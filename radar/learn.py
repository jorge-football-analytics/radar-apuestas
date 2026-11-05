"""Liquidar partidos, medir rendimiento real y recalibrar el peso del modelo frente al mercado."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from .config import Settings
from .db import DB
from .model.markets import devig, outcome

log = logging.getLogger(__name__)


def _closing_prob(row: pd.Series, market: str, sel: str, point: float) -> float | None:
    """Probabilidad justa de cierre (Pinnacle > bet365 > media) para calcular el CLV."""
    if market == "1x2":
        cols = ["close_H", "close_D", "close_A"]
        if row[cols].notna().all():
            return devig(list(row[cols].astype(float)))["HDA".index(sel)]
    if market == "ou" and point == 2.5:
        cols = ["close_O25", "close_U25"]
        if row[cols].notna().all():
            return devig(list(row[cols].astype(float)))["OU".index(sel)]
    return None


def settle(db: DB, hist: pd.DataFrame, now: datetime, ucl_scores: list[dict] | None = None) -> int:
    """Rellena resultados de partidos ya jugados y liquida predicciones, picks y combinadas."""
    pending = db.df("SELECT * FROM matches WHERE status='pendiente' AND kickoff < ?",
                    ((now - timedelta(hours=3)).isoformat(),))
    if pending.empty:
        return 0
    settled = 0
    for _, m in pending.iterrows():
        ko = pd.Timestamp(m["kickoff"]).tz_convert(None)
        row = None
        if not hist.empty:
            cand = hist[(hist["league"] == m["league"]) & (hist["home"] == m["home"]) & (hist["away"] == m["away"])
                        & ((hist["date"] - ko.normalize()).abs() <= pd.Timedelta(days=1))]
            if not cand.empty:
                row = cand.iloc[0]
        hg = ag = None
        if row is not None:
            hg, ag = int(row["hg"]), int(row["ag"])
        elif ucl_scores:
            for ev in ucl_scores:
                if ev.get("completed") and ev["home_team"] == m["home"] and ev["away_team"] == m["away"]:
                    sc = {s_["name"]: int(s_["score"]) for s_ in ev.get("scores") or []}
                    if m["home"] in sc and m["away"] in sc:
                        hg, ag = sc[m["home"]], sc[m["away"]]
        if hg is None:
            if ko < (now - timedelta(days=10)).replace(tzinfo=None):
                db.execute("UPDATE matches SET status='sin_resultado' WHERE id=?", (m["id"],))
            continue
        with db.tx() as c:
            c.execute("UPDATE matches SET hg=?, ag=?, status='finalizado' WHERE id=?", (hg, ag, m["id"]))
            for p in c.execute("SELECT market, selection, point FROM predictions WHERE match_id=?", (m["id"],)).fetchall():
                c.execute("UPDATE predictions SET won=? WHERE match_id=? AND market=? AND selection=? AND point=?",
                          (outcome(p[0], p[1], p[2], hg, ag), m["id"], p[0], p[1], p[2]))
            for p in c.execute("SELECT id, market, selection, point, odds, stake FROM picks WHERE match_id=?",
                               (m["id"],)).fetchall():
                won = outcome(p[1], p[2], p[3], hg, ag)
                profit = p[5] * (p[4] - 1) if won else -p[5]
                clv = closing = None
                if row is not None:
                    pc = _closing_prob(row, p[1], p[2], p[3])
                    if pc:
                        closing, clv = 1 / pc, p[4] * pc - 1
                c.execute("UPDATE picks SET won=?, profit=?, closing_odds=?, clv=? WHERE id=?",
                          (won, profit, closing, clv, p[0]))
        settled += 1
    _settle_combos(db)
    return settled


def _settle_combos(db: DB) -> None:
    res = {r["id"]: (r["hg"], r["ag"]) for r in db.conn.execute(
        "SELECT id, hg, ag FROM matches WHERE status='finalizado'").fetchall()}
    for c in db.conn.execute("SELECT id, legs, odds, stake FROM combos WHERE won IS NULL").fetchall():
        legs = json.loads(c["legs"])
        if not all(l["match_id"] in res for l in legs):
            continue
        won = int(all(outcome(l["market"], l["selection"], l["point"], *res[l["match_id"]]) for l in legs))
        profit = c["stake"] * (c["odds"] - 1) if won else -c["stake"]
        db.execute("UPDATE combos SET won=?, profit=? WHERE id=?", (won, profit, c["id"]))


def _logloss(p, y):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, float)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def metrics(db: DB) -> dict:
    picks = db.df("SELECT * FROM picks WHERE won IS NOT NULL")
    out: dict = {"n": len(picks)}
    if not picks.empty:
        out.update(
            hit_rate=float(picks["won"].mean()), avg_odds=float(picks["odds"].mean()),
            roi=float(((picks["odds"] - 1) * picks["won"] - (1 - picks["won"])).mean()),  # a 1 unidad
            profit=float(picks["profit"].sum()), staked=float(picks["stake"].sum()),
            clv=float(picks["clv"].mean()) if picks["clv"].notna().any() else None,
            expected_hit=float(picks["p_final"].mean()),
        )
        out["by_confidence"] = {
            k: {"n": len(g), "hit": float(g["won"].mean()),
                "roi": float(((g["odds"] - 1) * g["won"] - (1 - g["won"])).mean())}
            for k, g in picks.groupby("confidence")}
    combos = db.df("SELECT * FROM combos WHERE won IS NOT NULL")
    if not combos.empty:
        out["combos"] = {k: {"n": len(g), "hit": float(g["won"].mean()),
                             "roi": float(((g["odds"] - 1) * g["won"] - (1 - g["won"])).mean())}
                         for k, g in combos.groupby("risk")}
    preds = db.df("SELECT * FROM predictions WHERE won IS NOT NULL AND p_model IS NOT NULL")
    if len(preds) >= 30:
        out["calibration"] = {
            "n": len(preds),
            "logloss_modelo": _logloss(preds["p_model"], preds["won"]),
            "logloss_mercado": _logloss(preds["p_market"], preds["won"]),
            "logloss_final": _logloss(preds["p_final"], preds["won"]),
        }
    return out


def recalibrate(db: DB, s: Settings, min_n: int = 150, prior_n: int = 300) -> dict[str, float]:
    """Busca el peso modelo/mercado que mejor habría predicho los partidos ya jugados.
    Se encoge hacia el valor previo para no sobrerreaccionar con pocos datos."""
    weights = db.get("weights", {}) or {}
    prior = db.get("weights_backtest", {}) or {}
    preds = db.df("SELECT * FROM predictions WHERE won IS NOT NULL AND p_model IS NOT NULL")
    for market, g in preds.groupby("market"):
        base = prior.get(market, s.model_weight)
        if len(g) < min_n:
            weights[market] = base
            continue
        grid = np.linspace(0, 1, 21)
        losses = [_logloss(w * g["p_model"] + (1 - w) * g["p_market"], g["won"]) for w in grid]
        w_best = float(grid[int(np.argmin(losses))])
        weights[market] = round((len(g) * w_best + prior_n * base) / (len(g) + prior_n), 3)
    for market, w in prior.items():
        weights.setdefault(market, w)
    db.set("weights", weights)
    return weights
