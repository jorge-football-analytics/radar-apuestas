"""Liquidar partidos de todos los deportes, medir el rendimiento real y recalibrar el peso modelo/mercado."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from .config import DEFAULT_WEIGHT, Settings
from .db import DB
from .model.markets import devig, outcome
from .teams import match_team, player_key, us_key

log = logging.getLogger(__name__)


def _closing_prob(row: pd.Series, market: str, sel: str, point: float) -> float | None:
    """Probabilidad justa de cierre (Pinnacle > bet365 > media) para el CLV en fútbol."""
    try:
        if market == "1x2":
            cols = ["close_H", "close_D", "close_A"]
            if row[cols].notna().all():
                return devig(list(row[cols].astype(float)))["HDA".index(sel)]
        if market == "ou" and point == 2.5:
            cols = ["close_O25", "close_U25"]
            if row[cols].notna().all():
                return devig(list(row[cols].astype(float)))["OU".index(sel)]
    except KeyError:
        return None
    return None


def same_team(a: str, b: str, sport: str) -> bool:
    if a == b:
        return True
    if sport == "tennis":
        return player_key(a) == player_key(b)
    if sport in ("basketball", "nfl"):
        return us_key(a) == us_key(b)
    return match_team(a, [b]) is not None


def _find(results: pd.DataFrame, m: pd.Series, unordered: bool, days: int):
    """Busca el resultado del partido. Devuelve (hg, ag, fila) con el orden de nuestro partido, o None."""
    if results is None or results.empty:
        return None
    ko = pd.Timestamp(m["kickoff"]).tz_convert(None).normalize()
    r = results[(results["date"] - ko).abs() <= pd.Timedelta(days=days)]
    if "league" in r and not unordered:
        r = r[r["league"] == m["league"]]
    hit = r[(r["home"] == m["home"]) & (r["away"] == m["away"])]
    if not hit.empty:
        row = hit.iloc[0]
        return int(row["hg"]), int(row["ag"]), row
    if unordered:
        hit = r[(r["home"] == m["away"]) & (r["away"] == m["home"])]
        if not hit.empty:
            row = hit.iloc[0]
            return int(row["ag"]), int(row["hg"]), row
    return None


def settle(db: DB, results: dict[str, pd.DataFrame], now: datetime, api_scores: list[dict] | None = None) -> int:
    """results: {deporte: DataFrame(league, date, home, away, hg, ag, ...)} con los nombres canónicos."""
    pending = db.df("SELECT * FROM matches WHERE status='pendiente' AND kickoff < ?",
                    ((now - timedelta(hours=3)).isoformat(),))
    if pending.empty:
        return 0
    settled = 0
    for _, m in pending.iterrows():
        sport = m.get("sport") or "football"
        found = _find(results.get(sport), m, unordered=(sport == "tennis"), days=2 if sport == "tennis" else 1)
        hg = ag = row = None
        if found:
            hg, ag, row = found
        elif api_scores:
            for ev in api_scores:
                if not ev.get("completed") or not ev.get("scores"):
                    continue
                h, a = ev.get("home_team", ""), ev.get("away_team", "")
                if same_team(h, m["home"], sport) and same_team(a, m["away"], sport):
                    sc = {x["name"]: int(float(x["score"])) for x in ev["scores"]}
                    if h in sc and a in sc:
                        hg, ag = sc[h], sc[a]
                        break
        if hg is None:
            if pd.Timestamp(m["kickoff"]) < pd.Timestamp(now - timedelta(days=12)):
                db.execute("UPDATE matches SET status='sin_resultado' WHERE id=?", (m["id"],))
            continue
        void_all = sport == "tennis" and row is not None and "completed" in row and not bool(row["completed"])
        with db.tx() as c:
            c.execute("UPDATE matches SET hg=?, ag=?, status='finalizado' WHERE id=?", (hg, ag, m["id"]))
            for p in c.execute("SELECT market, selection, point FROM predictions WHERE match_id=?", (m["id"],)).fetchall():
                res = None if void_all else outcome(p[0], p[1], p[2], hg, ag)
                c.execute("UPDATE predictions SET won=? WHERE match_id=? AND market=? AND selection=? AND point=?",
                          (res, m["id"], p[0], p[1], p[2]))
            for p in c.execute("SELECT id, market, selection, point, odds, stake FROM picks WHERE match_id=?",
                               (m["id"],)).fetchall():
                res = None if void_all else outcome(p[1], p[2], p[3], hg, ag)
                if res is None:
                    c.execute("UPDATE picks SET won=-1, profit=0 WHERE id=?", (p[0],))
                    continue
                stake = p[5] or 0
                profit = stake * (p[4] - 1) if res else -stake
                clv = closing = None
                if row is not None and sport == "football":
                    pc = _closing_prob(row, p[1], p[2], p[3])
                    if pc:
                        closing, clv = 1 / pc, p[4] * pc - 1
                c.execute("UPDATE picks SET won=?, profit=?, closing_odds=?, clv=? WHERE id=?",
                          (res, profit, closing, clv, p[0]))
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
        odds, won = 1.0, 1
        for l in legs:
            r = outcome(l["market"], l["selection"], l["point"], *res[l["match_id"]])
            if r is None:
                continue  # pata nula: se quita de la combinada
            odds *= l["price"]
            won = won and r
        stake = c["stake"] or 0
        profit = stake * (odds - 1) if won else -stake
        db.execute("UPDATE combos SET won=?, profit=?, odds=? WHERE id=?", (int(won), profit, odds, c["id"]))


def _ll(p, y):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, float)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def _roi(g: pd.DataFrame) -> float:
    return float(((g["odds"] - 1) * g["won"] - (1 - g["won"])).mean())


def metrics(db: DB, tier: str = "valor") -> dict:
    picks = db.df("SELECT * FROM picks WHERE won IN (0, 1) AND tier=? AND bookmaker=?", (tier, "bet365"))
    out: dict = {"n": len(picks)}
    if not picks.empty:
        out.update(hit_rate=float(picks["won"].mean()), avg_odds=float(picks["odds"].mean()), roi=_roi(picks),
                   profit=float(picks["profit"].fillna(0).sum()), staked=float(picks["stake"].fillna(0).sum()),
                   clv=float(picks["clv"].mean()) if picks["clv"].notna().any() else None,
                   expected_hit=float(picks["p_final"].mean()))
        for col in ("sport", "risk", "confidence"):
            out[f"by_{col}"] = {k: {"n": len(g), "acierto": float(g["won"].mean()), "roi": _roi(g)}
                                for k, g in picks.groupby(col)}
    combos = db.df("SELECT * FROM combos WHERE won IS NOT NULL AND tier=?", (tier,))
    if not combos.empty:
        out["combos"] = {k: {"n": len(g), "acierto": float(g["won"].mean()), "roi": _roi(g)}
                         for k, g in combos.groupby("risk")}
    preds = db.df("SELECT p.*, m.sport FROM predictions p JOIN matches m ON m.id = p.match_id "
                  "WHERE p.won IN (0, 1) AND p.p_model IS NOT NULL AND p.p_market IS NOT NULL")
    if len(preds) >= 30:
        out["calibration"] = {"n": len(preds), "logloss_modelo": _ll(preds["p_model"], preds["won"]),
                              "logloss_mercado": _ll(preds["p_market"], preds["won"]),
                              "logloss_final": _ll(preds["p_final"], preds["won"])}
    return out


def recalibrate(db: DB, s: Settings, min_n: int = 150, prior_n: int = 300) -> dict[str, float]:
    """Peso modelo/mercado por deporte y mercado que mejor habría predicho lo ya jugado (encogido hacia el previo)."""
    weights = {k: v for k, v in (db.get("weights", {}) or {}).items() if "|" in k}
    prior = {k: v for k, v in (db.get("weights_backtest", {}) or {}).items() if "|" in k}
    preds = db.df("SELECT p.*, m.sport FROM predictions p JOIN matches m ON m.id = p.match_id "
                  "WHERE p.won IN (0, 1) AND p.p_model IS NOT NULL AND p.p_market IS NOT NULL")
    for (sport, market), g in preds.groupby(["sport", "market"]):
        key = f"{sport}|{market}"
        base = prior.get(key, DEFAULT_WEIGHT.get(sport, 0.2))
        if len(g) < min_n:
            weights[key] = base
            continue
        grid = np.linspace(0, 1, 21)
        losses = [_ll(w * g["p_model"] + (1 - w) * g["p_market"], g["won"]) for w in grid]
        w_best = float(grid[int(np.argmin(losses))])
        weights[key] = round((len(g) * w_best + prior_n * base) / (len(g) + prior_n), 3)
    for k, v in prior.items():
        weights.setdefault(k, v)
    db.set("weights", weights)
    return weights
