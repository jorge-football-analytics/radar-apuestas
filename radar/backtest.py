"""Backtest walk-forward: el modelo solo ve partidos anteriores a cada jornada, como en la vida real.
Mide si la estrategia habría ganado apostando en bet365, y fija el peso inicial modelo/mercado."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .config import Settings
from .model.dixon_coles import fit, score_matrix
from .model.markets import devig, market_probs

log = logging.getLogger(__name__)


def _ll(p, y):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, float)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def walk_forward(hist: pd.DataFrame, s: Settings, test_days: int = 365, refit_days: int = 14) -> pd.DataFrame:
    """Una fila por partido y selección (1X2 y más/menos 2.5) con probabilidades de modelo y mercado."""
    rows = []
    end = hist["date"].max()
    start = end - pd.Timedelta(days=test_days)
    for league, h in hist.groupby("league"):
        t = start
        while t <= end:
            model = fit(h, t, xi=s.time_decay_xi)
            window = h[(h["date"] >= t) & (h["date"] < t + pd.Timedelta(days=refit_days))]
            t += pd.Timedelta(days=refit_days)
            if model is None:
                continue
            for _, m in window.iterrows():
                r = model.rates(m["home"], m["away"])
                if r is None:
                    continue
                probs = market_probs(score_matrix(*r, model.rho), [2.5])
                specs = [("1x2", "HDA", ["open_H", "open_D", "open_A"], ["b365_H", "b365_D", "b365_A"],
                          ["close_H", "close_D", "close_A"]),
                         ("ou", "OU", ["open_O25", "open_U25"], ["b365_O25", "b365_U25"], ["close_O25", "close_U25"])]
                res = {"H": m["hg"] > m["ag"], "D": m["hg"] == m["ag"], "A": m["hg"] < m["ag"],
                       "O": m["hg"] + m["ag"] > 2.5, "U": m["hg"] + m["ag"] < 2.5}
                for market, sels, open_c, b365_c, close_c in specs:
                    if m[open_c].isna().any():
                        continue
                    fair = devig(list(m[open_c].astype(float)))
                    close = devig(list(m[close_c].astype(float))) if m[close_c].notna().all() else [np.nan] * len(sels)
                    pt = 2.5 if market == "ou" else 0.0
                    for k, sel in enumerate(sels):
                        rows.append(dict(league=league, date=m["date"], market=market, selection=sel,
                                         p_model=probs[(market, sel, pt)], p_market=fair[k], p_close=close[k],
                                         price=m[b365_c[k]], won=int(res[sel])))
    return pd.DataFrame(rows)


def summarize(bt: pd.DataFrame, s: Settings) -> dict:
    if bt.empty:
        return {}
    out: dict = {"n_predicciones": len(bt), "desde": str(bt["date"].min().date()), "hasta": str(bt["date"].max().date())}
    grid = np.linspace(0, 1, 21)
    weights = {}
    for market, g in bt.groupby("market"):
        losses = [_ll(w * g["p_model"] + (1 - w) * g["p_market"], g["won"]) for w in grid]
        weights[market] = float(grid[int(np.argmin(losses))])
        out[f"logloss_{market}"] = {"modelo": _ll(g["p_model"], g["won"]), "mercado": _ll(g["p_market"], g["won"]),
                                    "mejor_mezcla": float(min(losses)), "peso_modelo": weights[market]}
    out["pesos"] = weights
    # Estrategia: apuestas con EV ≥ min_ev a la cuota de bet365, usando la mezcla óptima
    bt = bt.dropna(subset=["price"]).copy()
    bt["p"] = [weights[mk] * pm + (1 - weights[mk]) * pk for mk, pm, pk in zip(bt["market"], bt["p_model"], bt["p_market"])]
    bt["ev"] = bt["p"] * bt["price"] - 1
    strat = {}
    for thr in (0.02, 0.04, 0.06, 0.10):
        b = bt[(bt["ev"] >= thr) & (bt["p"] >= s.min_prob) & (bt["price"] <= s.max_odds)]
        if b.empty:
            continue
        roi = ((b["price"] - 1) * b["won"] - (1 - b["won"])).mean()
        clv = (b["price"] * b["p_close"] - 1).mean()
        # intervalo aproximado del ROI (cuánto podría ser suerte)
        se = ((b["price"] * b["won"] - 1).std() / np.sqrt(len(b))) if len(b) > 1 else np.nan
        strat[f"ev_{int(thr * 100)}"] = {"apuestas": len(b), "acierto": float(b["won"].mean()),
                                         "cuota_media": float(b["price"].mean()), "roi": float(roi),
                                         "roi_ic95": [float(roi - 1.96 * se), float(roi + 1.96 * se)],
                                         "clv": float(clv) if pd.notna(clv) else None}
    out["estrategia_bet365"] = strat
    return out
