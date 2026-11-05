"""Combinadas por nivel de riesgo. Todas las patas en la MISMA casa (una combinada no se puede repartir),
una pata por partido (partidos distintos ≈ independientes) y solo patas con valor esperado ≥ 0 en esa casa."""
from __future__ import annotations

import itertools
import json
from math import prod

import pandas as pd

from .config import Settings
from .model.markets import label

RISK = {
    # nombre: (patas, prob. mínima por pata, prob. mínima combinada, EV mínimo combinado)
    "conservadora": ((2, 2), 0.58, 0.38, 0.02),
    "equilibrada": ((2, 3), 0.42, 0.18, 0.05),
    "arriesgada": ((3, 4), 0.30, 0.05, 0.10),
}


def build_combos(preds: pd.DataFrame, odds: pd.DataFrame, matches: dict[str, dict], s: Settings,
                 per_risk: int = 2) -> list[dict]:
    """preds: match_id, market, selection, point, p_final.  odds: match_id, bookmaker, market, selection, point, price."""
    if preds.empty or odds.empty:
        return []
    legs_all = preds.merge(odds, on=["match_id", "market", "selection", "point"])
    legs_all["ev"] = legs_all["p_final"] * legs_all["price"] - 1
    legs_all = legs_all[(legs_all["ev"] >= 0) & (legs_all["p_final"] >= 0.30) & legs_all["bookmaker"].map(s.can_bet)]
    results = []
    books = legs_all["bookmaker"].value_counts().index.tolist()
    if s.preferred_book in books:  # bet365 primero
        books.remove(s.preferred_book)
        books.insert(0, s.preferred_book)
    for book in books[:6]:
        legs = legs_all[legs_all["bookmaker"] == book]
        # mejor pata por partido
        legs = legs.sort_values("ev", ascending=False).drop_duplicates("match_id")
        for risk, ((kmin, kmax), pmin_leg, pmin, evmin) in RISK.items():
            pool = legs[legs["p_final"] >= pmin_leg].head(12)
            for k in range(kmin, min(kmax, s.max_combo_legs) + 1):
                for combo in itertools.combinations(pool.itertuples(index=False), k):
                    p = prod(c.p_final for c in combo)
                    o = prod(c.price for c in combo)
                    ev = p * o - 1
                    if p < pmin or ev < evmin:
                        continue
                    results.append(dict(risk=risk, bookmaker=book, odds=o, prob=p, ev=ev, legs=[
                        dict(match_id=c.match_id, market=c.market, selection=c.selection, point=c.point,
                             price=c.price, p=c.p_final,
                             label=label(c.market, c.selection, c.point, matches[c.match_id]["home"],
                                         matches[c.match_id]["away"]),
                             partido=f"{matches[c.match_id]['home']} - {matches[c.match_id]['away']}",
                             kickoff=matches[c.match_id]["kickoff"]) for c in combo]))
    out = []
    for risk in RISK:
        rs = [r for r in results if r["risk"] == risk]
        # ordenar por EV, a igualdad se prefiere bet365
        rs.sort(key=lambda r: (round(r["ev"], 3), r["bookmaker"] == s.preferred_book), reverse=True)
        seen_matches: set = set()
        for r in rs:
            ids = frozenset(l["match_id"] for l in r["legs"])
            if ids & seen_matches:  # no repetir partidos entre combinadas del mismo nivel
                continue
            # stake: Kelly fraccional reducido a la mitad (más varianza que una simple)
            edge = r["ev"]
            f = edge / (r["odds"] - 1) * s.kelly_fraction * 0.5 if r["odds"] > 1 else 0
            r["stake"] = round(min(f, s.max_stake_pct / 2) * s.bankroll, 2)
            r["signature"] = r["bookmaker"] + "|" + "|".join(sorted(
                f"{l['match_id']}:{l['market']}:{l['selection']}:{l['point']}" for l in r["legs"]))
            out.append(r)
            seen_matches |= ids
            if sum(1 for o in out if o["risk"] == risk) >= per_risk:
                break
    return out


def legs_json(c: dict) -> str:
    return json.dumps(c["legs"], ensure_ascii=False)
