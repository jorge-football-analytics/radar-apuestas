"""Combinadas en bet365 por nivel de riesgo. Una pata por partido y solo cuotas de bet365 confirmadas.
tier 'valor': todas las patas con EV ≥ 0 y la combinada con EV mínimo según el riesgo.
tier 'pronostico': las patas más sólidas del día aunque el margen de la casa deje el EV algo negativo (solo web)."""
from __future__ import annotations

import itertools
import json
from math import prod

import pandas as pd

from .config import SPORTS, Settings, sport_of
from .model.markets import label

RISK = {
    # nombre: (patas, prob. mínima por pata, prob. mínima combinada, EV mínimo combinado si es 'valor')
    "conservadora": ((2, 2), 0.60, 0.40, 0.02),
    "equilibrada": ((2, 3), 0.45, 0.20, 0.05),
    "arriesgada": ((3, 4), 0.33, 0.06, 0.10),
}


def _legs_table(preds: pd.DataFrame, odds: pd.DataFrame, s: Settings) -> pd.DataFrame:
    b = odds[odds["bookmaker"] == s.book]
    t = preds.merge(b, on=["match_id", "market", "selection", "point"])
    t["ev"] = t["p_final"] * t["price"] - 1
    return t[(t["price"] >= 1.15) & (t["price"] <= 5)]


def build_combos(preds: pd.DataFrame, odds: pd.DataFrame, matches: dict[str, dict], s: Settings,
                 per_risk: int = 2) -> list[dict]:
    if preds.empty or odds.empty:
        return []
    legs_all = _legs_table(preds, odds, s)
    out = []
    for tier, leg_ev in (("valor", 0.0), ("pronostico", -0.035)):
        legs = legs_all[legs_all["ev"] >= leg_ev].sort_values("ev", ascending=False).drop_duplicates("match_id")
        for risk, ((kmin, kmax), pmin_leg, pmin, evmin) in RISK.items():
            if tier == "pronostico" and any(c["risk"] == risk for c in out):
                continue  # ya hay combinadas con valor de este riesgo
            pool = legs[legs["p_final"] >= pmin_leg].head(10)
            found = []
            for k in range(kmin, min(kmax, s.max_combo_legs) + 1):
                for combo in itertools.combinations(pool.itertuples(index=False), k):
                    p = prod(c.p_final for c in combo)
                    o = prod(c.price for c in combo)
                    ev = p * o - 1
                    if p < pmin or (tier == "valor" and ev < evmin) or (tier == "pronostico" and ev < -0.08):
                        continue
                    found.append((ev, p, o, combo))
            found.sort(key=lambda x: (x[0], x[1]), reverse=True)
            used: set = set()
            n = 0
            for ev, p, o, combo in found:
                ids = {c.match_id for c in combo}
                if ids & used:
                    continue
                legs_out = []
                for c in combo:
                    m = matches[c.match_id]
                    sp = m.get("sport") or sport_of(m["league"])
                    legs_out.append(dict(match_id=c.match_id, sport=sp, league=m["league"], market=c.market,
                                         selection=c.selection, point=c.point, price=c.price, p=c.p_final,
                                         label=label(c.market, c.selection, c.point, m["home"], m["away"],
                                                     SPORTS[sp]["unidad"]),
                                         partido=f"{m['home']} - {m['away']}", kickoff=m["kickoff"]))
                f = ev / (o - 1) * s.kelly_fraction * 0.5 if ev > 0 and o > 1 else 0
                sig = s.book + "|" + "|".join(sorted(f"{l['match_id']}:{l['market']}:{l['selection']}:{l['point']}"
                                                     for l in legs_out))
                out.append(dict(tier=tier, risk=risk, bookmaker=s.book, odds=o, prob=p, ev=ev, legs=legs_out,
                                stake=round(min(f, s.max_stake_pct / 2) * s.bankroll, 2), signature=sig,
                                sports=",".join(sorted({l["sport"] for l in legs_out})),
                                first_kickoff=min(l["kickoff"] for l in legs_out)))
                used |= ids
                n += 1
                if n >= per_risk:
                    break
    return out


def legs_json(c: dict) -> str:
    return json.dumps(c["legs"], ensure_ascii=False)
