"""Combinadas en bet365 construidas SOLO con selecciones Élite ('valor') o Sólidas ('solida'), con cuota de bet365
confirmada y dentro del rango MIN_ODDS-MAX_ODDS.

Descorrelación: una selección por partido y, por defecto, como mucho una por liga (COMBO_MAX_PER_LEAGUE). Entre
partidos distintos de ligas distintas la independencia es una buena aproximación, así que la probabilidad real
conjunta es el producto de las probabilidades limpias, y se compara con la cuota total de bet365:
    valor esperado = Π p · Π cuota − 1
Niveles:
  valor  → valor esperado ≥ COMBO_MIN_EV (+2 %) y ninguna pata con valor negativo
  solida → valor esperado ≥ COMBO_SOLID_MIN_EV (−1,5 %): combinada fiable que casi no regala margen (sin ventaja)
"""
from __future__ import annotations

import itertools
import json
from collections import Counter
from math import prod

from .config import SPORTS, Settings, sport_of

RISK = {
    # nombre: (patas, prob. mínima por pata, prob. mínima conjunta)
    "conservadora": ((2, 2), 0.65, 0.45),
    "equilibrada": ((2, 3), 0.55, 0.30),
    "arriesgada": ((3, 4), 0.45, 0.15),
}
TIER_RANK = {"valor": 2, "solida": 1}


def _as_dict(c) -> dict:
    return c.as_row() if hasattr(c, "as_row") else dict(c)


def combo_legs(cands, s: Settings) -> list[dict]:
    """La mejor selección apta de cada partido (Élite antes que Sólida; a igualdad, más probabilidad)."""
    best: dict[str, dict] = {}
    for c in map(_as_dict, cands):
        if c.get("tier") not in TIER_RANK or not c.get("verified"):
            continue
        if not (s.min_odds <= float(c["odds"]) <= s.max_odds):
            continue
        key = (TIER_RANK[c["tier"]], c["ev"] if c["tier"] == "valor" else c["p_final"])
        prev = best.get(c["match_id"])
        if prev is None or key > prev["_key"]:
            best[c["match_id"]] = {**c, "_key": key}
    return sorted(best.values(), key=lambda c: c["_key"], reverse=True)


def build_combos(cands, matches: dict[str, dict], s: Settings, per_risk: int = 2, pool_size: int = 12) -> list[dict]:
    legs_all = combo_legs(cands, s)
    out = []
    for risk, ((kmin, kmax), pmin_leg, pmin) in RISK.items():
        pool = [c for c in legs_all if c["p_final"] >= pmin_leg][:pool_size]
        found = []
        for k in range(kmin, min(kmax, s.max_combo_legs) + 1):
            for combo in itertools.combinations(pool, k):
                leagues = Counter(c["league"] for c in combo)
                if max(leagues.values()) > s.combo_max_per_league:
                    continue  # correlación: misma liga
                p = prod(c["p_final"] for c in combo)
                if p < pmin:
                    continue
                o = prod(c["odds"] for c in combo)
                ev = p * o - 1
                if ev >= s.combo_min_ev and all(c["ev"] >= 0 for c in combo):
                    tier = "valor"
                elif ev >= s.combo_solid_min_ev:
                    tier = "solida"
                else:
                    continue
                found.append((TIER_RANK[tier], ev, p, o, combo, tier))
        found.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
        used: set = set()
        n = 0
        for _, ev, p, o, combo, tier in found:
            ids = {c["match_id"] for c in combo}
            if ids & used:
                continue
            legs_out = []
            for c in combo:
                m = matches.get(c["match_id"], c)
                sp = m.get("sport") or sport_of(m["league"])
                legs_out.append(dict(match_id=c["match_id"], sport=sp, league=m["league"], market=c["market"],
                                     selection=c["selection"], point=c["point"], price=c["odds"], p=c["p_final"],
                                     ev=c["ev"], tier=c["tier"], label=c["label"],
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
    return json.dumps(c["legs"], ensure_ascii=False, default=float)


__all__ = ["build_combos", "combo_legs", "legs_json", "RISK", "SPORTS"]
