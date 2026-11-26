"""Combinadas por perfil, listas para jugar en bet365 cada jornada.

Perfiles (siempre que haya partidos):
  segura      → cuota total 1,70-2,20 · patas con probabilidad real ≥ 70 % y el menor margen posible
  equilibrada → cuota total 2,50-4,00 · patas ≥ 55 %, mezclando deportes y mercados
  valor       → solo patas con ventaja (valor esperado > 0) y combinada con valor ≥ COMBO_MIN_EV. Solo si existe

Las patas pueden ser de cualquier mercado analizado: ganador, doble oportunidad, más/menos, ambos marcan o hándicap.
Descorrelación: una pata por partido y como mucho COMBO_MAX_PER_LEAGUE por liga (se relaja a 2 si no hay otra
forma). Entre partidos distintos la independencia es una buena aproximación: probabilidad conjunta = Π p.

Nivel de cada combinada (honesto: no siempre hay valor):
  valor       → valor esperado ≥ COMBO_MIN_EV y ninguna pata negativa
  solida      → valor esperado ≥ COMBO_SOLID_MIN_EV (−1,5 %): casi no regala margen
  sin_ventaja → jugable, pero bet365 se queda un margen mayor (se muestra cuánto)
  verificar   → alguna pata sin cuota de bet365 todavía: se indica la cuota mínima de cada pata
"""
from __future__ import annotations

import itertools
import json
from collections import Counter
from math import prod

import pandas as pd

from .config import SPORTS, Settings, sport_of
from .stats import price_line, scouting

PROFILES = {
    # nombre: (patas, prob. mínima por pata, cuota total mínima, máxima, margen máximo por pata)
    "segura": ((2, 3), 0.70, 1.70, 2.20, -0.04),
    "equilibrada": ((2, 4), 0.55, 2.50, 4.00, -0.04),
    "valor": ((2, 4), 0.40, 1.60, 6.00, 0.0),
}
MK_NAME = {"1x2": "Resultado", "ml": "Ganador", "ou": "Más/menos", "spread": "Hándicap", "btts": "Ambos marcan",
           "dc": "Doble oportunidad"}


def _records(preds) -> list[dict]:
    if preds is None:
        return []
    if isinstance(preds, pd.DataFrame):
        return preds.to_dict("records")
    return [c.as_row() if hasattr(c, "as_row") else dict(c) for c in preds]


def table_rows(tables: dict | None, league: str, home: str, away: str) -> dict:
    """Fila de la clasificación de cada equipo (tablas que guarda el pipeline: tabla_<liga>)."""
    t = (tables or {}).get(league) or []
    by = {r.get("equipo"): r for r in t}
    return {"local": by.get(home) or {}, "visit": by.get(away) or {}}


def leg_report(m: dict, sport: str, market: str, selection: str, point: float, info: dict, tables: dict | None,
               p: float, price: float | None, priced: bool, n_books=None, n_sharp=None, derived: bool = False
               ) -> list[str]:
    """Informe de una pata: scouting deportivo con datos reales + por qué merece (o no) la pena la cuota."""
    st = (info or {}).get("stats") or {}
    lines = scouting(st, m["home"], m["away"], sport, market, selection, point,
                     table_rows(tables, m["league"], m["home"], m["away"]), info)
    nb = int(n_books) if n_books is not None and not pd.isna(n_books) else None
    ns = int(n_sharp) if n_sharp is not None and not pd.isna(n_sharp) else None
    return lines + [price_line(p, price, priced, nb, ns, derived)]


def leg_pool(preds, matches: dict, s: Settings, infos: dict | None = None, tables: dict | None = None) -> list[dict]:
    """Selecciones candidatas a pata, con su informe. priced=True si hay cuota de bet365 coherente y reciente."""
    infos = infos or {}
    out = []
    for r in _records(preds):
        m = matches.get(r.get("match_id"))
        p = r.get("p_final")
        if m is None or p is None or pd.isna(p) or r.get("tier") in ("revisar", None) or (isinstance(r.get("tier"), float)):
            continue
        if r.get("weak"):
            continue
        if (r.get("n_books") or 0) < s.min_books and (r.get("n_sharp") or 0) < 1:
            continue
        age = r.get("b365_age_h")
        priced = bool(r.get("verified")) and (age is None or pd.isna(age) or age <= s.solid_b365_max_age_h)
        price = float(r["price"]) if priced else round(1 / p, 3)
        if not (1.15 <= price <= s.max_odds):
            continue
        ev = p * price - 1 if priced else 0.0
        sp = m.get("sport") or sport_of(m["league"])
        info = infos.get(r["match_id"]) or {}
        why = leg_report(m, sp, r["market"], r["selection"], float(r["point"]), info, tables, float(p),
                         price if priced else None, priced, r.get("n_books"), r.get("n_sharp"), bool(r.get("derived")))
        out.append(dict(match_id=r["match_id"], sport=sp, league=m["league"], kickoff=m["kickoff"],
                        partido=f"{m['home']} - {m['away']}", market=r["market"], selection=r["selection"],
                        point=float(r["point"]), label=r.get("label") or r["selection"], p=float(p), price=price,
                        ev=float(ev), priced=priced, tier=r.get("tier"), min_odds=round(1 / p, 2),
                        why=why))
    return out


def _best_per_match(legs: list[dict], key) -> list[dict]:
    best: dict[str, dict] = {}
    for l in legs:
        if l["match_id"] not in best or key(l) > key(best[l["match_id"]]):
            best[l["match_id"]] = l
    return sorted(best.values(), key=key, reverse=True)


def _search(pool: list[dict], kmin: int, kmax: int, omin: float, omax: float, max_league: int, s: Settings,
            diversity: bool = False) -> list[tuple]:
    found = []
    for k in range(kmin, min(kmax, s.max_combo_legs) + 1):
        for combo in itertools.combinations(pool, k):
            if len({c["match_id"] for c in combo}) < k:
                continue
            if max(Counter(c["league"] for c in combo).values()) > max_league:
                continue
            o = prod(c["price"] for c in combo)
            if not (omin <= o <= omax):
                continue
            p = prod(c["p"] for c in combo)
            ev = p * o - 1
            div = len({c["sport"] for c in combo}) + len({c["market"] for c in combo}) if diversity else 0
            found.append((ev + 0.004 * div, ev, p, o, combo))
    found.sort(key=lambda x: (x[0], x[2]), reverse=True)
    return found


def _tier(combo, ev: float, s: Settings) -> str:
    if not all(c["priced"] for c in combo):
        return "verificar"
    if ev >= s.combo_min_ev and all(c["ev"] >= 0 for c in combo):
        return "valor"
    if ev >= s.combo_solid_min_ev:
        return "solida"
    return "sin_ventaja"


def build_combos(preds, matches: dict[str, dict], s: Settings, infos: dict | None = None, per_profile: int = 2,
                 pool_size: int = 14, tables: dict | None = None) -> list[dict]:
    legs = leg_pool(preds, matches, s, infos, tables)
    out = []
    for profile, ((kmin, kmax), pmin, omin, omax, margin_min) in PROFILES.items():
        if profile == "valor":
            cand = [l for l in legs if l["priced"] and l["ev"] > 0 and l["p"] >= pmin]
            pool = _best_per_match(cand, key=lambda l: (l["ev"], l["p"]))[:pool_size]
        else:
            priced = [l for l in legs if l["priced"] and l["p"] >= pmin and l["ev"] >= margin_min]
            pool = _best_per_match(priced, key=lambda l: (l["ev"], l["p"]))[:pool_size]
        found = []
        for max_league in (s.combo_max_per_league, 2):
            found = _search(pool, kmin, kmax, omin, omax, max_league, s, diversity=profile == "equilibrada")
            if found:
                break
        if not found and profile != "valor":  # sin suficientes cuotas de bet365: combinada por confirmar
            unpriced = [l for l in legs if l["p"] >= pmin]
            pool = _best_per_match(unpriced, key=lambda l: (l["priced"], l["p"]))[:pool_size]
            found = _search(pool, kmin, kmax, omin, omax, 2, s)
        used: set = set()
        n = 0
        for _, ev, p, o, combo in found:
            ids = {c["match_id"] for c in combo}
            if ids & used:
                continue
            tier = _tier(combo, ev, s)
            if profile == "valor" and tier != "valor":
                continue
            f = ev / (o - 1) * s.kelly_fraction * 0.5 if tier == "valor" and o > 1 else 0
            legs_out = [{k: v for k, v in c.items()} for c in combo]
            sig = s.book + "|" + profile + "|" + "|".join(sorted(
                f"{l['match_id']}:{l['market']}:{l['selection']}:{l['point']}" for l in legs_out))
            out.append(dict(tier=tier, risk=profile, bookmaker=s.book, odds=o, prob=p, ev=ev, legs=legs_out,
                            stake=round(min(f, s.max_stake_pct / 2) * s.bankroll, 2), signature=sig,
                            sports=",".join(sorted({l["sport"] for l in legs_out})),
                            first_kickoff=min(l["kickoff"] for l in legs_out)))
            used |= ids
            n += 1
            if n >= per_profile:
                break
    return out


def legs_json(c: dict) -> str:
    return json.dumps(c["legs"], ensure_ascii=False, default=float)


__all__ = ["build_combos", "leg_pool", "leg_report", "legs_json", "table_rows", "PROFILES", "MK_NAME"]
