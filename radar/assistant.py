"""Asistente de apuesta: con un presupuesto y una fecha límite, elige la mejor apuesta entre todas las disponibles.

Estrategias:
  simple    → la mejor selección individual
  combinada → la mejor combinada (una selección por partido) dentro del rango de cuota del riesgo elegido
  repartir  → reparte el presupuesto entre varias simples, en proporción a su ventaja (Kelly)
Siempre en bet365 y solo con selecciones que se resuelven antes de la fecha límite."""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from math import prod

import pandas as pd

from .config import DURATION_MIN

# objetivo de cada nivel de riesgo: (prob. mínima por selección, cuota total mínima, cuota total máxima)
RISK_TARGET = {"bajo": (0.62, 1.25, 2.2), "medio": (0.45, 2.0, 4.5), "alto": (0.30, 4.0, 15.0)}


@dataclass
class Plan:
    tipo: str
    legs: list[dict]
    odds: float
    prob: float
    ev: float
    stakes: list[float]
    budget: float
    notes: list[str] = field(default_factory=list)

    @property
    def win(self) -> float:  # beneficio si se gana (simple/combinada)
        return sum(st * (l["odds"] - 1) for st, l in zip(self.stakes, self.legs)) if self.tipo == "repartir" else \
            self.budget * (self.odds - 1)

    @property
    def expected(self) -> float:
        if self.tipo == "repartir":
            return sum(st * (l["p_final"] * l["odds"] - 1) for st, l in zip(self.stakes, self.legs))
        return self.budget * self.ev

    @property
    def ends(self) -> str:
        return max(l["ends"] for l in self.legs)


def candidates(picks: pd.DataFrame, now: datetime, deadline: datetime, sports: list[str] | None = None,
               only_value: bool = False, include_unverified: bool = False) -> pd.DataFrame:
    """Selecciones apostables que terminan antes de la fecha límite."""
    if picks.empty:
        return picks
    df = picks.copy()
    ko = pd.to_datetime(df["kickoff"], utc=True, format="ISO8601")
    dur = df["sport"].map(DURATION_MIN).fillna(120)
    df["ends_ts"] = ko + pd.to_timedelta(dur, unit="m")
    df["ends"] = df["ends_ts"].map(lambda t: t.isoformat())
    df = df[(ko > pd.Timestamp(now)) & (df["ends_ts"] <= pd.Timestamp(deadline))]
    if sports:
        df = df[df["sport"].isin(sports)]
    if not include_unverified:
        df = df[df["verified"] == 1]
    if only_value:
        df = df[df["tier"] == "valor"]
    df = df[(df["odds"] > 1.01) & df["p_final"].notna()]
    df["ev"] = df["p_final"] * df["odds"] - 1
    return df.sort_values("ev", ascending=False)


def _legs(rows) -> list[dict]:
    keep = ["match_id", "sport", "league", "kickoff", "home", "away", "label", "odds", "p_final", "ev", "tier",
            "risk", "ends", "confidence"]
    return [{k: r[k] for k in keep} for r in rows]


def best_single(c: pd.DataFrame, risk: str, budget: float) -> Plan | None:
    pmin, omin, omax = RISK_TARGET[risk]
    pool = c[(c["p_final"] >= pmin * 0.95) & (c["odds"] >= max(1.15, omin * 0.85)) & (c["odds"] <= omax * 1.5)]
    if pool.empty:
        pool = c
    if pool.empty:
        return None
    # mejor valor esperado; a igualdad, más probabilidad
    r = pool.assign(score=pool["ev"] + 0.02 * pool["p_final"]).sort_values("score", ascending=False).iloc[0]
    leg = _legs([r])[0]
    return Plan("simple", [leg], leg["odds"], leg["p_final"], leg["ev"], [budget], budget)


def best_combo(c: pd.DataFrame, risk: str, budget: float, max_legs: int = 4) -> Plan | None:
    pmin, omin, omax = RISK_TARGET[risk]
    pool = c[(c["p_final"] >= pmin) & (c["odds"] >= 1.15)]
    pool = pool.sort_values(["ev", "p_final"], ascending=False).drop_duplicates("match_id").head(12)
    best = None
    for k in range(2, max_legs + 1):
        for combo in itertools.combinations(pool.to_dict("records"), k):
            o = prod(x["odds"] for x in combo)
            if not (omin <= o <= omax):
                continue
            p = prod(x["p_final"] for x in combo)
            ev = p * o - 1
            key = (ev, p)
            if best is None or key > best[0]:
                best = (key, combo, o, p, ev)
    if best is None:
        return None
    _, combo, o, p, ev = best
    return Plan("combinada", _legs(combo), o, p, ev, [budget], budget)


def spread_budget(c: pd.DataFrame, budget: float, n: int = 5) -> Plan | None:
    pool = c[c["ev"] > 0].drop_duplicates("match_id").head(n)
    if pool.empty:
        return None
    kelly = (pool["ev"] / (pool["odds"] - 1)).clip(lower=0)
    w = kelly / kelly.sum() if kelly.sum() > 0 else pd.Series(1 / len(pool), index=pool.index)
    stakes = [round(budget * x, 2) for x in w]
    legs = _legs(pool.to_dict("records"))
    p_any = 1 - prod(1 - l["p_final"] for l in legs)
    ev = sum(st * (l["p_final"] * l["odds"] - 1) for st, l in zip(stakes, legs)) / budget
    return Plan("repartir", legs, float("nan"), p_any, ev, stakes, budget)


def recommend(picks: pd.DataFrame, budget: float, deadline: datetime, now: datetime, risk: str = "medio",
              tipo: str = "auto", sports: list[str] | None = None, only_value: bool = False,
              include_unverified: bool = False) -> tuple[Plan | None, list[Plan], int]:
    """Devuelve (recomendación principal, alternativas, nº de selecciones consideradas)."""
    c = candidates(picks, now, deadline, sports, only_value, include_unverified)
    if c.empty:
        return None, [], 0
    plans = []
    if tipo in ("auto", "simple"):
        plans.append(best_single(c, risk, budget))
    if tipo in ("auto", "combinada"):
        plans.append(best_combo(c, risk, budget))
    if tipo in ("auto", "repartir"):
        plans.append(spread_budget(c, budget))
    plans = [p for p in plans if p is not None]
    if not plans:
        return None, [], len(c)
    if tipo == "auto":
        pmin, omin, omax = RISK_TARGET[risk]
        def fit(p: Plan) -> tuple:
            in_range = p.tipo == "repartir" or omin * 0.85 <= p.odds <= omax * 1.5
            return (in_range, p.ev, p.prob)
        plans.sort(key=fit, reverse=True)
    return plans[0], plans[1:], len(c)


def default_deadline(now: datetime, days: int = 2) -> datetime:
    return now + timedelta(days=days)


def pool_from_db(db, now: datetime, book: str = "bet365", min_ev: float = 0.03) -> pd.DataFrame:
    """Todas las selecciones futuras con cuota de bet365 y probabilidad estimada (no solo las de la portada)."""
    from .config import SPORTS
    from .model.markets import label
    from .value import risk_of

    df = db.df("""SELECT p.match_id, p.market, p.selection, p.point, p.p_final, p.p_model, p.p_market,
                         o.price AS odds, o.first_price, m.sport, m.league, m.kickoff, m.home, m.away
                  FROM predictions p
                  JOIN odds o ON o.match_id = p.match_id AND o.market = p.market AND o.selection = p.selection
                             AND o.point = p.point AND o.bookmaker = ?
                  JOIN matches m ON m.id = p.match_id
                  WHERE m.status = 'pendiente' AND m.kickoff > ? AND p.p_market IS NOT NULL""",
               (book, now.isoformat()))
    if df.empty:
        return df
    df["label"] = [label(r.market, r.selection, r.point, r.home, r.away, SPORTS.get(r.sport, SPORTS["football"])["unidad"])
                   for r in df.itertuples()]
    df["ev"] = df["p_final"] * df["odds"] - 1
    df["tier"] = df["ev"].map(lambda e: "valor" if e >= min_ev else "pronostico")
    df["risk"] = df["p_final"].map(risk_of)
    df["verified"] = 1
    df["confidence"] = "—"
    return df
