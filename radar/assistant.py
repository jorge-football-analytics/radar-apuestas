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
    stake_pct: float = 0.0  # stake sugerido sobre la banca (copiloto)
    rationale: list[str] = field(default_factory=list)

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
            "risk", "ends", "confidence", "market", "why"]
    return [{k: r.get(k) for k in keep} for r in rows]


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


# ======================================================================= copiloto (v3)
COPILOT_RISK = {
    # riesgo: (prob. mínima por selección, cuota total mínima, cuota total máxima, máx. patas)
    "bajo": (0.65, 1.25, 2.20, 3),
    "medio": (0.50, 1.80, 4.00, 3),
    "alto": (0.35, 3.50, 12.0, 4),
}


def stake_advice(ev: float, odds: float, bankroll: float, kelly_fraction: float = 0.25, max_pct: float = 0.03
                 ) -> tuple[float, float, str]:
    """(stake €, % de la banca, explicación). Kelly fraccionado solo si hay ventaja; si no, como mucho un 0,5 %."""
    if ev > 0 and odds > 1:
        f = min(kelly_fraction * ev / (odds - 1), max_pct)
        return round(f * bankroll, 2), f, _es(f"Kelly al {kelly_fraction:.0%} con ventaja de {ev:+.1%}: "
                                               f"{f:.1%} de la banca (tope {max_pct:.0%})")
    f = 0.005
    return round(f * bankroll, 2), f, _es(f"Sin ventaja matemática (valor {ev:+.1%}): si apuestas, que sea ocio, "
                                           f"como mucho un {f:.1%} de la banca. Pérdida esperada "
                                           f"{abs(ev) * f * bankroll:.2f} €")


def _es(txt: str) -> str:
    """Decimales con coma (los textos van tal cual a la web)."""
    import re
    return re.sub(r"(\d)\.(\d)", r"\1,\2", txt)


def _leg_why(r: dict) -> list[str]:
    p, o = r["p_final"], r["odds"]
    out = [_es(f"Probabilidad real {p:.0%} frente al {1 / o:.0%} que implica la cuota {o:.2f} → valor {r['ev']:+.1%}")]
    if r.get("n_books") is not None and not pd.isna(r.get("n_books")):
        out.append(f"Precio justo con {int(r['n_books'])} casas válidas y {int(r.get('n_sharp') or 0)} sharp"
                   + (" (mercado derivado del 1X2/ganador)" if r.get("derived") else ""))
    out += list(r.get("why") or [])[:3]
    return out


def copilot(pool: pd.DataFrame, bankroll: float, deadline: datetime, now: datetime, risk: str = "medio",
            tipo: str = "auto", sports: list[str] | None = None, odds_range: tuple[float, float] | None = None,
            only_good: bool = False, kelly_fraction: float = 0.25, max_pct: float = 0.03
            ) -> tuple[Plan | None, list[Plan], int]:
    """La mejor apuesta simple o combinada para una banca, un riesgo, unos deportes y una cuota objetivo.
    Cada plan trae el stake sugerido (Kelly fraccionado o, sin ventaja, como mucho un 0,5 %) y el porqué con datos:
    probabilidad real frente a la cuota, referencia de mercado, forma, rachas, local/visitante y cara a cara."""
    c = candidates(pool, now, deadline, sports, False, False)
    if only_good and not c.empty:
        c = c[c["tier"].isin(["valor", "solida"])]
    if c.empty:
        return None, [], 0
    pmin, omin, omax, kmax = COPILOT_RISK[risk]
    if odds_range:
        omin, omax = odds_range
    plans: list[Plan] = []
    if tipo in ("auto", "simple"):
        pool1 = c[(c["odds"] >= omin) & (c["odds"] <= omax) & (c["p_final"] >= min(pmin, 1 / omax))]
        for _, r in pool1.assign(score=pool1["ev"] + 0.01 * pool1["p_final"]).sort_values(
                "score", ascending=False).drop_duplicates("match_id").head(3).iterrows():
            leg = _legs([r])[0]
            plans.append(Plan("simple", [leg], leg["odds"], leg["p_final"], leg["ev"], [0.0], 0.0))
    if tipo in ("auto", "combinada"):
        pool2 = c[(c["p_final"] >= pmin) & (c["odds"] >= 1.15)]
        pool2 = pool2.sort_values(["ev", "p_final"], ascending=False).drop_duplicates("match_id").head(14)
        recs = pool2.to_dict("records")
        found = []
        for k in range(2, kmax + 1):
            for combo in itertools.combinations(recs, k):
                if len({x["league"] for x in combo}) < k - 1:  # como mucho dos de la misma liga
                    continue
                o = prod(x["odds"] for x in combo)
                if not (omin <= o <= omax):
                    continue
                p = prod(x["p_final"] for x in combo)
                found.append((p * o - 1, p, o, combo))
        found.sort(key=lambda x: (x[0], x[1]), reverse=True)
        used: set = set()
        for ev, p, o, combo in found:
            ids = {x["match_id"] for x in combo}
            if ids & used:
                continue
            plans.append(Plan("combinada", _legs(combo), o, p, ev, [0.0], 0.0))
            used |= ids
            if sum(pl.tipo == "combinada" for pl in plans) >= 3:
                break
    if not plans:
        return None, [], len(c)
    plans.sort(key=lambda pl: (pl.ev, pl.prob), reverse=True)
    for pl in plans:
        stake, pct, txt = stake_advice(pl.ev, pl.odds, bankroll, kelly_fraction, max_pct)
        pl.budget, pl.stakes, pl.stake_pct = stake, [stake] * len(pl.legs), pct
        pl.rationale = [txt, f"Probabilidad real {'conjunta ' if pl.tipo == 'combinada' else ''}{pl.prob:.0%} · "
                             f"la cuota implica {1 / pl.odds:.0%}"]
        for leg in pl.legs:
            src = c[(c["match_id"] == leg["match_id"]) & (c["label"] == leg["label"])]
            leg["why"] = _leg_why(src.iloc[0].to_dict() if not src.empty else {**leg, "n_books": None})
    return plans[0], plans[1:], len(c)


def default_deadline(now: datetime, days: int = 2) -> datetime:
    return now + timedelta(days=days)


def pool_from_db(db, now: datetime, book: str = "bet365", min_ev: float = 0.025, solid_min_prob: float = 0.65,
                 solid_min_ev: float = -0.005, max_age_h: float = 72) -> pd.DataFrame:
    """Todas las selecciones futuras con cuota de bet365 (de menos de max_age_h horas) y probabilidad estimada."""
    from .config import SPORTS
    from .model.markets import label
    from .value import risk_of

    def _has_table(db, name):
        return db.conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone() is not None

    sel = db.df("""SELECT s.*, s.price AS odds, m.sport, m.league, m.kickoff, m.home, m.away, i.info
                   FROM selections s JOIN matches m ON m.id = s.match_id
                   LEFT JOIN match_info i ON i.match_id = s.match_id
                   WHERE m.status = 'pendiente' AND m.kickoff > ? AND s.verified = 1 AND s.tier != 'revisar'
                     AND COALESCE(s.weak, 0) = 0 AND (s.b365_age_h IS NULL OR s.b365_age_h <= ?)""",
                (now.isoformat(), max_age_h)) if _has_table(db, "selections") else pd.DataFrame()
    if not sel.empty:
        import json as _json

        from .stats import why_lines
        sel["risk"] = sel["p_final"].map(risk_of)
        sel["confidence"] = "—"
        sel["ev"] = sel["p_final"] * sel["odds"] - 1
        sel["why"] = [why_lines(l, mk, sl, (_json.loads(i) if isinstance(i, str) else {}).get("stats") or {},
                                SPORTS.get(sp, SPORTS["football"])["unidad"])
                      for l, mk, sl, i, sp in zip(sel["label"], sel["market"], sel["selection"], sel["info"], sel["sport"])]
        return sel.drop(columns=["info"])
    df = db.df("""SELECT p.match_id, p.market, p.selection, p.point, p.p_final, p.p_model, p.p_market,
                         o.price AS odds, o.first_price, o.fetched_at, m.sport, m.league, m.kickoff, m.home, m.away
                  FROM predictions p
                  JOIN odds o ON o.match_id = p.match_id AND o.market = p.market AND o.selection = p.selection
                             AND o.point = p.point AND o.bookmaker = ?
                  JOIN matches m ON m.id = p.match_id
                  WHERE m.status = 'pendiente' AND m.kickoff > ? AND p.p_market IS NOT NULL""",
               (book, now.isoformat()))
    if df.empty:
        return df
    t = pd.to_datetime(df["fetched_at"], utc=True, format="ISO8601", errors="coerce")
    df = df[t.isna() | (t >= pd.Timestamp(now) - pd.Timedelta(hours=max_age_h))]
    if df.empty:
        return df
    df["label"] = [label(r.market, r.selection, r.point, r.home, r.away, SPORTS.get(r.sport, SPORTS["football"])["unidad"])
                   for r in df.itertuples()]
    df["ev"] = df["p_final"] * df["odds"] - 1
    df["tier"] = ["valor" if e >= min_ev else "solida" if (p >= solid_min_prob and e >= solid_min_ev) else "pronostico"
                  for e, p in zip(df["ev"], df["p_final"])]
    df["risk"] = df["p_final"].map(risk_of)
    df["verified"] = 1
    df["confidence"] = "—"
    return df
