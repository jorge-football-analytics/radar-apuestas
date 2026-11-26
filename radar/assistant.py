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
from math import ceil, prod

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
    role: str = "principal"  # principal · conservadora · retorno · cercana
    status: str = "exacta"  # exacta (cuotas confirmadas) · por_confirmar · cercana (fuera del filtro)
    in_range: bool = True

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
            "risk", "ends", "confidence", "market", "why", "priced", "min_odds"]
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


# ======================================================================= copiloto (v5)
# Rango de cuota TOTAL que pide cada nivel de riesgo cuando el usuario no fija otro. Se respeta siempre.
RISK_RANGE = {"bajo": (1.40, 1.90), "medio": (1.90, 2.80), "alto": (2.80, 5.00)}
RISK_LEGS = {"bajo": 2, "medio": 3, "alto": 4}  # patas máximas de una combinada
COPILOT_RISK = {k: (0.0, lo, hi, RISK_LEGS[k]) for k, (lo, hi) in RISK_RANGE.items()}  # compatibilidad


def stake_advice(ev: float, odds: float, bankroll: float, kelly_fraction: float = 0.25, max_pct: float = 0.03
                 ) -> tuple[float, float, str]:
    """(stake €, % de la banca, explicación). Kelly fraccionado si hay ventaja; a precio justo (valor 0) como mucho
    un 0,5 %; con valor negativo, nada."""
    if ev > 0.0005 and odds > 1:
        f = min(kelly_fraction * ev / (odds - 1), max_pct)
        return round(f * bankroll, 2), f, _es(f"Kelly al {kelly_fraction:.0%} con ventaja de {ev:+.1%}: "
                                               f"{f:.1%} de la banca (tope {max_pct:.0%})")
    if ev >= -0.0005:
        f = 0.005
        return round(f * bankroll, 2), f, _es(f"A precio justo (sin ventaja ni margen): como mucho un {f:.1%} de "
                                               f"la banca")
    return 0.0, 0.0, _es(f"Valor negativo ({ev:+.1%}): no se recomienda apostar")


def _es(txt: str) -> str:
    """Decimales con coma (los textos van tal cual a la web)."""
    import re
    return re.sub(r"(\d)\.(\d)", r"\1,\2", txt)


def _leg_why(r: dict) -> list[str]:
    """Informe de la selección (scouting + precio). Si la selección no lo trae, al menos el porqué de la cuota."""
    if r.get("why"):
        return list(r["why"])
    from .stats import price_line
    return [price_line(float(r["p_final"]), float(r["odds"]), bool(r.get("priced", True)))]


def _grade(r: dict) -> int:
    """0 = cuota de bet365 confirmada con valor ≥ 0 · 1 = por confirmar (cogerla a la cuota mínima) · 2 = valor < 0."""
    if r.get("priced", True):
        return 0 if r["ev"] >= -0.0005 else 2
    return 1


def _options(rows: list[dict], kmax: int, tipo: str) -> list[dict]:
    """Todas las apuestas posibles (simples y combinadas de 2..kmax patas de partidos distintos, como mucho dos de la
    misma liga) con su cuota, probabilidad, valor y nivel de confianza en la cuota (grado)."""
    opts = []
    if tipo in ("auto", "simple"):
        for r in rows:
            opts.append(dict(legs=[r], odds=r["odds"], prob=r["p_final"], ev=r["ev"], grade=_grade(r)))
    if tipo in ("auto", "combinada"):
        for k in range(2, kmax + 1):
            for combo in itertools.combinations(rows, k):
                if len({x["match_id"] for x in combo}) < k or len({x["league"] for x in combo}) < k - 1:
                    continue
                o = prod(x["odds"] for x in combo)
                if o > 15:
                    continue
                p = prod(x["p_final"] for x in combo)
                opts.append(dict(legs=list(combo), odds=o, prob=p, ev=p * o - 1, grade=max(_grade(x) for x in combo)))
    return opts


def _best(opts: list[dict], key) -> dict | None:
    return max(opts, key=key) if opts else None


def copilot(pool: pd.DataFrame, bankroll: float, deadline: datetime, now: datetime, risk: str = "medio",
            tipo: str = "auto", sports: list[str] | None = None, odds_range: tuple[float, float] | None = None,
            only_good: bool = False, kelly_fraction: float = 0.25, max_pct: float = 0.03
            ) -> tuple[Plan | None, list[Plan], int]:
    """Construye la mejor apuesta para una banca, un riesgo, unos deportes y una cuota objetivo.

    Reglas (en este orden):
      1. La cuota TOTAL siempre dentro del rango pedido (odds_range, o el del riesgo: bajo 1,40-1,90 · medio
         1,90-2,80 · alto 2,80-5,00).
      2. Primero, solo cuotas de bet365 confirmadas con valor ≥ 0 (ventaja o precio justo sin margen).
      3. Si no hay, se añaden las selecciones «por confirmar» (sin cuota de bet365 todavía): se indica la cuota
         mínima a la que hay que cogerlas para no apostar por debajo del precio justo.
      4. Nunca se recomienda una apuesta con valor negativo como opción principal. Si nada cumple el filtro, se dice y
         se muestra la alternativa más cercana, marcada como tal.
    Devuelve (opción principal, [más conservadora, mayor retorno], nº de selecciones consideradas)."""
    lo, hi = odds_range or RISK_RANGE.get(risk, RISK_RANGE["medio"])
    c = candidates(pool, now, deadline, sports, False, True)
    if c.empty:
        return None, [], 0
    if "priced" not in c:
        c = c.assign(priced=c["verified"].astype(bool))
    c = c.assign(priced=c["priced"].fillna(c["verified"]).astype(bool))
    if only_good:
        c = c[(~c["priced"]) | (c["tier"].isin(["valor", "solida", "moderado", "cuota_alta"])) | (c["ev"] >= 0)]
    c = c.sort_values(["ev", "p_final"], ascending=False).drop_duplicates(["match_id", "label"])
    # pool de patas: las mejores por partido de cada grado (para que la búsqueda sea rápida)
    recs = c.to_dict("records")
    for r in recs:
        if not r["priced"]:
            # redondeo hacia arriba: a la cuota mínima indicada nunca se pierde valor (≥ precio justo)
            r["min_odds"] = ceil(100 / r["p_final"] - 1e-9) / 100
            r["odds"], r["ev"] = r["min_odds"], r["p_final"] * r["min_odds"] - 1
    good = [r for r in recs if _grade(r) == 0]
    pend = [r for r in recs if _grade(r) == 1]
    bad = [r for r in recs if _grade(r) == 2]
    kmax = RISK_LEGS.get(risk, 3)

    def top(rows, n=14):
        best: dict[str, dict] = {}
        for r in sorted(rows, key=lambda r: (r["ev"], r["p_final"]), reverse=True):
            best.setdefault(r["match_id"], r)
        return list(best.values())[:n]

    in_r = lambda o: lo - 1e-9 <= o["odds"] <= hi + 1e-9
    score = lambda o: (round(o["ev"], 4), o["prob"], -len(o["legs"]))
    main, status = None, "exacta"
    for stage, rows in (("exacta", top(good)), ("por_confirmar", top(good) + top(pend))):
        opts = [o for o in _options(rows, kmax, tipo) if in_r(o)]
        if opts:
            main, status = _best(opts, score), stage
            break
    all_good = _options(top(good) + top(pend), kmax, tipo)
    plans: list[Plan] = []
    if main is None:  # nada cumple: la alternativa más cercana al rango (preferible sin valor negativo)
        cand = all_good or _options(top(bad), kmax, tipo)
        if not cand:
            return None, [], len(c)
        dist = lambda o: max(lo - o["odds"], o["odds"] - hi, 0)
        main = min(cand, key=lambda o: (dist(o), o["grade"], -o["ev"]))
        status = "cercana"
    plans.append(_plan(main, "principal" if status != "cercana" else "cercana", status, in_r(main), lo, hi))
    # alternativas: una más conservadora (menos cuota, más probabilidad) y otra de mayor retorno
    base = all_good if all_good else []
    cons = [o for o in base if 1.20 <= o["odds"] < main["odds"] - 1e-9 and o["prob"] > main["prob"]]
    ret_all = [o for o in base if main["odds"] + 1e-9 < o["odds"] <= max(hi * 1.6, main["odds"] * 1.8)]
    ret = [o for o in ret_all if o["odds"] >= main["odds"] * 1.25] or ret_all  # retorno claramente mayor si existe
    pick_c = _best(cons, lambda o: (o["grade"] == 0, round(o["ev"], 4), o["prob"]))
    pick_r = _best(ret, lambda o: (o["grade"] == 0, round(o["ev"], 4), o["prob"]))
    if pick_c:
        plans.append(_plan(pick_c, "conservadora", "exacta" if pick_c["grade"] == 0 else "por_confirmar",
                           in_r(pick_c), lo, hi))
    if pick_r:
        plans.append(_plan(pick_r, "retorno", "exacta" if pick_r["grade"] == 0 else "por_confirmar", in_r(pick_r), lo, hi))
    for pl in plans:
        stake, pct, txt = stake_advice(pl.ev, pl.odds, bankroll, kelly_fraction, max_pct)
        pl.budget, pl.stakes, pl.stake_pct = stake, [stake] * len(pl.legs), pct
        pl.rationale = [txt] + pl.rationale
        for leg in pl.legs:
            leg["why"] = _leg_why(leg)
    return plans[0], plans[1:], len(c)


def _plan(o: dict, role: str, status: str, in_range: bool, lo: float, hi: float) -> Plan:
    legs = _legs(o["legs"])
    pl = Plan("simple" if len(legs) == 1 else "combinada", legs, o["odds"], o["prob"], o["ev"], [0.0], 0.0,
              role=role, status=status, in_range=in_range)
    rng = _es(f"{lo:.2f}-{hi:.2f}")
    notes = []
    if status == "cercana":
        notes.append(_es(f"Ninguna apuesta cumple a la vez tu cuota objetivo ({rng}) y un valor ≥ 0 con lo analizado. "
                         f"Esta es la alternativa más cercana (cuota {o['odds']:.2f})"
                         + (", con valor negativo: no se recomienda." if o["ev"] < -0.0005 else ".")))
    elif status == "por_confirmar":
        notes.append("Incluye selecciones sin cuota de bet365 todavía: cógelas solo a la cuota mínima indicada (≥) "
                     "o superior; por debajo, la apuesta deja de tener sentido.")
    if role in ("conservadora", "retorno") and not in_range:
        notes.append(_es(f"Fuera de tu rango ({rng}) a propósito: "
                         + ("menos cuota y más probabilidad." if role == "conservadora" else "más cuota y más retorno.")))
    pl.rationale = notes + [_es(f"Probabilidad real {'conjunta ' if len(legs) > 1 else ''}{o['prob']:.0%} · la cuota "
                                f"implica {1 / o['odds']:.0%}")]
    return pl


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
                   WHERE m.status = 'pendiente' AND m.kickoff > ? AND s.tier != 'revisar' AND COALESCE(s.weak, 0) = 0
                     AND (COALESCE(s.n_books, 0) >= 2 OR COALESCE(s.n_sharp, 0) >= 1)
                     AND (s.verified = 0 OR s.b365_age_h IS NULL OR s.b365_age_h <= ?)""",
                (now.isoformat(), max_age_h)) if _has_table(db, "selections") else pd.DataFrame()
    if not sel.empty:
        import json as _json

        from .combos import leg_report
        tables = {r["key"][6:]: _json.loads(r["value"]) for r in db.conn.execute(
            "SELECT key, value FROM kv WHERE key LIKE 'tabla_%'").fetchall()}
        sel["priced"] = sel["verified"].fillna(0).astype(int) == 1
        sel.loc[~sel["priced"], "odds"] = (1 / sel.loc[~sel["priced"], "p_final"]).round(2)  # cuota mínima = justa
        sel["risk"] = sel["p_final"].map(risk_of)
        sel["confidence"] = "—"
        sel["ev"] = sel["p_final"] * sel["odds"] - 1
        sel["why"] = [leg_report(r, r["sport"], r["market"], r["selection"], float(r["point"]),
                                 _json.loads(r["info"]) if isinstance(r["info"], str) else {}, tables, float(r["p_final"]),
                                 float(r["odds"]) if r["priced"] else None, bool(r["priced"]), r.get("n_books"),
                                 r.get("n_sharp"), bool(r.get("derived")))
                      for r in sel.to_dict("records")]
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
