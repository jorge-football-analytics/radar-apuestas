"""Evaluación de cada selección SOLO con la cuota de bet365.

La probabilidad de referencia es el mercado sin margen, calculado SOLO con casas válidas (mercado completo, margen
creíble y cuotas tomadas a la misma hora que la de bet365). Si hay casas sharp (Pinnacle, exchanges) se usan solo
ellas. bet365 nunca entra en su propio consenso. El modelo propio solo pesa en los deportes no congelados.
Niveles (todos con cuota entre MIN_ODDS y MAX_ODDS, por defecto 1,30-2,60):
  valor      → «Élite» en la web. Cuota de bet365 confirmada, reciente y coherente, valor esperado entre MIN_EV y MAX_EV
  solida     → «Sólida». Probabilidad ≥ SOLID_MIN_PROB y bet365 paga al menos el precio justo limpio
               (valor esperado ≥ SOLID_MIN_EV, −0,5 %). No es ventaja: es no regalar margen. Base de las combinadas
  verificar  → falta la cuota de bet365 (o es vieja) y el resto del mercado sugiere valor: se indica la cuota mínima
  revisar    → valor esperado > MAX_EV: casi siempre es un error de datos. Nunca se apuesta (stake 0)
  pronostico → la mejor opción del partido aunque no tenga ventaja (información, no recomendación)
Telegram (elite=1) → 'valor' + referencia sharp (Pinnacle/exchange) + EV ≥ ALERT_EV + cuota y probabilidad razonables
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from .config import DEFAULT_WEIGHT, SPORTS, Settings, sport_of
from .model.derived import derived_probs
from .model.markets import GROUPS, SHARP, book_market_ok, label, market_fair_probs, odds_times

EXCLUDE_BOOKS = {"max", "avg"}


@dataclass
class Candidate:
    match_id: str
    sport: str
    league: str
    kickoff: str
    home: str
    away: str
    market: str
    selection: str
    point: float
    label: str
    p_model: float | None
    p_market: float
    p_final: float
    odds: float  # cuota de bet365 (o estimada si tier = verificar)
    bookmaker: str
    b365_odds: float | None
    ev: float
    stake: float
    confidence: str
    reasons: str
    n_books: int
    risk: str
    tier: str
    verified: int
    min_odds: float
    elite: int
    score: float
    ev_b365: float | None = None
    n_sharp: int = 0

    def as_row(self) -> dict:
        return asdict(self)


def risk_of(p: float) -> str:
    return "bajo" if p >= 0.60 else "medio" if p >= 0.40 else "alto"


def kelly_stake(p: float, odds: float, s: Settings) -> float:
    edge = p * odds - 1
    if edge <= 0 or odds <= 1 or edge > s.max_ev:  # un 'valor' enorme es un error de datos: nunca se apuesta
        return 0.0
    f = edge / (odds - 1) * s.kelly_fraction
    return round(min(f, s.max_stake_pct) * s.bankroll, 2)


def model_weight(sport: str, market: str, weights: dict[str, float], s: Settings) -> float:
    if sport in s.frozen_sports:
        return 0.0
    return float(weights.get(f"{sport}|{market}", DEFAULT_WEIGHT.get(sport, 0.2)))


def _b365_markets(odds: pd.DataFrame, s: Settings) -> tuple[dict, dict]:
    """Por (mercado, línea): ¿el mercado de bet365 está completo y es coherente? y hora de la cuota de bet365."""
    ok, ref = {}, {}
    b = odds[odds["bookmaker"] == s.book]
    if b.empty:
        return ok, ref
    times = odds_times(b)
    for (market, point), g in b.groupby(["market", "point"]):
        pr = g.drop_duplicates("selection", keep="last").set_index("selection")["price"]
        if market == "dc":  # doble oportunidad: las tres suman ~2 (cada resultado está en dos selecciones)
            complete = all(x in pr for x in ("1X", "12", "X2"))
            ok[(market, float(point))] = complete and 1.98 <= sum(1 / float(pr[x]) for x in ("1X", "12", "X2")) <= 2.30
        else:
            sels = GROUPS.get(market) or []
            complete = bool(sels) and all(x in pr for x in sels)
            ok[(market, float(point))] = complete and book_market_ok(
                s.book, [float(pr[x]) for x in sels], s.max_overround_b365, s.exchange_overround)
        if times is not None:
            ref[(market, float(point))] = times.loc[g.index].max()
    return ok, ref


def evaluate_match(match: dict, odds: pd.DataFrame, model_probs: dict | None, weights: dict[str, float],
                   s: Settings, notes: list[str] | None = None, now: datetime | None = None,
                   context: dict | None = None) -> tuple[list[Candidate], list[dict]]:
    """Devuelve (selecciones a mostrar, todas las predicciones del partido)."""
    sport = match.get("sport") or sport_of(match["league"])
    unit = SPORTS[sport]["unidad"]
    odds = odds[~odds["bookmaker"].isin(EXCLUDE_BOOKS)]
    b365_ok, b365_time = _b365_markets(odds, s)
    valid: dict = {}
    fair = market_fair_probs(odds, exclude={s.book}, ref_times=b365_time, max_gap_h=s.max_odds_gap_h,
                             max_overround=s.max_overround, exchange_band=s.exchange_overround, valid_books=valid)
    # mercados derivados del precio justo (doble oportunidad, más/menos, ambos marcan, hándicap): misma referencia
    # (y mismas casas válidas) que el mercado principal del que salen
    context = context or {}
    lines = {mk: sorted(set(odds.loc[odds["market"] == mk, "point"].astype(float))) for mk in ("ou", "spread")}
    derived = derived_probs(sport, match["league"], fair, lines, context.get("goal_avg"), context.get("model_total"))
    base_key = ("1x2", "H", 0.0) if sport == "football" else ("ml", "H", 0.0)
    derived_keys = set()
    goal_line_known = ("ou", "O", 2.5) in fair  # línea real de goles del mercado (si no, el total es una suposición)
    if derived and base_key in fair:
        _, nb, ns = fair[base_key]
        for k, p_d in derived.items():
            if k not in fair and 0.0 < p_d < 1.0:
                fair[k] = (p_d, nb, ns)
                derived_keys.add(k)
    now_ts = pd.Timestamp(now) if now is not None else None
    if now_ts is not None and now_ts.tzinfo is None:
        now_ts = now_ts.tz_localize("UTC")
    preds, cands = [], []
    for (market, sel, point), (p_mkt, n_books, n_sharp) in fair.items():
        is_derived = (market, sel, point) in derived_keys
        p_mod = model_probs.get((market, sel, point)) if model_probs else None
        w = model_weight(sport, market, weights, s) if p_mod is not None else 0.0
        p = w * p_mod + (1 - w) * p_mkt if p_mod is not None else p_mkt
        preds.append(dict(match_id=match["id"], market=market, selection=sel, point=point,
                          p_model=p_mod, p_market=p_mkt, p_final=p))
        g = odds[(odds["market"] == market) & (odds["selection"] == sel) & (odds["point"] == point)]
        b365 = g.loc[g["bookmaker"] == s.book, "price"]
        b365_bad = not b365.empty and not b365_ok.get((market, point), False)
        verified = not b365.empty and not b365_bad
        age_h = None
        if verified:
            price = float(b365.max())
            t = b365_time.get((market, point))
            if now_ts is not None and t is not None and pd.notna(t):
                age_h = (now_ts - t).total_seconds() / 3600
        else:
            soft = g.loc[~g["bookmaker"].isin(SHARP) & (g["bookmaker"] != s.book), "price"]
            if soft.empty and not is_derived:
                continue
            # estimación de lo que pagaría una casa 'blanda' como bet365 (mercado derivado sin cuotas: precio justo)
            price = float(np.median(soft)) if not soft.empty else round(1 / max(p_mkt, 1e-6), 3)
        stale = age_h is not None and age_h > s.b365_max_age_h
        ev = p * price - 1
        ev_mod = (p_mod * price - 1) if p_mod is not None else None
        ev_mkt = p_mkt * price - 1
        if ev_mkt > 0 and n_sharp >= 1 and n_books >= 3 and (ev_mod is None or ev_mod > 0):
            conf = "alta"
        elif ev_mkt > 0 and (n_books >= s.min_books or n_sharp >= 1):
            conf = "media"
        else:
            conf = "baja"
        # referencia suficiente: MIN_BOOKS casas válidas, o Pinnacle válido y a la misma hora (vale por sí solo:
        # es justo lo que trae OddsPapi en la misma consulta que bet365)
        enough = n_books >= s.min_books or "pinnacle" in valid.get((market, point), set())
        in_range = s.min_odds <= price <= s.max_odds
        ok = p >= s.min_prob and in_range and enough
        solid_fresh = age_h is None or age_h <= s.solid_b365_max_age_h
        # un mercado derivado solo puede ser Élite con referencia sharp y, si depende del número de goles, con una
        # línea real de más/menos del mercado (la doble oportunidad sale exacta del 1X2 y no lo necesita)
        # NBA/NFL: el hándicap y el total derivados del moneyline usan una normal sobre el margen, demasiado gruesa
        # para declarar ventaja (en la NFL los márgenes se concentran en 3 y 7): solo información
        weak_derived = is_derived and (n_sharp < 1 or (sport == "football" and market in ("ou", "btts", "spread")
                                                       and not goal_line_known) or sport in ("basketball", "nfl"))
        if ev >= s.min_ev and ok and weak_derived:
            tier = "verificar" if ev <= s.max_ev else "revisar"
        elif ev >= s.min_ev and ok:
            if ev > s.max_ev:
                tier = "revisar"
            elif verified and not stale:
                tier = "valor"
            else:
                tier = "verificar"
        elif ev > s.max_ev and verified and enough:
            tier = "revisar"
        elif (verified and enough and in_range and solid_fresh and p >= s.solid_min_prob
              and not weak_derived
              and not (is_derived and sport == "football" and market in ("ou", "btts", "spread") and not goal_line_known)
              and s.solid_min_ev <= ev <= s.max_ev):
            tier = "solida"
        else:
            tier = "pronostico"
        elite = int(tier == "valor" and not is_derived and conf == "alta" and n_sharp >= 1 and ev >= s.alert_ev
                    and p >= s.alert_min_prob and s.alert_odds_min <= price <= s.alert_odds_max)
        score = ev * (1.0 if conf == "alta" else 0.7 if conf == "media" else 0.4) * min(1.0, n_books / 4)
        reasons = [f"Probabilidad estimada {p:.0%} → cuota justa {1 / p:.2f}"]
        ref_txt = f"{n_sharp} sharp" if n_sharp else "sin casas sharp"
        if is_derived:
            reasons.append(f"Probabilidad derivada del precio justo del {'1X2' if sport == 'football' else 'ganador'}"
                           f"{' y del más/menos' if sport == 'football' else ''} ({n_books} casas válidas, {ref_txt}): "
                           f"{'modelo de goles Poisson' if sport == 'football' else 'normal sobre margen y total'}")
        elif p_mod is not None:
            reasons.append(f"Modelo {p_mod:.0%} · mercado sin margen {p_mkt:.0%} ({n_books} casas válidas, {ref_txt}, "
                           f"peso modelo {w:.0%})")
        else:
            reasons.append(f"Sin modelo propio: mercado sin margen {p_mkt:.0%} ({n_books} casas válidas, {ref_txt})")
        if verified:
            reasons.append(f"{s.book} paga {price:.2f}" + (f" (cuota de hace {age_h:.0f} h: confírmala)" if stale else ""))
        elif b365_bad:
            reasons.append(f"La cuota de {s.book} es incoherente (mercado incompleto o margen imposible) · "
                           f"casas similares pagan {price:.2f}")
        else:
            reasons.append(f"Sin cuota de {s.book} todavía · casas similares pagan {price:.2f}")
        if weak_derived and tier == "verificar":
            reasons.append("Referencia débil para este mercado (sin casa sharp o sin línea de goles del mercado): "
                           "confírmalo antes de apostar")
        if tier == "revisar":
            reasons.append(f"Valor esperado {ev:+.0%} demasiado alto: casi seguro es un error de datos. No apostar")
        if tier == "solida":
            reasons.append("Sólida: alta probabilidad y bet365 paga el precio justo (sin ventaja, sin regalar margen)"
                           + (f" · cuota de hace {age_h:.0f} h: confírmala" if age_h and age_h > s.b365_max_age_h else ""))
        if notes:
            reasons += notes
        cands.append(Candidate(
            match["id"], sport, match["league"], match["kickoff"], match["home"], match["away"], market, sel, point,
            label(market, sel, point, match["home"], match["away"], unit), p_mod, p_mkt, p, price, s.book,
            price if verified else None, ev, kelly_stake(p, price, s) if tier in ("valor", "solida") else 0.0, conf,
            " | ".join(reasons), n_books, risk_of(p), tier, int(verified), round((1 + s.min_ev) / p, 2), elite,
            round(score, 4), ev if verified else None, n_sharp))
        goal_guess = is_derived and sport == "football" and market in ("ou", "btts", "spread") and not goal_line_known
        preds[-1].update(label=cands[-1].label, price=price, verified=int(verified), ev=ev, tier=tier, derived=int(is_derived),
                         weak=int(bool(weak_derived or goal_guess)),
                         n_books=n_books, n_sharp=n_sharp, confidence=conf, b365_age_h=age_h,
                         min_odds=cands[-1].min_odds)
    # valor / verificar / revisar: la mejor por mercado. pronóstico: la mejor del partido (si no hay nada mejor)
    out: dict[tuple, Candidate] = {}
    for c in cands:
        if c.tier == "pronostico":
            continue
        k = (c.market, c.point)
        if k not in out or c.ev > out[k].ev:
            out[k] = c
    if not any(c.tier in ("valor", "verificar", "solida") for c in out.values()):
        pron = [c for c in cands if c.tier == "pronostico" and c.p_final >= 0.35 and c.ev >= -0.06
                and (c.n_books >= s.min_books or c.n_sharp >= 1)]
        if pron:
            best = max(pron, key=lambda c: (c.verified, c.ev + 0.05 * c.p_final))
            out[("pron",)] = best
    return list(out.values()), preds


def model_only(match: dict, probs: dict, s: Settings, notes: list[str] | None = None) -> tuple[list[Candidate], list[dict]]:
    """Partido sin cuotas todavía: pronóstico con la probabilidad del modelo y la cuota mínima a exigir en bet365."""
    sport = match.get("sport") or sport_of(match["league"])
    unit = SPORTS[sport]["unidad"]
    preds = [dict(match_id=match["id"], market=m, selection=sel, point=pt, p_model=p, p_market=None, p_final=p)
             for (m, sel, pt), p in probs.items()]
    main = {k: v for k, v in probs.items() if k[0] in ("1x2", "ml")}
    if not main:
        return [], preds
    (market, sel, point), p = max(main.items(), key=lambda kv: kv[1])
    fair = 1 / p
    reasons = [f"Aún no hay cuotas: probabilidad del modelo {p:.0%} → cuota justa {fair:.2f}",
               f"Apuesta solo si {s.book} paga {(1 + s.min_ev) / p:.2f} o más"] + (notes or [])
    c = Candidate(match["id"], sport, match["league"], match["kickoff"], match["home"], match["away"], market, sel,
                  point, label(market, sel, point, match["home"], match["away"], unit), p, float("nan"), p,
                  round(fair, 2), s.book, None, 0.0, 0.0, "modelo", " | ".join(reasons), 0, risk_of(p),
                  "pronostico", 0, round((1 + s.min_ev) / p, 2), 0, 0.0)
    return [c], preds
