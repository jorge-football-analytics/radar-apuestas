"""Evaluación de cada selección SOLO con la cuota de bet365.

La probabilidad sale de mezclar el modelo propio con el consenso del mercado sin margen (Pinnacle pesa doble).
Niveles:
  valor      → cuota de bet365 confirmada y valor esperado ≥ MIN_EV
  verificar  → no tenemos la cuota de bet365, pero el resto del mercado sugiere valor: se indica la cuota mínima
  pronostico → la mejor opción del partido aunque no tenga ventaja (información, no recomendación)
élite (Telegram) → 'valor' + confianza alta + EV ≥ ALERT_EV + cuota y probabilidad razonables
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .config import DEFAULT_WEIGHT, SPORTS, Settings, sport_of
from .model.markets import SHARP, label, market_fair_probs

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

    def as_row(self) -> dict:
        return asdict(self)


def risk_of(p: float) -> str:
    return "bajo" if p >= 0.60 else "medio" if p >= 0.40 else "alto"


def kelly_stake(p: float, odds: float, s: Settings) -> float:
    edge = p * odds - 1
    if edge <= 0 or odds <= 1:
        return 0.0
    f = edge / (odds - 1) * s.kelly_fraction
    return round(min(f, s.max_stake_pct) * s.bankroll, 2)


def evaluate_match(match: dict, odds: pd.DataFrame, model_probs: dict | None, weights: dict[str, float],
                   s: Settings, notes: list[str] | None = None) -> tuple[list[Candidate], list[dict]]:
    """Devuelve (selecciones a mostrar, todas las predicciones del partido)."""
    sport = match.get("sport") or sport_of(match["league"])
    unit = SPORTS[sport]["unidad"]
    odds = odds[~odds["bookmaker"].isin(EXCLUDE_BOOKS)]
    fair = market_fair_probs(odds)
    preds, cands = [], []
    for (market, sel, point), (p_mkt, n_books) in fair.items():
        p_mod = model_probs.get((market, sel, point)) if model_probs else None
        w = weights.get(f"{sport}|{market}", DEFAULT_WEIGHT.get(sport, 0.2)) if p_mod is not None else 0.0
        p = w * p_mod + (1 - w) * p_mkt if p_mod is not None else p_mkt
        preds.append(dict(match_id=match["id"], market=market, selection=sel, point=point,
                          p_model=p_mod, p_market=p_mkt, p_final=p))
        g = odds[(odds["market"] == market) & (odds["selection"] == sel) & (odds["point"] == point)]
        b365 = g.loc[g["bookmaker"] == s.book, "price"]
        verified = not b365.empty
        if verified:
            price = float(b365.max())
        else:
            soft = g.loc[~g["bookmaker"].isin(SHARP), "price"]
            if soft.empty:
                continue
            price = float(np.median(soft))  # estimación de lo que pagaría una casa 'blanda' como bet365
        ev = p * price - 1
        ev_mod = (p_mod * price - 1) if p_mod is not None else None
        ev_mkt = p_mkt * price - 1
        if ev_mod is not None and ev_mod > 0 and ev_mkt > 0 and n_books >= 4:
            conf = "alta"
        elif ev_mod is not None and ev_mod > 0 and abs(p_mod - p_mkt) <= 0.12:
            conf = "media"
        elif ev_mod is None and ev_mkt > 0.02 and n_books >= 6:
            conf = "media"
        else:
            conf = "baja"
        ok = p >= s.min_prob and price <= s.max_odds and n_books >= s.min_books
        if verified and ev >= s.min_ev and ok:
            tier = "valor"
        elif not verified and ev >= s.min_ev and ok:
            tier = "verificar"
        else:
            tier = "pronostico"
        elite = int(tier == "valor" and conf == "alta" and ev >= s.alert_ev and p >= s.alert_min_prob
                    and s.alert_odds_min <= price <= s.alert_odds_max)
        score = ev * (1.0 if conf == "alta" else 0.7 if conf == "media" else 0.4) * min(1.0, n_books / 6)
        reasons = [f"Probabilidad estimada {p:.0%} → cuota justa {1 / p:.2f}"]
        if p_mod is not None:
            reasons.append(f"Modelo {p_mod:.0%} · mercado sin margen {p_mkt:.0%} ({n_books} casas, peso modelo {w:.0%})")
        else:
            reasons.append(f"Sin modelo propio: consenso de {n_books} casas {p_mkt:.0%}")
        reasons.append(f"{s.book} paga {price:.2f}" if verified else
                       f"Sin cuota de {s.book} todavía · casas similares pagan {price:.2f}")
        if notes:
            reasons += notes
        cands.append(Candidate(
            match["id"], sport, match["league"], match["kickoff"], match["home"], match["away"], market, sel, point,
            label(market, sel, point, match["home"], match["away"], unit), p_mod, p_mkt, p, price, s.book,
            price if verified else None, ev, kelly_stake(p, price, s) if tier == "valor" else 0.0, conf,
            " | ".join(reasons), n_books, risk_of(p), tier, int(verified), round((1 + s.min_ev) / p, 2), elite,
            round(score, 4), ev if verified else None))
    # valor / verificar: la mejor por mercado. pronóstico: la mejor del partido (si no hay nada mejor)
    out: dict[tuple, Candidate] = {}
    for c in cands:
        if c.tier == "pronostico":
            continue
        k = (c.market, c.point)
        if k not in out or c.ev > out[k].ev:
            out[k] = c
    if not out:
        pron = [c for c in cands if c.p_final >= 0.35 and c.ev >= -0.06 and c.n_books >= s.min_books]
        if pron:
            best = max(pron, key=lambda c: (c.verified, c.ev + 0.05 * c.p_final))
            out[("pron",)] = best
    return list(out.values()), preds
