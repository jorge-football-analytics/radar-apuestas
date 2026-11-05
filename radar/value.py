"""Detector de apuestas de valor: compara la probabilidad estimada con las cuotas de todas las casas."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from .config import Settings
from .model.markets import label, market_fair_probs

EXCLUDE_BOOKS = {"max", "avg"}


@dataclass
class Candidate:
    match_id: str
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
    odds: float
    bookmaker: str
    b365_odds: float | None
    ev: float
    ev_b365: float | None
    stake: float
    confidence: str
    reasons: str
    n_books: int

    def as_row(self) -> dict:
        return asdict(self)


def kelly_stake(p: float, odds: float, s: Settings) -> float:
    edge = p * odds - 1
    if edge <= 0:
        return 0.0
    f = edge / (odds - 1) * s.kelly_fraction
    return round(min(f, s.max_stake_pct) * s.bankroll, 2)


def evaluate_match(match: dict, odds: pd.DataFrame, model_probs: dict | None, weights: dict[str, float],
                   s: Settings, notes: list[str] | None = None) -> tuple[list[Candidate], list[dict]]:
    """Devuelve (candidatos con valor, todas las predicciones de este partido)."""
    odds = odds[~odds["bookmaker"].isin(EXCLUDE_BOOKS)]
    fair = market_fair_probs(odds)
    preds, cands = [], []
    for (market, sel, point), (p_mkt, n_books) in fair.items():
        p_mod = model_probs.get((market, sel, point)) if model_probs else None
        w = weights.get(market, s.model_weight) if p_mod is not None else 0.0
        p = w * p_mod + (1 - w) * p_mkt if p_mod is not None else p_mkt
        preds.append(dict(match_id=match["id"], market=market, selection=sel, point=point,
                          p_model=p_mod, p_market=p_mkt, p_final=p))
        g = odds[(odds["market"] == market) & (odds["selection"] == sel) & (odds["point"] == point)]
        b365 = g[g["bookmaker"] == s.preferred_book]["price"]
        g = g[g["bookmaker"].map(s.can_bet)]
        if g.empty:
            continue
        best = g.loc[g["price"].idxmax()]
        b365_odds = float(b365.max()) if not b365.empty else None
        price = float(best["price"])
        ev = p * price - 1
        if ev < s.min_ev or p < s.min_prob or price > s.max_odds or n_books < s.min_books:
            continue
        # Confianza: ¿coinciden modelo y mercado en que la cuota está alta?
        ev_mod = (p_mod * price - 1) if p_mod is not None else None
        ev_mkt = p_mkt * price - 1
        if ev_mod is not None and ev_mod > 0 and ev_mkt > 0 and n_books >= 4:
            conf = "alta"
        elif ev_mod is not None and ev_mod > 0 and abs(p_mod - p_mkt) <= 0.12:
            conf = "media"
        elif ev_mod is None and ev_mkt > 0.03 and n_books >= 6:
            conf = "media"
        else:
            conf = "baja"
        reasons = [f"Probabilidad estimada {p:.0%} (cuota justa {1 / p:.2f})"]
        if p_mod is not None:
            reasons.append(f"Modelo {p_mod:.0%} · mercado sin margen {p_mkt:.0%} ({n_books} casas)")
        else:
            reasons.append(f"Sin modelo para este partido: consenso de {n_books} casas {p_mkt:.0%}")
        reasons.append(f"Mejor cuota {price:.2f} en {best['bookmaker']}"
                       + (f" · {s.preferred_book} {b365_odds:.2f}" if b365_odds else f" · {s.preferred_book} sin cuota"))
        if notes:
            reasons += notes
        cands.append(Candidate(
            match["id"], match["league"], match["kickoff"], match["home"], match["away"], market, sel, point,
            label(market, sel, point, match["home"], match["away"]), p_mod, p_mkt, p, price, str(best["bookmaker"]),
            b365_odds, ev, (p * b365_odds - 1) if b365_odds else None, kelly_stake(p, price, s), conf,
            " | ".join(reasons), n_books))
    # Una sola selección por mercado y partido (la de mayor valor)
    best_per_market: dict[tuple, Candidate] = {}
    for c in cands:
        k = (c.market, c.point)
        if k not in best_per_market or c.ev > best_per_market[k].ev:
            best_per_market[k] = c
    return list(best_per_market.values()), preds
