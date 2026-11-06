"""Probabilidades por mercado a partir de la matriz de marcadores, y probabilidades 'justas' del mercado."""
from __future__ import annotations

import numpy as np
import pandas as pd

# Selecciones que forman un mercado completo (para quitar el margen de la casa)
GROUPS = {"1x2": ["H", "D", "A"], "ou": ["O", "U"], "btts": ["Y", "N"], "ml": ["H", "A"], "spread": ["H", "A"]}


def market_probs(m: np.ndarray, points: list[float] | None = None) -> dict[tuple[str, str, float], float]:
    """{(mercado, selección, línea): probabilidad}"""
    points = points or [1.5, 2.5, 3.5]
    i, j = np.indices(m.shape)
    total = i + j
    out = {
        ("1x2", "H", 0.0): float(m[i > j].sum()),
        ("1x2", "D", 0.0): float(m[i == j].sum()),
        ("1x2", "A", 0.0): float(m[i < j].sum()),
        ("btts", "Y", 0.0): float(m[(i > 0) & (j > 0)].sum()),
    }
    out[("btts", "N", 0.0)] = 1 - out[("btts", "Y", 0.0)]
    for pt in points:
        if float(pt) % 1 == 0:  # líneas enteras tienen devolución; no las modelamos
            continue
        over = float(m[total > pt].sum())
        out[("ou", "O", float(pt))] = over
        out[("ou", "U", float(pt))] = 1 - over
    return out


def devig(prices: list[float]) -> list[float]:
    """Quita el margen con el método 'power' (más fiel que normalizar cuando hay favoritos claros)."""
    inv = np.array([1 / p for p in prices])
    if inv.sum() <= 1:
        return list(inv / inv.sum())
    lo, hi = 1.0, 3.0
    for _ in range(60):
        k = (lo + hi) / 2
        if (inv ** k).sum() > 1:
            lo = k
        else:
            hi = k
    q = inv ** ((lo + hi) / 2)
    return list(q / q.sum())


SHARP = ("pinnacle", "betfair_ex", "betfair_ex_eu", "betfair_ex_uk", "matchbook")


def market_fair_probs(odds: pd.DataFrame) -> dict[tuple[str, str, float], tuple[float, int]]:
    """Probabilidad justa por selección para un partido: mediana entre casas con el mercado completo.
    Si hay casa 'sharp' (Pinnacle, exchanges) se le da el doble de peso. Devuelve {(m, sel, pt): (p, n_casas)}."""
    out = {}
    for (market, point), g in odds.groupby(["market", "point"]):
        sels = GROUPS.get(market)
        if not sels or (float(point) * 2) % 1:  # líneas asiáticas de cuarto (3.25, -0.75): se liquidan a medias
            continue
        per_book = []
        for book, gb in g.groupby("bookmaker"):
            pr = gb.set_index("selection")["price"]
            if all(s in pr for s in sels):
                fair = devig([float(pr[s]) for s in sels])
                reps = 2 if book in SHARP else 1
                per_book += [dict(zip(sels, fair))] * reps
        if not per_book:
            continue
        n_books = len(g["bookmaker"].unique())
        med = {s: float(np.median([b[s] for b in per_book])) for s in sels}
        tot = sum(med.values())
        for s in sels:
            out[(market, s, float(point))] = (med[s] / tot, n_books)
    return out


def _fmt(x: float) -> str:
    return f"{x:+g}".replace("+0", "0")


def label(market: str, sel: str, point: float, home: str, away: str, unit: str = "goles") -> str:
    if market == "1x2":
        return {"H": f"Gana {home}", "D": "Empate", "A": f"Gana {away}"}[sel]
    if market == "ml":
        return f"Gana {home}" if sel == "H" else f"Gana {away}"
    if market == "spread":
        return f"{home} {_fmt(point)}" if sel == "H" else f"{away} {_fmt(-point)}"
    if market == "ou":
        return f"{'Más' if sel == 'O' else 'Menos'} de {point:g} {unit}"
    if market == "btts":
        return "Ambos marcan: sí" if sel == "Y" else "Ambos marcan: no"
    return f"{market} {sel}"


def outcome(market: str, sel: str, point: float, hg: int, ag: int) -> int | None:
    """1 ganada, 0 perdida, None nula (empate en hándicap/total exacto)."""
    if market == "1x2":
        return int({"H": hg > ag, "D": hg == ag, "A": hg < ag}[sel])
    if market == "ml":
        if hg == ag:
            return None
        return int(hg > ag) if sel == "H" else int(ag > hg)
    if market == "spread":
        diff = hg + point - ag
        if diff == 0:
            return None
        return int(diff > 0) if sel == "H" else int(diff < 0)
    if market == "ou":
        if hg + ag == point:
            return None
        return int((hg + ag > point) if sel == "O" else (hg + ag < point))
    if market == "btts":
        both = hg > 0 and ag > 0
        return int(both if sel == "Y" else not both)
    raise ValueError(market)
