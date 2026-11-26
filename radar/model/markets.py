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
        if float(pt) % 1 == 0 or (float(pt) * 2) % 1:  # enteras (devolución) y de cuarto (a medias): no se modelan
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
    lo, hi = 1.0, 10.0
    for _ in range(80):
        k = (lo + hi) / 2
        if (inv ** k).sum() > 1:
            lo = k
        else:
            hi = k
    q = inv ** ((lo + hi) / 2)
    return list(q / q.sum())


EXCHANGES = ("betfair_ex", "betfair_ex_eu", "betfair_ex_uk", "betfair-ex", "matchbook")
SHARP = ("pinnacle",) + EXCHANGES
DEFAULT_EXCHANGE_BAND = (0.99, 1.04)


def overround(prices: list[float]) -> float:
    return float(sum(1 / p for p in prices))


def book_market_ok(book: str, prices: list[float], max_overround: float = 1.12,
                   exchange_band: tuple = DEFAULT_EXCHANGE_BAND) -> bool:
    """¿Es creíble el mercado completo de esta casa? Un margen fuera de rango = cuota rota, desfasada o mal
    asignada (p. ej. Betfair Exchange de football-data con 1,10 / 1,16 / 1,33 en un mismo partido)."""
    if not prices or any((not np.isfinite(p)) or p <= 1 for p in prices):
        return False
    o = overround(prices)
    lo, hi = exchange_band if book in EXCHANGES else (1.0, max_overround)
    return lo <= o <= hi


def odds_times(odds: pd.DataFrame) -> pd.Series | None:
    """Momento en que se tomó cada cuota (fetched_at; en bases antiguas, updated)."""
    for col in ("fetched_at", "updated"):
        if col in odds and odds[col].notna().any():
            return pd.to_datetime(odds[col], utc=True, format="ISO8601", errors="coerce")
    return None


def market_fair_probs(odds: pd.DataFrame, exclude: tuple | set = ("bet365",),
                      ref_times: dict | None = None, max_gap_h: float | None = None, max_overround: float = 1.12,
                      exchange_band: tuple = DEFAULT_EXCHANGE_BAND, valid_books: dict | None = None
                      ) -> dict[tuple[str, str, float], tuple[float, int, int]]:
    """Probabilidad justa por selección para un partido. Devuelve {(m, sel, pt): (p, n_casas_validas, n_sharp)}.

    Solo cuentan casas con el mercado completo y un margen creíble, que NO sean la casa donde se apuesta (exclude)
    y, si hay horas, tomadas como mucho `max_gap_h` antes o después de la referencia (la cuota de bet365 del mercado
    o, si no la hay, la cuota más reciente). Si hay casas sharp válidas (Pinnacle, exchanges) la referencia es SOLO
    la mediana de las sharp; si no, la mediana de las demás. valid_books (opcional) se rellena con
    {(mercado, línea): {casas válidas}}."""
    out = {}
    if odds.empty:
        return out
    times = odds_times(odds) if max_gap_h else None
    if times is not None:
        odds = odds.assign(_t=times)
    for (market, point), g in odds.groupby(["market", "point"]):
        sels = GROUPS.get(market)
        if not sels or (float(point) * 2) % 1:  # líneas asiáticas de cuarto (3.25, -0.75): se liquidan a medias
            continue
        ref = None
        if times is not None:
            ref = (ref_times or {}).get((market, float(point)))
            if ref is None or pd.isna(ref):
                ref = g["_t"].max()
        sharp, soft, names = [], [], set()
        for book, gb in g.groupby("bookmaker"):
            if book in exclude:
                continue
            pr = gb.drop_duplicates("selection", keep="last").set_index("selection")["price"]
            if not all(s in pr for s in sels):
                continue
            prices = [float(pr[s]) for s in sels]
            if not book_market_ok(book, prices, max_overround, exchange_band):
                continue
            if ref is not None and pd.notna(ref):
                t = gb["_t"].max()
                if pd.isna(t) or abs((t - ref).total_seconds()) > max_gap_h * 3600:
                    continue
            (sharp if book in SHARP else soft).append(dict(zip(sels, devig(prices))))
            names.add(book)
        if valid_books is not None:
            valid_books[(market, float(point))] = names
        base = sharp or soft
        if not base:
            continue
        med = {s: float(np.median([b[s] for b in base])) for s in sels}
        tot = sum(med.values())
        for s in sels:
            out[(market, s, float(point))] = (med[s] / tot, len(sharp) + len(soft), len(sharp))
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
    if market == "dc":
        return {"1X": f"{home} o empate", "X2": f"Empate o {away}", "12": f"{home} o {away} (sin empate)"}.get(sel, sel)
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
    if market == "dc":
        return int({"1X": hg >= ag, "X2": hg <= ag, "12": hg != ag}[sel])
    raise ValueError(market)
