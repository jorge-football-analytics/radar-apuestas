"""Mercados derivados del precio justo del mercado (no del modelo propio).

Fútbol: se buscan las tasas de gol λ (local) y μ (visitante) cuya matriz de marcadores Dixon-Coles reproduce el 1X2
sin margen y, si existe, el más/menos 2,5 sin margen. De esa matriz salen doble oportunidad, más/menos de cualquier
línea, ambos marcan y hándicaps. Es la forma estándar de extender un mercado líquido (1X2 de Pinnacle) a mercados
menos líquidos sin meter el sesgo del modelo propio (que infraestima los goles).

NBA y NFL: margen ~ Normal(m, σ) con m tal que P(margen > 0) = probabilidad justa del moneyline, y total ~
Normal(t, σt) con t sacado de la línea de totales del mercado (o del modelo Elo si no hay línea)."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

from .dixon_coles import score_matrix
from .markets import market_probs

RHO = -0.05
OU_LINES = (0.5, 1.5, 2.5, 3.5, 4.5)
SPREAD_LINES = (-2.5, -1.5, -0.5, 0.5, 1.5, 2.5)
US_SD = {"nba": (12.5, 18.5), "euroleague": (11.5, 16.0), "nfl": (13.5, 13.5)}


def fit_goal_rates(p_home: float, p_draw: float, p_away: float, p_over25: float | None = None,
                   total_prior: float = 2.65, rho: float = RHO) -> tuple[float, float]:
    """λ, μ que mejor reproducen el 1X2 (y el más/menos 2,5 si se da)."""
    def loss(x):
        lam, mu = np.exp(x)
        p = market_probs(score_matrix(lam, mu, rho), [2.5])
        err = (p[("1x2", "H", 0.0)] - p_home) ** 2 + (p[("1x2", "A", 0.0)] - p_away) ** 2 + \
            (p[("1x2", "D", 0.0)] - p_draw) ** 2
        if p_over25 is not None:
            err += 2 * (p[("ou", "O", 2.5)] - p_over25) ** 2
        else:  # sin línea de goles: un empujón suave hacia la media de goles de la liga
            err += 0.0004 * (lam + mu - total_prior) ** 2
        return err

    x0 = np.log([max(total_prior * 0.55, 0.2), max(total_prior * 0.45, 0.2)])
    res = minimize(loss, x0, method="Nelder-Mead", options={"xatol": 1e-5, "fatol": 1e-10, "maxiter": 400})
    lam, mu = np.exp(res.x)
    return float(lam), float(mu)


def football_markets(lam: float, mu: float, rho: float = RHO, ou_lines=OU_LINES, spread_lines=SPREAD_LINES
                     ) -> dict[tuple[str, str, float], float]:
    m = score_matrix(lam, mu, rho)
    out = market_probs(m, list(ou_lines))
    h, d, a = out[("1x2", "H", 0.0)], out[("1x2", "D", 0.0)], out[("1x2", "A", 0.0)]
    out[("dc", "1X", 0.0)] = h + d
    out[("dc", "12", 0.0)] = h + a
    out[("dc", "X2", 0.0)] = d + a
    i, j = np.indices(m.shape)
    for pt in spread_lines:
        if float(pt) % 1 == 0 or (float(pt) * 2) % 1:  # solo líneas de medio gol (sin devolución)
            continue
        cover = float(m[i + pt - j > 0].sum())
        out[("spread", "H", float(pt))] = cover
        out[("spread", "A", float(pt))] = 1 - cover
    return out


def us_markets(p_home: float, league: str, spread_lines=(), total_lines=(), total_mean: float | None = None,
               ) -> dict[tuple[str, str, float], float]:
    sd_m, sd_t = US_SD.get(league, US_SD["nba"])
    p = min(max(p_home, 0.01), 0.99)
    mean_margin = sd_m * norm.ppf(p)
    out = {("ml", "H", 0.0): p, ("ml", "A", 0.0): 1 - p}
    for pt in spread_lines:
        if float(pt) % 1 == 0:
            continue
        cover = float(1 - norm.cdf(-pt, loc=mean_margin, scale=sd_m))
        out[("spread", "H", float(pt))] = cover
        out[("spread", "A", float(pt))] = 1 - cover
    if total_mean is not None:
        for pt in total_lines:
            if float(pt) % 1 == 0:
                continue
            over = float(1 - norm.cdf(pt, loc=total_mean, scale=sd_t))
            out[("ou", "O", float(pt))] = over
            out[("ou", "U", float(pt))] = 1 - over
    return out


def total_from_line(line: float, p_over: float, league: str) -> float:
    """Total esperado que hace que P(total > línea) = p_over."""
    _, sd_t = US_SD.get(league, US_SD["nba"])
    return float(line + sd_t * norm.ppf(min(max(p_over, 0.01), 0.99)))


def derived_probs(sport: str, league: str, fair: dict, odds_lines: dict, total_prior: float | None = None,
                  model_total: float | None = None) -> dict[tuple[str, str, float], float]:
    """Probabilidades de todos los mercados derivables a partir del precio justo (fair: {(m, sel, pt): (p, n, ns)}).
    odds_lines: {'ou': [líneas con cuota], 'spread': [...]} para calcular también esas líneas exactas."""
    g = lambda k: fair.get(k, (None,))[0]
    if sport == "football":
        h, d, a = g(("1x2", "H", 0.0)), g(("1x2", "D", 0.0)), g(("1x2", "A", 0.0))
        if None in (h, d, a):
            return {}
        lam, mu = fit_goal_rates(h, d, a, g(("ou", "O", 2.5)), total_prior or 2.65)
        ou = sorted(set(OU_LINES) | {x for x in odds_lines.get("ou", []) if (x * 2) % 1 == 0 and x % 1})
        sp = sorted(set(SPREAD_LINES) | {x for x in odds_lines.get("spread", []) if x % 1 and (x * 2) % 1 == 0})
        return football_markets(lam, mu, ou_lines=ou, spread_lines=sp)
    if sport in ("basketball", "nfl"):
        h = g(("ml", "H", 0.0))
        if h is None:
            return {}
        total_mean = model_total
        ou_lines = sorted(set(odds_lines.get("ou", [])))
        direct = [(pt, g(("ou", "O", pt))) for pt in ou_lines if g(("ou", "O", pt)) is not None]
        if direct:  # la línea de totales con más información del mercado
            pt, po = min(direct, key=lambda x: abs(x[1] - 0.5))
            total_mean = total_from_line(pt, po, league)
        return us_markets(h, league, odds_lines.get("spread", []), ou_lines, total_mean)
    return {}
