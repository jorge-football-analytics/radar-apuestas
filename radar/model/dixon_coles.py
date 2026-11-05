"""Modelo Dixon-Coles (1997) con decaimiento temporal y regularización ridge.

goles_local ~ Poisson(λ),  λ = exp(home + ataque_local + defensa_visitante)
goles_visit ~ Poisson(μ),  μ = exp(ataque_visitante + defensa_local)
con la corrección τ(ρ) para marcadores bajos (0-0, 1-0, 0-1, 1-1).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

MAX_GOALS = 10


@dataclass
class DCModel:
    teams: list[str]
    attack: np.ndarray
    defence: np.ndarray
    home: float
    rho: float
    fitted_at: str = ""
    n_matches: int = 0
    last_seen: dict[str, str] = field(default_factory=dict)

    @property
    def index(self) -> dict[str, int]:
        return {t: i for i, t in enumerate(self.teams)}

    def rates(self, home: str, away: str) -> tuple[float, float] | None:
        idx = self.index
        if home not in idx or away not in idx:
            return None
        h, a = idx[home], idx[away]
        lam = np.exp(self.home + self.attack[h] + self.defence[a])
        mu = np.exp(self.attack[a] + self.defence[h])
        return float(lam), float(mu)

    def to_dict(self) -> dict:
        return {"teams": self.teams, "attack": self.attack.tolist(), "defence": self.defence.tolist(),
                "home": self.home, "rho": self.rho, "fitted_at": self.fitted_at, "n_matches": self.n_matches,
                "last_seen": self.last_seen}

    @classmethod
    def from_dict(cls, d: dict) -> "DCModel":
        return cls(d["teams"], np.array(d["attack"]), np.array(d["defence"]), d["home"], d["rho"],
                   d.get("fitted_at", ""), d.get("n_matches", 0), d.get("last_seen", {}))


def score_matrix(lam: float, mu: float, rho: float, max_goals: int = MAX_GOALS) -> np.ndarray:
    """Matriz P[i, j] = P(local marca i, visitante marca j)."""
    g = np.arange(max_goals + 1)
    m = np.outer(poisson.pmf(g, lam), poisson.pmf(g, mu))
    m[0, 0] *= 1 - lam * mu * rho
    m[0, 1] *= 1 + lam * rho
    m[1, 0] *= 1 + mu * rho
    m[1, 1] *= 1 - rho
    m = np.clip(m, 0, None)
    return m / m.sum()


def fit(matches: pd.DataFrame, ref_date: datetime | pd.Timestamp, xi: float = 0.0019,
        ridge: float = 1.0, min_matches_team: int = 5) -> DCModel | None:
    """Ajusta el modelo con los partidos anteriores a ref_date. Columnas: date, home, away, hg, ag."""
    df = matches[matches["date"] < pd.Timestamp(ref_date)].copy()
    if len(df) < 50:
        return None
    counts = pd.concat([df["home"], df["away"]]).value_counts()
    keep = set(counts[counts >= min_matches_team].index)
    df = df[df["home"].isin(keep) & df["away"].isin(keep)]
    teams = sorted(keep)
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    hi = df["home"].map(idx).to_numpy()
    ai = df["away"].map(idx).to_numpy()
    x = df["hg"].to_numpy(float)
    y = df["ag"].to_numpy(float)
    age = (pd.Timestamp(ref_date) - df["date"]).dt.days.to_numpy(float)
    w = np.exp(-xi * age)
    w = w / w.mean()  # escala estable para el ridge

    m00 = (x == 0) & (y == 0)
    m01 = (x == 0) & (y == 1)
    m10 = (x == 1) & (y == 0)
    m11 = (x == 1) & (y == 1)

    def unpack(p):
        return p[:n], p[n:2 * n], p[2 * n], p[2 * n + 1]

    def nll_grad(p):
        att, de, home, rho = unpack(p)
        ll_l = home + att[hi] + de[ai]
        ll_m = att[ai] + de[hi]
        lam, mu = np.exp(ll_l), np.exp(ll_m)
        tau = np.ones_like(lam)
        tau[m00] = 1 - lam[m00] * mu[m00] * rho
        tau[m01] = 1 + lam[m01] * rho
        tau[m10] = 1 + mu[m10] * rho
        tau[m11] = 1 - rho
        tau = np.clip(tau, 1e-10, None)
        ll = w * (np.log(tau) + x * ll_l - lam + y * ll_m - mu)
        # derivadas de log τ
        dl, dm, dr = np.zeros_like(lam), np.zeros_like(lam), np.zeros_like(lam)
        dl[m00] = -lam[m00] * mu[m00] * rho / tau[m00]
        dm[m00] = dl[m00]
        dr[m00] = -lam[m00] * mu[m00] / tau[m00]
        dl[m01] = lam[m01] * rho / tau[m01]
        dr[m01] = lam[m01] / tau[m01]
        dm[m10] = mu[m10] * rho / tau[m10]
        dr[m10] = mu[m10] / tau[m10]
        dr[m11] = -1 / tau[m11]
        gl = w * (x - lam + dl)
        gm = w * (y - mu + dm)
        g_att = np.bincount(hi, gl, n) + np.bincount(ai, gm, n)
        g_def = np.bincount(ai, gl, n) + np.bincount(hi, gm, n)
        g_home = gl.sum()
        g_rho = (w * dr).sum()
        nll = -ll.sum() + ridge * (att @ att + de @ de)
        grad = -np.concatenate([g_att, g_def, [g_home, g_rho]])
        grad[:n] += 2 * ridge * att
        grad[n:2 * n] += 2 * ridge * de
        return nll, grad

    p0 = np.concatenate([np.zeros(n), np.zeros(n), [0.25, -0.05]])
    bounds = [(None, None)] * (2 * n) + [(None, None), (-0.25, 0.25)]
    res = minimize(nll_grad, p0, jac=True, method="L-BFGS-B", bounds=bounds, options={"maxiter": 500})
    att, de, home, rho = unpack(res.x)
    # Centrar ataque (la media de goles queda en 'defensa' + 'home'; no cambia λ ni μ)
    c = att.mean()
    att, de = att - c, de + c
    last = pd.concat([df[["date", "home"]].rename(columns={"home": "t"}),
                      df[["date", "away"]].rename(columns={"away": "t"})]).groupby("t")["date"].max()
    return DCModel(teams, att, de, float(home), float(rho), pd.Timestamp(ref_date).isoformat(), len(df),
                   {t: d.isoformat() for t, d in last.items()})
