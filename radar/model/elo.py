"""Modelos Elo.

TeamElo (NBA, NFL): Elo con ventaja de campo y multiplicador por diferencia de puntos (estilo FiveThirtyEight),
más medias móviles de puntos a favor/en contra para los mercados de totales y hándicap.
TennisElo: Elo general + Elo por superficie con K decreciente según partidos jugados (estilo FiveThirtyEight).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from math import log10

import numpy as np
import pandas as pd
from scipy.stats import norm

from ..teams import player_key

TEAM_PARAMS = {
    #       K, ventaja local (Elo), Elo por punto, desv. margen, desv. total, regresión entre temporadas, alfa EWMA
    "nba": dict(k=20, hfa=70, per_pt=28, sd_margin=12.5, sd_total=18.5, revert=0.25, alpha=0.06),
    "nfl": dict(k=20, hfa=48, per_pt=25, sd_margin=13.5, sd_total=13.5, revert=0.33, alpha=0.15),
}


def _p(dr: float) -> float:
    return 1 / (1 + 10 ** (-dr / 400))


@dataclass
class TeamElo:
    league: str
    rating: dict = field(default_factory=lambda: defaultdict(lambda: 1500.0))
    off: dict = field(default_factory=dict)
    deff: dict = field(default_factory=dict)
    games: dict = field(default_factory=lambda: defaultdict(int))
    avg_pts: float = 100.0
    last_date: pd.Timestamp | None = None

    @property
    def prm(self):
        return TEAM_PARAMS[self.league]

    def fit(self, hist: pd.DataFrame) -> "TeamElo":
        p = self.prm
        pts = pd.concat([hist["hg"], hist["ag"]])
        self.avg_pts = float(pts.mean()) if len(pts) else 100.0
        for r in hist.itertuples(index=False):
            if self.last_date is not None and (r.date - self.last_date).days > 60:  # nueva temporada
                for t in list(self.rating):
                    self.rating[t] = 1505 + (self.rating[t] - 1505) * (1 - p["revert"])
            self.last_date = r.date
            self.update(r.home, r.away, r.hg, r.ag, bool(getattr(r, "neutral", False)))
        return self

    def update(self, h, a, hg, ag, neutral=False):
        p = self.prm
        rh, ra = self.rating[h], self.rating[a]
        dr = rh - ra + (0 if neutral else p["hfa"])
        exp = _p(dr)
        res = 1.0 if hg > ag else 0.0 if hg < ag else 0.5
        mov = abs(hg - ag)
        mult = ((mov + 3) ** 0.8) / (7.5 + 0.006 * (dr if res == 1 else -dr)) if mov else 1.0
        delta = p["k"] * mult * (res - exp)
        self.rating[h], self.rating[a] = rh + delta, ra - delta
        al = p["alpha"]
        for t, pf, pa in ((h, hg, ag), (a, ag, hg)):
            self.off[t] = pf if t not in self.off else (1 - al) * self.off[t] + al * pf
            self.deff[t] = pa if t not in self.deff else (1 - al) * self.deff[t] + al * pa
            self.games[t] += 1

    def known(self, t: str) -> bool:
        return self.games.get(t, 0) >= 8

    def predict(self, home: str, away: str, lines: dict[str, list[float]] | None = None, neutral=False
                ) -> tuple[dict, dict] | None:
        """Devuelve ({(mercado, sel, línea): prob}, info) o None si no hay datos suficientes."""
        if not (self.known(home) and self.known(away)):
            return None
        p = self.prm
        dr = self.rating[home] - self.rating[away] + (0 if neutral else p["hfa"])
        ph = _p(dr)
        mu_m = dr / p["per_pt"]
        hp = (self.off[home] + self.deff[away]) / 2 + mu_m / 2
        ap = (self.off[away] + self.deff[home]) / 2 - mu_m / 2
        mu_t = hp + ap
        out = {("ml", "H", 0.0): ph, ("ml", "A", 0.0): 1 - ph}
        lines = lines or {}
        for pt in lines.get("spread", []):  # hándicap del local
            cover = 1 - norm.cdf(-pt, loc=mu_m, scale=p["sd_margin"])
            out[("spread", "H", float(pt))] = float(cover)
            out[("spread", "A", float(pt))] = float(1 - cover)
        for pt in lines.get("ou", []):
            over = 1 - norm.cdf(pt, loc=mu_t, scale=p["sd_total"])
            out[("ou", "O", float(pt))] = float(over)
            out[("ou", "U", float(pt))] = float(1 - over)
        info = dict(elo_home=self.rating[home], elo_away=self.rating[away], margen=mu_m, total=mu_t)
        return out, info


SURFACES = ("Hard", "Clay", "Grass")
CLAY = ("french", "roland", "madrid", "rome", "italian", "monte", "barcelona", "hamburg", "rio", "buenos",
        "estoril", "geneva", "lyon", "munich", "bastad", "gstaad", "kitzbuhel", "umag", "palermo", "parma",
        "bogota", "rabat", "strasbourg", "charleston", "stuttgart_wta", "prague", "iasi", "bucharest")
GRASS = ("wimbledon", "halle", "queens", "s_hertogenbosch", "hertogenbosch", "eastbourne", "mallorca",
         "newport", "berlin", "bad_homburg", "nottingham", "birmingham")


def surface_of(league_key: str) -> str:
    k = league_key.lower()
    if any(w in k for w in GRASS):
        return "Grass"
    if any(w in k for w in CLAY):
        return "Clay"
    return "Hard"


@dataclass
class TennisElo:
    overall: dict = field(default_factory=lambda: defaultdict(lambda: 1500.0))
    surf: dict = field(default_factory=lambda: defaultdict(lambda: 1500.0))  # clave (jugador, superficie)
    n: dict = field(default_factory=lambda: defaultdict(int))
    n_surf: dict = field(default_factory=lambda: defaultdict(int))
    names: dict = field(default_factory=dict)  # player_key → nombre en tennis-data
    last_seen: dict = field(default_factory=dict)

    @staticmethod
    def _k(n: int) -> float:
        return 250 / (n + 5) ** 0.4

    def prob(self, a: str, b: str, surface: str) -> float:
        po = _p(self.overall[a] - self.overall[b])
        if self.n_surf[(a, surface)] >= 8 and self.n_surf[(b, surface)] >= 8:
            ps = _p(self.surf[(a, surface)] - self.surf[(b, surface)])
            return 0.5 * po + 0.5 * ps
        return po

    def update(self, w: str, l: str, surface: str) -> None:
        pw = _p(self.overall[w] - self.overall[l])
        kw, kl = self._k(self.n[w]), self._k(self.n[l])
        self.overall[w] += kw * (1 - pw)
        self.overall[l] -= kl * (1 - pw)
        ps = _p(self.surf[(w, surface)] - self.surf[(l, surface)])
        kws, kls = self._k(self.n_surf[(w, surface)]), self._k(self.n_surf[(l, surface)])
        self.surf[(w, surface)] += kws * (1 - ps)
        self.surf[(l, surface)] -= kls * (1 - ps)
        self.n[w] += 1
        self.n[l] += 1
        self.n_surf[(w, surface)] += 1
        self.n_surf[(l, surface)] += 1

    def fit(self, hist: pd.DataFrame, record: bool = False) -> pd.DataFrame | None:
        """Recorre los partidos en orden. Con record=True devuelve la predicción previa a cada partido (backtest)."""
        rows = []
        for r in hist.itertuples(index=False):
            w, l = r.winner, r.loser
            self.names[player_key(w)] = w
            self.names[player_key(l)] = l
            surface = r.surface if r.surface in SURFACES else "Hard"
            if record and self.n[w] >= 10 and self.n[l] >= 10:
                rows.append(dict(date=r.date, tour=r.tour, p_w=self.prob(w, l, surface), b365_w=r.b365_w,
                                 b365_l=r.b365_l, ps_w=r.ps_w, ps_l=r.ps_l, avg_w=r.avg_w, avg_l=r.avg_l,
                                 completed=r.completed))
            self.update(w, l, surface)  # las retiradas también informan del nivel
            self.last_seen[w] = self.last_seen[l] = r.date
        return pd.DataFrame(rows) if record else None

    def predict(self, a: str, b: str, surface: str) -> tuple[dict, dict] | None:
        if self.n.get(a, 0) < 10 or self.n.get(b, 0) < 10:
            return None
        p = self.prob(a, b, surface)
        return ({("ml", "H", 0.0): p, ("ml", "A", 0.0): 1 - p},
                dict(elo_home=self.overall[a], elo_away=self.overall[b], superficie=surface,
                     partidos=(self.n[a], self.n[b])))


def logloss(p, y) -> float:
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, float)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


__all__ = ["TeamElo", "TennisElo", "surface_of", "logloss", "log10"]
