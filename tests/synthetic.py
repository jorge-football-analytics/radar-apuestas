"""Liga sintética con fuerzas conocidas para probar el sistema sin conexión."""
from __future__ import annotations

import numpy as np
import pandas as pd

from radar.model.dixon_coles import score_matrix
from radar.model.markets import market_probs

TEAMS = [f"Equipo {c}" for c in "ABCDEFGHIJKLMNOPQRST"]


def true_strengths(seed=0):
    rng = np.random.default_rng(seed)
    att = rng.normal(0, 0.3, len(TEAMS))
    de = rng.normal(0, 0.25, len(TEAMS))
    return att - att.mean(), de - de.mean()


def season_dates(start: str, n_rounds: int):
    return [pd.Timestamp(start) + pd.Timedelta(days=7 * r) for r in range(n_rounds)]


def make_history(league="laliga", seasons=("2223", "2324", "2425", "2526"), seed=0, home=0.25, base=0.1,
                 margin=1.05):
    rng = np.random.default_rng(seed + 1)
    att, de = true_strengths(seed)
    rows = []
    for si, code in enumerate(seasons):
        start = f"20{code[:2]}-08-15"
        n = len(TEAMS)
        pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
        rng.shuffle(pairs)
        per_round = n // 2
        dates = season_dates(start, len(pairs) // per_round + 1)
        for k, (i, j) in enumerate(pairs):
            lam = np.exp(base + home + att[i] + de[j])
            mu = np.exp(base + att[j] + de[i])
            hg, ag = rng.poisson(lam), rng.poisson(mu)
            pr = market_probs(score_matrix(lam, mu, -0.05), [2.5])
            noise = lambda: float(np.exp(rng.normal(0, 0.03)))
            odd = lambda p: round(1 / (p * margin) * noise(), 2)
            row = dict(league=league, season=code, date=dates[k // per_round], home=TEAMS[i], away=TEAMS[j],
                       hg=int(hg), ag=int(ag))
            for sel in "HDA":
                p = pr[("1x2", sel, 0.0)]
                row[f"open_{sel}"] = odd(p)
                row[f"close_{sel}"] = odd(p)
                row[f"b365_{sel}"] = odd(p)
                row[f"max_{sel}"] = row[f"b365_{sel}"] * 1.03
            for sel, key in (("O", "O25"), ("U", "U25")):
                p = pr[("ou", sel, 2.5)]
                row[f"open_{key}"] = odd(p)
                row[f"close_{key}"] = odd(p)
                row[f"b365_{key}"] = odd(p)
                row[f"max_{key}"] = row[f"b365_{key}"] * 1.03
            rows.append(row)
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True), (att, de, home, base)


def make_fixtures(pairs, kickoff, truth, league="laliga", generous_book="bwin", boost=1.15, seed=3):
    """Cuotas largas (formato de load_fixtures) para los próximos partidos. Una casa paga de más → valor."""
    att, de, home, base = truth
    rng = np.random.default_rng(seed)
    idx = {t: k for k, t in enumerate(TEAMS)}
    rows = []
    for h, a in pairs:
        i, j = idx[h], idx[a]
        lam = np.exp(base + home + att[i] + de[j])
        mu = np.exp(base + att[j] + de[i])
        pr = market_probs(score_matrix(lam, mu, -0.05), [2.5])
        for book in ("bet365", "pinnacle", "williamhill", "betvictor", "bwin"):
            m = 1.02 if book == "pinnacle" else 1.06
            for (market, sel, pt), p in pr.items():
                if market == "btts" or (market == "ou" and pt != 2.5):
                    continue
                price = 1 / (p * m) * float(np.exp(rng.normal(0, 0.01)))
                if book == generous_book and sel == "H":
                    price *= boost
                rows.append(dict(sport="football", league=league, kickoff=kickoff, home=h, away=a, source="football-data",
                                 bookmaker=book, market=market, selection=sel, point=pt, price=round(price, 2)))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ NBA y tenis sintéticos
NBA_TEAMS = ["Boston Celtics", "Los Angeles Lakers", "LA Clippers", "Denver Nuggets", "Miami Heat", "New York Knicks",
             "Golden State Warriors", "Phoenix Suns", "Dallas Mavericks", "Milwaukee Bucks"]


def make_nba(seed=0, n_days=500, start="2024-10-20"):
    rng = np.random.default_rng(seed)
    strength = {t: s for t, s in zip(NBA_TEAMS, np.linspace(-8, 8, len(NBA_TEAMS)))}
    rows = []
    for d in range(n_days):
        date = pd.Timestamp(start) + pd.Timedelta(days=d)
        if date.month in (7, 8, 9):
            continue
        teams = list(rng.permutation(NBA_TEAMS))
        for i in range(0, len(teams), 2):
            h, a = teams[i], teams[i + 1]
            margin = strength[h] - strength[a] + 3 + rng.normal(0, 12)
            total = 225 + rng.normal(0, 18)
            hg = int(round((total + margin) / 2))
            ag = int(round((total - margin) / 2))
            if hg == ag:
                hg += 1
            rows.append(dict(league="nba", date=date, home=h, away=a, hg=hg, ag=ag, neutral=False))
    return pd.DataFrame(rows), strength


PLAYERS = [f"Jugador{c} {c}." for c in "ABCDEFGHIJKLMNOP"]


def make_tennis(seed=0, n=4000):
    rng = np.random.default_rng(seed)
    skill = {p: s for p, s in zip(PLAYERS, np.linspace(1300, 1900, len(PLAYERS)))}
    rows = []
    for k in range(n):
        a, b = rng.choice(PLAYERS, 2, replace=False)
        pa = 1 / (1 + 10 ** ((skill[b] - skill[a]) / 400))
        w, l = (a, b) if rng.random() < pa else (b, a)
        pw = pa if w == a else 1 - pa
        rows.append(dict(tour="atp", date=pd.Timestamp("2023-01-01") + pd.Timedelta(hours=8 * k), tournament="X",
                         surface=["Hard", "Clay", "Grass"][k % 3], winner=w, loser=l, completed=True,
                         b365_w=round(1 / (pw * 1.06), 2), b365_l=round(1 / ((1 - pw) * 1.06), 2),
                         ps_w=round(1 / (pw * 1.02), 2), ps_l=round(1 / ((1 - pw) * 1.02), 2),
                         avg_w=np.nan, avg_l=np.nan, max_w=np.nan, max_l=np.nan))
    return pd.DataFrame(rows), skill
