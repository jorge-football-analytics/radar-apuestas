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
        for book in ("bet365", "pinnacle", "williamhill", "betvictor", generous_book):
            m = 1.02 if book == "pinnacle" else 1.06
            for (market, sel, pt), p in pr.items():
                if market == "btts" or (market == "ou" and pt != 2.5):
                    continue
                price = 1 / (p * m) * float(np.exp(rng.normal(0, 0.01)))
                if book == generous_book and sel == "H":
                    price *= boost
                rows.append(dict(league=league, kickoff=kickoff, home=h, away=a, source="football-data",
                                 bookmaker=book, market=market, selection=sel, point=pt, price=round(price, 2)))
    return pd.DataFrame(rows)
