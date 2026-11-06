"""Clasificaciones, forma reciente y rankings Elo que se guardan para la web."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .model.elo import TeamElo, TennisElo


def _form(results: list[str]) -> str:
    return "".join(results[-5:])


def football_table(hist: pd.DataFrame, model=None) -> list[dict]:
    """Clasificación de la temporada actual (la más reciente del histórico) con forma y fuerza del modelo."""
    if hist.empty:
        return []
    last = hist["date"].max()
    season = hist.loc[hist["date"] == last, "season"].iloc[0] if "season" in hist else None
    cur = hist[hist["season"] == season] if season is not None else hist[hist["date"] > last - pd.Timedelta(days=300)]
    if len(cur) < 10:  # temporada recién empezada en ligas de año natural: usar últimos 300 días
        cur = hist[hist["date"] > last - pd.Timedelta(days=300)]
    rows: dict[str, dict] = {}
    for r in cur.sort_values("date").itertuples(index=False):
        for team, gf, ga in ((r.home, r.hg, r.ag), (r.away, r.ag, r.hg)):
            t = rows.setdefault(team, dict(equipo=team, pj=0, g=0, e=0, p=0, gf=0, gc=0, pts=0, forma=[]))
            t["pj"] += 1
            t["gf"] += gf
            t["gc"] += ga
            res = "V" if gf > ga else "E" if gf == ga else "D"
            t[{"V": "g", "E": "e", "D": "p"}[res]] += 1
            t["pts"] += {"V": 3, "E": 1, "D": 0}[res]
            t["forma"].append(res)
    out = []
    for t in rows.values():
        t["dg"] = t["gf"] - t["gc"]
        t["forma"] = _form(t["forma"])
        if model is not None and t["equipo"] in model.index:
            i = model.index[t["equipo"]]
            t["ataque"] = float(np.exp(model.attack[i]))
            t["defensa"] = float(np.exp(-model.defence[i]))
        out.append(t)
    out.sort(key=lambda t: (t["pts"], t["dg"], t["gf"]), reverse=True)
    for k, t in enumerate(out, 1):
        t["pos"] = k
    return out


def team_elo_table(hist: pd.DataFrame, elo: TeamElo) -> list[dict]:
    if hist.empty:
        return []
    last = hist["date"].max()
    gaps = hist["date"].diff().dt.days.fillna(0)
    start = hist.loc[gaps > 60, "date"].max() if (gaps > 60).any() else hist["date"].min()
    cur = hist[hist["date"] >= start]
    rows: dict[str, dict] = {}
    for r in cur.itertuples(index=False):
        for team, pf, pa in ((r.home, r.hg, r.ag), (r.away, r.ag, r.hg)):
            t = rows.setdefault(team, dict(equipo=team, g=0, p=0, pf=0, pa=0, forma=[]))
            t["g" if pf > pa else "p"] += 1
            t["pf"] += pf
            t["pa"] += pa
            t["forma"].append("V" if pf > pa else "D")
    out = []
    for team, t in rows.items():
        n = t["g"] + t["p"]
        out.append(dict(equipo=team, elo=round(elo.rating[team]), g=t["g"], p=t["p"],
                        pct=t["g"] / n if n else 0, pf=t["pf"] / n if n else 0, pa=t["pa"] / n if n else 0,
                        forma=_form(t["forma"]), ultimo=str(last.date())))
    out.sort(key=lambda t: t["elo"], reverse=True)
    for k, t in enumerate(out, 1):
        t["pos"] = k
    return out


def tennis_ranking(hist: pd.DataFrame, elo: TennisElo, tour: str, top: int = 50) -> list[dict]:
    if hist.empty:
        return []
    h = hist[hist["tour"] == tour]
    if h.empty:
        return []
    recent = h["date"].max() - pd.Timedelta(days=150)
    active = set(h.loc[h["date"] >= recent, "winner"]) | set(h.loc[h["date"] >= recent, "loser"])
    out = []
    for p in active:
        if elo.n.get(p, 0) < 15:
            continue
        w = h[(h["date"] >= h["date"].max() - pd.Timedelta(days=365))]
        wins = int((w["winner"] == p).sum())
        losses = int((w["loser"] == p).sum())
        last = pd.concat([h.loc[h["winner"] == p, ["date"]].assign(r="V"), h.loc[h["loser"] == p, ["date"]].assign(r="D")]
                         ).sort_values("date")["r"].tolist()
        out.append(dict(jugador=p, elo=round(elo.overall[p]), dura=round(elo.surf[(p, "Hard")]),
                        tierra=round(elo.surf[(p, "Clay")]), hierba=round(elo.surf[(p, "Grass")]),
                        g=wins, p=losses, forma=_form(last)))
    out.sort(key=lambda t: t["elo"], reverse=True)
    out = out[:top]
    for k, t in enumerate(out, 1):
        t["pos"] = k
    return out
