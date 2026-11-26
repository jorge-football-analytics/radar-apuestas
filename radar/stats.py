"""Estadísticas de equipos y jugadores para justificar cada selección: forma, racha, rendimiento como local o
visitante, goles o puntos a favor y en contra, y cara a cara. Trabaja sobre cualquier histórico con columnas
date, home, away, hg, ag (fútbol, NBA, NFL y tenis con ganador=home)."""
from __future__ import annotations

import pandas as pd


def _naive(t) -> pd.Timestamp:
    t = pd.Timestamp(t)
    return t.tz_convert(None) if t.tzinfo else t


def _res(gf: int, ga: int, draws: bool) -> str:
    return "V" if gf > ga else ("E" if gf == ga and draws else "D")


def team_games(hist: pd.DataFrame, team: str, before=None) -> pd.DataFrame:
    """Partidos de un equipo, vistos desde él: date, venue (L/V), rival, gf, ga."""
    if hist is None or hist.empty:
        return pd.DataFrame(columns=["date", "venue", "rival", "gf", "ga"])
    h = hist if before is None else hist[hist["date"] < _naive(before)]
    a = h.loc[h["home"] == team, ["date", "away", "hg", "ag"]].rename(columns={"away": "rival", "hg": "gf", "ag": "ga"})
    b = h.loc[h["away"] == team, ["date", "home", "ag", "hg"]].rename(columns={"home": "rival", "ag": "gf", "hg": "ga"})
    return pd.concat([a.assign(venue="L"), b.assign(venue="V")]).sort_values("date")


def team_form(hist: pd.DataFrame, team: str, before=None, draws: bool = True, n: int = 5) -> dict:
    g = team_games(hist, team, before)
    if g.empty:
        return {}
    res = [_res(int(x), int(y), draws) for x, y in zip(g["gf"], g["ga"])]
    streak_kind, streak = res[-1], 0
    for r in reversed(res):
        if r != streak_kind:
            break
        streak += 1
    last10 = g.tail(10)
    season = g[g["date"] > g["date"].max() - pd.Timedelta(days=300)]

    def record(df):
        rr = [_res(int(x), int(y), draws) for x, y in zip(df["gf"], df["ga"])]
        return dict(pj=len(rr), v=rr.count("V"), e=rr.count("E"), d=rr.count("D"),
                    gf=float(df["gf"].mean()) if len(df) else 0.0, ga=float(df["ga"].mean()) if len(df) else 0.0)

    return dict(forma="".join(res[-n:]), racha=f"{streak}{streak_kind}", gf10=float(last10["gf"].mean()),
                ga10=float(last10["ga"].mean()), local=record(season[season["venue"] == "L"]),
                visitante=record(season[season["venue"] == "V"]), ultimo=str(g["date"].max().date()))


def head_to_head(hist: pd.DataFrame, home: str, away: str, before=None, n: int = 5) -> dict:
    g = team_games(hist, home, before)
    g = g[g["rival"] == away].tail(n)
    if g.empty:
        return {}
    games = [dict(fecha=str(d.date()), sede=v, marcador=f"{int(a)}-{int(b)}" if v == "L" else f"{int(b)}-{int(a)}",
                  resultado=_res(int(a), int(b), True)) for d, v, a, b in zip(g["date"], g["venue"], g["gf"], g["ga"])]
    rr = [x["resultado"] for x in games]
    return dict(partidos=games, v_local=rr.count("V"), empates=rr.count("E"), v_visit=rr.count("D"))


def match_stats(hist: pd.DataFrame, home: str, away: str, before=None, draws: bool = True) -> dict:
    out = {"local": team_form(hist, home, before, draws), "visit": team_form(hist, away, before, draws),
           "h2h": head_to_head(hist, home, away, before)}
    return {k: v for k, v in out.items() if v}


def why_lines(sel_label: str, market: str, selection: str, st: dict, unit: str = "goles") -> list[str]:
    """Frases cortas con los datos que apoyan (o no) una selección. Solo hechos, sin adornos."""
    out = []
    loc, vis, h2h = st.get("local") or {}, st.get("visit") or {}, st.get("h2h") or {}
    if loc.get("forma") and vis.get("forma"):
        out.append(f"Forma (últ. 5): local {loc['forma']} · visitante {vis['forma']}")
    if loc.get("racha", "").endswith("V") and int(loc["racha"][:-1]) >= 3:
        out.append(f"El local lleva {loc['racha'][:-1]} victorias seguidas")
    if vis.get("racha", "").endswith("D") and int(vis["racha"][:-1]) >= 3:
        out.append(f"El visitante encadena {vis['racha'][:-1]} derrotas")
    if loc.get("local", {}).get("pj") and vis.get("visitante", {}).get("pj"):
        l, v = loc["local"], vis["visitante"]
        out.append(f"En casa {l['v']}-{l['e']}-{l['d']} · fuera {v['v']}-{v['e']}-{v['d']} (esta temporada)")
    if market in ("ou", "btts") and loc.get("gf10") is not None and vis.get("gf10") is not None:
        out.append(f"{unit.capitalize()} por partido (últ. 10): local {loc['gf10']:.1f} a favor / {loc['ga10']:.1f} en "
                   f"contra · visitante {vis['gf10']:.1f} / {vis['ga10']:.1f}")
    if h2h.get("partidos"):
        out.append(f"Cara a cara (últ. {len(h2h['partidos'])}): {h2h['v_local']} victorias local, {h2h['empates']} "
                   f"empates, {h2h['v_visit']} visitante")
    return out
