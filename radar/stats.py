"""Estadísticas de equipos y jugadores para justificar cada selección: forma, racha, rendimiento como local o
visitante, goles o puntos a favor y en contra, y cara a cara. Trabaja sobre cualquier histórico con columnas
date, home, away, hg, ag (fútbol, NBA, NFL y tenis con ganador=home)."""
from __future__ import annotations

import math

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


def _season(g: pd.DataFrame, gap_days: int = 50) -> pd.DataFrame:
    """Partidos de la temporada en curso: desde el último parón de más de `gap_days` días (verano en fútbol,
    pretemporada en NBA/NFL). Antes se usaban 300 días fijos, que mezclaban la temporada anterior."""
    if g.empty:
        return g
    gaps = g["date"].diff().dt.days.fillna(0)
    starts = g.loc[gaps > gap_days, "date"]
    return g[g["date"] >= starts.max()] if not starts.empty else g


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
    season = _season(g)

    def record(df):
        rr = [_res(int(x), int(y), draws) for x, y in zip(df["gf"], df["ga"])]
        return dict(pj=len(rr), v=rr.count("V"), e=rr.count("E"), d=rr.count("D"),
                    gf=float(df["gf"].mean()) if len(df) else 0.0, ga=float(df["ga"].mean()) if len(df) else 0.0)

    tot = last10["gf"] + last10["ga"]
    return dict(forma="".join(res[-n:]), racha=f"{streak}{streak_kind}", gf10=float(last10["gf"].mean()),
                ga10=float(last10["ga"].mean()), n10=int(len(last10)), tot10=float(tot.mean()),
                over25_10=int((tot > 2.5).sum()), cs10=int((last10["ga"] == 0).sum()),
                scored10=int((last10["gf"] > 0).sum()), btts10=int(((last10["gf"] > 0) & (last10["ga"] > 0)).sum()),
                temporada=record(season), local=record(season[season["venue"] == "L"]),
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


def _rec(r: dict, draws: bool) -> str:
    return f"{r.get('v', 0)}-{r.get('e', 0)}-{r.get('d', 0)}" if draws else f"{r.get('v', 0)}-{r.get('d', 0)}"


def _streak(x: dict, draws: bool) -> str:
    r = x.get("racha", "")
    if not r:
        return ""
    n, k = int(r[:-1]), r[-1]
    if n == 1:
        return {"V": "ganó su último partido", "E": "empató el último", "D": "perdió el último"}[k]
    return {"V": f"lleva {n} victorias seguidas", "E": f"encadena {n} empates", "D": f"suma {n} derrotas seguidas"}[k]


def _num(x: float, d: int = 1) -> str:
    return f"{x:.{d}f}".replace(".", ",")


def scouting(st: dict, home: str, away: str, sport: str, market: str, selection: str, point: float = 0.0,
             table: dict | None = None, info: dict | None = None) -> list[str]:
    """Informe de scouting en lenguaje natural para una selección, solo con datos reales (sin inventar):
    tabla, forma y racha, rendimiento en casa/fuera, ataque/defensa y la tendencia que importa para el mercado."""
    draws = sport == "football"
    unit = {"football": "goles", "tennis": "juegos"}.get(sport, "puntos")
    loc, vis, h2h = (st or {}).get("local") or {}, (st or {}).get("visit") or {}, (st or {}).get("h2h") or {}
    table, info = table or {}, info or {}
    out: list[str] = []
    # tabla
    tl, tv = table.get("local") or {}, table.get("visit") or {}
    if tl.get("pos") and tv.get("pos"):
        if "pts" in tl:
            out.append(f"{home} es {tl['pos']}º con {tl['pts']} puntos en {tl['pj']} jornadas; "
                       f"{away}, {tv['pos']}º con {tv['pts']}.")
        else:
            out.append(f"{home} ({tl['g']}-{tl['p']}, {tl['pos']}º del ranking Elo) recibe a {away} "
                       f"({tv['g']}-{tv['p']}, {tv['pos']}º).")
    # forma y racha
    if loc.get("forma") and vis.get("forma"):
        out.append(f"{home} {_streak(loc, draws)} (últimos 5: {loc['forma']}); {away} {_streak(vis, draws)} "
                   f"({vis['forma']}).")
    # casa / fuera
    lr, vr = loc.get("local") or {}, vis.get("visitante") or {}
    if lr.get("pj") and vr.get("pj") and all(k in d for d in (lr, vr) for k in ("gf", "ga")):
        # estadísticas guardadas por la versión anterior (ventana fija de 300 días): no decir «esta temporada»
        when = "esta temporada" if "temporada" in loc else "en los últimos meses"
        out.append(f"En casa {home} va {_rec(lr, draws)} {when} ({_num(lr['gf'])} {unit} a favor y "
                   f"{_num(lr['ga'])} en contra por partido); fuera {away} va {_rec(vr, draws)} "
                   f"({_num(vr['gf'])} a favor, {_num(vr['ga'])} en contra).")
    # lo que importa según el mercado
    n_l, n_v = loc.get("n10") or 0, vis.get("n10") or 0
    if market == "ou" and n_l and n_v:
        if sport == "football":
            out.append(f"Tendencia de goles: más de 2,5 en {loc.get('over25_10', 0)} de los últimos {n_l} partidos de "
                       f"{home} y en {vis.get('over25_10', 0)} de {n_v} de {away}; promedian {_num(loc.get('tot10', 0))} y "
                       f"{_num(vis.get('tot10', 0))} goles totales por partido.")
        else:
            comb = (loc.get("gf10", 0) + vis.get("ga10", 0) + vis.get("gf10", 0) + loc.get("ga10", 0)) / 2
            out.append(f"Ritmo anotador: los dos equipos suman de media {_num(comb, 0)} puntos combinados en sus últimos "
                       f"partidos frente a una línea de {_num(point)}.")
    elif market == "btts" and n_l and n_v:
        out.append(f"{home} ha marcado en {loc.get('scored10', 0)} de {n_l} y dejado su portería a cero en "
                   f"{loc.get('cs10', 0)}; {away} ha marcado en {vis.get('scored10', 0)} de {n_v} y mantenido la "
                   f"portería a cero en {vis.get('cs10', 0)}.")
    elif market in ("1x2", "ml", "dc", "spread") and loc.get("gf10") is not None and vis.get("gf10") is not None:
        backs_home = selection in ("H", "1X") or (market == "dc" and selection == "12" and
                                                  loc.get("gf10", 0) >= vis.get("gf10", 0))
        a, b = (loc, vis) if backs_home else (vis, loc)
        na, nb = (home, away) if backs_home else (away, home)
        if sport == "football":
            out.append(f"Últimos 10: {na} marca {_num(a['gf10'])} y encaja {_num(a['ga10'])} por partido "
                       f"({a.get('cs10', 0)} porterías a cero); {nb}, {_num(b['gf10'])} y {_num(b['ga10'])}.")
        else:
            out.append(f"Últimos 10: {na} anota {_num(a['gf10'], 0)} y recibe {_num(a['ga10'], 0)} de media; "
                       f"{nb}, {_num(b['gf10'], 0)} y {_num(b['ga10'], 0)}.")
    # modelo propio (contexto, no decide)
    if info.get("tipo") == "puntos" and info.get("margen") is not None:
        fav = home if info["margen"] >= 0 else away
        out.append(f"El Elo da favorito a {fav} por unos {_num(abs(info['margen']), 1)} puntos (total esperado "
                   f"{_num(info.get('total', 0), 0)}).")
    elif info.get("tipo") == "goles" and info.get("xg_local") is not None:
        out.append(f"El modelo de goles espera {_num(info['xg_local'], 2)} – {_num(info['xg_visit'], 2)}.")
    # cara a cara
    if h2h.get("partidos"):
        last = h2h["partidos"][-1]
        n_h = len(h2h["partidos"])
        vl, em = h2h["v_local"], h2h.get("empates", 0)
        out.append(f"Cara a cara ({'último partido' if n_h == 1 else f'últimos {n_h}'}): {vl} "
                   f"{'victoria' if vl == 1 else 'victorias'} de {home}"
                   + (f", {em} {'empate' if em == 1 else 'empates'}" if draws else "") + f" y {h2h['v_visit']} de {away}; "
                   f"el último, {last['marcador']} ({last['fecha'][:7]}).")
    # bajas y lesiones (solo lo que devuelve la fuente; si no hay fuente, se dice)
    b = info.get("bajas")
    if isinstance(b, dict):
        def fmt(lst):
            if not lst:
                return "sin bajas confirmadas"
            items = [f"{a.get('jugador')} ({a.get('tipo') or 'baja'}{': ' + a['motivo'] if a.get('motivo') else ''})"
                     for a in lst[:4]]
            return ", ".join(items) + (f" y {len(lst) - 4} más" if len(lst) > 4 else "")
        if b.get("fuente") or b.get("local") or b.get("visit"):
            out.append(f"Bajas ({b.get('fuente') or 'apuntadas a mano'}): {home}: {fmt(b.get('local'))}; "
                       f"{away}: {fmt(b.get('visit'))}.")
        else:
            out.append(f"Bajas: sin datos para este partido: {b.get('nota') or 'sin fuente'}.")
    if not out or all(x.startswith("Bajas") for x in out):
        out.insert(0, "Sin histórico propio de estos equipos en nuestras fuentes (competición europea, jugador nuevo o "
                   "liga sin datos): la probabilidad sale solo del mercado sin margen.")
    return out


def price_line(p: float, price: float | None, priced: bool, n_books: int | None = None, n_sharp: int | None = None,
               derived: bool = False, ref_book: str | None = None, b365_est: float | None = None,
               ref_price: float | None = None, ref_age_h: float | None = None) -> str:
    """Por qué merece (o no) la pena la cuota, en una frase. Con ref_book, la cuota es la mejor real de otra casa."""
    from .config import book_name
    fair = 1 / p if p else None
    ref = ""
    if n_books:
        more = n_books - 1
        ref = (f" Precio justo calculado con Pinnacle"
               + (f" y {more} casa{'s' if more > 1 else ''} más" if more > 0 else "")
               if n_sharp else f" Precio justo calculado con {n_books} casa{'s' if n_books > 1 else ''} sin margen")
        ref += " (mercado derivado del 1X2/ganador)." if derived else "."
    if not priced or not price:
        return (f"Aún no hay cuota fiable de bet365: solo merece la pena si paga {_num(math.ceil(100 * fair - 1e-9) / 100, 2)} o más "
                f"(probabilidad real del {p:.0%}).{ref}")
    ev = p * price - 1
    if ref_book:
        need = math.ceil(100 * fair - 1e-9) / 100
        refp = f" (cuota ref. {_num(ref_price, 2)} en {book_name(ref_book)}, −5,5 %)" if ref_price else \
            f" (desde {book_name(ref_book)})"
        age = f" La cuota de referencia es de hace {_num(ref_age_h, 0)} h: compruébala." \
            if ref_age_h is not None and ref_age_h > 6 else ""
        txt = (f"bet365 estimada {_num(price, 2)}{refp}; lo justo sería {_num(fair, 2)} "
               f"({'+' if ev >= 0 else ''}{_num(ev * 100)} % de valor). En bet365, cógela solo a {_num(need, 2)} o más.{age}")
        return txt + ref
    if ev > 0.005:
        txt = f"bet365 paga {_num(price, 2)} y lo justo sería {_num(fair, 2)}: cobras un {_num(ev * 100)} % más de lo que vale."
    elif ev >= -0.015:
        txt = (f"bet365 paga {_num(price, 2)} frente a un justo de {_num(fair, 2)}: prácticamente sin margen "
               f"({_num(ev * 100)} %), por eso encaja en una combinada.")
    else:
        txt = (f"bet365 paga {_num(price, 2)} y lo justo sería {_num(fair, 2)}: la casa se queda un {_num(-ev * 100)} % "
               f"de margen; solo como pata de bajo riesgo.")
    return txt + ref


def why_lines(sel_label: str, market: str, selection: str, st: dict, unit: str = "goles") -> list[str]:
    """Compatibilidad: frases cortas de scouting (sin nombres de equipo)."""
    sport = "football" if unit == "goles" else "tennis" if unit == "juegos" else "basketball"
    return scouting(st, "El local", "el visitante", sport, market, selection)[:4] if st else []
