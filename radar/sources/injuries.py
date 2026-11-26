"""Bajas (lesionados y sancionados).

- API-Football (/injuries?date=AAAA-MM-DD): todas las ligas de un día en UNA consulta (plan gratis: 100/día), con caché
  de 6 h. Si la API responde que la temporada no está en tu plan, se desactiva esa temporada sin romper nada.
- ESPN (/injuries de NBA y NFL): gratis, una consulta por liga.
- data/bajas_manual.csv: bajas clave que apuntes tú (columnas: liga,equipo,jugador,importancia,hasta).
  importancia 1 = rotación, 2 = titular, 3 = figura. 'hasta' = fecha AAAA-MM-DD en que deja de contar.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from datetime import date

import pandas as pd
import requests

from ..config import DATA_DIR, LEAGUES, Settings, season_start_year
from ..db import DB
from ..teams import match_team

log = logging.getLogger(__name__)
BASE = "https://v3.football.api-sports.io"


def fetch_api_football(s: Settings, db: DB, league_key: str, dates: set[str], known: list[str], today: date
                       ) -> dict[str, list[dict]]:
    """{equipo_canónico: [{jugador, tipo, motivo, importancia}]}"""
    out: dict[str, list[dict]] = defaultdict(list)
    lg = LEAGUES[league_key]
    if not s.apifootball_key or db.get("apif_disabled_season") == season_start_year(today):
        return out
    for d in sorted(dates):
        try:
            r = requests.get(f"{BASE}/injuries", headers={"x-apisports-key": s.apifootball_key},
                             params={"league": lg["apif"], "season": season_start_year(today), "date": d}, timeout=20)
            r.raise_for_status()
            js = r.json()
        except requests.RequestException as e:
            log.warning("API-Football %s %s: %s", league_key, d, e)
            continue
        errors = js.get("errors")
        if errors:
            msg = str(errors)
            log.warning("API-Football: %s", msg)
            if "plan" in msg.lower() or "season" in msg.lower():
                # plan gratis sin temporada actual → no volver a intentarlo esta temporada
                db.set("apif_disabled_season", season_start_year(today))
                return out
            continue
        for it in js.get("response", []):
            team = match_team(it["team"]["name"], known) if known else it["team"]["name"]
            if not team:
                continue
            kind = it["player"].get("type", "")
            out[team].append({"jugador": it["player"]["name"], "tipo": kind,
                              "motivo": it["player"].get("reason", ""),
                              "importancia": 1.0 if kind == "Missing Fixture" else 0.5})
    return out


def load_manual(today: date) -> dict[tuple[str, str], list[dict]]:
    path = DATA_DIR / "bajas_manual.csv"
    out: dict[tuple[str, str], list[dict]] = defaultdict(list)
    if not path.exists():
        return out
    df = pd.read_csv(path)
    for _, r in df.iterrows():
        hasta = str(r.get("hasta", "") or "")
        if hasta and hasta != "nan" and hasta < today.isoformat():
            continue
        out[(str(r["liga"]).strip(), str(r["equipo"]).strip())].append(
            {"jugador": r["jugador"], "tipo": "manual", "motivo": "apuntada a mano",
             "importancia": float(r.get("importancia", 2) or 2)})
    return out


def absence_factor(absences: list[dict], impact: float, cap: float) -> float:
    """Multiplicador de los goles esperados del equipo (1.0 = sin efecto). Estimación deliberadamente prudente:
    el mercado ya descuenta muchas bajas, y el aprendizaje medirá si este ajuste ayuda."""
    total = sum(a.get("importancia", 1.0) for a in absences) * impact
    return 1.0 - min(total, cap)


# ======================================================================= v5: bajas por fecha y ESPN
ESPN_INJ = {"nba": "basketball/nba", "nfl": "football/nfl"}
STATUS_ES = {"out": "baja", "questionable": "duda", "doubtful": "duda seria", "day-to-day": "día a día",
             "injured reserve": "lesionado (reserva)", "suspension": "sancionado", "probable": "probable"}


def apifootball_by_date(s: Settings, db: DB, dates: set[str], today: date) -> tuple[list[dict], str]:
    """Bajas de todas las ligas para cada fecha (1 consulta por fecha, caché 6 h). Devuelve (bajas, estado)."""
    if not s.apifootball_key:
        return [], "API-Football no configurada (falta el secreto APIFOOTBALL_KEY)"
    season = season_start_year(today)
    if db.get("apif_disabled_season") == season:
        return [], "el plan de API-Football no cubre la temporada actual"
    out: list[dict] = []
    for d in sorted(dates):
        key = f"apif_inj_{d}"
        cached = db.get(key)
        if cached and (pd.Timestamp.now(tz="UTC") - pd.Timestamp(cached["ts"])).total_seconds() < 6 * 3600:
            out += cached["data"]
            continue
        try:
            r = requests.get(f"{BASE}/injuries", headers={"x-apisports-key": s.apifootball_key},
                             params={"date": d}, timeout=20)
            r.raise_for_status()
            js = r.json()
        except requests.RequestException as e:
            log.warning("API-Football %s: %s", d, e)
            continue
        errors = js.get("errors")
        if errors:
            msg = str(errors)
            log.warning("API-Football: %s", msg)
            if "plan" in msg.lower() or "season" in msg.lower():
                db.set("apif_disabled_season", season)
                return out, "el plan de API-Football no cubre la temporada actual"
            continue
        rows = [dict(league_id=(it.get("league") or {}).get("id"), team=(it.get("team") or {}).get("name", ""),
                     jugador=(it.get("player") or {}).get("name", ""), tipo=(it.get("player") or {}).get("type", ""),
                     motivo=(it.get("player") or {}).get("reason", ""), fixture=(it.get("fixture") or {}).get("id"))
                for it in js.get("response", [])]
        db.set(key, {"ts": pd.Timestamp.now(tz="UTC").isoformat(), "data": rows})
        out += rows
    return out, "ok"


def espn_injuries(path: str) -> dict[str, list[dict]]:
    """{equipo (nombre ESPN): [{jugador, estado, detalle, vuelta, comentario}]} de NBA o NFL."""
    try:
        r = requests.get(f"https://site.api.espn.com/apis/site/v2/sports/{path}/injuries", timeout=30,
                         headers={"User-Agent": "radar-apuestas/3.0 (uso personal)"})
        r.raise_for_status()
        js = r.json()
    except (requests.RequestException, ValueError) as e:
        log.warning("ESPN lesiones %s: %s", path, e)
        return {}
    out: dict[str, list[dict]] = {}
    for team in js.get("injuries", []):
        lst = []
        for it in team.get("injuries", []):
            st = str(it.get("status", "")).strip()
            det = it.get("details") or {}
            detail = " ".join(x for x in (det.get("type"), det.get("detail")) if x)
            lst.append(dict(jugador=(it.get("athlete") or {}).get("displayName", ""),
                            tipo=STATUS_ES.get(st.lower(), st.lower() or "baja"), motivo=detail,
                            vuelta=det.get("returnDate"), comentario=it.get("shortComment", ""),
                            importancia=1.0 if st.lower() == "out" else 0.4))
        if team.get("displayName"):
            out[team["displayName"]] = lst
    return out


def collect(s: Settings, db: DB, matches: list[dict], known: dict[str, list[str]], today: date
            ) -> tuple[dict[tuple[str, str], list[dict]], dict[str, tuple[str | None, str]]]:
    """Bajas por (liga, equipo) para los partidos próximos + estado de la fuente por liga: (fuente, nota)."""
    from ..config import FOOTBALL
    from ..teams import us_key

    out: dict[tuple[str, str], list[dict]] = defaultdict(list)
    status: dict[str, tuple[str | None, str]] = {}
    for (lg, team), lst in load_manual(today).items():
        out[(lg, team)] += lst
    fb = [m for m in matches if m.get("sport") == "football" and m["league"] in FOOTBALL]
    if fb:
        items, st = apifootball_by_date(s, db, {m["kickoff"][:10] for m in fb}, today)
        by_league: dict[int, list[dict]] = defaultdict(list)
        for it in items:
            by_league[it["league_id"]].append(it)
        for m in fb:
            lid = FOOTBALL[m["league"]].get("apif")
            status[m["league"]] = ("API-Football", "ok") if st == "ok" else (None, st)
            for side in ("home", "away"):
                for it in by_league.get(lid, []):
                    if match_team(it["team"], [m[side]]):
                        out[(m["league"], m[side])].append(
                            {"jugador": it["jugador"], "tipo": "baja" if it["tipo"] == "Missing Fixture" else "duda",
                             "motivo": it["motivo"], "importancia": 1.0 if it["tipo"] == "Missing Fixture" else 0.5})
    for lg, path in ESPN_INJ.items():
        ms = [m for m in matches if m["league"] == lg]
        if not ms:
            continue
        inj = espn_injuries(path)
        status[lg] = ("ESPN", "ok") if inj else (None, "ESPN no devolvió el parte de lesiones")
        keyed = {us_key(t): v for t, v in inj.items()}
        for m in ms:
            for side in ("home", "away"):
                out[(lg, m[side])] += keyed.get(us_key(m[side]), [])
    for m in matches:
        if m["league"] not in status:
            status[m["league"]] = (None, "sin fuente de bajas para esta competición" if m.get("sport") != "tennis"
                                   else "sin parte de lesiones en tenis")
    # sin duplicados (manual + API)
    return {k: list({a["jugador"]: a for a in v}.values()) for k, v in out.items()}, status
