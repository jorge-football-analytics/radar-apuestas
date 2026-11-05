"""Bajas (lesionados y sancionados).

- API-Football (/injuries): el plan gratis NO incluye la temporada actual; con el plan Pro funciona solo.
  Si la API responde que la temporada no está en tu plan, el módulo se desactiva sin romper nada.
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
