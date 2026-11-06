"""Resultados de NBA y NFL desde la API pública de ESPN (sin clave). Se usan para el modelo Elo y para liquidar."""
from __future__ import annotations

import calendar
import json
import logging
import time
from datetime import date

import pandas as pd
import requests

from ..config import RAW_DIR

log = logging.getLogger(__name__)
BASE = "https://site.api.espn.com/apis/site/v2/sports"
UA = {"User-Agent": "radar-apuestas/2.0 (uso personal)"}
# Meses con partidos (para no pedir meses vacíos)
SEASON_MONTHS = {"basketball/nba": {10, 11, 12, 1, 2, 3, 4, 5, 6}, "football/nfl": {9, 10, 11, 12, 1, 2}}


def _month_range(today: date, years: float):
    y, m = today.year, today.month
    n = int(years * 12)
    for _ in range(n):
        yield y, m
        m -= 1
        if m == 0:
            y, m = y - 1, 12


def _fetch_month(path: str, y: int, m: int, current: bool) -> list[dict]:
    cache = RAW_DIR / f"espn_{path.replace('/', '_')}_{y}{m:02d}.json"
    if cache.exists() and (not current or time.time() - cache.stat().st_mtime < 3 * 3600):
        return json.loads(cache.read_text())
    last = calendar.monthrange(y, m)[1]
    try:
        r = requests.get(f"{BASE}/{path}/scoreboard", headers=UA, timeout=30,
                         params={"dates": f"{y}{m:02d}01-{y}{m:02d}{last}", "limit": 1000})
        r.raise_for_status()
        events = r.json().get("events", [])
    except requests.RequestException as e:
        log.warning("ESPN %s %s-%s: %s", path, y, m, e)
        return json.loads(cache.read_text()) if cache.exists() else []
    out = []
    for ev in events:
        try:
            comp = ev["competitions"][0]
            if not comp["status"]["type"].get("completed"):
                continue
            if ev.get("season", {}).get("type") not in (2, 3):  # temporada regular y playoffs
                continue
            teams = {c["homeAway"]: c for c in comp["competitors"]}
            out.append(dict(date=ev["date"][:10], home=teams["home"]["team"]["displayName"],
                            away=teams["away"]["team"]["displayName"],
                            hg=int(float(teams["home"]["score"])), ag=int(float(teams["away"]["score"])),
                            neutral=bool(comp.get("neutralSite"))))
        except (KeyError, ValueError, TypeError):
            continue
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out))
    time.sleep(0.5)
    return out


def load(league: str, path: str, today: date, years: float = 2.6) -> pd.DataFrame:
    rows = []
    months = SEASON_MONTHS.get(path, set(range(1, 13)))
    for y, m in _month_range(today, years):
        if m not in months:
            continue
        current = (y, m) == (today.year, today.month) or (today.day <= 3 and (y, m) == _prev(today))
        rows += _fetch_month(path, y, m, current)
    if not rows:
        return pd.DataFrame(columns=["league", "date", "home", "away", "hg", "ag", "neutral"])
    df = pd.DataFrame(rows).drop_duplicates(["date", "home", "away"])
    df["date"] = pd.to_datetime(df["date"])
    df["league"] = league
    return df.sort_values("date").reset_index(drop=True)


def _prev(d: date):
    return (d.year, d.month - 1) if d.month > 1 else (d.year - 1, 12)
