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


# ------------------------------------------------------------------ próximos partidos (gratis)
def _american_to_decimal(v) -> float | None:
    try:
        x = float(str(v).replace("+", ""))
    except (TypeError, ValueError):
        return None
    if x == 0:
        return None
    return round(1 + (x / 100 if x > 0 else 100 / -x), 3)


def _odds_rows(comp: dict, base: dict) -> list[dict]:
    """Cuotas de ESPN BET/DraftKings que trae el marcador de ESPN. Sirven de referencia de mercado."""
    rows = []
    for od in comp.get("odds") or []:
        book = "espn_" + str((od.get("provider") or {}).get("name", "bet")).lower().replace(" ", "")
        add = lambda market, sel, pt, price: rows.append({**base, "bookmaker": book, "market": market,
                                                          "selection": sel, "point": float(pt), "price": price}) \
            if price and price > 1 else None
        ml = od.get("moneyline") or {}
        h = _american_to_decimal(((ml.get("home") or {}).get("close") or {}).get("odds")) or \
            _american_to_decimal((od.get("homeTeamOdds") or {}).get("moneyLine"))
        a = _american_to_decimal(((ml.get("away") or {}).get("close") or {}).get("odds")) or \
            _american_to_decimal((od.get("awayTeamOdds") or {}).get("moneyLine"))
        add("ml", "H", 0, h)
        add("ml", "A", 0, a)
        spread = od.get("spread")
        if spread is not None:
            try:
                sp = float(spread)
                ps = od.get("pointSpread") or {}
                hs = _american_to_decimal(((ps.get("home") or {}).get("close") or {}).get("odds")) or \
                    _american_to_decimal((od.get("homeTeamOdds") or {}).get("spreadOdds")) or 1.91
                as_ = _american_to_decimal(((ps.get("away") or {}).get("close") or {}).get("odds")) or \
                    _american_to_decimal((od.get("awayTeamOdds") or {}).get("spreadOdds")) or 1.91
                add("spread", "H", sp, hs)
                add("spread", "A", sp, as_)
            except (TypeError, ValueError):
                pass
        ou = od.get("overUnder")
        if ou is not None:
            try:
                tot = od.get("total") or {}
                o = _american_to_decimal(((tot.get("over") or {}).get("close") or {}).get("odds")) or \
                    _american_to_decimal(od.get("overOdds")) or 1.91
                u = _american_to_decimal(((tot.get("under") or {}).get("close") or {}).get("odds")) or \
                    _american_to_decimal(od.get("underOdds")) or 1.91
                add("ou", "O", float(ou), o)
                add("ou", "U", float(ou), u)
            except (TypeError, ValueError):
                pass
    return rows


def upcoming(league: str, path: str, sport: str, now, days: int = 4) -> pd.DataFrame:
    """Partidos de los próximos días con cuotas de referencia (sin gastar créditos de ninguna API)."""
    start = now.strftime("%Y%m%d")
    end = (now + pd.Timedelta(days=days)).strftime("%Y%m%d")
    try:
        r = requests.get(f"{BASE}/{path}/scoreboard", headers=UA, timeout=30,
                         params={"dates": f"{start}-{end}", "limit": 500})
        r.raise_for_status()
        events = r.json().get("events", [])
    except requests.RequestException as e:
        log.warning("ESPN próximos %s: %s", league, e)
        return pd.DataFrame()
    rows, fixtures = [], []
    for ev in events:
        try:
            comp = ev["competitions"][0]
            if comp["status"]["type"].get("state") != "pre":
                continue
            teams = {c["homeAway"]: c["team"]["displayName"] for c in comp["competitors"]}
            ko = ev["date"].replace("Z", "+00:00")
            if len(ko) == 22:  # '2026-11-07T00:30+00:00'
                ko = ko[:16] + ":00" + ko[16:]
            base = dict(sport=sport, league=league, kickoff=ko, home=teams["home"], away=teams["away"], source="espn")
            fixtures.append(base)
            rows += _odds_rows(comp, base)
        except (KeyError, TypeError, IndexError):
            continue
    df = pd.DataFrame(rows)
    df.attrs["fixtures"] = fixtures
    return df


def _walk_matches(obj, out: list, tour: str, tname: str = ""):
    """Recorre el JSON de tenis de ESPN buscando partidos (competiciones con 2 jugadores)."""
    if isinstance(obj, dict):
        name = obj.get("name") or obj.get("shortName") or tname
        comps = obj.get("competitors")
        if isinstance(comps, list) and len(comps) == 2 and all(isinstance(c, dict) for c in comps):
            try:
                names = [(c.get("athlete") or {}).get("displayName") or (c.get("athlete") or {}).get("fullName")
                         for c in comps]
                st = ((obj.get("status") or {}).get("type") or {})
                date = obj.get("date") or obj.get("startDate")
                if all(names) and date:
                    winner = [c.get("winner") for c in comps]
                    out.append(dict(tour=tour, tournament=tname, date=date, p1=names[0], p2=names[1],
                                    state=st.get("state"), completed=bool(st.get("completed")), winner=winner))
            except (AttributeError, TypeError):
                pass
            return
        for k, v in obj.items():
            _walk_matches(v, out, tour, obj.get("name", tname) if k in ("groupings", "competitions") else tname)
    elif isinstance(obj, list):
        for v in obj:
            _walk_matches(v, out, tour, tname)


def tennis_scoreboard(tour: str, dates: str | None = None) -> list[dict]:
    params = {"limit": 1000}
    if dates:
        params["dates"] = dates
    try:
        r = requests.get(f"{BASE}/tennis/{tour}/scoreboard", headers=UA, timeout=30, params=params)
        r.raise_for_status()
        js = r.json()
    except requests.RequestException as e:
        log.warning("ESPN tenis %s: %s", tour, e)
        return []
    out: list[dict] = []
    for ev in js.get("events", []):
        _walk_matches(ev, out, tour, ev.get("name", ""))
    return out
