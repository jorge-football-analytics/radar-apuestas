"""Resultados de NBA y NFL desde la API pública de ESPN (sin clave). Se usan para el modelo Elo y para liquidar."""
from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta

import pandas as pd
import requests

from ..config import RAW_DIR

log = logging.getLogger(__name__)
BASE = "https://site.api.espn.com/apis/site/v2/sports"
UA = {"User-Agent": "radar-apuestas/2.0 (uso personal)"}
# Meses con partidos (para no pedir días vacíos)
SEASON_MONTHS = {"basketball/nba": {10, 11, 12, 1, 2, 3, 4, 5, 6}, "football/nfl": {9, 10, 11, 12, 1, 2}}
# Copas europeas en ESPN (para liquidar sin gastar créditos de The Odds API)
SOCCER_CUPS = {"champions": "uefa.champions", "europa": "uefa.europa", "conference": "uefa.europa.conf"}
WORKERS = 6

# OJO (noviembre de 2026): el marcador de ESPN ya NO acepta rangos de fechas (dates=AAAAMMDD-AAAAMMDD → 400 Bad
# Request), ni con limit ni sin él. Acepta un día suelto (dates=AAAAMMDD) y, en la NFL, temporada y semana
# (dates=AAAA&seasontype=2&week=N). Por eso todo se pide día a día (o semana a semana) en paralelo y con caché:
# un día ya jugado no cambia, así que solo se descarga una vez.


def _get_json(url: str, params: dict) -> dict | None:
    try:
        r = requests.get(url, headers=UA, timeout=30, params=params)
        r.raise_for_status()
        return r.json()
    except (requests.RequestException, ValueError) as e:
        log.warning("ESPN %s %s: %s", url.rsplit("/sports/", 1)[-1], params, e)
        return None


def _completed_rows(events: list[dict], season_filter: bool = True) -> list[dict]:
    out = []
    for ev in events:
        try:
            comp = ev["competitions"][0]
            if not comp["status"]["type"].get("completed"):
                continue
            if season_filter and ev.get("season", {}).get("type") not in (2, 3, None):  # temporada regular y playoffs
                continue
            teams = {c["homeAway"]: c for c in comp["competitors"]}
            out.append(dict(date=ev["date"][:10], home=teams["home"]["team"]["displayName"],
                            away=teams["away"]["team"]["displayName"],
                            hg=int(float(teams["home"]["score"])), ag=int(float(teams["away"]["score"])),
                            neutral=bool(comp.get("neutralSite"))))
        except (KeyError, ValueError, TypeError):
            continue
    return out


def _cached(cache, fresh_h: float | None, fetch) -> list[dict] | None:
    """fresh_h=None → la caché vale para siempre (días ya jugados)."""
    if cache.exists() and (fresh_h is None or time.time() - cache.stat().st_mtime < fresh_h * 3600):
        return json.loads(cache.read_text())
    got = fetch()
    if got is None:
        return json.loads(cache.read_text()) if cache.exists() else None
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(got))
    return got


def _fetch_day(path: str, day: date, today: date) -> list[dict]:
    cache = RAW_DIR / f"espn_{path.replace('/', '_')}_d{day:%Y%m%d}.json"
    settled = day < today - timedelta(days=2)

    def fetch():
        js = _get_json(f"{BASE}/{path}/scoreboard", {"dates": f"{day:%Y%m%d}"})
        return None if js is None else _completed_rows(js.get("events", []), not path.startswith("soccer/"))

    return _cached(cache, None if settled else 3, fetch) or []


def _fetch_nfl_week(year: int, stype: int, week: int, today: date) -> list[dict]:
    cache = RAW_DIR / f"espn_nfl_{year}_{stype}_{week:02d}.json"
    if cache.exists():
        c = json.loads(cache.read_text())
        if c.get("done") or time.time() - cache.stat().st_mtime < 3 * 3600:  # semana cerrada → para siempre
            return c.get("rows", [])
    js = _get_json(f"{BASE}/football/nfl/scoreboard", {"dates": year, "seasontype": stype, "week": week})
    if js is None:
        return json.loads(cache.read_text()).get("rows", []) if cache.exists() else []
    evs = js.get("events", [])
    rows = _completed_rows(evs)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"done": bool(evs) and len(rows) == len(evs), "rows": rows}))
    return rows


def _days(today: date, years: float, months: set[int]):
    d = today - timedelta(days=int(years * 365))
    while d <= today:
        if d.month in months:
            yield d
        d += timedelta(days=1)


def load(league: str, path: str, today: date, years: float = 1.6) -> pd.DataFrame:
    """Resultados de la NBA (día a día) o de la NFL (semana a semana) de las últimas temporadas."""
    rows: list[dict] = []
    if path == "football/nfl":
        start = today.year - 2 if today.month >= 9 else today.year - 3
        jobs = [(y, st, w) for y in range(start, today.year + 1) for st, n in ((2, 18), (3, 5))
                for w in range(1, n + 1) if date(y, 9, 1) <= today]
        with ThreadPoolExecutor(WORKERS) as ex:
            for got in ex.map(lambda j: _fetch_nfl_week(*j, today), jobs):
                rows += got
    else:
        days = list(_days(today, years, SEASON_MONTHS.get(path, set(range(1, 13)))))
        with ThreadPoolExecutor(WORKERS) as ex:
            for got in ex.map(lambda d: _fetch_day(path, d, today), days):
                rows += got
    if not rows:
        return pd.DataFrame(columns=["league", "date", "home", "away", "hg", "ag", "neutral"])
    df = pd.DataFrame(rows).drop_duplicates(["date", "home", "away"])
    df["date"] = pd.to_datetime(df["date"])
    df["league"] = league
    return df.sort_values("date").reset_index(drop=True)


def cup_results(today: date, days: int = 4) -> list[dict]:
    """Resultados de Champions, Europa y Conference de los últimos días, con el formato de /scores de The Odds API
    (home_team, away_team, completed, scores) para que la liquidación los trate igual."""
    out = []
    for league, slug in SOCCER_CUPS.items():
        for k in range(days):
            d = today - timedelta(days=k)
            for r in _fetch_day(f"soccer/{slug}", d, today):
                out.append(dict(league=league, home_team=r["home"], away_team=r["away"], completed=True,
                                scores=[{"name": r["home"], "score": r["hg"]}, {"name": r["away"], "score": r["ag"]}]))
    return out


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
    """Cuotas de ESPN BET/DraftKings que trae el marcador de ESPN. Sirven de referencia de mercado.
    Si falta la cuota de un lado NO se inventa (antes se rellenaba con 1,91 y entraba en el consenso como una casa)."""
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
                    _american_to_decimal((od.get("homeTeamOdds") or {}).get("spreadOdds"))
                as_ = _american_to_decimal(((ps.get("away") or {}).get("close") or {}).get("odds")) or \
                    _american_to_decimal((od.get("awayTeamOdds") or {}).get("spreadOdds"))
                add("spread", "H", sp, hs)
                add("spread", "A", sp, as_)
            except (TypeError, ValueError):
                pass
        ou = od.get("overUnder")
        if ou is not None:
            try:
                tot = od.get("total") or {}
                o = _american_to_decimal(((tot.get("over") or {}).get("close") or {}).get("odds")) or \
                    _american_to_decimal(od.get("overOdds"))
                u = _american_to_decimal(((tot.get("under") or {}).get("close") or {}).get("odds")) or \
                    _american_to_decimal(od.get("underOdds"))
                add("ou", "O", float(ou), o)
                add("ou", "U", float(ou), u)
            except (TypeError, ValueError):
                pass
    return rows


def upcoming(league: str, path: str, sport: str, now, days: int = 4) -> pd.DataFrame:
    """Partidos de los próximos días con cuotas de referencia de DraftKings (gratis), pedidos día a día."""
    rows, fixtures, seen = [], [], set()
    start = now.date() if hasattr(now, "date") else now
    for k in range(days + 1):
        js = _get_json(f"{BASE}/{path}/scoreboard", {"dates": f"{start + timedelta(days=k):%Y%m%d}"})
        for ev in (js or {}).get("events", []):
            try:
                if ev.get("id") in seen:
                    continue
                seen.add(ev.get("id"))
                comp = ev["competitions"][0]
                if comp["status"]["type"].get("state") != "pre":
                    continue
                teams = {c["homeAway"]: c["team"]["displayName"] for c in comp["competitors"]}
                ko = ev["date"].replace("Z", "+00:00")
                if len(ko) == 22:  # '2026-11-07T00:30+00:00'
                    ko = ko[:16] + ":00" + ko[16:]
                base = dict(sport=sport, league=league, kickoff=ko, home=teams["home"], away=teams["away"],
                            source="espn", fetched_at=pd.Timestamp.now(tz="UTC").isoformat())
                fixtures.append({k2: v for k2, v in base.items() if k2 != "fetched_at"})
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
