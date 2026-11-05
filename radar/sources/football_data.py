"""football-data.co.uk: histórico de resultados con cuotas de apertura y cierre (gratis) y cuotas de la próxima jornada."""
from __future__ import annotations

import io
import logging
import time
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from ..config import FD_TO_LEAGUE, LEAGUES, RAW_DIR, season_code

log = logging.getLogger(__name__)
BASE = "https://www.football-data.co.uk"
UA = {"User-Agent": "radar-apuestas/1.0 (uso personal)"}

# prefijo de columna → nombre de casa
BOOK_PREFIX = {"B365": "bet365", "BW": "bwin", "IW": "interwetten", "PS": "pinnacle", "WH": "williamhill",
               "VC": "betvictor", "1XB": "1xbet", "BF": "betfair", "BFE": "betfair_ex", "LB": "ladbrokes"}
OU_PREFIX = {"B365": "bet365", "P": "pinnacle", "BFE": "betfair_ex", "1XB": "1xbet"}


def _get(url: str, cache: Path, max_age_h: float) -> bytes | None:
    if cache.exists() and (time.time() - cache.stat().st_mtime) < max_age_h * 3600:
        return cache.read_bytes()
    try:
        r = requests.get(url, headers=UA, timeout=30)
        r.raise_for_status()
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(r.content)
        time.sleep(1)  # educado con el servidor
        return r.content
    except requests.RequestException as e:
        log.warning("No se pudo descargar %s: %s", url, e)
        return cache.read_bytes() if cache.exists() else None


def _read_csv(content: bytes) -> pd.DataFrame:
    for enc in ("utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(content), encoding=enc, on_bad_lines="skip")
        except UnicodeDecodeError:
            continue
    return pd.DataFrame()


def _parse_dates(df: pd.DataFrame) -> pd.Series:
    d = pd.to_datetime(df["Date"], format="%d/%m/%Y", errors="coerce")
    d2 = pd.to_datetime(df["Date"], format="%d/%m/%y", errors="coerce")
    return d.fillna(d2)


def load_history(today: date, seasons: int) -> pd.DataFrame:
    """Partidos jugados de las ligas configuradas, últimas `seasons` temporadas (la actual incluida)."""
    frames = []
    for key, lg in LEAGUES.items():
        if not lg["fd"]:
            continue
        for off in range(-(seasons - 1), 1):
            code = season_code(today, off)
            current = off == 0
            content = _get(f"{BASE}/mmz4281/{code}/{lg['fd']}.csv", RAW_DIR / f"{code}_{lg['fd']}.csv",
                           max_age_h=6 if current else 24 * 365)
            if not content:
                continue
            df = _read_csv(content)
            if df.empty or "HomeTeam" not in df:
                continue
            df["league"] = key
            df["season"] = code
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    raw = pd.concat(frames, ignore_index=True)
    return tidy_history(raw)


def tidy_history(raw: pd.DataFrame) -> pd.DataFrame:
    raw = raw.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"]).copy()
    raw["date"] = _parse_dates(raw)
    out = pd.DataFrame({
        "league": raw["league"], "season": raw.get("season"), "date": raw["date"],
        "home": raw["HomeTeam"].str.strip(), "away": raw["AwayTeam"].str.strip(),
        "hg": raw["FTHG"].astype(int), "ag": raw["FTAG"].astype(int),
    })
    # cuotas útiles para backtest / CLV (si existen)
    keep = {
        "open_H": ["PSH", "B365H", "AvgH"], "open_D": ["PSD", "B365D", "AvgD"], "open_A": ["PSA", "B365A", "AvgA"],
        "close_H": ["PSCH", "B365CH", "AvgCH"], "close_D": ["PSCD", "B365CD", "AvgCD"], "close_A": ["PSCA", "B365CA", "AvgCA"],
        "b365_H": ["B365H"], "b365_D": ["B365D"], "b365_A": ["B365A"],
        "max_H": ["MaxH"], "max_D": ["MaxD"], "max_A": ["MaxA"],
        "open_O25": ["P>2.5", "B365>2.5", "Avg>2.5"], "open_U25": ["P<2.5", "B365<2.5", "Avg<2.5"],
        "close_O25": ["PC>2.5", "B365C>2.5", "AvgC>2.5"], "close_U25": ["PC<2.5", "B365C<2.5", "AvgC<2.5"],
        "max_O25": ["Max>2.5"], "max_U25": ["Max<2.5"],
        "b365_O25": ["B365>2.5"], "b365_U25": ["B365<2.5"],
    }
    for col, candidates in keep.items():
        s = pd.Series(float("nan"), index=raw.index)
        for c in candidates:  # primera columna disponible con valor (Pinnacle > bet365 > media)
            if c in raw:
                s = s.fillna(pd.to_numeric(raw[c], errors="coerce"))
        out[col] = s
    return out.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)


def load_fixtures(cache_h: float = 3) -> pd.DataFrame:
    """Próximos partidos con cuotas (se actualiza martes y viernes). Devuelve filas largas: una por casa/selección."""
    content = _get(f"{BASE}/fixtures.csv", RAW_DIR / "fixtures.csv", max_age_h=cache_h)
    if not content:
        return pd.DataFrame()
    return tidy_fixtures(_read_csv(content))


def tidy_fixtures(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty or "Div" not in raw:
        return pd.DataFrame()
    raw = raw[raw["Div"].isin(FD_TO_LEAGUE)].copy()
    raw["date"] = _parse_dates(raw)
    rows = []
    for _, r in raw.iterrows():
        ko = r["date"]
        if pd.isna(ko):
            continue
        t = str(r.get("Time", "") or "")
        if ":" in t:
            hh, mm = t.split(":")[:2]
            ko = ko.replace(hour=int(hh), minute=int(mm))
        # football-data publica horas del Reino Unido; lo pasamos a UTC aproximando con zoneinfo
        ko = pd.Timestamp(ko).tz_localize("Europe/London", ambiguous="NaT", nonexistent="NaT")
        if pd.isna(ko):
            continue
        ko_iso = ko.tz_convert("UTC").isoformat()
        base = dict(league=FD_TO_LEAGUE[r["Div"]], kickoff=ko_iso, home=str(r["HomeTeam"]).strip(),
                    away=str(r["AwayTeam"]).strip(), source="football-data")
        for pre, book in BOOK_PREFIX.items():
            for sel in "HDA":
                v = pd.to_numeric(r.get(f"{pre}{sel}"), errors="coerce")
                if pd.notna(v) and v > 1:
                    rows.append({**base, "bookmaker": book, "market": "1x2", "selection": sel, "point": 0.0, "price": float(v)})
        for pre, book in OU_PREFIX.items():
            for sel, sym in (("O", ">"), ("U", "<")):
                v = pd.to_numeric(r.get(f"{pre}{sym}2.5"), errors="coerce")
                if pd.notna(v) and v > 1:
                    rows.append({**base, "bookmaker": book, "market": "ou", "selection": sel, "point": 2.5, "price": float(v)})
    return pd.DataFrame(rows)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)
