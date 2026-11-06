"""tennis-data.co.uk: todos los partidos ATP y WTA con superficie, resultado y cuotas de bet365 y Pinnacle (gratis)."""
from __future__ import annotations

import io
import logging
import time
from datetime import date

import pandas as pd
import requests

from ..config import RAW_DIR

log = logging.getLogger(__name__)
BASE = "http://www.tennis-data.co.uk"
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"}
SACKMANN = "https://raw.githubusercontent.com/JeffSackmann"
STATUS: dict = {}  # diagnóstico de la última carga


def _get(url: str, cache, max_age_h: float) -> bytes | None:
    if cache.exists() and (time.time() - cache.stat().st_mtime) < max_age_h * 3600:
        return cache.read_bytes()
    try:
        r = requests.get(url, headers=UA, timeout=60)
        r.raise_for_status()
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(r.content)
        time.sleep(1)
        return r.content
    except requests.RequestException as e:
        log.warning("tennis-data %s: %s", url, e)
        return cache.read_bytes() if cache.exists() else None


def load(today: date, years: int = 4) -> pd.DataFrame:
    """tennis-data.co.uk (con cuotas) → respaldo: repositorio de Jeff Sackmann → respaldo: resultados de ESPN."""
    df = _load_tennis_data(today, years)
    STATUS.clear()
    STATUS["tennis-data"] = len(df)
    if df.empty or df["date"].max() < pd.Timestamp(today) - pd.Timedelta(days=45):
        extra = _load_sackmann(today, years)
        STATUS["sackmann"] = len(extra)
        espn_rows = _load_espn(today)
        STATUS["espn"] = len(espn_rows)
        df = pd.concat([x for x in (df, extra, espn_rows) if not x.empty], ignore_index=True) if \
            any(not x.empty for x in (df, extra, espn_rows)) else df
        if not df.empty:
            from ..teams import player_key
            df["_k"] = df["date"].dt.strftime("%Y-%m-%d") + df["winner"].map(player_key) + df["loser"].map(player_key)
            df = df.drop_duplicates("_k").drop(columns="_k").sort_values("date", kind="stable").reset_index(drop=True)
    return df


def _load_sackmann(today: date, years: int) -> pd.DataFrame:
    frames = []
    for tour in ("atp", "wta"):
        for y in range(today.year - years + 1, today.year + 1):
            c = _get(f"{SACKMANN}/tennis_{tour}/master/{tour}_matches_{y}.csv", RAW_DIR / f"sack_{tour}_{y}.csv",
                     max_age_h=12 if y == today.year else 24 * 60)
            if not c:
                continue
            try:
                d = pd.read_csv(io.BytesIO(c))
            except Exception:  # noqa: BLE001
                continue
            frames.append(pd.DataFrame({
                "tour": tour, "date": pd.to_datetime(d["tourney_date"].astype(str), format="%Y%m%d", errors="coerce"),
                "tournament": d.get("tourney_name"), "surface": d.get("surface", "Hard").fillna("Hard"),
                "winner": d["winner_name"].astype(str), "loser": d["loser_name"].astype(str),
                "completed": ~d.get("score", pd.Series("", index=d.index)).astype(str).str.contains("RET|W/O|DEF"),
            }))
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    for c in ("b365_w", "b365_l", "ps_w", "ps_l", "avg_w", "avg_l", "max_w", "max_l"):
        out[c] = float("nan")
    return out.dropna(subset=["date"])


def _load_espn(today: date) -> pd.DataFrame:
    """Últimos ~5 meses de resultados ATP/WTA desde ESPN (por si las otras fuentes van con retraso)."""
    import calendar
    import json

    from . import espn
    rows = []
    for tour in ("atp", "wta"):
        y, m = today.year, today.month
        for _ in range(5):
            cache = RAW_DIR / f"espn_tennis_{tour}_{y}{m:02d}.json"
            current = (y, m) == (today.year, today.month)
            if cache.exists() and (not current or time.time() - cache.stat().st_mtime < 6 * 3600):
                got = json.loads(cache.read_text())
            else:
                last = calendar.monthrange(y, m)[1]
                got = espn.tennis_scoreboard(tour, f"{y}{m:02d}01-{y}{m:02d}{last}")
                if got:
                    cache.parent.mkdir(parents=True, exist_ok=True)
                    cache.write_text(json.dumps(got))
            for g in got:
                if g.get("completed") and any(g.get("winner") or []):
                    w = 0 if (g["winner"] or [False])[0] else 1
                    names = [g["p1"], g["p2"]]
                    rows.append(dict(tour=tour, date=pd.to_datetime(g["date"][:10], errors="coerce"),
                                     tournament=g.get("tournament"), surface=_surface_guess(g.get("tournament", "")),
                                     winner=names[w], loser=names[1 - w], completed=True))
            m -= 1
            if m == 0:
                y, m = y - 1, 12
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).dropna(subset=["date"])
    for c in ("b365_w", "b365_l", "ps_w", "ps_l", "avg_w", "avg_l", "max_w", "max_l"):
        out[c] = float("nan")
    return out


def _surface_guess(name: str) -> str:
    from ..model.elo import surface_of
    return surface_of(str(name).lower().replace(" ", "_"))


def _load_tennis_data(today: date, years: int) -> pd.DataFrame:
    frames = []
    for tour, suffix in (("atp", ""), ("wta", "w")):
        for y in range(today.year - years + 1, today.year + 1):
            current = y == today.year
            content = None
            for ext in ("xlsx", "xls"):
                content = _get(f"{BASE}/{y}{suffix}/{y}.{ext}", RAW_DIR / f"tennis_{tour}_{y}.{ext}",
                               max_age_h=6 if current else 24 * 365)
                if content:
                    break
            if not content:
                continue
            try:
                df = pd.read_excel(io.BytesIO(content))
            except Exception as e:  # noqa: BLE001
                log.warning("tennis-data %s %s ilegible: %s", tour, y, e)
                continue
            df["tour"] = tour
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return tidy(pd.concat(frames, ignore_index=True))


def tidy(raw: pd.DataFrame) -> pd.DataFrame:
    raw = raw.dropna(subset=["Winner", "Loser", "Date"]).copy()
    out = pd.DataFrame({
        "tour": raw["tour"], "date": pd.to_datetime(raw["Date"], errors="coerce"),
        "tournament": raw.get("Tournament"), "surface": raw.get("Surface", "Hard").fillna("Hard"),
        "winner": raw["Winner"].astype(str).str.strip(), "loser": raw["Loser"].astype(str).str.strip(),
        "completed": raw.get("Comment", "Completed").fillna("Completed").astype(str).str.startswith("Compl"),
    })
    for col, cands in {"b365_w": ["B365W"], "b365_l": ["B365L"], "ps_w": ["PSW"], "ps_l": ["PSL"],
                       "avg_w": ["AvgW"], "avg_l": ["AvgL"], "max_w": ["MaxW"], "max_l": ["MaxL"]}.items():
        s = pd.Series(float("nan"), index=raw.index)
        for c in cands:
            if c in raw:
                s = s.fillna(pd.to_numeric(raw[c], errors="coerce"))
        out[col] = s
    return out.dropna(subset=["date"]).sort_values("date", kind="stable").reset_index(drop=True)
