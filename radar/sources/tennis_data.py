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
UA = {"User-Agent": "radar-apuestas/2.0 (uso personal)"}


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
