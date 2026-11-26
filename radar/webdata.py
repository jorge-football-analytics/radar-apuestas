"""Capa de datos de la web: todo lo que app.py necesita, en funciones puras (sin Streamlit) para poder probarlas y
cachearlas. La web llama a load_dashboard() una vez por cambio del fichero de la base de datos."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import assistant
from .config import Settings
from .db import DB
from .learn import metrics

ODDS_BANDS = [(0, 1.30, "< 1,30"), (1.30, 1.60, "1,30-1,60"), (1.60, 2.00, "1,60-2,00"), (2.00, 2.60, "2,00-2,60"),
              (2.60, 99, "> 2,60")]
TRACKED = ("valor", "solida")


def metrics_since_default() -> str:
    """METRICS_SINCE: variable de entorno (GitHub Actions) o secreto de Streamlit (se expone como variable)."""
    v = (os.getenv("METRICS_SINCE") or "").strip()
    try:
        return datetime.fromisoformat(v).date().isoformat() if v else ""
    except ValueError:
        return ""


def odds_band(o: float) -> str:
    for lo, hi, name in ODDS_BANDS:
        if lo <= o < hi or (hi == 2.60 and o == 2.60):
            return name
    return "> 2,60"


def load_dashboard(db_path: str | Path, now: datetime | None = None, since: str = "") -> dict:
    s = Settings(db_path=Path(db_path))
    now = now or datetime.now(timezone.utc)
    nowi = now.isoformat()
    db = DB(db_path)
    since = since or ""
    tracked = "','".join(TRACKED)
    d = {
        "analysis": db.df("SELECT * FROM analysis WHERE kickoff > ? ORDER BY kickoff", (nowi,)),
        "picks": db.df("SELECT * FROM picks WHERE kickoff > ? AND bookmaker=? ORDER BY score DESC", (nowi, s.book)),
        "combos": db.df("SELECT * FROM combos WHERE won IS NULL AND first_kickoff > ? AND bookmaker=? "
                        "ORDER BY ev DESC", (nowi, s.book)),
        "resolved": db.df(f"SELECT * FROM picks WHERE won IN (0,1) AND tier IN ('{tracked}') AND bookmaker=? "
                          "AND COALESCE(shown_at, created) >= ? ORDER BY kickoff", (s.book, since)),
        "resolved_combos": db.df("SELECT * FROM combos WHERE won IN (0,1) AND tier IN ('valor','solida') AND "
                                 "created >= ? ORDER BY first_kickoff", (since,)),
        "matches": db.df("SELECT * FROM matches WHERE status='pendiente' AND kickoff > ? ORDER BY kickoff", (nowi,)),
        "odds": db.df("SELECT o.* FROM odds o JOIN matches m ON m.id=o.match_id WHERE m.status='pendiente' "
                      "AND m.kickoff > ?", (nowi,)),
        "preds": db.df("SELECT p.* FROM predictions p JOIN matches m ON m.id=p.match_id WHERE m.status='pendiente' "
                       "AND m.kickoff > ?", (nowi,)),
        "info": {r["match_id"]: json.loads(r["info"]) for r in db.conn.execute(
            "SELECT i.match_id, i.info FROM match_info i JOIN matches m ON m.id=i.match_id WHERE m.kickoff > ?",
            (nowi,)).fetchall()},
        "pool": assistant.pool_from_db(db, now, s.book, s.min_ev, s.solid_min_prob, s.solid_min_ev,
                                       s.solid_b365_max_age_h),
        "calib": db.df("SELECT p.p_final, p.won, m.sport, m.kickoff FROM predictions p JOIN matches m "
                       "ON m.id=p.match_id WHERE p.won IN (0,1) AND p.p_market IS NOT NULL AND m.kickoff >= ?",
                       (since,)),
        "metrics": {t: metrics(db, t, since) for t in TRACKED},
        "backtest": db.get("backtest"), "backtest_tennis": db.get("backtest_tennis"), "weights": db.get("weights"),
        "last_run": db.get("last_run"), "credits": db.get("odds_api_remaining"), "fuentes": db.get("fuentes") or {},
        "oddspapi_used": db.get(f"oddspapi_used_{now:%Y-%m}", 0),
        "runs": db.df("SELECT ts, detail FROM runs ORDER BY rowid DESC LIMIT 15"),
        "tables": {r["key"][6:]: json.loads(r["value"]) for r in db.conn.execute(
            "SELECT key, value FROM kv WHERE key LIKE 'tabla_%'").fetchall()},
        "since": since, "now": nowi,
    }
    db.close()
    return d


# ------------------------------------------------------------------ rendimiento
def _units(df: pd.DataFrame) -> pd.Series:
    return (df["odds"] - 1) * df["won"] - (1 - df["won"])


def perf_series(resolved: pd.DataFrame, bankroll: float = 100.0, mode: str = "unit") -> pd.DataFrame:
    """Curva de banca y yield acumulado. mode='stake' usa los stakes recomendados; 'unit', 1 unidad por jugada
    (las sólidas sin ventaja tienen stake 0, así que para compararlas sirve más la vista por unidad)."""
    if resolved.empty:
        return pd.DataFrame(columns=["date", "n", "profit", "staked", "bank", "yield", "drawdown"])
    df = resolved.sort_values("kickoff").copy()
    if mode == "stake":
        stake = df["stake"].fillna(0)
        profit = df["profit"].fillna(0)
    else:
        stake = pd.Series(1.0, index=df.index)
        profit = _units(df)
    out = pd.DataFrame({"date": pd.to_datetime(df["kickoff"], utc=True, format="ISO8601"),
                        "profit": profit.cumsum().values, "staked": stake.cumsum().values})
    out["n"] = np.arange(1, len(out) + 1)
    out["bank"] = (bankroll if mode == "stake" else 0) + out["profit"]
    out["yield"] = np.where(out["staked"] > 0, out["profit"] / out["staked"].replace(0, np.nan), np.nan)
    peak = out["bank"].cummax()
    out["drawdown"] = out["bank"] - peak
    return out


def kpis(resolved: pd.DataFrame, mode: str = "unit") -> dict:
    if resolved.empty:
        return {"n": 0}
    u = _units(resolved)
    if mode == "stake" and resolved["stake"].fillna(0).sum() > 0:
        staked, profit = resolved["stake"].fillna(0).sum(), resolved["profit"].fillna(0).sum()
    else:
        staked, profit = float(len(resolved)), float(u.sum())
    ser = perf_series(resolved, 0, "stake" if mode == "stake" else "unit")
    clv = resolved["clv"].dropna() if "clv" in resolved else pd.Series(dtype=float)
    return {"n": len(resolved), "hit": float(resolved["won"].mean()), "expected": float(resolved["p_final"].mean()),
            "profit": float(profit), "staked": float(staked), "yield": float(profit / staked) if staked else None,
            "avg_odds": float(resolved["odds"].mean()), "clv": float(clv.mean()) if len(clv) else None,
            "clv_n": int(len(clv)), "max_dd": float(ser["drawdown"].min()) if not ser.empty else 0.0}


def breakdown(resolved: pd.DataFrame, by: str) -> pd.DataFrame:
    """Desglose por deporte, mercado, nivel o rango de cuota ('band')."""
    if resolved.empty:
        return pd.DataFrame(columns=["grupo", "n", "acierto", "esperado", "yield", "unidades"])
    df = resolved.assign(u=_units(resolved), band=resolved["odds"].map(odds_band))
    g = df.groupby(by).agg(n=("u", "size"), acierto=("won", "mean"), esperado=("p_final", "mean"),
                           unidades=("u", "sum"))
    g["yield"] = g["unidades"] / g["n"]
    g = g.reset_index().rename(columns={by: "grupo"})
    if by == "band":
        order = {name: i for i, (_, _, name) in enumerate(ODDS_BANDS)}
        return g.sort_values("grupo", key=lambda x: x.map(order)).reset_index(drop=True)
    return g.sort_values("n", ascending=False).reset_index(drop=True)


def calibration(cal: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    if cal.empty:
        return pd.DataFrame(columns=["p", "real", "n"])
    c = cal.assign(b=pd.cut(cal["p_final"], np.linspace(0, 1, bins + 1)))
    return c.groupby("b", observed=True).agg(p=("p_final", "mean"), real=("won", "mean"),
                                             n=("won", "size")).reset_index(drop=True)


# ------------------------------------------------------------------ radar del día
LIGHT_ORDER = {"verde": 0, "ambar": 1, "rojo": 2, "gris": 3}


def radar_table(analysis: pd.DataFrame, sport: str | None = None, light: list[str] | None = None,
                hours: float | None = None, now: datetime | None = None, query: str = "") -> pd.DataFrame:
    if analysis.empty:
        return analysis
    df = analysis.copy()
    if sport and sport != "todos":
        df = df[df["sport"] == sport]
    if light:
        df = df[df["light"].isin(light)]
    if hours:
        now = now or datetime.now(timezone.utc)
        ko = pd.to_datetime(df["kickoff"], utc=True, format="ISO8601")
        df = df[ko <= pd.Timestamp(now) + pd.Timedelta(hours=hours)]
    if query:
        q = query.lower()
        df = df[df["home"].str.lower().str.contains(q, regex=False) | df["away"].str.lower().str.contains(q, regex=False)
                | df["league"].str.lower().str.contains(q, regex=False)]
    return df.assign(_l=df["light"].map(LIGHT_ORDER).fillna(9)).sort_values(["kickoff", "_l"]).drop(columns="_l")


def day_summary(analysis: pd.DataFrame) -> dict:
    if analysis.empty:
        return {"partidos": 0, "verde": 0, "ambar": 0, "rojo": 0, "gris": 0, "elite": 0, "solida": 0}
    lc = analysis["light"].value_counts()
    tc = analysis["tier"].value_counts()
    return {"partidos": len(analysis), **{k: int(lc.get(k, 0)) for k in LIGHT_ORDER},
            "elite": int(tc.get("valor", 0)), "solida": int(tc.get("solida", 0)),
            "verificar": int(tc.get("verificar", 0)), "revisar": int(tc.get("revisar", 0))}
