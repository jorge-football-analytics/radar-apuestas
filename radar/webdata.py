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
TRACKED = ("valor", "solida", "moderado", "cuota_alta")
TIER_NAMES = {"valor": "Élite", "solida": "Sólidas", "moderado": "Valor moderado", "cuota_alta": "Cuota alta"}


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


def _has(db: DB, table: str) -> bool:
    return db.conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone() is not None


def combos_as_picks(rc: pd.DataFrame) -> pd.DataFrame:
    """Combinadas resueltas con las columnas de una jugada (para usar las mismas métricas)."""
    if rc.empty:
        return pd.DataFrame(columns=["kickoff", "odds", "won", "stake", "profit", "p_final", "tier", "sport", "market"])
    return rc.assign(kickoff=rc["first_kickoff"], p_final=rc["prob"], market="combinada",
                     sport=rc["sports"].fillna("varios"), tier=rc["risk"])


OPERABLE = "bookmaker=? OR reasons LIKE '%Cuota ref. mercado%'"


def load_dashboard(db_path: str | Path, now: datetime | None = None, since: str = "") -> dict:
    s = Settings(db_path=Path(db_path))
    now = now or datetime.now(timezone.utc)
    nowi = now.isoformat()
    db = DB(db_path)
    since = since or ""
    tracked = "','".join(TRACKED)
    d = {
        "analysis": db.df("SELECT * FROM analysis WHERE kickoff > ? ORDER BY kickoff", (nowi,)),
        # bet365 o, desde la v6, la mejor cuota real de otra casa como referencia operable (se nombra la casa)
        "picks": db.df(f"SELECT * FROM picks WHERE kickoff > ? AND ({OPERABLE}) ORDER BY score DESC", (nowi, s.book)),
        "combos": db.df("SELECT * FROM combos WHERE won IS NULL AND first_kickoff > ? AND bookmaker=? "
                        "ORDER BY ev DESC", (nowi, s.book)),
        "resolved": db.df(f"SELECT * FROM picks WHERE won IN (0,1) AND tier IN ('{tracked}') AND ({OPERABLE}) "
                          "AND COALESCE(shown_at, created) >= ? ORDER BY kickoff", (s.book, since)),
        "resolved_combos": db.df("SELECT * FROM combos WHERE won IN (0,1) AND created >= ? ORDER BY first_kickoff",
                                 (since,)),
        "selections": db.df("SELECT s.* FROM selections s JOIN matches m ON m.id=s.match_id WHERE m.status='pendiente' "
                            "AND m.kickoff > ?", (nowi,)) if _has(db, "selections") else pd.DataFrame(),
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
    # partidos de cuadros de tenis aún sin rival (TBD): no son partidos todavía
    for k in ("analysis", "matches"):
        df = d[k]
        if not df.empty and {"home", "away"} <= set(df.columns):
            d[k] = df[~df["home"].isin(PLACEHOLDER) & ~df["away"].isin(PLACEHOLDER)].reset_index(drop=True)
    if not d["pool"].empty:
        d["pool"] = d["pool"][~d["pool"]["home"].isin(PLACEHOLDER) & ~d["pool"]["away"].isin(PLACEHOLDER)]
    return d


PLACEHOLDER = {"TBD", "TBA", "Por determinar", "", None}


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


def kpis(resolved: pd.DataFrame, mode: str = "unit", bankroll: float = 100.0) -> dict:
    """n, acierto real y esperado, beneficio, yield (beneficio / apostado), ROI (beneficio / banca inicial; en
    unidades, igual al yield), cuota media, CLV y caída máxima."""
    if resolved.empty:
        return {"n": 0}
    u = _units(resolved)
    if mode == "stake" and resolved["stake"].fillna(0).sum() > 0:
        staked, profit = resolved["stake"].fillna(0).sum(), resolved["profit"].fillna(0).sum()
    else:
        staked, profit = float(len(resolved)), float(u.sum())
    ser = perf_series(resolved, 0, "stake" if mode == "stake" else "unit")
    clv = resolved["clv"].dropna() if "clv" in resolved else pd.Series(dtype=float)
    real_money = mode == "stake" and resolved["stake"].fillna(0).sum() > 0
    return {"n": len(resolved), "hit": float(resolved["won"].mean()), "expected": float(resolved["p_final"].mean()),
            "profit": float(profit), "staked": float(staked), "yield": float(profit / staked) if staked else None,
            "roi": float(profit / bankroll) if real_money and bankroll else (float(profit / staked) if staked else None),
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
        return {"partidos": 0, "verde": 0, "ambar": 0, "rojo": 0, "gris": 0, "elite": 0, "solida": 0, "moderado": 0,
                "cuota_alta": 0}
    lc = analysis["light"].value_counts()
    tc = analysis["tier"].value_counts()
    return {"partidos": len(analysis), **{k: int(lc.get(k, 0)) for k in LIGHT_ORDER},
            "elite": int(tc.get("valor", 0)), "solida": int(tc.get("solida", 0)),
            "moderado": int(tc.get("moderado", 0)), "cuota_alta": int(tc.get("cuota_alta", 0)),
            "verificar": int(tc.get("verificar", 0)), "revisar": int(tc.get("revisar", 0))}


def level_summary(resolved: pd.DataFrame, resolved_combos: pd.DataFrame, bankroll: float = 100.0) -> pd.DataFrame:
    """Una fila por nivel: Élite, Sólidas y cada perfil de combinada, con jugadas, acierto, yield y ROI."""
    rows = []
    names = TIER_NAMES
    for t in TRACKED:
        r = resolved[resolved["tier"] == t] if not resolved.empty else resolved
        k = kpis(r, "unit")
        rows.append(dict(nivel=names[t], **{x: k.get(x) for x in ("n", "hit", "expected", "yield", "profit")},
                         roi=kpis(r, "stake", bankroll).get("roi") if k["n"] else None))
    cp = combos_as_picks(resolved_combos)
    for prof in (cp["tier"].dropna().unique().tolist() if not cp.empty else []):
        r = cp[cp["tier"] == prof]
        k = kpis(r, "unit")
        rows.append(dict(nivel=f"Combinada {prof}", **{x: k.get(x) for x in ("n", "hit", "expected", "yield", "profit")},
                         roi=kpis(r, "stake", bankroll).get("roi")))
    return pd.DataFrame(rows)


def confidence_score(p: float, ev: float | None = None, n_books: int | None = None, n_sharp: int | None = None,
                     priced: bool = True, age_h: float | None = None, derived: bool = False) -> int:
    """Confianza 0-100 en la recomendación (para el velocímetro). Mide la solidez de la estimación, no la
    probabilidad de ganar: referencia sharp, número de casas válidas, cuota de bet365 confirmada y reciente,
    margen frente al precio justo, probabilidad y si el mercado es directo o derivado."""
    def ok(x):
        return x is not None and not (isinstance(x, float) and pd.isna(x))
    sc = 0.0
    sc += 25 if ok(n_sharp) and n_sharp >= 1 else 0
    sc += 15 * min(1.0, (n_books or 0) / 4) if ok(n_books) else 0
    sc += 20 if priced else 0
    sc -= 10 if ok(age_h) and age_h > 24 else 0
    if ok(ev):
        sc += 25 * min(1.0, max(0.0, (ev + 0.03) / 0.08))
    sc += 15 * min(1.0, max(0.0, (p - 0.25) / 0.5)) if ok(p) else 0
    sc -= 10 if derived else 0
    return int(round(min(100, max(0, sc))))


# ------------------------------------------------------------------ recomendaciones del día
TIER_RANK = {"valor": 0, "cuota_alta": 1, "solida": 2, "moderado": 3, "objetivo": 4}


def daily_picks(pool: pd.DataFrame, now: datetime | None = None, n_min: int = 8, n_max: int = 15,
                hours: float = 48) -> pd.DataFrame:
    """Recomendaciones del día (8-15): primero los 4 niveles (élite, cuota alta, sólidas, valor moderado) con cuota
    operable, una por partido y repartidas entre deportes; si no llegan a n_min, se completa con selecciones de
    probabilidad alta y valor ≥ 0 a su cuota mínima («objetivo»: cogerla solo a esa cuota o más en bet365).
    Nunca entra una selección con valor negativo a la cuota indicada."""
    cols = list(pool.columns) + ["rank_tier", "conf", "objetivo"]
    if pool is None or pool.empty:
        return pd.DataFrame(columns=cols)
    now = now or datetime.now(timezone.utc)
    df = pool.copy()
    ko = pd.to_datetime(df["kickoff"], utc=True, format="ISO8601", errors="coerce")
    df = df[(ko > pd.Timestamp(now)) & (ko <= pd.Timestamp(now) + pd.Timedelta(hours=hours))]
    if df.empty:
        return pd.DataFrame(columns=cols)
    df["conf"] = [confidence_score(r.p_final, r.ev, r.n_books, r.n_sharp, bool(r.priced), r.b365_age_h, bool(r.derived))
                  for r in df.itertuples()]
    main = df[df["priced"] & df["tier"].isin(["valor", "cuota_alta", "solida", "moderado"])].copy()
    main["rank_tier"], main["objetivo"] = main["tier"].map(TIER_RANK), False
    main = main.sort_values(["rank_tier", "conf", "ev"], ascending=[True, False, False])
    out, seen, per_sport = [], set(), {}
    cap = max(4, n_max // max(1, main["sport"].nunique())) if not main.empty else n_max
    for r in main.to_dict("records"):  # una por partido y reparto entre deportes
        if r["match_id"] in seen or per_sport.get(r["sport"], 0) >= cap:
            continue
        out.append(r); seen.add(r["match_id"]); per_sport[r["sport"]] = per_sport.get(r["sport"], 0) + 1
    for r in main.to_dict("records"):  # si sobra hueco, sin límite por deporte
        if len(out) >= n_max:
            break
        if r["match_id"] not in seen:
            out.append(r); seen.add(r["match_id"])
    if len(out) < n_min:
        rest = df[~df["match_id"].isin(seen) & (df["p_final"] >= 0.55) & (df["tier"] != "revisar")].copy()
        rest["objetivo"] = True
        rest["cur_odds"] = rest["odds"].where(rest["priced"])
        rest["rank_tier"] = TIER_RANK["objetivo"]
        rest["min_odds"] = (100 / rest["p_final"]).map(lambda x: int(-(-x // 1)) / 100)  # justo, redondeo arriba
        # a la cuota indicada el valor es ≥ 0: si la cuota operable ya la paga, se usa esa
        rest["odds"] = rest[["odds", "min_odds"]].max(axis=1).where(rest["priced"], rest["min_odds"])
        rest["ev"] = rest["p_final"] * rest["odds"] - 1
        rest = rest.sort_values(["conf", "p_final"], ascending=False)
        for r in rest.to_dict("records"):
            if len(out) >= n_min:
                break
            if r["match_id"] not in seen:
                out.append(r); seen.add(r["match_id"])
    res = pd.DataFrame(out)
    return res.reset_index(drop=True) if not res.empty else pd.DataFrame(columns=cols)


def combo_of_day(combos: pd.DataFrame) -> dict | None:
    """La combinada recomendada del día: con valor si la hay; si no, la sólida de más probabilidad; si no, la de menor
    margen con todas las cuotas operables. Nunca una «por confirmar» si existe otra."""
    if combos is None or combos.empty:
        return None
    pref = {"valor": 0, "solida": 1, "sin_ventaja": 2, "pronostico": 2, "verificar": 3}
    c = combos.assign(_p=combos["tier"].map(pref).fillna(4))
    c = c.sort_values(["_p", "ev", "prob"], ascending=[True, False, False])
    return c.iloc[0].drop(labels="_p").to_dict()


def day_table(analysis: pd.DataFrame, now: datetime | None = None) -> pd.DataFrame:
    """Tabla interactiva de la jornada (una fila por partido) ya lista para mostrar: sin NaN ni None."""
    from .config import SPORTS, book_name, league_name
    cols = ["Hora", "Deporte", "Competición", "Partido", "Selección", "Nivel", "Prob. real", "Cuota", "Casa", "Valor",
            "Confianza", "Recomendación", "_tier", "_sport", "_kickoff"]
    if analysis is None or analysis.empty:
        return pd.DataFrame(columns=cols)
    names = {"valor": "💎 Élite", "cuota_alta": "🔥 Cuota alta", "solida": "🛡️ Sólida", "moderado": "⚖️ Moderado",
             "verificar": "🎯 Por confirmar", "revisar": "⚠️ Revisar", "pronostico": "· Sin ventaja",
             "modelo": "· Sin cuotas", "sin_datos": "· Sin datos"}
    tz = __import__("zoneinfo").ZoneInfo("Europe/Madrid")
    rows = []
    for r in analysis.to_dict("records"):
        if r.get("tier") in ("sin_datos", None) or (isinstance(r.get("p_final"), float) and np.isnan(r["p_final"])) \
                or r.get("p_final") is None:
            continue  # sin cuotas ni modelo todavía: no hay nada que mostrar en la tabla
        def g(k):
            v = r.get(k)
            return None if v is None or (isinstance(v, float) and np.isnan(v)) else v
        src, price = g("price_src"), g("price")
        if price is None and g("b365_odds") is not None:
            price, src = g("b365_odds"), "bet365"
        operable = src in ("bet365", "mercado")
        casa = "bet365" if src == "bet365" else book_name(g("ref_book")) if src == "mercado" else "—"
        p = g("p_final")
        ev = (p * price - 1) if (p is not None and price and operable) else None
        tier = g("tier") or "sin_datos"
        conf = confidence_score(p, ev, g("n_books"), g("n_sharp"), operable, g("b365_age_h")) if p is not None else 0
        ko = pd.Timestamp(r["kickoff"]).tz_convert(tz)
        sp = SPORTS.get(r["sport"], SPORTS["football"])
        rows.append({"Hora": ko.strftime("%a %d/%m %H:%M").replace("Mon", "lun").replace("Tue", "mar")
                     .replace("Wed", "mié").replace("Thu", "jue").replace("Fri", "vie").replace("Sat", "sáb")
                     .replace("Sun", "dom"),
                     "Deporte": f'{sp["icono"]} {sp["nombre"]}', "Competición": league_name(r["league"]),
                     "Partido": f'{r["home"]} – {r["away"]}', "Selección": g("label") or "Sin selección todavía",
                     "Nivel": names.get(tier, tier), "Prob. real": round(p * 100, 1) if p is not None else 0.0,
                     "Cuota": f"{float(price):.2f}".replace(".", ",") if price and operable else "—", "Casa": casa,
                     "Valor": f"{ev * 100:+.1f} %".replace(".", ",") if ev is not None else "—", "Confianza": conf,
                     "Recomendación": g("recommendation") or "Sin cuotas todavía: se analizará en la próxima ejecución",
                     "_tier": tier, "_sport": r["sport"], "_kickoff": r["kickoff"]})
    return pd.DataFrame(rows, columns=cols)
