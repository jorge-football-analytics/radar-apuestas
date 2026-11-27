"""Orquestación: datos → modelos → cuotas → bajas → evaluación con bet365 → verificación → combinadas → avisos → aprender."""
from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd

from . import notify
from .backtest import summarize, tennis_backtest, walk_forward
from .combos import build_combos, legs_json
from .model.markets import label as market_label
from .config import CUPS, FOOTBALL, OTHER, Settings, book_name, sport_of
from .db import DB
from .learn import metrics, recalibrate, settle
from .model.dixon_coles import DCModel, fit, score_matrix
from .model.elo import TeamElo, TennisElo, surface_of
from .model.markets import SHARP, market_probs
from .sources import espn
from .sources import football_data as fd
from .sources import tennis_data
from .sources.injuries import absence_factor
from .sources.injuries import collect as collect_injuries
from .sources.odds_api import OddsAPI
from .sources.oddspapi import OddsPapi
from . import tables
from .stats import match_stats
from .teams import match_player, match_team, match_us, norm, player_key
from .value import evaluate_match, model_only

log = logging.getLogger("radar")
TZ = ZoneInfo("Europe/Madrid")
HORIZON_H = 96  # partidos de los próximos 4 días


@dataclass
class Ctx:
    fb_hist: pd.DataFrame = field(default_factory=pd.DataFrame)
    fb_models: dict = field(default_factory=dict)
    fb_known: dict = field(default_factory=dict)
    tennis_hist: pd.DataFrame = field(default_factory=pd.DataFrame)
    tennis: TennisElo | None = None
    tennis_keys: dict = field(default_factory=dict)
    us_hist: dict = field(default_factory=dict)
    us_elo: dict = field(default_factory=dict)
    diag: dict = field(default_factory=dict)  # estado de cada fuente para la web
    injury_status: dict = field(default_factory=dict)  # liga → (fuente de bajas o None, nota)


# ------------------------------------------------------------------ datos y modelos
def load_context(s: Settings, db: DB, now: datetime) -> Ctx:
    ctx = Ctx()
    act = s.active_sports
    if "football" in act:
        ctx.fb_hist = fd.load_history(now.date(), s.seasons_history)
        log.info("Fútbol: %d partidos históricos", len(ctx.fb_hist))
        if not ctx.fb_hist.empty:
            today = now.date().isoformat()
            for league, h in ctx.fb_hist.groupby("league"):
                cached = db.get(f"model_{league}")
                if cached and cached.get("fitted_at", "")[:10] == today:
                    ctx.fb_models[league] = DCModel.from_dict(cached)
                    continue
                m = fit(h, pd.Timestamp(now).tz_convert(None), xi=s.time_decay_xi)
                if m:
                    db.set(f"model_{league}", m.to_dict())
                    ctx.fb_models[league] = m
            recent = ctx.fb_hist[ctx.fb_hist["date"] >= pd.Timestamp(now).tz_convert(None) - pd.Timedelta(days=500)]
            ctx.fb_known = {lg: sorted(set(g["home"]) | set(g["away"])) for lg, g in recent.groupby("league")}
        counts = ctx.fb_hist.groupby("league").size().to_dict() if not ctx.fb_hist.empty else {}
        ctx.diag["football-data (histórico)"] = dict(ok=bool(counts), filas=int(sum(counts.values())),
                                                     detalle=f"{len(counts)} ligas con histórico")
    if "tennis" in act:
        ctx.tennis_hist = tennis_data.load(now.date())
        if not ctx.tennis_hist.empty:
            ctx.tennis = TennisElo()
            ctx.tennis.fit(ctx.tennis_hist)
            ctx.tennis_keys = dict(ctx.tennis.names)
            log.info("Tenis: %d partidos, %d jugadores", len(ctx.tennis_hist), len(ctx.tennis_keys))
        st = dict(tennis_data.STATUS)
        ctx.diag["Tenis (histórico)"] = dict(ok=not ctx.tennis_hist.empty, filas=len(ctx.tennis_hist),
                                             detalle=", ".join(f"{k}: {v}" for k, v in st.items()) or "sin datos")
    for league, v in OTHER.items():
        if v["sport"] in act and v.get("espn"):
            h = espn.load(league, v["espn"], now.date())
            ctx.us_hist[league] = h
            if not h.empty:
                ctx.us_elo[league] = TeamElo(league).fit(h)
                log.info("%s: %d partidos", league, len(h))
            ctx.diag[f"ESPN {v['nombre']} (histórico)"] = dict(ok=not h.empty, filas=len(h), detalle="")
    return ctx


def results_by_sport(ctx: Ctx) -> dict[str, pd.DataFrame]:
    out = {"football": ctx.fb_hist}
    if not ctx.tennis_hist.empty:
        t = ctx.tennis_hist
        out["tennis"] = pd.DataFrame({"date": t["date"], "home": t["winner"], "away": t["loser"], "hg": 1, "ag": 0,
                                      "completed": t["completed"]})
    for league, h in ctx.us_hist.items():
        sp = OTHER[league]["sport"]
        out[sp] = pd.concat([out[sp], h]) if sp in out else h
    return out


def run_backtest(s: Settings, db: DB, ctx: Ctx) -> dict:
    res, pesos = {}, {}
    if not ctx.fb_hist.empty:
        res = summarize(walk_forward(ctx.fb_hist, s), s)
        if res:
            db.set("backtest", res)
            pesos.update(res["pesos"])
    if not ctx.tennis_hist.empty:
        rt = tennis_backtest(ctx.tennis_hist, s)
        if rt:
            db.set("backtest_tennis", rt)
            pesos.update(rt["pesos"])
    if pesos:
        db.set("weights_backtest", pesos)
    return {"futbol": res, "tenis": db.get("backtest_tennis")}


# ------------------------------------------------------------------ cuotas
def canonicalize(odds: pd.DataFrame, ctx: Ctx, db: DB) -> pd.DataFrame:
    """Pone a todos los equipos/jugadores el nombre de la fuente histórica y crea el id del partido."""
    if odds.empty:
        return odds
    odds = odds.copy()
    if "sport" not in odds:
        odds["sport"] = odds["league"].map(sport_of)
    cache: dict = {}

    def canon(sport, league, raw, src):
        k = (league, raw, src)
        if k in cache:
            return cache[k]
        c = None
        if sport == "football" and src != "football-data" and ctx.fb_known.get(league):
            c = match_team(raw, ctx.fb_known[league])
        elif sport == "tennis" and ctx.tennis_keys:
            c = match_player(raw, ctx.tennis_keys)
        elif sport in ("basketball", "nfl") and league in ctx.us_hist and not ctx.us_hist[league].empty:
            known = sorted(set(ctx.us_hist[league]["home"]) | set(ctx.us_hist[league]["away"]))
            c = match_us(raw, known)
        c = c or raw
        if c != raw:
            db.execute("INSERT OR REPLACE INTO team_map VALUES (?,?,?,?)", (src, league, raw, c))
        cache[k] = c
        return c

    for col in ("home", "away"):
        odds[col] = [canon(sp, lg, r, src) for sp, lg, r, src in zip(odds["sport"], odds["league"], odds[col],
                                                                     odds["source"])]
    ko = pd.to_datetime(odds["kickoff"], utc=True, format="ISO8601")
    odds["kickoff"] = ko.map(lambda t: t.isoformat())
    # Tenis: el orden de los jugadores cambia según la fuente → orden alfabético y se invierten las selecciones
    ten = (odds["sport"] == "tennis") & (odds["home"] > odds["away"])
    if ten.any():
        odds.loc[ten, ["home", "away"]] = odds.loc[ten, ["away", "home"]].to_numpy()
        if "selection" in odds:
            odds.loc[ten, "selection"] = odds.loc[ten, "selection"].map({"H": "A", "A": "H"}).fillna(
                odds.loc[ten, "selection"])
    league_part = odds["league"].where(odds["sport"] != "tennis", "tennis")
    odds["match_id"] = league_part + "|" + ko.dt.strftime("%Y-%m-%d") + "|" + odds["home"] + "|" + odds["away"]
    return odds


def store_fixtures(db: DB, fx: pd.DataFrame) -> None:
    """Partidos sin cuotas todavía (ESPN): se guardan para poder pronosticarlos con el modelo."""
    if fx.empty:
        return
    with db.tx() as c:
        for m in fx.drop_duplicates("match_id").itertuples():
            if str(m.home).upper() in ("TBD", "TBA", "") or str(m.away).upper() in ("TBD", "TBA", ""):
                continue  # cuadro de tenis aún sin rival
            c.execute("INSERT OR IGNORE INTO matches (id, league, kickoff, home, away, sport) VALUES (?,?,?,?,?,?)",
                      (m.match_id, m.league, m.kickoff, m.home, m.away, m.sport))


def save_tables(db: DB, ctx: Ctx) -> None:
    """Clasificaciones y rankings para la web."""
    for league, h in (ctx.fb_hist.groupby("league") if not ctx.fb_hist.empty else []):
        db.set(f"tabla_{league}", tables.football_table(h, ctx.fb_models.get(league)))
    for league, h in ctx.us_hist.items():
        if league in ctx.us_elo:
            db.set(f"tabla_{league}", tables.team_elo_table(h, ctx.us_elo[league]))
    if ctx.tennis is not None:
        for tour in ("atp", "wta"):
            db.set(f"tabla_tennis_{tour}", tables.tennis_ranking(ctx.tennis_hist, ctx.tennis, tour))


ODDS_COLS = "match_id, bookmaker, market, selection, point, price, first_price, first_seen, updated, source, fetched_at"


def store_odds(db: DB, odds: pd.DataFrame, now: datetime, purge: bool = True) -> None:
    """Guarda cuotas. fetched_at = cuándo publicó la fuente esa cuota (no cuándo la miramos nosotros).
    purge: por cada partido que trae una fuente, se borran las filas de ESA fuente que ya no ofrece (casas o líneas
    que desaparecen), para que nunca se mezclen cuotas de hace días con las actuales."""
    if odds.empty:
        return
    ts = now.isoformat()
    if "fetched_at" not in odds:
        odds = odds.assign(fetched_at=ts)
    odds = odds.assign(fetched_at=odds["fetched_at"].fillna(ts))
    fresher = {"oddspapi": 3, "the-odds-api": 2, "football-data": 1}
    with db.tx() as c:
        for m in odds.drop_duplicates("match_id").itertuples():
            c.execute("INSERT OR IGNORE INTO matches (id, league, kickoff, home, away, sport) VALUES (?,?,?,?,?,?)",
                      (m.match_id, m.league, m.kickoff, m.home, m.away, m.sport))
        for o in odds.itertuples():
            key = (o.match_id, o.bookmaker, o.market, o.selection, o.point)
            prev = c.execute("SELECT source, price, fetched_at FROM odds WHERE match_id=? AND bookmaker=? AND market=? "
                             "AND selection=? AND point=?", key).fetchone()
            if prev is None:
                c.execute(f"INSERT INTO odds ({ODDS_COLS}) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                          (*key, o.price, o.price, ts, ts, o.source, o.fetched_at))
                continue
            old_t = prev[2] or ""
            if fresher.get(o.source, 0) >= fresher.get(prev[0], 0) or old_t < (now - timedelta(hours=12)).isoformat():
                changed = prev[1] != o.price or prev[0] != o.source or (o.fetched_at or "") > old_t
                if changed:
                    c.execute("UPDATE odds SET price=?, updated=?, source=?, fetched_at=? WHERE match_id=? AND "
                              "bookmaker=? AND market=? AND selection=? AND point=?",
                              (o.price, ts, o.source, o.fetched_at, *key))
                else:
                    c.execute("UPDATE odds SET updated=? WHERE match_id=? AND bookmaker=? AND market=? AND "
                              "selection=? AND point=?", (ts, *key))
        if purge:
            for (mid, src), g in odds.groupby(["match_id", "source"]):
                keep = {(b, mk, sel, float(pt)) for b, mk, sel, pt in
                        zip(g["bookmaker"], g["market"], g["selection"], g["point"])}
                for r in c.execute("SELECT rowid, bookmaker, market, selection, point FROM odds WHERE match_id=? "
                                   "AND source=?", (mid, src)).fetchall():
                    if (r[1], r[2], r[3], float(r[4])) not in keep:
                        c.execute("DELETE FROM odds WHERE rowid=?", (r[0],))


def football_data_timestamp(db: DB, f: pd.DataFrame, now: datetime) -> str:
    """football-data publica martes y viernes: la cuota tiene la hora en que CAMBIÓ el fichero, no la de cada lectura."""
    if f.empty:
        return now.isoformat()
    cols = ["league", "kickoff", "home", "away", "bookmaker", "market", "selection", "point", "price"]
    sig = str(pd.util.hash_pandas_object(f[cols].sort_values(cols).reset_index(drop=True), index=False).sum())
    if db.get("fd_fixtures_sig") != sig or not db.get("fd_fixtures_ts"):
        db.set("fd_fixtures_sig", sig)
        db.set("fd_fixtures_ts", now.isoformat())
    return db.get("fd_fixtures_ts")


def _absences(s: Settings, db: DB, upcoming: pd.DataFrame, ctx: Ctx, now: datetime):
    """Bajas de los partidos próximos (API-Football por fecha, ESPN en NBA/NFL, CSV manual) y estado por liga."""
    ms = [dict(m, sport=m.get("sport") or sport_of(m["league"])) for m in upcoming.to_dict("records")]
    ms = [m for m in ms if pd.Timestamp(m["kickoff"]) < pd.Timestamp(now) + pd.Timedelta(hours=HORIZON_H)]
    absences, status = collect_injuries(s, db, ms, ctx.fb_known, now.date())
    ok = sorted({src for src, _ in status.values() if src})
    notes = sorted({n for src, n in status.values() if not src and "tenis" not in n})
    ctx.diag["Bajas y lesiones"] = dict(ok=bool(ok), filas=sum(len(v) for v in absences.values()),
                                        detalle=("fuentes: " + ", ".join(ok) if ok else "") +
                                        ("; " if ok and notes else "") + "; ".join(notes))
    ctx.injury_status = status
    return absences


def model_for(m: dict, o: pd.DataFrame, ctx: Ctx, absences: dict, s: Settings, now: datetime):
    sport, league = m["sport"], m["league"]
    notes: list[str] = []
    if sport == "football":
        model = ctx.fb_models.get(league)
        if model is None:
            return None, notes, {}
        r = model.rates(m["home"], m["away"])
        for t in (m["home"], m["away"]):
            seen = model.last_seen.get(t)
            if seen and (pd.Timestamp(now).tz_convert(None) - pd.Timestamp(seen)).days > 200:
                return None, notes, {}  # recién ascendido: sin datos recientes
        if not r:
            return None, notes, {}
        lam, mu = r
        fh = absence_factor(absences.get((league, m["home"]), []), s.injury_impact, s.injury_cap)
        fa = absence_factor(absences.get((league, m["away"]), []), s.injury_impact, s.injury_cap)
        lam, mu = lam * fh / fa ** 0.5, mu * fa / fh ** 0.5
        for team, f in ((m["home"], fh), (m["away"], fa)):
            if f < 1:
                names = ", ".join(str(a["jugador"]) for a in absences[(league, team)][:5])
                notes.append(f"Bajas {team}: {names} (−{1 - f:.0%} goles esperados)")
        points = sorted(set(o.loc[o["market"] == "ou", "point"])) or [1.5, 2.5, 3.5]
        notes.append(f"Goles esperados {lam:.2f} - {mu:.2f}")
        mat = score_matrix(lam, mu, model.rho)
        top = sorted(((mat[i, j], f"{i}-{j}") for i in range(6) for j in range(6)), reverse=True)[:5]
        info = dict(tipo="goles", xg_local=lam, xg_visit=mu, marcadores=[[sc, float(p)] for p, sc in top],
                    bajas_local=[a["jugador"] for a in absences.get((league, m["home"]), [])],
                    bajas_visit=[a["jugador"] for a in absences.get((league, m["away"]), [])])
        probs = market_probs(mat, points)
        probs.update({k: v for k, v in market_probs(mat, [1.5, 2.5, 3.5]).items() if k not in probs})
        return probs, notes, info
    if sport == "tennis" and ctx.tennis:
        surface = surface_of(league)
        res = ctx.tennis.predict(m["home"], m["away"], surface)
        if not res:
            return None, notes, {}
        probs, info = res
        sup = {"Hard": "pista dura", "Clay": "tierra", "Grass": "hierba"}[surface]
        notes.append(f"Elo {info['elo_home']:.0f} vs {info['elo_away']:.0f} · superficie: {sup} · "
                     f"{info['partidos'][0]} y {info['partidos'][1]} partidos en la base")
        h2h = ctx.tennis_hist[((ctx.tennis_hist["winner"] == m["home"]) & (ctx.tennis_hist["loser"] == m["away"])) |
                              ((ctx.tennis_hist["winner"] == m["away"]) & (ctx.tennis_hist["loser"] == m["home"]))]
        info = dict(tipo="tenis", elo_local=info["elo_home"], elo_visit=info["elo_away"], superficie=sup,
                    elo_sup_local=ctx.tennis.surf[(m["home"], surface)], elo_sup_visit=ctx.tennis.surf[(m["away"], surface)],
                    h2h_local=int((h2h["winner"] == m["home"]).sum()), h2h_visit=int((h2h["winner"] == m["away"]).sum()))
        return probs, notes, info
    if sport in ("basketball", "nfl") and league in ctx.us_elo:
        lines = {"spread": sorted(set(o.loc[o["market"] == "spread", "point"])),
                 "ou": sorted(set(o.loc[o["market"] == "ou", "point"]))}
        res = ctx.us_elo[league].predict(m["home"], m["away"], lines)
        if not res:
            return None, notes, {}
        probs, info = res
        notes.append(f"Elo {info['elo_home']:.0f} vs {info['elo_away']:.0f} · margen esperado {info['margen']:+.1f} · "
                     f"total esperado {info['total']:.0f}")
        info = dict(tipo="puntos", elo_local=info["elo_home"], elo_visit=info["elo_away"], margen=info["margen"],
                    total=info["total"])
        return probs, notes, info
    return None, notes, {}


def analyze(s: Settings, db: DB, ctx: Ctx, now: datetime, absences: dict, weights: dict):
    upcoming = db.df("SELECT * FROM matches WHERE status='pendiente' AND kickoff > ? AND kickoff < ?",
                     (now.isoformat(), (now + timedelta(hours=HORIZON_H)).isoformat()))
    all_odds = db.df("SELECT * FROM odds WHERE match_id IN (SELECT id FROM matches WHERE status='pendiente' "
                     "AND kickoff > ?)", (now.isoformat(),))
    candidates, preds, matches = [], [], {}
    infos = {}
    results = results_by_sport(ctx)
    goal_avg = league_goal_avgs(ctx.fb_hist, now)
    ratios = b365_ratios(db, now)
    for m in upcoming.to_dict("records"):
        m["sport"] = m.get("sport") or sport_of(m["league"])
        if m["sport"] not in s.active_sports:
            continue
        matches[m["id"]] = m
        o = all_odds[all_odds["match_id"] == m["id"]]
        probs, notes, info = model_for(m, o, ctx, absences, s, now)
        st = match_stats(results.get(m["sport"]), m["home"], m["away"], now, draws=m["sport"] == "football")
        if st:
            info = {**(info or {}), "stats": st}
        src, note = ctx.injury_status.get(m["league"], (None, "sin fuente de bajas para esta competición"))
        info = {**(info or {}), "bajas": {"local": absences.get((m["league"], m["home"]), []),
                                          "visit": absences.get((m["league"], m["away"]), []),
                                          "fuente": src, "nota": note}}
        if info:
            infos[m["id"]] = info
        if o.empty:
            if probs:
                cs, ps = model_only(m, probs, s, notes)
                candidates += cs
                preds += ps
            continue
        b365 = o[o["bookmaker"] == s.book]
        if not b365.empty and (b365["source"] == "football-data").all():
            notes.append("Cuota de bet365 tomada el martes/viernes: confírmala antes de apostar")
        context = {"goal_avg": goal_avg.get(m["league"]), "model_total": (info or {}).get("total"), "b365_ratio": ratios}
        cs, ps = evaluate_match(m, o, probs, weights, s, notes, now=now, context=context)
        candidates += cs
        preds += ps
    if infos:
        with db.tx() as c:
            for mid, info in infos.items():
                c.execute("INSERT OR REPLACE INTO match_info VALUES (?,?,?)", (mid, json.dumps(info, default=float),
                                                                            now.isoformat()))
    return candidates, pd.DataFrame(preds), matches, all_odds


def b365_ratios(db: DB, now: datetime) -> dict[str, float]:
    """Cuánto paga bet365 frente a la cuota mediana de las casas generalistas, por deporte (histórico de la base).
    Sirve para estimar «bet365 ~1,83» cuando solo hay cuota de referencia de otras casas. Caché diaria."""
    key = f"b365_ratio_{now:%Y-%m-%d}"
    cached = db.get(key)
    if cached:
        return cached
    from .value import OFFSHORE
    o = db.df("SELECT o.match_id, o.bookmaker, o.market, o.selection, o.point, o.price, m.sport FROM odds o "
              "JOIN matches m ON m.id = o.match_id")
    out = {}
    if not o.empty:
        k = ["match_id", "market", "selection", "point"]
        b = o[o["bookmaker"] == "bet365"]
        soft = o[~o["bookmaker"].isin({"bet365", *SHARP, *OFFSHORE})]
        med = soft.groupby(k)["price"].agg(["median", "count"]).reset_index()
        j = b.merge(med, on=k)
        j = j[j["count"] >= 3]
        for sp, g in j.groupby("sport"):
            if len(g) >= 30:
                out[sp] = float(min(1.05, max(0.95, (g["price"] / g["median"]).median())))
    db.set(key, out)
    return out


def league_goal_avgs(hist: pd.DataFrame, now: datetime, days: int = 365) -> dict[str, float]:
    """Goles por partido de cada liga en el último año (para los mercados derivados cuando no hay línea de goles)."""
    if hist is None or hist.empty:
        return {}
    h = hist[hist["date"] >= pd.Timestamp(now).tz_convert(None) - pd.Timedelta(days=days)]
    return {lg: float((g["hg"] + g["ag"]).mean()) for lg, g in h.groupby("league") if len(g) >= 30}


def verify_with_oddspapi(s: Settings, db: DB, ctx: Ctx, candidates: list, matches: dict, now: datetime) -> int:
    """Pide bet365 + Pinnacle (misma llamada, misma hora) solo para los mejores candidatos de las próximas horas."""
    op = OddsPapi(s, db)
    if not op.enabled:
        return 0
    limit = pd.Timestamp(now) + pd.Timedelta(hours=s.oddspapi_window_h)
    soon = [c for c in candidates if pd.Timestamp(c.kickoff) <= limit]
    # 1º lo que parece tener valor (incluido fútbol con la cuota de football-data vieja); 2º pronósticos sin bet365
    # de competiciones sin bet365 gratis
    todo = sorted([c for c in soon if c.tier == "verificar"], key=lambda c: c.score, reverse=True) + \
        sorted([c for c in soon if c.tier == "pronostico" and not c.verified
                and not FOOTBALL.get(c.league, {}).get("fd")], key=lambda c: c.ev, reverse=True)
    done, added = set(), 0
    for c in todo:
        if c.match_id in done:
            continue
        if op.allowance(now) < 1:
            break
        done.add(c.match_id)
        checked = db.get(f"op_checked_{c.match_id}")
        if checked and (now - datetime.fromisoformat(checked)).total_seconds() < 6 * 3600:
            continue
        rows = op.match_odds(matches[c.match_id], now)
        db.set(f"op_checked_{c.match_id}", now.isoformat())
        if rows:
            df = pd.DataFrame(rows)
            df["match_id"] = c.match_id
            store_odds(db, df, now)
            added += len(rows)
    return added


def save_picks(db: DB, candidates: list, ts: str, now: datetime) -> list[dict]:
    """Inserta/actualiza jugadas. Una jugada que se ha mostrado como 'valor' (o se avisó) queda CONGELADA: se guarda
    la cuota, probabilidad y stake de ese momento y ya no se toca ni se borra (el seguimiento mide lo que se
    recomendó, no la última cuota antes del partido). Su estado actual va a current_odds / current_ev / active."""
    seen = set()
    changed = []
    with db.tx() as c:
        for cd in candidates:
            r = cd.as_row()
            key = (r["match_id"], r["market"], r["selection"], r["point"])
            seen.add(key)
            ex = c.execute("SELECT id, alerted, locked, tier, ev FROM picks WHERE match_id=? AND market=? AND "
                           "selection=? AND point=?", key).fetchone()
            vals = dict(p_final=r["p_final"], p_model=r["p_model"], p_market=r["p_market"], odds=r["odds"],
                        bookmaker=r["bookmaker"], b365_odds=r["b365_odds"], ev=r["ev"], stake=r["stake"],
                        confidence=r["confidence"], reasons=r["reasons"], sport=r["sport"], risk=r["risk"],
                        tier=r["tier"], elite=r["elite"], verified=r["verified"], min_odds=r["min_odds"],
                        score=r["score"], label=r["label"], updated=ts, current_odds=r["odds"], current_ev=r["ev"],
                        active=1)
            lock = r["tier"] in TRACKED_TIERS  # lo que se recomienda queda registrado tal cual
            if ex is None:
                vals.update(locked=int(lock), shown_at=ts if lock else None)
                cols = ["created", "match_id", "league", "kickoff", "home", "away", "market", "selection", "point",
                        *vals]
                c.execute(f"INSERT INTO picks ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                          (ts, r["match_id"], r["league"], r["kickoff"], r["home"], r["away"], r["market"],
                           r["selection"], r["point"], *vals.values()))
                changed.append(r)
            elif not (ex["alerted"] or ex["locked"]):
                vals.update(locked=int(lock), shown_at=ts if lock else None)
                c.execute(f"UPDATE picks SET {', '.join(f'{k}=?' for k in vals)} WHERE id=?",
                          (*vals.values(), ex["id"]))
                if r["elite"]:
                    changed.append(r)
            else:  # congelada: solo su estado actual
                c.execute("UPDATE picks SET current_odds=?, current_ev=?, active=?, updated=? WHERE id=?",
                          (r["odds"], r["ev"], int(r["tier"] == ex["tier"]), ts, ex["id"]))
        for row in c.execute("SELECT id, match_id, market, selection, point, alerted, locked FROM picks "
                             "WHERE won IS NULL AND kickoff > ?", (now.isoformat(),)).fetchall():
            if (row[1], row[2], row[3], row[4]) in seen:
                continue
            if row[5] or row[6]:
                c.execute("UPDATE picks SET active=0, updated=? WHERE id=?", (ts, row[0]))  # se conserva
            else:
                c.execute("DELETE FROM picks WHERE id=?", (row[0],))
    return changed


TRACKED_TIERS = ("valor", "solida", "moderado", "cuota_alta")
TIER_ORDER = {"valor": 0, "cuota_alta": 1, "solida": 2, "moderado": 3, "verificar": 4, "revisar": 5, "pronostico": 6}
MAIN_MARKET = {"football": "1x2"}


def analysis_rows(preds: pd.DataFrame, matches: dict, now: datetime, s: Settings | None = None) -> list[dict]:
    """Una fila por partido: la selección destacada, su semáforo y una recomendación en una frase."""
    s = s or Settings()
    rows = []
    groups = dict(tuple(preds.groupby("match_id"))) if not preds.empty else {}
    for mid, m in matches.items():
        g = groups.get(mid)
        main = MAIN_MARKET.get(m["sport"], "ml")
        base = dict(match_id=mid, sport=m["sport"], league=m["league"], kickoff=m["kickoff"], home=m["home"],
                    away=m["away"], updated=now.isoformat(), p_home=None, p_draw=None, p_away=None, n_markets=0)
        if g is None or g.empty:
            rows.append({**base, "tier": "sin_datos", "light": "gris", "recommendation": "Sin cuotas ni modelo todavía"})
            continue
        mp = g[g["market"] == main].set_index("selection")["p_final"]
        base.update(p_home=mp.get("H"), p_draw=mp.get("D"), p_away=mp.get("A"),
                    n_markets=int(g[["market", "point"]].drop_duplicates().shape[0]))
        if "tier" not in g or g["tier"].isna().all():  # solo modelo (aún sin cuotas)
            r = g[g["market"] == main].sort_values("p_final", ascending=False).head(1)
            r = (r if not r.empty else g.sort_values("p_final", ascending=False).head(1)).iloc[0]
            lab = market_label(r["market"], r["selection"], r["point"], m["home"], m["away"],
                               "goles" if m["sport"] == "football" else "puntos")
            rows.append({**base, "market": r["market"], "selection": r["selection"], "point": r["point"], "label": lab,
                         "p_final": r["p_final"], "fair_odds": 1 / r["p_final"], "tier": "modelo", "light": "gris",
                         "min_odds": (1 + s.min_ev) / r["p_final"],
                         "recommendation": f"Aún sin cuotas: solo con bet365 a {(1 + s.min_ev) / r['p_final']:.2f} o más"
                         .replace(".", ",")})
            continue
        g = g.dropna(subset=["tier"]).assign(_o=lambda d: d["tier"].map(TIER_ORDER).fillna(9))
        top = g.sort_values("_o").iloc[0]["tier"]
        cand = g[g["tier"] == top]
        if top in ("valor", "cuota_alta", "moderado"):
            r = cand.sort_values("ev", ascending=False).iloc[0]
        elif top == "solida":
            r = cand.sort_values("p_final", ascending=False).iloc[0]
        elif top in ("verificar", "revisar"):
            r = cand.sort_values("ev", ascending=False).iloc[0]
        else:  # sin nada recomendable: lo más probable con cuota de bet365, si no lo más probable
            v = cand[cand["verified"] == 1]
            pool = v if not v.empty else cand
            pool_main = pool[pool["market"] == main]
            r = (pool_main if not pool_main.empty else pool).sort_values("p_final", ascending=False).iloc[0]
        ev, p, verified = float(r["ev"]), float(r["p_final"]), int(r["verified"])
        src = r.get("price_src") if isinstance(r.get("price_src"), str) else ("bet365" if verified else "estimada")
        rb = r.get("ref_book") if isinstance(r.get("ref_book"), str) else None
        who = "bet365" if src == "bet365" else f"la cuota de mercado ({book_name(rb)})" if src == "mercado" else "bet365"

        def fp(x: float, signed: bool = False) -> str:  # porcentaje con coma decimal
            return (f"{x:+.1%}" if signed else f"{x:.1%}").replace(".", ",")
        if top == "valor":
            light, rec = "verde", f"Élite: apostar · valor {fp(ev, True)}"
        elif top == "solida":
            light, rec = "verde", f"Sólida: {p:.0%} de probabilidad a precio justo · apta para combinadas"
        elif top == "cuota_alta":
            light, rec = "verde", f"Cuota alta: {who} paga un {fp(ev, True)} sobre el precio justo · riesgo controlado"
        elif top == "moderado":
            light, rec = "verde", f"Valor moderado: ventaja ligera de {fp(ev, True)} sobre el precio justo"
        elif top == "verificar":
            light, rec = "ambar", f"Por confirmar: apostar solo si bet365 paga {r['min_odds']:.2f} o más".replace(".", ",")
        elif top == "revisar":
            light, rec = "rojo", "Revisar: valor anómalo, probable error de datos. No apostar"
        elif src not in ("bet365", "mercado"):
            light, rec = "gris", "Sin cuota fiable todavía: solo informativo"
        elif ev >= -0.03:
            light, rec = "ambar", f"Sin ventaja: {who} cobra un margen de {fp(-ev)} sobre el precio justo"
        else:
            light, rec = "rojo", f"No apostar: {who} paga un {fp(-ev)} por debajo del precio justo"
        rows.append({**base, "market": r["market"], "selection": r["selection"], "point": r["point"],
                     "label": r["label"], "p_final": p, "fair_odds": 1 / p if p else None,
                     "b365_odds": r["price"] if src == "bet365" else None, "b365_age_h": r.get("b365_age_h"), "ev": ev,
                     "price": r["price"], "price_src": src, "ref_book": rb, "b365_est": r.get("b365_est"),
                     "tier": top, "light": light, "recommendation": rec, "n_books": int(r["n_books"]),
                     "n_sharp": int(r["n_sharp"]), "confidence": r["confidence"], "min_odds": r["min_odds"]})
    return rows


SEL_COLS = ["match_id", "market", "selection", "point", "label", "p_final", "p_market", "price", "verified", "ev", "tier",
            "derived", "weak", "n_books", "n_sharp", "b365_age_h", "min_odds", "updated", "price_src", "ref_book",
            "b365_est"]


def save_selections(db: DB, preds: pd.DataFrame, now: datetime) -> int:
    if preds.empty or "tier" not in preds:
        with db.tx() as c:
            c.execute("DELETE FROM selections")
        return 0
    rows = preds.dropna(subset=["tier"]).assign(updated=now.isoformat())
    with db.tx() as c:
        c.execute("DELETE FROM selections")
        for r in rows.to_dict("records"):
            vals = [r.get(k) for k in SEL_COLS]
            vals = [None if (isinstance(v, float) and pd.isna(v)) else v for v in vals]
            c.execute(f"INSERT OR REPLACE INTO selections ({','.join(SEL_COLS)}) VALUES ({','.join('?' * len(SEL_COLS))})",
                      vals)
    return len(rows)


ANALYSIS_COLS = ["match_id", "sport", "league", "kickoff", "home", "away", "market", "selection", "point", "label",
                 "p_final", "fair_odds", "b365_odds", "b365_age_h", "ev", "tier", "light", "recommendation", "n_books",
                 "n_sharp", "confidence", "min_odds", "p_home", "p_draw", "p_away", "n_markets", "updated", "price",
                 "price_src", "ref_book", "b365_est"]


def save_analysis(db: DB, preds: pd.DataFrame, matches: dict, now: datetime) -> int:
    rows = analysis_rows(preds, matches, now)
    with db.tx() as c:
        c.execute("DELETE FROM analysis")
        for r in rows:
            vals = [r.get(k) for k in ANALYSIS_COLS]
            vals = [None if (isinstance(v, float) and pd.isna(v)) else v for v in vals]
            c.execute(f"INSERT INTO analysis ({','.join(ANALYSIS_COLS)}) VALUES ({','.join('?' * len(ANALYSIS_COLS))})",
                      vals)
    return len(rows)


def send_alerts(s: Settings, db: DB, now: datetime, matches: dict, combos: list[dict]) -> int:
    sent = 0
    today = now.astimezone(TZ).strftime("%Y-%m-%d")
    n_today = db.get(f"alerts_{today}", 0)
    elite = db.df("SELECT * FROM picks WHERE elite=1 AND alerted=0 AND tier='valor' AND kickoff > ? "
                  "ORDER BY score DESC", (now.isoformat(),))
    used_matches = set()
    for p in elite.to_dict("records"):
        if n_today >= s.alert_max_day:
            break
        if p["match_id"] in used_matches:
            continue
        used_matches.add(p["match_id"])
        if notify.send(s, notify.format_pick(p)):
            sent += 1
        n_today += 1
        db.execute("UPDATE picks SET alerted=1 WHERE id=?", (p["id"],))
    combo_key = f"combo_alert_{today}"
    if not db.get(combo_key):
        best = [c for c in combos if c["tier"] == "valor" and c["risk"] != "arriesgada" and c["ev"] >= s.alert_combo_ev]
        if best:
            c = max(best, key=lambda x: x["ev"])
            if notify.send(s, notify.format_combo(c)):
                sent += 1
            db.execute("UPDATE combos SET alerted=1 WHERE signature=?", (c["signature"],))
            db.set(combo_key, True)
    db.set(f"alerts_{today}", n_today)
    local = now.astimezone(TZ)
    if s.daily_digest and local.hour >= 9 and db.get("digest_day") != today:
        day = db.df("SELECT * FROM picks WHERE tier='valor' AND kickoff > ? AND kickoff < ? ORDER BY score DESC",
                    (now.isoformat(), (now + timedelta(hours=24)).isoformat())).to_dict("records")
        counts = Counter(m["sport"] for m in matches.values()
                         if pd.Timestamp(m["kickoff"]) < pd.Timestamp(now) + pd.Timedelta(hours=24))
        notify.send(s, notify.format_digest(day, dict(counts), db.get("web_url", "") or ""))
        db.set("digest_day", today)
    week = now.strftime("%G-%V")
    if now.weekday() == 0 and db.get("summary_week") != week:
        msg = notify.format_summary(metrics(db))
        if msg:
            notify.send(s, msg)
        db.set("summary_week", week)
    return sent


# ------------------------------------------------------------------ ejecución completa
def run(s: Settings | None = None, now: datetime | None = None, send: bool = True, ctx: Ctx | None = None) -> dict:
    s = s or Settings()
    now = now or datetime.now(timezone.utc)
    db = DB(s.db_path)
    ts = now.isoformat()
    ctx = ctx or load_context(s, db, now)

    last_bt = db.get("backtest_date")
    if (not last_bt or (now.date() - datetime.fromisoformat(last_bt).date()).days > 30 or
            ("tennis" in s.active_sports and not db.get("backtest_tennis") and not ctx.tennis_hist.empty)):
        run_backtest(s, db, ctx)
        db.set("backtest_date", now.date().isoformat())

    # cachés de la versión anterior de OddsPapi (listas de miles de torneos): ya no se usan
    db.execute("DELETE FROM kv WHERE key LIKE 'op_tournaments_%' OR key LIKE 'op_fix_%' OR key LIKE 'op_checked_%'")

    # 1) Resultados y aprendizaje
    api = OddsAPI(s, db)
    scores = []
    # copas europeas: resultados gratis de ESPN (antes costaban 2 créditos de The Odds API por copa y día)
    if "football" in s.active_sports and not db.df(
            "SELECT 1 FROM matches WHERE league IN ('champions','europa','conference') AND status='pendiente' "
            "AND kickoff < ?", ((now - timedelta(hours=3)).isoformat(),)).empty:
        scores += espn.cup_results(now.date())
    # resto sin fuente gratuita (Euroliga): The Odds API
    for lg in [k for k, v in OTHER.items() if not v.get("espn")]:
        if not db.df("SELECT 1 FROM matches WHERE league=? AND status='pendiente' AND kickoff < ?",
                     (lg, (now - timedelta(hours=3)).isoformat())).empty:
            scores += api.fetch_scores(lg)
    n_settled = settle(db, results_by_sport(ctx), now, scores or None)
    weights = recalibrate(db, s)

    # 2) Cuotas: football-data (bet365 gratis en fútbol) + The Odds API (consenso)
    frames, fixtures = [], []
    if "football" in s.active_sports:
        f = fd.load_fixtures()
        if not f.empty:
            f = f.assign(fetched_at=football_data_timestamp(db, f, now))
        frames.append(f)
        ctx.diag["football-data (cuotas bet365)"] = dict(ok=not f.empty, filas=len(f), detalle=(
            f"{f['league'].nunique()} ligas, {f[['home', 'away']].drop_duplicates().shape[0]} partidos" if not f.empty
            else "sin partidos próximos publicados (se actualiza martes y viernes por la tarde)"))
    oa = api.fetch(now)
    frames.append(oa)
    ctx.diag["The Odds API (consenso)"] = dict(ok=api.enabled, filas=len(oa), detalle=(
        f"{db.get('odds_api_remaining')} créditos restantes" if api.enabled else "sin clave ODDS_API_KEY"))
    # ESPN: próximos partidos de NBA y NFL con hándicap y totales de referencia (gratis)
    for league, v in OTHER.items():
        if v["sport"] in s.active_sports and v.get("espn"):
            up = espn.upcoming(league, v["espn"], v["sport"], now)
            frames.append(up)
            fx = up.attrs.get("fixtures", [])
            fixtures += fx
            ctx.diag[f"ESPN {v['nombre']} (próximos)"] = dict(ok=bool(fx), filas=len(fx), detalle=f"{len(up)} cuotas")
    # ESPN: cuadro de los torneos de tenis en juego (partidos aún sin cuotas)
    if "tennis" in s.active_sports:
        n_t = 0
        for tour in ("atp", "wta"):
            for g in espn.tennis_scoreboard(tour):
                if g.get("state") != "pre":
                    continue
                slug = "_".join(norm(g.get("tournament") or "torneo").split())[:40]
                fixtures.append(dict(sport="tennis", league=f"tennis_{tour}_{slug}", kickoff=g["date"],
                                     home=g["p1"], away=g["p2"], source="espn"))
                n_t += 1
        ctx.diag["ESPN tenis (cuadros)"] = dict(ok=n_t > 0, filas=n_t, detalle="partidos programados")
    frames = [f for f in frames if f is not None and not f.empty]
    n_odds = 0
    if frames:
        odds = canonicalize(pd.concat(frames, ignore_index=True), ctx, db)
        odds = odds[pd.to_datetime(odds["kickoff"], utc=True, format="ISO8601") > now]
        store_odds(db, odds, now)
        n_odds = len(odds)
    if fixtures:
        fx = canonicalize(pd.DataFrame(fixtures), ctx, db)
        store_fixtures(db, fx[pd.to_datetime(fx["kickoff"], utc=True, format="ISO8601") > now])

    # 3) OddsPapi en bloque: bet365 (y Pinnacle si sobra presupuesto) de todos los partidos de las próximas horas
    upcoming = db.df("SELECT * FROM matches WHERE status='pendiente' AND kickoff > ?", (ts,))
    op = OddsPapi(s, db)
    op_rows = []
    if op.enabled and not upcoming.empty:
        ups = [m for m in upcoming.to_dict("records") if (m.get("sport") or sport_of(m["league"])) in s.active_sports]
        op_rows = op.fetch_bulk(ups, now)
        if op_rows:
            store_odds(db, pd.DataFrame(op_rows).dropna(subset=["match_id"]), now)
    ctx.diag["OddsPapi (bet365 en bloque)"] = dict(ok=op.enabled, filas=len(op_rows), detalle=(
        f"{len({(r['home'], r['away']) for r in op_rows})} partidos con cuota en esta ejecución · "
        f"{db.get(f'oddspapi_used_{now:%Y-%m}', 0)} de {s.oddspapi_monthly} consultas este mes" if op.enabled
        else "sin clave ODDSPAPI_KEY"))

    # 4) Análisis de todos los partidos y mercados
    absences = _absences(s, db, upcoming, ctx, now) if not upcoming.empty else {}
    candidates, preds, matches, all_odds = analyze(s, db, ctx, now, absences, weights)

    with db.tx() as c:
        for p in preds.to_dict("records"):
            c.execute("INSERT OR REPLACE INTO predictions VALUES (?,?,?,?,?,?,?,?,NULL)",
                      (p["match_id"], p["market"], p["selection"], p["point"], p["p_model"], p["p_market"],
                       p["p_final"], ts))
    save_picks(db, candidates, ts, now)
    save_tables(db, ctx)

    # 5) Combinadas por perfil (partidos de las próximas 36 h)
    soon = {k: v for k, v in matches.items() if pd.Timestamp(v["kickoff"]) < pd.Timestamp(now) + pd.Timedelta(hours=36)}
    infos = {r["match_id"]: json.loads(r["info"]) for r in db.conn.execute(
        "SELECT match_id, info FROM match_info WHERE updated=?", (ts,)).fetchall()}
    tables = {r["key"][6:]: json.loads(r["value"]) for r in db.conn.execute(
        "SELECT key, value FROM kv WHERE key LIKE 'tabla_%'").fetchall()}
    combos = build_combos(preds[preds["match_id"].isin(soon)] if not preds.empty else preds, soon, s, infos,
                          tables=tables) if soon else []
    save_analysis(db, preds, matches, now)
    save_selections(db, preds, now)
    with db.tx() as c:
        c.execute("DELETE FROM combos WHERE alerted=0 AND won IS NULL AND first_kickoff > ?", (ts,))
        for cb in combos:
            c.execute("INSERT OR IGNORE INTO combos (created, risk, bookmaker, legs, odds, prob, ev, stake, signature,"
                      " sports, first_kickoff, tier) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                      (ts, cb["risk"], cb["bookmaker"], legs_json(cb), cb["odds"], cb["prob"], cb["ev"], cb["stake"],
                       cb["signature"], cb["sports"], cb["first_kickoff"], cb["tier"]))

    sent = send_alerts(s, db, now, matches, combos) if send else 0
    tiers = Counter(c.tier for c in candidates)
    summary = dict(partidos=len(matches), por_deporte=dict(Counter(m["sport"] for m in matches.values())),
                   cuotas=n_odds, liquidados=n_settled, valor=tiers.get("valor", 0),
                   verificar=tiers.get("verificar", 0), pronosticos=tiers.get("pronostico", 0),
                   solidas=tiers.get("solida", 0), moderadas=tiers.get("moderado", 0),
                   cuota_alta=tiers.get("cuota_alta", 0), revisar=tiers.get("revisar", 0),
                   elite=sum(c.elite for c in candidates), combinadas=len(combos), avisos=sent, pesos=weights,
                   creditos_odds_api=db.get("odds_api_remaining"),
                   oddspapi_mes=db.get(f"oddspapi_used_{now:%Y-%m}", 0))
    # mantener la base de datos pequeña: cuotas de partidos terminados hace más de 30 días
    db.execute("DELETE FROM odds WHERE match_id IN (SELECT id FROM matches WHERE status != 'pendiente' AND kickoff < ?)",
               ((now - timedelta(days=30)).isoformat(),))
    db.set("fuentes", {"ts": ts, **ctx.diag})
    db.log(ts, "run", str(summary))
    db.set("last_run", ts)
    db.close()
    return summary
