"""Orquestación: descargar → ajustar modelos → cuotas → bajas → valor → combinadas → avisos → aprender."""
from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import pandas as pd

from . import notify
from .backtest import summarize, walk_forward
from .combos import build_combos, legs_json
from .config import LEAGUES, Settings
from .db import DB
from .learn import metrics, recalibrate, settle
from .model.dixon_coles import DCModel, fit, score_matrix
from .model.markets import market_probs
from .sources import football_data as fd
from .sources.injuries import absence_factor, fetch_api_football, load_manual
from .sources.odds_api import OddsAPI
from .teams import match_team

log = logging.getLogger("radar")
HORIZON_H = 120  # analizamos partidos de los próximos 5 días


def _models(db: DB, hist: pd.DataFrame, s: Settings, now: datetime) -> dict[str, DCModel]:
    models = {}
    today = now.date().isoformat()
    for league, h in hist.groupby("league"):
        cached = db.get(f"model_{league}")
        if cached and cached.get("fitted_at", "")[:10] == today:
            models[league] = DCModel.from_dict(cached)
            continue
        m = fit(h, pd.Timestamp(now).tz_convert(None), xi=s.time_decay_xi)
        if m:
            db.set(f"model_{league}", m.to_dict())
            models[league] = m
            log.info("Modelo %s ajustado con %d partidos", league, m.n_matches)
    return models


def _known_teams(hist: pd.DataFrame, now: datetime) -> dict[str, list[str]]:
    recent = hist[hist["date"] >= pd.Timestamp(now).tz_convert(None) - pd.Timedelta(days=500)]
    return {lg: sorted(set(g["home"]) | set(g["away"])) for lg, g in recent.groupby("league")}


def canonicalize(odds: pd.DataFrame, known: dict[str, list[str]], db: DB) -> pd.DataFrame:
    """Pone a todos los equipos el nombre de football-data y crea el id del partido."""
    if odds.empty:
        return odds
    odds = odds.copy()
    cache: dict = {}
    for col in ("home", "away"):
        vals = []
        for lg, raw, src in zip(odds["league"], odds[col], odds["source"]):
            key = (lg, raw)
            if key not in cache:
                canon = raw
                if src != "football-data" and known.get(lg):
                    canon = match_team(raw, known[lg]) or raw
                    db.execute("INSERT OR REPLACE INTO team_map VALUES (?,?,?,?)", (src, lg, raw, canon))
                cache[key] = canon
            vals.append(cache[key])
        odds[col] = vals
    ko = pd.to_datetime(odds["kickoff"], utc=True)
    odds["kickoff"] = ko.map(lambda t: t.isoformat())
    odds["match_id"] = odds["league"] + "|" + ko.dt.strftime("%Y-%m-%d") + "|" + odds["home"] + "|" + odds["away"]
    return odds


def store_odds(db: DB, odds: pd.DataFrame, now: datetime) -> None:
    ts = now.isoformat()
    with db.tx() as c:
        for m in odds.drop_duplicates("match_id").itertuples():
            c.execute("INSERT OR IGNORE INTO matches (id, league, kickoff, home, away) VALUES (?,?,?,?,?)",
                      (m.match_id, m.league, m.kickoff, m.home, m.away))
        for o in odds.itertuples():
            prev = c.execute("SELECT source FROM odds WHERE match_id=? AND bookmaker=? AND market=? AND selection=? AND point=?",
                             (o.match_id, o.bookmaker, o.market, o.selection, o.point)).fetchone()
            if prev is None:
                c.execute("INSERT INTO odds VALUES (?,?,?,?,?,?,?,?,?,?)",
                          (o.match_id, o.bookmaker, o.market, o.selection, o.point, o.price, o.price, ts, ts, o.source))
            elif not (prev[0] == "the-odds-api" and o.source == "football-data"):  # la API es más fresca
                c.execute("UPDATE odds SET price=?, updated=?, source=? WHERE match_id=? AND bookmaker=? AND market=? "
                          "AND selection=? AND point=?",
                          (o.price, ts, o.source, o.match_id, o.bookmaker, o.market, o.selection, o.point))


def _absences(s: Settings, db: DB, upcoming: pd.DataFrame, known: dict, now: datetime) -> dict:
    out: dict = defaultdict(list)
    manual = load_manual(now.date())
    for (lg, team), lst in manual.items():
        out[(lg, team)] += lst
    for lg, g in upcoming.groupby("league"):
        dates = {k[:10] for k in g["kickoff"]}
        for team, lst in fetch_api_football(s, db, lg, dates, known.get(lg, []), now.date()).items():
            out[(lg, team)] += lst
    return out


def analyze(s: Settings, db: DB, hist: pd.DataFrame, models: dict, now: datetime, absences: dict,
            weights: dict) -> tuple[list, pd.DataFrame, dict, pd.DataFrame]:
    upcoming = db.df("SELECT * FROM matches WHERE status='pendiente' AND kickoff > ? AND kickoff < ?",
                     (now.isoformat(), (now + timedelta(hours=HORIZON_H)).isoformat()))
    all_odds = db.df("SELECT * FROM odds WHERE match_id IN (SELECT id FROM matches WHERE status='pendiente' AND kickoff > ?)",
                     (now.isoformat(),))
    from .value import evaluate_match

    candidates, preds = [], []
    matches = {}
    for m in upcoming.to_dict("records"):
        matches[m["id"]] = m
        o = all_odds[all_odds["match_id"] == m["id"]]
        if o.empty:
            continue
        model = models.get(m["league"])
        probs, notes = None, []
        if model is not None:
            r = model.rates(m["home"], m["away"])
            stale = False
            for t in (m["home"], m["away"]):
                seen = model.last_seen.get(t)
                if seen and (pd.Timestamp(now).tz_convert(None) - pd.Timestamp(seen)).days > 200:
                    stale = True  # recién ascendido: sus datos son de hace mucho
            if r and not stale:
                lam, mu = r
                fh = absence_factor(absences.get((m["league"], m["home"]), []), s.injury_impact, s.injury_cap)
                fa = absence_factor(absences.get((m["league"], m["away"]), []), s.injury_impact, s.injury_cap)
                lam, mu = lam * fh / fa ** 0.5, mu * fa / fh ** 0.5
                for team, f in ((m["home"], fh), (m["away"], fa)):
                    if f < 1:
                        names = ", ".join(a["jugador"] for a in absences[(m["league"], team)][:5])
                        notes.append(f"Bajas {team}: {names} (−{1 - f:.0%} goles esperados)")
                points = sorted(set(o.loc[o["market"] == "ou", "point"])) or [2.5]
                probs = market_probs(score_matrix(lam, mu, model.rho), points)
                notes.append(f"Goles esperados {lam:.2f} - {mu:.2f}")
        if (o["source"] == "football-data").all():
            notes.append("Cuotas de football-data (pueden tener 1-2 días): comprueba la cuota actual antes de apostar")
        cs, ps = evaluate_match(m, o, probs, weights, s, notes)
        candidates += cs
        preds += ps
    return candidates, pd.DataFrame(preds), matches, all_odds


def run(s: Settings | None = None, now: datetime | None = None, send_alerts: bool = True) -> dict:
    s = s or Settings()
    now = now or datetime.now(timezone.utc)
    db = DB(s.db_path)
    ts = now.isoformat()

    hist = fd.load_history(now.date(), s.seasons_history)
    log.info("Histórico: %d partidos", len(hist))
    models = _models(db, hist, s, now) if not hist.empty else {}
    known = _known_teams(hist, now) if not hist.empty else {}

    # Si nunca se ha hecho backtest (o hace más de 30 días), hacerlo para fijar pesos iniciales
    last_bt = db.get("backtest_date")
    if not hist.empty and (not last_bt or (now.date() - datetime.fromisoformat(last_bt).date()).days > 30):
        run_backtest(s, db, hist)
        db.set("backtest_date", now.date().isoformat())

    # 1) Resultados y aprendizaje
        # Competiciones europeas (sin histórico en football-data): resultados vía The Odds API (2 créditos/liga)
    ucl_scores = []
    for lg in [k for k, v in LEAGUES.items() if not v["fd"]]:
        if not db.df("SELECT 1 FROM matches WHERE league=? AND status='pendiente' AND kickoff < ?",
                     (lg, (now - timedelta(hours=3)).isoformat())).empty:
            ucl_scores += OddsAPI(s, db).fetch_scores(lg)
    n_settled = settle(db, hist, now, ucl_scores or None)

    # 2) Cuotas
    frames = [fd.load_fixtures(), OddsAPI(s, db).fetch(now)]
    odds = pd.concat([f for f in frames if not f.empty], ignore_index=True) if any(not f.empty for f in frames) else pd.DataFrame()
    if not odds.empty:
        odds = canonicalize(odds, known, db)
        odds = odds[pd.to_datetime(odds["kickoff"], utc=True) > now]
        store_odds(db, odds, now)

    # 3) Bajas y análisis
    upcoming = db.df("SELECT * FROM matches WHERE status='pendiente' AND kickoff > ?", (ts,))
    absences = _absences(s, db, upcoming, known, now) if not upcoming.empty else {}
    candidates, preds, matches, all_odds = analyze(s, db, hist, models, now, absences, weights)

    with db.tx() as c:
        for p in preds.to_dict("records"):
            c.execute("INSERT OR REPLACE INTO predictions VALUES (?,?,?,?,?,?,?,?,NULL)",
                      (p["match_id"], p["market"], p["selection"], p["point"], p["p_model"], p["p_market"],
                       p["p_final"], ts))
        new_picks = []
        for cd in candidates:
            r = cd.as_row()
            exists = c.execute("SELECT id, ev, alerted FROM picks WHERE match_id=? AND market=? AND selection=? AND point=?",
                               (r["match_id"], r["market"], r["selection"], r["point"])).fetchone()
            vals = (r["p_final"], r["p_model"], r["p_market"], r["odds"], r["bookmaker"], r["b365_odds"], r["ev"],
                    r["stake"], r["confidence"], r["reasons"])
            if exists is None:
                c.execute("INSERT INTO picks (created, match_id, league, kickoff, home, away, market, selection, point, label,"
                          " p_final, p_model, p_market, odds, bookmaker, b365_odds, ev, stake, confidence, reasons)"
                          " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                          (ts, r["match_id"], r["league"], r["kickoff"], r["home"], r["away"], r["market"],
                           r["selection"], r["point"], r["label"], *vals))
                new_picks.append(r)
            elif not exists["alerted"] and r["ev"] > exists["ev"]:
                c.execute("UPDATE picks SET p_final=?, p_model=?, p_market=?, odds=?, bookmaker=?, b365_odds=?, ev=?,"
                          " stake=?, confidence=?, reasons=? WHERE id=?", (*vals, exists["id"]))
                new_picks.append(r)

    # 4) Combinadas (con todas las selecciones y cuotas de partidos de las próximas 48 h)
    soon = {k: v for k, v in matches.items()
            if pd.Timestamp(v["kickoff"]) < pd.Timestamp(now) + pd.Timedelta(hours=48)}
    combos = []
    if soon and not preds.empty:
        combos = build_combos(preds[preds["match_id"].isin(soon)], all_odds[all_odds["match_id"].isin(soon)], soon, s)
    new_combos = []
    with db.tx() as c:
        for cb in combos:
            cur = c.execute("INSERT OR IGNORE INTO combos (created, risk, bookmaker, legs, odds, prob, ev, stake, signature)"
                            " VALUES (?,?,?,?,?,?,?,?,?)",
                            (ts, cb["risk"], cb["bookmaker"], legs_json(cb), cb["odds"], cb["prob"], cb["ev"],
                             cb["stake"], cb["signature"]))
            if cur.rowcount:
                new_combos.append(cb)

    # 5) Avisos
    sent = 0
    if send_alerts:
        to_alert = sorted([p for p in new_picks if p["ev"] >= s.alert_ev and p["confidence"] != "baja"],
                          key=lambda p: p["ev"], reverse=True)[:8]
        for p in to_alert:
            if notify.send(s, notify.format_pick(p)):
                sent += 1
            db.execute("UPDATE picks SET alerted=1 WHERE match_id=? AND market=? AND selection=? AND point=?",
                       (p["match_id"], p["market"], p["selection"], p["point"]))
        for cb in new_combos[:3]:
            if notify.send(s, notify.format_combo(cb)):
                sent += 1
            db.execute("UPDATE combos SET alerted=1 WHERE signature=?", (cb["signature"],))
        week = now.strftime("%G-%V")
        if now.weekday() == 0 and db.get("summary_week") != week:
            msg = notify.format_summary(metrics(db))
            if msg:
                notify.send(s, msg)
            db.set("summary_week", week)

    summary = dict(historico=len(hist), ligas_modelo=list(models), cuotas=len(odds), partidos=len(matches),
                   liquidados=n_settled, picks_nuevos=len(new_picks), combinadas_nuevas=len(new_combos),
                   avisos=sent, pesos=weights, creditos_odds_api=db.get("odds_api_remaining"))
    db.log(ts, "run", str(summary))
    db.set("last_run", ts)
    db.close()
    return summary


def run_backtest(s: Settings, db: DB, hist: pd.DataFrame) -> dict:
    bt = walk_forward(hist, s)
    res = summarize(bt, s)
    if res:
        db.set("backtest", res)
        db.set("weights_backtest", res["pesos"])
        log.info("Backtest: %s", res)
    return res
