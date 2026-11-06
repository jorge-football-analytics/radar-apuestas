"""Orquestación: datos → modelos → cuotas → bajas → evaluación con bet365 → verificación → combinadas → avisos → aprender."""
from __future__ import annotations

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd

from . import notify
from .backtest import summarize, tennis_backtest, walk_forward
from .combos import build_combos, legs_json
from .config import FOOTBALL, OTHER, Settings, sport_of
from .db import DB
from .learn import metrics, recalibrate, settle
from .model.dixon_coles import DCModel, fit, score_matrix
from .model.elo import TeamElo, TennisElo, surface_of
from .model.markets import market_probs
from .sources import espn
from .sources import football_data as fd
from .sources import tennis_data
from .sources.injuries import absence_factor, fetch_api_football, load_manual
from .sources.odds_api import OddsAPI
from .sources.oddspapi import OddsPapi
from .teams import match_player, match_team, match_us, player_key
from .value import evaluate_match

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
    if "tennis" in act:
        ctx.tennis_hist = tennis_data.load(now.date())
        if not ctx.tennis_hist.empty:
            ctx.tennis = TennisElo()
            ctx.tennis.fit(ctx.tennis_hist)
            ctx.tennis_keys = dict(ctx.tennis.names)
            log.info("Tenis: %d partidos, %d jugadores", len(ctx.tennis_hist), len(ctx.tennis_keys))
    for league, v in OTHER.items():
        if v["sport"] in act and v.get("espn"):
            h = espn.load(league, v["espn"], now.date())
            ctx.us_hist[league] = h
            if not h.empty:
                ctx.us_elo[league] = TeamElo(league).fit(h)
                log.info("%s: %d partidos", league, len(h))
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
    ko = pd.to_datetime(odds["kickoff"], utc=True)
    odds["kickoff"] = ko.map(lambda t: t.isoformat())
    odds["match_id"] = odds["league"] + "|" + ko.dt.strftime("%Y-%m-%d") + "|" + odds["home"] + "|" + odds["away"]
    return odds


def store_odds(db: DB, odds: pd.DataFrame, now: datetime) -> None:
    ts = now.isoformat()
    fresher = {"oddspapi": 3, "the-odds-api": 2, "football-data": 1}
    with db.tx() as c:
        for m in odds.drop_duplicates("match_id").itertuples():
            c.execute("INSERT OR IGNORE INTO matches (id, league, kickoff, home, away, sport) VALUES (?,?,?,?,?,?)",
                      (m.match_id, m.league, m.kickoff, m.home, m.away, m.sport))
        for o in odds.itertuples():
            prev = c.execute("SELECT source, updated FROM odds WHERE match_id=? AND bookmaker=? AND market=? AND "
                             "selection=? AND point=?", (o.match_id, o.bookmaker, o.market, o.selection, o.point)).fetchone()
            if prev is None:
                c.execute("INSERT INTO odds VALUES (?,?,?,?,?,?,?,?,?,?)",
                          (o.match_id, o.bookmaker, o.market, o.selection, o.point, o.price, o.price, ts, ts, o.source))
            elif fresher.get(o.source, 0) >= fresher.get(prev[0], 0) or prev[1] < (now - timedelta(hours=12)).isoformat():
                c.execute("UPDATE odds SET price=?, updated=?, source=? WHERE match_id=? AND bookmaker=? AND market=? "
                          "AND selection=? AND point=?",
                          (o.price, ts, o.source, o.match_id, o.bookmaker, o.market, o.selection, o.point))


def _absences(s: Settings, db: DB, upcoming: pd.DataFrame, ctx: Ctx, now: datetime) -> dict:
    out: dict = defaultdict(list)
    for (lg, team), lst in load_manual(now.date()).items():
        out[(lg, team)] += lst
    fb = upcoming[upcoming["sport"] == "football"]
    for lg, g in fb.groupby("league"):
        if lg not in FOOTBALL:
            continue
        for team, lst in fetch_api_football(s, db, lg, {k[:10] for k in g["kickoff"]}, ctx.fb_known.get(lg, []),
                                            now.date()).items():
            out[(lg, team)] += lst
    return out


# ------------------------------------------------------------------ análisis
def model_for(m: dict, o: pd.DataFrame, ctx: Ctx, absences: dict, s: Settings, now: datetime):
    sport, league = m["sport"], m["league"]
    notes: list[str] = []
    if sport == "football":
        model = ctx.fb_models.get(league)
        if model is None:
            return None, notes
        r = model.rates(m["home"], m["away"])
        for t in (m["home"], m["away"]):
            seen = model.last_seen.get(t)
            if seen and (pd.Timestamp(now).tz_convert(None) - pd.Timestamp(seen)).days > 200:
                return None, notes  # recién ascendido: sin datos recientes
        if not r:
            return None, notes
        lam, mu = r
        fh = absence_factor(absences.get((league, m["home"]), []), s.injury_impact, s.injury_cap)
        fa = absence_factor(absences.get((league, m["away"]), []), s.injury_impact, s.injury_cap)
        lam, mu = lam * fh / fa ** 0.5, mu * fa / fh ** 0.5
        for team, f in ((m["home"], fh), (m["away"], fa)):
            if f < 1:
                names = ", ".join(str(a["jugador"]) for a in absences[(league, team)][:5])
                notes.append(f"Bajas {team}: {names} (−{1 - f:.0%} goles esperados)")
        points = sorted(set(o.loc[o["market"] == "ou", "point"])) or [2.5]
        notes.append(f"Goles esperados {lam:.2f} - {mu:.2f}")
        return market_probs(score_matrix(lam, mu, model.rho), points), notes
    if sport == "tennis" and ctx.tennis:
        surface = surface_of(league)
        res = ctx.tennis.predict(m["home"], m["away"], surface)
        if not res:
            return None, notes
        probs, info = res
        sup = {"Hard": "pista dura", "Clay": "tierra", "Grass": "hierba"}[surface]
        notes.append(f"Elo {info['elo_home']:.0f} vs {info['elo_away']:.0f} · superficie: {sup} · "
                     f"{info['partidos'][0]} y {info['partidos'][1]} partidos en la base")
        return probs, notes
    if sport in ("basketball", "nfl") and league in ctx.us_elo:
        lines = {"spread": sorted(set(o.loc[o["market"] == "spread", "point"])),
                 "ou": sorted(set(o.loc[o["market"] == "ou", "point"]))}
        res = ctx.us_elo[league].predict(m["home"], m["away"], lines)
        if not res:
            return None, notes
        probs, info = res
        notes.append(f"Elo {info['elo_home']:.0f} vs {info['elo_away']:.0f} · margen esperado {info['margen']:+.1f} · "
                     f"total esperado {info['total']:.0f}")
        return probs, notes
    return None, notes


def analyze(s: Settings, db: DB, ctx: Ctx, now: datetime, absences: dict, weights: dict):
    upcoming = db.df("SELECT * FROM matches WHERE status='pendiente' AND kickoff > ? AND kickoff < ?",
                     (now.isoformat(), (now + timedelta(hours=HORIZON_H)).isoformat()))
    all_odds = db.df("SELECT * FROM odds WHERE match_id IN (SELECT id FROM matches WHERE status='pendiente' "
                     "AND kickoff > ?)", (now.isoformat(),))
    candidates, preds, matches = [], [], {}
    for m in upcoming.to_dict("records"):
        m["sport"] = m.get("sport") or sport_of(m["league"])
        if m["sport"] not in s.active_sports:
            continue
        matches[m["id"]] = m
        o = all_odds[all_odds["match_id"] == m["id"]]
        if o.empty:
            continue
        probs, notes = model_for(m, o, ctx, absences, s, now)
        b365 = o[o["bookmaker"] == s.book]
        if not b365.empty and (b365["source"] == "football-data").all():
            notes.append("Cuota de bet365 tomada el martes/viernes: confírmala antes de apostar")
        cs, ps = evaluate_match(m, o, probs, weights, s, notes)
        candidates += cs
        preds += ps
    return candidates, pd.DataFrame(preds), matches, all_odds


def verify_with_oddspapi(s: Settings, db: DB, ctx: Ctx, candidates: list, matches: dict, now: datetime) -> int:
    """Pide la cuota real de bet365 solo para los mejores candidatos que no la tienen."""
    op = OddsPapi(s, db)
    if not op.enabled:
        return 0
    # solo competiciones sin bet365 gratis (las ligas de fútbol ya lo traen de football-data)
    candidates = [c for c in candidates if not FOOTBALL.get(c.league, {}).get("fd")]
    # primero los que parecen tener valor; con lo que sobre, los pronósticos sin cuota de bet365
    todo = sorted([c for c in candidates if c.tier == "verificar"], key=lambda c: c.score, reverse=True) + \
        sorted([c for c in candidates if c.tier == "pronostico" and not c.verified], key=lambda c: c.ev, reverse=True)
    done, added = set(), 0
    for c in todo:
        if c.match_id in done or op.allowance(now) < 1:
            continue
        done.add(c.match_id)
        checked = db.get(f"op_checked_{c.match_id}")
        if checked and (now - datetime.fromisoformat(checked)).total_seconds() < 8 * 3600:
            continue
        rows = op.bet365_odds(matches[c.match_id], now)
        db.set(f"op_checked_{c.match_id}", now.isoformat())
        if rows:
            df = pd.DataFrame(rows)
            df["match_id"] = c.match_id
            store_odds(db, df, now)
            added += len(rows)
    return added


def save_picks(db: DB, candidates: list, ts: str, now: datetime) -> list[dict]:
    """Inserta/actualiza jugadas. Las ya avisadas no se tocan (queda registro de lo que se recomendó)."""
    seen = set()
    changed = []
    with db.tx() as c:
        for cd in candidates:
            r = cd.as_row()
            key = (r["match_id"], r["market"], r["selection"], r["point"])
            seen.add(key)
            ex = c.execute("SELECT id, alerted, tier, ev FROM picks WHERE match_id=? AND market=? AND selection=? "
                           "AND point=?", key).fetchone()
            vals = dict(p_final=r["p_final"], p_model=r["p_model"], p_market=r["p_market"], odds=r["odds"],
                        bookmaker=r["bookmaker"], b365_odds=r["b365_odds"], ev=r["ev"], stake=r["stake"],
                        confidence=r["confidence"], reasons=r["reasons"], sport=r["sport"], risk=r["risk"],
                        tier=r["tier"], elite=r["elite"], verified=r["verified"], min_odds=r["min_odds"],
                        score=r["score"], label=r["label"], updated=ts)
            if ex is None:
                cols = ["created", "match_id", "league", "kickoff", "home", "away", "market", "selection", "point",
                        *vals]
                c.execute(f"INSERT INTO picks ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                          (ts, r["match_id"], r["league"], r["kickoff"], r["home"], r["away"], r["market"],
                           r["selection"], r["point"], *vals.values()))
                changed.append(r)
            elif not ex["alerted"]:
                c.execute(f"UPDATE picks SET {', '.join(f'{k}=?' for k in vals)} WHERE id=?",
                          (*vals.values(), ex["id"]))
                if r["elite"]:
                    changed.append(r)
        # jugadas de partidos futuros que ya no pasan el filtro (la cuota cambió) y no se avisaron → fuera
        for row in c.execute("SELECT id, match_id, market, selection, point FROM picks WHERE alerted=0 AND won IS NULL "
                             "AND kickoff > ?", (now.isoformat(),)).fetchall():
            if (row[1], row[2], row[3], row[4]) not in seen:
                c.execute("DELETE FROM picks WHERE id=?", (row[0],))
    return changed


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

    # 1) Resultados y aprendizaje
    api = OddsAPI(s, db)
    no_source = [lg for lg in {*FOOTBALL, *OTHER} if (lg in FOOTBALL and not FOOTBALL[lg]["fd"]) or
                 (lg in OTHER and not OTHER[lg].get("espn"))]
    scores = []
    for lg in no_source:
        if not db.df("SELECT 1 FROM matches WHERE league=? AND status='pendiente' AND kickoff < ?",
                     (lg, (now - timedelta(hours=3)).isoformat())).empty:
            scores += api.fetch_scores(lg)
    n_settled = settle(db, results_by_sport(ctx), now, scores or None)
    weights = recalibrate(db, s)

    # 2) Cuotas: football-data (bet365 gratis en fútbol) + The Odds API (consenso)
    frames = []
    if "football" in s.active_sports:
        frames.append(fd.load_fixtures())
    frames.append(api.fetch(now))
    frames = [f for f in frames if f is not None and not f.empty]
    n_odds = 0
    if frames:
        odds = canonicalize(pd.concat(frames, ignore_index=True), ctx, db)
        odds = odds[pd.to_datetime(odds["kickoff"], utc=True) > now]
        store_odds(db, odds, now)
        n_odds = len(odds)

    # 3) Análisis → verificación de bet365 en OddsPapi → nuevo análisis
    upcoming = db.df("SELECT * FROM matches WHERE status='pendiente' AND kickoff > ?", (ts,))
    absences = _absences(s, db, upcoming, ctx, now) if not upcoming.empty else {}
    candidates, preds, matches, all_odds = analyze(s, db, ctx, now, absences, weights)
    if verify_with_oddspapi(s, db, ctx, candidates, matches, now):
        candidates, preds, matches, all_odds = analyze(s, db, ctx, now, absences, weights)

    with db.tx() as c:
        for p in preds.to_dict("records"):
            c.execute("INSERT OR REPLACE INTO predictions VALUES (?,?,?,?,?,?,?,?,NULL)",
                      (p["match_id"], p["market"], p["selection"], p["point"], p["p_model"], p["p_market"],
                       p["p_final"], ts))
    save_picks(db, candidates, ts, now)

    # 4) Combinadas (partidos de las próximas 36 h)
    soon = {k: v for k, v in matches.items() if pd.Timestamp(v["kickoff"]) < pd.Timestamp(now) + pd.Timedelta(hours=36)}
    combos = build_combos(preds[preds["match_id"].isin(soon)], all_odds[all_odds["match_id"].isin(soon)], soon, s) \
        if soon and not preds.empty else []
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
                   elite=sum(c.elite for c in candidates), combinadas=len(combos), avisos=sent, pesos=weights,
                   creditos_odds_api=db.get("odds_api_remaining"),
                   oddspapi_mes=db.get(f"oddspapi_used_{now:%Y-%m}", 0))
    # mantener la base de datos pequeña: cuotas de partidos terminados hace más de 30 días
    db.execute("DELETE FROM odds WHERE match_id IN (SELECT id FROM matches WHERE status != 'pendiente' AND kickoff < ?)",
               ((now - timedelta(days=30)).isoformat(),))
    db.log(ts, "run", str(summary))
    db.set("last_run", ts)
    db.close()
    return summary
