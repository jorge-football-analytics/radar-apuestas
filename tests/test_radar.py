import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from radar.backtest import summarize, tennis_backtest, walk_forward
from radar.combos import build_combos
from radar.config import Settings, league_name, season_code, sport_of
from radar.model.dixon_coles import fit, score_matrix
from radar.model.elo import TeamElo, TennisElo, surface_of
from radar.model.markets import devig, label, market_fair_probs, market_probs, outcome
from radar.sources.football_data import tidy_fixtures, tidy_history
from radar.sources.odds_api import parse_odds
from radar.teams import match_team, match_us, player_key
from radar.value import evaluate_match

from .synthetic import NBA_TEAMS, PLAYERS, TEAMS, make_fixtures, make_history, make_nba, make_tennis


def test_season_code_and_registry():
    assert season_code(datetime(2026, 11, 5).date()) == "2627"
    assert sport_of("tennis_atp_paris_masters") == "tennis" and sport_of("nba") == "basketball"
    assert sport_of("laliga2") == "football" and sport_of("nfl") == "nfl"
    assert league_name("tennis_wta_finals") == "WTA Finals"
    assert surface_of("tennis_atp_french_open") == "Clay" and surface_of("tennis_atp_wimbledon") == "Grass"


def test_markets_sum_to_one_and_devig():
    p = market_probs(score_matrix(1.6, 1.1, -0.08))
    assert sum(p[("1x2", s, 0.0)] for s in "HDA") == pytest.approx(1)
    fair = devig([1.8, 3.6, 4.8])
    assert sum(fair) == pytest.approx(1) and fair[0] > fair[1] > fair[2]


def test_fit_recovers_strengths():
    hist, (att, de, home, base) = make_history()
    model = fit(hist, pd.Timestamp("2026-07-01"))
    est = np.array([model.attack[model.index[t]] for t in TEAMS])
    assert np.corrcoef(est, att)[0, 1] > 0.85


def test_name_matching():
    known = ["Man United", "Ath Madrid", "M'gladbach", "Brentford", "Paris SG", "Betis", "Inter"]
    assert match_team("Manchester United", known) == "Man United"
    assert match_team("Atlético Madrid", known) == "Ath Madrid"
    assert match_team("Borussia Monchengladbach", known) == "M'gladbach"
    assert match_us("Los Angeles Clippers", NBA_TEAMS) == "LA Clippers"
    assert match_us("Los Angeles Lakers", NBA_TEAMS) == "Los Angeles Lakers"
    assert player_key("Carlos Alcaraz") == player_key("Alcaraz C.")
    assert player_key("Alex de Minaur") == player_key("De Minaur A.")


def test_parse_odds_api_all_markets():
    ev = [{"id": "x", "commence_time": "2026-11-08T01:00:00Z", "home_team": "Boston Celtics",
           "away_team": "Miami Heat", "bookmakers": [{"key": "pinnacle", "markets": [
               {"key": "h2h", "outcomes": [{"name": "Boston Celtics", "price": 1.4}, {"name": "Miami Heat", "price": 3.1}]},
               {"key": "spreads", "outcomes": [{"name": "Boston Celtics", "price": 1.91, "point": -7.5},
                                               {"name": "Miami Heat", "price": 1.91, "point": 7.5}]},
               {"key": "totals", "outcomes": [{"name": "Over", "price": 1.9, "point": 221.5},
                                              {"name": "Under", "price": 1.9, "point": 221.5}]}]}]}]
    rows = parse_odds(ev, "nba")
    got = {(r["market"], r["selection"], r["point"]) for r in rows}
    assert ("ml", "H", 0.0) in got and ("spread", "H", -7.5) in got and ("spread", "A", -7.5) in got
    assert ("ou", "O", 221.5) in got and all(r["sport"] == "basketball" for r in rows)
    fair = market_fair_probs(pd.DataFrame(rows))
    assert fair[("spread", "H", -7.5)][0] == pytest.approx(0.5)


def test_tidy_football_data():
    raw = pd.DataFrame({"Div": ["SP2"], "Date": ["08/11/2026"], "Time": ["20:00"], "HomeTeam": ["Zaragoza"],
                        "AwayTeam": ["Racing"], "B365H": [2.1], "B365D": [3.3], "B365A": [3.6],
                        "PSH": [2.15], "PSD": [3.4], "PSA": [3.7]})
    fx = tidy_fixtures(raw)
    assert set(fx["bookmaker"]) == {"bet365", "pinnacle"} and fx["league"].iloc[0] == "laliga2"
    hist = tidy_history(raw.assign(league="laliga2", season="2627", FTHG=1, FTAG=0))
    assert hist["open_H"].iloc[0] == 2.15


def test_outcomes_and_labels():
    assert outcome("1x2", "D", 0, 1, 1) == 1
    assert outcome("spread", "H", -5.5, 110, 100) == 1 and outcome("spread", "A", -5.5, 110, 105) == 1
    assert outcome("ou", "U", 3, 2, 1) is None and outcome("ml", "A", 0, 1, 2) == 1
    assert label("spread", "A", -5.5, "Lakers", "Celtics") == "Celtics +5.5"
    assert label("ou", "O", 224.5, "a", "b", "puntos") == "Más de 224.5 puntos"


def test_team_elo_learns_and_prices_spreads():
    hist, strength = make_nba()
    elo = TeamElo("nba").fit(hist)
    best, worst = max(strength, key=strength.get), min(strength, key=strength.get)
    probs, info = elo.predict(best, worst, {"spread": [-10.5], "ou": [225.5]})
    assert probs[("ml", "H", 0.0)] > 0.75
    assert 8 < info["margen"] < 26
    assert probs[("ou", "O", 225.5)] == pytest.approx(0.5, abs=0.15)
    assert probs[("spread", "H", -10.5)] + probs[("spread", "A", -10.5)] == pytest.approx(1)


def test_tennis_elo_and_backtest():
    hist, skill = make_tennis()
    elo = TennisElo()
    elo.fit(hist)
    best, worst = PLAYERS[-1], PLAYERS[0]
    probs, _ = elo.predict(best, worst, "Hard")
    assert probs[("ml", "H", 0.0)] > 0.85
    bt = tennis_backtest(hist, Settings(), test_days=3000)
    assert bt["n_partidos"] > 1000 and 0 <= bt["pesos"]["tennis|ml"] <= 1


def test_backtest_football():
    hist, _ = make_history()
    s = Settings()
    res = summarize(walk_forward(hist, s, test_days=150, refit_days=28), s)
    assert res["n_predicciones"] > 500 and "football|1x2" in res["pesos"]


def _odds(rows):
    return pd.DataFrame([dict(bookmaker=b, market="ml", selection=sel, point=0.0, price=p, source="x")
                         for b, sel, p in rows])


def test_value_only_bet365_and_tiers():
    s = Settings()
    m = dict(id="nba|2026-11-08|A|B", sport="basketball", league="nba", kickoff="2026-11-08T01:00:00+00:00",
             home="A", away="B")
    base = [("pinnacle", "H", 1.80), ("pinnacle", "A", 2.10), ("williamhill", "H", 1.75), ("williamhill", "A", 2.05),
            ("onexbet", "H", 1.78), ("onexbet", "A", 2.08), ("betsson", "H", 2.0), ("betsson", "A", 1.85)]
    model = {("ml", "H", 0.0): 0.62, ("ml", "A", 0.0): 0.38}
    # sin bet365 → 'verificar' (nunca 'valor'), con cuota mínima
    cands, _ = evaluate_match(m, _odds(base), model, {}, s)
    assert all(c.tier != "valor" and c.verified == 0 for c in cands)
    # bet365 paga bien al local → 'valor' y élite
    cands, _ = evaluate_match(m, _odds(base + [("bet365", "H", 1.95), ("bet365", "A", 1.85)]), model, {}, s)
    v = [c for c in cands if c.tier == "valor"]
    assert len(v) == 1 and v[0].selection == "H" and v[0].bookmaker == "bet365" and v[0].odds == 1.95
    assert v[0].elite == 1 and v[0].risk == "medio" and v[0].stake > 0
    # betsson paga más que bet365 pero no cuenta: solo bet365
    assert all(c.bookmaker == "bet365" for c in cands)


def test_combos_only_bet365():
    s = Settings()
    matches = {f"m{i}": {"home": f"H{i}", "away": f"A{i}", "kickoff": "2026-11-08T20:00:00+00:00",
                         "league": "laliga", "sport": "football"} for i in range(5)}
    preds = pd.DataFrame([dict(match_id=f"m{i}", market="1x2", selection="H", point=0.0, p_final=0.66) for i in range(5)])
    odds = pd.DataFrame([dict(match_id=f"m{i}", bookmaker=b, market="1x2", selection="H", point=0.0,
                              price=1.62 if b == "bet365" else 1.80) for i in range(5) for b in ("bet365", "bwin")])
    combos = build_combos(preds, odds, matches, s)
    assert combos and all(c["bookmaker"] == "bet365" for c in combos)
    assert all(l["price"] == 1.62 for c in combos for l in c["legs"])
    assert {c["tier"] for c in combos} == {"valor"}


def test_migrates_old_database(tmp_path):
    old = Path(__file__).parent.parent / "data" / "radar.db"
    if not old.exists():
        pytest.skip("sin base de datos antigua")
    copy = tmp_path / "old.db"
    shutil.copy(old, copy)
    from radar.db import DB

    db = DB(copy)
    cols = {r[1] for r in db.conn.execute("PRAGMA table_info(picks)")}
    assert {"sport", "risk", "tier", "elite", "verified"} <= cols
    assert db.get("schema_version") == 2
    db.close()


def test_pipeline_end_to_end(tmp_path, monkeypatch):
    from radar import pipeline
    from radar.sources import espn
    from radar.sources import football_data as fd
    from radar.sources import tennis_data
    from radar.sources.odds_api import OddsAPI
    from radar.sources.oddspapi import OddsPapi

    hist, truth = make_history(seasons=("2324", "2425", "2526", "2627"))
    now = datetime(2026, 11, 5, 10, tzinfo=timezone.utc)
    past = hist[hist["date"] < pd.Timestamp("2026-11-01")]
    ko = (now + timedelta(days=1)).isoformat()
    pairs = [(TEAMS[2 * i], TEAMS[2 * i + 1]) for i in range(4)]
    fixtures = make_fixtures(pairs, ko, truth, generous_book="bet365", boost=1.13)
    nba_hist, strength = make_nba(start="2025-01-01", n_days=300)
    best, worst = max(strength, key=strength.get), min(strength, key=strength.get)

    def nba_odds(self, now=None):  # The Odds API: consenso sin bet365
        rows = []
        for b, ph, pa in (("pinnacle", 1.30, 3.80), ("williamhill", 1.28, 3.60), ("onexbet", 1.29, 3.75),
                          ("betsson", 1.27, 3.70)):
            for sel, p in (("H", ph), ("A", pa)):
                rows.append(dict(sport="basketball", league="nba", kickoff=ko, home="Boston Celtics" if best ==
                                 "Boston Celtics" else best, away=worst, source="the-odds-api", bookmaker=b,
                                 market="ml", selection=sel, point=0.0, price=p))
        return pd.DataFrame(rows)

    monkeypatch.setattr(fd, "load_history", lambda today, seasons, leagues=None: past)
    monkeypatch.setattr(fd, "load_fixtures", lambda cache_h=3: fixtures)
    monkeypatch.setattr(tennis_data, "load", lambda today, years=4: pd.DataFrame())
    monkeypatch.setattr(espn, "load", lambda league, path, today, years=2.6: nba_hist if league == "nba" else
                        pd.DataFrame(columns=["league", "date", "home", "away", "hg", "ag", "neutral"]))
    monkeypatch.setattr(OddsAPI, "fetch", nba_odds)
    monkeypatch.setattr(OddsAPI, "fetch_scores", lambda self, lg: [])
    monkeypatch.setattr(pipeline, "run_backtest", lambda s, db, ctx: {})

    def b365(self, match, now):  # OddsPapi confirma bet365 con una cuota alta para el favorito
        return [dict(sport="basketball", league="nba", kickoff=match["kickoff"], home=match["home"],
                     away=match["away"], source="oddspapi", bookmaker="bet365", market="ml", selection=sel,
                     point=0.0, price=p) for sel, p in (("H", 1.40), ("A", 3.20))]

    monkeypatch.setattr(OddsPapi, "bet365_odds", b365)
    sent = []
    monkeypatch.setattr(pipeline.notify, "send", lambda s, text: sent.append(text) or True)

    s = Settings(db_path=tmp_path / "radar.db", odds_api_key="x", oddspapi_key="y", telegram_token="x",
                 telegram_chat_id="y", alert_max_day=3, sports="football,basketball")
    out = pipeline.run(s, now=now)
    assert out["por_deporte"].get("football") == 4 and out["por_deporte"].get("basketball") == 1, out
    from radar.db import DB

    db = DB(s.db_path)
    picks = db.df("SELECT * FROM picks")
    assert set(picks["bookmaker"]) == {"bet365"}
    assert set(picks["tier"]) <= {"valor", "verificar", "pronostico"}
    assert (picks["tier"] == "valor").any()
    nba = picks[picks["sport"] == "basketball"]
    assert not nba.empty and nba["verified"].eq(1).all(), "OddsPapi debería haber confirmado bet365"
    elite_msgs = [t for t in sent if "JUGADA ÉLITE" in t]
    assert 1 <= len(elite_msgs) <= 3
    assert any("Radar del día" in t for t in sent)
    db.close()

    # segunda ejecución: no repite avisos ni el resumen diario
    sent.clear()
    pipeline.run(s, now=now + timedelta(hours=3))
    assert not any("Radar del día" in t for t in sent)
    assert len([t for t in sent if "JUGADA ÉLITE" in t]) + len(elite_msgs) <= 3  # tope diario respetado

    # se juegan los partidos → se liquidan
    played = pd.DataFrame([dict(league="laliga", season="2627", date=pd.Timestamp(ko[:10]), home=h, away=a, hg=2, ag=0,
                                close_H=1.9, close_D=3.5, close_A=4.2, close_O25=1.9, close_U25=1.95) for h, a in pairs])
    monkeypatch.setattr(fd, "load_history", lambda today, seasons, leagues=None: pd.concat([past, played]))
    nba_played = pd.concat([nba_hist, pd.DataFrame([dict(league="nba", date=pd.Timestamp(ko[:10]), home=best,
                                                         away=worst, hg=120, ag=100, neutral=False)])])
    monkeypatch.setattr(espn, "load", lambda league, path, today, years=2.6: nba_played if league == "nba" else
                        pd.DataFrame(columns=["league", "date", "home", "away", "hg", "ag", "neutral"]))
    out3 = pipeline.run(s, now=now + timedelta(days=3))
    assert out3["liquidados"] == 5
    db = DB(s.db_path)
    assert db.df("SELECT count(*) n FROM picks WHERE won IS NULL")["n"].iloc[0] == 0
    db.close()
