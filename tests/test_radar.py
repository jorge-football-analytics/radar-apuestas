from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from radar.backtest import summarize, walk_forward
from radar.combos import build_combos
from radar.config import Settings, season_code
from radar.model.dixon_coles import fit, score_matrix
from radar.model.markets import devig, market_fair_probs, market_probs, outcome
from radar.sources.football_data import tidy_fixtures, tidy_history
from radar.sources.odds_api import parse_odds
from radar.teams import match_team

from .synthetic import TEAMS, make_fixtures, make_history


def test_season_code():
    assert season_code(datetime(2026, 11, 5).date()) == "2627"
    assert season_code(datetime(2027, 3, 1).date()) == "2627"
    assert season_code(datetime(2026, 11, 5).date(), -1) == "2526"


def test_markets_sum_to_one():
    m = score_matrix(1.6, 1.1, -0.08)
    assert m.sum() == pytest.approx(1)
    p = market_probs(m)
    assert sum(p[("1x2", s, 0.0)] for s in "HDA") == pytest.approx(1)
    assert p[("ou", "O", 2.5)] + p[("ou", "U", 2.5)] == pytest.approx(1)
    assert p[("1x2", "H", 0.0)] > p[("1x2", "A", 0.0)]


def test_devig():
    fair = devig([1.8, 3.6, 4.8])
    assert sum(fair) == pytest.approx(1)
    assert fair[0] > fair[1] > fair[2]


def test_fit_recovers_strengths():
    hist, (att, de, home, base) = make_history()
    model = fit(hist, pd.Timestamp("2026-07-01"))
    idx = model.index
    est_att = np.array([model.attack[idx[t]] for t in TEAMS])
    est_def = np.array([model.defence[idx[t]] for t in TEAMS])
    assert np.corrcoef(est_att, att)[0, 1] > 0.85
    assert np.corrcoef(est_def, de)[0, 1] > 0.75
    assert model.home == pytest.approx(home, abs=0.08)


def test_team_matching():
    known = ["Man United", "Ath Madrid", "Dortmund", "M'gladbach", "Brentford", "Paris SG", "Betis", "Inter"]
    assert match_team("Manchester United", known) == "Man United"
    assert match_team("Atlético Madrid", known) == "Ath Madrid"
    assert match_team("Borussia Monchengladbach", known) == "M'gladbach"
    assert match_team("Brentford FC", known) == "Brentford"
    assert match_team("Paris Saint Germain", known) == "Paris SG"
    assert match_team("Real Betis", known) == "Betis"
    assert match_team("Inter Milan", known) == "Inter"


def test_parse_odds_api():
    ev = [{"id": "x", "commence_time": "2026-11-08T20:00:00Z", "home_team": "Real Madrid", "away_team": "Sevilla",
           "bookmakers": [{"key": "bet365", "markets": [
               {"key": "h2h", "outcomes": [{"name": "Real Madrid", "price": 1.5}, {"name": "Sevilla", "price": 6.5},
                                           {"name": "Draw", "price": 4.4}]},
               {"key": "totals", "outcomes": [{"name": "Over", "price": 1.7, "point": 2.5},
                                              {"name": "Under", "price": 2.1, "point": 2.5}]}]}]}]
    rows = parse_odds(ev, "laliga")
    sels = {(r["market"], r["selection"]): r["price"] for r in rows}
    assert sels[("1x2", "H")] == 1.5 and sels[("1x2", "D")] == 4.4 and sels[("ou", "U")] == 2.1


def test_tidy_football_data_formats():
    raw = pd.DataFrame({"Div": ["SP1"], "Date": ["08/11/2026"], "Time": ["20:00"], "HomeTeam": ["Betis"],
                        "AwayTeam": ["Sevilla"], "B365H": [2.1], "B365D": [3.3], "B365A": [3.6],
                        "PSH": [2.15], "PSD": [3.4], "PSA": [3.7], "B365>2.5": [1.9], "B365<2.5": [1.9]})
    fx = tidy_fixtures(raw)
    assert set(fx["bookmaker"]) == {"bet365", "pinnacle"}
    assert fx["kickoff"].iloc[0].startswith("2026-11-08T20:00")  # en noviembre Londres = UTC
    hist = tidy_history(raw.assign(league="laliga", season="2627", FTHG=1, FTAG=0))
    assert hist["open_H"].iloc[0] == 2.15  # Pinnacle tiene prioridad


def test_outcomes():
    assert outcome("1x2", "D", 0, 1, 1) == 1
    assert outcome("ou", "O", 2.5, 2, 1) == 1
    assert outcome("ou", "U", 2.5, 2, 1) == 0
    assert outcome("btts", "Y", 0, 2, 0) == 0


def test_backtest_runs_and_market_is_hard_to_beat():
    hist, _ = make_history()
    s = Settings()
    bt = walk_forward(hist, s, test_days=150, refit_days=28)
    res = summarize(bt, s)
    assert res["n_predicciones"] > 500
    assert 0 <= res["pesos"]["1x2"] <= 1
    # En la liga sintética el mercado conoce las fuerzas reales: el modelo no debería superarlo
    assert res["logloss_1x2"]["mercado"] <= res["logloss_1x2"]["modelo"] + 0.01


def test_pipeline_end_to_end(tmp_path, monkeypatch):
    from radar import pipeline
    from radar.sources import football_data as fd
    from radar.sources.odds_api import OddsAPI

    hist, truth = make_history(seasons=("2324", "2425", "2526", "2627"))
    cut = pd.Timestamp("2026-11-01")
    past = hist[hist["date"] < cut]
    now = datetime(2026, 11, 5, 18, tzinfo=timezone.utc)
    ko = (now + timedelta(days=2)).isoformat()
    pairs = [(TEAMS[0], TEAMS[1]), (TEAMS[2], TEAMS[3]), (TEAMS[4], TEAMS[5]), (TEAMS[6], TEAMS[7])]
    fixtures = make_fixtures(pairs, ko, truth)

    state = {"hist": past}
    monkeypatch.setattr(fd, "load_history", lambda today, seasons: state["hist"])
    monkeypatch.setattr(fd, "load_fixtures", lambda cache_h=3: fixtures)
    monkeypatch.setattr(OddsAPI, "fetch", lambda self, now=None: pd.DataFrame())
    monkeypatch.setattr(pipeline, "run_backtest", lambda s, db, hist: {})
    sent = []
    monkeypatch.setattr(pipeline.notify, "send", lambda s, text: sent.append(text) or True)

    s = Settings(db_path=tmp_path / "radar.db", odds_api_key="", telegram_token="x", telegram_chat_id="y")
    out = pipeline.run(s, now=now)
    assert out["partidos"] == 4
    assert out["picks_nuevos"] >= 1, out
    assert any("Gana" in t for t in sent), "debería avisar de la cuota inflada de bwin"

    # Segunda ejecución igual: no repite avisos
    sent.clear()
    out2 = pipeline.run(s, now=now + timedelta(hours=3))
    assert not any("Apuesta de valor" in t for t in sent)

    # Se juegan los partidos → se liquidan y se miden
    played = pd.DataFrame([dict(league="laliga", season="2627", date=pd.Timestamp(ko[:10]), home=h, away=a,
                                hg=2, ag=0, close_H=1.9, close_D=3.5, close_A=4.2,
                                close_O25=1.9, close_U25=1.95) for h, a in pairs])
    state["hist"] = pd.concat([past, played], ignore_index=True)
    out3 = pipeline.run(s, now=now + timedelta(days=3))
    assert out3["liquidados"] == 4
    from radar.db import DB
    from radar.learn import metrics

    db = DB(s.db_path)
    m = metrics(db)
    assert m["n"] >= 1
    home_picks = db.df("SELECT won FROM picks WHERE selection='H'")
    assert len(home_picks) >= 1 and home_picks["won"].eq(1).all()  # todos los locales ganaron 2-0
    assert db.df("SELECT clv FROM picks WHERE market='1x2'")["clv"].notna().all()
    db.close()


def test_combos_single_book_and_risk_levels():
    s = Settings()
    matches = {f"m{i}": {"home": f"H{i}", "away": f"A{i}", "kickoff": "2026-11-08T20:00:00+00:00"} for i in range(5)}
    preds = pd.DataFrame([dict(match_id=f"m{i}", market="1x2", selection="H", point=0.0, p_final=0.66) for i in range(5)])
    odds = pd.DataFrame([dict(match_id=f"m{i}", bookmaker=b, market="1x2", selection="H", point=0.0,
                              price=1.62 if b == "bet365" else 1.55) for i in range(5) for b in ("bet365", "bwin")])
    combos = build_combos(preds, odds, matches, s)
    assert combos and all(c["bookmaker"] == "bet365" for c in combos)
    assert {c["risk"] for c in combos} >= {"conservadora"}
    for c in combos:
        assert len({l["match_id"] for l in c["legs"]}) == len(c["legs"])
        assert c["ev"] > 0
