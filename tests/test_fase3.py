"""Fase 3: conectores reparados (ESPN, OddsPapi, tenis), mercados derivados, estadísticas, combinadas por perfil,
copiloto y liquidación de copas."""
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from radar.config import Settings


class Resp:
    def __init__(self, data, code=200):
        self.data, self.status_code = data, code

    def json(self):
        return self.data

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests
            raise requests.HTTPError(f"{self.status_code}")


def event(home, away, hg, ag, d, done=True, state="post", season=2):
    return {"id": f"{home}{away}{d}", "date": f"{d}T19:00Z", "season": {"type": season},
            "competitions": [{"status": {"type": {"completed": done, "state": state}}, "competitors": [
                {"homeAway": "home", "team": {"displayName": home}, "score": str(hg)},
                {"homeAway": "away", "team": {"displayName": away}, "score": str(ag)}]}]}


# ------------------------------------------------------------------ ESPN
def test_espn_never_asks_for_date_ranges_and_caches_settled_days(tmp_path, monkeypatch):
    from radar.sources import espn

    monkeypatch.setattr(espn, "RAW_DIR", tmp_path)
    calls = []

    def fake_get(url, headers=None, timeout=None, params=None):
        calls.append(dict(params))
        d = str(params["dates"])
        assert "-" not in d, "ESPN devuelve 400 con rangos de fechas"
        return Resp({"events": [event("Boston Celtics", "Miami Heat", 110, 100, f"{d[:4]}-{d[4:6]}-{d[6:]}")]})

    monkeypatch.setattr(espn.requests, "get", fake_get)
    today = date(2026, 11, 26)
    df = espn.load("nba", "basketball/nba", today, years=0.05)
    assert len(df) > 10 and (df["hg"] == 110).all()
    n1 = len(calls)
    espn.load("nba", "basketball/nba", today, years=0.05)
    assert len(calls) - n1 <= 3  # solo se repiten los 3 últimos días (pueden cambiar); el resto, caché


def test_espn_nfl_by_week_and_cup_results(tmp_path, monkeypatch):
    from radar.sources import espn

    monkeypatch.setattr(espn, "RAW_DIR", tmp_path)
    seen = []

    def fake_get(url, headers=None, timeout=None, params=None):
        seen.append((url, dict(params)))
        if "nfl" in url:
            assert "week" in params and "seasontype" in params
            return Resp({"events": [event("Kansas City Chiefs", "Denver Broncos", 27, 20, "2026-11-22")]})
        return Resp({"events": [event("Sabah FK", "Barcelona", 1, 4, "2026-11-25", season=13)]})

    monkeypatch.setattr(espn.requests, "get", fake_get)
    nfl = espn.load("nfl", "football/nfl", date(2026, 11, 26))
    assert not nfl.empty and nfl["home"].iloc[0] == "Kansas City Chiefs"
    cups = espn.cup_results(date(2026, 11, 26), days=1)
    assert cups and cups[0]["home_team"] == "Sabah FK" and cups[0]["completed"]
    assert {x["name"]: x["score"] for x in cups[0]["scores"]} == {"Sabah FK": 1, "Barcelona": 4}


def test_tennis_data_403_is_remembered(tmp_path, monkeypatch):
    from radar.sources import tennis_data

    monkeypatch.setattr(tennis_data, "RAW_DIR", tmp_path)
    calls = []
    monkeypatch.setattr(tennis_data.requests, "get", lambda url, headers=None, timeout=None: calls.append(url) or Resp({}, 403))
    assert tennis_data._get("http://www.tennis-data.co.uk/2026/2026.xlsx", tmp_path / "x.xlsx", 6) is None
    assert tennis_data._blocked()
    tennis_data._get("http://www.tennis-data.co.uk/2025/2025.xlsx", tmp_path / "y.xlsx", 6)
    assert len(calls) == 1  # no se insiste durante 7 días


# ------------------------------------------------------------------ OddsPapi
MARKETS_FB = [
    {"marketId": 101, "marketType": "1x2", "period": "fulltime", "handicap": 0,
     "outcomes": [{"outcomeId": 103, "outcomeName": "2"}, {"outcomeId": 102, "outcomeName": "X"},
                  {"outcomeId": 101, "outcomeName": "1"}]},
    {"marketId": 1010, "marketType": "totals", "period": "fulltime", "handicap": 2.5,
     "outcomes": [{"outcomeId": 1011, "outcomeName": "Under"}, {"outcomeId": 1010, "outcomeName": "Over"}]},
    {"marketId": 5000, "marketType": "totals-shots", "period": "fulltime", "handicap": 2.5,
     "outcomes": [{"outcomeId": 5001, "outcomeName": "Over"}, {"outcomeId": 5002, "outcomeName": "Under"}]},
    {"marketId": 101902, "marketType": "doublechance", "period": "fulltime", "handicap": 0,
     "outcomes": [{"outcomeId": 101904, "outcomeName": "2X"}, {"outcomeId": 101902, "outcomeName": "1X"},
                  {"outcomeId": 101903, "outcomeName": "12"}]},
    {"marketId": 1060, "marketType": "spreads", "period": "fulltime", "handicap": -1.5,
     "outcomes": [{"outcomeId": 1061, "outcomeName": "2"}, {"outcomeId": 1060, "outcomeName": "1"}]},
    {"marketId": 1070, "marketType": "spreads", "period": "fulltime", "handicap": -0.25,
     "outcomes": [{"outcomeId": 1070, "outcomeName": "1"}, {"outcomeId": 1071, "outcomeName": "2"}]},
    {"marketId": 104, "marketType": "bothteamsscore", "period": "fulltime", "handicap": 0,
     "outcomes": [{"outcomeId": 104, "outcomeName": "Yes"}, {"outcomeId": 105, "outcomeName": "No"}]},
]


def test_market_map_uses_outcome_names_not_order():
    from radar.sources.oddspapi import build_market_map

    m = build_market_map(MARKETS_FB, "football")
    assert m["101"] == ("1x2", "H", 0.0) and m["103"] == ("1x2", "A", 0.0) and m["102"] == ("1x2", "D", 0.0)
    assert m["1010"] == ("ou", "O", 2.5) and m["1011"] == ("ou", "U", 2.5)
    assert "5001" not in m  # tiros: nunca se confunden con goles
    assert m["101904"] == ("dc", "X2", 0.0) and m["104"] == ("btts", "Y", 0.0)
    assert m["1060"] == ("spread", "H", -1.5) and m["1061"] == ("spread", "A", -1.5)
    assert "1070" not in m  # hándicap de cuarto: se liquida a medias, no se usa
    tennis = build_market_map([{"marketId": 121, "marketType": "moneyline", "period": "result", "handicap": 0,
                                "outcomes": [{"outcomeId": 122, "outcomeName": "2"},
                                             {"outcomeId": 121, "outcomeName": "1"}]}], "tennis")
    assert tennis == {"121": ("ml", "H", 0.0), "122": ("ml", "A", 0.0)}


def test_swapped_fixture_flips_spread_and_double_chance():
    from radar.sources.oddspapi import build_market_map, parse_match_odds

    mmap = build_market_map(MARKETS_FB, "football")
    pl = lambda p: {"players": {"0": {"price": p}}}
    data = {"bookmakerOdds": {"bet365": {"markets": {
        "1060": {"outcomes": {"1060": pl(2.80), "1061": pl(1.40)}},
        "101902": {"outcomes": {"101902": pl(1.25), "101904": pl(1.70), "101903": pl(1.30)}}}}}}
    match = dict(id="x", league="laliga", kickoff="2026-11-28T20:00:00+00:00", home="Betis", away="Real Madrid")
    rows = {(r["market"], r["selection"], r["point"]): r["price"]
            for r in parse_match_odds(data, mmap, match, "football", True, "bet365", datetime.now(timezone.utc))}
    # OddsPapi: participante 1 = Real Madrid (nuestro visitante) −1,5 → nuestro local (Betis) +1,5
    assert rows[("spread", "A", 1.5)] == 2.80 and rows[("spread", "H", 1.5)] == 1.40
    assert rows[("dc", "X2", 0.0)] == 1.25 and rows[("dc", "1X", 0.0)] == 1.70


def test_bulk_fetch_costs_one_call_per_sport_and_per_five_tournaments(tmp_path, monkeypatch):
    from radar.db import DB
    from radar.sources import oddspapi

    now = datetime(2026, 11, 27, 12, tzinfo=timezone.utc)
    ko = lambda h: (now + timedelta(hours=h)).isoformat()
    ours = [dict(id=f"nba{i}", league="nba", sport="basketball", kickoff=ko(5 + i), home=f"Team{i}", away=f"Rival{i}")
            for i in range(7)]
    fixtures = [dict(fixtureId=f"f{i}", participant1Name=f"Team{i}", participant2Name=f"Rival{i}",
                     startTime=ko(5 + i).replace("+00:00", "Z"), tournamentId=100 + i % 6, tournamentName="x")
                for i in range(7)]
    mk = [{"marketId": 1, "marketType": "moneyline", "period": "result", "handicap": 0,
           "outcomes": [{"outcomeId": 11, "outcomeName": "1"}, {"outcomeId": 12, "outcomeName": "2"}]}]
    calls = []

    def fake_get(url, params, timeout):
        ep = url.rsplit("/", 1)[1]
        calls.append((ep, dict(params)))
        if ep == "fixtures":
            return Resp(fixtures)
        if ep == "markets":
            return Resp(mk)
        tids = [int(t) for t in params["tournamentIds"].split(",")]
        assert len(tids) <= 5
        return Resp([{"fixtureId": f["fixtureId"], "bookmakerOdds": {params["bookmaker"]: {"markets": {"1": {"outcomes": {
            "11": {"players": {"0": {"price": 1.80}}}, "12": {"players": {"0": {"price": 2.05}}}}}}}}}
            for f in fixtures if f["tournamentId"] in tids])

    monkeypatch.setattr(oddspapi.requests, "get", fake_get)
    monkeypatch.setattr(oddspapi.time, "sleep", lambda x: None)
    db = DB(tmp_path / "op.db")
    op = oddspapi.OddsPapi(Settings(oddspapi_key="k", oddspapi_books="bet365"), db)
    rows = op.fetch_bulk(ours, now)
    eps = [c[0] for c in calls]
    assert eps.count("fixtures") == 1 and eps.count("odds-by-tournaments") == 2  # 6 competiciones → 2 consultas
    assert {r["match_id"] for r in rows} == {m["id"] for m in ours}
    assert all(r["bookmaker"] == "bet365" for r in rows)
    calls.clear()
    assert op.fetch_bulk(ours, now + timedelta(hours=1)) == [] and not [c for c in calls if c[0] != "markets"]
    db.close()


# ------------------------------------------------------------------ mercados derivados
def test_goal_rates_reproduce_the_market_and_derive_consistent_markets():
    from radar.model.derived import fit_goal_rates, football_markets
    from radar.model.markets import label, outcome

    lam, mu = fit_goal_rates(0.55, 0.25, 0.20, 0.52)
    m = football_markets(lam, mu)
    assert m[("1x2", "H", 0.0)] == pytest.approx(0.55, abs=0.01) and m[("ou", "O", 2.5)] == pytest.approx(0.52, abs=0.01)
    assert m[("dc", "1X", 0.0)] == pytest.approx(m[("1x2", "H", 0.0)] + m[("1x2", "D", 0.0)])
    assert m[("spread", "H", -0.5)] == pytest.approx(m[("1x2", "H", 0.0)], abs=1e-9)  # −0,5 = gana el local
    assert m[("spread", "H", 0.5)] == pytest.approx(m[("dc", "1X", 0.0)], abs=1e-9)  # +0,5 = no pierde
    assert m[("ou", "O", 1.5)] > m[("ou", "O", 2.5)] > m[("ou", "O", 3.5)]
    assert 0 < m[("btts", "Y", 0.0)] < 1
    assert outcome("dc", "1X", 0, 1, 1) == 1 and outcome("dc", "12", 0, 1, 1) == 0 and outcome("dc", "X2", 0, 0, 2) == 1
    assert label("dc", "1X", 0, "Betis", "Sevilla") == "Betis o empate"


def test_derived_markets_flow_through_evaluation_with_honest_limits():
    from radar.value import evaluate_match

    now = datetime(2026, 11, 27, 12, tzinfo=timezone.utc)
    t = now.isoformat()
    rows = lambda b, mk, sels, ps, pt=0.0: [dict(bookmaker=b, market=mk, selection=x, point=pt, price=p, fetched_at=t,
                                                 source="x") for x, p in zip(sels, ps)]
    m = dict(id="laliga|x", sport="football", league="laliga", kickoff="2026-11-28T20:00:00+00:00", home="A", away="B")
    odds = pd.DataFrame(rows("pinnacle", "1x2", "HDA", [1.60, 4.0, 6.0]) + rows("pinnacle", "ou", "OU", [1.90, 1.95], 2.5)
                        + rows("bet365", "1x2", "HDA", [1.62, 3.8, 5.5]) +
                        rows("bet365", "dc", ["1X", "12", "X2"], [1.16, 1.30, 2.30]))
    cands, preds = evaluate_match(m, odds, None, {}, Settings(), now=now)
    got = {(p["market"], p["selection"], p["point"]) for p in preds}
    assert {("dc", "1X", 0.0), ("btts", "Y", 0.0), ("ou", "O", 1.5), ("spread", "H", -1.5)} <= got
    dc = [p for p in preds if p["market"] == "dc" and p["selection"] == "1X"][0]
    assert dc["derived"] == 1 and dc["verified"] == 1 and dc["weak"] == 0
    # sin línea real de goles, los mercados de goles derivados quedan marcados como débiles (nunca Élite)
    odds2 = odds[~((odds["market"] == "ou"))]
    _, preds2 = evaluate_match(m, odds2, None, {}, Settings(), now=now)
    assert all(p["weak"] == 1 for p in preds2 if p["market"] in ("ou", "btts", "spread"))


# ------------------------------------------------------------------ estadísticas
def test_form_streak_venue_and_head_to_head():
    from radar.stats import match_stats, why_lines

    d = pd.to_datetime(["2026-10-01", "2026-10-08", "2026-10-15", "2026-10-22", "2026-10-29"])
    h = pd.DataFrame(dict(date=d, home=["A", "C", "A", "D", "A"], away=["B", "A", "E", "A", "F"],
                          hg=[2, 0, 3, 0, 1], ag=[0, 1, 1, 2, 0]))
    st = match_stats(h, "A", "B", "2026-11-01")
    assert st["local"]["forma"] == "VVVVV" and st["local"]["racha"] == "5V"
    assert st["local"]["local"]["v"] == 3 and st["local"]["visitante"]["v"] == 2
    assert st["h2h"]["v_local"] == 1
    lines = why_lines("Gana A", "1x2", "H", st)
    assert any("5 victorias seguidas" in x for x in lines) and any("Cara a cara" in x for x in lines)


# ------------------------------------------------------------------ combinadas y copiloto
def _preds(n=6):
    leagues = ["laliga", "premier", "seriea", "bundesliga", "ligue1", "eredivisie"]
    matches = {f"m{i}": dict(home=f"H{i}", away=f"A{i}", league=leagues[i], sport="football",
                             kickoff=f"2026-11-28T{12 + i}:00:00+00:00") for i in range(n)}
    base = [(0.82, 1.21), (0.78, 1.27), (0.74, 1.33), (0.70, 1.41), (0.62, 1.58), (0.58, 1.68)]
    preds = pd.DataFrame([dict(match_id=f"m{i}", market=mk, selection=sel, point=0.0, p_final=p, price=o, verified=1,
                               ev=p * o - 1, tier="pronostico", label=f"{sel} {i}", n_books=3, n_sharp=1, derived=0,
                               weak=0, b365_age_h=3.0)
                          for i, (p, o) in enumerate(base[:n]) for mk, sel in (("dc", "1X"),)])
    return preds, matches


def test_three_profiles_with_odds_windows():
    from radar.combos import build_combos

    preds, matches = _preds()
    combos = build_combos(preds, matches, Settings())
    seg = [c for c in combos if c["risk"] == "segura"]
    eq = [c for c in combos if c["risk"] == "equilibrada"]
    assert seg and all(1.70 <= c["odds"] <= 2.20 for c in seg)
    assert eq and all(2.50 <= c["odds"] <= 4.00 for c in eq)
    assert all(c["tier"] in ("valor", "solida", "sin_ventaja") for c in combos)
    assert not [c for c in combos if c["risk"] == "valor"]  # sin patas con ventaja no se inventa una


def test_combos_without_bet365_prices_are_por_confirmar():
    from radar.combos import build_combos

    preds, matches = _preds()
    preds = preds.assign(verified=0)
    combos = build_combos(preds, matches, Settings())
    assert combos and all(c["tier"] == "verificar" for c in combos)
    assert all(l["priced"] is False and l["min_odds"] > 1 for c in combos for l in c["legs"])


def test_copilot_respects_target_odds_and_sizes_stake():
    from radar import assistant

    now = datetime(2026, 11, 28, 8, tzinfo=timezone.utc)
    preds, matches = _preds()
    pool = preds.merge(pd.DataFrame([dict(match_id=k, **v) for k, v in matches.items()]), on="match_id")
    pool = pool.assign(odds=pool["price"], risk="bajo", confidence="—", why=[["Forma VVVVE"]] * len(pool))
    plan, alts, n = assistant.copilot(pool, 1000, now + timedelta(days=2), now, "medio", "combinada",
                                      odds_range=(2.5, 4.0))
    assert plan and plan.tipo == "combinada" and 2.5 <= plan.odds <= 4.0 and n == 6
    assert plan.budget == pytest.approx(5.0) and plan.stake_pct == 0.005  # sin ventaja: como mucho un 0,5 %
    assert all(l["why"] for l in plan.legs) and plan.rationale
    stake, pct, txt = assistant.stake_advice(0.05, 2.0, 1000)
    assert pct == pytest.approx(0.0125) and stake == pytest.approx(12.5) and "Kelly" in txt


# ------------------------------------------------------------------ liquidación de copas con ESPN
def test_cup_matches_settle_with_espn_results(tmp_path):
    from radar.db import DB
    from radar.learn import settle

    db = DB(tmp_path / "s.db")
    db.execute("INSERT INTO matches (id, league, kickoff, home, away, sport) VALUES (?,?,?,?,?,?)",
               ("champions|2026-11-25|Sabah|FC Barcelona", "champions", "2026-11-25T17:45:00+00:00", "Sabah",
                "FC Barcelona", "football"))
    db.execute("INSERT INTO picks (match_id, market, selection, point, odds, stake, tier, bookmaker, kickoff) "
               "VALUES (?,?,?,?,?,?,?,?,?)", ("champions|2026-11-25|Sabah|FC Barcelona", "dc", "X2", 0.0, 1.10, 1.0,
                                              "solida", "bet365", "2026-11-25T17:45:00+00:00"))
    scores = [dict(home_team="Sabah FK", away_team="Barcelona", completed=True,
                   scores=[{"name": "Sabah FK", "score": 1}, {"name": "Barcelona", "score": 4}])]
    n = settle(db, {"football": pd.DataFrame()}, datetime(2026, 11, 26, 12, tzinfo=timezone.utc), scores)
    assert n == 1
    assert db.df("SELECT won FROM picks")["won"].iloc[0] == 1
    db.close()


def test_level_summary_includes_combo_profiles():
    from radar.webdata import level_summary

    r = pd.DataFrame(dict(kickoff=["2026-11-27T12:00:00+00:00"] * 2, odds=[1.8, 1.4], won=[1, 0], stake=[1.0, 0.0],
                          profit=[0.8, 0.0], p_final=[0.58, 0.72], tier=["valor", "solida"]))
    rc = pd.DataFrame(dict(first_kickoff=["2026-11-27T12:00:00+00:00"], odds=[2.0], won=[1], stake=[0.0], profit=[0.0],
                           prob=[0.5], risk=["segura"], sports=["football"], created=["2026-11-27"]))
    t = level_summary(r, rc)
    assert list(t["nivel"]) == ["Élite", "Sólidas", "Combinada segura"]
    assert t.set_index("nivel").loc["Combinada segura", "yield"] == pytest.approx(1.0)
