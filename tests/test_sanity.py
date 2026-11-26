"""Pruebas de la limpieza de cuotas, el seguimiento inmutable y OddsPapi (v2).

Incluye una prueba de regresión con las 21 jugadas 'valor' reales de noviembre de 2026 (tests/data/casos_21_*.csv):
20 de ellas eran errores de datos (Betfair Exchange roto en football-data, cuotas de hace 14 días mezcladas con las
actuales y peso del modelo en más/menos) y la 21ª tenía bet365 dentro de su propio consenso. Ninguna debe volver a
salir como 'valor'."""
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from radar.config import Settings
from radar.db import DB
from radar.model.markets import book_market_ok, market_fair_probs
from radar.value import evaluate_match, kelly_stake

DATA = Path(__file__).parent / "data"
T0 = "2026-11-20T11:51:45+00:00"


def rows(book, market, prices, point=0.0, t=T0, source="x"):
    sels = {"1x2": "HDA", "ml": "HA", "ou": "OU"}[market]
    return [dict(bookmaker=book, market=market, selection=s, point=point, price=p, fetched_at=t, source=source)
            for s, p in zip(sels, prices)]


MATCH = dict(id="ligue2|2026-11-23|Nantes|Grenoble", sport="football", league="ligue2",
             kickoff="2026-11-23T19:45:00+00:00", home="Nantes", away="Grenoble")


# ------------------------------------------------------------------ 1. limpieza de cuotas
def test_rejects_impossible_margins():
    assert not book_market_ok("betfair_ex", [1.35, 1.16, 1.33])  # Nantes-Grenoble real: suma 2,35
    assert not book_market_ok("betfair_ex", [1.29, 6.5, 8.0])  # 1,05 es demasiado para un exchange
    assert book_market_ok("betfair_ex", [1.29, 7.2, 10.5])
    assert book_market_ok("bwin", [1.49, 4.1, 6.5]) and not book_market_ok("bwin", [1.0, 4.1, 6.5])


def test_bet365_never_in_its_own_consensus_and_sharp_only():
    odds = pd.DataFrame(rows("bet365", "1x2", [1.44, 3.7, 6.5]) + rows("bwin", "1x2", [1.49, 4.1, 6.5]) +
                        rows("pinnacle", "1x2", [1.52, 4.3, 7.2]))
    fair = market_fair_probs(odds, exclude={"bet365"})
    p, n, n_sharp = fair[("1x2", "H", 0.0)]
    assert n == 2 and n_sharp == 1
    inv = np.array([1 / 1.52, 1 / 4.3, 1 / 7.2])
    assert p == pytest.approx(0.643, abs=0.01) and p < inv[0]  # solo Pinnacle sin margen


def test_garbage_exchange_cannot_create_value():
    """Nantes-Grenoble (empate a 3,70): el exchange roto daba un 33 % al empate (+21 % de valor). Ya no."""
    odds = pd.DataFrame(rows("bet365", "1x2", [1.44, 3.7, 6.5]) + rows("bwin", "1x2", [1.49, 4.1, 6.5]) +
                        rows("betfair_ex", "1x2", [1.33, 1.16, 1.35]))
    cands, _ = evaluate_match(MATCH, odds, None, {}, Settings(), now=datetime(2026, 11, 20, 12, tzinfo=timezone.utc))
    assert not [c for c in cands if c.tier in ("valor", "revisar")]
    fair = market_fair_probs(odds, exclude={"bet365"})
    assert fair[("1x2", "D", 0.0)][1] == 1  # solo queda bwin


def test_stale_rows_are_not_mixed_with_fresh_ones():
    old = "2026-11-06T11:04:56+00:00"
    odds = pd.DataFrame(rows("bet365", "1x2", [2.15, 3.5, 3.4]) + rows("bwin", "1x2", [2.2, 3.4, 3.3]) +
                        rows("pinnacle", "1x2", [1.95, 3.6, 4.2], t=old) + rows("onexbet", "1x2", [1.98, 3.6, 4.1], t=old))
    ref = {("1x2", 0.0): pd.Timestamp(T0)}
    fair = market_fair_probs(odds, exclude={"bet365"}, ref_times=ref, max_gap_h=6)
    assert fair[("1x2", "H", 0.0)][1:] == (1, 0)  # Pinnacle y 1xBet de hace 14 días fuera
    # también al revés: un consenso más NUEVO que la cuota de bet365 no se compara con ella
    newer = "2026-11-22T10:00:00+00:00"
    odds2 = pd.DataFrame(rows("bet365", "1x2", [2.15, 3.5, 3.4]) + rows("pinnacle", "1x2", [1.95, 3.6, 4.2], t=newer))
    assert market_fair_probs(odds2, exclude={"bet365"}, ref_times=ref, max_gap_h=6) == {}


# ------------------------------------------------------------------ 3 y 4. valor, tope y frescura
def _clean(t=T0, b365=(2.30, 3.4, 3.3)):
    return pd.DataFrame(rows("bet365", "1x2", list(b365), t=t) + rows("pinnacle", "1x2", [2.05, 3.5, 3.9], t=t) +
                        rows("bwin", "1x2", [2.0, 3.4, 3.8], t=t) + rows("williamhill", "1x2", [2.0, 3.3, 3.7], t=t))


def test_value_when_everything_is_coherent():
    now = datetime(2026, 11, 20, 13, tzinfo=timezone.utc)
    cands, _ = evaluate_match(MATCH, _clean(), None, {}, Settings(), now=now)
    v = [c for c in cands if c.tier == "valor"]
    assert len(v) == 1 and v[0].selection == "H" and v[0].stake > 0 and v[0].n_sharp == 1


def test_bet365_and_pinnacle_from_the_same_call_are_enough():
    """Lo que trae OddsPapi: solo bet365 + Pinnacle, con la misma hora. Pinnacle basta como referencia."""
    now = datetime(2026, 11, 27, 12, tzinfo=timezone.utc)
    t = now.isoformat()
    nba = dict(id="nba|x", sport="basketball", league="nba", kickoff="2026-11-28T01:00:00+00:00", home="A", away="B")
    odds = pd.DataFrame(rows("bet365", "ml", [1.95, 1.90], t=t) + rows("pinnacle", "ml", [1.80, 2.10], t=t))
    cands, _ = evaluate_match(nba, odds, None, {}, Settings(), now=now)
    v = [c for c in cands if c.tier == "valor"]
    assert len(v) == 1 and v[0].selection == "H" and v[0].n_books == 1 and v[0].n_sharp == 1
    # con una sola casa blanda (no Pinnacle) NO basta
    odds2 = pd.DataFrame(rows("bet365", "ml", [1.95, 1.90], t=t) + rows("williamhill", "ml", [1.78, 2.02], t=t))
    assert not [c for c in evaluate_match(nba, odds2, None, {}, Settings(), now=now)[0] if c.tier == "valor"]


def test_huge_ev_goes_to_revisar_with_zero_stake():
    now = datetime(2026, 11, 20, 13, tzinfo=timezone.utc)
    cands, _ = evaluate_match(MATCH, _clean(b365=(3.20, 3.4, 2.2)), None, {}, Settings(), now=now)
    h = [c for c in cands if c.selection == "H"][0]
    assert h.ev > 0.15 and h.tier == "revisar" and h.stake == 0 and h.elite == 0
    assert kelly_stake(0.6, 2.87, Settings()) == 0  # +72 %: nunca se apuesta


def test_old_bet365_price_is_only_verificar():
    now = datetime(2026, 11, 22, 13, tzinfo=timezone.utc)  # 49 h después de la cuota de bet365
    cands, _ = evaluate_match(MATCH, _clean(), None, {}, Settings(), now=now)
    h = [c for c in cands if c.selection == "H"][0]
    assert h.tier == "verificar" and h.stake == 0 and "confírmala" in h.reasons


def test_incoherent_bet365_market_is_not_verified():
    now = datetime(2026, 11, 20, 13, tzinfo=timezone.utc)
    odds = _clean(b365=(2.30, 1.30, 1.40))  # margen de bet365 imposible
    cands, _ = evaluate_match(MATCH, odds, None, {}, Settings(), now=now)
    assert all(c.tier != "valor" and c.verified == 0 for c in cands)


def test_frozen_model_has_no_weight():
    now = datetime(2026, 11, 20, 13, tzinfo=timezone.utc)
    model = {("1x2", "H", 0.0): 0.10, ("1x2", "D", 0.0): 0.30, ("1x2", "A", 0.0): 0.60}
    _, preds = evaluate_match(MATCH, _clean(), model, {"football|1x2": 0.9}, Settings(), now=now)
    assert all(p["p_final"] == pytest.approx(p["p_market"]) for p in preds)


def test_recalibrate_frozen_and_prior_from_backtest(tmp_path):
    from radar.learn import recalibrate

    db = DB(tmp_path / "r.db")
    db.set("weights", {"football|ou": 0.236, "nba|ml": 0.5})
    db.set("weights_backtest", {"football|ou": 0.0})
    w = recalibrate(db, Settings())
    assert w["football|ou"] == 0.0
    db.close()


# ------------------------------------------------------------------ 2. almacenamiento
def _odds_frame(prices, src="the-odds-api", t=None, book="pinnacle"):
    d = pd.DataFrame(rows(book, "1x2", prices, t=t, source=src))
    return d.assign(match_id="m1", sport="football", league="laliga", kickoff="2026-11-22T20:00:00+00:00",
                    home="A", away="B")


def test_store_odds_purges_and_keeps_real_timestamp(tmp_path):
    from radar.pipeline import store_odds

    db = DB(tmp_path / "s.db")
    t1 = datetime(2026, 11, 20, 12, tzinfo=timezone.utc)
    store_odds(db, pd.concat([_odds_frame([2, 3.5, 4], t=t1.isoformat()),
                              _odds_frame([2.1, 3.4, 3.9], t=t1.isoformat(), book="onexbet")]), t1)
    # 9 h después la fuente ya no trae 1xBet y la cuota de Pinnacle no ha cambiado
    t2 = t1 + timedelta(hours=9)
    store_odds(db, _odds_frame([2, 3.5, 4], t=t1.isoformat()), t2)
    o = db.df("SELECT * FROM odds")
    assert set(o["bookmaker"]) == {"pinnacle"}
    assert (o["fetched_at"] == t1.isoformat()).all() and (o["updated"] == t2.isoformat()).all()
    db.close()


def test_football_data_timestamp_only_changes_with_the_file(tmp_path):
    from radar.pipeline import football_data_timestamp

    db = DB(tmp_path / "f.db")
    f = _odds_frame([2, 3.5, 4], src="football-data", book="bet365")
    t1 = datetime(2026, 11, 20, 12, tzinfo=timezone.utc)
    assert football_data_timestamp(db, f, t1) == t1.isoformat()
    assert football_data_timestamp(db, f, t1 + timedelta(hours=9)) == t1.isoformat()
    t3 = t1 + timedelta(days=3)
    assert football_data_timestamp(db, f.assign(price=f["price"] + 0.05), t3) == t3.isoformat()
    db.close()


def test_odds_api_only_stores_the_lookahead_window(tmp_path, monkeypatch):
    from radar.sources.odds_api import OddsAPI

    now = datetime(2026, 11, 20, 12, tzinfo=timezone.utc)

    def ev(days):
        return {"commence_time": (now + timedelta(days=days)).isoformat().replace("+00:00", "Z"), "home_team": "A",
                "away_team": f"B{days}", "bookmakers": [{"key": "pinnacle", "last_update": now.isoformat(),
                                                        "markets": [{"key": "h2h", "outcomes": [
                                                            {"name": "A", "price": 2.0}, {"name": f"B{days}", "price": 3.6},
                                                            {"name": "Draw", "price": 3.5}]}]}]}

    db = DB(tmp_path / "o.db")
    api = OddsAPI(Settings(odds_api_key="k", sports="football"), db)
    monkeypatch.setattr(api, "plan", lambda: [("champions", "soccer_uefa_champs_league", "h2h")])
    monkeypatch.setattr(api, "upcoming_events", lambda key, now: [ev(1)])
    monkeypatch.setattr(api, "_get", lambda path, **kw: [ev(1), ev(14)])
    out = api.fetch(now)
    assert set(out["away"]) == {"B1"} and out["fetched_at"].notna().all()
    db.close()


# ------------------------------------------------------------------ 5. seguimiento inmutable y CLV
def test_valor_picks_are_frozen(tmp_path):
    from radar.pipeline import save_picks

    db = DB(tmp_path / "p.db")
    s = Settings()
    now = datetime(2026, 11, 20, 13, tzinfo=timezone.utc)
    cands, _ = evaluate_match(MATCH, _clean(), None, {}, s, now=now)
    save_picks(db, cands, now.isoformat(), now)
    first = db.df("SELECT * FROM picks WHERE tier='valor'").iloc[0]
    assert first["locked"] == 1 and first["shown_at"] == now.isoformat()
    # la cuota baja y deja de haber valor: la jugada NO se borra ni se reescribe
    later = now + timedelta(hours=6)
    cands2, _ = evaluate_match(MATCH, _clean(t=later.isoformat(), b365=(2.0, 3.4, 3.6)), None, {}, s, now=later)
    save_picks(db, cands2, later.isoformat(), later)
    p = db.df(f"SELECT * FROM picks WHERE id={first['id']}").iloc[0]
    assert p["odds"] == 2.30 and p["tier"] == "valor" and p["stake"] == first["stake"]
    assert p["current_odds"] == 2.0 and p["active"] == 0
    db.close()


def test_clv_only_against_pinnacle_close():
    from radar.learn import _closing_prob
    from radar.sources.football_data import tidy_history

    raw = pd.DataFrame({"Div": ["SP1"], "Date": ["22/11/2026"], "HomeTeam": ["A"], "AwayTeam": ["B"], "FTHG": [1],
                        "FTAG": [0], "B365H": [2.1], "B365D": [3.3], "B365A": [3.6], "B365CH": [2.0], "B365CD": [3.4],
                        "B365CA": [3.8]}).assign(league="laliga", season="2627")
    h = tidy_history(raw)
    assert pd.isna(h["open_H"].iloc[0]) and pd.isna(h["close_H"].iloc[0])  # bet365 nunca es la referencia
    assert _closing_prob(h.iloc[0], "1x2", "H", 0.0) is None
    h2 = tidy_history(raw.assign(PSCH=1.95, PSCD=3.5, PSCA=4.1))
    assert 0.45 < _closing_prob(h2.iloc[0], "1x2", "H", 0.0) < 0.52


# ------------------------------------------------------------------ 6. OddsPapi
class FakeResp:
    def __init__(self, data):
        self.data, self.status_code = data, 200

    def json(self):
        return self.data

    def raise_for_status(self):
        pass


def test_oddspapi_one_fixture_call_per_sport_and_both_books(tmp_path, monkeypatch):
    from radar.sources import oddspapi

    now = datetime(2026, 11, 27, 12, tzinfo=timezone.utc)
    calls = []
    fixtures = [dict(fixtureId=f"id{i}", participant1Name=h, participant2Name=a,
                     startTime=(now + timedelta(hours=10 + i)).isoformat().replace("+00:00", "Z"),
                     tournamentName="NFL", categoryName="USA")
                for i, (h, a) in enumerate([("Dallas Cowboys", "New York Giants"), ("Miami Dolphins", "Buffalo Bills"),
                                            ("Chicago Bears", "Detroit Lions")])]
    markets = [{"marketId": 141, "playerProp": False, "period": "result", "marketType": "moneyline", "handicap": 0,
                "outcomes": [{"outcomeId": 141, "outcomeName": "1"}, {"outcomeId": 142, "outcomeName": "2"}]}]
    odds = {"bookmakerOdds": {
        "bet365": {"markets": {"141": {"outcomes": {"141": {"players": {"0": {"price": 1.80}}},
                                                    "142": {"players": {"0": {"price": 2.05}}}}}}},
        "pinnacle": {"markets": {"141": {"outcomes": {"141": {"players": {"0": {"price": 1.76}}},
                                                      "142": {"players": {"0": {"price": 2.18}}}}}}}}}

    def fake_get(url, params, timeout):
        calls.append((url.rsplit("/", 1)[1], dict(params)))
        return FakeResp({"fixtures": fixtures, "markets": markets, "odds": odds}[url.rsplit("/", 1)[1]])

    monkeypatch.setattr(oddspapi.requests, "get", fake_get)
    monkeypatch.setattr(oddspapi.time, "sleep", lambda x: None)
    db = DB(tmp_path / "op.db")
    op = oddspapi.OddsPapi(Settings(oddspapi_key="k"), db)
    got = []
    for f in fixtures:  # nuestro partido con local y visitante al revés en el último
        h, a = f["participant1Name"], f["participant2Name"]
        if f["fixtureId"] == "id2":
            h, a = a, h
        got.append(op.match_odds(dict(league="nfl", sport="nfl", kickoff=f["startTime"].replace("Z", "+00:00"),
                                      home=h, away=a), now))
    names = [c[0] for c in calls]
    assert names.count("fixtures") == 1 and names.count("odds") == 3 and "tournaments" not in names
    fx = [p for n, p in calls if n == "fixtures"][0]
    assert fx["sportId"] == 14 and fx["statusId"] == 0 and fx["bookmakers"] == "bet365"
    span = pd.Timestamp(fx["to"]) - pd.Timestamp(fx["from"])
    assert span < pd.Timedelta(hours=48)
    assert all(p["bookmakers"] == "bet365,pinnacle" for n, p in calls if n == "odds")
    r0 = {(r["bookmaker"], r["selection"]): r["price"] for r in got[0]}
    assert r0 == {("bet365", "H"): 1.80, ("bet365", "A"): 2.05, ("pinnacle", "H"): 1.76, ("pinnacle", "A"): 2.18}
    r2 = {(r["bookmaker"], r["selection"]): r["price"] for r in got[2]}
    assert r2[("bet365", "A")] == 1.80  # invertido: el local de OddsPapi es nuestro visitante
    assert len({r["fetched_at"] for r in got[0]}) == 1  # las dos casas con la misma hora
    db.close()


# ------------------------------------------------------------------ regresión: las 21 jugadas reales
def _replay(settings: Settings) -> pd.DataFrame:
    """Vuelve a evaluar cada una de las 21 jugadas con el código nuevo, en el momento de la última ejecución antes del
    partido y con el peso del modelo que había entonces (0,236 en más/menos). Por cada selección devuelve el nivel,
    la probabilidad limpia, el valor esperado contra la cuota de bet365 de entonces y las casas válidas que quedan."""
    picks = pd.read_csv(DATA / "casos_21_picks.csv")
    odds = pd.read_csv(DATA / "casos_21_odds.csv")
    model = pd.read_csv(DATA / "casos_21_model.csv")
    out = []
    for r in picks.itertuples():
        m = dict(id=r.match_id, sport=r.sport, league=r.league, kickoff=r.kickoff, home=r.home, away=r.away)
        probs = {(x.market, x.selection, float(x.point)): x.p_model
                 for x in model[model.match_id == r.match_id].itertuples()}
        o = odds[odds.match_id == r.match_id]
        cands, preds = evaluate_match(m, o, probs, {"football|ou": 0.236, "football|1x2": 0.0}, settings,
                                      now=datetime.fromisoformat(r.updated))
        key = (r.market, r.selection, float(r.point))
        cand = [c for c in cands if (c.market, c.selection, float(c.point)) == key]
        pred = [x for x in preds if (x["market"], x["selection"], float(x["point"])) == key]
        b365 = o[(o.bookmaker == "bet365") & (o.market == r.market) & (o.point == r.point)]
        fair = market_fair_probs(o, exclude={"bet365"}, ref_times={(r.market, float(r.point)): pd.Timestamp(
            b365.fetched_at.max())}, max_gap_h=settings.max_odds_gap_h, max_overround=settings.max_overround)
        n_books, n_sharp = (fair[key][1], fair[key][2]) if key in fair else (0, 0)
        p = pred[0]["p_final"] if pred else None
        tier = cand[0].tier if cand else ("sin consenso válido" if not pred else "pronostico")
        if tier == "pronostico" and p is not None and n_books < settings.min_books:
            tier = "pocas casas válidas"
        out.append(dict(id=r.id, tier=tier, p_clean=p, ev=(p * r.odds - 1) if p is not None else None,
                        n_books=n_books, n_sharp=n_sharp, any_valor=any(c.tier == "valor" for c in cands)))
    return pd.DataFrame(out)


@pytest.mark.skipif(not (DATA / "casos_21_picks.csv").exists(), reason="sin casos históricos")
def test_the_21_real_picks_no_longer_slip_through():
    # 1) reglas completas: ninguna de las 21 sale como 'valor' y no hay ni una sola jugada 'valor' en esos partidos
    full = _replay(Settings())
    assert not (full["tier"] == "valor").any() and not full["any_valor"].any(), full
    # 2) incluso relajando las dos reglas que podrían "salvarlas por falta de datos" (antigüedad de bet365 y mínimo de
    #    casas), ninguna de las 21 vuelve a ser 'valor': con el consenso limpio todas tienen valor esperado < +3 %
    lax = _replay(Settings(b365_max_age_h=1e9, min_books=1))
    assert not lax["any_valor"].any() and not (lax["tier"] == "valor").any(), lax
    lx = lax.set_index("id")
    assert (lx["ev"] < 0.03).all(), lax
    # las 4 jugadas con +21 % a +95 % de 'valor' (Betfair Exchange roto) tienen ahora valor esperado negativo
    assert (lx.loc[[134, 588, 591, 625], "ev"] < 0).all()
    # las 11 de más/menos solo tenían valor por el peso del modelo: con el modelo congelado, todas negativas
    ou = [419, 430, 434, 448, 455, 482, 491, 492, 523, 537, 538]
    assert (lx.loc[ou, "ev"] < 0).all()
    # se evalúan todas de verdad (hay consenso limpio), ninguna se salva por falta de datos
    assert lx["p_clean"].notna().all()
