"""Fase 5: niveles Valor moderado y Cuota alta, bajas reales (API-Football y ESPN) y confianza del velocímetro."""
from datetime import date, datetime, timezone

import pandas as pd
import pytest

from radar.config import Settings
from radar.value import evaluate_match

NOW = datetime(2026, 11, 27, 12, tzinfo=timezone.utc)
T = NOW.isoformat()


def rows(book, prices, market="1x2", sels="HDA"):
    return [dict(bookmaker=book, market=market, selection=x, point=0.0, price=p, fetched_at=T, source="x")
            for x, p in zip(sels, prices)]


M = dict(id="laliga|m", sport="football", league="laliga", kickoff="2026-11-28T20:00:00+00:00", home="A", away="B")


def tier_of(cands, sel):
    return [c for c in cands if c.selection == sel][0].tier


def test_moderate_value_tier():
    # local: justo ≈ 2,13 (Pinnacle sin margen); bet365 2,16 → ventaja ligera (+1,4 %) → moderado
    odds = pd.DataFrame(rows("pinnacle", [2.10, 3.40, 3.80]) + rows("bet365", [2.16, 3.20, 3.40]))
    cands, _ = evaluate_match(M, odds, None, {}, Settings(), now=NOW)
    h = [c for c in cands if c.selection == "H"][0]
    assert h.tier == "moderado" and 0 <= h.ev < Settings().min_ev and h.stake > 0


def test_high_odds_tier():
    # visitante: prob. justa ≈ 30 %; bet365 3,45 → +3,8 % a cuota alta (fuera del rango élite 1,30-2,60) → cuota_alta
    odds = pd.DataFrame(rows("pinnacle", [2.20, 3.50, 3.10]) + rows("bet365", [2.05, 3.30, 3.45]))
    cands, _ = evaluate_match(M, odds, None, {}, Settings(), now=NOW)
    a = [c for c in cands if c.selection == "A"][0]
    assert a.tier == "cuota_alta" and 2.40 <= a.odds <= 3.80 and a.ev >= 0.01 and a.elite == 0


def test_no_value_stays_out_of_the_four_levels():
    odds = pd.DataFrame(rows("pinnacle", [2.10, 3.40, 3.80]) + rows("bet365", [2.00, 3.20, 3.40]))
    cands, _ = evaluate_match(M, odds, None, {}, Settings(), now=NOW)
    assert all(c.tier not in ("valor", "solida", "moderado", "cuota_alta") for c in cands)


class R:
    def __init__(self, data, code=200):
        self.data, self.status_code = data, code

    def json(self):
        return self.data

    def raise_for_status(self):
        pass


def test_apifootball_by_date_one_call_per_day_cached_and_honest(tmp_path, monkeypatch):
    from radar.db import DB
    from radar.sources import injuries

    db = DB(tmp_path / "i.db")
    s = Settings(apifootball_key="")
    items, st = injuries.apifootball_by_date(s, db, {"2026-11-28"}, date(2026, 11, 27))
    assert items == [] and "falta el secreto APIFOOTBALL_KEY" in st
    calls = []

    def get(url, headers=None, params=None, timeout=None):
        calls.append(params)
        return R({"errors": [], "response": [{"league": {"id": 140}, "team": {"name": "Real Madrid"},
                                              "player": {"name": "Vinícius Júnior", "type": "Missing Fixture",
                                                         "reason": "Hamstring Injury"}, "fixture": {"id": 1}}]})

    monkeypatch.setattr(injuries.requests, "get", get)
    s = Settings(apifootball_key="k")
    items, st = injuries.apifootball_by_date(s, db, {"2026-11-28", "2026-11-29"}, date(2026, 11, 27))
    assert st == "ok" and len(calls) == 2 and all(set(c) == {"date"} for c in calls)
    injuries.apifootball_by_date(s, db, {"2026-11-28"}, date(2026, 11, 27))
    assert len(calls) == 2  # caché
    matches = [dict(league="laliga", sport="football", home="Real Madrid", away="Getafe", kickoff="2026-11-28T20:00:00+00:00")]
    out, status = injuries.collect(s, db, matches, {}, date(2026, 11, 27))
    assert out[("laliga", "Real Madrid")][0]["jugador"] == "Vinícius Júnior" and status["laliga"][0] == "API-Football"
    # plan sin la temporada actual → se avisa y no se insiste
    monkeypatch.setattr(injuries.requests, "get", lambda *a, **k: R({"errors": {"plan": "Free plans do not have access to this season"}}))
    db.execute("DELETE FROM kv WHERE key LIKE 'apif_inj_%'")
    items, st = injuries.apifootball_by_date(s, db, {"2026-11-30"}, date(2026, 11, 27))
    assert "no cubre la temporada actual" in st
    db.close()


def test_espn_injuries_parsing(monkeypatch):
    from radar.sources import injuries

    data = {"injuries": [{"displayName": "Atlanta Hawks", "injuries": [
        {"status": "Out", "athlete": {"displayName": "Jock Landale"}, "shortComment": "won't play",
         "details": {"type": "Hip", "detail": "Inflammation", "returnDate": "2026-11-27"}}]}]}
    monkeypatch.setattr(injuries.requests, "get", lambda *a, **k: R(data))
    inj = injuries.espn_injuries("basketball/nba")
    a = inj["Atlanta Hawks"][0]
    assert a["jugador"] == "Jock Landale" and a["tipo"] == "baja" and a["motivo"] == "Hip Inflammation"


def test_injuries_reach_the_scouting_report():
    from radar.stats import scouting

    info = {"bajas": {"local": [{"jugador": "Vinícius Júnior", "tipo": "baja", "motivo": "Hamstring Injury"}],
                      "visit": [], "fuente": "API-Football", "nota": "ok"}}
    txt = " ".join(scouting({}, "Real Madrid", "Getafe", "football", "1x2", "H", info=info))
    assert "Bajas (API-Football): Real Madrid: Vinícius Júnior (baja: Hamstring Injury); Getafe: sin bajas" in txt
    none = " ".join(scouting({}, "A", "B", "football", "1x2", "H",
                             info={"bajas": {"local": [], "visit": [], "fuente": None, "nota": "sin fuente de bajas"}}))
    assert "sin datos para este partido" in none


def test_confidence_score_orders_sensibly():
    from radar.webdata import confidence_score

    strong = confidence_score(0.70, 0.04, 4, 1, True, 2)
    weak = confidence_score(0.45, -0.03, 1, 0, False, None, derived=True)
    assert 0 <= weak < 40 < 65 <= strong <= 100
