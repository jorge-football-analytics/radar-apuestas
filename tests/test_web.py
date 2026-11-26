"""Capa de datos de la web (radar/webdata.py): rendimiento, desgloses, radar del día y METRICS_SINCE."""
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from radar import webdata


def resolved():
    base = datetime(2026, 11, 27, 18, tzinfo=timezone.utc)
    rows = [  # odds, won, stake, tier, sport, market
        (1.70, 1, 1.0, "valor", "football", "1x2"), (2.10, 0, 1.0, "valor", "football", "ou"),
        (1.45, 1, 0.0, "solida", "basketball", "ml"), (1.40, 1, 0.0, "solida", "football", "1x2"),
        (2.40, 0, 0.5, "valor", "tennis", "ml")]
    return pd.DataFrame([dict(kickoff=(base + timedelta(hours=i)).isoformat(), odds=o, won=w, stake=st,
                              profit=st * (o - 1) if w else -st, tier=t, sport=sp, market=mk, league="laliga",
                              p_final=1 / o, clv=0.01 if i % 2 else None, created=(base - timedelta(days=i)).isoformat(),
                              shown_at=(base - timedelta(days=i)).isoformat())
                         for i, (o, w, st, t, sp, mk) in enumerate(rows)])


def test_perf_series_units_and_stakes():
    r = resolved()
    u = webdata.perf_series(r, 100, "unit")
    assert u["profit"].iloc[-1] == pytest.approx(0.70 - 1 + 0.45 + 0.40 - 1)
    assert u["yield"].iloc[-1] == pytest.approx(u["profit"].iloc[-1] / 5)
    assert (u["drawdown"] <= 0).all()
    st = webdata.perf_series(r, 100, "stake")
    assert st["bank"].iloc[-1] == pytest.approx(100 + 0.70 - 1 - 0.5)


def test_kpis_and_breakdowns():
    r = resolved()
    k = webdata.kpis(r)
    assert k["n"] == 5 and k["hit"] == pytest.approx(0.6) and k["clv_n"] == 2
    b = webdata.breakdown(r, "band")
    assert list(b["grupo"]) == ["1,30-1,60", "1,60-2,00", "2,00-2,60"]
    assert b["n"].sum() == 5
    assert set(webdata.breakdown(r, "sport")["grupo"]) == {"football", "basketball", "tennis"}
    assert webdata.kpis(r.iloc[0:0]) == {"n": 0}


def test_metrics_since_env(monkeypatch):
    monkeypatch.setenv("METRICS_SINCE", "2026-11-27")
    assert webdata.metrics_since_default() == "2026-11-27"
    monkeypatch.setenv("METRICS_SINCE", "no es una fecha")
    assert webdata.metrics_since_default() == ""


def test_radar_table_filters():
    now = datetime(2026, 11, 27, 12, tzinfo=timezone.utc)
    A = pd.DataFrame([dict(match_id=f"m{i}", sport=sp, league="laliga", home=h, away="X", light=l,
                           kickoff=(now + timedelta(hours=hh)).isoformat(), tier="pronostico")
                      for i, (sp, h, l, hh) in enumerate([("football", "Real Madrid", "verde", 3),
                                                          ("tennis", "Alcaraz", "rojo", 5),
                                                          ("football", "Betis", "ambar", 30)])])
    assert len(webdata.radar_table(A, "football")) == 2
    assert len(webdata.radar_table(A, None, ["verde", "ambar"])) == 2
    assert len(webdata.radar_table(A, None, None, hours=12, now=now)) == 2
    assert list(webdata.radar_table(A, query="madrid")["home"]) == ["Real Madrid"]
    sm = webdata.day_summary(A)
    assert sm["partidos"] == 3 and sm["verde"] == 1 and sm["rojo"] == 1


def test_load_dashboard_end_to_end(tmp_path):
    """Base de datos real del pipeline sintético → todo lo que pinta la web, sin errores."""
    from radar.db import DB
    from radar.pipeline import Ctx, analyze, save_analysis, save_picks, store_odds
    from radar.config import Settings

    s = Settings(db_path=tmp_path / "w.db")
    db = DB(s.db_path)
    now = datetime(2026, 11, 27, 12, tzinfo=timezone.utc)
    ko = (now + timedelta(hours=6)).isoformat()
    rows = []
    for book, pr in (("pinnacle", [1.40, 4.9, 9.0]), ("bet365", [1.43, 4.5, 7.5])):
        rows += [dict(match_id="laliga|x|A|B", sport="football", league="laliga", kickoff=ko, home="A", away="B",
                      source="oddspapi", bookmaker=book, market="1x2", selection=sel, point=0.0, price=p,
                      fetched_at=now.isoformat()) for sel, p in zip("HDA", pr)]
    store_odds(db, pd.DataFrame(rows), now)
    cands, preds, matches, _ = analyze(s, db, Ctx(), now, {}, {})
    save_picks(db, cands, now.isoformat(), now)
    save_analysis(db, preds, matches, now)
    db.close()
    d = webdata.load_dashboard(s.db_path, now=now)
    assert len(d["analysis"]) == 1 and d["analysis"]["tier"].iloc[0] == "solida"
    assert (d["picks"]["tier"] == "solida").any() and d["picks"]["locked"].iloc[0] == 1
    assert set(d["metrics"]) == {"valor", "solida"}
