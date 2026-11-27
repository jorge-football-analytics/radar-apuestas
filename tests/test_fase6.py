"""Fase 6: cuota de referencia operable cuando bet365 no tiene cuota, recomendaciones del día, tres propuestas del
asistente dentro del rango, tabla de la jornada sin textos rotos y presupuesto de The Odds API."""
from datetime import datetime, timedelta, timezone

import pandas as pd

from radar import assistant, webdata
from radar.config import Settings, book_name
from radar.value import evaluate_match, ref_price

NOW = datetime(2026, 11, 27, 12, tzinfo=timezone.utc)
T = NOW.isoformat()
M = dict(id="nba|x", sport="basketball", league="nba", kickoff="2026-11-28T01:00:00+00:00", home="Boston Celtics",
         away="Atlanta Hawks")


def rows(book, prices, market="ml", sels="HA", point=0.0, t=T):
    return [dict(bookmaker=book, market=market, selection=x, point=point, price=p, fetched_at=t, source="x")
            for x, p in zip(sels, prices)]


def test_ref_price_best_real_book_with_outlier_guard_and_no_offshore():
    s = Settings()
    o = pd.DataFrame(rows("williamhill", [1.62, 2.40]) + rows("unibet_se", [1.66, 2.30]) +
                     rows("gtbets", [1.90, 2.20]) + rows("pinnacle", [1.70, 2.30]))
    r = ref_price(o, "ml", "H", 0.0, s, pd.Timestamp(NOW))
    assert r["book"] == "unibet_se" and r["price"] == 1.66  # ni Pinnacle (referencia) ni la offshore
    # una cuota suelta un 4 % por encima de la segunda no se usa
    o2 = pd.DataFrame(rows("williamhill", [1.62, 2.40]) + rows("betsson", [1.80, 2.10]))
    assert ref_price(o2, "ml", "H", 0.0, s, pd.Timestamp(NOW))["price"] == 1.62
    # cuotas viejas (más de REF_MAX_AGE_H) no cuentan
    old = (NOW - timedelta(hours=30)).isoformat()
    o3 = pd.DataFrame(rows("williamhill", [1.62, 2.40], t=old))
    assert ref_price(o3, "ml", "H", 0.0, s, pd.Timestamp(NOW)) is None


def test_without_bet365_the_engine_uses_the_market_reference_and_names_the_book():
    s = Settings()
    # Pinnacle: 1,55 / 2,60 → local justo ≈ 1,61; William Hill paga 1,68 → +4 % de valor con referencia sharp
    o = pd.DataFrame(rows("pinnacle", [1.55, 2.60]) + rows("williamhill", [1.68, 2.25]) + rows("unibet_se", [1.60, 2.35])
                     + rows("betsson", [1.62, 2.30]))
    cands, preds = evaluate_match(M, o, None, {}, s, now=NOW, context={"b365_ratio": {"basketball": 0.99}})
    h = [c for c in cands if c.selection == "H"][0]
    assert h.price_src == "mercado" and h.ref_book == "williamhill" and h.bookmaker == "williamhill"
    assert h.tier == "valor" and h.verified == 0 and h.elite == 0  # Telegram solo con bet365 confirmada
    assert "Cuota ref. mercado 1.68 (William Hill) · bet365 ~1.60" in h.reasons
    assert h.min_odds * h.p_final >= 1 + s.min_ev - 1e-9  # cuota mínima en bet365 con la ventaja mínima
    pr = [p for p in preds if p["selection"] == "H"][0]
    assert pr["price_src"] == "mercado" and pr["ref_book"] == "williamhill" and pr["b365_est"] == 1.6


def test_fresh_bet365_always_wins_over_other_books():
    o = pd.DataFrame(rows("pinnacle", [1.55, 2.60]) + rows("williamhill", [1.68, 2.25]) + rows("bet365", [1.57, 2.40]))
    cands, _ = evaluate_match(M, o, None, {}, Settings(), now=NOW)
    assert all(c.price_src == "bet365" for c in cands if c.selection == "H")


def test_daily_picks_volume_order_and_never_negative():
    recs = []
    for i in range(14):
        tier = ["valor", "solida", "moderado", "cuota_alta", "pronostico"][i % 5]
        p = 0.70 - 0.01 * i
        price = round((1.03 if tier != "pronostico" else 0.97) / p, 2)
        recs.append(dict(match_id=f"m{i}", sport=["football", "basketball", "nfl"][i % 3], league="laliga",
                         kickoff=(NOW + timedelta(hours=3 + i)).isoformat(), home=f"L{i}", away=f"V{i}", market="1x2",
                         selection="H", point=0.0, label=f"Gana L{i}", p_final=p, odds=price, ev=p * price - 1,
                         tier=tier, priced=True, n_books=5, n_sharp=1, b365_age_h=None, derived=0,
                         min_odds=round(1 / p, 2)))
    dp = webdata.daily_picks(pd.DataFrame(recs), NOW)
    assert 8 <= len(dp) <= 15 and dp["match_id"].is_unique
    assert (dp["p_final"] * dp["odds"] - 1 >= -1e-9).all()  # a la cuota indicada, nunca valor negativo
    order = dp["rank_tier"].tolist()
    assert order == sorted(order)  # élite primero, luego cuota alta, sólidas, moderado y al final cuota objetivo
    assert set(dp["sport"]) == {"football", "basketball", "nfl"}


def test_assistant_three_distinct_proposals_inside_the_range():
    P = []
    for i, (p, o) in enumerate([(0.62, 1.66), (0.70, 1.45), (0.55, 1.85), (0.66, 1.55), (0.58, 1.75), (0.75, 1.36)]):
        P.append(dict(match_id=f"m{i}", sport="football", league=f"lg{i}", kickoff=(NOW + timedelta(hours=5 + i)).isoformat(),
                      home=f"L{i}", away=f"V{i}", label=f"Gana L{i}", market="1x2", odds=o, p_final=p, ev=p * o - 1,
                      tier="solida", verified=1, priced=True, risk="bajo", confidence="—", why=["x"]))
    plan, alts, _ = assistant.copilot(pd.DataFrame(P), 200, NOW + timedelta(days=2), NOW, risk="medio")
    allp = [plan, *alts]
    assert [p.role for p in allp] == ["equilibrada", "conservadora", "retorno"]
    assert all(p.in_range and 1.90 <= p.odds <= 2.80 and p.ev >= -1e-9 for p in allp)
    assert len({tuple(sorted(l["label"] for l in p.legs)) for p in allp}) == 3
    by = {p.role: p for p in allp}
    assert by["conservadora"].prob >= by["equilibrada"].prob and by["retorno"].odds >= by["equilibrada"].odds
    assert all(p.budget >= 0 for p in allp)


def test_negative_priced_legs_become_target_odds_never_negative():
    P = pd.DataFrame([dict(match_id="a", sport="nba", league="nba", kickoff=(NOW + timedelta(hours=5)).isoformat(),
                           home="A", away="B", label="Gana A", market="ml", odds=1.40, p_final=0.70, ev=0.70 * 1.40 - 1,
                           tier="pronostico", verified=0, priced=True, ref_book="unibet_se", risk="bajo", confidence="—",
                           why=["x"])])
    plan, _, _ = assistant.copilot(P, 100, NOW + timedelta(days=2), NOW, risk="bajo", odds_range=(1.40, 1.60))
    leg = plan.legs[0]
    assert leg["priced"] is False and leg["odds"] == 1.43 and leg["cur_odds"] == 1.40 and leg["cur_book"] == "unibet_se"
    assert plan.ev >= 0


def test_day_table_has_no_broken_values():
    A = pd.DataFrame([
        dict(match_id="1", sport="football", league="laliga", kickoff="2026-11-28T20:00:00+00:00", home="A", away="B",
             label="Gana A", p_final=0.55, price=1.90, price_src="mercado", ref_book="williamhill", ev=0.045, tier="valor",
             light="verde", recommendation="Élite", n_books=6, n_sharp=1, b365_age_h=None),
        dict(match_id="2", sport="football", league="mexico", kickoff="2026-11-28T21:00:00+00:00", home="Pachuca",
             away="Guadalajara", label=float("nan"), p_final=float("nan"), price=None, price_src=None, ref_book=None,
             ev=None, tier="sin_datos", light="gris", recommendation=None, n_books=None, n_sharp=None, b365_age_h=None)])
    T = webdata.day_table(A, NOW)
    assert len(T) == 1 and T.iloc[0]["Casa"] == "William Hill" and T.iloc[0]["Cuota"] == "1,90"
    flat = " ".join(str(v) for v in T.drop(columns=["_kickoff"]).to_numpy().ravel())
    assert "nan" not in flat.lower() and "None" not in flat


def test_combo_of_day_prefers_value_then_solid():
    C = pd.DataFrame([dict(tier="verificar", ev=0.0, prob=0.6, odds=1.7), dict(tier="solida", ev=-0.004, prob=0.5, odds=2.0),
                      dict(tier="valor", ev=0.03, prob=0.4, odds=2.6)])
    assert webdata.combo_of_day(C)["tier"] == "valor"
    assert webdata.combo_of_day(C[C["tier"] != "valor"])["tier"] == "solida"
    assert webdata.combo_of_day(pd.DataFrame()) is None


def test_book_names():
    assert book_name("williamhill") == "William Hill" and book_name("espn_draftkings") == "DraftKings"
    assert book_name("unibet_se") == "Unibet" and book_name("nueva_casa") == "Nueva"


def test_odds_api_spends_where_it_matters(tmp_path, monkeypatch):
    """Ligas que juegan en las próximas 24 h: cada 6 h; el resto, una vez al día. El fútbol entra por defecto."""
    from radar.db import DB
    from radar.sources import odds_api

    s = Settings(odds_api_key="k", db_path=tmp_path / "o.db", sports="football,basketball")
    assert s.odds_api_football
    db = DB(s.db_path)
    db.set("odds_api_remaining", 400)
    api = odds_api.OddsAPI(s, db)
    soon = (NOW + timedelta(hours=5)).isoformat().replace("+00:00", "Z")
    later = (NOW + timedelta(hours=40)).isoformat().replace("+00:00", "Z")
    events = {"basketball_nba": [{"commence_time": soon}], "soccer_spain_la_liga": [{"commence_time": later}]}
    monkeypatch.setattr(api, "plan", lambda: [("nba", "basketball_nba", "h2h,spreads,totals"),
                                              ("laliga", "soccer_spain_la_liga", "h2h,totals")])
    monkeypatch.setattr(api, "upcoming_events", lambda key, now: events.get(key, []))
    called = []
    monkeypatch.setattr(api, "_get", lambda path, **k: called.append(path) or [])
    db.set("odds_refresh_nba", (NOW - timedelta(hours=7)).isoformat())
    db.set("odds_refresh_laliga", (NOW - timedelta(hours=7)).isoformat())
    api.fetch(NOW)
    assert called == ["/sports/basketball_nba/odds"]  # LaLiga juega dentro de 40 h y se pidió hace 7 h: espera
    db.close()


def test_broken_text_detector():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    from check_pages import broken_text
    assert broken_text("<b>nan</b> <p>Equipo None</p>")
    assert not broken_text("<p>Nantes · Nancy · Nonesuch</p><style>.nan{}</style>")
