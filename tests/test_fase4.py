"""Fase 4: informes de scouting en combinadas y asistente, temporada real en las estadísticas y firmas que usa app.py."""
import inspect

import pandas as pd
import pytest


def test_app_calls_match_module_signatures():
    """Las dos llamadas que daban AttributeError en Streamlit Cloud existen con los parámetros que usa app.py."""
    from radar import assistant, webdata

    sig = inspect.signature(assistant.copilot)
    for name in ("pool", "bankroll", "deadline", "now", "risk", "tipo", "sports", "odds_range", "only_good",
                 "kelly_fraction", "max_pct"):
        assert name in sig.parameters
    assert list(inspect.signature(webdata.level_summary).parameters)[:3] == ["resolved", "resolved_combos", "bankroll"]
    src = open("app.py", encoding="utf-8").read()
    assert "assistant.copilot(D[\"pool\"], bankroll, deadline, now, risk, tipo, sports," in src
    assert "webdata.level_summary(R, RC_all, s.bankroll)" in src
    assert "_reload_radar()" in src  # recarga de módulos antiguos en memoria (causa del AttributeError)


def test_season_window_ignores_previous_season():
    from radar.stats import team_form

    old = pd.date_range("2026-01-01", periods=30, freq="3D")
    new = pd.date_range("2026-10-25", periods=10, freq="3D")
    h = pd.DataFrame(dict(date=list(old) + list(new), home="A", away="B", hg=[100] * 40, ag=[90] * 30 + [120] * 10))
    f = team_form(h, "A", draws=False)
    assert f["temporada"]["pj"] == 10 and f["local"]["v"] == 0 and f["local"]["d"] == 10


def test_scouting_is_sport_aware_and_never_invents():
    from radar.stats import price_line, scouting

    st = {"local": {"forma": "VVVVV", "racha": "6V", "gf10": 115.0, "ga10": 104.0, "n10": 10, "temporada": {},
                    "local": {"pj": 9, "v": 8, "e": 0, "d": 1, "gf": 117.0, "ga": 105.0}},
          "visit": {"forma": "DDVDV", "racha": "1V", "gf10": 108.0, "ga10": 112.0, "n10": 10,
                    "visitante": {"pj": 8, "v": 3, "e": 0, "d": 5, "gf": 107.0, "ga": 113.0}}}
    lines = scouting(st, "Knicks", "Heat", "basketball", "ml", "H",
                     table={"local": {"pos": 2, "g": 14, "p": 5}, "visit": {"pos": 18, "g": 8, "p": 11}},
                     info={"tipo": "puntos", "margen": 6.4, "total": 226})
    txt = " ".join(lines)
    assert "Knicks lleva 6 victorias seguidas" in txt and "va 8-1 esta temporada" in txt  # sin empates en NBA
    assert "14-5, 2º" in txt and "favorito a Knicks por unos 6,4 puntos" in txt
    # estadísticas antiguas (sin 'temporada'): no se afirma «esta temporada»
    st["local"].pop("temporada")
    assert "en los últimos meses" in " ".join(scouting(st, "Knicks", "Heat", "basketball", "ml", "H"))
    # sin datos: se dice, sin inventar
    assert "Sin histórico propio" in scouting({}, "A", "B", "football", "1x2", "H")[0]
    assert "solo merece la pena si paga 1,25" in price_line(0.80, None, False)
    assert "cobras un 5,0 % más" in price_line(0.60, 1.75, True, 3, 1)
    assert "Pinnacle y 2 casas más" in price_line(0.60, 1.75, True, 3, 1)


def test_football_goal_markets_use_goal_trends():
    from radar.stats import scouting

    st = {"local": {"forma": "VEVVD", "racha": "1D", "gf10": 2.1, "ga10": 1.2, "n10": 10, "tot10": 3.3, "over25_10": 7,
                    "cs10": 2, "scored10": 9},
          "visit": {"forma": "DDEVE", "racha": "1E", "gf10": 1.4, "ga10": 1.9, "n10": 10, "tot10": 3.3, "over25_10": 6,
                    "cs10": 1, "scored10": 8}}
    ou = " ".join(scouting(st, "Betis", "Celta", "football", "ou", "O", 2.5))
    assert "más de 2,5 en 7 de los últimos 10 partidos de Betis" in ou
    btts = " ".join(scouting(st, "Betis", "Celta", "football", "btts", "Y"))
    assert "Betis ha marcado en 9 de 10" in btts and "portería a cero en 1" in btts


def test_combo_legs_carry_scouting_reports():
    from radar.combos import build_combos
    from radar.config import Settings

    leagues = ["laliga", "premier", "seriea"]
    matches = {f"m{i}": dict(home=f"Local{i}", away=f"Visit{i}", league=leagues[i], sport="football",
                             kickoff=f"2026-11-28T{12 + i}:00:00+00:00") for i in range(3)}
    preds = pd.DataFrame([dict(match_id=f"m{i}", market="dc", selection="1X", point=0.0, p_final=p, price=o, verified=1,
                               ev=p * o - 1, tier="pronostico", label="x", n_books=3, n_sharp=1, derived=1, weak=0,
                               b365_age_h=2.0) for i, (p, o) in enumerate([(0.80, 1.24), (0.78, 1.27), (0.76, 1.30)])])
    infos = {"m0": {"stats": {"local": {"forma": "VVVEV", "racha": "1V"}, "visit": {"forma": "DDDVD", "racha": "1D"}}}}
    tables = {"laliga": [{"equipo": "Local0", "pos": 3, "pts": 30, "pj": 14}, {"equipo": "Visit0", "pos": 17, "pts": 12, "pj": 14}]}
    combos = build_combos(preds, matches, Settings(), infos, tables=tables)
    legs = [l for c in combos for l in c["legs"] if l["match_id"] == "m0"]
    assert legs, combos
    why = " ".join(legs[0]["why"])
    assert "Local0 es 3º con 30 puntos" in why and "últimos 5: VVVEV" in why and "bet365 paga 1,24" in why
    assert "Probabilidad real" not in why  # nada de textos robóticos
