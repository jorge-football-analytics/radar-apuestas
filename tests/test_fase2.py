"""Fase 2 del motor: rango de cuotas 1,30-2,60, nivel Sólida, Telegram solo con referencia sharp, combinadas
descorrelacionadas y análisis completo de la jornada."""
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from radar.config import Settings
from radar.value import evaluate_match

NOW = datetime(2026, 11, 27, 12, tzinfo=timezone.utc)
T = NOW.isoformat()


def rows(book, market, prices, t=T):
    sels = {"1x2": "HDA", "ml": "HA"}[market]
    return [dict(bookmaker=book, market=market, selection=s, point=0.0, price=p, fetched_at=t, source="x")
            for s, p in zip(sels, prices)]


def match(i=0, league="laliga"):
    return dict(id=f"{league}|m{i}", sport="football", league=league, kickoff="2026-11-28T20:00:00+00:00",
                home=f"Local{i}", away=f"Visit{i}")


def by_sel(cands, sel):
    return [c for c in cands if c.selection == sel]


# ------------------------------------------------------------------ niveles
def test_solid_pick_high_probability_at_fair_price():
    # Pinnacle sin margen: local ≈ 70 %. bet365 paga 1,43 ≈ precio justo (1/0,70 = 1,43)
    odds = pd.DataFrame(rows("pinnacle", "1x2", [1.40, 4.9, 9.0]) + rows("bet365", "1x2", [1.43, 4.5, 7.5]))
    cands, preds = evaluate_match(match(), odds, None, {}, Settings(), now=NOW)
    h = by_sel(cands, "H")[0]
    assert h.tier == "solida" and h.elite == 0 and h.p_final >= 0.65 and -0.005 <= h.ev < 0.025
    assert h.stake == 0 or h.ev > 0  # sin ventaja → sin stake de Kelly
    assert [p for p in preds if p["selection"] == "H"][0]["tier"] == "solida"


def test_solid_requires_not_giving_margin_away():
    odds = pd.DataFrame(rows("pinnacle", "1x2", [1.40, 4.9, 9.0]) + rows("bet365", "1x2", [1.36, 4.5, 7.5]))
    h = by_sel(evaluate_match(match(), odds, None, {}, Settings(), now=NOW)[0], "H")[0]
    assert h.tier == "pronostico" and h.ev < -0.005  # bet365 cobra margen: no es sólida


def test_odds_outside_range_are_never_recommended():
    # visitante con +8 % de valor pero a cuota 5,00 → fuera de 1,30-2,60
    odds = pd.DataFrame(rows("pinnacle", "1x2", [1.60, 4.0, 5.5]) + rows("bet365", "1x2", [1.55, 3.9, 5.0 * 1.25]))
    cands, _ = evaluate_match(match(), odds, None, {}, Settings(), now=NOW)
    assert all(c.tier not in ("valor", "solida") or c.odds <= 2.60 for c in cands)
    # favorito clarísimo a 1,15: probabilidad alta pero cuota por debajo de 1,30
    odds2 = pd.DataFrame(rows("pinnacle", "1x2", [1.12, 9.0, 21.0]) + rows("bet365", "1x2", [1.15, 8.0, 17.0]))
    assert not [c for c in evaluate_match(match(), odds2, None, {}, Settings(), now=NOW)[0]
                if c.tier in ("valor", "solida")]


def test_telegram_only_with_sharp_reference():
    soft = pd.DataFrame(rows("bwin", "1x2", [2.0, 3.4, 3.9]) + rows("williamhill", "1x2", [2.0, 3.3, 3.8]) +
                        rows("bet365", "1x2", [2.25, 3.3, 3.3]))
    h = by_sel(evaluate_match(match(), soft, None, {}, Settings(), now=NOW)[0], "H")[0]
    assert h.tier == "valor" and h.elite == 0  # Élite en la web, pero sin casa sharp no va a Telegram
    sharp = pd.DataFrame(rows("pinnacle", "1x2", [2.02, 3.5, 4.0]) + rows("bwin", "1x2", [2.0, 3.4, 3.9]) +
                         rows("williamhill", "1x2", [2.0, 3.3, 3.8]) + rows("bet365", "1x2", [2.25, 3.3, 3.3]))
    h = by_sel(evaluate_match(match(), sharp, None, {}, Settings(), now=NOW)[0], "H")[0]
    assert h.tier == "valor" and h.elite == 1 and h.n_sharp == 1


def test_solid_accepts_older_bet365_price_up_to_72h():
    old = (NOW - timedelta(hours=50)).isoformat()
    odds = pd.DataFrame(rows("pinnacle", "1x2", [1.40, 4.9, 9.0], t=old) + rows("bet365", "1x2", [1.43, 4.5, 7.5], t=old))
    h = by_sel(evaluate_match(match(), odds, None, {}, Settings(), now=NOW)[0], "H")[0]
    assert h.tier == "solida" and "confírmala" in h.reasons
    older = (NOW - timedelta(hours=80)).isoformat()
    odds2 = pd.DataFrame(rows("pinnacle", "1x2", [1.40, 4.9, 9.0], t=older) +
                         rows("bet365", "1x2", [1.43, 4.5, 7.5], t=older))
    assert by_sel(evaluate_match(match(), odds2, None, {}, Settings(), now=NOW)[0], "H")[0].tier == "pronostico"


# ------------------------------------------------------------------ combinadas
def test_combos_decorrelated_and_honest_joint_probability():
    from radar.combos import build_combos

    s = Settings()
    cands, matches = [], {}
    # 4 partidos sólidos: dos de LaLiga, uno de Premier, uno de Serie A
    for i, (lg, pin, b) in enumerate([("laliga", 1.40, 1.43), ("laliga", 1.42, 1.45), ("premier", 1.45, 1.48),
                                      ("seriea", 1.38, 1.41)]):
        m = match(i, lg)
        matches[m["id"]] = m
        o = pd.DataFrame(rows("pinnacle", "1x2", [pin, 4.8, 9.0]) + rows("bet365", "1x2", [b, 4.4, 7.5]))
        cands += evaluate_match(m, o, None, {}, s, now=NOW)[0]
    combos = build_combos(cands, matches, s)
    assert combos
    for c in combos:
        leagues = [l["league"] for l in c["legs"]]
        assert len(set(leagues)) == len(leagues)  # nunca dos patas de la misma liga
        assert c["prob"] == pytest.approx(np.prod([l["p"] for l in c["legs"]]))
        assert c["odds"] == pytest.approx(np.prod([l["price"] for l in c["legs"]]))
        if c["tier"] == "valor":
            assert c["ev"] >= s.combo_min_ev and all(l["ev"] >= 0 for l in c["legs"]) and c["stake"] > 0
        else:
            assert c["tier"] == "solida" and c["ev"] >= s.combo_solid_min_ev
        assert all(l["tier"] in ("valor", "solida") for l in c["legs"])


# ------------------------------------------------------------------ análisis de la jornada
def test_analysis_rows_cover_every_match_with_light_and_recommendation():
    from radar.pipeline import analysis_rows

    s = Settings()
    cases = {
        "elite": pd.DataFrame(rows("pinnacle", "1x2", [2.02, 3.5, 4.0]) + rows("bwin", "1x2", [2.0, 3.4, 3.9]) +
                              rows("bet365", "1x2", [2.25, 3.3, 3.3])),
        "solida": pd.DataFrame(rows("pinnacle", "1x2", [1.40, 4.9, 9.0]) + rows("bet365", "1x2", [1.43, 4.5, 7.5])),
        "margen": pd.DataFrame(rows("pinnacle", "1x2", [2.10, 3.4, 3.8]) + rows("bet365", "1x2", [2.10, 3.25, 3.6])),
        "caro": pd.DataFrame(rows("pinnacle", "1x2", [2.10, 3.4, 3.8]) + rows("bet365", "1x2", [1.92, 3.2, 3.5])),
        "sin_b365": pd.DataFrame(rows("pinnacle", "1x2", [2.10, 3.4, 3.8]) + rows("bwin", "1x2", [2.05, 3.3, 3.7])),
    }
    preds, matches = [], {}
    for i, (name, o) in enumerate(cases.items()):
        m = match(i, name)
        matches[m["id"]] = m
        preds += evaluate_match(m, o, None, {}, s, now=NOW)[1]
    matches["vacio|x"] = match(9, "vacio") | {"id": "vacio|x"}
    out = {r["league"]: r for r in analysis_rows(pd.DataFrame(preds), matches, NOW, s)}
    assert len(out) == len(matches)
    assert out["elite"]["tier"] == "valor" and out["elite"]["light"] == "verde"
    assert out["solida"]["tier"] == "solida" and out["solida"]["light"] == "verde"
    assert out["margen"]["light"] == "ambar" and "margen" in out["margen"]["recommendation"]
    assert out["caro"]["light"] == "rojo" and out["caro"]["recommendation"].startswith("No apostar")
    assert out["sin_b365"]["light"] in ("gris", "ambar")
    assert out["vacio"]["light"] == "gris"
    for r in out.values():
        assert r["recommendation"]
        if r.get("p_home") is not None:
            assert r["p_home"] + r["p_draw"] + r["p_away"] == pytest.approx(1)
