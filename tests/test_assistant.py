"""Asistente (copiloto v5): el rango de cuota y el riesgo se respetan siempre, primero valor ≥ 0 con cuotas
confirmadas, después «por confirmar», nunca una apuesta con valor negativo si hay otra opción, y tres planes."""
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from radar import assistant

NOW = datetime(2026, 11, 28, 8, tzinfo=timezone.utc)
DEADLINE = NOW + timedelta(days=2)
LEAGUES = ["laliga", "premier", "seriea", "bundesliga", "ligue1", "eredivisie", "portugal", "brasil", "mexico", "mls"]


def pool(rows):
    out = []
    for i, (p, odds, priced, sport) in enumerate(rows):
        out.append(dict(match_id=f"m{i}", sport=sport, league=LEAGUES[i % len(LEAGUES)],
                        kickoff=(NOW + timedelta(hours=6 + i)).isoformat(), home=f"Local{i}", away=f"Visit{i}",
                        label=f"Gana Local{i}", market="1x2", odds=odds if priced else round(1 / p, 2), p_final=p,
                        ev=(p * odds - 1) if priced else 0.0, tier="pronostico", verified=int(priced), priced=priced,
                        risk="medio", confidence="—", why=[f"Informe {i}"]))
    return pd.DataFrame(out)


def test_risk_range_applies_even_without_slider():
    """El caso real: había una 1,39 con valor (Sao Paulo) y se pedía riesgo medio → nunca 1,39."""
    P = pool([(0.73, 1.39, True, "football"), (0.40, 2.60, True, "football"), (0.50, 2.05, True, "basketball"),
              (0.36, 2.70, True, "nfl")])
    plan, alts, n = assistant.copilot(P, 100, DEADLINE, NOW, risk="medio")
    lo, hi = assistant.RISK_RANGE["medio"]
    assert plan.in_range and lo <= plan.odds <= hi and plan.status == "exacta"
    assert plan.ev >= 0
    # riesgo bajo (1,40-1,90): no existe ninguna simple ni combinada en ese rango → se dice, no se cuela la 1,39
    plan_b, _, _ = assistant.copilot(P, 100, DEADLINE, NOW, risk="bajo")
    assert plan_b.status == "cercana" and not plan_b.in_range and "1,40-1,90" in plan_b.rationale[1]


def test_explicit_odds_range_is_strict():
    P = pool([(0.73, 1.39, True, "football"), (0.42, 2.45, True, "football"), (0.47, 2.20, True, "football"),
              (0.31, 3.30, True, "basketball")])
    plan, alts, _ = assistant.copilot(P, 100, DEADLINE, NOW, risk="bajo", odds_range=(2.00, 3.40))
    assert 2.00 <= plan.odds <= 3.40 and plan.role == "equilibrada"
    assert all(l["odds"] != 1.39 for l in plan.legs) or plan.tipo == "combinada"


def test_positive_value_first_then_por_confirmar_never_negative():
    # en rango solo hay: una con valor negativo confirmada y una por confirmar
    P = pool([(0.40, 2.30, True, "football"), (0.45, 0, False, "football")])
    plan, _, _ = assistant.copilot(P, 100, DEADLINE, NOW, risk="medio", tipo="simple")
    assert plan.status == "por_confirmar" and plan.ev >= 0
    leg = plan.legs[0]
    assert leg["priced"] is False and leg["min_odds"] == pytest.approx(2.23) and leg["min_odds"] * leg["p_final"] >= 1 and "cuota mínima" in " ".join(plan.rationale)
    # con una confirmada con valor, esa va primero
    P2 = pool([(0.40, 2.30, True, "football"), (0.45, 0, False, "football"), (0.46, 2.25, True, "nfl")])
    plan2, _, _ = assistant.copilot(P2, 100, DEADLINE, NOW, risk="medio", tipo="simple")
    assert plan2.status == "exacta" and plan2.legs[0]["match_id"] == "m2"


def test_nothing_fits_explains_and_shows_closest():
    P = pool([(0.73, 1.39, True, "football"), (0.80, 1.26, True, "football")])
    plan, _, _ = assistant.copilot(P, 100, DEADLINE, NOW, risk="alto", tipo="simple")
    assert plan.status == "cercana" and not plan.in_range and plan.role == "cercana"
    assert "Ninguna apuesta cumple" in plan.rationale[1] and "2,80-5,00" in plan.rationale[1]


def test_three_plans_conservative_and_higher_return():
    P = pool([(0.62, 1.66, True, "football"), (0.60, 1.72, True, "basketball"), (0.45, 2.30, True, "nfl"),
              (0.44, 2.35, True, "football"), (0.80, 1.27, True, "football"), (0.30, 3.50, True, "basketball")])
    plan, alts, _ = assistant.copilot(P, 1000, DEADLINE, NOW, risk="medio")
    roles = {a.role: a for a in alts}
    assert plan.role == "equilibrada" and 1.90 <= plan.odds <= 2.80
    assert set(roles) == {"conservadora", "retorno"}
    assert roles["conservadora"].odds <= plan.odds and roles["conservadora"].prob >= plan.prob
    assert roles["retorno"].odds >= plan.odds
    sigs = {tuple(sorted(l["label"] for l in p.legs)) for p in [plan, *alts]}
    assert len(sigs) == 3  # tres propuestas distintas
    assert all(a.ev >= -0.0005 for a in [plan, *alts])  # ninguna con valor negativo
    assert all(l["why"] for a in [plan, *alts] for l in a.legs)


def test_stake_rules():
    s, pct, txt = assistant.stake_advice(0.05, 2.0, 1000)
    assert pct == pytest.approx(0.0125) and "Kelly" in txt
    s, pct, txt = assistant.stake_advice(0.0, 2.0, 1000)
    assert pct == 0.005 and "precio justo" in txt
    s, pct, txt = assistant.stake_advice(-0.02, 2.0, 1000)
    assert s == 0 and "no se recomienda" in txt


def test_deadline_and_sports_filters():
    P = pool([(0.47, 2.20, True, "football"), (0.47, 2.25, True, "basketball")])
    plan, _, _ = assistant.copilot(P, 100, DEADLINE, NOW, risk="medio", tipo="simple", sports=["basketball"])
    assert plan.legs[0]["sport"] == "basketball"
    plan2, _, n = assistant.copilot(P, 100, NOW + timedelta(hours=1), NOW, risk="medio")
    assert plan2 is None and n == 0
