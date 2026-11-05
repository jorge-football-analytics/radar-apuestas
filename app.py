"""Panel web del radar. Local: `streamlit run app.py` · Gratis en internet: Streamlit Community Cloud."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.express as px
import streamlit as st

from radar.config import LEAGUES, Settings
from radar.db import DB
from radar.learn import metrics

TZ = ZoneInfo("Europe/Madrid")
st.set_page_config(page_title="Radar de apuestas", page_icon="⚽", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.6rem; max-width: 1300px;}
div[data-testid="stMetricValue"] {font-size: 1.6rem;}
.conf-alta {color:#1a7f37; font-weight:600} .conf-media {color:#9a6700; font-weight:600} .conf-baja {color:#bc4c00}
</style>""", unsafe_allow_html=True)

s = Settings()


@st.cache_data(ttl=300)
def load(_mtime: float):
    db = DB(s.db_path)
    now = datetime.now(timezone.utc).isoformat()
    d = {
        "picks_next": db.df("SELECT * FROM picks WHERE kickoff > ? ORDER BY ev DESC", (now,)),
        "picks_done": db.df("SELECT * FROM picks WHERE won IS NOT NULL ORDER BY kickoff"),
        "combos": db.df("SELECT * FROM combos WHERE won IS NULL ORDER BY ev DESC"),
        "combos_done": db.df("SELECT * FROM combos WHERE won IS NOT NULL ORDER BY created"),
        "matches": db.df("SELECT * FROM matches WHERE status='pendiente' AND kickoff > ? ORDER BY kickoff", (now,)),
        "preds": db.df("SELECT * FROM predictions"),
        "metrics": metrics(db), "backtest": db.get("backtest"), "weights": db.get("weights"),
        "last_run": db.get("last_run"), "credits": db.get("odds_api_remaining"),
        "runs": db.df("SELECT * FROM runs ORDER BY rowid DESC LIMIT 20"),
    }
    db.close()
    return d


if not s.db_path.exists():
    st.info("Todavía no hay datos. Ejecuta `python -m radar run` o espera a la primera ejecución automática.")
    st.stop()

d = load(s.db_path.stat().st_mtime)


def when(iso: str) -> str:
    return datetime.fromisoformat(iso).astimezone(TZ).strftime("%d/%m %H:%M")


# Cabecera
left, right = st.columns([3, 2])
with left:
    st.title("⚽ Radar de apuestas")
    st.caption("Apuestas de valor: la probabilidad estimada supera la que implica la cuota. "
               "Comprueba siempre la cuota actual en la casa antes de apostar.")
with right:
    lr = d["last_run"]
    st.caption(f"Último análisis: {when(lr) if lr else '—'} · créditos The Odds API: "
               f"{int(d['credits']) if d['credits'] is not None else '—'}")

m = d["metrics"]
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Apuestas liquidadas", m.get("n", 0))
c2.metric("Acierto", f"{m['hit_rate']:.0%}" if m.get("n") else "—",
          help="Comparado con el acierto que esperaba el modelo: " + (f"{m['expected_hit']:.0%}" if m.get("n") else "—"))
c3.metric("ROI (1 u. por apuesta)", f"{m['roi']:+.1%}" if m.get("n") else "—")
c4.metric("Beneficio con stakes", f"{m['profit']:+.2f} €" if m.get("n") else "—")
c5.metric("CLV medio", f"{m['clv']:+.1%}" if m.get("clv") is not None else "—",
          help="Si ganas a la cuota de cierre de forma sostenida, la ventaja es real y no suerte.")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["🎯 Jugadas de valor", "🧩 Combinadas", "📅 Partidos", "📈 Rendimiento",
                                        "🧪 Backtest y estado"])

with tab1:
    p = d["picks_next"].copy()
    if p.empty:
        st.info("Ahora mismo no hay apuestas que pasen los filtros. Es normal: el sistema solo marca las que tienen ventaja.")
    else:
        f1, f2, f3 = st.columns(3)
        ligas = f1.multiselect("Liga", sorted(p["league"].unique()), format_func=lambda k: LEAGUES[k]["nombre"])
        confs = f2.multiselect("Confianza", ["alta", "media", "baja"], default=["alta", "media"])
        solo365 = f3.toggle("Solo con cuota en bet365")
        if ligas:
            p = p[p["league"].isin(ligas)]
        if confs:
            p = p[p["confidence"].isin(confs)]
        if solo365:
            p = p[p["b365_odds"].notna()]
        for r in p.to_dict("records"):
            with st.container(border=True):
                a, b, c, e = st.columns([3, 2, 2, 2])
                a.markdown(f"**{r['home']} - {r['away']}**  \n{LEAGUES[r['league']]['nombre']} · {when(r['kickoff'])}")
                b.markdown(f"🎯 **{r['label']}**  \n<span class='conf-{r['confidence']}'>confianza {r['confidence']}</span>",
                           unsafe_allow_html=True)
                c.markdown(f"**{r['odds']:.2f}** en {r['bookmaker']}  \n"
                           + (f"bet365: {r['b365_odds']:.2f}" if pd.notna(r['b365_odds']) else "bet365: sin cuota"))
                e.markdown(f"Prob. **{r['p_final']:.0%}** · EV **{r['ev']:+.1%}**  \nStake: {r['stake']:.2f} €")
                with st.expander("Por qué"):
                    for line in str(r["reasons"]).split(" | "):
                        st.write("•", line)

with tab2:
    cb = d["combos"]
    if cb.empty:
        st.info("Sin combinadas con valor para las próximas 48 h.")
    for risk, icon in (("conservadora", "🛡️"), ("equilibrada", "⚖️"), ("arriesgada", "🔥")):
        sub = cb[cb["risk"] == risk] if not cb.empty else cb
        if sub.empty:
            continue
        st.subheader(f"{icon} {risk.capitalize()}")
        for r in sub.to_dict("records"):
            with st.container(border=True):
                legs = json.loads(r["legs"])
                st.markdown(f"**{r['bookmaker']}** · cuota **{r['odds']:.2f}** · prob. **{r['prob']:.0%}** · "
                            f"EV **{r['ev']:+.1%}** · stake {r['stake']:.2f} €")
                for l in legs:
                    st.write(f"• {l['partido']} ({when(l['kickoff'])}): {l['label']} @ {l['price']:.2f} ({l['p']:.0%})")
    st.caption("Cada pata que añades multiplica también el margen de la casa: las combinadas ganan menos veces "
               "de lo que parece. El sistema solo propone las que mantienen valor esperado positivo.")

with tab3:
    mt, pr = d["matches"], d["preds"]
    if mt.empty:
        st.info("No hay partidos próximos cargados.")
    else:
        x = pr[(pr["market"] == "1x2")].pivot_table(index="match_id", columns="selection",
                                                     values=["p_final", "p_model", "p_market"])
        rows = []
        for r in mt.to_dict("records"):
            row = {"Fecha": when(r["kickoff"]), "Liga": LEAGUES[r["league"]]["nombre"],
                   "Partido": f"{r['home']} - {r['away']}"}
            if r["id"] in x.index:
                for sel, name in (("H", "1"), ("D", "X"), ("A", "2")):
                    row[f"{name} final"] = x.loc[r["id"], ("p_final", sel)]
                    row[f"{name} modelo"] = x.loc[r["id"], ("p_model", sel)] if ("p_model", sel) in x.columns else None
            rows.append(row)
        t = pd.DataFrame(rows)
        st.dataframe(t, hide_index=True, width="stretch",
                     column_config={c: st.column_config.NumberColumn(format="percent") for c in t.columns
                                    if "final" in c or "modelo" in c})

with tab4:
    done = d["picks_done"]
    if done.empty:
        st.info("Aún no hay apuestas liquidadas. El rendimiento aparecerá en cuanto se jueguen los primeros partidos.")
    else:
        done = done.copy()
        done["unidades"] = (done["odds"] - 1) * done["won"] - (1 - done["won"])
        done["acumulado"] = done["unidades"].cumsum()
        done["fecha"] = pd.to_datetime(done["kickoff"])
        fig = px.line(done, x="fecha", y="acumulado", markers=True,
                      labels={"acumulado": "Unidades ganadas (1 u. por apuesta)", "fecha": ""})
        fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, width="stretch")
        a, b = st.columns(2)
        if m.get("by_confidence"):
            a.markdown("**Por confianza**")
            a.dataframe(pd.DataFrame(m["by_confidence"]).T.rename(columns={"n": "apuestas", "hit": "acierto"}),
                        width="stretch")
        if m.get("combos"):
            b.markdown("**Combinadas**")
            b.dataframe(pd.DataFrame(m["combos"]).T.rename(columns={"n": "apuestas", "hit": "acierto"}),
                        width="stretch")
        if m.get("calibration"):
            cal = m["calibration"]
            st.markdown(f"**Calidad de las probabilidades** (log loss, menos es mejor, {cal['n']} predicciones): "
                        f"modelo {cal['logloss_modelo']:.4f} · mercado {cal['logloss_mercado']:.4f} · "
                        f"mezcla {cal['logloss_final']:.4f}")
        st.dataframe(done[["kickoff", "home", "away", "label", "odds", "bookmaker", "p_final", "ev", "won", "clv"]]
                     .sort_values("kickoff", ascending=False), hide_index=True, width="stretch")

with tab5:
    bt = d["backtest"]
    if bt:
        st.markdown(f"**Backtest walk-forward** ({bt['desde']} → {bt['hasta']}, {bt['n_predicciones']} predicciones). "
                    "El modelo solo veía partidos anteriores a cada jornada; apuestas a la cuota de bet365.")
        st.dataframe(pd.DataFrame(bt["estrategia_bet365"]).T, width="stretch")
        st.caption("Si el intervalo del ROI (roi_ic95) incluye el 0, no hay evidencia de ventaja todavía.")
        st.write("Peso del modelo frente al mercado:", d["weights"] or bt.get("pesos"))
    else:
        st.info("El backtest se ejecuta en el primer análisis con datos reales.")
    st.markdown("**Últimas ejecuciones**")
    st.dataframe(d["runs"], hide_index=True, width="stretch")
