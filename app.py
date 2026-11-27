"""Radar · plataforma de análisis de apuestas. Local: `streamlit run app.py` · Internet: Streamlit Community Cloud.

Arquitectura: radar/webdata.py lee la base de datos (funciones puras, probadas) y aquí solo se pinta. Los datos se
cachean con @st.cache_data por fecha de modificación del fichero: la web carga al instante y se refresca sola
cuando GitHub Actions sube un análisis nuevo."""
from __future__ import annotations

import ast
import hashlib
import html
import json
import math
from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

import importlib
import pathlib
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import radar


def _reload_radar() -> str:
    """Streamlit Cloud reutiliza el mismo proceso de Python tras cada push: vuelve a ejecutar app.py, pero los módulos
    ya importados (radar.assistant, radar.webdata...) siguen en memoria con el código ANTIGUO. Si algún fichero de
    radar/ ha cambiado desde la última carga, se recargan todos en orden de dependencias."""
    root = pathlib.Path(radar.__file__).parent
    sig = str(hash(tuple(sorted((str(f), f.stat().st_mtime_ns, f.stat().st_size) for f in root.rglob("*.py")))))
    if getattr(radar, "_code_sig", None) != sig:
        order = ["radar.config", "radar.db", "radar.teams", "radar.model.dixon_coles", "radar.model.markets",
                 "radar.model.derived", "radar.model.elo", "radar.stats", "radar.tables", "radar.learn", "radar.value",
                 "radar.combos", "radar.assistant", "radar.webdata"]
        loaded_before = {n for n in order if n in sys.modules}
        for name in order:
            if name in loaded_before:
                importlib.reload(sys.modules[name])
            else:
                importlib.import_module(name)
        radar._code_sig = sig
    return sig


CODE_SIG = _reload_radar()

from radar import assistant, webdata  # noqa: E402  (después de la recarga)
from radar.combos import leg_report  # noqa: E402
from radar.config import FOOTBALL, OTHER, SPORTS, Settings, book_name, league_flag, league_name  # noqa: E402
from radar.model.derived import RHO, US_SD, fit_goal_rates  # noqa: E402
from radar.model.dixon_coles import score_matrix  # noqa: E402
from radar.model.markets import label as mlabel  # noqa: E402

TZ = ZoneInfo("Europe/Madrid")
DIAS = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
DIAS_C = ["lun", "mar", "mié", "jue", "vie", "sáb", "dom"]
MK = {"1x2": "Resultado", "ml": "Ganador", "ou": "Más/menos", "spread": "Hándicap", "btts": "Ambos marcan",
      "dc": "Doble oportunidad", "combinada": "Combinada"}
TIER = {  # nivel → (texto del badge, clase css)
    "valor": ("💎 Élite", "b-elite"), "solida": ("🛡️ Sólida", "b-solid"), "moderado": ("⚖️ Valor moderado", "b-mod"),
    "cuota_alta": ("🔥 Cuota alta", "b-high"), "verificar": ("🎯 Por confirmar", "b-check"),
    "objetivo": ("🎯 Cuota objetivo", "b-check"), "revisar": ("⚠️ Revisar", "b-review"),
    "pronostico": ("Sin ventaja", "b-none"), "modelo": ("Sin cuotas", "b-none"), "sin_datos": ("Sin datos", "b-none")}
TAB_NAME = {"valor": "💎 Élite", "solida": "🛡️ Sólidas", "moderado": "⚖️ Valor moderado", "cuota_alta": "🔥 Cuota alta",
            "verificar": "🎯 Por confirmar", "revisar": "⚠️ Revisar"}
LIGHT = {"verde": "Recomendable", "ambar": "Con reservas", "rojo": "No apostar", "gris": "Sin datos fiables"}
COMBO_NAME = {"segura": "Segura", "equilibrada": "Equilibrada", "valor": "Valor EV+", "conservadora": "Conservadora",
              "arriesgada": "Arriesgada"}
COMBO_TIER = {"valor": ("💎 Con valor", "b-elite"), "solida": ("🛡️ Sólida", "b-solid"), "sin_ventaja": ("Sin ventaja", "b-none"),
              "verificar": ("🎯 Por confirmar", "b-check"), "pronostico": ("Sin ventaja", "b-none")}
C = dict(bg="#070A0F", panel="#0D131C", line="#1E2A3A", text="#EAF0F6", mute="#8592A3", amber="#FFB020",
         green="#22D37F", red="#FF5C66", cyan="#3BC9F5", violet="#A78BFA", blue="#5B8CFF")

st.set_page_config(page_title="Radar EV · Centro de mando", page_icon="📡", layout="wide",
                   initial_sidebar_state="collapsed")
s = Settings()
e = html.escape

# ======================================================================= estilo
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');
:root{--bg:#070A0F;--panel:#0D131C;--panel2:#121A26;--panel3:#18223127;--line:#1E2A3A;--line2:#2A3850;--text:#EAF0F6;
--mute:#8592A3;--dim:#5C6878;--amber:#FFB020;--green:#22D37F;--red:#FF5C66;--cyan:#3BC9F5;--violet:#A78BFA;--blue:#5B8CFF;
--orange:#FF8A4C;--mono:'JetBrains Mono',ui-monospace,SFMono-Regular,Menlo,monospace;
--sans:'Inter',system-ui,-apple-system,Segoe UI,sans-serif;--r:14px}
html,body,[data-testid="stAppViewContainer"],[data-testid="stHeader"],[data-testid="stMain"]{background:var(--bg)!important;color:var(--text)}
[data-testid="stAppViewContainer"]{background:radial-gradient(1200px 500px at 10% -10%,rgba(91,140,255,.08),transparent 60%),
radial-gradient(900px 400px at 100% 0%,rgba(34,211,127,.06),transparent 60%),var(--bg)!important}
[data-testid="stHeader"]{height:2.6rem;background:rgba(7,10,15,.85)!important;backdrop-filter:blur(8px);border-bottom:1px solid var(--line)}
.block-container{padding:3.6rem 1.6rem 4rem;max-width:1480px}
*{font-family:var(--sans)}
html,body{font-size:15.5px}
p,li{line-height:1.5}
code,.mono,.num{font-family:var(--mono)!important;font-variant-numeric:tabular-nums}
a{color:var(--amber)}
.pos{color:var(--green)!important}.neg{color:var(--red)!important}.am{color:var(--amber)!important}.mute{color:var(--mute)}
/* cabecera */
.hero{display:flex;align-items:flex-end;justify-content:space-between;gap:1.2rem;flex-wrap:wrap;margin:.2rem 0 1.1rem}
.hero .ttl{display:flex;flex-direction:column;gap:.25rem}
.hero .eyebrow{font-family:var(--mono);font-size:.74rem;letter-spacing:.16em;color:var(--amber);font-weight:600}
.hero h1{font-size:2.05rem;font-weight:800;letter-spacing:-.02em;margin:0;padding:0;color:var(--text);line-height:1.1}
.hero p{color:var(--mute);font-size:.98rem;margin:0;max-width:70ch}
.status{display:flex;gap:.5rem;flex-wrap:wrap;align-items:center}
.chip{display:inline-flex;align-items:center;gap:.4rem;background:var(--panel);border:1px solid var(--line);border-radius:999px;
padding:.32rem .75rem;font-size:.8rem;color:var(--mute);white-space:nowrap}.chip b{color:var(--text);font-weight:600}
.live{display:inline-block;width:8px;height:8px;border-radius:50%;background:var(--green);box-shadow:0 0 0 0 rgba(34,211,127,.6);
animation:pulse 2s infinite}.live.off{background:var(--red);animation:none}
@keyframes pulse{0%{box-shadow:0 0 0 0 rgba(34,211,127,.55)}70%{box-shadow:0 0 0 8px rgba(34,211,127,0)}100%{box-shadow:0 0 0 0 rgba(34,211,127,0)}}
/* KPIs */
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(165px,1fr));gap:12px;margin:.2rem 0 1.3rem}
.kpi{background:linear-gradient(180deg,var(--panel2),var(--panel));border:1px solid var(--line);border-radius:var(--r);
padding:.9rem 1rem;position:relative;overflow:hidden}
.kpi::before{content:"";position:absolute;inset:0 0 auto 0;height:3px;background:var(--kc,var(--line2))}
.kpi .l{font-size:.78rem;color:var(--mute);font-weight:500}
.kpi .v{font-family:var(--mono);font-weight:700;font-size:1.9rem;line-height:1.1;margin-top:.25rem;letter-spacing:-.02em}
.kpi .d{font-size:.8rem;color:var(--mute);margin-top:.15rem}
.k-amber{--kc:var(--amber)}.k-green{--kc:var(--green)}.k-red{--kc:var(--red)}.k-cyan{--kc:var(--cyan)}
.k-violet{--kc:var(--violet)}.k-orange{--kc:var(--orange)}.k-blue{--kc:var(--blue)}
/* secciones */
.sec{display:flex;align-items:center;gap:.7rem;margin:1.8rem 0 .7rem}
.sec h3{margin:0;padding:0;font-size:1.28rem;font-weight:700;letter-spacing:-.01em}
.sec .n{font-family:var(--mono);color:#1A1200;background:var(--amber);font-size:.78rem;font-weight:700;border-radius:999px;padding:.05rem .55rem}
.sec .line{flex:1;height:1px;background:linear-gradient(90deg,var(--line2),transparent)}
.hint{color:var(--mute);font-size:.92rem;margin:.0rem 0 .9rem;max-width:100ch}
.empty{background:var(--panel);border:1px dashed var(--line2);border-radius:var(--r);padding:1.1rem 1.2rem;color:var(--mute);font-size:.95rem}
.note{background:rgba(91,140,255,.07);border:1px solid rgba(91,140,255,.35);border-radius:var(--r);padding:.8rem 1rem;font-size:.9rem;color:var(--text)}
.warnbox{background:rgba(255,92,102,.08);border:1px solid rgba(255,92,102,.45);border-radius:var(--r);padding:.8rem 1rem;
color:var(--text);font-size:.92rem;margin-bottom:.8rem}
/* badges */
.bdg{display:inline-flex;align-items:center;gap:.25rem;font-size:.74rem;font-weight:700;letter-spacing:.02em;
padding:.22rem .6rem;border-radius:999px;border:1px solid transparent;white-space:nowrap;vertical-align:middle}
.b-elite{background:linear-gradient(90deg,#FFB020,#FFCF5C);color:#1A1200}
.b-solid{background:rgba(34,211,127,.14);color:var(--green);border-color:rgba(34,211,127,.45)}
.b-mod{background:rgba(59,201,245,.12);color:var(--cyan);border-color:rgba(59,201,245,.42)}
.b-high{background:rgba(255,138,76,.14);color:var(--orange);border-color:rgba(255,138,76,.5)}
.b-check{background:rgba(167,139,250,.12);color:var(--violet);border-color:rgba(167,139,250,.45)}
.b-review{color:var(--red);border-color:rgba(255,92,102,.6)}
.b-none{color:var(--mute);border-color:var(--line2)}.b-tg{background:rgba(167,139,250,.15);color:var(--violet)}
.b-warn{background:rgba(255,92,102,.12);color:var(--red);border-color:rgba(255,92,102,.5)}
.b-risk{color:var(--text);border-color:var(--line2)}
.lt{display:inline-block;width:9px;height:9px;border-radius:50%;flex:none}
.lt.verde{background:var(--green);box-shadow:0 0 8px rgba(34,211,127,.55)}.lt.ambar{background:var(--amber)}
.lt.rojo{background:var(--red)}.lt.gris{background:var(--dim)}
/* escudos (monograma) */
.crest{display:inline-flex;align-items:center;justify-content:center;border-radius:50%;font-weight:800;color:#fff;flex:none;line-height:1;
font-family:var(--sans);letter-spacing:-.02em;box-shadow:inset 0 0 0 2px rgba(255,255,255,.14),0 6px 16px rgba(0,0,0,.35)}
/* tarjeta de oportunidad */
.opps{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:14px}
.opp{background:linear-gradient(180deg,var(--panel2),var(--panel) 55%);border:1px solid var(--line);border-radius:18px;
padding:1rem 1.05rem 1.05rem;position:relative;overflow:hidden;transition:transform .15s ease,border-color .15s ease,box-shadow .15s ease}
.opp:hover{transform:translateY(-3px);box-shadow:0 14px 34px rgba(0,0,0,.45)}
.opp::after{content:"";position:absolute;inset:0 0 auto 0;height:3px;background:var(--tc,var(--line2))}
.t-valor{--tc:linear-gradient(90deg,#FFB020,#FFD36B);border-color:rgba(255,176,32,.45)}
.t-solida{--tc:var(--green);border-color:rgba(34,211,127,.32)}
.t-moderado{--tc:var(--cyan);border-color:rgba(59,201,245,.3)}
.t-cuota_alta{--tc:var(--orange);border-color:rgba(255,138,76,.4)}
.t-objetivo,.t-verificar{--tc:var(--violet);border-color:rgba(167,139,250,.32)}
.opp-top{display:flex;align-items:flex-start;gap:.5rem}
.opp-top .meta{color:var(--mute);font-size:.8rem;margin-top:.35rem}
.opp-top .g{margin-left:auto}
.vs{display:grid;grid-template-columns:1fr auto 1fr;align-items:center;gap:.6rem;margin:.85rem 0 .7rem}
.vs .tm{display:flex;flex-direction:column;align-items:center;gap:.4rem;text-align:center;min-width:0}
.vs .tm .nm{font-weight:600;font-size:.92rem;line-height:1.2;overflow:hidden;text-overflow:ellipsis;display:-webkit-box;
-webkit-line-clamp:2;-webkit-box-orient:vertical}
.vs .x{font-family:var(--mono);color:var(--dim);font-size:.78rem;font-weight:700}
.pick{background:rgba(255,255,255,.025);border:1px solid var(--line);border-radius:12px;padding:.6rem .75rem;margin-bottom:.65rem}
.pick .k{font-size:.72rem;color:var(--mute);text-transform:uppercase;letter-spacing:.08em;font-weight:600}
.pick .v{font-size:1.12rem;font-weight:700;margin-top:.1rem}
.nums{display:grid;grid-template-columns:1.25fr 1fr 1fr;gap:8px}
.nums div{background:var(--panel2);border:1px solid var(--line);border-radius:10px;padding:.45rem .6rem;min-width:0}
.nums b{display:block;font-family:var(--mono);font-size:1.32rem;font-weight:700;line-height:1.15;white-space:nowrap}
.nums span{font-size:.72rem;color:var(--mute)}
.nums .odd b{color:var(--amber);font-size:1.6rem}
.src{font-size:.8rem;color:var(--mute);margin:.5rem 0 .2rem}.src b{color:var(--text)}
.cmp{margin-top:.55rem;display:grid;grid-template-columns:86px 1fr 46px;gap:.25rem .5rem;align-items:center;font-size:.76rem;color:var(--mute)}
.cmp .tr{height:8px;background:var(--panel2);border-radius:99px;overflow:hidden}
.cmp .tr i{display:block;height:100%;border-radius:99px}
.cmp .p{font-family:var(--mono);color:var(--text);text-align:right}
.pay{font-family:var(--mono);font-size:.8rem;color:var(--mute);margin-top:.55rem}.pay b{color:var(--text)}
.opp details,.card details{margin-top:.6rem}
details{font-size:.86rem;color:var(--mute)}
details summary{cursor:pointer;color:var(--text);list-style:none;font-size:.86rem;font-weight:600;padding:.45rem .65rem;
border:1px solid var(--line);border-radius:10px;background:var(--panel2)}
details summary::-webkit-details-marker{display:none}
details summary::before{content:"▸ ";color:var(--amber)}
details[open] summary::before{content:"▾ "}
details ul{margin:.5rem 0 0;padding-left:1.1rem}details li{margin:.18rem 0}
.gauge{display:inline-flex;flex-direction:column;align-items:center;line-height:1}
.gauge svg{display:block}.gauge span{font-size:.62rem;color:var(--mute);letter-spacing:.06em;text-transform:uppercase;margin-top:1px}
/* lista compacta */
.rows{border:1px solid var(--line);border-radius:var(--r);overflow:hidden;background:var(--panel)}
.row{display:grid;grid-template-columns:44px minmax(180px,1.6fr) minmax(160px,1.3fr) 92px 84px 84px 64px;gap:.8rem;align-items:center;
padding:.65rem .9rem;border-top:1px solid var(--line)}
.row:first-child{border-top:0}.row:hover{background:#101824}
.row .m b{display:block;font-weight:600;font-size:.92rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.row .m span,.row .s span{color:var(--mute);font-size:.76rem}
.row .s b{display:block;font-size:.92rem}
.row .o{font-family:var(--mono);font-weight:700;font-size:1.05rem;color:var(--amber);text-align:right}
.row .o small{display:block;font-size:.66rem;color:var(--mute);font-weight:500}
.row .n{font-family:var(--mono);text-align:right;font-weight:600}
.row .n small{display:block;font-size:.66rem;color:var(--mute);font-weight:500}
/* combinadas */
.card{background:linear-gradient(180deg,var(--panel2),var(--panel) 55%);border:1px solid var(--line);border-radius:18px;padding:1rem 1.1rem;position:relative}
.card.hero-c{border-color:rgba(255,176,32,.5);box-shadow:0 0 0 1px rgba(255,176,32,.12),0 20px 50px rgba(0,0,0,.35)}
.card-h{display:flex;gap:.45rem;align-items:center;flex-wrap:wrap}.card-h .w{margin-left:auto;color:var(--mute);font-size:.8rem}
.legs{margin:.6rem 0 .3rem;padding:0;list-style:none}
.legs li{display:flex;justify-content:space-between;gap:.7rem;padding:.55rem 0;border-top:1px solid var(--line);font-size:.92rem}
.legs li:first-child{border-top:0}.legs .lp{color:var(--mute);font-size:.8rem}
.legs .lo{font-family:var(--mono);font-weight:700;font-size:1.08rem;color:var(--amber);white-space:nowrap;text-align:right}
.legs .lo small{display:block;color:var(--mute);font-size:.68rem;font-weight:500}
.big4{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:8px;margin:.75rem 0 .3rem}
.big4 div{background:var(--panel2);border:1px solid var(--line);border-radius:10px;padding:.5rem .65rem}
.big4 b{display:block;font-family:var(--mono);font-size:1.3rem;font-weight:700;line-height:1.15}
.big4 span{color:var(--mute);font-size:.74rem}
.why{margin:.3rem 0 0;padding-left:1.1rem;color:var(--mute);font-size:.84rem}
.legs .why li{display:list-item;border:0;padding:.1rem 0;font-size:.84rem}
/* fichas */
.mcs{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:12px}
.mc{background:var(--panel);border:1px solid var(--line);border-radius:var(--r);padding:.85rem .95rem}
.mc-h{display:flex;justify-content:space-between;align-items:center;gap:.6rem;color:var(--mute);font-size:.8rem}
.mc-t{display:grid;grid-template-columns:auto 1fr auto;gap:.35rem .6rem;align-items:center;margin:.55rem 0}
.mc-t .nm{font-weight:600;font-size:.98rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.mc-t .pp{font-family:var(--mono);font-weight:700}
.tri{display:flex;height:7px;border-radius:99px;overflow:hidden;margin-top:.45rem;background:var(--panel2)}
.tri span:nth-child(1){background:var(--blue)}.tri span:nth-child(2){background:#5E6B7A}.tri span:nth-child(3){background:var(--violet)}
.mc-r{font-size:.84rem;margin-top:.55rem}
.det{background:linear-gradient(180deg,var(--panel2),var(--panel));border:1px solid var(--line2);border-radius:20px;padding:1.2rem 1.3rem;margin-bottom:1rem}
.det-vs{display:grid;grid-template-columns:1fr auto 1fr;align-items:center;gap:1rem}
.det-vs .tm{display:flex;align-items:center;gap:.9rem}.det-vs .tm.r{flex-direction:row-reverse;text-align:right}
.det-vs .tm b{font-size:1.35rem;font-weight:800;letter-spacing:-.01em}.det-vs .tm small{display:block;color:var(--mute);font-size:.82rem}
.det-vs .mid{text-align:center;color:var(--mute);font-size:.84rem}.det-vs .mid b{display:block;font-family:var(--mono);
font-size:1.5rem;color:var(--text)}
.inj{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.inj div{background:var(--panel2);border:1px solid var(--line);border-radius:12px;padding:.7rem .85rem;font-size:.86rem}
.inj b{display:block;margin-bottom:.3rem}
table.mk{width:100%;border-collapse:collapse;font-size:.86rem;color:var(--text)}
table.mk th{color:var(--mute);font-weight:500;text-align:right;padding:.4rem .45rem;border-bottom:1px solid var(--line2);font-size:.74rem;text-transform:uppercase;letter-spacing:.04em}
table.mk th:first-child,table.mk td:first-child{text-align:left}
table.mk td{padding:.4rem .45rem;text-align:right;border-bottom:1px solid var(--line);font-family:var(--mono)}
table.mk td:first-child{font-family:var(--sans)}table.mk tr.v td{color:var(--green)}
table.tb{width:100%;border-collapse:collapse;font-size:.88rem;background:var(--panel);border:1px solid var(--line);border-radius:12px;overflow:hidden}
table.tb th{color:var(--mute);font-weight:500;padding:.5rem .5rem;text-align:right;border-bottom:1px solid var(--line2);
background:var(--panel2);font-size:.72rem;text-transform:uppercase;letter-spacing:.05em}
table.tb td{padding:.45rem .5rem;text-align:right;border-bottom:1px solid var(--line);font-family:var(--mono)}
table.tb th.l,table.tb td.l{text-align:left}table.tb td.l{font-family:var(--sans)}table.tb td.b{font-weight:600}
.fm{display:inline-flex;gap:3px}.fm i{display:inline-block;width:18px;height:18px;border-radius:5px;font-style:normal;
font-size:.64rem;line-height:18px;text-align:center;color:#06080C;font-weight:800;font-family:var(--mono)}
.fm .V{background:var(--green)}.fm .E{background:#8592A3}.fm .D{background:var(--red)}
.bar{display:inline-block;height:7px;border-radius:99px;background:var(--blue);vertical-align:middle}
.scroll{overflow-x:auto}
/* asistente */
.plan{background:linear-gradient(180deg,var(--panel2),var(--panel) 50%);border:1px solid var(--line);border-radius:18px;padding:1rem 1.05rem;height:100%}
.plan.mainp{border-color:rgba(255,176,32,.55);box-shadow:0 0 0 1px rgba(255,176,32,.12),0 16px 40px rgba(0,0,0,.35)}
.plan h4{font-size:1.15rem;margin:0 0 .25rem;font-weight:800}
.plan .sub{color:var(--mute);font-size:.82rem}
/* streamlit */
[data-testid="stTabs"] button p{font-size:.92rem;font-weight:600}
[data-testid="stTabs"] [aria-selected="true"] p{color:var(--amber)}
[data-baseweb="tab-highlight"]{background:var(--amber)!important}
[data-baseweb="tab-border"]{background:var(--line)!important}
[data-testid="stButtonGroup"]>div{flex-wrap:wrap;row-gap:6px}
[data-testid="stDataFrame"]{border:1px solid var(--line);border-radius:12px;overflow:hidden}
[data-testid="stExpander"] details{border:1px solid var(--line)!important;border-radius:12px!important;background:var(--panel)!important}
div[data-testid="stTextInput"] input{background:var(--panel)!important;border-color:var(--line2)!important}
.stButton>button[kind="primary"]{background:linear-gradient(90deg,#FFB020,#FFCF5C);color:#1A1200;border:0;font-weight:700;border-radius:12px;height:3rem}
button:focus-visible,summary:focus-visible{outline:2px solid var(--amber);outline-offset:2px}
@media (max-width:900px){
.block-container{padding:3.2rem .8rem 3rem}.hero h1{font-size:1.5rem}.kpi .v{font-size:1.45rem}
.row{grid-template-columns:36px 1fr 86px;grid-template-areas:"c m o" "c s s" "c n1 n2"}
.row .cr{grid-area:c}.row .m{grid-area:m}.row .s{grid-area:s}.row .o{grid-area:o}.row .n1{grid-area:n1;text-align:left}
.row .n2{grid-area:n2}.row .n3{display:none}
.inj{grid-template-columns:1fr}.det-vs .tm b{font-size:1.05rem}.nums b{font-size:1.1rem}.nums .odd b{font-size:1.3rem}}
</style>""", unsafe_allow_html=True)


# ======================================================================= utilidades
def isnum(x) -> bool:
    return x is not None and not isinstance(x, str) and not (isinstance(x, float) and np.isnan(x))


def txt(x, default: str = "—") -> str:
    """Texto seguro: nunca 'nan', 'None' ni 'null' en pantalla."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return default
    t = str(x).strip()
    return default if t.lower() in ("", "nan", "none", "null", "nat") else t


def pct(x, signed=False, dec=0) -> str:
    if not isnum(x):
        return "—"
    return (f"{x:+.{max(dec, 1)}%}" if signed else f"{x:.{dec}%}").replace(".", ",")


def num(x, d=2) -> str:
    if not isnum(x):
        return "—"
    return f"{x:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def sign_cls(x) -> str:
    return "" if not isnum(x) else "pos" if x >= 0 else "neg"


def loc(iso: str) -> datetime:
    return datetime.fromisoformat(iso).astimezone(TZ)


def when(iso: str) -> str:
    d = loc(iso)
    today = datetime.now(TZ).date()
    day = "hoy" if d.date() == today else "mañana" if d.date() == today + timedelta(days=1) else \
        f"{DIAS_C[d.weekday()]} {d:%d/%m}"
    return f"{day} {d:%H:%M}"


def when_nat(iso: str) -> str:
    """Fecha en lenguaje natural: «hoy a las 21:00», «mañana a las 01:55», «el sábado 28/11 a las 18:00»."""
    d = loc(iso)
    today = datetime.now(TZ).date()
    if d.date() == today:
        return f"hoy a las {d:%H:%M}"
    if d.date() == today + timedelta(days=1):
        return f"mañana a las {d:%H:%M}"
    return f"el {DIAS[d.weekday()]} {d:%d/%m} a las {d:%H:%M}"


def day_title(d) -> str:
    today = datetime.now(TZ).date()
    if d == today:
        return "Hoy"
    if d == today + timedelta(days=1):
        return "Mañana"
    return f"{DIAS[d.weekday()].capitalize()} {d:%d/%m}"


def crest(name: str, size: int = 52) -> str:
    """Escudo-monograma del equipo o jugador: iniciales sobre un color estable derivado del nombre."""
    nm = txt(name, "?")
    words = [w for w in nm.replace(".", " ").replace("-", " ").split() if w[:1].isalnum()]
    if not words:
        ini = "?"
    elif len(words) == 1:
        ini = words[0][:2].upper()
    else:
        skip = {"fc", "cf", "cd", "sc", "ac", "afc", "real", "club", "de", "la", "los", "las", "the", "united", "city"}
        core = [w for w in words if w.lower() not in skip] or words
        ini = (core[0][0] + (core[1][0] if len(core) > 1 else core[0][1:2])).upper()
    h = int(hashlib.md5(nm.encode()).hexdigest()[:6], 16)
    hue, hue2 = h % 360, (h // 360) % 360
    fs = max(10, int(size * 0.36))
    return (f'<span class="crest" style="width:{size}px;height:{size}px;font-size:{fs}px;background:linear-gradient(135deg,'
            f'hsl({hue},62%,42%),hsl({(hue + 40 + hue2 % 40) % 360},70%,28%))">{e(ini)}</span>')


def gauge(score: int, label: str = "confianza", size: int = 64) -> str:
    """Velocímetro semicircular 0-100 (SVG en línea)."""
    sc = max(0, min(100, int(score or 0)))
    ang = math.pi * (1 - sc / 100)
    x, y = 30 + 22 * math.cos(ang), 30 - 22 * math.sin(ang)
    col = "#22D37F" if sc >= 65 else "#FFB020" if sc >= 40 else "#FF5C66"
    arc = f"M8,30 A22,22 0 0,1 {x:.1f},{y:.1f}" if sc > 0 else ""  # nunca más de media vuelta
    hgt = int(size * 0.6)
    return (f'<span class="gauge" title="Confianza en la estimación: {sc}/100"><svg width="{size}" height="{hgt}" viewBox="0 0 60 36">'
            f'<path d="M8,30 A22,22 0 0,1 52,30" fill="none" stroke="#1E2A3A" stroke-width="5.5" stroke-linecap="round"/>'
            + (f'<path d="{arc}" fill="none" stroke="{col}" stroke-width="5.5" stroke-linecap="round"/>' if arc else "")
            + f'<text x="30" y="29" text-anchor="middle" font-family="JetBrains Mono" font-size="12" font-weight="700" '
            f'fill="#EAF0F6">{sc}</text></svg><span>{label}</span></span>')


def badge(tier: str) -> str:
    t, cls = TIER.get(tier, (txt(tier), "b-none"))
    return f'<span class="bdg {cls}">{e(t)}</span>'


def form_html(f) -> str:
    return '<span class="fm">' + "".join(f'<i class="{c}">{c}</i>' for c in txt(f, "") if c in "VED") + "</span>"


def section(title: str, n=None, hint: str = ""):
    st.markdown(f'<div class="sec"><h3>{e(title)}</h3>{f"<span class=n>{n}</span>" if n is not None else ""}'
                f'<span class="line"></span></div>' + (f'<p class="hint">{hint}</p>' if hint else ""),
                unsafe_allow_html=True)


def empty(msg: str):
    st.markdown(f'<div class="empty">{msg}</div>', unsafe_allow_html=True)


def kpi_strip(items: list[tuple]):
    """items: (etiqueta, valor, detalle, color, clase del valor)."""
    st.markdown('<div class="kpis">' + "".join(
        f'<div class="kpi k-{c}"><div class="l">{e(l)}</div><div class="v {vc}">{v}</div>'
        f'<div class="d">{d}</div></div>' for l, v, d, c, vc in items) + "</div>", unsafe_allow_html=True)


def show_df(df: pd.DataFrame, fmt: dict | None = None, **k):
    """Tabla sin NaN ni None: todo se formatea a texto (los vacíos salen como «—»)."""
    fmt = fmt or {}

    def cell(v, f):
        if isinstance(v, str):
            return txt(v)
        if not isnum(v):
            return "—"
        if f:
            return f(v)
        return num(v, 0) if float(v).is_integer() else num(v)
    out = df.copy()
    for col in out.columns:
        out[col] = [cell(v, fmt.get(col)) for v in out[col]]
    st.dataframe(out, hide_index=True, width="stretch", **k)


def chart_layout(fig: go.Figure, h=340, legend=True):
    fig.update_layout(height=h, margin=dict(l=8, r=8, t=28, b=8), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter, sans-serif", color=C["text"], size=12),
                      legend=dict(orientation="h", y=1.12, x=0, bgcolor="rgba(0,0,0,0)"), showlegend=legend,
                      hoverlabel=dict(bgcolor=C["panel"], bordercolor=C["line"], font_family="JetBrains Mono"))
    fig.update_xaxes(gridcolor=C["line"], zeroline=False, linecolor=C["line"])
    fig.update_yaxes(gridcolor=C["line"], zeroline=False, linecolor=C["line"])
    return fig


PLOT_CFG = {"displayModeBar": False, "responsive": True}


# ======================================================================= datos (caché)
@st.cache_data(ttl=180, show_spinner="Cargando el radar…")
def get_data(path: str, mtime: float, code_sig: str) -> dict:
    """code_sig: si cambia el código, la caché se invalida."""
    return webdata.load_dashboard(path)


if not s.db_path.exists():
    empty("Aún no hay análisis. Lanza el workflow en GitHub (Actions → Run workflow).")
    st.stop()
D = get_data(str(s.db_path), s.db_path.stat().st_mtime, CODE_SIG)
MATCHES = {r["id"]: r for r in D["matches"].to_dict("records")} if not D["matches"].empty else {}
_SX = D.get("selections", pd.DataFrame())
SEL_IDX = {(r["match_id"], r["market"], r["selection"], float(r["point"] or 0)): r
           for r in (_SX.to_dict("records") if _SX is not None and not _SX.empty else [])}
SINCE_DEFAULT = webdata.metrics_since_default()
NOW = datetime.now(timezone.utc)
DAILY = webdata.daily_picks(D["pool"], NOW)


def report_for(match_id: str, sport: str, market: str, selection: str, point: float, p: float, price, priced: bool,
               n_books=None, n_sharp=None, derived: bool = False, stored=None, ref_book=None, b365_est=None) -> list[str]:
    """Informe de scouting calculado al pintar. Si el partido ya no está en la lista de próximos, el texto guardado."""
    m = MATCHES.get(match_id)
    if m is None:
        return list(stored or [])
    try:
        return leg_report(m, sport, market, selection, float(point or 0), D["info"].get(match_id) or {}, D["tables"],
                          float(p), float(price) if priced and price else None, priced, n_books, n_sharp, derived,
                          ref_book, b365_est)
    except Exception:  # noqa: BLE001  (nunca romper la página por un informe)
        return list(stored or [])


def since_filter(df: pd.DataFrame, since: str) -> pd.DataFrame:
    if df.empty or not since:
        return df
    col = df["shown_at"].fillna(df["created"]) if "shown_at" in df else df["created"]
    return df[col >= since]


def hero(eyebrow: str, title: str, sub: str):
    lr = D["last_run"]
    stale = not lr or (NOW - datetime.fromisoformat(lr)) > timedelta(hours=14)
    sm = webdata.day_summary(D["analysis"])
    st.markdown(
        f'<div class="hero"><div class="ttl"><span class="eyebrow">{e(eyebrow)}</span><h1>{e(title)}</h1>'
        f'<p>{e(sub)}</p></div><div class="status">'
        f'<span class="chip"><span class="live{" off" if stale else ""}"></span>Actualizado <b>{when(lr) if lr else "—"}</b></span>'
        f'<span class="chip">Partidos <b>{sm["partidos"]}</b></span>'
        f'<span class="chip">Recomendaciones <b>{len(DAILY)}</b></span></div></div>', unsafe_allow_html=True)


# ======================================================================= oportunidades (tarjetas)
def _v(r: dict, k: str):
    x = r.get(k)
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else x


def opp_from_pool(r: dict) -> dict:
    """Selección del pool (selecciones del último análisis) → datos de tarjeta."""
    src = _v(r, "price_src")
    priced = bool(_v(r, "priced"))
    objetivo = bool(_v(r, "objetivo"))
    book = (_v(r, "ref_book") if src == "mercado" else "bet365") if priced else None
    tier = "objetivo" if objetivo else (_v(r, "tier") or "pronostico")
    odds, p, ev = float(r["odds"]), float(r["p_final"]), _v(r, "ev")
    stake = 0.0
    if priced and not objetivo and isnum(ev) and ev >= -0.0005:
        stake = assistant.stake_advice(ev, odds, s.bankroll, s.kelly_fraction, s.max_stake_pct)[0]
    return dict(match_id=r["match_id"], sport=r["sport"], league=r["league"], kickoff=r["kickoff"], home=r["home"],
                away=r["away"], market=r["market"], label=txt(r.get("label")), p=p, odds=odds, priced=priced and not objetivo,
                book=book, b365_est=_v(r, "b365_est"), ev=ev, tier=tier, objetivo=objetivo or not priced,
                min_odds=_v(r, "min_odds") or math.ceil(100 / p - 1e-9) / 100, cur_odds=_v(r, "cur_odds"),
                cur_book=_v(r, "ref_book") if src == "mercado" else ("bet365" if src == "bet365" else None),
                conf=int(_v(r, "conf") or webdata.confidence_score(p, ev, _v(r, "n_books"), _v(r, "n_sharp"), priced,
                                                                   _v(r, "b365_age_h"), bool(_v(r, "derived")))),
                stake=stake, why=list(r.get("why") or []), elite=0, off=False)


def opp_from_pick(r: dict) -> dict:
    """Jugada registrada (tabla picks) → datos de tarjeta."""
    sx = SEL_IDX.get((r["match_id"], r["market"], r["selection"], float(r.get("point") or 0))) or {}
    book = txt(r.get("bookmaker"), "bet365")
    tier = r["tier"]
    priced = tier not in ("verificar", "revisar", "pronostico") or r.get("verified") == 1
    p, odds = float(r["p_final"]), float(r["odds"])
    ref = book if book != "bet365" else None
    nb = sx.get("n_books") if sx.get("n_books") is not None else r.get("n_books")
    why = report_for(r["match_id"], r["sport"], r["market"], r["selection"], r.get("point", 0), p,
                     odds if priced else None, priced, nb, sx.get("n_sharp"), bool(sx.get("derived")),
                     stored=str(r.get("reasons") or "").split(" | "), ref_book=ref, b365_est=sx.get("b365_est"))
    return dict(match_id=r["match_id"], sport=r["sport"], league=r["league"], kickoff=r["kickoff"], home=r["home"],
                away=r["away"], market=r["market"], label=txt(r.get("label")), p=p, odds=odds, priced=priced,
                book=book if priced else None, b365_est=sx.get("b365_est"), ev=_v(r, "ev"), tier=tier,
                objetivo=not priced, min_odds=_v(r, "min_odds") or math.ceil(100 / p - 1e-9) / 100, cur_odds=None,
                cur_book=None, conf=webdata.confidence_score(p, _v(r, "ev"), nb, sx.get("n_sharp"), priced,
                                                             sx.get("b365_age_h"), bool(sx.get("derived"))),
                stake=float(_v(r, "stake") or 0), why=[w for w in why if txt(w) != "—"], elite=int(r.get("elite") or 0),
                off=r.get("locked") == 1 and r.get("active") == 0, cur=_v(r, "current_odds"))


def price_src_html(o: dict) -> str:
    if o["objetivo"]:
        hoy = (f' · hoy paga <b>{num(o["cur_odds"])}</b> ({e(book_name(o["cur_book"]))})'
               if isnum(o.get("cur_odds")) and o.get("cur_book") else "")
        return f'<div class="src">Cógela en bet365 solo a <b>{num(o["min_odds"])}</b> o más{hoy}</div>'
    if o["book"] and o["book"] != "bet365":
        est = f' · bet365 ~<b>{num(o["b365_est"])}</b>' if isnum(o.get("b365_est")) else ""
        return (f'<div class="src">Cuota ref. mercado <b>{num(o["odds"])}</b> en <b>{e(book_name(o["book"]))}</b>{est}'
                f' · en bet365, a <b>{num(o["min_odds"])}</b> o más</div>')
    return '<div class="src">Cuota confirmada en <b>bet365</b></div>'


def opp_card(o: dict, big: bool = False) -> str:
    sp = SPORTS.get(o["sport"], SPORTS["football"])
    p, odds = o["p"], o["odds"]
    implied = 1 / odds if odds > 1 else 1
    badges = badge(o["tier"]) + ('<span class="bdg b-tg">Telegram</span>' if o.get("elite") else "") + \
        (f'<span class="bdg b-none">Ya no cumple · ahora {num(o.get("cur"))}</span>' if o.get("off") else "")
    size = 58 if big else 44
    odds_k = "cuota objetivo" if o["objetivo"] else ("bet365" if o["book"] == "bet365" else book_name(o["book"]))
    ev_txt = pct(o["ev"], True) if isnum(o["ev"]) and not o["objetivo"] else "≥ 0 %"
    why = "".join(f"<li>{e(txt(w))}</li>" for w in o["why"] if txt(w) != "—")
    pay = (f'<div class="pay">Stake sugerido <b>{num(o["stake"])} €</b> · con 10 € cobras <b>{num(10 * odds)} €</b></div>'
           if o["stake"] else f'<div class="pay">Con 10 € cobras <b>{num(10 * odds)} €</b></div>')
    return (
        f'<div class="opp t-{o["tier"]}"><div class="opp-top"><div>{badges}<div class="meta">{sp["icono"]} '
        f'{league_flag(o["league"])} {e(league_name(o["league"]))} · {when(o["kickoff"])}</div></div>'
        f'<div class="g">{gauge(o["conf"], size=70 if big else 60)}</div></div>'
        f'<div class="vs"><div class="tm">{crest(o["home"], size)}<span class="nm">{e(o["home"])}</span></div><div class="x">VS</div>'
        f'<div class="tm">{crest(o["away"], size)}<span class="nm">{e(o["away"])}</span></div></div>'
        f'<div class="pick"><div class="k">{e(MK.get(o["market"], o["market"]))}</div><div class="v">{e(o["label"])}</div></div>'
        f'<div class="nums"><div class="odd"><b>{"≥ " if o["objetivo"] else ""}{num(o["min_odds"] if o["objetivo"] else odds)}</b>'
        f'<span>{e(odds_k)}</span></div><div><b>{pct(p)}</b><span>prob. real</span></div>'
        f'<div><b class="{sign_cls(o["ev"]) if not o["objetivo"] else "pos"}">{ev_txt}</b><span>valor</span></div></div>'
        f'{price_src_html(o)}'
        f'<div class="cmp"><span>Prob. real</span><div class="tr"><i style="width:{min(p, 1) * 100:.1f}%;'
        f'background:linear-gradient(90deg,#14975A,#22D37F)"></i></div><span class="p">{pct(p)}</span>'
        f'<span>Mercado</span><div class="tr"><i style="width:{min(implied, 1) * 100:.1f}%;background:#5B6B80"></i></div>'
        f'<span class="p">{pct(implied)}</span></div>{pay}'
        + (f'<details><summary>Análisis táctico</summary><ul>{why}</ul></details>' if why else "") + '</div>')


def opp_rows(opps: list[dict]) -> str:
    out = []
    for o in opps:
        sp = SPORTS.get(o["sport"], SPORTS["football"])
        odds_v = f'≥ {num(o["min_odds"])}' if o["objetivo"] else num(o["odds"])
        odds_k = "objetivo bet365" if o["objetivo"] else ("bet365" if o["book"] == "bet365" else book_name(o["book"]))
        out.append(
            f'<div class="row"><div class="cr">{crest(o["home"], 34)}</div>'
            f'<div class="m"><b>{e(o["home"])} – {e(o["away"])}</b><span>{sp["icono"]} {e(league_name(o["league"]))} · '
            f'{when(o["kickoff"])}</span></div>'
            f'<div class="s"><b>{e(o["label"])}</b><span>{badge(o["tier"])}</span></div>'
            f'<div class="o">{odds_v}<small>{e(odds_k)}</small></div>'
            f'<div class="n n1">{pct(o["p"])}<small>prob. real</small></div>'
            f'<div class="n n2 {sign_cls(o["ev"]) if not o["objetivo"] else "pos"}">'
            f'{pct(o["ev"], True) if not o["objetivo"] else "≥ 0 %"}<small>valor</small></div>'
            f'<div class="n n3">{o["conf"]}<small>confianza</small></div></div>')
    return f'<div class="rows">{"".join(out)}</div>'


def opp_grid(opps: list[dict], big=False):
    if opps:
        st.markdown(f'<div class="opps">{"".join(opp_card(o, big) for o in opps)}</div>', unsafe_allow_html=True)


def stake_calculator(options: list[dict], key: str):
    """Calculadora: elige una apuesta y un importe → cobro, beneficio, valor esperado y stake sugerido para tu banca."""
    if not options:
        return
    with st.expander("🧮 Calculadora de stake y beneficio", expanded=False):
        c1, c2, c3 = st.columns([2.4, 1, 1])
        i = c1.selectbox("Apuesta", list(range(len(options))), format_func=lambda k: options[k]["name"], key=f"{key}_sel")
        stake = c2.number_input("Importe (€)", min_value=0.5, max_value=100000.0, value=10.0, step=0.5, key=f"{key}_st")
        bank = c3.number_input("Banca (€)", min_value=10.0, max_value=1_000_000.0, value=float(s.bankroll), step=10.0,
                               key=f"{key}_bk")
        o = options[i or 0]
        odds, p = float(o["odds"]), float(o["p"])
        ev = p * odds - 1
        sug, pct_, msg = assistant.stake_advice(ev, odds, bank, s.kelly_fraction, s.max_stake_pct)
        kpi_strip([("Cobras si aciertas", num(stake * odds) + " €", f"cuota {num(odds)}", "amber", "am"),
                   ("Beneficio", num(stake * (odds - 1)) + " €", f"probabilidad real {pct(p)}", "green", "pos"),
                   ("Valor esperado", num(stake * ev) + " €", f"{pct(ev, True)} por euro", "cyan", sign_cls(ev)),
                   ("Stake sugerido", num(sug) + " €", f"{pct(pct_, dec=1)} de la banca", "violet", "")])
        st.caption(msg)


# ======================================================================= combinadas
def combo_card(r: dict, hero_c: bool = False) -> str:
    legs = json.loads(r["legs"])
    t_txt, t_cls = COMBO_TIER.get(r["tier"], (txt(r["tier"]), "b-none"))
    pend = r["tier"] == "verificar"
    items = []
    for l in legs:
        unpriced = l.get("priced") is False
        rb = l.get("ref_book")
        lines = report_for(l["match_id"], l.get("sport", "football"), l["market"], l["selection"], l.get("point", 0),
                           l["p"], l.get("price"), not unpriced, stored=l.get("why"), ref_book=rb)
        why = "".join(f"<li>{e(txt(w))}</li>" for w in lines if txt(w) != "—")
        src = "objetivo bet365" if unpriced else (book_name(rb) if rb else "bet365")
        items.append(
            f'<li><span>{SPORTS.get(l.get("sport", "football"), SPORTS["football"])["icono"]} <b>{e(txt(l.get("label")))}</b>'
            f' <span class="lp">· {MK.get(l.get("market"), "")}</span>'
            f'<br><span class="lp">{e(txt(l.get("partido")))} · {e(league_name(l["league"]))} · {when(l["kickoff"])} · '
            f'prob. {pct(l["p"])}</span>'
            + (f'<details><summary>Por qué</summary><ul class="why">{why}</ul></details>' if why else "")
            + f'</span><span class="lo">{("≥ " + num(l.get("min_odds"))) if unpriced else num(l["price"])}'
              f'<small>{e(src)}</small></span></li>')
    implied = 1 / r["odds"] if r["odds"] > 1 else 1
    stake = float(r.get("stake") or 0)
    if not stake and not pend and r["ev"] >= -0.0005:
        stake = assistant.stake_advice(r["ev"], r["odds"], s.bankroll, s.kelly_fraction, s.max_stake_pct)[0]
    conf = webdata.confidence_score(r["prob"], r["ev"] if not pend else None, 3, 1, not pend)
    return (f'<div class="card{" hero-c" if hero_c else ""}"><div class="card-h"><span class="bdg {t_cls}">{t_txt}</span>'
            f'<span class="bdg b-risk">{COMBO_NAME.get(r["risk"], txt(r["risk"]))}</span>'
            f'<span class="w">{len(legs)} selecciones · empieza {when_nat(r["first_kickoff"])}</span>{gauge(conf)}</div>'
            f'<div class="big4"><div><b class="am">{"≥ " if pend else ""}{num(r["odds"])}</b><span>cuota total</span></div>'
            f'<div><b>{pct(r["prob"], dec=1)}</b><span>prob. real conjunta</span></div>'
            f'<div><b class="{"" if pend else sign_cls(r["ev"])}">{"—" if pend else pct(r["ev"], True)}</b>'
            f'<span>{"valor" if r["ev"] >= 0 else "margen de la casa"}</span></div>'
            f'<div><b>{num(stake) + " €" if stake else "—"}</b><span>stake sugerido</span></div>'
            f'<div><b>{num(10 * r["odds"])} €</b><span>cobras con 10 €</span></div></div>'
            f'<div class="cmp"><span>Prob. real</span><div class="tr"><i style="width:{min(r["prob"], 1) * 100:.1f}%;'
            f'background:linear-gradient(90deg,#14975A,#22D37F)"></i></div><span class="p">{pct(r["prob"])}</span>'
            f'<span>Cuota implica</span><div class="tr"><i style="width:{min(implied, 1) * 100:.1f}%;background:#5B6B80"></i>'
            f'</div><span class="p">{pct(implied)}</span></div>'
            f'<ul class="legs">{"".join(items)}</ul></div>')


# ======================================================================= tabla de la jornada
def day_table_widget(key: str):
    T = webdata.day_table(D["analysis"], NOW)
    n_wait = len(D["analysis"]) - len(T) if not D["analysis"].empty else 0
    if T.empty:
        empty("Todavía no hay partidos analizados. Aparecerán tras la próxima ejecución del radar (cada 3 horas).")
        return
    c1, c2, c3 = st.columns([2.2, 2.2, 1.6])
    sports = ["todos"] + [k for k in SPORTS if k in set(T["_sport"])]
    sp = c1.pills("Deporte", sports, default="todos", key=f"{key}_sp", label_visibility="collapsed",
                  format_func=lambda k: "Todos" if k == "todos" else f'{SPORTS[k]["icono"]} {SPORTS[k]["nombre"]}') or "todos"
    lv_opts = ["rec", "todos", "elite"]
    lv = c2.segmented_control("Nivel", lv_opts, default="todos", key=f"{key}_lv", label_visibility="collapsed",
                              format_func=lambda x: {"rec": "Solo recomendadas", "todos": "Todos los partidos",
                                                     "elite": "Élite y cuota alta"}[x]) or "todos"
    q = c3.text_input("Buscar", placeholder="Equipo o competición…", key=f"{key}_q", label_visibility="collapsed")
    if sp != "todos":
        T = T[T["_sport"] == sp]
    if lv == "rec":
        T = T[T["_tier"].isin(["valor", "cuota_alta", "solida", "moderado"])]
    elif lv == "elite":
        T = T[T["_tier"].isin(["valor", "cuota_alta"])]
    if q:
        ql = q.lower()
        T = T[T["Partido"].str.lower().str.contains(ql, regex=False) | T["Competición"].str.lower().str.contains(ql, regex=False)]
    if T.empty:
        empty("Ningún partido con estos filtros.")
        return
    show = T.drop(columns=["_tier", "_sport", "_kickoff"])
    st.dataframe(show, hide_index=True, width="stretch", height=min(640, 42 + 35 * len(show)), column_config={
        "Prob. real": st.column_config.ProgressColumn("Prob. real", format="%.0f%%", min_value=0, max_value=100),
        "Confianza": st.column_config.ProgressColumn("Confianza", format="%d", min_value=0, max_value=100),
        "Recomendación": st.column_config.TextColumn("Recomendación", width="large"),
        "Partido": st.column_config.TextColumn("Partido", width="medium"),
    })
    st.caption("Prob. real: probabilidad sin margen (Pinnacle/exchange o consenso de casas válidas). Cuota: bet365 si está "
               "confirmada; si no, la mejor cuota real de una casa generalista (se indica cuál). Valor = prob. × cuota − 1."
               + (f" {n_wait} partidos más aún sin cuotas: aparecerán en cuanto se publiquen." if n_wait > 0 else ""))


# ======================================================================= partidos
def _main_market(sport: str) -> str:
    return "1x2" if sport == "football" else "ml"


def _sel_rows(mid: str) -> pd.DataFrame:
    Sx = D.get("selections", pd.DataFrame())
    if Sx is None or Sx.empty:
        return pd.DataFrame()
    return Sx[Sx["match_id"] == mid]


def match_card(m: dict, a: dict | None) -> str:
    sp = SPORTS.get(m["sport"], SPORTS["football"])
    probs = ""
    if a and isnum(a.get("p_home")):
        parts = [a["p_home"], a.get("p_draw") if isnum(a.get("p_draw")) else 0, a["p_away"]]
        probs = '<div class="tri">' + "".join(f'<span style="width:{x * 100:.1f}%"></span>' for x in parts) + "</div>"
    ph = pct(a.get("p_home")) if a else "—"
    pa = pct(a.get("p_away")) if a else "—"
    rec = ""
    if a:
        price, srcb = a.get("price"), a.get("price_src")
        casa = "bet365" if srcb == "bet365" else book_name(a.get("ref_book")) if srcb == "mercado" else ""
        cuota = f' · <b class="am">{num(price)}</b> {e(casa)}' if isnum(price) and casa else ""
        rec = (f'<div class="mc-r"><span class="lt {txt(a.get("light"), "gris")}"></span> {badge(txt(a.get("tier"), "sin_datos"))} '
               f'<b>{e(txt(a.get("label"), "Sin selección todavía"))}</b>{cuota}<br>'
               f'<span class="mute">{e(txt(a.get("recommendation"), "Sin cuotas todavía"))}</span></div>')
    return (f'<div class="mc"><div class="mc-h"><span>{sp["icono"]} {league_flag(m["league"])} {e(league_name(m["league"]))}</span>'
            f'<span class="mono">{when(m["kickoff"])}</span></div>'
            f'<div class="mc-t">{crest(m["home"], 30)}<span class="nm">{e(m["home"])}</span><span class="pp">{ph}</span>'
            f'{crest(m["away"], 30)}<span class="nm">{e(m["away"])}</span><span class="pp">{pa}</span></div>{probs}{rec}</div>')


def _team_bars(stt: dict, home: str, away: str, sport: str) -> go.Figure | None:
    loc_, vis = (stt or {}).get("local") or {}, (stt or {}).get("visit") or {}
    if not (isnum(loc_.get("gf10")) and isnum(vis.get("gf10"))):
        return None
    unit = "goles" if sport == "football" else "puntos" if sport != "tennis" else "juegos"
    lr, vr = loc_.get("local") or {}, vis.get("visitante") or {}
    cats = [f"Anota (últ. 10)", f"Recibe (últ. 10)"]
    hv, av = [loc_["gf10"], loc_.get("ga10", 0)], [vis["gf10"], vis.get("ga10", 0)]
    if isnum(lr.get("gf")) and isnum(vr.get("gf")):
        cats += ["Anota (casa / fuera)", "Recibe (casa / fuera)"]
        hv += [lr["gf"], lr.get("ga", 0)]
        av += [vr["gf"], vr.get("ga", 0)]
    fig = go.Figure()
    fig.add_trace(go.Bar(name=home, x=cats, y=hv, marker_color=C["blue"], text=[num(v, 1) for v in hv],
                         textposition="outside", hovertemplate="%{x}: %{y:.2f}<extra>" + e(home) + "</extra>"))
    fig.add_trace(go.Bar(name=away, x=cats, y=av, marker_color=C["violet"], text=[num(v, 1) for v in av],
                         textposition="outside", hovertemplate="%{x}: %{y:.2f}<extra>" + e(away) + "</extra>"))
    fig.update_layout(barmode="group", title=dict(text=f"Ataque y defensa · {unit} por partido", x=0, font=dict(size=13)))
    return fig


def _streak_fig(stt: dict, home: str, away: str) -> go.Figure | None:
    loc_, vis = (stt or {}).get("local") or {}, (stt or {}).get("visit") or {}
    fh, fa = txt(loc_.get("forma"), ""), txt(vis.get("forma"), "")
    if not fh and not fa:
        return None
    col = {"V": C["green"], "E": "#8592A3", "D": C["red"]}
    fig = go.Figure()
    for team, f in ((away, fa), (home, fh)):
        f = [c for c in f if c in col][-5:]
        fig.add_trace(go.Scatter(x=list(range(1, len(f) + 1)), y=[team] * len(f), mode="markers+text", text=f,
                                 textfont=dict(color="#06080C", size=13, family="JetBrains Mono"),
                                 marker=dict(symbol="square", size=34, color=[col[c] for c in f]),
                                 hovertemplate="%{text}<extra>" + e(team) + "</extra>", showlegend=False))
    fig.update_xaxes(showgrid=False, tickvals=[1, 2, 3, 4, 5], ticktext=["−5", "−4", "−3", "−2", "último"], range=[0.4, 5.6])
    fig.update_yaxes(showgrid=False)
    fig.update_layout(title=dict(text="Racha · últimos 5 partidos", x=0, font=dict(size=13)))
    return fig


def _score_fig(a: dict, info: dict, sel: pd.DataFrame, home: str, away: str) -> go.Figure | None:
    lam, mu = info.get("xg_local"), info.get("xg_visit")
    if not (isnum(lam) and isnum(mu)):
        if not (a and isnum(a.get("p_home")) and isnum(a.get("p_draw")) and isnum(a.get("p_away"))):
            return None
        over = None
        if sel is not None and not sel.empty:
            o = sel[(sel["market"] == "ou") & (sel["selection"] == "O") & (sel["point"] == 2.5)]
            over = float(o["p_final"].iloc[0]) if not o.empty else None
        try:
            lam, mu = fit_goal_rates(float(a["p_home"]), float(a["p_draw"]), float(a["p_away"]), over)
        except Exception:  # noqa: BLE001
            return None
    mat = score_matrix(float(lam), float(mu), RHO)[:6, :6]
    z = (mat * 100).round(1)
    fig = go.Figure(go.Heatmap(z=z, x=[str(i) for i in range(6)], y=[str(i) for i in range(6)],
                               colorscale=[[0, "#0D131C"], [0.35, "#1E3A5F"], [0.7, "#2F7BD9"], [1, "#FFB020"]],
                               text=[[f"{v:.0f}%" if v >= 1 else "" for v in row] for row in z], texttemplate="%{text}",
                               showscale=False, hovertemplate=e(home) + " %{y} - %{x} " + e(away) + ": %{z:.1f}%<extra></extra>"))
    fig.update_xaxes(title=f"Goles {away}", side="bottom", showgrid=False)
    fig.update_yaxes(title=f"Goles {home}", showgrid=False, autorange="reversed")
    i, j = np.unravel_index(np.argmax(mat), mat.shape)
    fig.update_layout(title=dict(text=f"Marcadores más probables · {i}-{j} ({mat[i, j]:.0%})  ·  goles esperados "
                                      f"{num(lam, 2)} - {num(mu, 2)}", x=0, font=dict(size=13)))
    return fig


def _margin_fig(a: dict, info: dict, league: str, home: str, away: str) -> go.Figure | None:
    from scipy.stats import norm
    sd = US_SD.get(league, US_SD["nba"])[0]
    mean = info.get("margen")
    if not isnum(mean):
        if not (a and isnum(a.get("p_home"))):
            return None
        mean = sd * norm.ppf(min(max(float(a["p_home"]), 0.01), 0.99))
    x = np.linspace(mean - 3.2 * sd, mean + 3.2 * sd, 140)
    y = norm.pdf(x, mean, sd) * 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x[x <= 0], y=y[x <= 0], fill="tozeroy", mode="lines", line=dict(color=C["violet"]),
                             fillcolor="rgba(167,139,250,.25)", name=f"Gana {away}", hovertemplate="%{x:.0f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=x[x >= 0], y=y[x >= 0], fill="tozeroy", mode="lines", line=dict(color=C["blue"]),
                             fillcolor="rgba(91,140,255,.28)", name=f"Gana {home}", hovertemplate="%{x:.0f}<extra></extra>"))
    fig.add_vline(x=0, line_color=C["mute"], line_dash="dot")
    fig.update_xaxes(title=f"Margen final (puntos de {home})")
    fig.update_yaxes(showticklabels=False)
    fig.update_layout(title=dict(text=f"Distribución del margen · esperado {mean:+.1f}", x=0, font=dict(size=13)))
    return fig


def _prob_fig(m: dict, a: dict | None, sel: pd.DataFrame) -> go.Figure | None:
    if not a or not isnum(a.get("p_home")):
        return None
    main = _main_market(m["sport"])
    names = {"H": m["home"], "D": "Empate", "A": m["away"]}
    keys = ["H", "D", "A"] if main == "1x2" and isnum(a.get("p_draw")) else ["H", "A"]
    real = [float(a["p_home"] if k == "H" else a["p_draw"] if k == "D" else a["p_away"]) for k in keys]
    impl = []
    for k in keys:
        r = sel[(sel["market"] == main) & (sel["selection"] == k)] if sel is not None and not sel.empty else pd.DataFrame()
        pr = r["price"].iloc[0] if not r.empty and isnum(r["price"].iloc[0]) else None
        impl.append(1 / float(pr) if pr else None)
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Probabilidad real", x=[names[k] for k in keys], y=real, marker_color=C["green"],
                         text=[pct(v) for v in real], textposition="outside", hovertemplate="%{x}: %{y:.1%}<extra></extra>"))
    if any(v is not None for v in impl):
        fig.add_trace(go.Bar(name="Implícita en la cuota", x=[names[k] for k in keys], y=[v or 0 for v in impl],
                             marker_color="#5B6B80", text=[pct(v) for v in impl], textposition="outside",
                             hovertemplate="%{x}: %{y:.1%}<extra></extra>"))
    fig.update_yaxes(tickformat=".0%", range=[0, max(real + [v or 0 for v in impl]) * 1.25])
    fig.update_layout(barmode="group", title=dict(text="Probabilidad real frente a la del mercado", x=0, font=dict(size=13)))
    return fig


def injuries_block(info: dict, home: str, away: str) -> str:
    b = (info or {}).get("bajas") or {}

    def fmt(lst):
        if not lst:
            return '<span class="mute">Sin bajas confirmadas</span>'
        return "<br>".join(f'• {e(txt(x.get("jugador")))} <span class="mute">({e(txt(x.get("tipo"), "baja"))}'
                           + (f': {e(txt(x.get("motivo"), ""))}' if txt(x.get("motivo"), "") else "") + ")</span>"
                           for x in lst[:8])
    src = txt(b.get("fuente"), "")
    note = txt(b.get("nota"), "sin fuente de bajas para esta competición")
    head = f'Fuente: <b>{e(src)}</b>' if src else f'Sin datos de bajas para este partido: {e(note)}'
    return (f'<p class="hint" style="margin:.2rem 0 .5rem">{head}</p><div class="inj"><div><b>{e(home)}</b>'
            f'{fmt(b.get("local") or [])}</div><div><b>{e(away)}</b>{fmt(b.get("visit") or [])}</div></div>')


def markets_table(m: dict, sel: pd.DataFrame) -> str:
    if sel is None or sel.empty:
        return ""
    unit = SPORTS[m["sport"]]["unidad"]
    rows = []
    for r in sel.sort_values(["market", "point", "selection"]).to_dict("records"):
        p = r.get("p_final")
        src = r.get("price_src")
        price = r.get("price") if src in ("bet365", "mercado") else None
        casa = "bet365" if src == "bet365" else book_name(r.get("ref_book")) if src == "mercado" else "—"
        ev = p * price - 1 if isnum(p) and isnum(price) else None
        tier = txt(r.get("tier"), "pronostico")
        rows.append(f'<tr class="{"v" if isnum(ev) and ev >= 0 else ""}"><td>{e(MK.get(r["market"], r["market"]))} · '
                    f'{e(txt(r.get("label")) if txt(r.get("label")) != "—" else mlabel(r["market"], r["selection"], r["point"], m["home"], m["away"], unit))}</td>'
                    f'<td>{pct(p)}</td><td>{num(1 / p) if isnum(p) and p else "—"}</td><td>{num(price)}</td>'
                    f'<td style="font-family:var(--sans)">{e(casa)}</td><td>{pct(ev, True)}</td>'
                    f'<td style="font-family:var(--sans)">{badge(tier) if tier in TIER and tier != "pronostico" else ""}</td></tr>')
    return (f'<div class="scroll"><table class="mk"><tr><th>Mercado</th><th>Prob.</th><th>Justa</th><th>Cuota</th>'
            f'<th>Casa</th><th>Valor</th><th>Nivel</th></tr>{"".join(rows)}</table></div>')


def match_detail(mid: str):
    m = MATCHES.get(mid)
    if m is None:
        empty("Este partido ya no está en la lista de próximos.")
        return
    A = D["analysis"]
    a = A[A["match_id"] == mid].iloc[0].to_dict() if not A.empty and (A["match_id"] == mid).any() else None
    info = D["info"].get(mid) or {}
    stt = info.get("stats") or {}
    sel = _sel_rows(mid)
    sp = SPORTS.get(m["sport"], SPORTS["football"])
    conf = webdata.confidence_score(a.get("p_final"), a.get("ev"), a.get("n_books"), a.get("n_sharp"),
                                    a.get("price_src") in ("bet365", "mercado"), a.get("b365_age_h")) \
        if a and isnum(a.get("p_final")) else 0
    loc_, vis = stt.get("local") or {}, stt.get("visit") or {}
    sub_h = f'{form_html(loc_.get("forma"))}' if loc_.get("forma") else ""
    sub_a = f'{form_html(vis.get("forma"))}' if vis.get("forma") else ""
    rec = ""
    if a:
        rec = (f'<div style="margin-top:.9rem"><span class="lt {txt(a.get("light"), "gris")}"></span> {badge(txt(a.get("tier"), "sin_datos"))}'
               f' <b>{e(txt(a.get("label"), "Sin selección todavía"))}</b> · {e(txt(a.get("recommendation"), ""))}</div>')
    st.markdown(
        f'<div class="det"><div class="det-vs"><div class="tm">{crest(m["home"], 72)}<div><b>{e(m["home"])}</b>'
        f'<small>Local · prob. {pct(a.get("p_home")) if a else "—"}</small>{sub_h}</div></div>'
        f'<div class="mid">{sp["icono"]} {league_flag(m["league"])} {e(league_name(m["league"]))}<b>{loc(m["kickoff"]):%H:%M}</b>'
        f'{when_nat(m["kickoff"])}<div style="margin-top:.4rem">{gauge(conf, size=78)}</div></div>'
        f'<div class="tm r">{crest(m["away"], 72)}<div><b>{e(m["away"])}</b><small>Visitante · prob. '
        f'{pct(a.get("p_away")) if a else "—"}</small>{sub_a}</div></div></div>{rec}</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    f1 = _prob_fig(m, a, sel)
    f2 = _score_fig(a, info, sel, m["home"], m["away"]) if m["sport"] == "football" else \
        _margin_fig(a, info, m["league"], m["home"], m["away"]) if m["sport"] in ("basketball", "nfl") else None
    with c1:
        if f1 is not None:
            st.plotly_chart(chart_layout(f1, 320), width="stretch", config=PLOT_CFG)
        else:
            empty("Sin probabilidades todavía para este partido.")
    with c2:
        if f2 is not None:
            st.plotly_chart(chart_layout(f2, 320, legend=m["sport"] != "football"), width="stretch", config=PLOT_CFG)
        elif m["sport"] == "tennis" and info.get("elo_local"):
            fe = go.Figure(go.Bar(x=[m["home"], m["away"]], y=[info["elo_local"], info["elo_visit"]],
                                  marker_color=[C["blue"], C["violet"]], text=[f'{info["elo_local"]:.0f}', f'{info["elo_visit"]:.0f}'],
                                  textposition="outside"))
            fe.update_layout(title=dict(text=f'Elo · superficie: {txt(info.get("superficie"), "")}', x=0, font=dict(size=13)))
            st.plotly_chart(chart_layout(fe, 320, legend=False), width="stretch", config=PLOT_CFG)
        else:
            empty("Sin modelo de marcador para este partido.")
    c3, c4 = st.columns(2)
    f3, f4 = _team_bars(stt, m["home"], m["away"], m["sport"]), _streak_fig(stt, m["home"], m["away"])
    with c3:
        if f3 is not None:
            st.plotly_chart(chart_layout(f3, 300), width="stretch", config=PLOT_CFG)
        else:
            empty("Sin histórico propio de ataque y defensa para estos equipos (competición sin datos en nuestras fuentes).")
    with c4:
        if f4 is not None:
            st.plotly_chart(chart_layout(f4, 300, legend=False), width="stretch", config=PLOT_CFG)
        else:
            empty("Sin racha reciente disponible.")
    section("Bajas y lesiones")
    st.markdown(injuries_block(info, m["home"], m["away"]), unsafe_allow_html=True)
    lines = []
    if a and isnum(a.get("p_final")) and txt(a.get("market"), "") and txt(a.get("selection"), ""):
        src = a.get("price_src")
        lines = report_for(mid, m["sport"], a["market"], a["selection"], a.get("point") or 0, a["p_final"],
                           a.get("price"), src in ("bet365", "mercado"), a.get("n_books"), a.get("n_sharp"),
                           ref_book=a.get("ref_book") if src == "mercado" else None, b365_est=a.get("b365_est"))
    if lines:
        section("Análisis")
        st.markdown('<ul class="why" style="font-size:.92rem;color:var(--text)">' +
                    "".join(f"<li>{e(txt(x))}</li>" for x in lines if txt(x) != "—") + "</ul>", unsafe_allow_html=True)
    tb = markets_table(m, sel)
    if tb:
        section("Todos los mercados", len(sel))
        st.markdown(tb, unsafe_allow_html=True)


def standings(league: str):
    t = D["tables"].get(league)
    if not t:
        empty("Sin clasificación todavía para esta competición.")
        return
    kind = "football" if league in FOOTBALL else "tennis" if league.startswith("tennis_") else "us"
    if kind == "football":
        mx_a = max((r.get("ataque", 0) for r in t), default=1) or 1
        head = ('<tr><th>#</th><th class="l">Equipo</th><th>PJ</th><th>G</th><th>E</th><th>P</th><th>GF</th><th>GC</th>'
                '<th>DG</th><th>Pts</th><th>Local</th><th>Fuera</th><th class="l">Forma</th><th>Racha</th>'
                '<th class="l">Ataque</th><th class="l">Defensa</th></tr>')
        body = "".join(
            f'<tr><td>{r["pos"]}</td><td class="l b">{e(r["equipo"])}</td><td>{r["pj"]}</td><td>{r["g"]}</td>'
            f'<td>{r["e"]}</td><td>{r["p"]}</td><td>{r["gf"]}</td><td>{r["gc"]}</td><td>{r["dg"]:+d}</td>'
            f'<td class="b">{r["pts"]}</td><td>{r.get("pts_l", "—")}/{r.get("pj_l", "—")}</td>'
            f'<td>{r.get("pts_v", "—")}/{r.get("pj_v", "—")}</td><td class="l">{form_html(r["forma"])}</td>'
            f'<td>{e(str(r.get("racha", "")))}</td>'
            f'<td class="l"><span class="bar" style="width:{60 * r.get("ataque", 0) / mx_a:.0f}px"></span></td>'
            f'<td class="l"><span class="bar" style="background:#2BD47D;width:{60 * min(r.get("defensa", 0), 2) / 2:.0f}px">'
            f'</span></td></tr>' for r in t)
    elif kind == "tennis":
        head = ('<tr><th>#</th><th class="l">Jugador</th><th>Elo</th><th>Dura</th><th>Tierra</th><th>Hierba</th>'
                '<th>G-P (12 m)</th><th class="l">Forma</th></tr>')
        body = "".join(
            f'<tr><td>{r["pos"]}</td><td class="l b">{e(r["jugador"])}</td><td class="b">{r["elo"]}</td><td>{r["dura"]}</td>'
            f'<td>{r["tierra"]}</td><td>{r["hierba"]}</td><td>{r["g"]}-{r["p"]}</td><td class="l">{form_html(r["forma"])}'
            f'</td></tr>' for r in t)
    else:
        head = ('<tr><th>#</th><th class="l">Equipo</th><th>Elo</th><th>G</th><th>P</th><th>%</th><th>Local</th>'
                '<th>Fuera</th><th>A favor</th><th>En contra</th><th class="l">Forma</th></tr>')
        body = "".join(
            f'<tr><td>{r["pos"]}</td><td class="l b">{e(r["equipo"])}</td><td class="b">{r["elo"]}</td><td>{r["g"]}</td>'
            f'<td>{r["p"]}</td><td>{pct(r["pct"])}</td><td>{r.get("local", "—")}</td><td>{r.get("visitante", "—")}</td>'
            f'<td>{num(r["pf"], 1)}</td><td>{num(r["pa"], 1)}</td>'
            f'<td class="l">{form_html(r["forma"])}</td></tr>' for r in t)
    st.markdown(f'<div class="scroll"><table class="tb">{head}{body}</table></div>', unsafe_allow_html=True)




# ======================================================================= páginas
def page_home():
    hero("CENTRO DE MANDO", "Las mejores apuestas del día",
         "Probabilidad real sin margen frente a la mejor cuota disponible, en fútbol, NBA, NFL y tenis.")
    opps = [opp_from_pool(r) for r in DAILY.to_dict("records")] if not DAILY.empty else []
    rec = [o for o in opps if not o["objetivo"]]
    cnt = {t: sum(o["tier"] == t for o in rec) for t in ("valor", "cuota_alta", "solida", "moderado")}
    sm = webdata.day_summary(D["analysis"])
    avg_ev = np.mean([o["ev"] for o in rec if isnum(o["ev"])]) if rec else None
    kpi_strip([
        ("Recomendaciones del día", str(len(opps)), f'{len(rec)} con cuota operable · {len(opps) - len(rec)} con cuota objetivo',
         "amber", "am"),
        ("💎 Élite", str(cnt["valor"]), f"valor ≥ {pct(s.min_ev, True)} con referencia sharp", "amber", ""),
        ("🔥 Cuota alta", str(cnt["cuota_alta"]), f"{num(s.high_odds_min)}-{num(s.high_odds_max)} con valor", "orange", ""),
        ("🛡️ Sólidas", str(cnt["solida"]), f"prob. ≥ {pct(s.solid_min_prob)} a precio justo", "green", ""),
        ("⚖️ Valor moderado", str(cnt["moderado"]), "ventaja ligera", "cyan", ""),
        ("Valor medio", pct(avg_ev, True) if isnum(avg_ev) else "—", f'{sm["partidos"]} partidos analizados', "violet",
         sign_cls(avg_ev))])
    section("Mejores oportunidades del día", min(4, len(opps)) if opps else None,
            "Las cuatro selecciones con mejor nivel y confianza de las próximas 48 horas. Abre «Análisis táctico» para ver "
            "el porqué: forma, local/visitante, cara a cara, bajas y precio.")
    if not opps:
        empty("Ahora mismo no hay selecciones analizadas para las próximas 48 horas. El radar se actualiza cada 3 horas.")
    else:
        opp_grid(opps[:4], big=True)
    c1, c2 = st.columns([1.05, 1])
    with c1:
        section("Combinada recomendada del día")
        cb = webdata.combo_of_day(D["combos"])
        if cb:
            c1.markdown(combo_card(cb, hero_c=True), unsafe_allow_html=True)
        else:
            empty("No hay partidos suficientes en las próximas 36 horas para una combinada. Prueba el Asistente.")
    with c2:
        section("Más recomendaciones", max(0, len(opps) - 4) if len(opps) > 4 else None)
        if len(opps) > 4:
            c2.markdown(opp_rows(opps[4:]), unsafe_allow_html=True)
        else:
            empty("Las recomendaciones del día están arriba. Mira toda la jornada en la tabla.")
    section("Toda la jornada", len(D["analysis"]) if not D["analysis"].empty else 0,
            "Un partido por fila con su selección más interesante. Ordena por cualquier columna o filtra por deporte y nivel.")
    day_table_widget("t")


def page_picks():
    hero("JUGADAS", "Jugadas por nivel", "Élite, cuota alta, sólidas y valor moderado, más las pendientes de confirmar.")
    P = D["picks"]
    if P.empty:
        empty("No hay jugadas registradas para los próximos días. El radar se actualiza cada 3 horas.")
        return
    c1, c2 = st.columns([2, 1])
    sports = ["todos"] + [k for k in SPORTS if k in set(P["sport"])]
    sp = c1.pills("Deporte", sports, default="todos", key="p_sp", label_visibility="collapsed",
                  format_func=lambda k: "Todos" if k == "todos" else f'{SPORTS[k]["icono"]} {SPORTS[k]["nombre"]}') or "todos"
    order = c2.segmented_control("Orden", ["calidad", "hora", "prob"], default="calidad", key="p_o",
                                 label_visibility="collapsed",
                                 format_func=lambda o: {"calidad": "Calidad", "hora": "Hora", "prob": "Probabilidad"}[o])
    if sp != "todos":
        P = P[P["sport"] == sp]
    P = P.sort_values({"calidad": "score", "hora": "kickoff", "prob": "p_final"}.get(order or "calidad", "score"),
                      ascending=(order == "hora"))
    counts = P["tier"].value_counts()
    rec = P[P["tier"].isin(["valor", "solida", "moderado", "cuota_alta"])]
    stake_calculator([dict(name=f'{TIER.get(r["tier"], (r["tier"],))[0]} · {txt(r["label"])} ({r["home"]} – {r["away"]}) · '
                                f'{num(r["odds"])}', odds=r["odds"], p=r["p_final"]) for r in rec.to_dict("records")],
                     "calc_p")
    tier_list = ["valor", "cuota_alta", "solida", "moderado", "verificar", "revisar"]
    tabs = st.tabs([f'{TAB_NAME[t]} · {counts.get(t, 0)}' for t in tier_list])
    texts = {
        "valor": ("La cuota supera el precio justo limpio en al menos un " + pct(s.min_ev, True) + ", con referencia sharp "
                  "(Pinnacle o exchange). Si la cuota es de otra casa, se indica cuál y a cuánto cogerla en bet365.",
                  "Ahora mismo ninguna cuota supera el precio justo con margen suficiente. Mira las sólidas."),
        "cuota_alta": (f"Cuota {num(s.high_odds_min)}-{num(s.high_odds_max)} con al menos un {pct(s.high_min_ev, True)} sobre el "
                       "precio justo: más varianza, stake pequeño (Kelly fraccionado).", "Ninguna cuota alta con valor ahora mismo."),
        "solida": ("Probabilidad real alta y la cuota paga al menos el precio justo. No es ventaja: es no regalar margen. "
                   "Son la base de las combinadas.", "No hay sólidas para los próximos días."),
        "moderado": ("Ventaja ligera (entre 0 % y el umbral élite) frente al precio justo limpio.",
                     "Ninguna selección con ventaja ligera ahora mismo."),
        "verificar": ("El mercado sugiere valor pero no hay cuota operable reciente: apuesta solo si bet365 paga la "
                      "cuota mínima indicada.", "Nada pendiente de confirmar."),
        "revisar": ("Valor esperado anormalmente alto: casi siempre es una cuota mal publicada. Nunca se apuesta.",
                    "Ninguna anomalía detectada."),
    }
    for tab, tier in zip(tabs, tier_list):
        with tab:
            hint, msg = texts[tier]
            st.markdown(f'<p class="hint">{hint}</p>', unsafe_allow_html=True)
            sub = P[P["tier"] == tier]
            if sub.empty:
                empty(msg)
            else:
                opp_grid([opp_from_pick(r) for r in sub.head(60).to_dict("records")])


def page_combos():
    hero("COMBINADAS", "Combinadas listas para jugar", "Tres perfiles cada jornada, con el porqué de cada selección.")
    Cb = D["combos"]
    if Cb.empty:
        empty("No hay partidos suficientes en las próximas 36 horas para montar combinadas. Prueba el Asistente.")
        return
    cb = webdata.combo_of_day(Cb)
    section("Combinada recomendada del día")
    st.markdown(combo_card(cb, hero_c=True), unsafe_allow_html=True)
    st.markdown('<p class="hint" style="margin-top:.8rem"><b>Segura</b>: cuota 1,70-2,20 con las selecciones más fiables. '
                '<b>Equilibrada</b>: 2,50-4,00 mezclando deportes y mercados. <b>Valor EV+</b>: solo cuando hay patas con '
                'ventaja. La probabilidad conjunta es el producto de las de cada pata. Si una pata no tiene cuota operable, '
                'sale su cuota mínima (≥).</p>', unsafe_allow_html=True)
    stake_calculator([dict(name=f'{COMBO_NAME.get(r["risk"], r["risk"])} · {COMBO_TIER.get(r["tier"], (r["tier"],))[0]} · '
                                f'cuota {num(r["odds"])} · {len(json.loads(r["legs"]))} patas', odds=r["odds"], p=r["prob"])
                      for r in Cb.to_dict("records")], "calc_c")
    profiles = [p for p in ("segura", "equilibrada", "valor") if p in set(Cb["risk"])] + \
        [p for p in Cb["risk"].unique() if p not in ("segura", "equilibrada", "valor")]
    tabs = st.tabs([f'{COMBO_NAME.get(p, p)} · {int((Cb["risk"] == p).sum())}' for p in profiles])
    for tab, prof in zip(tabs, profiles):
        with tab:
            g = Cb[Cb["risk"] == prof].sort_values("ev", ascending=False)
            st.markdown('<div class="opps">' + "".join(combo_card(r) for r in g.to_dict("records")) + "</div>",
                        unsafe_allow_html=True)


def page_matches():
    hero("PARTIDOS", "Fichas de partido", "Probabilidades, marcadores, ataque y defensa, racha, bajas y todos los mercados.")
    M = D["matches"]
    sports = [k for k in SPORTS if k in s.active_sports and (M.empty or k in set(M["sport"]))] or \
        [k for k in SPORTS if k in s.active_sports]
    sp = st.pills("Deporte", sports, default=sports[0] if sports else None, key="m_sp", label_visibility="collapsed",
                  format_func=lambda k: f'{SPORTS[k]["icono"]} {SPORTS[k]["nombre"]}') or (sports[0] if sports else "football")
    Ms = M[M["sport"] == sp] if not M.empty else M
    A = {r["match_id"]: r for r in D["analysis"].to_dict("records")} if not D["analysis"].empty else {}
    t_f, t_p, t_c = st.tabs(["Ficha del partido", "Todos los partidos", "Clasificación"])
    with t_f:
        if Ms.empty:
            empty(f"No hay partidos de {SPORTS[sp]['nombre'].lower()} en los próximos días.")
        else:
            rank = {"valor": 0, "cuota_alta": 1, "solida": 2, "moderado": 3, "verificar": 4, "pronostico": 5, "revisar": 6}
            ids = sorted(Ms["id"], key=lambda i: (rank.get((A.get(i) or {}).get("tier"), 9), MATCHES[i]["kickoff"]))
            mid = st.selectbox("Partido", ids, key=f"m_det_{sp}",
                               format_func=lambda i: f'{when(MATCHES[i]["kickoff"])} · {MATCHES[i]["home"]} – '
                                                     f'{MATCHES[i]["away"]} · {league_name(MATCHES[i]["league"])}')
            match_detail(mid)
    with t_p:
        if Ms.empty:
            empty(f"No hay partidos de {SPORTS[sp]['nombre'].lower()} en los próximos días.")
        else:
            counts = Ms["league"].value_counts()
            order = sorted(counts.index, key=lambda k: (FOOTBALL.get(k, {}).get("tier", 1), -counts[k]))
            lg = st.selectbox("Competición", ["todas"] + order, key=f"m_lg_{sp}",
                              format_func=lambda k: f"Todas ({len(Ms)})" if k == "todas" else
                              f"{league_flag(k)} {league_name(k)} ({counts[k]})")
            Ml = (Ms if lg == "todas" else Ms[Ms["league"] == lg]).head(90)
            Ml = Ml.assign(_d=Ml["kickoff"].map(lambda k: loc(k).date()))
            for d, g in Ml.groupby("_d", sort=True):
                section(day_title(d), len(g))
                st.markdown('<div class="mcs">' + "".join(match_card(m, A.get(m["id"])) for m in g.to_dict("records"))
                            + "</div>", unsafe_allow_html=True)
    with t_c:
        if sp == "tennis":
            a, w = st.tabs(["Ranking ATP", "Ranking WTA"])
            with a:
                standings("tennis_atp")
            with w:
                standings("tennis_wta")
        else:
            keys = [k for k in D["tables"] if (k in FOOTBALL and sp == "football") or (OTHER.get(k, {}).get("sport") == sp)]
            if not keys:
                empty("Sin clasificaciones todavía.")
            else:
                keys = sorted(keys, key=lambda k: (FOOTBALL.get(k, {}).get("tier", 1), league_name(k)))
                pick = st.selectbox("Competición", keys, format_func=lambda k: f"{league_flag(k)} {league_name(k)}",
                                    key=f"m_tb_{sp}")
                standings(pick)


def _risk_changed():
    st.session_state["as_odds"] = tuple(assistant.RISK_RANGE[st.session_state.get("as_risk") or "medio"])


ROLE_TITLE = {"conservadora": "🛡️ Conservadora", "equilibrada": "⚖️ Equilibrada", "retorno": "🚀 Alto retorno",
              "cercana": "Alternativa más cercana", "principal": "⚖️ Equilibrada"}
STATUS_BADGE = {"exacta": ("Cuotas operables", "b-solid"), "por_confirmar": ("Con cuota objetivo", "b-check"),
                "cercana": ("Fuera del filtro", "b-warn")}


def plan_card(p: assistant.Plan, n: int) -> str:
    title = "Simple" if p.tipo == "simple" else f"Combinada de {len(p.legs)}"
    rows = []
    for l in p.legs:
        sp = SPORTS.get(l["sport"], SPORTS["football"])["icono"]
        why = "".join(f"<li>{e(txt(w))}</li>" for w in (l.get("why") or []) if txt(w) != "—")
        unpriced = l.get("priced") is False
        rb = l.get("ref_book") if isinstance(l.get("ref_book"), str) else None
        src = "objetivo bet365" if unpriced else (book_name(rb) if rb else "bet365")
        odds_txt = f'≥ {num(l.get("min_odds") or l["odds"])}' if unpriced else num(l["odds"])
        hoy = (f' · hoy paga {num(l["cur_odds"])} en {e(book_name(l.get("cur_book")))}'
               if unpriced and isnum(l.get("cur_odds")) else "")
        rows.append(f'<li><span>{sp} <b>{e(txt(l["label"]))}</b> <span class="lp">· {MK.get(l.get("market"), "")}</span><br>'
                    f'<span class="lp">{e(l["home"])} – {e(l["away"])} · {when_nat(l["kickoff"])} · prob. {pct(l["p_final"])}'
                    + (hoy if unpriced else f' · valor {pct(l["ev"], True)}') + '</span>'
                    + (f'<details><summary>Por qué apostar</summary><ul class="why">{why}</ul></details>' if why else "")
                    + f'</span><span class="lo">{odds_txt}<small>{e(src)}</small></span></li>')
    stt, stc = STATUS_BADGE.get(p.status, (p.status, "b-none"))
    rat = "".join(f"<li>{e(x)}</li>" for x in p.rationale)
    pend = p.status == "por_confirmar"
    return (f'<div class="plan{" mainp" if p.role in ("equilibrada", "principal") else ""}"><h4>{ROLE_TITLE.get(p.role, p.role)}</h4>'
            f'<div class="sub">{title} · se resuelve {when_nat(p.ends)}</div>'
            f'<div style="margin:.45rem 0"><span class="bdg {stc}">{stt}</span>'
            + ("" if p.in_range else ' <span class="bdg b-warn">fuera del rango pedido</span>') + '</div>'
            f'<div class="big4"><div><b class="am">{"≥ " if pend else ""}{num(p.odds)}</b><span>cuota total</span></div>'
            f'<div><b>{pct(p.prob)}</b><span>prob. real</span></div>'
            f'<div><b class="{sign_cls(p.ev)}">{pct(p.ev, True)}</b><span>valor</span></div>'
            f'<div><b>{num(p.budget)} €</b><span>stake ({pct(p.stake_pct, dec=1)})</span></div>'
            f'<div><b>{num(p.budget * p.odds)} €</b><span>cobras si aciertas</span></div></div>'
            + (f'<ul class="why">{rat}</ul>' if rat else "") + f'<ul class="legs">{"".join(rows)}</ul></div>')


def page_assistant():
    hero("ASISTENTE", "Tu copiloto de apuestas",
         "Banca, riesgo, deportes y cuota objetivo: tres propuestas jugables con stake en euros y el porqué de cada pata.")
    now = NOW
    if "as_odds" not in st.session_state:
        st.session_state["as_odds"] = tuple(assistant.RISK_RANGE["medio"])
    c1, c2, c3 = st.columns(3)
    bankroll = c1.number_input("Banca (€)", min_value=10.0, max_value=1_000_000.0, value=float(s.bankroll), step=10.0,
                               key="as_bank")
    dl_date = c2.date_input("Que se resuelva antes del día", value=(datetime.now(TZ) + timedelta(days=2)).date(),
                            min_value=datetime.now(TZ).date(), format="DD/MM/YYYY", key="as_date")
    dl_time = c3.time_input("a las", value=time(23, 59), step=timedelta(minutes=30), key="as_time")
    c4, c5 = st.columns(2)
    risk = c4.segmented_control("Riesgo", ["bajo", "medio", "alto"], default="medio", key="as_risk",
                                on_change=_risk_changed, format_func=lambda r: {
                                    "bajo": "Bajo · 1,40-1,90", "medio": "Medio · 1,90-2,80",
                                    "alto": "Alto · 2,80-5,00"}[r]) or "medio"
    tipo = c5.segmented_control("Tipo", ["auto", "simple", "combinada"], default="auto", key="as_tipo",
                                format_func=lambda t: {"auto": "Lo mejor", "simple": "Simple",
                                                       "combinada": "Combinada"}[t]) or "auto"
    odds_range = st.slider("Cuota total objetivo (el riesgo la ajusta; puedes afinarla)", 1.20, 8.00, step=0.05,
                           key="as_odds")
    c6, c7 = st.columns([2, 1])
    sports = c6.multiselect("Deportes", [k for k in SPORTS if k in s.active_sports],
                            default=[k for k in SPORTS if k in s.active_sports], key="as_sports",
                            format_func=lambda k: f'{SPORTS[k]["icono"]} {SPORTS[k]["nombre"]}')
    only_good = c7.toggle("Solo niveles recomendados", value=False, key="as_good",
                          help="Solo élite, cuota alta, sólidas y valor moderado (además de las de cuota objetivo).")
    go_ = st.button("Construir mis 3 propuestas", type="primary", use_container_width=True)
    lo, hi = odds_range
    if not go_:
        st.markdown(f'<div class="note">Cuota total pedida: <b>{num(lo)}-{num(hi)}</b>, y se respeta siempre. Primero '
                    'busca cuotas operables (bet365 o la mejor cuota real de otra casa, que se indica) con valor ≥ 0; si no '
                    'hay, usa selecciones con cuota objetivo (la mínima a la que cogerlas en bet365). Nunca propone una '
                    'apuesta con valor negativo. Recibirás tres propuestas: <b>Conservadora</b> (más probabilidad), '
                    '<b>Equilibrada</b> (mejor valor) y <b>Alto retorno</b> (más cuota).</div>', unsafe_allow_html=True)
        return
    deadline = datetime.combine(dl_date, dl_time, TZ).astimezone(timezone.utc)
    plan, alts, n = assistant.copilot(D["pool"], bankroll, deadline, now, risk, tipo, sports, (lo, hi), only_good,
                                      s.kelly_fraction, s.max_stake_pct)
    if plan is None:
        empty(f"No hay selecciones analizadas que terminen antes del {dl_date:%d/%m} a las {dl_time:%H:%M} en los "
              "deportes elegidos. Amplía la fecha o los deportes.")
        return
    if plan.role == "cercana":
        st.markdown(f'<div class="warnbox">{e(plan.rationale[1] if len(plan.rationale) > 1 else "")}</div>',
                    unsafe_allow_html=True)
    by = {p.role: p for p in [plan, *alts]}
    ordered = [by[r] for r in ("conservadora", "equilibrada", "cercana", "retorno") if r in by]
    st.markdown(f'<p class="hint">Elegidas entre {n} selecciones analizadas.</p>', unsafe_allow_html=True)
    cols = st.columns(len(ordered))
    for col, p in zip(cols, ordered):
        col.markdown(plan_card(p, n), unsafe_allow_html=True)
    st.caption("Las probabilidades son estimaciones. Comprueba la cuota en bet365 antes de apostar: si paga menos que la "
               "cuota indicada, la apuesta deja de tener sentido. Apuesta solo lo que puedas permitirte perder.")


def backtest_view():
    bt = D.get("backtest") or {}
    strat = bt.get("estrategia_bet365") or {}
    if not strat:
        empty("El backtest histórico aparecerá tras el próximo análisis completo (se recalcula cada 30 días).")
        return
    t = pd.DataFrame(strat).T
    t["umbral"] = [f"Valor ≥ {k.split('_')[1]} %" for k in t.index]
    ll = bt.get("logloss_1x2") or {}
    best = t.sort_values("apuestas", ascending=False).iloc[0]
    kpi_strip([
        ("Partidos del backtest", num(bt.get("n_predicciones"), 0), f'{txt(bt.get("desde"))} a {txt(bt.get("hasta"))}', "cyan", ""),
        ("Apuestas simuladas", num(best["apuestas"], 0), f'con {best["umbral"].lower()}', "violet", ""),
        ("Acierto", pct(best["acierto"], dec=1), f'cuota media {num(best["cuota_media"])}', "blue", ""),
        ("Rentabilidad", pct(best["roi"], True), "por euro apostado a la cuota de bet365", "green" if best["roi"] >= 0 else "red",
         sign_cls(best["roi"])),
        ("Log-loss 1X2", num(ll.get("mercado"), 3), f'mercado · modelo {num(ll.get("modelo"), 3)}', "amber", "")])
    c1, c2 = st.columns([1.2, 1])
    with c1:
        lo = [x[0] if isinstance(x, (list, tuple)) else None for x in t["roi_ic95"]]
        hi_ = [x[1] if isinstance(x, (list, tuple)) else None for x in t["roi_ic95"]]
        fig = go.Figure(go.Bar(x=t["umbral"], y=t["roi"], marker_color=[C["green"] if v >= 0 else C["red"] for v in t["roi"]],
                               text=[pct(v, True) for v in t["roi"]], textposition="outside",
                               error_y=dict(type="data", symmetric=False,
                                            array=[(h - r) if isnum(h) else 0 for h, r in zip(hi_, t["roi"])],
                                            arrayminus=[(r - l) if isnum(l) else 0 for l, r in zip(lo, t["roi"])],
                                            color=C["mute"]),
                               hovertemplate="%{x}: %{y:.1%}<extra></extra>"))
        fig.update_yaxes(tickformat=".0%")
        fig.update_layout(title=dict(text="Rentabilidad por umbral de valor (con intervalo del 95 %)", x=0, font=dict(size=13)))
        st.plotly_chart(chart_layout(fig, 340, legend=False), width="stretch", config=PLOT_CFG)
    with c2:
        show_df(t[["umbral", "apuestas", "acierto", "cuota_media", "roi"]].reset_index(drop=True).rename(columns={
            "umbral": "Estrategia", "apuestas": "Apuestas", "acierto": "Acierto", "cuota_media": "Cuota media",
            "roi": "Rentabilidad"}), {"Acierto": lambda v: pct(v), "Rentabilidad": lambda v: pct(v, True)})
    st.markdown('<div class="note">Cómo leerlo: es el resultado de haber apostado el último año en las ligas de football-data '
                'cada vez que el modelo antiguo veía valor frente a la cuota de bet365. Sirvió para descubrir que ese modelo '
                'no batía al mercado (su log-loss es peor que el del mercado sin margen): por eso el radar actual usa el precio '
                'justo de Pinnacle y las exchanges como referencia, congela el modelo propio y exige referencia sharp para '
                'Élite. El seguimiento en vivo mide el motor actual desde METRICS_SINCE.</div>', unsafe_allow_html=True)


def page_perf():
    hero("RENDIMIENTO", "Resultados y backtest", "Lo que el radar recomendó, congelado en el momento de mostrarlo.")
    R = D["resolved"]
    c1, c2, c3 = st.columns([1.5, 1.3, 1.2])
    opts = (["since", "all", "custom"] if SINCE_DEFAULT else ["all", "custom"])
    scope = c1.segmented_control(
        "Periodo", opts, default=opts[0], key="pf_scope", label_visibility="collapsed",
        format_func=lambda o: {"since": f"Desde {SINCE_DEFAULT}", "all": "Todo el historial", "custom": "Elegir fecha"}[o]) \
        or opts[0]
    since = SINCE_DEFAULT if scope == "since" else ""
    if scope == "custom":
        since = st.date_input("Desde", value=(datetime.now(TZ) - timedelta(days=30)).date(), format="DD/MM/YYYY",
                              key="pf_date").isoformat()
    tiers = c2.selectbox("Nivel", ["todas", "valor", "cuota_alta", "solida", "moderado", "combinadas"], key="pf_tier",
                         label_visibility="collapsed",
                         format_func=lambda t: {"todas": "Todos los niveles", "combinadas": "Combinadas"}.get(
                             t, TIER.get(t, (t,))[0])) or "todas"
    mode = c3.segmented_control("Medida", ["unit", "stake"], default="unit", key="pf_mode", label_visibility="collapsed",
                                format_func=lambda m: {"unit": "1 unidad por jugada", "stake": "Stakes reales (€)"}[m]) or "unit"
    R = since_filter(R, since)
    RC_all = since_filter(D["resolved_combos"], since)
    if tiers == "combinadas":
        Rt = webdata.combos_as_picks(RC_all)
    elif tiers != "todas" and not R.empty:
        Rt = R[R["tier"] == tiers]
    else:
        Rt = R
    k = webdata.kpis(Rt, mode, s.bankroll)
    view = st.segmented_control("Vista", ["vivo", "backtest"], default="vivo" if k["n"] else "backtest", key="pf_view",
                                label_visibility="collapsed",
                                format_func=lambda v: {"vivo": f"📈 Seguimiento en vivo · {k['n']} resueltas",
                                                       "backtest": "🧪 Backtest histórico validado"}[v]) or "vivo"
    if view == "backtest":
        if not k["n"]:
            st.markdown('<div class="note">Aún no hay apuestas resueltas en este periodo: mientras llegan, aquí tienes el '
                        'backtest histórico. Cambia a «Seguimiento en vivo» cuando se jueguen los primeros partidos.</div>',
                        unsafe_allow_html=True)
        backtest_view()
        return
    if not k["n"]:
        kpi_strip([("Jugadas resueltas", "0", "en este periodo", "cyan", "")])
        empty("Todavía no hay jugadas resueltas en este periodo. En cuanto se jueguen los primeros partidos verás aquí la "
              "curva de banca, el yield y el desglose. Mientras, mira el backtest histórico.")
        return
    lv = webdata.level_summary(R, RC_all, s.bankroll)
    unit = " €" if mode == "stake" and not Rt.empty and Rt["stake"].fillna(0).sum() > 0 else " u"
    kpi_strip([
        ("Jugadas resueltas", str(k["n"]), f'cuota media {num(k["avg_odds"])}', "cyan", ""),
        ("Acierto", pct(k["hit"], dec=1), f'esperado {pct(k["expected"], dec=1)}', "violet", ""),
        ("Yield", pct(k["yield"], True), f'{num(k["staked"])}{unit} apostados', "green" if k["yield"] >= 0 else "red",
         sign_cls(k["yield"])),
        ("Beneficio", num(k["profit"]) + unit, "con " + ("stakes" if unit == " €" else "1 u por jugada"),
         "green" if k["profit"] >= 0 else "red", sign_cls(k["profit"])),
        ("CLV", pct(k["clv"], True) if k["clv_n"] else "—", f'{k["clv_n"]} con cierre de Pinnacle', "amber",
         sign_cls(k["clv"])),
        ("Máx. caída", num(k["max_dd"]) + unit, "desde el máximo", "red", "neg" if k["max_dd"] < 0 else "")])
    ser = webdata.perf_series(Rt, s.bankroll, "stake" if unit == " €" else "unit")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.74, 0.26], vertical_spacing=0.04,
                        specs=[[{"secondary_y": True}], [{}]])
    fig.add_trace(go.Scatter(x=ser["date"], y=ser["bank"], mode="lines", line=dict(color=C["amber"], width=2.5, shape="hv"),
                             name="Banca" if unit == " €" else "Unidades", fill="tozeroy" if unit == " u" else None,
                             fillcolor="rgba(255,176,32,.07)", hovertemplate="%{y:.2f}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=ser["date"], y=ser["yield"], mode="lines", line=dict(color=C["cyan"], width=1.6, dash="dot"),
                             name="Yield acumulado", hovertemplate="%{y:.1%}<extra></extra>"), row=1, col=1, secondary_y=True)
    fig.add_trace(go.Scatter(x=ser["date"], y=ser["drawdown"], mode="lines", line=dict(color=C["red"], width=1),
                             fill="tozeroy", fillcolor="rgba(255,92,102,.15)", name="Caída",
                             hovertemplate="%{y:.2f}<extra></extra>"), row=2, col=1)
    fig.add_hline(y=s.bankroll if unit == " €" else 0, line_color=C["mute"], line_width=1, line_dash="dash", row=1, col=1)
    fig.update_yaxes(tickformat=".0%", secondary_y=True, row=1, col=1, showgrid=False)
    section("Evolución de la banca y del yield")
    st.plotly_chart(chart_layout(fig, 430), width="stretch", config=PLOT_CFG)
    if not lv.empty and lv["n"].fillna(0).sum() > 0:
        section("Por nivel", hint="Yield = beneficio / apostado (1 u por jugada). ROI = beneficio / banca inicial con stakes reales.")
        show_df(lv[lv["n"].fillna(0) > 0].rename(columns={"nivel": "Nivel", "n": "Jugadas", "hit": "Acierto",
                                                         "expected": "Esperado", "yield": "Yield", "profit": "Unidades",
                                                         "roi": "ROI"}),
                {"Acierto": lambda v: pct(v), "Esperado": lambda v: pct(v), "Yield": lambda v: pct(v, True),
                 "Unidades": lambda v: f"{v:+.2f}".replace(".", ","), "ROI": lambda v: pct(v, True)})
    section("Desglose")
    tabs = st.tabs(["Deporte", "Mercado", "Rango de cuota", "Competición", "Nivel", "Casa"])
    names = {"sport": lambda x: SPORTS.get(x, {}).get("nombre", txt(x)), "market": lambda x: MK.get(x, txt(x)),
             "band": lambda x: txt(x), "league": lambda x: f"{league_flag(x)} {league_name(x)}" if txt(x) != "—" else "—",
             "tier": lambda x: COMBO_NAME.get(x, TIER.get(x, (txt(x),))[0]), "bookmaker": lambda x: book_name(x) or "—"}
    for tab, by in zip(tabs, ["sport", "market", "band", "league", "tier", "bookmaker"]):
        with tab:
            if by not in Rt and by != "band":
                empty("Este desglose no aplica a las combinadas.")
                continue
            b = webdata.breakdown(Rt.assign(**{by: Rt[by].fillna("—")}) if by in Rt else Rt, by)
            b["grupo"] = b["grupo"].map(names[by])
            cA, cB = st.columns([1.1, 1])
            bf = go.Figure(go.Bar(x=b["yield"], y=b["grupo"], orientation="h",
                                  marker_color=[C["green"] if v >= 0 else C["red"] for v in b["yield"]],
                                  text=[pct(v, True) for v in b["yield"]], textposition="outside",
                                  hovertemplate="%{y}: %{x:.1%}<extra></extra>"))
            bf.update_xaxes(tickformat=".0%")
            bf.update_yaxes(autorange="reversed")
            with cA:
                st.plotly_chart(chart_layout(bf, max(200, 46 * len(b) + 60), legend=False), width="stretch", config=PLOT_CFG)
            with cB:
                show_df(b.rename(columns={"grupo": "Grupo", "n": "Jugadas", "acierto": "Acierto", "esperado": "Esperado",
                                          "yield": "Yield", "unidades": "Unidades"}),
                        {"Acierto": lambda v: pct(v), "Esperado": lambda v: pct(v), "Yield": lambda v: pct(v, True),
                         "Unidades": lambda v: f"{v:+.2f}".replace(".", ",")})
    cal = D["calib"]
    if since and not cal.empty:
        cal = cal[cal["kickoff"] >= since]
    section("Calibración", len(cal), "¿Ocurre lo que el radar dice que ocurrirá? Cuanto más cerca de la diagonal, mejor.")
    if len(cal) < 50:
        empty("Hacen falta al menos 50 resultados para dibujar la calibración.")
    else:
        g = webdata.calibration(cal)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=C["mute"], dash="dot"), name="Perfecto"))
        fig.add_trace(go.Scatter(x=g["p"], y=g["real"], mode="markers+lines", line=dict(color=C["amber"]), name="Radar",
                                 marker=dict(size=np.sqrt(g["n"]) * 1.4 + 6, color=C["amber"]),
                                 hovertemplate="Previsto %{x:.0%}<br>Real %{y:.0%}<extra></extra>"))
        fig.update_xaxes(tickformat=".0%", title="Probabilidad prevista")
        fig.update_yaxes(tickformat=".0%", title="Frecuencia real")
        st.plotly_chart(chart_layout(fig, 340), width="stretch", config=PLOT_CFG)


def page_status():
    hero("ESTADO", "Estado y método", "De dónde salen los datos y cómo decide el radar.")
    F = D["fuentes"]
    kpi_strip([("Créditos The Odds API", num(D["credits"], 0) if D["credits"] is not None else "—", "restantes este mes",
                "cyan", ""),
               ("OddsPapi", f'{D["oddspapi_used"]}/{s.oddspapi_monthly}', "consultas este mes", "violet", ""),
               ("Rango de cuotas", f"{num(s.min_odds)}-{num(s.max_odds)}", "fuera de él, solo información", "amber", ""),
               ("Umbral élite", pct(s.min_ev, True), f"Telegram: {pct(s.alert_ev, True)} con ref. sharp", "green", "")])
    section("Fuentes de datos", hint=f"Última comprobación {when(F['ts'])}" if F.get("ts") else "")
    rows = [f'<tr><td class="l b"><span class="lt {"verde" if v.get("ok") else "rojo"}"></span> {e(k)}</td>'
            f'<td>{num(v.get("filas", 0), 0)}</td><td class="l mute">{e(str(v.get("detalle", "")))}</td></tr>'
            for k, v in F.items() if k != "ts"]
    if rows:
        st.markdown('<div class="scroll"><table class="tb"><tr><th class="l">Fuente</th><th>Registros</th>'
                    f'<th class="l">Detalle</th></tr>{"".join(rows)}</table></div>', unsafe_allow_html=True)
    else:
        empty("El estado de las fuentes aparecerá tras el próximo análisis.")
    section("Cómo decide")
    st.markdown(
        '<p class="hint">1. <b>Precio justo limpio</b>: mercado sin margen con casas válidas (mercado completo y margen '
        'creíble). Si hay Pinnacle o exchanges, solo ellos. bet365 nunca entra en su propio consenso.<br>'
        '2. <b>Cuota operable</b>: la de bet365 si es reciente; si no, la mejor cuota real de una casa generalista '
        '(William Hill, Unibet, Betsson, 888sport, DraftKings…), con su nombre y la estimación de lo que pagaría bet365. '
        'Si la mejor supera en más de un 4 % a la segunda, se usa la segunda (cuotas sueltas fuera).<br>'
        f'3. <b>Élite</b>: la cuota operable supera el precio justo en al menos un {pct(s.min_ev, True)}, cuota '
        f'{num(s.min_odds)}-{num(s.max_odds)} y referencia sharp. A Telegram solo van las de bet365 confirmada con '
        f'{pct(s.alert_ev, True)} o más.<br>4. <b>Cuota alta</b>: {num(s.high_odds_min)}-{num(s.high_odds_max)} con '
        f'{pct(s.high_min_ev, True)} o más. <b>Sólida</b>: probabilidad ≥ {pct(s.solid_min_prob)} a precio justo '
        f'(cuota desde {num(s.solid_min_odds)}). <b>Valor moderado</b>: valor entre 0 y {pct(s.min_ev, True)}.<br>'
        f'5. <b>Revisar</b>: valor de más del {pct(s.max_ev)}: casi siempre es un error de datos.<br>'
        '6. <b>Seguimiento</b>: lo recomendado queda congelado con su cuota, su casa y su probabilidad. El CLV solo se mide '
        'contra el cierre de Pinnacle.<br>7. <b>Datos</b>: The Odds API (fútbol, NBA, NFL y tenis con unas 15 casas, '
        'Pinnacle incluida), ESPN (calendario, resultados, DraftKings y lesiones de NBA/NFL), football-data (ligas y '
        'bet365), OddsPapi (bet365 en bloque) y API-Football (bajas de fútbol).</p>', unsafe_allow_html=True)
    w = D["weights"] or {}
    if w:
        rows = [{"Deporte": SPORTS.get(k.split("|")[0], {}).get("nombre", k), "Mercado": MK.get(k.split("|")[1], k),
                 "Peso del modelo": v} for k, v in sorted(w.items())]
        show_df(pd.DataFrame(rows), {"Peso del modelo": lambda v: pct(v)})
    for key, title in (("backtest", "Backtest de fútbol (último año)"), ("backtest_tennis", "Backtest de tenis")):
        bt = D.get(key)
        if not bt or not bt.get("estrategia_bet365"):
            continue
        section(title, f'{bt["desde"]} a {bt["hasta"]}')
        t = pd.DataFrame(bt["estrategia_bet365"]).T
        t.index = [f"Valor ≥ {k.split('_')[1]} %" for k in t.index]
        t["Intervalo"] = t["roi_ic95"].map(lambda x: f"{x[0]:+.0%} a {x[1]:+.0%}" if isnum(x[0]) else "—")
        t = t[["apuestas", "acierto", "cuota_media", "roi", "Intervalo"]].reset_index().rename(columns={"index": "Estrategia"})
        t.columns = ["Estrategia", "Apuestas", "Acierto", "Cuota media", "Rentabilidad", "Intervalo 95 %"]
        show_df(t, {"Acierto": lambda v: pct(v), "Rentabilidad": lambda v: pct(v, True)})
    section("Últimos análisis")
    R = D["runs"]
    if not R.empty:
        def resume(txt: str) -> dict:
            try:
                d = ast.literal_eval(txt)
            except (ValueError, SyntaxError):
                return {}
            dep = d.get("por_deporte") or {}
            return {"Partidos": d.get("partidos", 0),
                    "Por deporte": " ".join(f'{SPORTS[k]["icono"]} {v}' for k, v in dep.items() if k in SPORTS) or "—",
                    "Élite": d.get("valor", 0), "Cuota alta": d.get("cuota_alta", 0), "Sólidas": d.get("solidas", 0),
                    "Moderadas": d.get("moderadas", 0), "Por confirmar": d.get("verificar", 0),
                    "Liquidados": d.get("liquidados", 0)}
        show_df(pd.DataFrame([{"Hora": when(r.ts), **resume(r.detail)} for r in R.itertuples()]))
    st.markdown('<p class="hint" style="margin-top:1.2rem">Juega solo con dinero que puedas perder. Si deja de ser '
                'divertido: <a href="https://www.jugarbien.es">jugarbien.es</a> · autoexclusión en el RGIAJ.</p>',
                unsafe_allow_html=True)


# ======================================================================= navegación
pages = {
    "Radar": [st.Page(page_home, title="Centro de mando", icon="📡", default=True),
              st.Page(page_picks, title="Jugadas", icon="🎯", url_path="jugadas"),
              st.Page(page_combos, title="Combinadas", icon="🧩", url_path="combinadas"),
              st.Page(page_matches, title="Partidos", icon="🗓️", url_path="partidos")],
    "Herramientas": [st.Page(page_assistant, title="Asistente de apuesta", icon="🧮", url_path="asistente")],
    "Seguimiento": [st.Page(page_perf, title="Rendimiento", icon="📈", url_path="rendimiento"),
                    st.Page(page_status, title="Estado y método", icon="🛰️", url_path="estado")],
}
st.navigation(pages, position="top").run()
