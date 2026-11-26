"""Radar · terminal de análisis. Local: `streamlit run app.py` · Internet: Streamlit Community Cloud.

Arquitectura: radar/webdata.py lee la base de datos (funciones puras, probadas) y aquí solo se pinta. Los datos se
cachean con @st.cache_data por fecha de modificación del fichero: la web carga al instante y se refresca sola
cuando GitHub Actions sube un análisis nuevo."""
from __future__ import annotations

import ast
import html
import json
from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from radar import assistant, webdata
from radar.config import FOOTBALL, OTHER, SPORTS, Settings, league_flag, league_name
from radar.model.markets import label as mlabel

TZ = ZoneInfo("Europe/Madrid")
DIAS = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
DIAS_C = ["lun", "mar", "mié", "jue", "vie", "sáb", "dom"]
MK = {"1x2": "Resultado", "ml": "Ganador", "ou": "Más/menos", "spread": "Hándicap", "btts": "Ambos marcan"}
TIER = {  # nivel → (texto del badge, clase css)
    "valor": ("Élite", "b-elite"), "solida": ("Sólida", "b-solid"), "verificar": ("Por confirmar", "b-check"),
    "revisar": ("Revisar", "b-review"), "pronostico": ("Sin ventaja", "b-none"), "modelo": ("Sin cuotas", "b-none"),
    "sin_datos": ("Sin datos", "b-none")}
LIGHT = {"verde": "Recomendable", "ambar": "Con reservas", "rojo": "No apostar", "gris": "Sin datos fiables"}
COMBO_NAME = {"conservadora": "Conservadora", "equilibrada": "Equilibrada", "arriesgada": "Arriesgada"}
RISK_NAME = {"bajo": "Riesgo bajo", "medio": "Riesgo medio", "alto": "Riesgo alto"}
C = dict(bg="#06080C", panel="#0C1118", line="#1C2633", text="#E6EDF3", mute="#7D8A9A", amber="#FFB020",
         green="#2BD47D", red="#FF5C5C", cyan="#3BC9F5", violet="#A78BFA")

st.set_page_config(page_title="Radar · terminal EV", page_icon="📡", layout="wide", initial_sidebar_state="collapsed")
s = Settings()
e = html.escape

# ======================================================================= estilo
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');
:root{--bg:#06080C;--panel:#0C1118;--panel2:#111823;--panel3:#162030;--line:#1C2633;--line2:#263243;--text:#E6EDF3;
--mute:#7D8A9A;--dim:#56636F;--amber:#FFB020;--green:#2BD47D;--red:#FF5C5C;--cyan:#3BC9F5;--violet:#A78BFA;
--mono:'JetBrains Mono',ui-monospace,SFMono-Regular,Menlo,monospace;--sans:'Inter',system-ui,-apple-system,sans-serif}
html,body,[data-testid="stAppViewContainer"],[data-testid="stHeader"],[data-testid="stMain"]{background:var(--bg)!important;color:var(--text)}
[data-testid="stHeader"]{height:2.4rem;border-bottom:1px solid var(--line)}
[data-testid="stSidebar"]{background:#080B10!important;border-right:1px solid var(--line)}
[data-testid="stSidebarNav"] span{font-family:var(--sans);font-size:.92rem}
.block-container{padding:3.2rem 1.4rem 4rem;max-width:1440px}
*{font-family:var(--sans)}
code,.mono,.num{font-family:var(--mono)!important;font-variant-numeric:tabular-nums}
a{color:var(--amber)}
.pos{color:var(--green)!important}.neg{color:var(--red)!important}.am{color:var(--amber)!important}.mute{color:var(--mute)}
/* barra superior */
.top{display:flex;align-items:center;justify-content:space-between;gap:1rem;flex-wrap:wrap;border:1px solid var(--line);
background:linear-gradient(180deg,#0D131C,#0A0F15);border-radius:10px;padding:.7rem 1rem;margin-bottom:1rem}
.brand{display:flex;align-items:baseline;gap:.7rem}
.brand b{font-family:var(--mono);font-weight:700;letter-spacing:.08em;color:var(--amber);font-size:1.05rem}
.brand h1{font-size:1.35rem;font-weight:600;margin:0;padding:0;color:var(--text)}
.brand span{color:var(--mute);font-size:.86rem}
.tick{display:flex;gap:1.1rem;flex-wrap:wrap;font-family:var(--mono);font-size:.78rem;color:var(--mute)}
.tick b{color:var(--text);font-weight:600}
.live{display:inline-block;width:7px;height:7px;border-radius:50%;background:var(--green);margin-right:.35rem;
box-shadow:0 0 0 0 rgba(43,212,125,.6);animation:pulse 2s infinite}
.live.off{background:var(--red);animation:none}
@keyframes pulse{0%{box-shadow:0 0 0 0 rgba(43,212,125,.55)}70%{box-shadow:0 0 0 7px rgba(43,212,125,0)}100%{box-shadow:0 0 0 0 rgba(43,212,125,0)}}
/* KPIs */
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:.2rem 0 1.1rem}
.kpi{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:.75rem .9rem;position:relative;overflow:hidden}
.kpi::after{content:"";position:absolute;left:0;top:0;height:2px;width:100%;background:var(--kc,var(--line2))}
.kpi .l{font-family:var(--mono);font-size:.68rem;letter-spacing:.08em;text-transform:uppercase;color:var(--mute)}
.kpi .v{font-family:var(--mono);font-weight:600;font-size:1.65rem;line-height:1.15;margin-top:.2rem}
.kpi .d{font-size:.78rem;color:var(--mute);margin-top:.1rem}
.k-amber{--kc:var(--amber)}.k-green{--kc:var(--green)}.k-red{--kc:var(--red)}.k-cyan{--kc:var(--cyan)}.k-violet{--kc:var(--violet)}
/* secciones */
.sec{display:flex;align-items:baseline;gap:.6rem;margin:1.4rem 0 .55rem;border-bottom:1px solid var(--line);padding-bottom:.35rem}
.sec h3{margin:0;padding:0;font-size:1.02rem;font-weight:600;letter-spacing:.01em}
.sec .n{font-family:var(--mono);color:var(--amber);font-size:.85rem}
.hint{color:var(--mute);font-size:.86rem;margin:.1rem 0 .8rem;max-width:90ch}
.empty{background:var(--panel);border:1px dashed var(--line2);border-radius:10px;padding:.9rem 1.1rem;color:var(--mute);font-size:.9rem}
/* badges */
.bdg{display:inline-flex;align-items:center;gap:.25rem;font-family:var(--mono);font-size:.66rem;font-weight:600;letter-spacing:.06em;
text-transform:uppercase;padding:.16rem .5rem;border-radius:4px;border:1px solid transparent;white-space:nowrap;vertical-align:middle}
.b-elite{background:var(--amber);color:#1A1200}.b-solid{background:rgba(43,212,125,.14);color:var(--green);border-color:rgba(43,212,125,.45)}
.b-check{color:var(--cyan);border-color:rgba(59,201,245,.55)}.b-review{color:var(--red);border-color:rgba(255,92,92,.6)}
.b-none{color:var(--mute);border-color:var(--line2)}.b-tg{background:rgba(167,139,250,.15);color:var(--violet)}
.b-risk{color:var(--text);border-color:var(--line2)}
/* semáforo */
.lt{display:inline-block;width:9px;height:9px;border-radius:50%;flex:none}
.lt.verde{background:var(--green);box-shadow:0 0 8px rgba(43,212,125,.55)}.lt.ambar{background:var(--amber)}
.lt.rojo{background:var(--red)}.lt.gris{background:var(--dim)}
/* barra de probabilidad */
.pbar{position:relative;height:6px;background:var(--panel3);border-radius:3px;margin-top:.35rem;overflow:visible}
.pbar i{position:absolute;left:0;top:0;bottom:0;border-radius:3px;background:linear-gradient(90deg,#1B8F57,var(--green))}
.pbar u{position:absolute;top:-3px;bottom:-3px;width:2px;background:var(--amber);text-decoration:none}
.pbar.big{height:8px}
.tri{display:flex;height:5px;border-radius:3px;overflow:hidden;margin-top:.35rem;background:var(--panel3)}
.tri span:nth-child(1){background:#4C8DFF}.tri span:nth-child(2){background:#5E6B7A}.tri span:nth-child(3){background:#C77DFF}
/* radar */
.rr{display:grid;grid-template-columns:62px minmax(170px,1.5fr) minmax(150px,1.2fr) 112px 64px 70px 74px minmax(190px,1.5fr);
gap:.75rem;align-items:center;padding:.6rem .85rem;border-bottom:1px solid var(--line);background:var(--panel)}
.rr:hover{background:#0F1620}
.rr.h{background:transparent;border-bottom:1px solid var(--line2);font-family:var(--mono);font-size:.66rem;letter-spacing:.08em;
text-transform:uppercase;color:var(--dim);padding:.35rem .85rem}
.rw{border:1px solid var(--line);border-radius:10px;overflow:hidden;margin-bottom:.9rem}
.rr-t b{font-family:var(--mono);font-size:.95rem;display:block}.rr-t span{color:var(--mute);font-size:.72rem}
.rr-m .lg{color:var(--mute);font-size:.74rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.rr-m .tm{font-weight:600;font-size:.92rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.rr-m .tm i{color:var(--dim);font-style:normal;font-weight:400;margin:0 .25rem}
.rr-s .lb{font-weight:600;font-size:.9rem}.rr-s .sub{color:var(--mute);font-size:.72rem}
.rr-n .k{display:none}.rr-n b{font-family:var(--mono);font-size:.95rem;font-weight:600}
.rr-r{display:flex;gap:.55rem;align-items:flex-start}.rr-r p{margin:.2rem 0 0;font-size:.78rem;color:var(--mute);line-height:1.3}
.day{font-family:var(--mono);font-size:.78rem;letter-spacing:.1em;text-transform:uppercase;color:var(--amber);margin:1rem 0 .45rem}
/* tarjetas de jugada */
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(310px,1fr));gap:12px}
.grid2{display:grid;grid-template-columns:repeat(auto-fill,minmax(400px,1fr));gap:12px}
.pk{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:.85rem .95rem;position:relative}
.pk.t-valor{border-color:rgba(255,176,32,.55);background:linear-gradient(180deg,rgba(255,176,32,.06),var(--panel) 40%)}
.pk.t-solida{border-color:rgba(43,212,125,.35)}
.pk.off{opacity:.62}
.pk-h{display:flex;gap:.35rem;align-items:center;flex-wrap:wrap}.pk-h .w{margin-left:auto;color:var(--mute);font-size:.74rem;font-family:var(--mono)}
.pk-lg{color:var(--mute);font-size:.76rem;margin-top:.45rem}
.pk-m{font-weight:600;font-size:.98rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.pk-s{font-size:1.18rem;font-weight:700;margin:.15rem 0 .55rem}
.cells{display:grid;grid-template-columns:repeat(4,1fr);gap:6px}
.cell{background:var(--panel2);border:1px solid var(--line);border-radius:7px;padding:.35rem .5rem;min-width:0}
.cell .k{font-family:var(--mono);font-size:.6rem;letter-spacing:.08em;text-transform:uppercase;color:var(--mute);
white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.cell .o{font-family:var(--mono);font-weight:600;font-size:1.08rem;line-height:1.2;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.cell.hl{border-color:rgba(255,176,32,.45)}.cell.hl .o{color:var(--amber)}
.cell.val{border-color:rgba(43,212,125,.6);background:rgba(43,212,125,.08)}.cell.val .o{color:var(--green)}
.pk-f{display:flex;justify-content:space-between;gap:.5rem;color:var(--mute);font-size:.74rem;margin-top:.3rem;font-family:var(--mono)}
details{margin-top:.5rem;font-size:.8rem;color:var(--mute)}
details summary{cursor:pointer;color:var(--text);opacity:.8;list-style:none;font-size:.78rem}
details summary::-webkit-details-marker{display:none}
details summary::before{content:"+ ";color:var(--amber);font-family:var(--mono)}
details[open] summary::before{content:"− "}
details ul{margin:.35rem 0 0;padding-left:1rem}
/* combinadas */
.legs{margin:.5rem 0 .2rem;padding:0;list-style:none}
.legs li{display:flex;justify-content:space-between;gap:.6rem;padding:.42rem 0;border-top:1px solid var(--line);font-size:.86rem}
.legs li:first-child{border-top:0}.legs .lp{color:var(--mute);font-size:.74rem}
.legs .lo{font-family:var(--mono);font-weight:600;font-size:1rem}
/* fichas de partido */
.mc{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:.8rem .9rem}
.mc-h{display:flex;justify-content:space-between;gap:.6rem;color:var(--mute);font-size:.76rem;flex-wrap:wrap}
.mc-h .t{color:var(--text);font-family:var(--mono);font-weight:600}
.mc-teams{display:grid;grid-template-columns:1fr auto;gap:.1rem .8rem;margin:.45rem 0 .45rem;align-items:center}
.mc-teams .nm{font-weight:600;font-size:.98rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.mc-teams .xs{color:var(--mute);font-size:.74rem;text-align:right;font-family:var(--mono)}
.oc{display:grid;gap:6px;margin-top:.5rem}.oc3{grid-template-columns:repeat(3,1fr)}.oc2{grid-template-columns:repeat(2,1fr)}
.mv{font-size:.66rem;margin-left:.2rem}.mv.up{color:var(--green)}.mv.dn{color:var(--red)}
.tagrow{display:flex;gap:.35rem;flex-wrap:wrap;margin-top:.5rem}
.tag{font-size:.7rem;color:var(--mute);background:var(--panel2);border:1px solid var(--line);border-radius:4px;padding:.1rem .45rem}
table.mk{width:100%;border-collapse:collapse;margin-top:.45rem;font-size:.78rem;color:var(--text)}
table.mk th{color:var(--mute);font-weight:500;text-align:right;padding:.25rem .3rem;border-bottom:1px solid var(--line);font-family:var(--mono);font-size:.64rem;text-transform:uppercase}
table.mk th:first-child,table.mk td:first-child{text-align:left}
table.mk td{padding:.25rem .3rem;text-align:right;border-bottom:1px solid rgba(28,38,51,.7);font-family:var(--mono)}
table.mk td:first-child{font-family:var(--sans)}table.mk tr.v td{color:var(--green)}
/* clasificación */
table.tb{width:100%;border-collapse:collapse;font-size:.85rem;background:var(--panel);border:1px solid var(--line);border-radius:10px;overflow:hidden}
table.tb th{color:var(--mute);font-weight:500;padding:.45rem .45rem;text-align:right;border-bottom:1px solid var(--line2);
background:var(--panel2);font-family:var(--mono);font-size:.66rem;text-transform:uppercase;letter-spacing:.05em}
table.tb td{padding:.4rem .45rem;text-align:right;border-bottom:1px solid var(--line);font-family:var(--mono)}
table.tb th.l,table.tb td.l{text-align:left}table.tb td.l{font-family:var(--sans)}table.tb td.b{font-weight:600}
.fm{display:inline-flex;gap:2px}.fm i{display:inline-block;width:15px;height:15px;border-radius:3px;font-style:normal;
font-size:.58rem;line-height:15px;text-align:center;color:#06080C;font-weight:700;font-family:var(--mono)}
.fm .V{background:var(--green)}.fm .E{background:#7D8A9A}.fm .D{background:var(--red)}
.bar{display:inline-block;height:6px;border-radius:3px;background:#4C8DFF;vertical-align:middle}
.scroll{overflow-x:auto}
/* asistente */
.plan{background:var(--panel);border:1px solid rgba(255,176,32,.5);border-radius:12px;padding:1rem 1.1rem}
.plan h4{font-size:1.15rem;margin:0 0 .2rem;font-weight:600}
.plan .big{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:8px;margin:.8rem 0}
.plan .big div{background:var(--panel2);border:1px solid var(--line);border-radius:8px;padding:.55rem .7rem}
.plan .big b{display:block;font-family:var(--mono);font-size:1.35rem;font-weight:600;line-height:1.15}
.plan .big span{color:var(--mute);font-size:.74rem}
.alt{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:.8rem 1rem;margin-top:.7rem}
/* streamlit */
[data-testid="stTabs"] button p{font-family:var(--mono);font-size:.8rem;font-weight:600;letter-spacing:.06em;text-transform:uppercase}
[data-testid="stTabs"] [aria-selected="true"] p{color:var(--amber)}
[data-baseweb="tab-highlight"]{background:var(--amber)!important}
[data-baseweb="tab-border"]{background:var(--line)!important}
[data-testid="stButtonGroup"]>div{flex-wrap:wrap;row-gap:6px}
[data-testid="stDataFrame"]{border:1px solid var(--line);border-radius:8px}
div[data-testid="stTextInput"] input{background:var(--panel)!important;border-color:var(--line2)!important}
button:focus-visible,summary:focus-visible{outline:2px solid var(--amber);outline-offset:2px}
/* móvil */
@media (max-width:900px){
.block-container{padding:3rem .7rem 3rem}
.rr.h{display:none}
.rr{grid-template-columns:1fr 1fr 1fr 1fr;grid-template-areas:"t m m m" "s s s s" "p j b v" "r r r r";gap:.45rem .6rem;padding:.7rem}
.rr-t{grid-area:t}.rr-m{grid-area:m}.rr-s{grid-area:s}.rr-n.np{grid-area:p}.rr-n.nj{grid-area:j}.rr-n.nb{grid-area:b}.rr-n.nv{grid-area:v}.rr-r{grid-area:r}
.rr-n .k{display:block;font-family:var(--mono);font-size:.58rem;letter-spacing:.08em;text-transform:uppercase;color:var(--dim)}
.grid2{grid-template-columns:1fr}.cells{grid-template-columns:repeat(2,1fr)}
.brand h1{font-size:1.1rem}.brand span{display:none}.kpi .v{font-size:1.3rem}}
</style>""", unsafe_allow_html=True)


# ======================================================================= utilidades
def loc(iso: str) -> datetime:
    return datetime.fromisoformat(iso).astimezone(TZ)


def when(iso: str) -> str:
    d = loc(iso)
    today = datetime.now(TZ).date()
    day = "hoy" if d.date() == today else "mañana" if d.date() == today + timedelta(days=1) else \
        f"{DIAS_C[d.weekday()]} {d:%d/%m}"
    return f"{day} {d:%H:%M}"


def day_title(d) -> str:
    today = datetime.now(TZ).date()
    if d == today:
        return "Hoy"
    if d == today + timedelta(days=1):
        return "Mañana"
    return f"{DIAS[d.weekday()].capitalize()} {d:%d/%m}"


def isnum(x) -> bool:
    return x is not None and not (isinstance(x, float) and np.isnan(x))


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


def badge(tier: str) -> str:
    t, cls = TIER.get(tier, (tier, "b-none"))
    return f'<span class="bdg {cls}">{e(t)}</span>'


def form_html(f: str) -> str:
    return '<span class="fm">' + "".join(f'<i class="{c}">{c}</i>' for c in str(f)) + "</span>"


def section(title: str, n=None, hint: str = ""):
    st.markdown(f'<div class="sec"><h3>{e(title)}</h3>{f"<span class=n>{n}</span>" if n is not None else ""}</div>'
                + (f'<p class="hint">{hint}</p>' if hint else ""), unsafe_allow_html=True)


def empty(msg: str):
    st.markdown(f'<div class="empty">{msg}</div>', unsafe_allow_html=True)


def kpi_strip(items: list[tuple]):
    """items: (etiqueta, valor, detalle, color, clase del valor)."""
    st.markdown('<div class="kpis">' + "".join(
        f'<div class="kpi k-{c}"><div class="l">{e(l)}</div><div class="v {vc}">{v}</div>'
        f'<div class="d">{d}</div></div>' for l, v, d, c, vc in items) + "</div>", unsafe_allow_html=True)


def chart_layout(fig: go.Figure, h=340):
    fig.update_layout(height=h, margin=dict(l=6, r=6, t=12, b=6), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(family="JetBrains Mono, monospace", color=C["text"], size=11),
                      legend=dict(orientation="h", y=1.08, x=0, bgcolor="rgba(0,0,0,0)"),
                      hoverlabel=dict(bgcolor=C["panel"], bordercolor=C["line"], font_family="JetBrains Mono"),
                      hovermode="x unified")
    fig.update_xaxes(gridcolor=C["line"], zeroline=False, linecolor=C["line"])
    fig.update_yaxes(gridcolor=C["line"], zeroline=False, linecolor=C["line"])
    return fig


PLOT_CFG = {"displayModeBar": False, "responsive": True}


# ======================================================================= datos (caché)
@st.cache_data(ttl=180, show_spinner="Cargando el radar…")
def get_data(path: str, mtime: float) -> dict:
    return webdata.load_dashboard(path)


if not s.db_path.exists():
    empty("Aún no hay análisis. Lanza el workflow en GitHub (Actions → Run workflow).")
    st.stop()
D = get_data(str(s.db_path), s.db_path.stat().st_mtime)
SINCE_DEFAULT = webdata.metrics_since_default()


def since_filter(df: pd.DataFrame, since: str) -> pd.DataFrame:
    if df.empty or not since:
        return df
    col = df["shown_at"].fillna(df["created"]) if "shown_at" in df else df["created"]
    return df[col >= since]


def topbar(title: str, sub: str):
    lr = D["last_run"]
    stale = not lr or (datetime.now(timezone.utc) - datetime.fromisoformat(lr)) > timedelta(hours=14)
    sm = webdata.day_summary(D["analysis"])
    st.markdown(
        f'<div class="top"><div class="brand"><b>RADAR//EV</b><h1>{e(title)}</h1><span>{e(sub)}</span></div>'
        f'<div class="tick"><span><span class="live{" off" if stale else ""}"></span>Análisis <b>{when(lr) if lr else "—"}</b></span>'
        f'<span>Partidos <b>{sm["partidos"]}</b></span><span>Élite <b class="am">{sm["elite"]}</b></span>'
        f'<span>Sólidas <b class="pos">{sm["solida"]}</b></span><span>Casa <b>{e(s.book)}</b></span>'
        f'<span>Cuotas <b>{num(s.min_odds)}-{num(s.max_odds)}</b></span></div></div>', unsafe_allow_html=True)


# ======================================================================= componentes
def pick_card(r: dict) -> str:
    sp = SPORTS.get(r["sport"], SPORTS["football"])
    p, odds, tier = float(r["p_final"]), float(r["odds"]), r["tier"]
    implied = 1 / odds if odds > 1 else 1
    fair = 1 / p if p else None
    ev = r.get("ev")
    off = r.get("locked") == 1 and r.get("active") == 0
    badges = badge(tier)
    if r.get("elite") == 1:
        badges += '<span class="bdg b-tg">Telegram</span>'
    if off:
        badges += f'<span class="bdg b-none">Ya no cumple · ahora {num(r.get("current_odds"))}</span>'
    if tier == "verificar":
        cells = (f'<div class="cell hl"><div class="k">Mín. bet365</div><div class="o">{num(r["min_odds"])}</div></div>'
                 f'<div class="cell"><div class="k">Otras casas</div><div class="o">{num(odds)}</div></div>'
                 f'<div class="cell"><div class="k">Justa</div><div class="o">{num(fair)}</div></div>'
                 f'<div class="cell"><div class="k">Prob.</div><div class="o">{pct(p)}</div></div>')
    else:
        stake = r.get("stake") or 0
        cells = (f'<div class="cell hl"><div class="k">bet365</div><div class="o">{num(odds)}</div></div>'
                 f'<div class="cell"><div class="k">Justa</div><div class="o">{num(fair)}</div></div>'
                 f'<div class="cell{" val" if isnum(ev) and ev > 0 else ""}"><div class="k">Valor</div>'
                 f'<div class="o {sign_cls(ev)}">{pct(ev, True)}</div></div>'
                 f'<div class="cell"><div class="k">Stake €</div><div class="o">{num(stake) if stake else "—"}</div></div>')
    reasons = "".join(f"<li>{e(x)}</li>" for x in str(r.get("reasons", "")).split(" | ") if x)
    return (f'<div class="pk t-{tier}{" off" if off else ""}"><div class="pk-h">{badges}'
            f'<span class="w">{when(r["kickoff"])}</span></div>'
            f'<div class="pk-lg">{sp["icono"]} {league_flag(r["league"])} {e(league_name(r["league"]))} · '
            f'{MK.get(r["market"], r["market"])}</div>'
            f'<div class="pk-m">{e(r["home"])} – {e(r["away"])}</div><div class="pk-s">{e(r["label"])}</div>'
            f'<div class="cells">{cells}</div>'
            f'<div class="pbar big"><i style="width:{min(p, 1) * 100:.1f}%"></i>'
            f'<u style="left:calc({min(implied, 1) * 100:.1f}% - 1px)"></u></div>'
            f'<div class="pk-f"><span>Prob. real {pct(p)}</span><span>bet365 implica {pct(implied)}</span>'
            f'<span>Conf. {e(str(r.get("confidence", "—")))}</span></div>'
            f'<details><summary>Por qué</summary><ul>{reasons}</ul></details></div>')


def cards(df: pd.DataFrame, limit=60):
    if df.empty:
        return
    st.markdown(f'<div class="grid">{"".join(pick_card(r) for r in df.head(limit).to_dict("records"))}</div>',
                unsafe_allow_html=True)
    if len(df) > limit:
        st.caption(f"Se muestran {limit} de {len(df)}.")


def radar_rows(df: pd.DataFrame) -> str:
    head = ('<div class="rr h"><div>Hora</div><div>Partido</div><div>Selección</div><div>Prob. real</div>'
            '<div>Justa</div><div>bet365</div><div>Valor</div><div>Semáforo · recomendación</div></div>')
    out = []
    for r in df.to_dict("records"):
        sp = SPORTS.get(r["sport"], SPORTS["football"])
        p, b = r.get("p_final"), r.get("b365_odds")
        ev = r.get("ev") if isnum(b) else None  # sin cuota de bet365 no hay valor que mostrar
        tri = ""
        if isnum(r.get("p_home")):
            parts = [r["p_home"], r.get("p_draw") if isnum(r.get("p_draw")) else None, r["p_away"]]
            tri = '<div class="tri">' + "".join(f'<span style="width:{x * 100:.1f}%"></span>' if x is not None
                                                else '<span style="width:0"></span>' for x in parts) + "</div>"
        pbar = ""
        if isnum(p):
            mk = f'<u style="left:calc({min(1 / b, 1) * 100:.1f}% - 1px)"></u>' if isnum(b) and b > 1 else ""
            pbar = f'<div class="pbar"><i style="width:{p * 100:.1f}%"></i>{mk}</div>'
        books = f'{int(r["n_books"])} casas · {int(r["n_sharp"])} sharp' if isnum(r.get("n_books")) else "sin cuotas"
        age = f' · cuota de hace {r["b365_age_h"]:.0f} h' if isnum(r.get("b365_age_h")) and r["b365_age_h"] > 24 else ""
        d = loc(r["kickoff"])
        out.append(
            f'<div class="rr"><div class="rr-t"><b>{d:%H:%M}</b><span>{DIAS_C[d.weekday()]} {d:%d/%m}</span></div>'
            f'<div class="rr-m"><div class="lg">{sp["icono"]} {league_flag(r["league"])} {e(league_name(r["league"]))}</div>'
            f'<div class="tm">{e(r["home"])}<i>vs</i>{e(r["away"])}</div>{tri}</div>'
            f'<div class="rr-s"><div class="lb">{e(str(r.get("label") or "—"))}</div>'
            f'<div class="sub">{MK.get(r.get("market"), "")} · {books}{age}</div></div>'
            f'<div class="rr-n np"><span class="k">Prob. real</span><b>{pct(p)}</b>{pbar}</div>'
            f'<div class="rr-n nj"><span class="k">Justa</span><b>{num(r.get("fair_odds"))}</b></div>'
            f'<div class="rr-n nb"><span class="k">bet365</span><b class="am">{num(b)}</b></div>'
            f'<div class="rr-n nv"><span class="k">Valor</span><b class="{sign_cls(ev)}">{pct(ev, True)}</b></div>'
            f'<div class="rr-r"><span class="lt {r["light"]}" title="{LIGHT.get(r["light"], "")}"></span>'
            f'<div>{badge(r["tier"]) if not (r["light"] == "gris" and r["tier"] == "pronostico") else badge("modelo")}'
            f'<p>{e(str(r.get("recommendation") or ""))}</p></div></div></div>')
    return f'<div class="rw">{head}{"".join(out)}</div>'


def combo_card(r: dict) -> str:
    legs = json.loads(r["legs"])
    tier = "valor" if r["tier"] == "valor" else "solida"
    items = "".join(
        f'<li><span>{SPORTS.get(l.get("sport", "football"), SPORTS["football"])["icono"]} <b>{e(l["label"])}</b>'
        f'<br><span class="lp">{e(l["partido"])} · {e(league_name(l["league"]))} · {when(l["kickoff"])} · '
        f'prob. {pct(l["p"])}</span></span><span class="lo">{num(l["price"])}</span></li>' for l in legs)
    implied = 1 / r["odds"] if r["odds"] > 1 else 1
    stake = r.get("stake") or 0
    return (f'<div class="pk t-{tier}"><div class="pk-h">'
            f'<span class="bdg {"b-elite" if tier == "valor" else "b-solid"}">{"Con valor" if tier == "valor" else "Sólida"}</span>'
            f'<span class="bdg b-risk">{COMBO_NAME.get(r["risk"], r["risk"])}</span>'
            f'<span class="w">{len(legs)} selecciones · {when(r["first_kickoff"])}</span></div>'
            f'<ul class="legs">{items}</ul><div class="cells">'
            f'<div class="cell hl"><div class="k">Cuota total</div><div class="o">{num(r["odds"])}</div></div>'
            f'<div class="cell"><div class="k">Prob. real</div><div class="o">{pct(r["prob"], dec=1)}</div></div>'
            f'<div class="cell{" val" if r["ev"] > 0 else ""}"><div class="k">Valor</div>'
            f'<div class="o {sign_cls(r["ev"])}">{pct(r["ev"], True)}</div></div>'
            f'<div class="cell"><div class="k">Stake €</div><div class="o">{num(stake) if stake else "—"}</div></div>'
            f'</div><div class="pbar big"><i style="width:{min(r["prob"], 1) * 100:.1f}%"></i>'
            f'<u style="left:calc({min(implied, 1) * 100:.1f}% - 1px)"></u></div>'
            f'<div class="pk-f"><span>Prob. real conjunta {pct(r["prob"], dec=1)}</span>'
            f'<span>La cuota implica {pct(implied, dec=1)}</span></div></div>')


def match_card(m: dict, odds: pd.DataFrame, preds: pd.DataFrame, info: dict, a: dict | None) -> str:
    sport = m["sport"]
    unit = SPORTS[sport]["unidad"]
    main = "1x2" if sport == "football" else "ml"
    sels = ["H", "D", "A"] if main == "1x2" else ["H", "A"]
    pm = preds[preds["market"] == main].drop_duplicates("selection").set_index("selection") if not preds.empty \
        else pd.DataFrame()
    b = odds[odds["bookmaker"] == s.book] if not odds.empty else odds
    probs = {x: float(pm.loc[x, "p_final"]) for x in sels if not pm.empty and x in pm.index}
    tags, xs_h, xs_a = [], "", ""
    if info.get("tipo") == "goles":
        xs_h, xs_a = f'xG {num(info["xg_local"])}', f'xG {num(info["xg_visit"])}'
        if info.get("marcadores"):
            tags.append(f'Marcador más probable {info["marcadores"][0][0]} ({pct(info["marcadores"][0][1])})')
        for side, key in (("local", "bajas_local"), ("visitante", "bajas_visit")):
            if info.get(key):
                tags.append(f'Bajas {side}: {len(info[key])}')
    elif info.get("tipo") in ("tenis", "puntos"):
        xs_h, xs_a = f'Elo {info["elo_local"]:.0f}', f'Elo {info["elo_visit"]:.0f}'
        if info.get("tipo") == "tenis":
            tags.append(f'{info["superficie"].capitalize()} · Elo superficie {info["elo_sup_local"]:.0f}–{info["elo_sup_visit"]:.0f}')
            if info["h2h_local"] + info["h2h_visit"]:
                tags.append(f'Cara a cara {info["h2h_local"]}-{info["h2h_visit"]}')
        else:
            tags.append(f'Margen esperado {info["margen"]:+.1f} · total {info["total"]:.0f}')
    head = (f'<div class="mc-h"><span class="t">{loc(m["kickoff"]):%H:%M}</span>'
            f'<span>{league_flag(m["league"])} {e(league_name(m["league"]))}</span>'
            + (f'<span class="lt {a["light"]}"></span>' if a else "") + '</div>'
            f'<div class="mc-teams"><span class="nm">{e(m["home"])}</span><span class="xs">{xs_h}</span>'
            f'<span class="nm">{e(m["away"])}</span><span class="xs">{xs_a}</span></div>')
    tri = ""
    if probs:
        tri = '<div class="tri" style="height:7px">' + "".join(
            f'<span style="width:{probs.get(x, 0) * 100:.1f}%"></span>' for x in ("H", "D", "A")) + "</div>"
    cells = []
    names = {"H": "1" if main == "1x2" else m["home"], "D": "X", "A": "2" if main == "1x2" else m["away"]}
    for x in sels:
        row = b[(b["market"] == main) & (b["selection"] == x)] if not b.empty else b
        p = probs.get(x)
        if not row.empty:
            price, first = float(row["price"].iloc[0]), float(row["first_price"].iloc[0])
            ev = p * price - 1 if p else None
            mv = ""
            if first and abs(price - first) / first > 0.01:
                mv = f'<span class="mv {"up" if price > first else "dn"}">{"▲" if price > first else "▼"}{num(first)}</span>'
            cells.append(f'<div class="cell{" val" if isnum(ev) and ev >= s.min_ev else ""}"><div class="k">{e(names[x])} · '
                         f'{pct(p)}</div><div class="o">{num(price)}{mv}</div><div class="k">justa {num(1 / p) if p else "—"}'
                         f' · {pct(ev, True)}</div></div>')
        else:
            cells.append(f'<div class="cell"><div class="k">{e(names[x])} · {pct(p)}</div><div class="o mute">—</div>'
                         f'<div class="k">justa {num(1 / p) if p else "—"}</div></div>')
    grid = f'<div class="oc oc{len(sels)}">{"".join(cells)}</div>'
    rows = []
    if not preds.empty:
        for r in preds.drop_duplicates(["market", "selection", "point"]).sort_values(["market", "point", "selection"]).itertuples():
            ob = b[(b["market"] == r.market) & (b["selection"] == r.selection) & (b["point"] == r.point)] if not b.empty else b
            price = float(ob["price"].iloc[0]) if not ob.empty else None
            ev = r.p_final * price - 1 if price else None
            rows.append(f'<tr class="{"v" if isnum(ev) and ev >= s.min_ev else ""}"><td>'
                        f'{e(mlabel(r.market, r.selection, r.point, m["home"], m["away"], unit))}</td>'
                        f'<td>{pct(r.p_final)}</td><td>{num(1 / r.p_final) if r.p_final else "—"}</td>'
                        f'<td>{num(price)}</td><td>{pct(ev, True)}</td></tr>')
    table = (f'<details><summary>Todos los mercados ({len(rows)})</summary><table class="mk"><tr><th>Selección</th>'
             f'<th>Prob.</th><th>Justa</th><th>bet365</th><th>Valor</th></tr>{"".join(rows)}</table></details>') if rows else ""
    if a and a.get("recommendation"):
        tags.insert(0, a["recommendation"])
    n_books = odds["bookmaker"].nunique() if not odds.empty else 0
    tags.append(f"{n_books} casas" if n_books else "Aún sin cuotas")
    tg = '<div class="tagrow">' + "".join(f'<span class="tag">{e(t)}</span>' for t in tags) + "</div>"
    return f'<div class="mc">{head}{tri}{grid}{tg}{table}</div>'


def render_matches(M: pd.DataFrame, limit=60):
    O, Pd, I = D["odds"], D["preds"], D["info"]
    A = {r["match_id"]: r for r in D["analysis"].to_dict("records")} if not D["analysis"].empty else {}
    oo = dict(tuple(O.groupby("match_id"))) if not O.empty else {}
    pp = dict(tuple(Pd.groupby("match_id"))) if not Pd.empty else {}
    M = M.head(limit)
    M = M.assign(_d=M["kickoff"].map(lambda k: loc(k).date()))
    for d, g in M.groupby("_d", sort=True):
        st.markdown(f'<div class="day">{day_title(d)} · {len(g)} partido{"s" if len(g) != 1 else ""}</div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="grid2">' + "".join(
            match_card(m, oo.get(m["id"], pd.DataFrame()), pp.get(m["id"], pd.DataFrame()), I.get(m["id"], {}),
                       A.get(m["id"])) for m in g.to_dict("records")) + "</div>", unsafe_allow_html=True)


def standings(league: str):
    t = D["tables"].get(league)
    if not t:
        empty("Sin clasificación todavía para esta competición.")
        return
    kind = "football" if league in FOOTBALL else "tennis" if league.startswith("tennis_") else "us"
    if kind == "football":
        mx_a = max((r.get("ataque", 0) for r in t), default=1) or 1
        head = ('<tr><th>#</th><th class="l">Equipo</th><th>PJ</th><th>G</th><th>E</th><th>P</th><th>GF</th><th>GC</th>'
                '<th>DG</th><th>Pts</th><th class="l">Forma</th><th class="l">Ataque</th><th class="l">Defensa</th></tr>')
        body = "".join(
            f'<tr><td>{r["pos"]}</td><td class="l b">{e(r["equipo"])}</td><td>{r["pj"]}</td><td>{r["g"]}</td>'
            f'<td>{r["e"]}</td><td>{r["p"]}</td><td>{r["gf"]}</td><td>{r["gc"]}</td><td>{r["dg"]:+d}</td>'
            f'<td class="b">{r["pts"]}</td><td class="l">{form_html(r["forma"])}</td>'
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
        head = ('<tr><th>#</th><th class="l">Equipo</th><th>Elo</th><th>G</th><th>P</th><th>%</th><th>A favor</th>'
                '<th>En contra</th><th class="l">Forma</th></tr>')
        body = "".join(
            f'<tr><td>{r["pos"]}</td><td class="l b">{e(r["equipo"])}</td><td class="b">{r["elo"]}</td><td>{r["g"]}</td>'
            f'<td>{r["p"]}</td><td>{pct(r["pct"])}</td><td>{num(r["pf"], 1)}</td><td>{num(r["pa"], 1)}</td>'
            f'<td class="l">{form_html(r["forma"])}</td></tr>' for r in t)
    st.markdown(f'<div class="scroll"><table class="tb">{head}{body}</table></div>', unsafe_allow_html=True)


# ======================================================================= páginas
def page_radar():
    topbar("Radar de la jornada", "Probabilidad real frente a la cuota de bet365, partido a partido")
    A = D["analysis"]
    sm = webdata.day_summary(A)
    res = since_filter(D["resolved"], SINCE_DEFAULT)
    k = webdata.kpis(res[res["tier"] == "valor"]) if not res.empty else {"n": 0}
    kpi_strip([
        ("Partidos analizados", str(sm["partidos"]), f'{sm["verde"]} verdes · {sm["ambar"]} ámbar · {sm["rojo"]} rojos',
         "cyan", ""),
        ("Élite", str(sm["elite"]), "valor ≥ +" + num(s.min_ev * 100, 1) + " % frente al precio justo", "amber", "am"),
        ("Sólidas", str(sm["solida"]), f"prob. ≥ {pct(s.solid_min_prob)} a precio justo", "green", "pos"),
        ("Por confirmar", str(sm.get("verificar", 0)), "falta cuota de bet365 reciente", "violet", ""),
        ("Yield Élite", pct(k.get("yield"), True) if k["n"] else "—",
         f'{k["n"]} resueltas' + (f" desde {SINCE_DEFAULT}" if SINCE_DEFAULT else " · todo el historial"),
         "green" if k.get("yield", 0) and k["yield"] >= 0 else "red", sign_cls(k.get("yield"))),
        ("CLV", pct(k.get("clv"), True) if k.get("clv_n") else "—", f'{k.get("clv_n", 0)} con cierre de Pinnacle',
         "cyan", sign_cls(k.get("clv")))])
    P = D["picks"]
    top = P[P["tier"].isin(["valor", "solida"])] if not P.empty else P
    if not top.empty:
        top = top.assign(_o=top["tier"].map({"valor": 0, "solida": 1}), _a=top["active"].fillna(1) * -1) \
            .sort_values(["_o", "_a", "score"], ascending=[True, True, False])
        section("Destacadas", len(top), "Élite: bet365 paga más que el precio justo limpio. Sólidas: alta probabilidad "
                                        "y bet365 paga el precio justo (sin ventaja, sin regalar margen).")
        cards(top, 6)

    section("Todos los partidos", len(A) if not A.empty else 0,
            "Cada fila es un partido: la selección más interesante, su probabilidad real (barra verde) frente a la que "
            "implica la cuota de bet365 (marca ámbar), el valor esperado y una recomendación.")
    if A.empty:
        empty("Todavía no hay análisis de partidos. Aparecerán tras la próxima ejecución del radar "
              "(GitHub Actions, cada 3 horas).")
        return
    c1, c2, c3 = st.columns([2.2, 1.6, 1.4])
    sports = ["todos"] + [k for k in SPORTS if k in set(A["sport"])]
    sp = c1.pills("Deporte", sports, default="todos", key="r_sp", label_visibility="collapsed",
                  format_func=lambda k: "Todos" if k == "todos" else f'{SPORTS[k]["icono"]} {SPORTS[k]["nombre"]}') or "todos"
    hours = c2.segmented_control("Ventana", [12, 24, 48, 0], default=48, key="r_h", label_visibility="collapsed",
                                 format_func=lambda h: "Todo" if h == 0 else f"{h} h")
    q = c3.text_input("Buscar", placeholder="Equipo o liga…", key="r_q", label_visibility="collapsed")
    lights = st.segmented_control("Semáforo", list(LIGHT), selection_mode="multi", default=list(LIGHT), key="r_l",
                                  label_visibility="collapsed", format_func=lambda x: LIGHT[x]) or list(LIGHT)
    T = webdata.radar_table(A, sp, lights, hours or None, query=q or "")
    if T.empty:
        empty("Ningún partido con estos filtros.")
        return
    T = T.assign(_d=T["kickoff"].map(lambda x: loc(x).date()))
    for d, g in T.groupby("_d", sort=True):
        st.markdown(f'<div class="day">{day_title(d)} · {len(g)} partidos</div>', unsafe_allow_html=True)
        st.markdown(radar_rows(g.head(150)), unsafe_allow_html=True)


def page_picks():
    topbar("Jugadas", "Élite, sólidas, por confirmar y a revisar")
    P = D["picks"]
    if P.empty:
        empty("No hay jugadas para los próximos días. El radar se actualiza cada 3 horas.")
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
    tabs = st.tabs([f'Élite · {counts.get("valor", 0)}', f'Sólidas · {counts.get("solida", 0)}',
                    f'Por confirmar · {counts.get("verificar", 0)}', f'Revisar · {counts.get("revisar", 0)}'])
    texts = {
        "valor": ("bet365 paga más que el precio justo limpio (Pinnacle o consenso de casas válidas a la misma hora). "
                  "Las marcadas con Telegram tienen referencia sharp y se han enviado como aviso.",
                  "Ahora mismo ninguna cuota de bet365 supera el precio justo en el rango "
                  f"{num(s.min_odds)}-{num(s.max_odds)}. Mira las sólidas y el radar de la jornada."),
        "solida": ("Probabilidad real alta y bet365 paga al menos el precio justo. No es ventaja: es no regalar margen. "
                   "Son la base de las combinadas.", "No hay sólidas para los próximos días."),
        "verificar": ("El mercado sugiere valor pero falta la cuota de bet365 reciente: apuesta solo si bet365 paga la "
                      "cuota mínima indicada.", "Nada pendiente de confirmar."),
        "revisar": ("Valor esperado anormalmente alto: casi siempre es una cuota mal publicada. Nunca se apuesta.",
                    "Ninguna anomalía detectada."),
    }
    for tab, tier in zip(tabs, ["valor", "solida", "verificar", "revisar"]):
        with tab:
            hint, msg = texts[tier]
            st.markdown(f'<p class="hint">{hint}</p>', unsafe_allow_html=True)
            sub = P[P["tier"] == tier]
            if sub.empty:
                empty(msg)
            else:
                cards(sub, 60)


def page_combos():
    topbar("Combinadas", "Solo selecciones élite o sólidas, de partidos y ligas distintas")
    st.markdown('<p class="hint">Cada combinada cruza selecciones de partidos y ligas diferentes, así que la probabilidad '
                'real conjunta es el producto de sus probabilidades. Se compara con la probabilidad que implica la cuota '
                'total de bet365: si la supera, la combinada tiene valor. Las sólidas casi no regalan margen, pero no '
                'tienen ventaja: apuesta poco.</p>', unsafe_allow_html=True)
    Cb = D["combos"]
    if Cb.empty:
        empty("No hay combinadas fiables para las próximas 36 horas: hacen falta al menos dos selecciones élite o sólidas "
              "de ligas distintas. Prueba el Asistente con tus condiciones.")
        return
    t1, t2 = st.tabs([f'Con valor · {int((Cb["tier"] == "valor").sum())}',
                      f'Sólidas · {int((Cb["tier"] != "valor").sum())}'])
    for tab, sub in ((t1, Cb[Cb["tier"] == "valor"]), (t2, Cb[Cb["tier"] != "valor"])):
        with tab:
            if sub.empty:
                empty("Ninguna combinada de este tipo ahora mismo.")
                continue
            for risk in ("conservadora", "equilibrada", "arriesgada"):
                g = sub[sub["risk"] == risk]
                if not g.empty:
                    section(COMBO_NAME[risk], len(g))
                    st.markdown(f'<div class="grid">{"".join(combo_card(r) for r in g.to_dict("records"))}</div>',
                                unsafe_allow_html=True)


def page_matches():
    topbar("Partidos", "Fichas por competición, todos los mercados y clasificaciones")
    M = D["matches"]
    sports = [k for k in SPORTS if k in s.active_sports]
    sp = st.pills("Deporte", sports, default=sports[0] if sports else None, key="m_sp", label_visibility="collapsed",
                  format_func=lambda k: f'{SPORTS[k]["icono"]} {SPORTS[k]["nombre"]}') or (sports[0] if sports else "football")
    Ms = M[M["sport"] == sp] if not M.empty else M
    t_p, t_c = st.tabs(["Partidos", "Clasificación"])
    with t_p:
        if Ms.empty:
            empty(f"No hay partidos de {SPORTS[sp]['nombre'].lower()} en los próximos días.")
        else:
            counts = Ms["league"].value_counts()
            order = sorted(counts.index, key=lambda k: (FOOTBALL.get(k, {}).get("tier", 1), -counts[k]))
            lg = st.selectbox("Competición", ["todas"] + order, key=f"m_lg_{sp}",
                              format_func=lambda k: f"Todas ({len(Ms)})" if k == "todas" else
                              f"{league_flag(k)} {league_name(k)} ({counts[k]})")
            Ml = Ms if lg == "todas" else Ms[Ms["league"] == lg]
            n = len(Ml)
            if n > 60:
                page = st.number_input(f"Página (60 por página, {n} partidos)", 1, (n - 1) // 60 + 1, 1, key=f"m_pg_{sp}")
                Ml = Ml.iloc[(page - 1) * 60:]
            render_matches(Ml)
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
                if sp == "football":
                    st.caption("Ataque y defensa: fuerza estimada por el modelo. Forma: últimos 5 partidos.")


def page_assistant():
    topbar("Asistente de apuesta", "Presupuesto, fecha límite y riesgo: elige entre todo lo analizado")
    now = datetime.now(timezone.utc)
    with st.form("asis"):
        c1, c2, c3 = st.columns(3)
        budget = c1.number_input("Presupuesto (€)", min_value=1.0, max_value=10000.0, value=10.0, step=1.0)
        dl_date = c2.date_input("Que se resuelva antes del día", value=(datetime.now(TZ) + timedelta(days=2)).date(),
                                min_value=datetime.now(TZ).date(), format="DD/MM/YYYY")
        dl_time = c3.time_input("a las", value=time(23, 59), step=timedelta(minutes=30))
        c4, c5 = st.columns(2)
        risk = c4.segmented_control("Riesgo", ["bajo", "medio", "alto"], default="bajo",
                                    format_func=lambda r: RISK_NAME[r]) or "bajo"
        tipo = c5.segmented_control("Tipo", ["auto", "simple", "combinada", "repartir"], default="auto",
                                    format_func=lambda t: {"auto": "Lo mejor", "simple": "Simple",
                                                           "combinada": "Combinada", "repartir": "Repartir"}[t]) or "auto"
        c6, c7 = st.columns([2, 1])
        sports = c6.multiselect("Deportes", [k for k in SPORTS if k in s.active_sports],
                                default=[k for k in SPORTS if k in s.active_sports],
                                format_func=lambda k: f'{SPORTS[k]["icono"]} {SPORTS[k]["nombre"]}')
        only_good = c7.toggle("Solo élite y sólidas", value=True,
                              help="Descarta las selecciones donde bet365 cobra margen por encima del precio justo.")
        go_ = st.form_submit_button("Buscar la mejor apuesta", type="primary", use_container_width=True)
    if not go_:
        st.markdown('<p class="hint">Usa todas las selecciones con cuota de bet365 de menos de 72 h y descarta las que no '
                    'terminan antes de tu fecha límite. Riesgo bajo busca cuotas totales de 1,25 a 2,2; medio de 2 a 4,5; '
                    'alto de 4 a 15.</p>', unsafe_allow_html=True)
        return
    deadline = datetime.combine(dl_date, dl_time, TZ).astimezone(timezone.utc)
    pool = D["pool"]
    if only_good and not pool.empty:
        pool = pool[pool["tier"].isin(["valor", "solida"])]
    plan, alts, n = assistant.recommend(pool, budget, deadline, now, risk, tipo, sports)
    if plan is None:
        empty(f"No encuentro selecciones que terminen antes del {dl_date:%d/%m} a las {dl_time:%H:%M}"
              f"{' entre las élite y sólidas' if only_good else ''}. Amplía la fecha o desactiva el filtro.")
        return
    render_plan(plan, n, main=True)
    for a in alts:
        render_plan(a, n, main=False)
    st.caption("Las probabilidades son estimaciones. Comprueba la cuota en bet365 antes de apostar: si ha bajado, la "
               "recomendación puede dejar de tener sentido. Apuesta solo lo que puedas permitirte perder.")


def render_plan(p: assistant.Plan, n: int, main: bool):
    title = {"simple": "Apuesta simple", "combinada": f"Combinada de {len(p.legs)}",
             "repartir": f"Reparto en {len(p.legs)} simples"}[p.tipo]
    stakes = p.stakes if p.tipo == "repartir" else [p.budget] * len(p.legs)
    rows = []
    for st_, l in zip(stakes, p.legs):
        sp = SPORTS.get(l["sport"], SPORTS["football"])["icono"]
        stake = f' · apuesta {num(st_)} €' if p.tipo == "repartir" else ""
        rows.append(f'<li><span>{sp} {badge(l.get("tier", "pronostico"))} <b>{e(l["label"])}</b><br><span class="lp">'
                    f'{e(l["home"])} – {e(l["away"])} · {when(l["kickoff"])} · prob. {pct(l["p_final"])} · valor '
                    f'{pct(l["ev"], True)}{stake}</span></span><span class="lo">{num(l["odds"])}</span></li>')
    if p.tipo == "repartir":
        big = (f'<div><b>{num(p.budget)} €</b><span>repartidos</span></div>'
               f'<div><b>{pct(p.prob)}</b><span>prob. de acertar al menos una</span></div>'
               f'<div><b class="{sign_cls(p.ev)}">{num(p.expected)} €</b><span>ganancia esperada</span></div>'
               f'<div><b>{num(p.win)} €</b><span>beneficio si aciertan todas</span></div>')
    else:
        big = (f'<div><b class="am">{num(p.odds)}</b><span>cuota bet365</span></div>'
               f'<div><b>{pct(p.prob)}</b><span>probabilidad real</span></div>'
               f'<div><b>{num(p.budget * p.odds)} €</b><span>cobras si aciertas</span></div>'
               f'<div><b class="{sign_cls(p.ev)}">{pct(p.ev, True)}</b><span>valor esperado ({num(p.expected)} €)</span></div>')
    st.markdown(
        f'<div class="{"plan" if main else "alt"}"><h4>{"Recomendación · " if main else "Alternativa · "}{title}</h4>'
        f'<div class="mute" style="font-size:.8rem">Elegida entre {n} selecciones · se resuelve hacia el {when(p.ends)}</div>'
        f'{f"<div class=big>{big}</div>" if main else ""}<ul class="legs">{"".join(rows)}</ul>'
        + ("" if main else f'<div class="mute" style="font-size:.8rem;margin-top:.4rem">valor {pct(p.ev, True)} · '
                           f'prob. {pct(p.prob)}</div>') + "</div>", unsafe_allow_html=True)
    if main and p.ev < 0:
        st.caption("Ninguna opción tiene valor positivo con este filtro: es la menos mala. A la larga, el margen de "
                   "bet365 hace perder dinero en este tipo de apuesta.")


def page_perf():
    topbar("Rendimiento", "Resultados reales de lo que el radar recomendó, congelado en el momento de mostrarlo")
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
    tiers = c2.segmented_control("Nivel", ["valor", "solida", "ambos"], default="valor", key="pf_tier",
                                 label_visibility="collapsed",
                                 format_func=lambda t: {"valor": "Élite", "solida": "Sólidas", "ambos": "Ambas"}[t]) or "valor"
    mode = c3.segmented_control("Medida", ["unit", "stake"], default="unit", key="pf_mode", label_visibility="collapsed",
                                format_func=lambda m: {"unit": "1 unidad por jugada", "stake": "Stakes reales (€)"}[m]) or "unit"
    if not SINCE_DEFAULT:
        st.caption("Consejo: crea la variable METRICS_SINCE con la fecha del arreglo del motor para ver aquí el "
                   "rendimiento limpio por defecto.")
    R = since_filter(R, since)
    if tiers != "ambos" and not R.empty:
        R = R[R["tier"] == tiers]
    k = webdata.kpis(R, mode)
    unit = " €" if mode == "stake" and not R.empty and R["stake"].fillna(0).sum() > 0 else " u"
    if not k["n"]:
        kpi_strip([("Jugadas resueltas", "0", "en este periodo", "cyan", "")])
        empty("Todavía no hay jugadas resueltas en este periodo. En cuanto se jueguen los primeros partidos verás aquí "
              "la curva de banca, el yield y el desglose por deporte, mercado y cuota.")
    else:
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
        ser = webdata.perf_series(R, s.bankroll, "stake" if unit == " €" else "unit")
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.74, 0.26], vertical_spacing=0.04,
                            specs=[[{"secondary_y": True}], [{}]])
        fig.add_trace(go.Scatter(x=ser["date"], y=ser["bank"], mode="lines", line=dict(color=C["amber"], width=2.5, shape="hv"),
                                 name="Banca" if unit == " €" else "Unidades", fill="tozeroy" if unit == " u" else None,
                                 fillcolor="rgba(255,176,32,.07)", hovertemplate="%{y:.2f}<extra></extra>"), row=1, col=1)
        fig.add_trace(go.Scatter(x=ser["date"], y=ser["yield"], mode="lines", line=dict(color=C["cyan"], width=1.6, dash="dot"),
                                 name="Yield acumulado", hovertemplate="%{y:.1%}<extra></extra>"), row=1, col=1,
                      secondary_y=True)
        fig.add_trace(go.Scatter(x=ser["date"], y=ser["drawdown"], mode="lines", line=dict(color=C["red"], width=1),
                                 fill="tozeroy", fillcolor="rgba(255,92,92,.15)", name="Caída",
                                 hovertemplate="%{y:.2f}<extra></extra>"), row=2, col=1)
        fig.add_hline(y=s.bankroll if unit == " €" else 0, line_color=C["mute"], line_width=1, line_dash="dash", row=1, col=1)
        fig.update_yaxes(tickformat=".0%", secondary_y=True, row=1, col=1, showgrid=False)
        section("Evolución de la banca y del yield", hint="Línea ámbar: banca (o unidades acumuladas). Punteada: yield "
                                                           "acumulado. Abajo: caída desde el máximo.")
        st.plotly_chart(chart_layout(fig, 430), width="stretch", config=PLOT_CFG)
        section("Desglose")
        tabs = st.tabs(["Deporte", "Mercado", "Rango de cuota", "Competición", "Nivel"])
        names = {"sport": lambda x: SPORTS.get(x, {}).get("nombre", x), "market": lambda x: MK.get(x, x),
                 "band": lambda x: x, "league": lambda x: f"{league_flag(x)} {league_name(x)}",
                 "tier": lambda x: TIER.get(x, (x,))[0]}
        for tab, by in zip(tabs, ["sport", "market", "band", "league", "tier"]):
            with tab:
                b = webdata.breakdown(R, by)
                b["grupo"] = b["grupo"].map(names[by])
                cA, cB = st.columns([1.1, 1])
                bf = go.Figure(go.Bar(x=b["yield"], y=b["grupo"], orientation="h",
                                      marker_color=[C["green"] if v >= 0 else C["red"] for v in b["yield"]],
                                      text=[f"{v:+.1%}" for v in b["yield"]], textposition="outside",
                                      hovertemplate="%{y}: %{x:.1%}<extra></extra>"))
                bf.update_xaxes(tickformat=".0%")
                bf.update_yaxes(autorange="reversed")
                cA.plotly_chart(chart_layout(bf, max(180, 46 * len(b) + 60)), width="stretch", config=PLOT_CFG)
                t = b.rename(columns={"grupo": "Grupo", "n": "Jugadas", "acierto": "Acierto", "esperado": "Esperado",
                                      "yield": "Yield", "unidades": "Unidades"})
                cB.dataframe(t.style.format({"Acierto": "{:.0%}", "Esperado": "{:.0%}", "Yield": "{:+.1%}",
                                             "Unidades": "{:+.2f}"}), hide_index=True, width="stretch")
    RC = since_filter(D["resolved_combos"], since)
    if not RC.empty:
        section("Combinadas resueltas", len(RC))
        RC = RC.assign(u=(RC["odds"] - 1) * RC["won"] - (1 - RC["won"]))
        g = RC.groupby("tier").agg(n=("won", "size"), acierto=("won", "mean"), unidades=("u", "sum"))
        g["yield"] = g["unidades"] / g["n"]
        g.index = [{"valor": "Con valor", "solida": "Sólidas"}.get(i, i) for i in g.index]
        st.dataframe(g.rename(columns={"n": "Combinadas", "acierto": "Acierto", "unidades": "Unidades", "yield": "Yield"})
                     .style.format({"Acierto": "{:.0%}", "Unidades": "{:+.2f}", "Yield": "{:+.1%}"}), width="stretch")
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
        fig.update_layout(hovermode="closest")
        st.plotly_chart(chart_layout(fig, 340), width="stretch", config=PLOT_CFG)


def page_status():
    topbar("Estado y método", "De dónde salen los datos y cómo decide el radar")
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
        '<p class="hint">1. <b>Precio justo limpio</b>: mercado sin margen con casas válidas (mercado completo, margen '
        'creíble y cuotas tomadas como mucho 6 h antes o después de la de bet365). Si hay Pinnacle o exchanges, solo '
        'ellos. bet365 nunca entra en su propio consenso.<br>2. <b>Élite</b>: bet365 paga al menos un '
        f'{pct(s.min_ev, True)} más que ese precio, con cuota entre {num(s.min_odds)} y {num(s.max_odds)}. A Telegram solo '
        f'van las de {pct(s.alert_ev, True)} o más con referencia sharp.<br>3. <b>Sólida</b>: probabilidad real de al menos '
        f'{pct(s.solid_min_prob)} y bet365 paga el precio justo (valor ≥ {pct(s.solid_min_ev, True)}). Sin ventaja, sin '
        'regalar margen.<br>4. <b>Revisar</b>: valor de más del '
        f'{pct(s.max_ev)}: casi siempre es un error de datos.<br>5. <b>Seguimiento</b>: lo recomendado queda congelado con '
        'su cuota y probabilidad. El CLV solo se mide contra el cierre de Pinnacle.</p>', unsafe_allow_html=True)
    w = D["weights"] or {}
    if w:
        rows = [{"Deporte": SPORTS.get(k.split("|")[0], {}).get("nombre", k), "Mercado": MK.get(k.split("|")[1], k),
                 "Peso del modelo": v} for k, v in sorted(w.items())]
        st.dataframe(pd.DataFrame(rows).style.format({"Peso del modelo": "{:.0%}"}), hide_index=True, width="stretch")
    for key, title in (("backtest", "Backtest de fútbol (último año)"), ("backtest_tennis", "Backtest de tenis")):
        bt = D.get(key)
        if not bt or not bt.get("estrategia_bet365"):
            continue
        section(title, f'{bt["desde"]} a {bt["hasta"]}')
        t = pd.DataFrame(bt["estrategia_bet365"]).T
        t.index = [f"Valor ≥ {k.split('_')[1]} %" for k in t.index]
        t["Intervalo"] = t["roi_ic95"].map(lambda x: f"{x[0]:+.0%} a {x[1]:+.0%}" if isnum(x[0]) else "—")
        t = t[["apuestas", "acierto", "cuota_media", "roi", "Intervalo"]]
        t.columns = ["Apuestas", "Acierto", "Cuota media", "Rentabilidad", "Intervalo 95 %"]
        st.dataframe(t.style.format({"Acierto": "{:.0%}", "Cuota media": "{:.2f}", "Rentabilidad": "{:+.1%}"}),
                     width="stretch")
    section("Últimos análisis")
    R = D["runs"]
    if not R.empty:
        def resume(txt: str) -> dict:
            try:
                d = ast.literal_eval(txt)
            except (ValueError, SyntaxError):
                return {}
            dep = d.get("por_deporte") or {}
            return {"Partidos": d.get("partidos"),
                    "Por deporte": " ".join(f'{SPORTS[k]["icono"]} {v}' for k, v in dep.items() if k in SPORTS),
                    "Élite": d.get("valor"), "Sólidas": d.get("solidas"), "Por confirmar": d.get("verificar"),
                    "Avisos": d.get("avisos"), "Liquidados": d.get("liquidados"), "OddsPapi": d.get("oddspapi_mes")}
        st.dataframe(pd.DataFrame([{"Hora": when(r.ts), **resume(r.detail)} for r in R.itertuples()]),
                     hide_index=True, width="stretch")
    st.markdown('<p class="hint" style="margin-top:1.2rem">Juega solo con dinero que puedas perder. Si deja de ser '
                'divertido: <a href="https://www.jugarbien.es">jugarbien.es</a> · autoexclusión en el RGIAJ.</p>',
                unsafe_allow_html=True)


# ======================================================================= navegación
pages = {
    "Radar": [st.Page(page_radar, title="Radar de la jornada", icon="📡", default=True),
              st.Page(page_picks, title="Jugadas", icon="🎯", url_path="jugadas"),
              st.Page(page_combos, title="Combinadas", icon="🧩", url_path="combinadas"),
              st.Page(page_matches, title="Partidos", icon="🗓️", url_path="partidos")],
    "Herramientas": [st.Page(page_assistant, title="Asistente de apuesta", icon="🧮", url_path="asistente")],
    "Seguimiento": [st.Page(page_perf, title="Rendimiento", icon="📈", url_path="rendimiento"),
                    st.Page(page_status, title="Estado y método", icon="🛰️", url_path="estado")],
}
st.navigation(pages, position="top").run()
