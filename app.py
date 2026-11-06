"""Radar de apuestas · panel web. Local: `streamlit run app.py` · Internet: Streamlit Community Cloud."""
from __future__ import annotations

import html
import json
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from radar.config import SPORTS, Settings, league_flag, league_name
from radar.db import DB
from radar.learn import metrics

TZ = ZoneInfo("Europe/Madrid")
DIAS = ["lun", "mar", "mié", "jue", "vie", "sáb", "dom"]
RISK_NAME = {"bajo": "Riesgo bajo", "medio": "Riesgo medio", "alto": "Riesgo alto"}
COMBO_NAME = {"conservadora": "Conservadora", "equilibrada": "Equilibrada", "arriesgada": "Arriesgada"}
COMBO_RISK = {"conservadora": "bajo", "equilibrada": "medio", "arriesgada": "alto"}

st.set_page_config(page_title="Radar · bet365", page_icon="📡", layout="wide", initial_sidebar_state="collapsed")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow:wght@400;500;600&family=Barlow+Condensed:wght@500;600;700&display=swap');
:root{--ink:#0E1726;--panel:#172338;--panel2:#1D2B44;--line:#2A3A57;--text:#ECE8DF;--mute:#93A0B8;
--gold:#F0B43C;--lo:#3DB4A2;--mid:#E9A23B;--hi:#E4685A;--pos:#3DB4A2;--neg:#E4685A}
html,body,[data-testid="stAppViewContainer"],[data-testid="stHeader"]{background:var(--ink)!important;color:var(--text)}
[data-testid="stHeader"]{height:0}
.block-container{padding:3rem 1rem 4rem;max-width:1240px}
[data-testid="stButtonGroup"]>div{flex-wrap:wrap;row-gap:6px}
*{font-family:'Barlow',system-ui,sans-serif}
h1,h2,h3,.cond{font-family:'Barlow Condensed','Arial Narrow',sans-serif!important;letter-spacing:.01em}
p,li,span,div{color:inherit}
a{color:var(--gold)}
/* cabecera */
.top{display:flex;align-items:flex-end;justify-content:space-between;gap:1rem;flex-wrap:wrap;margin-bottom:1.2rem}
.brand{font-family:'Barlow Condensed','Arial Narrow',sans-serif;font-weight:700;font-size:2.6rem;line-height:1;color:var(--text)}
.brand small{display:block;font-family:'Barlow',system-ui,sans-serif;font-weight:500;font-size:.95rem;color:var(--mute);margin-top:.35rem}
.status{color:var(--mute);font-size:.9rem;text-align:right}
.status b{color:var(--text);font-weight:600}
.dot{display:inline-block;width:.55rem;height:.55rem;border-radius:50%;background:var(--lo);margin-right:.4rem;
box-shadow:0 0 0 0 rgba(61,180,162,.6);animation:pulse 2.4s ease-out 1}
@keyframes pulse{0%{box-shadow:0 0 0 0 rgba(61,180,162,.6)}100%{box-shadow:0 0 0 12px rgba(61,180,162,0)}}
@media (prefers-reduced-motion:reduce){.dot{animation:none}}
/* cifras */
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:1px;background:var(--line);
border:1px solid var(--line);border-radius:14px;overflow:hidden;margin:.4rem 0 1.4rem}
.kpi{background:var(--panel);padding:.9rem 1rem}
.kpi .v{font-family:'Barlow Condensed','Arial Narrow',sans-serif;font-weight:600;font-size:1.9rem;line-height:1.1}
.kpi .l{color:var(--mute);font-size:.85rem}
.pos{color:var(--pos)!important}.neg{color:var(--neg)!important}
/* sección */
.sec{display:flex;align-items:baseline;gap:.7rem;margin:1.6rem 0 .7rem}
.sec h3{margin:0;font-size:1.55rem;font-weight:600}
.sec .n{color:var(--mute);font-size:.95rem}
.hint{color:var(--mute);font-size:.92rem;margin:-.3rem 0 .8rem;max-width:72ch}
/* boletos */
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:14px}
.tk{display:grid;grid-template-columns:1fr 112px;background:var(--panel);border:1px solid var(--line);
border-radius:12px;overflow:hidden;position:relative}
.tk::before{content:"";position:absolute;left:0;top:0;bottom:0;width:4px;background:var(--rc)}
.r-bajo{--rc:var(--lo)}.r-medio{--rc:var(--mid)}.r-alto{--rc:var(--hi)}
.tk-main{padding:.85rem .9rem .8rem 1.1rem;min-width:0}
.meta{color:var(--mute);font-size:.82rem;display:flex;gap:.5rem;flex-wrap:wrap}
.meta .rk{color:var(--rc);font-weight:600}
.match{font-weight:600;font-size:1.02rem;margin:.25rem 0 .1rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.sel{font-family:'Barlow Condensed','Arial Narrow',sans-serif;font-weight:600;font-size:1.45rem;line-height:1.15;color:var(--text)}
.meter{position:relative;height:8px;background:var(--panel2);border-radius:4px;margin:.7rem 0 .35rem}
.meter .fill{position:absolute;left:0;top:0;bottom:0;border-radius:4px;background:var(--rc);opacity:.85}
.meter .tick{position:absolute;top:-4px;bottom:-4px;width:2px;background:var(--gold)}
.mlab{display:flex;justify-content:space-between;color:var(--mute);font-size:.78rem}
.mlab b{color:var(--text);font-weight:600}
details{margin-top:.55rem;font-size:.85rem;color:var(--mute)}
details summary{cursor:pointer;color:var(--text);opacity:.8;list-style:none}
details summary::-webkit-details-marker{display:none}
details summary::before{content:"＋ ";color:var(--gold)}
details[open] summary::before{content:"－ "}
details ul{margin:.4rem 0 0;padding-left:1.1rem}
.stub{background:var(--panel2);border-left:2px dashed var(--line);display:flex;flex-direction:column;
align-items:center;justify-content:center;padding:.6rem .4rem;text-align:center;position:relative}
.stub::before,.stub::after{content:"";position:absolute;left:-9px;width:16px;height:16px;border-radius:50%;background:var(--ink)}
.stub::before{top:-8px}.stub::after{bottom:-8px}
.stub .bk{font-size:.75rem;color:var(--mute)}
.stub .od{font-family:'Barlow Condensed','Arial Narrow',sans-serif;font-weight:700;font-size:2.3rem;line-height:1;color:var(--gold)}
.stub .od.muted{color:var(--text);opacity:.75}
.stub .ev{font-weight:600;font-size:.95rem;margin-top:.15rem}
.stub .sk{font-size:.78rem;color:var(--mute);margin-top:.25rem}
.badge{display:inline-block;font-size:.72rem;font-weight:600;padding:.1rem .45rem;border-radius:20px;
background:rgba(240,180,60,.14);color:var(--gold);margin-left:.2rem}
.tk.elite{border-color:rgba(240,180,60,.55)}
/* combinadas */
.legs{margin:.45rem 0 0;padding:0;list-style:none}
.legs li{padding:.35rem 0;border-top:1px solid var(--line);font-size:.9rem;display:flex;justify-content:space-between;gap:.6rem}
.legs li:first-child{border-top:0}
.legs .lp{color:var(--mute);font-size:.8rem}
.legs .lo{font-family:'Barlow Condensed','Arial Narrow',sans-serif;font-weight:600;font-size:1.1rem;color:var(--text)}
.empty{background:var(--panel);border:1px dashed var(--line);border-radius:12px;padding:1.1rem 1.2rem;color:var(--mute)}
/* controles de streamlit */
[data-testid="stTabs"] button p{font-family:'Barlow Condensed','Arial Narrow',sans-serif;font-size:1.15rem;font-weight:600}
[data-testid="stTabs"] [aria-selected="true"] p{color:var(--gold)}
[data-baseweb="tab-highlight"]{background:var(--gold)!important}
[data-testid="stMarkdownContainer"] p{color:var(--text)}
[data-testid="stDataFrame"]{border:1px solid var(--line);border-radius:10px}
button:focus-visible,summary:focus-visible{outline:2px solid var(--gold);outline-offset:2px}
@media (max-width:600px){.kpi:last-child{grid-column:1/-1}.brand{font-size:2.1rem}.status{text-align:left}.tk{grid-template-columns:1fr 92px}
.stub .od{font-size:1.9rem}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

s = Settings()
e = html.escape


def when(iso: str) -> str:
    d = datetime.fromisoformat(iso).astimezone(TZ)
    today = datetime.now(TZ).date()
    day = "hoy" if d.date() == today else "mañana" if d.date() == today + timedelta(days=1) else \
        f"{DIAS[d.weekday()]} {d:%d/%m}"
    return f"{day} {d:%H:%M}"


def pct(x, signed=False) -> str:
    if x is None or pd.isna(x):
        return "—"
    return f"{x:+.1%}".replace(".", ",") if signed else f"{x:.0%}"


def num(x: float, d=2) -> str:
    return f"{x:.{d}f}".replace(".", ",")


@st.cache_data(ttl=300)
def load(_mtime: float):
    db = DB(s.db_path)
    now = datetime.now(timezone.utc).isoformat()
    d = {
        "picks": db.df("SELECT * FROM picks WHERE kickoff > ? AND bookmaker = 'bet365' ORDER BY score DESC", (now,)),
        "done": db.df("SELECT * FROM picks WHERE won IN (0,1) AND tier='valor' AND bookmaker='bet365' ORDER BY kickoff"),
        "pron_done": db.df("SELECT * FROM picks WHERE won IN (0,1) AND tier='pronostico' AND bookmaker='bet365'"),
        "combos": db.df("SELECT * FROM combos WHERE won IS NULL AND first_kickoff > ? AND bookmaker='bet365' "
                        "ORDER BY ev DESC", (now,)),
        "matches": db.df("SELECT sport, count(*) n FROM matches WHERE status='pendiente' AND kickoff > ? GROUP BY sport",
                         (now,)),
        "metrics": metrics(db), "backtest": db.get("backtest"), "backtest_tennis": db.get("backtest_tennis"),
        "weights": db.get("weights"), "last_run": db.get("last_run"), "credits": db.get("odds_api_remaining"),
        "oddspapi": db.get(f"oddspapi_used_{datetime.now(timezone.utc):%Y-%m}", 0),
    }
    db.close()
    return d


if not s.db_path.exists():
    st.markdown('<div class="empty">Aún no hay análisis. Lanza el workflow en GitHub (Actions → Run workflow) '
                'o ejecuta <code>python -m radar run</code>.</div>', unsafe_allow_html=True)
    st.stop()

d = load(s.db_path.stat().st_mtime)
P = d["picks"]
m = d["metrics"]

# ------------------------------------------------------------------ cabecera
lr = d["last_run"]
n_val = int((P["tier"] == "valor").sum()) if not P.empty else 0
st.markdown(f"""<div class="top"><div class="brand">Radar<small>Cuotas de bet365 frente a la probabilidad real</small></div>
<div class="status"><span class="dot"></span>Último análisis <b>{when(lr) if lr else '—'}</b><br>
{n_val} jugada{'s' if n_val != 1 else ''} con valor ahora mismo</div></div>""", unsafe_allow_html=True)

roi_cls = "" if not m.get("n") else "pos" if m["roi"] >= 0 else "neg"
clv = m.get("clv")
st.markdown(f"""<div class="kpis">
<div class="kpi"><div class="v">{m.get('n', 0)}</div><div class="l">Jugadas de valor resueltas</div></div>
<div class="kpi"><div class="v">{pct(m['hit_rate']) if m.get('n') else '—'}</div><div class="l">Acierto (esperado {pct(m['expected_hit']) if m.get('n') else '—'})</div></div>
<div class="kpi"><div class="v {roi_cls}">{pct(m['roi'], True) if m.get('n') else '—'}</div><div class="l">Rentabilidad por apuesta</div></div>
<div class="kpi"><div class="v {roi_cls}">{(num(m['profit']) + ' €') if m.get('n') else '—'}</div><div class="l">Beneficio con los stakes</div></div>
<div class="kpi"><div class="v {'' if clv is None else 'pos' if clv >= 0 else 'neg'}">{pct(clv, True)}</div><div class="l">Ventaja frente al cierre</div></div>
</div>""", unsafe_allow_html=True)

# ------------------------------------------------------------------ filtros
sport_opts = ["todos"] + [k for k in SPORTS if k in s.active_sports]
counts = {r["sport"]: r["n"] for r in d["matches"].to_dict("records")}
fmt_sport = lambda k: "Todos" if k == "todos" else f"{SPORTS[k]['icono']} {SPORTS[k]['nombre']} ({counts.get(k, 0)})"
c1, c2 = st.columns([3, 2])
with c1:
    sport = st.segmented_control("Deporte", sport_opts, default="todos", format_func=fmt_sport,
                                 label_visibility="collapsed") or "todos"
with c2:
    risks = st.segmented_control("Riesgo", ["bajo", "medio", "alto"], selection_mode="multi",
                                 default=["bajo", "medio", "alto"], format_func=lambda r: RISK_NAME[r],
                                 label_visibility="collapsed") or ["bajo", "medio", "alto"]


def filt(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df if sport == "todos" else df[df["sport"] == sport]
    return out[out["risk"].isin(risks)] if "risk" in out else out


# ------------------------------------------------------------------ componentes
def ticket(r: dict) -> str:
    sp = SPORTS.get(r["sport"], SPORTS["football"])
    p = float(r["p_final"])
    odds = float(r["odds"])
    implied = 1 / odds
    tier = r["tier"]
    elite = tier == "valor" and r.get("elite") == 1
    reasons = "".join(f"<li>{e(x)}</li>" for x in str(r["reasons"]).split(" | "))
    if tier == "valor":
        stub = (f'<div class="bk">bet365</div><div class="od">{num(odds)}</div>'
                f'<div class="ev pos">{pct(r["ev"], True)}</div><div class="sk">stake {num(r["stake"])} €</div>')
    elif tier == "verificar":
        stub = (f'<div class="bk">mínimo en bet365</div><div class="od">{num(r["min_odds"])}</div>'
                f'<div class="sk">otras casas {num(odds)}</div>')
    else:
        lab = "bet365" if r["verified"] else "casas similares"
        cls = "pos" if r["ev"] >= 0 else "neg"
        stub = (f'<div class="bk">{lab}</div><div class="od muted">{num(odds)}</div>'
                f'<div class="ev {cls}">{pct(r["ev"], True)}</div>')
    badge = '<span class="badge">enviada a Telegram</span>' if r.get("alerted") else \
        '<span class="badge">élite</span>' if elite else ""
    return (f'<div class="tk r-{r["risk"]}{" elite" if elite else ""}"><div class="tk-main">'
            f'<div class="meta"><span>{sp["icono"]} {league_flag(r["league"])} {e(league_name(r["league"]))}</span>'
            f'<span>{when(r["kickoff"])}</span><span class="rk">{RISK_NAME[r["risk"]]}</span></div>'
            f'<div class="match">{e(r["home"])} – {e(r["away"])}</div>'
            f'<div class="sel">{e(r["label"])}{badge}</div>'
            f'<div class="meter" title="Barra: probabilidad estimada. Marca dorada: probabilidad que paga la cuota">'
            f'<div class="fill" style="width:{min(p, 1) * 100:.1f}%"></div>'
            f'<div class="tick" style="left:calc({min(implied, 1) * 100:.1f}% - 1px)"></div></div>'
            f'<div class="mlab"><span>Probabilidad <b>{pct(p)}</b></span><span>la cuota paga {pct(implied)} · '
            f'justa {num(1 / p)}</span></div>'
            f'<details><summary>Por qué · confianza {e(r["confidence"])}</summary><ul>{reasons}</ul></details>'
            f'</div><div class="stub">{stub}</div></div>')


def section(title: str, df: pd.DataFrame, hint: str, empty: str, limit: int = 60):
    st.markdown(f'<div class="sec"><h3>{title}</h3><span class="n">{len(df)}</span></div>'
                f'<p class="hint">{hint}</p>', unsafe_allow_html=True)
    if df.empty:
        st.markdown(f'<div class="empty">{empty}</div>', unsafe_allow_html=True)
        return
    cards = "".join(ticket(r) for r in df.head(limit).to_dict("records"))
    st.markdown(f'<div class="grid">{cards}</div>', unsafe_allow_html=True)
    if len(df) > limit:
        st.caption(f"Se muestran las {limit} mejores de {len(df)}.")


def combo_card(r: dict) -> str:
    legs = json.loads(r["legs"])
    risk = COMBO_RISK[r["risk"]]
    items = "".join(
        f'<li><span>{SPORTS.get(l.get("sport", "football"), SPORTS["football"])["icono"]} {e(l["label"])}'
        f'<br><span class="lp">{e(l["partido"])} · {when(l["kickoff"])} · {pct(l["p"])}</span></span>'
        f'<span class="lo">{num(l["price"])}</span></li>' for l in legs)
    cls = "pos" if r["ev"] >= 0 else "neg"
    stake = f'<div class="sk">stake {num(r["stake"])} €</div>' if r["stake"] else ""
    return (f'<div class="tk r-{risk}"><div class="tk-main"><div class="meta"><span class="rk">'
            f'{COMBO_NAME[r["risk"]]}</span><span>{len(legs)} selecciones</span><span>prob. {pct(r["prob"])}</span></div>'
            f'<ul class="legs">{items}</ul></div><div class="stub"><div class="bk">bet365</div>'
            f'<div class="od{"" if r["tier"] == "valor" else " muted"}">{num(r["odds"])}</div>'
            f'<div class="ev {cls}">{pct(r["ev"], True)}</div>{stake}</div></div>')


def chart_layout(fig: go.Figure, h=320):
    fig.update_layout(height=h, margin=dict(l=8, r=8, t=10, b=8), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Barlow", color="#ECE8DF"),
                      xaxis=dict(gridcolor="#2A3A57", zeroline=False), yaxis=dict(gridcolor="#2A3A57", zeroline=False),
                      legend=dict(orientation="h", y=-0.2))
    return fig


# ------------------------------------------------------------------ pestañas
t1, t2, t3, t4 = st.tabs(["Jugadas", "Combinadas", "Rendimiento", "Cómo decide"])

with t1:
    F = filt(P)
    tiers = {k: F[F["tier"] == k] if not F.empty else F for k in ("valor", "verificar", "pronostico")}
    section("Con valor en bet365", tiers["valor"],
            "La cuota de bet365 está por encima de su precio justo. Las marcadas como élite son las que llegan a Telegram.",
            "Ahora mismo ninguna cuota de bet365 supera su precio justo con este filtro. Es lo habitual en las grandes "
            "ligas: bet365 rara vez se equivoca, y cuando lo hace aparece aquí.")
    section("Pendientes de confirmar en bet365", tiers["verificar"],
            "El mercado sugiere valor, pero aún no tenemos la cuota de bet365. Apuesta solo si bet365 paga la cuota mínima "
            "del talón o más.",
            "Nada pendiente de confirmar.")
    section("Pronósticos del día", tiers["pronostico"].sort_values("kickoff") if not tiers["pronostico"].empty else
            tiers["pronostico"],
            "La opción más probable o mejor pagada de cada partido. Sin ventaja demostrada: el margen de la casa se come "
            "el valor. Útiles para informarte o para combinadas pequeñas.",
            "Sin pronósticos para este filtro.", limit=40)

with t2:
    C = d["combos"]
    if not C.empty and sport != "todos":
        C = C[C["sports"].fillna("football").str.contains(sport)]
    if not C.empty:
        C = C[C["risk"].map(COMBO_RISK).isin(risks)]
    for tier, title, hint in (
            ("valor", "Combinadas con valor", "Todas las selecciones tienen valor en bet365 y la combinada también."),
            ("pronostico", "Combinadas de pronósticos",
             "Las selecciones más sólidas del día. El valor esperado suele ser algo negativo: apuesta poco.")):
        sub = C[C["tier"] == tier] if not C.empty else C
        st.markdown(f'<div class="sec"><h3>{title}</h3><span class="n">{len(sub)}</span></div>'
                    f'<p class="hint">{hint}</p>', unsafe_allow_html=True)
        if sub.empty:
            st.markdown('<div class="empty">No hay combinadas de este tipo para las próximas 36 horas.</div>',
                        unsafe_allow_html=True)
            continue
        for risk in ("conservadora", "equilibrada", "arriesgada"):
            g = sub[sub["risk"] == risk]
            if not g.empty:
                st.markdown(f'<div class="grid">{"".join(combo_card(r) for r in g.to_dict("records"))}</div>'
                            '<div style="height:14px"></div>', unsafe_allow_html=True)

with t3:
    done = d["done"]
    if done.empty:
        st.markdown('<div class="empty">Cuando se jueguen los primeros partidos con jugadas de valor verás aquí la '
                    'curva de beneficio, el acierto por deporte y por riesgo.</div>', unsafe_allow_html=True)
    else:
        done = done.copy()
        done["u"] = (done["odds"] - 1) * done["won"] - (1 - done["won"])
        done["acum"] = done["u"].cumsum()
        fig = go.Figure(go.Scatter(x=pd.to_datetime(done["kickoff"]), y=done["acum"], mode="lines",
                                   line=dict(color="#F0B43C", width=3, shape="hv"), fill="tozeroy",
                                   fillcolor="rgba(240,180,60,.08)", name="Unidades"))
        fig.add_hline(y=0, line_color="#93A0B8", line_width=1)
        st.markdown('<div class="sec"><h3>Unidades ganadas</h3><span class="n">apostando 1 unidad por jugada</span></div>',
                    unsafe_allow_html=True)
        st.plotly_chart(chart_layout(fig), width="stretch", config={"displayModeBar": False})
        cols = st.columns(3)
        for col, key, title, names in (
                (cols[0], "by_sport", "Por deporte", {k: SPORTS[k]["nombre"] for k in SPORTS}),
                (cols[1], "by_risk", "Por riesgo", RISK_NAME),
                (cols[2], "by_confidence", "Por confianza", {"alta": "Alta", "media": "Media", "baja": "Baja"})):
            if m.get(key):
                t = pd.DataFrame(m[key]).T.rename(index=names)
                t.columns = ["Jugadas", "Acierto", "Rentabilidad"]
                col.markdown(f"**{title}**")
                col.dataframe(t.style.format({"Acierto": "{:.0%}", "Rentabilidad": "{:+.1%}"}), width="stretch")
    pdn = d["pron_done"]
    if not pdn.empty:
        st.markdown(f'<div class="sec"><h3>Pronósticos</h3><span class="n">{len(pdn)} resueltos</span></div>'
                    f'<p class="hint">Acierto real {pct(pdn["won"].mean())} frente al {pct(pdn["p_final"].mean())} '
                    f'esperado. Si se parecen, las probabilidades del radar están bien calibradas.</p>',
                    unsafe_allow_html=True)

with t4:
    st.markdown('<div class="sec"><h3>Cómo decide el radar</h3></div><p class="hint">Para cada selección estima la '
                'probabilidad real mezclando su modelo con el consenso de muchas casas sin su margen (Pinnacle pesa doble). '
                'Si la cuota de bet365 paga más que esa probabilidad, hay valor. El peso del modelo lo decide lo que '
                'habría acertado en el pasado y se ajusta con cada partido.</p>', unsafe_allow_html=True)
    w = d["weights"] or {}
    if w:
        MK = {"1x2": "1X2", "ou": "Más/menos", "ml": "Ganador", "spread": "Hándicap", "btts": "Ambos marcan"}
        rows = [{"Deporte": SPORTS.get(k.split("|")[0], {}).get("nombre", k), "Mercado": MK.get(k.split("|")[1], k),
                 "Peso del modelo": v} for k, v in sorted(w.items())]
        st.dataframe(pd.DataFrame(rows).style.format({"Peso del modelo": "{:.0%}"}), hide_index=True, width="stretch")
    for key, title in (("backtest", "Fútbol · último año"), ("backtest_tennis", "Tenis · último año")):
        bt = d.get(key)
        if not bt:
            continue
        st.markdown(f'<div class="sec"><h3>{title}</h3><span class="n">{bt["desde"]} a {bt["hasta"]}</span></div>',
                    unsafe_allow_html=True)
        strat = bt.get("estrategia_bet365") or {}
        if strat:
            t = pd.DataFrame(strat).T
            t.index = [f"Valor ≥ {k.split('_')[1]} %" for k in t.index]
            t["Intervalo"] = t["roi_ic95"].map(lambda x: f"{x[0]:+.0%} a {x[1]:+.0%}")
            t = t[["apuestas", "acierto", "cuota_media", "roi", "Intervalo"]]
            t.columns = ["Apuestas", "Acierto", "Cuota media", "Rentabilidad", "Intervalo 95 %"]
            st.dataframe(t.style.format({"Acierto": "{:.0%}", "Cuota media": "{:.2f}", "Rentabilidad": "{:+.1%}"}),
                         width="stretch")
            st.caption("Si el intervalo incluye el 0 todavía no hay pruebas de ventaja: puede ser suerte.")
        else:
            st.markdown('<div class="empty">En el último año casi ninguna cuota de bet365 superó el precio justo con '
                        'este filtro.</div>', unsafe_allow_html=True)
    st.markdown(f'<p class="hint" style="margin-top:1.2rem">Consumo del mes · The Odds API: '
                f'{int(d["credits"]) if d["credits"] is not None else "—"} créditos restantes · '
                f'OddsPapi: {d["oddspapi"]} de {s.oddspapi_monthly} consultas</p>', unsafe_allow_html=True)
