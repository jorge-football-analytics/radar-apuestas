"""Radar · panel web. Local: `streamlit run app.py` · Internet: Streamlit Community Cloud."""
from __future__ import annotations

import html
import json
from collections import Counter
from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from radar import assistant
from radar.config import FOOTBALL, OTHER, SPORTS, Settings, league_flag, league_name
from radar.db import DB
from radar.learn import metrics
from radar.model.markets import label as mlabel

TZ = ZoneInfo("Europe/Madrid")
DIAS = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
DIAS_C = ["lun", "mar", "mié", "jue", "vie", "sáb", "dom"]
RISK_NAME = {"bajo": "Riesgo bajo", "medio": "Riesgo medio", "alto": "Riesgo alto"}
COMBO_NAME = {"conservadora": "Conservadora", "equilibrada": "Equilibrada", "arriesgada": "Arriesgada"}
COMBO_RISK = {"conservadora": "bajo", "equilibrada": "medio", "arriesgada": "alto"}
MK = {"1x2": "Resultado", "ml": "Ganador", "ou": "Más/menos", "spread": "Hándicap", "btts": "Ambos marcan"}

st.set_page_config(page_title="Radar · bet365", page_icon="📡", layout="wide")
s = Settings()
e = html.escape

# ======================================================================= estilo
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow:wght@400;500;600&family=Barlow+Condensed:wght@500;600;700&display=swap');
:root{--ink:#0E1726;--panel:#172338;--panel2:#1D2B44;--line:#2A3A57;--text:#ECE8DF;--mute:#93A0B8;
--gold:#F0B43C;--lo:#3DB4A2;--mid:#E9A23B;--hi:#E4685A;--pos:#3DB4A2;--neg:#E4685A;--blue:#6C9BD2}
html,body,[data-testid="stAppViewContainer"],[data-testid="stHeader"]{background:var(--ink)!important;color:var(--text)}
[data-testid="stSidebar"]{background:#0B1320!important;border-right:1px solid var(--line)}
[data-testid="stSidebarNav"] span{font-family:'Barlow',system-ui,sans-serif;font-size:1rem}
.block-container{padding:2.6rem 1rem 4rem;max-width:1280px}
*{font-family:'Barlow',system-ui,sans-serif}
h1,h2,h3,.cond{font-family:'Barlow Condensed','Arial Narrow',sans-serif!important}
a{color:var(--gold)}
.pg{display:flex;align-items:flex-end;justify-content:space-between;gap:1rem;flex-wrap:wrap;margin-bottom:1rem}
.pg h1{font-size:2.5rem;font-weight:700;line-height:1;margin:0;padding:0}
.pg .sub{color:var(--mute);font-size:.95rem;margin-top:.35rem}
.pg .st{color:var(--mute);font-size:.88rem;text-align:right}
.pg .st b{color:var(--text)}
.dot{display:inline-block;width:.5rem;height:.5rem;border-radius:50%;background:var(--lo);margin-right:.35rem}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:1px;background:var(--line);
border:1px solid var(--line);border-radius:14px;overflow:hidden;margin:.3rem 0 1.2rem}
.kpi{background:var(--panel);padding:.85rem 1rem}
.kpi .v{font-family:'Barlow Condensed','Arial Narrow',sans-serif;font-weight:600;font-size:1.85rem;line-height:1.1}
.kpi .l{color:var(--mute);font-size:.84rem}
.pos{color:var(--pos)!important}.neg{color:var(--neg)!important}.mute{color:var(--mute)}
.sec{display:flex;align-items:baseline;gap:.7rem;margin:1.5rem 0 .6rem}
.sec h3{margin:0;font-size:1.5rem;font-weight:600;padding:0}
.sec .n{color:var(--mute);font-size:.95rem}
.hint{color:var(--mute);font-size:.92rem;margin:-.25rem 0 .8rem;max-width:75ch}
.empty{background:var(--panel);border:1px dashed var(--line);border-radius:12px;padding:1rem 1.2rem;color:var(--mute)}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:14px}
.grid2{display:grid;grid-template-columns:repeat(auto-fill,minmax(420px,1fr));gap:14px}
/* boletos */
.tk{display:grid;grid-template-columns:1fr 112px;background:var(--panel);border:1px solid var(--line);
border-radius:12px;overflow:hidden;position:relative}
.tk::before{content:"";position:absolute;left:0;top:0;bottom:0;width:4px;background:var(--rc)}
.r-bajo{--rc:var(--lo)}.r-medio{--rc:var(--mid)}.r-alto{--rc:var(--hi)}
.tk-main{padding:.85rem .9rem .8rem 1.1rem;min-width:0}
.meta{color:var(--mute);font-size:.82rem;display:flex;gap:.55rem;flex-wrap:wrap}
.meta .rk{color:var(--rc);font-weight:600}
.match{font-weight:600;font-size:1.02rem;margin:.25rem 0 .1rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.sel{font-family:'Barlow Condensed','Arial Narrow',sans-serif;font-weight:600;font-size:1.45rem;line-height:1.15}
.meter{position:relative;height:8px;background:var(--panel2);border-radius:4px;margin:.7rem 0 .35rem}
.meter .fill{position:absolute;left:0;top:0;bottom:0;border-radius:4px;background:var(--rc);opacity:.85}
.meter .tick{position:absolute;top:-4px;bottom:-4px;width:2px;background:var(--gold)}
.mlab{display:flex;justify-content:space-between;color:var(--mute);font-size:.78rem;gap:.5rem}
.mlab b{color:var(--text)}
details{margin-top:.55rem;font-size:.85rem;color:var(--mute)}
details summary{cursor:pointer;color:var(--text);opacity:.85;list-style:none}
details summary::-webkit-details-marker{display:none}
details summary::before{content:"＋ ";color:var(--gold)}
details[open] summary::before{content:"－ "}
details ul{margin:.4rem 0 0;padding-left:1.1rem}
.stub{background:var(--panel2);border-left:2px dashed var(--line);display:flex;flex-direction:column;
align-items:center;justify-content:center;padding:.6rem .4rem;text-align:center;position:relative}
.stub::before,.stub::after{content:"";position:absolute;left:-9px;width:16px;height:16px;border-radius:50%;background:var(--ink)}
.stub::before{top:-8px}.stub::after{bottom:-8px}
.stub .bk{font-size:.74rem;color:var(--mute)}
.stub .od{font-family:'Barlow Condensed','Arial Narrow',sans-serif;font-weight:700;font-size:2.25rem;line-height:1;color:var(--gold)}
.stub .od.muted{color:var(--text);opacity:.75}
.stub .ev{font-weight:600;font-size:.95rem;margin-top:.15rem}
.stub .sk{font-size:.76rem;color:var(--mute);margin-top:.25rem}
.badge{display:inline-block;font-size:.72rem;font-weight:600;padding:.1rem .45rem;border-radius:20px;
background:rgba(240,180,60,.14);color:var(--gold);margin-left:.3rem;vertical-align:middle}
.tk.elite{border-color:rgba(240,180,60,.6)}
.legs{margin:.45rem 0 0;padding:0;list-style:none}
.legs li{padding:.38rem 0;border-top:1px solid var(--line);font-size:.9rem;display:flex;justify-content:space-between;gap:.6rem}
.legs li:first-child{border-top:0}
.legs .lp{color:var(--mute);font-size:.8rem}
.legs .lo{font-family:'Barlow Condensed','Arial Narrow',sans-serif;font-weight:600;font-size:1.12rem}
/* fichas de partido */
.day{font-family:'Barlow Condensed','Arial Narrow',sans-serif;font-size:1.25rem;font-weight:600;color:var(--mute);
margin:1.2rem 0 .5rem}
.mc{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:.8rem .9rem}
.mc-h{display:flex;justify-content:space-between;gap:.6rem;color:var(--mute);font-size:.82rem;flex-wrap:wrap}
.mc-h .t{color:var(--text);font-weight:600}
.mc-teams{display:grid;grid-template-columns:1fr auto;gap:.1rem .8rem;margin:.45rem 0 .5rem;align-items:center}
.mc-teams .nm{font-weight:600;font-size:1.05rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.mc-teams .xs{color:var(--mute);font-size:.8rem;text-align:right;font-variant-numeric:tabular-nums}
.pb{display:flex;height:22px;border-radius:6px;overflow:hidden;font-size:.74rem;font-weight:600;margin:.25rem 0 .55rem}
.pb span{display:flex;align-items:center;justify-content:center;color:#0E1726;min-width:0;overflow:hidden}
.pb .h{background:#6C9BD2}.pb .d{background:#93A0B8}.pb .a{background:#C98BB9}
.oc{display:grid;gap:6px}
.oc3{grid-template-columns:repeat(3,1fr)}.oc2{grid-template-columns:repeat(2,1fr)}
.cell{background:var(--panel2);border:1px solid transparent;border-radius:8px;padding:.35rem .5rem;min-width:0}
.cell .k{color:var(--mute);font-size:.72rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.cell .o{font-family:'Barlow Condensed','Arial Narrow',sans-serif;font-weight:600;font-size:1.3rem;line-height:1.1}
.cell .f{color:var(--mute);font-size:.72rem}
.cell.val{border-color:rgba(61,180,162,.7);background:rgba(61,180,162,.10)}
.cell.val .o{color:var(--pos)}
.mv{font-size:.72rem;margin-left:.2rem}
.mv.up{color:var(--pos)}.mv.dn{color:var(--neg)}
.tagrow{display:flex;gap:.35rem;flex-wrap:wrap;margin-top:.5rem}
.tag{font-size:.74rem;color:var(--mute);background:var(--panel2);border-radius:20px;padding:.12rem .55rem}
.tag.g{color:var(--gold);background:rgba(240,180,60,.12)}
table.mk{width:100%;border-collapse:collapse;margin-top:.45rem;font-size:.82rem;color:var(--text)}
table.mk th{color:var(--mute);font-weight:500;text-align:right;padding:.25rem .3rem;border-bottom:1px solid var(--line)}
table.mk th:first-child,table.mk td:first-child{text-align:left}
table.mk td{padding:.25rem .3rem;text-align:right;border-bottom:1px solid rgba(42,58,87,.5);font-variant-numeric:tabular-nums}
table.mk tr.v td{color:var(--pos)}
/* clasificación */
table.tb{width:100%;border-collapse:collapse;font-size:.9rem;background:var(--panel);border-radius:12px;overflow:hidden}
table.tb th{color:var(--mute);font-weight:500;padding:.5rem .45rem;text-align:right;border-bottom:1px solid var(--line);
background:var(--panel2)}
table.tb td{padding:.42rem .45rem;text-align:right;border-bottom:1px solid rgba(42,58,87,.55);font-variant-numeric:tabular-nums}
table.tb th.l,table.tb td.l{text-align:left}
table.tb td.b{font-weight:600}
.fm{display:inline-flex;gap:2px}
.fm i{display:inline-block;width:16px;height:16px;border-radius:3px;font-style:normal;font-size:.62rem;line-height:16px;
text-align:center;color:#0E1726;font-weight:700}
.fm .V{background:var(--lo)}.fm .E{background:#93A0B8}.fm .D{background:var(--hi)}
.bar{display:inline-block;height:6px;border-radius:3px;background:var(--blue);vertical-align:middle}
.scroll{overflow-x:auto}
/* asistente */
.plan{background:var(--panel);border:1px solid rgba(240,180,60,.5);border-radius:14px;padding:1.1rem 1.2rem}
.plan h4{font-family:'Barlow Condensed','Arial Narrow',sans-serif;font-size:1.6rem;margin:0 0 .2rem;font-weight:600}
.plan .big{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:10px;margin:.8rem 0}
.plan .big div{background:var(--panel2);border-radius:10px;padding:.6rem .7rem}
.plan .big b{display:block;font-family:'Barlow Condensed','Arial Narrow',sans-serif;font-size:1.6rem;font-weight:600;line-height:1.1}
.plan .big span{color:var(--mute);font-size:.8rem}
.alt{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:.8rem 1rem;margin-top:.7rem}
/* streamlit */
[data-testid="stTabs"] button p{font-family:'Barlow Condensed','Arial Narrow',sans-serif;font-size:1.12rem;font-weight:600}
[data-testid="stTabs"] [aria-selected="true"] p{color:var(--gold)}
[data-baseweb="tab-highlight"]{background:var(--gold)!important}
[data-testid="stButtonGroup"]>div{flex-wrap:wrap;row-gap:6px}
button:focus-visible,summary:focus-visible{outline:2px solid var(--gold);outline-offset:2px}
@media (max-width:640px){.grid2{grid-template-columns:1fr}.pg h1{font-size:2.1rem}.pg .st{text-align:left}
.tk{grid-template-columns:1fr 92px}.stub .od{font-size:1.9rem}.kpi:last-child{grid-column:1/-1}}
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


def pct(x, signed=False, dec=0) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return (f"{x:+.{max(dec, 1)}%}" if signed else f"{x:.{dec}%}").replace(".", ",")


def num(x, d=2) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:.{d}f}".replace(".", ",")


def form_html(f: str) -> str:
    return '<span class="fm">' + "".join(f'<i class="{c}">{c}</i>' for c in str(f)) + "</span>"


def page_head(title: str, sub: str = ""):
    lr = D["last_run"]
    st.markdown(f'<div class="pg"><div><h1>{e(title)}</h1><div class="sub">{e(sub)}</div></div>'
                f'<div class="st"><span class="dot"></span>Último análisis <b>{when(lr) if lr else "—"}</b><br>'
                f'{len(D["value"])} jugada{"s" if len(D["value"]) != 1 else ""} con valor ahora mismo</div></div>',
                unsafe_allow_html=True)


def section(title: str, n=None, hint: str = ""):
    st.markdown(f'<div class="sec"><h3>{e(title)}</h3>{f"<span class=n>{n}</span>" if n is not None else ""}</div>'
                + (f'<p class="hint">{hint}</p>' if hint else ""), unsafe_allow_html=True)


def empty(msg: str):
    st.markdown(f'<div class="empty">{msg}</div>', unsafe_allow_html=True)


def chart_layout(fig: go.Figure, h=320):
    fig.update_layout(height=h, margin=dict(l=8, r=8, t=10, b=8), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Barlow", color="#ECE8DF"),
                      xaxis=dict(gridcolor="#2A3A57", zeroline=False), yaxis=dict(gridcolor="#2A3A57", zeroline=False),
                      legend=dict(orientation="h", y=-0.2), hoverlabel=dict(font_family="Barlow"))
    return fig


# ======================================================================= datos
@st.cache_data(ttl=300, show_spinner="Cargando análisis…")
def load(_mtime: float) -> dict:
    db = DB(s.db_path)
    now = datetime.now(timezone.utc)
    nowi = now.isoformat()
    d = {
        "picks": db.df("SELECT * FROM picks WHERE kickoff > ? AND bookmaker='bet365' ORDER BY score DESC", (nowi,)),
        "done": db.df("SELECT * FROM picks WHERE won IN (0,1) AND tier='valor' AND bookmaker='bet365' ORDER BY kickoff"),
        "pron_done": db.df("SELECT * FROM picks WHERE won IN (0,1) AND tier='pronostico' AND bookmaker='bet365'"),
        "combos": db.df("SELECT * FROM combos WHERE won IS NULL AND first_kickoff > ? AND bookmaker='bet365' "
                        "ORDER BY ev DESC", (nowi,)),
        "matches": db.df("SELECT * FROM matches WHERE status='pendiente' AND kickoff > ? ORDER BY kickoff", (nowi,)),
        "odds": db.df("SELECT o.* FROM odds o JOIN matches m ON m.id=o.match_id WHERE m.status='pendiente' AND m.kickoff > ?",
                      (nowi,)),
        "preds": db.df("SELECT p.* FROM predictions p JOIN matches m ON m.id=p.match_id WHERE m.status='pendiente' "
                       "AND m.kickoff > ?", (nowi,)),
        "info": {r["match_id"]: json.loads(r["info"]) for r in db.conn.execute(
            "SELECT i.match_id, i.info FROM match_info i JOIN matches m ON m.id=i.match_id WHERE m.kickoff > ?",
            (nowi,)).fetchall()},
        "pool": assistant.pool_from_db(db, now, s.book, s.min_ev),
        "calib": db.df("SELECT p.p_final, p.won, m.sport FROM predictions p JOIN matches m ON m.id=p.match_id "
                       "WHERE p.won IN (0,1) AND p.p_market IS NOT NULL"),
        "metrics": metrics(db), "backtest": db.get("backtest"), "backtest_tennis": db.get("backtest_tennis"),
        "weights": db.get("weights"), "last_run": db.get("last_run"), "credits": db.get("odds_api_remaining"),
        "fuentes": db.get("fuentes") or {},
        "runs": db.df("SELECT ts, detail FROM runs ORDER BY rowid DESC LIMIT 15"),
        "tables": {r["key"][6:]: json.loads(r["value"]) for r in db.conn.execute(
            "SELECT key, value FROM kv WHERE key LIKE 'tabla_%'").fetchall()},
    }
    db.close()
    p = d["picks"]
    d["value"] = p[p["tier"] == "valor"] if not p.empty else p
    return d


if not s.db_path.exists():
    st.markdown('<div class="empty">Aún no hay análisis. Lanza el workflow en GitHub (Actions → Run workflow).</div>',
                unsafe_allow_html=True)
    st.stop()
D = load(s.db_path.stat().st_mtime)


# ======================================================================= componentes
def ticket(r: dict) -> str:
    sp = SPORTS.get(r["sport"], SPORTS["football"])
    p = float(r["p_final"])
    odds = float(r["odds"])
    implied = 1 / odds if odds > 1 else 1
    tier = r["tier"]
    elite = tier == "valor" and r.get("elite") == 1
    reasons = "".join(f"<li>{e(x)}</li>" for x in str(r["reasons"]).split(" | "))
    if tier == "valor":
        stub = (f'<div class="bk">bet365</div><div class="od">{num(odds)}</div>'
                f'<div class="ev pos">{pct(r["ev"], True)}</div><div class="sk">stake {num(r["stake"])} €</div>')
    elif tier == "verificar" or (not r["verified"] and r["confidence"] == "modelo"):
        stub = (f'<div class="bk">mínimo en bet365</div><div class="od">{num(r["min_odds"])}</div>'
                f'<div class="sk">{"otras casas " + num(odds) if tier == "verificar" else "aún sin cuotas"}</div>')
    else:
        lab = "bet365" if r["verified"] else "otras casas"
        cls = "pos" if r["ev"] >= 0 else "neg"
        stub = (f'<div class="bk">{lab}</div><div class="od muted">{num(odds)}</div>'
                f'<div class="ev {cls}">{pct(r["ev"], True)}</div>')
    badge = '<span class="badge">enviada a Telegram</span>' if r.get("alerted") else \
        '<span class="badge">élite</span>' if elite else ""
    tick = "" if r["confidence"] == "modelo" else f'<div class="tick" style="left:calc({min(implied, 1) * 100:.1f}% - 1px)"></div>'
    right = f'justa {num(1 / p)}' if r["confidence"] == "modelo" else f'la cuota paga {pct(implied)} · justa {num(1 / p)}'
    return (f'<div class="tk r-{r["risk"]}{" elite" if elite else ""}"><div class="tk-main">'
            f'<div class="meta"><span>{sp["icono"]} {league_flag(r["league"])} {e(league_name(r["league"]))}</span>'
            f'<span>{when(r["kickoff"])}</span><span class="rk">{RISK_NAME[r["risk"]]}</span></div>'
            f'<div class="match">{e(r["home"])} – {e(r["away"])}</div>'
            f'<div class="sel">{e(r["label"])}{badge}</div>'
            f'<div class="meter"><div class="fill" style="width:{min(p, 1) * 100:.1f}%"></div>{tick}</div>'
            f'<div class="mlab"><span>Probabilidad <b>{pct(p)}</b></span><span>{right}</span></div>'
            f'<details><summary>Por qué · confianza {e(str(r["confidence"]))}</summary><ul>{reasons}</ul></details>'
            f'</div><div class="stub">{stub}</div></div>')


def tickets(df: pd.DataFrame, limit=60):
    if df.empty:
        return
    st.markdown(f'<div class="grid">{"".join(ticket(r) for r in df.head(limit).to_dict("records"))}</div>',
                unsafe_allow_html=True)
    if len(df) > limit:
        st.caption(f"Se muestran {limit} de {len(df)}.")


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


def match_card(m: dict, odds: pd.DataFrame, preds: pd.DataFrame, info: dict) -> str:
    sport = m["sport"]
    unit = SPORTS[sport]["unidad"]
    main = "1x2" if sport == "football" else "ml"
    sels = ["H", "D", "A"] if main == "1x2" else ["H", "A"]
    pm = preds[(preds["market"] == main)].set_index("selection") if not preds.empty else pd.DataFrame()
    b = odds[odds["bookmaker"] == s.book] if not odds.empty else odds
    probs = {x: float(pm.loc[x, "p_final"]) for x in sels if not pm.empty and x in pm.index}
    # cabecera y equipos
    tags = []
    xs_h = xs_a = ""
    if info.get("tipo") == "goles":
        xs_h, xs_a = f'xG {num(info["xg_local"], 2)}', f'xG {num(info["xg_visit"], 2)}'
        if info.get("marcadores"):
            tags.append(f'Marcador más probable {info["marcadores"][0][0]} ({pct(info["marcadores"][0][1])})')
        for side, key in (("local", "bajas_local"), ("visitante", "bajas_visit")):
            if info.get(key):
                tags.append(f'Bajas {side}: {len(info[key])}')
    elif info.get("tipo") in ("tenis", "puntos"):
        xs_h, xs_a = f'Elo {info["elo_local"]:.0f}', f'Elo {info["elo_visit"]:.0f}'
        if info.get("tipo") == "tenis":
            tags.append(f'{info["superficie"].capitalize()} · Elo superficie {info["elo_sup_local"]:.0f} – {info["elo_sup_visit"]:.0f}')
            if info["h2h_local"] + info["h2h_visit"]:
                tags.append(f'Cara a cara {info["h2h_local"]}-{info["h2h_visit"]}')
        else:
            tags.append(f'Margen esperado {info["margen"]:+.1f}')
            tags.append(f'Total esperado {info["total"]:.0f}')
    head = (f'<div class="mc-h"><span class="t">{loc(m["kickoff"]):%H:%M}</span>'
            f'<span>{league_flag(m["league"])} {e(league_name(m["league"]))}</span></div>'
            f'<div class="mc-teams"><span class="nm">{e(m["home"])}</span><span class="xs">{xs_h}</span>'
            f'<span class="nm">{e(m["away"])}</span><span class="xs">{xs_a}</span></div>')
    # barra de probabilidades
    bar = ""
    if probs:
        cls = {"H": "h", "D": "d", "A": "a"}
        bar = '<div class="pb">' + "".join(
            f'<span class="{cls[x]}" style="width:{probs[x] * 100:.1f}%">{pct(probs[x]) if probs[x] > .09 else ""}</span>'
            for x in sels if x in probs) + "</div>"
    # cuotas de bet365 del mercado principal
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
            cells.append(f'<div class="cell{" val" if ev is not None and ev >= s.min_ev else ""}"><div class="k">{e(names[x])}</div>'
                         f'<div class="o">{num(price)}{mv}</div><div class="f">justa {num(1 / p) if p else "—"} · '
                         f'{pct(ev, True) if ev is not None else ""}</div></div>')
        else:
            cells.append(f'<div class="cell"><div class="k">{e(names[x])}</div><div class="o mute">—</div>'
                         f'<div class="f">justa {num(1 / p) if p else "—"}</div></div>')
    grid = f'<div class="oc oc{len(sels)}">{"".join(cells)}</div>'
    # todos los mercados
    rows = []
    if not preds.empty:
        for r in preds.sort_values(["market", "point", "selection"]).itertuples():
            ob = b[(b["market"] == r.market) & (b["selection"] == r.selection) & (b["point"] == r.point)] if not b.empty else b
            om = odds[(odds["market"] == r.market) & (odds["selection"] == r.selection) & (odds["point"] == r.point)] \
                if not odds.empty else odds
            price = float(ob["price"].iloc[0]) if not ob.empty else None
            best = float(om["price"].max()) if not om.empty else None
            ev = r.p_final * price - 1 if price else None
            rows.append(f'<tr class="{"v" if ev is not None and ev >= s.min_ev else ""}"><td>'
                        f'{e(mlabel(r.market, r.selection, r.point, m["home"], m["away"], unit))}</td>'
                        f'<td>{pct(r.p_final)}</td><td>{num(1 / r.p_final) if r.p_final else "—"}</td>'
                        f'<td>{num(price) if price else "—"}</td><td>{pct(ev, True) if ev is not None else "—"}</td>'
                        f'<td>{num(best) if best else "—"}</td></tr>')
    table = (f'<details><summary>Todos los mercados ({len(rows)})</summary><table class="mk"><tr><th>Selección</th>'
             f'<th>Prob.</th><th>Justa</th><th>bet365</th><th>Valor</th><th>Mejor</th></tr>{"".join(rows)}</table>'
             f'</details>') if rows else ""
    n_books = odds["bookmaker"].nunique() if not odds.empty else 0
    tags.append(f"{n_books} casas comparadas" if n_books else "Aún sin cuotas")
    tg = '<div class="tagrow">' + "".join(f'<span class="tag">{e(t)}</span>' for t in tags) + "</div>"
    return f'<div class="mc">{head}{bar}{grid}{tg}{table}</div>'


def standings(league: str):
    t = D["tables"].get(league if not league.startswith("tennis_") else league)
    if not t:
        empty("Sin clasificación todavía para esta competición.")
        return
    sport = "football" if league in FOOTBALL else "tennis" if league.startswith("tennis_") else "us"
    if sport == "football":
        mx_a = max((r.get("ataque", 0) for r in t), default=1) or 1
        head = ('<tr><th>#</th><th class="l">Equipo</th><th>PJ</th><th>G</th><th>E</th><th>P</th><th>GF</th><th>GC</th>'
                '<th>DG</th><th>Pts</th><th class="l">Forma</th><th class="l">Ataque</th><th class="l">Defensa</th></tr>')
        body = "".join(
            f'<tr><td>{r["pos"]}</td><td class="l b">{e(r["equipo"])}</td><td>{r["pj"]}</td><td>{r["g"]}</td>'
            f'<td>{r["e"]}</td><td>{r["p"]}</td><td>{r["gf"]}</td><td>{r["gc"]}</td><td>{r["dg"]:+d}</td>'
            f'<td class="b">{r["pts"]}</td><td class="l">{form_html(r["forma"])}</td>'
            f'<td class="l"><span class="bar" style="width:{60 * r.get("ataque", 0) / mx_a:.0f}px"></span></td>'
            f'<td class="l"><span class="bar" style="background:#3DB4A2;width:{60 * min(r.get("defensa", 0), 2) / 2:.0f}px">'
            f'</span></td></tr>' for r in t)
    elif sport == "tennis":
        head = ('<tr><th>#</th><th class="l">Jugador</th><th>Elo</th><th>Dura</th><th>Tierra</th><th>Hierba</th>'
                '<th>G-P (12 m)</th><th class="l">Forma</th></tr>')
        body = "".join(
            f'<tr><td>{r["pos"]}</td><td class="l b">{e(r["jugador"])}</td><td class="b">{r["elo"]}</td><td>{r["dura"]}</td>'
            f'<td>{r["tierra"]}</td><td>{r["hierba"]}</td><td>{r["g"]}-{r["p"]}</td><td class="l">{form_html(r["forma"])}'
            f'</td></tr>' for r in t)
    else:
        head = ('<tr><th>#</th><th class="l">Equipo</th><th>Elo</th><th>G</th><th>P</th><th>%</th><th>Puntos a favor</th>'
                '<th>En contra</th><th class="l">Forma</th></tr>')
        body = "".join(
            f'<tr><td>{r["pos"]}</td><td class="l b">{e(r["equipo"])}</td><td class="b">{r["elo"]}</td><td>{r["g"]}</td>'
            f'<td>{r["p"]}</td><td>{pct(r["pct"])}</td><td>{num(r["pf"], 1)}</td><td>{num(r["pa"], 1)}</td>'
            f'<td class="l">{form_html(r["forma"])}</td></tr>' for r in t)
    st.markdown(f'<div class="scroll"><table class="tb">{head}{body}</table></div>', unsafe_allow_html=True)
    if sport == "football":
        st.caption("Ataque y defensa: fuerza estimada por el modelo (más barra = mejor). Forma: últimos 5 partidos.")


def kpis(m: dict):
    roi_cls = "" if not m.get("n") else "pos" if m["roi"] >= 0 else "neg"
    clv = m.get("clv")
    st.markdown(f"""<div class="kpis">
<div class="kpi"><div class="v">{m.get('n', 0)}</div><div class="l">Jugadas de valor resueltas</div></div>
<div class="kpi"><div class="v">{pct(m['hit_rate']) if m.get('n') else '—'}</div><div class="l">Acierto (esperado {pct(m['expected_hit']) if m.get('n') else '—'})</div></div>
<div class="kpi"><div class="v {roi_cls}">{pct(m['roi'], True) if m.get('n') else '—'}</div><div class="l">Rentabilidad por apuesta</div></div>
<div class="kpi"><div class="v {roi_cls}">{(num(m['profit']) + ' €') if m.get('n') else '—'}</div><div class="l">Beneficio con los stakes</div></div>
<div class="kpi"><div class="v {'' if clv is None else 'pos' if clv >= 0 else 'neg'}">{pct(clv, True)}</div><div class="l">Ventaja frente al cierre</div></div>
</div>""", unsafe_allow_html=True)


# ======================================================================= páginas
def page_home():
    page_head("Radar", "Cuotas de bet365 frente a la probabilidad real")
    kpis(D["metrics"])
    M = D["matches"]
    by_sport = Counter(M["sport"]) if not M.empty else Counter()
    val_sport = Counter(D["value"]["sport"]) if not D["value"].empty else Counter()
    cards = "".join(
        f'<div class="kpi"><div class="v">{SPORTS[k]["icono"]} {by_sport.get(k, 0)}</div>'
        f'<div class="l">{SPORTS[k]["nombre"]}: partidos · <b class="pos">{val_sport.get(k, 0)}</b> con valor</div></div>'
        for k in SPORTS if k in s.active_sports)
    st.markdown(f'<div class="kpis">{cards}</div>', unsafe_allow_html=True)

    v = D["value"]
    section("Las mejores ahora", len(v), "Cuotas de bet365 por encima de su precio justo, ordenadas por calidad. "
                                         "Las élite son las que llegan a Telegram.")
    if v.empty:
        empty("Ahora mismo ninguna cuota de bet365 supera su precio justo. Es lo habitual en las grandes ligas: "
              "cuando bet365 se equivoca, aparece aquí y en Telegram.")
    else:
        tickets(v, 9)

    P = D["picks"]
    ver = P[P["tier"] == "verificar"] if not P.empty else P
    if not ver.empty:
        section("Pendientes de confirmar", len(ver), "El mercado sugiere valor, pero falta la cuota de bet365: "
                                                     "apuesta solo si paga la cuota mínima del talón.")
        tickets(ver, 6)

    section("Empiezan pronto", hint="Los próximos partidos analizados con sus cuotas de bet365.")
    soon = M[pd.to_datetime(M["kickoff"], utc=True, format="ISO8601") < pd.Timestamp.now(tz="UTC") + pd.Timedelta(hours=12)] if not M.empty else M
    soon = soon[soon["id"].isin(set(D["odds"]["match_id"]) | set(D["info"]))] if not soon.empty else soon
    if soon.empty:
        empty("No hay partidos analizados en las próximas 12 horas.")
    else:
        render_matches(soon.head(8), group_days=False)


def render_matches(M: pd.DataFrame, group_days=True, limit=80):
    O, Pd, I = D["odds"], D["preds"], D["info"]
    oo = dict(tuple(O.groupby("match_id"))) if not O.empty else {}
    pp = dict(tuple(Pd.groupby("match_id"))) if not Pd.empty else {}
    M = M.head(limit)
    if not group_days:
        st.markdown('<div class="grid2">' + "".join(
            match_card(m, oo.get(m["id"], pd.DataFrame()), pp.get(m["id"], pd.DataFrame()), I.get(m["id"], {}))
            for m in M.to_dict("records")) + "</div>", unsafe_allow_html=True)
        return
    M = M.assign(_d=M["kickoff"].map(lambda k: loc(k).date()))
    for d, g in M.groupby("_d", sort=True):
        st.markdown(f'<div class="day">{day_title(d)} · {len(g)} partido{"s" if len(g) != 1 else ""}</div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="grid2">' + "".join(
            match_card(m, oo.get(m["id"], pd.DataFrame()), pp.get(m["id"], pd.DataFrame()), I.get(m["id"], {}))
            for m in g.to_dict("records")) + "</div>", unsafe_allow_html=True)


def sport_page(sport: str):
    meta = SPORTS[sport]
    M = D["matches"][D["matches"]["sport"] == sport] if not D["matches"].empty else D["matches"]
    P = D["picks"][D["picks"]["sport"] == sport] if not D["picks"].empty else D["picks"]
    page_head(f'{meta["icono"]} {meta["nombre"]}', "Por competición: partidos, jugadas y clasificación")
    if M.empty:
        empty(f"No hay partidos de {meta['nombre'].lower()} en los próximos días. Revisa la página Estado si crees que "
              "debería haberlos.")
        if sport == "tennis":
            section("Ranking Elo")
            t1, t2 = st.tabs(["ATP", "WTA"])
            with t1:
                standings("tennis_atp")
            with t2:
                standings("tennis_wta")
        else:
            for lg in [k for k, v in OTHER.items() if v["sport"] == sport]:
                if D["tables"].get(lg):
                    section(f"{league_name(lg)} · ranking Elo")
                    standings(lg)
        return
    counts = M["league"].value_counts()
    vals = P[P["tier"] == "valor"]["league"].value_counts() if not P.empty else pd.Series(dtype=int)
    order = sorted(counts.index, key=lambda k: (FOOTBALL.get(k, {}).get("tier", 1), -counts[k]))
    opts = ["todas"] + order
    fmt = lambda k: f"Todas ({len(M)})" if k == "todas" else \
        f"{league_flag(k)} {league_name(k)} ({counts[k]}{' · ' + str(vals[k]) + ' 💎' if k in vals else ''})"
    lg = st.pills("Competición", opts, default="todas", format_func=fmt, key=f"lg_{sport}",
                  label_visibility="collapsed") or "todas"
    Ml = M if lg == "todas" else M[M["league"] == lg]
    Pl = P if lg == "todas" else P[P["league"] == lg]
    risks = st.segmented_control("Riesgo", ["bajo", "medio", "alto"], selection_mode="multi",
                                 default=["bajo", "medio", "alto"], format_func=lambda r: RISK_NAME[r],
                                 key=f"rk_{sport}", label_visibility="collapsed") or ["bajo", "medio", "alto"]
    Pl = Pl[Pl["risk"].isin(risks)] if not Pl.empty else Pl
    t_j, t_p, t_c = st.tabs(["Jugadas", "Partidos", "Clasificación"])
    with t_j:
        for tier, title, hint, msg in (
                ("valor", "Con valor en bet365", "La cuota de bet365 supera su precio justo.",
                 "Ninguna cuota de bet365 supera su precio justo en esta selección."),
                ("verificar", "Pendientes de confirmar", "Apuesta solo si bet365 paga la cuota mínima del talón.", ""),
                ("pronostico", "Pronósticos", "Lo más probable de cada partido. Sin ventaja demostrada.",
                 "Sin pronósticos.")):
            sub = Pl[Pl["tier"] == tier] if not Pl.empty else Pl
            if tier == "verificar" and sub.empty:
                continue
            if tier == "pronostico" and not sub.empty:
                sub = sub.sort_values("kickoff")
            section(title, len(sub), hint)
            if sub.empty:
                empty(msg)
            else:
                tickets(sub, 45)
    with t_p:
        n = len(Ml)
        if n > 60:
            page = st.number_input(f"Página (60 partidos por página, {n} en total)", 1, (n - 1) // 60 + 1, 1,
                                   key=f"pg_{sport}_{lg}")
            Ml = Ml.iloc[(page - 1) * 60:]
        render_matches(Ml, limit=60)
    with t_c:
        if sport == "tennis":
            a, w = st.tabs(["Ranking ATP", "Ranking WTA"])
            with a:
                standings("tennis_atp")
            with w:
                standings("tennis_wta")
        elif lg == "todas":
            leagues_t = [k for k in order if D["tables"].get(k)]
            if not leagues_t:
                empty("Elige una competición para ver su clasificación.")
            else:
                pick = st.selectbox("Competición", leagues_t, format_func=lambda k: f"{league_flag(k)} {league_name(k)}",
                                    key=f"tb_{sport}")
                standings(pick)
        else:
            standings(lg)


def page_football():
    sport_page("football")


def page_tennis():
    sport_page("tennis")


def page_basket():
    sport_page("basketball")


def page_nfl():
    sport_page("nfl")


def page_assistant():
    page_head("Asistente de apuesta", "Dime cuánto y hasta cuándo: elijo la mejor opción entre todo lo analizado")
    now = datetime.now(timezone.utc)
    with st.form("asis"):
        c1, c2, c3 = st.columns(3)
        budget = c1.number_input("Presupuesto (€)", min_value=1.0, max_value=10000.0, value=10.0, step=1.0)
        dl_date = c2.date_input("Que se resuelva antes del día", value=(datetime.now(TZ) + timedelta(days=2)).date(),
                                min_value=datetime.now(TZ).date(), format="DD/MM/YYYY")
        dl_time = c3.time_input("a las", value=time(23, 59), step=timedelta(minutes=30))
        c4, c5 = st.columns([1, 1])
        risk = c4.segmented_control("Riesgo", ["bajo", "medio", "alto"], default="medio",
                                    format_func=lambda r: RISK_NAME[r]) or "medio"
        tipo = c5.segmented_control("Tipo", ["auto", "simple", "combinada", "repartir"], default="auto",
                                    format_func=lambda t: {"auto": "Lo mejor", "simple": "Simple",
                                                           "combinada": "Combinada", "repartir": "Repartir"}[t]) or "auto"
        c6, c7 = st.columns([2, 1])
        sports = c6.multiselect("Deportes", [k for k in SPORTS if k in s.active_sports],
                                default=[k for k in SPORTS if k in s.active_sports],
                                format_func=lambda k: f'{SPORTS[k]["icono"]} {SPORTS[k]["nombre"]}')
        only_value = c7.toggle("Solo con valor", value=False,
                               help="Solo selecciones donde bet365 paga más que el precio justo. Habrá menos opciones.")
        go_ = st.form_submit_button("Buscar la mejor apuesta", type="primary", use_container_width=True)
    if not go_:
        st.markdown('<p class="hint">Uso todas las selecciones con cuota de bet365 analizadas (no solo las de la portada) '
                    'y descarto las que no terminan antes de tu fecha límite. Riesgo bajo busca cuotas totales de 1,25 '
                    'a 2,2; medio de 2 a 4,5; alto de 4 a 15.</p>', unsafe_allow_html=True)
        return
    deadline = datetime.combine(dl_date, dl_time, TZ).astimezone(timezone.utc)
    plan, alts, n = assistant.recommend(D["pool"], budget, deadline, now, risk, tipo, sports, only_value)
    if plan is None:
        empty(f"No encuentro selecciones con cuota de bet365 que terminen antes del {dl_date:%d/%m} a las {dl_time:%H:%M}"
              f"{' y tengan valor' if only_value else ''}. Prueba a ampliar la fecha o quitar «Solo con valor».")
        return
    render_plan(plan, n, main=True)
    for a in alts:
        render_plan(a, n, main=False)
    st.caption("Las probabilidades son estimaciones. Comprueba la cuota en bet365 antes de apostar: si ha bajado, "
               "la recomendación puede dejar de tener sentido. Apuesta solo lo que puedas permitirte perder.")


def render_plan(p: assistant.Plan, n: int, main: bool):
    title = {"simple": "Apuesta simple", "combinada": f"Combinada de {len(p.legs)}",
             "repartir": f"Reparto en {len(p.legs)} simples"}[p.tipo]
    rows = []
    stakes = p.stakes if p.tipo == "repartir" else [p.budget] * len(p.legs)
    for st_, l in zip(stakes, p.legs):
        sp = SPORTS.get(l["sport"], SPORTS["football"])["icono"]
        stake = f' · apuesta {num(st_)} €' if p.tipo == "repartir" else ""
        rows.append(f'<li><span>{sp} <b>{e(l["label"])}</b><br><span class="lp">{e(l["home"])} – {e(l["away"])} · '
                    f'{when(l["kickoff"])} · prob. {pct(l["p_final"])} · valor {pct(l["ev"], True)}{stake}</span></span>'
                    f'<span class="lo">{num(l["odds"])}</span></li>')
    ev_cls = "pos" if p.ev >= 0 else "neg"
    if p.tipo == "repartir":
        big = (f'<div><b>{num(p.budget)} €</b><span>repartidos</span></div>'
               f'<div><b>{pct(p.prob)}</b><span>prob. de acertar al menos una</span></div>'
               f'<div><b class="{ev_cls}">{num(p.expected)} €</b><span>ganancia esperada</span></div>'
               f'<div><b>{num(p.win)} €</b><span>beneficio si aciertan todas</span></div>')
    else:
        big = (f'<div><b>{num(p.odds)}</b><span>cuota bet365</span></div>'
               f'<div><b>{pct(p.prob)}</b><span>probabilidad</span></div>'
               f'<div><b>{num(p.budget * p.odds)} €</b><span>cobras si aciertas</span></div>'
               f'<div><b class="{ev_cls}">{pct(p.ev, True)}</b><span>valor esperado ({num(p.expected)} €)</span></div>')
    html_ = (f'<div class="{"plan" if main else "alt"}"><h4>{"Recomendación · " if main else "Alternativa · "}{title}</h4>'
             f'<div class="mute" style="font-size:.85rem">Elegida entre {n} selecciones · se resuelve hacia el '
             f'{when(p.ends)}</div>{f"<div class=big>{big}</div>" if main else ""}'
             f'<ul class="legs">{"".join(rows)}</ul>'
             + ("" if main else f'<div class="mute" style="font-size:.85rem;margin-top:.4rem">'
                                f'{"cuota " + num(p.odds) + " · prob. " if p.tipo != "repartir" else "prob. de acertar al menos una "}{pct(p.prob)} · '
                                f'valor {pct(p.ev, True)}</div>') + "</div>")
    st.markdown(html_, unsafe_allow_html=True)
    if main and p.ev < 0:
        st.caption("Ninguna opción tiene valor positivo con este filtro: es la menos mala. El margen de bet365 hace que, "
                   "a la larga, este tipo de apuesta pierda dinero.")


def page_combos():
    page_head("Combinadas", "Una selección por partido, todas en bet365, por nivel de riesgo")
    C = D["combos"]
    for tier, title, hint in (
            ("valor", "Con valor", "Todas las selecciones tienen valor en bet365 y la combinada también."),
            ("pronostico", "De pronósticos", "Las selecciones más sólidas del día. El valor esperado suele ser algo "
                                             "negativo: apuesta poco.")):
        sub = C[C["tier"] == tier] if not C.empty else C
        section(title, len(sub), hint)
        if sub.empty:
            empty("No hay combinadas de este tipo para las próximas 36 horas. Prueba el Asistente para montar una "
                  "con tus condiciones.")
            continue
        for risk in ("conservadora", "equilibrada", "arriesgada"):
            g = sub[sub["risk"] == risk]
            if not g.empty:
                st.markdown(f'<div class="grid">{"".join(combo_card(r) for r in g.to_dict("records"))}</div>'
                            '<div style="height:14px"></div>', unsafe_allow_html=True)


def page_perf():
    page_head("Rendimiento", "Lo que ha pasado con las jugadas del radar")
    m = D["metrics"]
    kpis(m)
    done = D["done"]
    if done.empty:
        empty("Cuando se jueguen los primeros partidos con jugadas de valor verás aquí la curva de beneficio y el "
              "desglose por deporte, liga y riesgo.")
    else:
        done = done.copy()
        done["u"] = (done["odds"] - 1) * done["won"] - (1 - done["won"])
        done["acum"] = done["u"].cumsum()
        fig = go.Figure(go.Scatter(x=pd.to_datetime(done["kickoff"]), y=done["acum"], mode="lines",
                                   line=dict(color="#F0B43C", width=3, shape="hv"), fill="tozeroy",
                                   fillcolor="rgba(240,180,60,.08)", name="Unidades"))
        fig.add_hline(y=0, line_color="#93A0B8", line_width=1)
        section("Unidades ganadas", hint="Apostando 1 unidad por jugada.")
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
        lg = done.groupby("league").agg(Jugadas=("u", "size"), Acierto=("won", "mean"), Unidades=("u", "sum"))
        lg.index = [f"{league_flag(k)} {league_name(k)}" for k in lg.index]
        section("Por competición")
        st.dataframe(lg.sort_values("Unidades", ascending=False).style.format({"Acierto": "{:.0%}", "Unidades": "{:+.2f}"}),
                     width="stretch")
    cal = D["calib"]
    section("¿Acierta las probabilidades?", len(cal) if not cal.empty else 0,
            "Cada punto agrupa predicciones con probabilidad parecida. Cuanto más cerca de la diagonal, mejor calibrado.")
    if len(cal) < 50:
        empty("Hacen falta al menos 50 resultados para dibujar la calibración.")
    else:
        cal = cal.assign(b=pd.cut(cal["p_final"], np.linspace(0, 1, 11)))
        g = cal.groupby("b", observed=True).agg(p=("p_final", "mean"), real=("won", "mean"), n=("won", "size")).reset_index()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="#93A0B8", dash="dot"), name="Perfecto"))
        fig.add_trace(go.Scatter(x=g["p"], y=g["real"], mode="markers+lines", marker=dict(size=np.sqrt(g["n"]) * 1.6 + 6,
                                 color="#F0B43C"), line=dict(color="#F0B43C"), name="Radar",
                                 hovertemplate="Previsto %{x:.0%}<br>Real %{y:.0%}<extra></extra>"))
        fig.update_xaxes(tickformat=".0%", title="Probabilidad prevista")
        fig.update_yaxes(tickformat=".0%", title="Frecuencia real")
        st.plotly_chart(chart_layout(fig, 360), width="stretch", config={"displayModeBar": False})
    pdn = D["pron_done"]
    if not pdn.empty:
        section("Pronósticos", f"{len(pdn)} resueltos",
                f"Acierto real {pct(pdn['won'].mean())} frente al {pct(pdn['p_final'].mean())} esperado.")


def page_status():
    page_head("Estado y método", "De dónde salen los datos y cómo decide el radar")
    F = D["fuentes"]
    section("Fuentes de datos", hint=f"Última comprobación {when(F['ts'])}" if F.get("ts") else "")
    rows = []
    for k, v in F.items():
        if k == "ts":
            continue
        rows.append(f'<tr><td class="l b">{"🟢" if v.get("ok") else "🔴"} {e(k)}</td><td>{v.get("filas", 0):,}'
                    f'</td><td class="l mute">{e(str(v.get("detalle", "")))}</td></tr>'.replace(",", "."))
    if rows:
        st.markdown('<div class="scroll"><table class="tb"><tr><th class="l">Fuente</th><th>Registros</th>'
                    f'<th class="l">Detalle</th></tr>{"".join(rows)}</table></div>', unsafe_allow_html=True)
    else:
        empty("El estado de las fuentes aparecerá tras el próximo análisis.")
    section("Cómo decide", hint="Para cada selección estima la probabilidad real mezclando su modelo (Dixon-Coles en "
            "fútbol, Elo por superficie en tenis, Elo con puntos esperados en NBA y NFL) con el consenso de muchas casas "
            "sin su margen, con Pinnacle pesando doble. Si la cuota de bet365 paga más que esa probabilidad, hay valor. "
            "El peso del modelo lo fija el backtest y se reajusta con cada partido jugado.")
    w = D["weights"] or {}
    if w:
        rows = [{"Deporte": SPORTS.get(k.split("|")[0], {}).get("nombre", k), "Mercado": MK.get(k.split("|")[1], k),
                 "Peso del modelo": v} for k, v in sorted(w.items())]
        st.dataframe(pd.DataFrame(rows).style.format({"Peso del modelo": "{:.0%}"}), hide_index=True, width="stretch")
    for key, title in (("backtest", "Backtest de fútbol (último año)"), ("backtest_tennis", "Backtest de tenis (último año)")):
        bt = D.get(key)
        if not bt:
            continue
        section(title, f'{bt["desde"]} a {bt["hasta"]}')
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
            empty("En el último año casi ninguna cuota de bet365 superó el precio justo con este filtro.")
    section("Últimos análisis")
    R = D["runs"].copy()
    if not R.empty:
        import ast

        def resume(txt: str) -> dict:
            try:
                d = ast.literal_eval(txt)
            except (ValueError, SyntaxError):
                return {}
            dep = d.get("por_deporte") or {}
            return {"Partidos": d.get("partidos"),
                    "Por deporte": " ".join(f'{SPORTS[k]["icono"]} {v}' for k, v in dep.items() if k in SPORTS),
                    "Con valor": d.get("valor", d.get("picks_nuevos")), "Élite": d.get("elite"),
                    "Avisos": d.get("avisos"), "Liquidados": d.get("liquidados"),
                    "Créditos Odds API": d.get("creditos_odds_api")}
        T = pd.DataFrame([{"Hora": when(r.ts), **resume(r.detail)} for r in R.itertuples()])
        st.dataframe(T, hide_index=True, width="stretch")
    st.markdown('<p class="hint" style="margin-top:1.2rem">Juega solo con dinero que puedas perder. Si deja de ser '
                'divertido: <a href="https://www.jugarbien.es">jugarbien.es</a>.</p>', unsafe_allow_html=True)


# ======================================================================= navegación
pages = {
    "Radar": [st.Page(page_home, title="Inicio", icon="📡", default=True),
              st.Page(page_assistant, title="Asistente de apuesta", icon="🧮"),
              st.Page(page_combos, title="Combinadas", icon="🧩")],
    "Deportes": [st.Page(f, title=SPORTS[k]["nombre"], icon=SPORTS[k]["icono"], url_path=k)
                 for k, f in (("football", page_football), ("tennis", page_tennis), ("basketball", page_basket),
                              ("nfl", page_nfl)) if k in s.active_sports],
    "Seguimiento": [st.Page(page_perf, title="Rendimiento", icon="📈"),
                    st.Page(page_status, title="Estado y método", icon="🛰️")],
}
st.navigation(pages).run()
