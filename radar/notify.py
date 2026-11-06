"""Telegram: solo lo mejor. Jugadas élite, combinadas con valor claro, un resumen diario y otro semanal."""
from __future__ import annotations

import html
import logging
from datetime import datetime
from zoneinfo import ZoneInfo

import requests

from .config import SPORTS, Settings, league_flag, league_name

log = logging.getLogger(__name__)
TZ = ZoneInfo("Europe/Madrid")
RISK_TXT = {"bajo": "🟢 Riesgo bajo", "medio": "🟡 Riesgo medio", "alto": "🔴 Riesgo alto"}
COMBO_ICON = {"conservadora": "🛡️", "equilibrada": "⚖️", "arriesgada": "🔥"}
DIAS = ["lun", "mar", "mié", "jue", "vie", "sáb", "dom"]


def _when(iso: str) -> str:
    d = datetime.fromisoformat(iso).astimezone(TZ)
    return f"{DIAS[d.weekday()]} {d:%d/%m %H:%M}"


def send(s: Settings, text: str) -> bool:
    if not (s.telegram_token and s.telegram_chat_id):
        log.info("Telegram no configurado. Mensaje:\n%s", text)
        return False
    try:
        r = requests.post(f"https://api.telegram.org/bot{s.telegram_token}/sendMessage",
                          json={"chat_id": s.telegram_chat_id, "text": text, "parse_mode": "HTML",
                                "disable_web_page_preview": True}, timeout=20)
        r.raise_for_status()
        return True
    except requests.RequestException as e:
        log.warning("Telegram: %s", e)
        return False


def format_pick(p: dict) -> str:
    e = html.escape
    sp = SPORTS[p["sport"]]
    return "\n".join([
        f"💎 <b>JUGADA ÉLITE</b> · {RISK_TXT[p['risk']]}",
        f"{sp['icono']} {e(p['home'])} - {e(p['away'])}",
        f"{league_flag(p['league'])} {e(league_name(p['league']))} · {_when(p['kickoff'])}",
        "",
        f"🎯 <b>{e(p['label'])}</b>",
        f"💰 bet365: <b>{p['odds']:.2f}</b>  (justa {1 / p['p_final']:.2f})",
        f"📊 Probabilidad {p['p_final']:.0%} · valor esperado <b>{p['ev']:+.1%}</b>",
        f"💶 Stake: {p['stake']:.2f} € · apuesta solo si la cuota sigue ≥ {p['min_odds']:.2f}",
        "",
        f"<i>{e(p['reasons'].replace(' | ', ' · '))}</i>",
    ])


def format_combo(c: dict) -> str:
    e = html.escape
    lines = [f"{COMBO_ICON.get(c['risk'], '')} <b>COMBINADA {c['risk'].upper()}</b> en bet365"]
    for l in c["legs"]:
        lines.append(f"{SPORTS[l['sport']]['icono']} {e(l['partido'])} ({_when(l['kickoff'])})\n"
                     f"   → {e(l['label'])} @ {l['price']:.2f} ({l['p']:.0%})")
    lines.append(f"\nCuota <b>{c['odds']:.2f}</b> · prob. {c['prob']:.0%} · EV <b>{c['ev']:+.1%}</b> · "
                 f"stake {c['stake']:.2f} €")
    return "\n".join(lines)


def format_digest(day_picks: list[dict], counts: dict, web_url: str = "") -> str:
    lines = ["☀️ <b>Radar del día</b>"]
    if day_picks:
        lines.append(f"Hoy hay {len(day_picks)} jugada(s) con valor en bet365:")
        for p in day_picks[:6]:
            lines.append(f"{SPORTS[p['sport']]['icono']} {html.escape(p['label'])} @ {p['odds']:.2f} "
                         f"(EV {p['ev']:+.1%}) · {_when(p['kickoff'])}")
    else:
        lines.append("Hoy ninguna cuota de bet365 supera el filtro de valor. No apostar también es una decisión.")
    if counts:
        lines.append("\nAnalizados: " + " · ".join(f"{SPORTS[k]['icono']} {v}" for k, v in counts.items()))
    if web_url:
        lines.append(f"\n👉 {web_url}")
    return "\n".join(lines)


def format_summary(stats: dict) -> str:
    if not stats or not stats.get("n"):
        return ""
    return (f"📈 <b>Resumen semanal</b>\n{stats['n']} jugadas de valor liquidadas · acierto {stats['hit_rate']:.0%} · "
            f"ROI {stats['roi']:+.1%} · CLV medio {(stats.get('clv') or 0):+.1%}")
