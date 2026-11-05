"""Avisos por Telegram (bot creado con @BotFather)."""
from __future__ import annotations

import html
import logging
from datetime import datetime
from zoneinfo import ZoneInfo

import requests

from .config import LEAGUES, Settings

log = logging.getLogger(__name__)
TZ = ZoneInfo("Europe/Madrid")
CONF_ICON = {"alta": "🟢", "media": "🟡", "baja": "🟠"}
RISK_ICON = {"conservadora": "🛡️", "equilibrada": "⚖️", "arriesgada": "🔥"}


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
    lines = [
        f"{CONF_ICON.get(p['confidence'], '')} <b>Apuesta de valor</b> · confianza {p['confidence']}",
        f"⚽ {e(p['home'])} - {e(p['away'])} ({LEAGUES[p['league']]['nombre']}, {_when(p['kickoff'])})",
        f"🎯 <b>{e(p['label'])}</b> @ <b>{p['odds']:.2f}</b> en {e(p['bookmaker'])}",
    ]
    if p.get("b365_odds"):
        lines.append(f"   bet365: {p['b365_odds']:.2f}" + (f" (EV {p['ev_b365']:+.1%})" if p.get("ev_b365") is not None else ""))
    lines += [
        f"📊 Prob. {p['p_final']:.0%} · valor esperado <b>{p['ev']:+.1%}</b>",
        f"💶 Stake sugerido: {p['stake']:.2f} €",
        f"<i>{e(p['reasons'])}</i>",
    ]
    return "\n".join(lines)


def format_combo(c: dict) -> str:
    e = html.escape
    lines = [f"{RISK_ICON.get(c['risk'], '')} <b>Combinada {c['risk']}</b> en {e(c['bookmaker'])}"]
    for l in c["legs"]:
        lines.append(f"• {e(l['partido'])} ({_when(l['kickoff'])}): {e(l['label'])} @ {l['price']:.2f} ({l['p']:.0%})")
    lines.append(f"Cuota total <b>{c['odds']:.2f}</b> · prob. {c['prob']:.0%} · EV <b>{c['ev']:+.1%}</b> · stake {c['stake']:.2f} €")
    return "\n".join(lines)


def format_summary(stats: dict) -> str:
    if not stats or not stats.get("n"):
        return ""
    return (f"📈 <b>Rendimiento acumulado</b>: {stats['n']} apuestas · acierto {stats['hit_rate']:.0%} · "
            f"ROI {stats['roi']:+.1%} · CLV medio {stats.get('clv', 0) or 0:+.1%}")
