"""Uso:
  python -m radar run            análisis completo + avisos (lo que ejecuta GitHub Actions cada 3 h)
  python -m radar backtest       backtest de fútbol y tenis del último año (fija los pesos iniciales)
  python -m radar status         estado: créditos, último análisis, rendimiento
  python -m radar telegram-test  envía un mensaje de prueba y muestra tu chat_id
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone

import requests

from .config import Settings
from .db import DB


def main(argv: list[str]) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cmd = argv[1] if len(argv) > 1 else "run"
    s = Settings()
    if cmd == "run":
        from .pipeline import run

        print(json.dumps(run(s), indent=2, ensure_ascii=False, default=str))
    elif cmd == "backtest":
        from .pipeline import load_context, run_backtest

        db = DB(s.db_path)
        ctx = load_context(s, db, datetime.now(timezone.utc))
        print(json.dumps(run_backtest(s, db, ctx), indent=2, ensure_ascii=False, default=str))
        db.set("backtest_date", datetime.now(timezone.utc).date().isoformat())
        db.close()
    elif cmd == "status":
        from .learn import metrics

        db = DB(s.db_path)
        print(json.dumps({"ultimo_analisis": db.get("last_run"), "creditos_odds_api": db.get("odds_api_remaining"),
                          "pesos_modelo": db.get("weights"), "rendimiento": metrics(db)},
                         indent=2, ensure_ascii=False, default=str))
        db.close()
    elif cmd == "telegram-test":
        if not s.telegram_token:
            print("Falta TELEGRAM_BOT_TOKEN en el .env")
            return 1
        r = requests.get(f"https://api.telegram.org/bot{s.telegram_token}/getUpdates", timeout=20).json()
        chats = {u["message"]["chat"]["id"]: u["message"]["chat"].get("first_name", "")
                 for u in r.get("result", []) if "message" in u}
        print("Chats que han escrito al bot (usa el id como TELEGRAM_CHAT_ID):", chats or
              "ninguno: escribe cualquier cosa a tu bot en Telegram y vuelve a ejecutar esto")
        if s.telegram_chat_id:
            from .notify import send

            print("Mensaje de prueba enviado" if send(s, "✅ Radar de apuestas conectado") else "No se pudo enviar")
    else:
        print(__doc__)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
