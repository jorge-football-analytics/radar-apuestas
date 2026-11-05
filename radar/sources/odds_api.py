"""The Odds API (v4). Plan gratis: 500 créditos/mes; cada llamada a /odds cuesta mercados × regiones.
El endpoint /events no gasta créditos, así que lo usamos para saber qué ligas tienen partidos pronto."""
from __future__ import annotations

import calendar
import logging
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

from ..config import LEAGUES, Settings
from ..db import DB

log = logging.getLogger(__name__)
BASE = "https://api.the-odds-api.com/v4"


class OddsAPI:
    def __init__(self, s: Settings, db: DB):
        self.s, self.db = s, db

    @property
    def enabled(self) -> bool:
        return bool(self.s.odds_api_key)

    def _cost(self) -> int:
        n_markets = len([m for m in self.s.odds_markets.split(",") if m.strip()])
        if self.s.odds_bookmakers:
            n_regions = max(1, -(-len(self.s.odds_bookmakers.split(",")) // 10))
        else:
            n_regions = len(self.s.odds_regions.split(","))
        return n_markets * n_regions

    def _update_budget(self, r: requests.Response) -> None:
        rem = r.headers.get("x-requests-remaining")
        if rem is not None:
            self.db.set("odds_api_remaining", float(rem))

    def allowance_now(self, now: datetime) -> int:
        """Créditos que podemos gastar en esta ejecución sin quedarnos sin crédito antes de fin de mes."""
        today = now.strftime("%Y-%m-%d")
        spent = self.db.get(f"odds_spent_{today}", 0)
        # créditos que había al empezar el día (los de hoy ya están descontados del restante)
        remaining = self.db.get("odds_api_remaining", 500) + spent - self.s.odds_reserve
        days_in_month = calendar.monthrange(now.year, now.month)[1]
        days_left = days_in_month - now.day + 1
        per_day = remaining / max(days_left, 1)
        return int(max(0, per_day - spent))

    def upcoming_events(self, sport: str, now: datetime) -> list[dict]:
        try:
            r = requests.get(f"{BASE}/sports/{sport}/events", params={"apiKey": self.s.odds_api_key}, timeout=20)
            r.raise_for_status()
        except requests.RequestException as e:
            log.warning("events %s: %s", sport, e)
            return []
        limit = now + timedelta(hours=self.s.odds_lookahead_h)
        out = []
        for ev in r.json():
            ko = datetime.fromisoformat(ev["commence_time"].replace("Z", "+00:00"))
            if now < ko <= limit:
                out.append(ev)
        return out

    def fetch(self, now: datetime | None = None) -> pd.DataFrame:
        """Descarga cuotas de las ligas con partidos próximos, respetando el presupuesto de créditos."""
        if not self.enabled:
            return pd.DataFrame()
        now = now or datetime.now(timezone.utc)
        budget = self.allowance_now(now)
        cost = self._cost()
        rows: list[dict] = []
        # ligas ordenadas por cercanía del próximo partido
        candidates = []
        for key, lg in LEAGUES.items():
            evs = self.upcoming_events(lg["odds"], now)
            if evs:
                candidates.append((min(e["commence_time"] for e in evs), key, lg))
        candidates.sort()
        spent = 0
        for _, key, lg in candidates:
            if spent + cost > budget:
                log.info("Presupuesto de The Odds API agotado por hoy (gastado %s/%s)", spent, budget)
                break
            params = {"apiKey": self.s.odds_api_key, "markets": self.s.odds_markets, "oddsFormat": "decimal"}
            if self.s.odds_bookmakers:
                params["bookmakers"] = self.s.odds_bookmakers
            else:
                params["regions"] = self.s.odds_regions
            try:
                r = requests.get(f"{BASE}/sports/{lg['odds']}/odds", params=params, timeout=30)
                self._update_budget(r)
                r.raise_for_status()
            except requests.RequestException as e:
                log.warning("odds %s: %s", key, e)
                continue
            spent += cost
            rows += parse_odds(r.json(), key)
        today = now.strftime("%Y-%m-%d")
        self.db.set(f"odds_spent_{today}", self.db.get(f"odds_spent_{today}", 0) + spent)
        return pd.DataFrame(rows)

    def fetch_scores(self, league_key: str) -> list[dict]:
        """Resultados recientes (cuesta 2 créditos). Solo se usa para la Champions, que no está en football-data."""
        if not self.enabled:
            return []
        try:
            r = requests.get(f"{BASE}/sports/{LEAGUES[league_key]['odds']}/scores",
                             params={"apiKey": self.s.odds_api_key, "daysFrom": 3}, timeout=20)
            self._update_budget(r)
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            log.warning("scores %s: %s", league_key, e)
            return []


def parse_odds(events: list[dict], league_key: str) -> list[dict]:
    rows = []
    for ev in events:
        base = dict(league=league_key, kickoff=ev["commence_time"].replace("Z", "+00:00"),
                    home=ev["home_team"], away=ev["away_team"], source="the-odds-api")
        for bk in ev.get("bookmakers", []):
            for m in bk.get("markets", []):
                for o in m.get("outcomes", []):
                    price = o.get("price")
                    if not price or price <= 1:
                        continue
                    if m["key"] == "h2h":
                        sel = "H" if o["name"] == ev["home_team"] else "A" if o["name"] == ev["away_team"] else "D"
                        rows.append({**base, "bookmaker": bk["key"], "market": "1x2", "selection": sel,
                                     "point": 0.0, "price": float(price)})
                    elif m["key"] == "totals" and o.get("point") is not None:
                        rows.append({**base, "bookmaker": bk["key"], "market": "ou",
                                     "selection": "O" if o["name"].lower() == "over" else "U",
                                     "point": float(o["point"]), "price": float(price)})
                    elif m["key"] == "btts":
                        rows.append({**base, "bookmaker": bk["key"], "market": "btts",
                                     "selection": "Y" if o["name"].lower() == "yes" else "N",
                                     "point": 0.0, "price": float(price)})
    return rows
