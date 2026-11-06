"""The Odds API (v4): consenso del mercado (Pinnacle, Betfair, 1xBet, William Hill...). OJO: no incluye bet365 España.
Plan gratis: 500 créditos/mes; /odds cuesta mercados × regiones. /sports y /events no gastan créditos."""
from __future__ import annotations

import calendar
import logging
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

from ..config import FOOTBALL, ODDS_MARKETS, OTHER, Settings, sport_of
from ..db import DB

log = logging.getLogger(__name__)
BASE = "https://api.the-odds-api.com/v4"


class OddsAPI:
    def __init__(self, s: Settings, db: DB):
        self.s, self.db = s, db

    @property
    def enabled(self) -> bool:
        return bool(self.s.odds_api_key)

    def _get(self, path: str, **params):
        params["apiKey"] = self.s.odds_api_key
        r = requests.get(f"{BASE}{path}", params=params, timeout=30)
        rem = r.headers.get("x-requests-remaining")
        if rem is not None:
            self.db.set("odds_api_remaining", float(rem))
        r.raise_for_status()
        return r.json()

    def allowance_now(self, now: datetime) -> int:
        """Créditos que podemos gastar hoy sin quedarnos sin crédito antes de fin de mes."""
        today = now.strftime("%Y-%m-%d")
        spent = self.db.get(f"odds_spent_{today}", 0)
        remaining = self.db.get("odds_api_remaining", 500) + spent - self.s.odds_reserve
        days_left = calendar.monthrange(now.year, now.month)[1] - now.day + 1
        return int(max(0, remaining / max(days_left, 1) - spent))

    def plan(self) -> list[tuple[str, str, str]]:
        """(liga, clave de The Odds API, mercados) para todo lo que hay que seguir."""
        act = self.s.active_sports
        out = []
        if "football" in act:
            for k, v in FOOTBALL.items():
                if v["fd"] is None or self.s.odds_api_football:
                    out.append((k, v["odds"], ODDS_MARKETS["football"]))
        for k, v in OTHER.items():
            if v["sport"] in act:
                out.append((k, v["odds"], ODDS_MARKETS[v["sport"]]))
        if "tennis" in act:
            try:
                for sp in self._get("/sports"):  # gratis
                    key = sp["key"]
                    if sp.get("active") and (key.startswith("tennis_atp_") or key.startswith("tennis_wta_")):
                        out.append((key, key, ODDS_MARKETS["tennis"]))
            except requests.RequestException as e:
                log.warning("sports: %s", e)
        return out

    def upcoming_events(self, sport_key: str, now: datetime) -> list[dict]:
        try:
            evs = self._get(f"/sports/{sport_key}/events")
        except requests.RequestException as e:
            log.warning("events %s: %s", sport_key, e)
            return []
        limit = now + timedelta(hours=self.s.odds_lookahead_h)
        return [e for e in evs if now < datetime.fromisoformat(e["commence_time"].replace("Z", "+00:00")) <= limit]

    def fetch(self, now: datetime | None = None) -> pd.DataFrame:
        if not self.enabled:
            return pd.DataFrame()
        now = now or datetime.now(timezone.utc)
        budget = self.allowance_now(now)
        n_regions = len(self.s.odds_regions.split(","))
        cands = []
        for league, key, markets in self.plan():
            last = self.db.get(f"odds_refresh_{league}")
            if last and (now - datetime.fromisoformat(last)).total_seconds() < self.s.odds_min_refresh_h * 3600:
                continue
            evs = self.upcoming_events(key, now)
            if evs:
                cands.append((min(e["commence_time"] for e in evs), league, key, markets))
        cands.sort()
        rows, spent = [], 0
        for _, league, key, markets in cands:
            cost = len(markets.split(",")) * n_regions
            if spent + cost > budget:
                log.info("Créditos de The Odds API reservados para otros días (hoy %s/%s)", spent, budget)
                break
            try:
                data = self._get(f"/sports/{key}/odds", regions=self.s.odds_regions, markets=markets, oddsFormat="decimal")
            except requests.RequestException as e:
                log.warning("odds %s: %s", league, e)
                continue
            spent += cost
            rows += parse_odds(data, league)
            self.db.set(f"odds_refresh_{league}", now.isoformat())
        today = now.strftime("%Y-%m-%d")
        self.db.set(f"odds_spent_{today}", self.db.get(f"odds_spent_{today}", 0) + spent)
        return pd.DataFrame(rows)

    def fetch_scores(self, league: str) -> list[dict]:
        """Resultados de los últimos 3 días (2 créditos). Para ligas sin otra fuente de resultados."""
        if not self.enabled:
            return []
        key = FOOTBALL.get(league, {}).get("odds") or OTHER.get(league, {}).get("odds") or league
        try:
            return self._get(f"/sports/{key}/scores", daysFrom=3)
        except requests.RequestException as e:
            log.warning("scores %s: %s", league, e)
            return []


def parse_odds(events: list[dict], league: str) -> list[dict]:
    sport = sport_of(league)
    rows = []
    for ev in events:
        base = dict(sport=sport, league=league, kickoff=ev["commence_time"].replace("Z", "+00:00"),
                    home=ev["home_team"], away=ev["away_team"], source="the-odds-api")
        for bk in ev.get("bookmakers", []):
            for m in bk.get("markets", []):
                outs = m.get("outcomes", [])
                three_way = any(o["name"] == "Draw" for o in outs)
                for o in outs:
                    price = o.get("price")
                    if not price or price <= 1:
                        continue
                    side = "H" if o["name"] == ev["home_team"] else "A" if o["name"] == ev["away_team"] else "D"
                    row = {**base, "bookmaker": bk["key"], "price": float(price)}
                    if m["key"] == "h2h":
                        rows.append({**row, "market": "1x2" if three_way else "ml", "selection": side, "point": 0.0})
                    elif m["key"] == "spreads" and o.get("point") is not None and side in "HA":
                        pt = float(o["point"]) if side == "H" else -float(o["point"])  # siempre hándicap del local
                        rows.append({**row, "market": "spread", "selection": side, "point": pt})
                    elif m["key"] == "totals" and o.get("point") is not None:
                        rows.append({**row, "market": "ou", "selection": "O" if o["name"].lower() == "over" else "U",
                                     "point": float(o["point"])})
                    elif m["key"] == "btts":
                        rows.append({**row, "market": "btts", "selection": "Y" if o["name"].lower() == "yes" else "N",
                                     "point": 0.0})
    return rows
