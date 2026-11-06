"""OddsPapi: la única fuente gratuita con cuotas de bet365 para tenis, NBA, NFL y copas europeas.
Plan gratis: 250 peticiones/mes → solo se usa para CONFIRMAR la cuota de bet365 de los mejores candidatos.

Estructura (v4): /tournaments?sportId → /fixtures?tournamentId&from&to → /odds?fixtureId&bookmakers=bet365
respuesta de /odds: bookmakerOdds[bookmaker].markets[marketId].outcomes[outcomeId].players["0"].price
Catálogo de mercados: /markets?sportId (marketType, period, handicap, outcomes[outcomeId, outcomeName]).
Todo es defensivo: si la API cambia o falla, el radar sigue funcionando sin esta fuente."""
from __future__ import annotations

import calendar
import logging
import time
from datetime import datetime, timedelta

import requests

from ..config import FOOTBALL, OTHER, Settings, sport_of
from ..db import DB
from ..teams import match_team, norm, player_key, us_key

log = logging.getLogger(__name__)
BASE = "https://api.oddspapi.io/v4"
SPORT_ID = {"football": 10, "basketball": 11, "tennis": 12, "nfl": 14}


class OddsPapi:
    def __init__(self, s: Settings, db: DB):
        self.s, self.db = s, db
        self._last_call = 0.0

    @property
    def enabled(self) -> bool:
        return bool(self.s.oddspapi_key)

    # ------------------------------------------------------------ presupuesto
    def _month(self, now: datetime) -> str:
        return now.strftime("%Y-%m")

    def allowance(self, now: datetime) -> int:
        used = self.db.get(f"oddspapi_used_{self._month(now)}", 0)
        today = self.db.get(f"oddspapi_today_{now:%Y-%m-%d}", 0)
        left = self.s.oddspapi_monthly - self.s.oddspapi_reserve - used + today
        days_left = calendar.monthrange(now.year, now.month)[1] - now.day + 1
        return int(max(0, left / max(days_left, 1) - today))

    def _get(self, path: str, now: datetime, **params):
        wait = 1.1 - (time.time() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        params["apiKey"] = self.s.oddspapi_key
        r = requests.get(f"{BASE}{path}", params=params, timeout=30)
        self._last_call = time.time()
        for k in (f"oddspapi_used_{self._month(now)}", f"oddspapi_today_{now:%Y-%m-%d}"):
            self.db.set(k, self.db.get(k, 0) + 1)
        if r.status_code == 404:
            return []
        r.raise_for_status()
        return r.json()

    def _cached(self, key: str, max_age_h: float, now: datetime, path: str, **params):
        c = self.db.get(key)
        if c and (now - datetime.fromisoformat(c["ts"])).total_seconds() < max_age_h * 3600:
            return c["data"]
        data = self._get(path, now, **params)
        self.db.set(key, {"ts": now.isoformat(), "data": data})
        return data

    # ------------------------------------------------------------ mercados
    def market_map(self, sport: str, now: datetime) -> dict[str, tuple[str, str, float]]:
        """outcomeId → (mercado, selección, línea)."""
        data = self._cached(f"op_markets_{sport}", 24 * 30, now, "/markets", sportId=SPORT_ID[sport])
        out = {}
        for m in data if isinstance(data, list) else []:
            if m.get("playerProp"):
                continue
            mtype = str(m.get("marketType", "")).lower()
            period = str(m.get("period", "")).lower()
            outs = m.get("outcomes", [])
            hcp = float(m.get("handicap") or 0)
            want_period = "fulltime" if sport == "football" else "result"
            if period != want_period:
                continue
            if mtype == "1x2" and sport == "football" and len(outs) == 3:
                for o, sel in zip(outs, "HDA"):
                    out[str(o["outcomeId"])] = ("1x2", sel, 0.0)
            elif mtype == "moneyline" and sport != "football" and len(outs) == 2:
                for o, sel in zip(outs, "HA"):
                    out[str(o["outcomeId"])] = ("ml", sel, 0.0)
            elif "total" in mtype and len(outs) == 2 and hcp:
                for o in outs:
                    name = str(o.get("outcomeName", "")).lower()
                    if name.startswith("o"):
                        out[str(o["outcomeId"])] = ("ou", "O", hcp)
                    elif name.startswith("u"):
                        out[str(o["outcomeId"])] = ("ou", "U", hcp)
        return out

    # ------------------------------------------------------------ fixtures
    def _tournament_ids(self, league: str, sport: str, now: datetime) -> list[int]:
        tours = self._cached(f"op_tournaments_{sport}", 24, now, "/tournaments", sportId=SPORT_ID[sport])
        if not isinstance(tours, list):
            return []
        if sport == "tennis":
            parts = league.split("_")
            tour = parts[1].upper()  # ATP / WTA
            words = [w for w in parts[2:] if len(w) > 2]
            want = lambda t: tour in str(t.get("categoryName", "")).upper() and any(
                w in norm(str(t.get("tournamentName", ""))) for w in words)
        else:
            kw = (FOOTBALL.get(league) or OTHER.get(league) or {}).get("oddspapi", league)
            want = lambda t: kw in norm(str(t.get("tournamentName", ""))) and "women" not in norm(
                str(t.get("tournamentName", "")))
        return [t["tournamentId"] for t in tours if want(t) and t.get("futureFixtures", 1)][:2]

    def _fixtures(self, league: str, sport: str, now: datetime) -> list[dict]:
        out = []
        for tid in self._tournament_ids(league, sport, now):
            data = self._cached(f"op_fix_{tid}_{now:%Y-%m-%d}", 12, now, "/fixtures", tournamentId=tid,
                                **{"from": now.strftime("%Y-%m-%d"),
                                   "to": (now + timedelta(days=3)).strftime("%Y-%m-%d")})
            out += data if isinstance(data, list) else []
        return out

    @staticmethod
    def _same(a: str, b: str, sport: str) -> bool:
        if sport == "tennis":
            return player_key(a) == player_key(b)
        if sport in ("basketball", "nfl"):
            return us_key(a) == us_key(b)
        return match_team(a, [b]) is not None

    def find_fixture(self, match: dict, now: datetime) -> tuple[str, bool] | None:
        """(fixtureId, invertido) del partido, o None."""
        sport = sport_of(match["league"])
        ko = datetime.fromisoformat(match["kickoff"])
        for f in self._fixtures(match["league"], sport, now):
            st = f.get("startTime")
            try:
                t = datetime.fromtimestamp(st, ko.tzinfo) if isinstance(st, (int, float)) else datetime.fromisoformat(
                    str(st).replace("Z", "+00:00"))
            except (ValueError, TypeError):
                continue
            if abs((t - ko).total_seconds()) > 6 * 3600:
                continue
            p1, p2 = str(f.get("participant1Name", "")), str(f.get("participant2Name", ""))
            if self._same(p1, match["home"], sport) and self._same(p2, match["away"], sport):
                return str(f["fixtureId"]), False
            if self._same(p2, match["home"], sport) and self._same(p1, match["away"], sport):
                return str(f["fixtureId"]), True
        return None

    # ------------------------------------------------------------ cuotas
    def bet365_odds(self, match: dict, now: datetime) -> list[dict]:
        """Filas de cuotas de bet365 (mismo formato que el resto de fuentes) para un partido."""
        if not self.enabled:
            return []
        sport = sport_of(match["league"])
        try:
            fx = self.find_fixture(match, now)
            if not fx:
                log.info("OddsPapi: no encuentro %s - %s", match["home"], match["away"])
                return []
            fid, swapped = fx
            mmap = self.market_map(sport, now)
            data = self._get("/odds", now, fixtureId=fid, bookmakers=self.s.book)
        except (requests.RequestException, ValueError, KeyError) as e:
            log.warning("OddsPapi: %s", e)
            return []
        book = (data.get("bookmakerOdds") or {}).get(self.s.book) if isinstance(data, dict) else None
        rows = []
        for mid, m in ((book or {}).get("markets") or {}).items():
            for oid, o in (m.get("outcomes") or {}).items():
                key = mmap.get(str(oid))
                pl = (o.get("players") or {}).get("0") or {}
                price = pl.get("price") if isinstance(pl, dict) else None
                if not key or not price or (isinstance(pl, dict) and pl.get("active") is False):
                    continue
                market, sel, pt = key
                if swapped and sel in "HA":
                    sel = "A" if sel == "H" else "H"
                rows.append(dict(sport=sport, league=match["league"], kickoff=match["kickoff"], home=match["home"],
                                 away=match["away"], source="oddspapi", bookmaker=self.s.book, market=market,
                                 selection=sel, point=pt, price=float(price)))
        return rows
