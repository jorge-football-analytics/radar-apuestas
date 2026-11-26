"""OddsPapi: cuotas de bet365 (y de Pinnacle EN LA MISMA LLAMADA) para tenis, NBA, NFL, copas y fútbol sin bet365 fresco.
Plan gratis: 250 peticiones/mes → solo se usa para CONFIRMAR los mejores candidatos de las próximas 48 h.

Coste por ejecución (v2, sin listas de torneos):
  /fixtures?sportId&from&to&statusId=0&hasOdds=true&bookmakers=bet365   1 petición por deporte, cacheada 12 h.
      Ventana < 48 h (la API lo exige si se filtra solo por deporte y fechas). Trae nombres de participantes,
      torneo y país, así que NO hace falta /tournaments (antes: miles de torneos y 1-2 peticiones por liga).
  /odds?fixtureId&bookmakers=bet365,pinnacle                              1 petición por partido.
      Las dos casas salen de la misma respuesta: misma hora → la comparación bet365 vs Pinnacle es válida.
  /markets?sportId                                                        1 petición por deporte cada 30 días.
Límite de ritmo: ≥ 2 s entre llamadas a /fixtures (1 s en el resto).
respuesta de /odds: bookmakerOdds[slug].markets[marketId].outcomes[outcomeId].players["0"].price
Todo es defensivo: si la API cambia o falla, el radar sigue funcionando sin esta fuente."""
from __future__ import annotations

import calendar
import logging
import time
from datetime import datetime, timedelta, timezone

import requests

from ..config import Settings, sport_of
from ..db import DB
from ..teams import match_team, player_key, us_key

log = logging.getLogger(__name__)
BASE = "https://api.oddspapi.io/v4"
SPORT_ID = {"football": 10, "basketball": 11, "tennis": 12, "nfl": 14}
FIXTURE_FIELDS = ("fixtureId", "participant1Name", "participant2Name", "startTime", "tournamentName", "categoryName")


def _parse_time(st, tz=timezone.utc) -> datetime | None:
    try:
        if isinstance(st, (int, float)):
            return datetime.fromtimestamp(st, tz)
        return datetime.fromisoformat(str(st).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


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

    def _get(self, path: str, now: datetime, min_gap: float = 1.1, **params):
        wait = min_gap - (time.time() - self._last_call)
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

    # ------------------------------------------------------------ partidos de las próximas horas
    def upcoming_fixtures(self, sport: str, now: datetime) -> list[dict]:
        """Partidos de un deporte que empiezan en la ventana y tienen cuota de bet365: 1 petición cada 12 h."""
        key = f"op_upcoming_{sport}"
        c = self.db.get(key)
        if c and (now - datetime.fromisoformat(c["ts"])).total_seconds() < self.s.oddspapi_fixtures_ttl_h * 3600:
            return c["data"]
        end = now + timedelta(hours=min(self.s.oddspapi_window_h, 47.5))
        fmt = "%Y-%m-%dT%H:%M:%SZ"
        data = self._get("/fixtures", now, min_gap=2.1, sportId=SPORT_ID[sport],
                         **{"from": now.astimezone(timezone.utc).strftime(fmt),
                            "to": end.astimezone(timezone.utc).strftime(fmt)},
                         statusId=0, hasOdds="true", bookmakers=self.s.book)
        fx = [{k: f.get(k) for k in FIXTURE_FIELDS} for f in (data if isinstance(data, list) else [])]
        self.db.set(key, {"ts": now.isoformat(), "data": fx})
        log.info("OddsPapi: %d partidos de %s con bet365 en las próximas horas", len(fx), sport)
        return fx

    @staticmethod
    def _same(a: str, b: str, sport: str) -> bool:
        if sport == "tennis":
            return player_key(a) == player_key(b)
        if sport in ("basketball", "nfl"):
            return us_key(a) == us_key(b)
        return match_team(a, [b]) is not None

    def find_fixture(self, match: dict, now: datetime) -> tuple[str, bool] | None:
        """(fixtureId, invertido) del partido, o None."""
        sport = match.get("sport") or sport_of(match["league"])
        if sport not in SPORT_ID:
            return None
        ko = datetime.fromisoformat(match["kickoff"])
        for f in self.upcoming_fixtures(sport, now):
            t = _parse_time(f.get("startTime"), ko.tzinfo or timezone.utc)
            if t is None or abs((t - ko).total_seconds()) > 6 * 3600:
                continue
            p1, p2 = str(f.get("participant1Name") or ""), str(f.get("participant2Name") or "")
            if self._same(p1, match["home"], sport) and self._same(p2, match["away"], sport):
                return str(f["fixtureId"]), False
            if self._same(p2, match["home"], sport) and self._same(p1, match["away"], sport):
                return str(f["fixtureId"]), True
        return None

    # ------------------------------------------------------------ cuotas
    def match_odds(self, match: dict, now: datetime) -> list[dict]:
        """Filas de cuotas de bet365 y de las casas sharp (ODDSPAPI_BOOKS) de UNA misma consulta, con la misma hora."""
        if not self.enabled:
            return []
        sport = match.get("sport") or sport_of(match["league"])
        try:
            fx = self.find_fixture(match, now)
            if not fx:
                log.info("OddsPapi: no encuentro %s - %s", match["home"], match["away"])
                return []
            fid, swapped = fx
            mmap = self.market_map(sport, now)
            data = self._get("/odds", now, fixtureId=fid, bookmakers=self.s.oddspapi_books)
        except (requests.RequestException, ValueError, KeyError) as e:
            log.warning("OddsPapi: %s", e)
            return []
        return parse_match_odds(data, mmap, match, sport, swapped, self.s.book, now)

    bet365_odds = match_odds  # nombre antiguo


def parse_match_odds(data, mmap: dict, match: dict, sport: str, swapped: bool, book: str, now: datetime) -> list[dict]:
    rows = []
    books = (data.get("bookmakerOdds") or {}) if isinstance(data, dict) else {}
    stamp = now.isoformat()
    for slug, bk in books.items():
        if not isinstance(bk, dict) or bk.get("suspended") or bk.get("bookmakerIsActive") is False:
            continue
        name = str(slug).split("+")[0].replace("-", "_")  # 'pinnacle+30' → pinnacle, 'betfair-ex' → betfair_ex
        for _, m in (bk.get("markets") or {}).items():
            if not isinstance(m, dict) or m.get("marketActive") is False:
                continue
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
                                 away=match["away"], source="oddspapi", bookmaker=book if name == book else name,
                                 market=market, selection=sel, point=pt, price=float(price), fetched_at=stamp))
    return rows
