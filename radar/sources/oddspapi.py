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
FIXTURE_FIELDS = ("fixtureId", "participant1Name", "participant2Name", "startTime", "tournamentId", "tournamentName",
                  "categoryName")
DC_SWAP = {"1X": "X2", "X2": "1X", "12": "12"}


def build_market_map(data, sport: str) -> dict[str, tuple[str, str, float]]:
    """outcomeId → (mercado, selección, línea), usando el NOMBRE de cada resultado ('1', 'X', '2', 'Over'...).

    Antes se asignaba local/empate/visitante por el orden de la lista, pero OddsPapi los lista como '2, X, 1' en el
    1X2 y '2, 1' en el ganador de tenis: las cuotas salían con local y visitante invertidos. Y cualquier tipo que
    contuviera 'total' (tiros, córners, goles de un equipo...) se tomaba como más/menos goles del partido."""
    out: dict[str, tuple[str, str, float]] = {}
    want = "fulltime" if sport == "football" else "result"
    for m in data if isinstance(data, list) else []:
        if m.get("playerProp") or str(m.get("period", "")).lower() != want:
            continue
        mtype = str(m.get("marketType", "")).lower()
        hcp = float(m.get("handicap") or 0)
        outs = {str(o.get("outcomeName", "")).strip().lower(): str(o["outcomeId"]) for o in m.get("outcomes", [])}
        half = hcp % 1 == 0.5 or hcp % 1 == -0.5 or abs(hcp % 1 - 0.5) < 1e-9
        if mtype == "1x2" and sport == "football":
            for name, sel in (("1", "H"), ("x", "D"), ("2", "A")):
                if name in outs:
                    out[outs[name]] = ("1x2", sel, 0.0)
        elif mtype == "moneyline" and sport != "football":
            for name, sel in (("1", "H"), ("2", "A")):
                if name in outs:
                    out[outs[name]] = ("ml", sel, 0.0)
        elif mtype == "doublechance" and sport == "football":
            for name, sel in (("1x", "1X"), ("12", "12"), ("2x", "X2"), ("x2", "X2")):
                if name in outs:
                    out[outs[name]] = ("dc", sel, 0.0)
        elif mtype == "bothteamsscore" and sport == "football":
            for name, sel in (("yes", "Y"), ("no", "N")):
                if name in outs:
                    out[outs[name]] = ("btts", sel, 0.0)
        elif mtype == "totals" and hcp > 0 and half:
            for name, sel in (("over", "O"), ("under", "U")):
                if name in outs:
                    out[outs[name]] = ("ou", sel, hcp)
        elif mtype == "spreads" and sport in ("football", "basketball", "nfl") and half:
            for name, sel in (("1", "H"), ("2", "A")):  # línea = hándicap del participante 1
                if name in outs:
                    out[outs[name]] = ("spread", sel, hcp)
    return out


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
        return build_market_map(data, sport)

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
        self.db.set(key, {"ts": now.isoformat(), "data": fx})  # solo los campos útiles: la base no engorda
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
        hit = self._match_in(self.upcoming_fixtures(sport, now), match, sport)
        return (str(hit[0]["fixtureId"]), hit[1]) if hit else None

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

    # ------------------------------------------------------------ carga masiva (v3)
    def fetch_bulk(self, matches: list[dict], now: datetime, books: tuple[str, ...] | None = None,
                   refresh_h: float = 6) -> list[dict]:
        """bet365 (y, si sobra presupuesto, Pinnacle) para TODOS nuestros partidos de las próximas horas con
        /odds-by-tournaments: 1 consulta = una casa en todos los partidos de hasta 5 competiciones.
        Coste por ejecución: 1 consulta por deporte (lista de partidos, cada 12 h) + 1 por cada 5 competiciones
        y casa. Las competiciones con más partidos nuestros van primero; se para al agotar el presupuesto del día."""
        if not self.enabled or not matches:
            return []
        books = books or tuple(b.strip() for b in self.s.oddspapi_books.split(",") if b.strip())
        limit = now + timedelta(hours=self.s.oddspapi_window_h)
        by_sport: dict[str, list[dict]] = {}
        for m in matches:
            sp = m.get("sport") or sport_of(m["league"])
            ko = datetime.fromisoformat(m["kickoff"])
            if sp in SPORT_ID and now < ko <= limit:
                by_sport.setdefault(sp, []).append(m)
        rows: list[dict] = []
        plan: list[tuple[int, str, list[int], dict]] = []  # (prioridad, deporte, torneos, mapa fixture→partido)
        for sp, ms in sorted(by_sport.items(), key=lambda kv: -len(kv[1])):
            if self.allowance(now) < 1:
                break
            try:
                fixtures = self.upcoming_fixtures(sp, now)
            except (requests.RequestException, ValueError) as e:
                log.warning("OddsPapi fixtures %s: %s", sp, e)
                continue
            fmap: dict[str, tuple[dict, bool]] = {}
            count: dict[int, int] = {}
            for m in ms:
                hit = self._match_in(fixtures, m, sp)
                if hit:
                    f, swapped = hit
                    fmap[str(f["fixtureId"])] = (m, swapped)
                    tid = f.get("tournamentId")
                    if tid is not None:
                        count[tid] = count.get(tid, 0) + 1
            log.info("OddsPapi %s: %d de %d partidos encontrados en %d competiciones", sp, len(fmap), len(ms), len(count))
            tids = [t for t, _ in sorted(count.items(), key=lambda kv: -kv[1])]
            for i in range(0, len(tids), 5):
                chunk = tids[i:i + 5]
                plan.append((sum(count[t] for t in chunk), sp, chunk, fmap))
        plan.sort(key=lambda x: -x[0])
        for book in books:  # primero bet365 en todo; después la referencia sharp con lo que sobre
            for _, sp, chunk, fmap in plan:
                if self.allowance(now) < 1:
                    log.info("OddsPapi: presupuesto del día agotado")
                    return rows
                key = f"op_bulk_{book}_{'-'.join(map(str, chunk))}"
                last = self.db.get(key)
                if last and (now - datetime.fromisoformat(last)).total_seconds() < refresh_h * 3600:
                    continue
                try:
                    mmap = self.market_map(sp, now)
                    data = self._get("/odds-by-tournaments", now, tournamentIds=",".join(map(str, chunk)), bookmaker=book)
                except (requests.RequestException, ValueError) as e:
                    log.warning("OddsPapi odds-by-tournaments %s: %s", chunk, e)
                    continue
                self.db.set(key, now.isoformat())
                fixtures = data if isinstance(data, list) else (data or {}).get("data", []) if isinstance(data, dict) else []
                n = 0
                for fx in fixtures:
                    hit = fmap.get(str(fx.get("fixtureId")))
                    if not hit:
                        continue
                    m, swapped = hit
                    got = parse_match_odds(fx, mmap, m, sp, swapped, self.s.book, now)
                    rows += got
                    n += bool(got)
                log.info("OddsPapi %s %s: cuotas de %d partidos", book, sp, n)
        return rows

    def _match_in(self, fixtures: list[dict], match: dict, sport: str):
        ko = datetime.fromisoformat(match["kickoff"])
        for f in fixtures:
            t = _parse_time(f.get("startTime"), ko.tzinfo or timezone.utc)
            if t is None or abs((t - ko).total_seconds()) > 6 * 3600:
                continue
            p1, p2 = str(f.get("participant1Name") or ""), str(f.get("participant2Name") or "")
            if self._same(p1, match["home"], sport) and self._same(p2, match["away"], sport):
                return f, False
            if self._same(p2, match["home"], sport) and self._same(p1, match["away"], sport):
                return f, True
        return None


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
                if swapped:  # el participante 1 de OddsPapi es nuestro visitante
                    if market == "dc":
                        sel = DC_SWAP[sel]
                    elif sel in ("H", "A"):
                        sel = "A" if sel == "H" else "H"
                        if market == "spread":
                            pt = -pt  # siempre guardamos el hándicap del local
                rows.append(dict(sport=sport, league=match["league"], kickoff=match["kickoff"], home=match["home"],
                                 away=match["away"], source="oddspapi", bookmaker=book if name == book else name,
                                 market=market, selection=sel, point=pt, price=float(price), fetched_at=stamp,
                                 match_id=match.get("id")))
    return rows
