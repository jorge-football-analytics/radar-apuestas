"""Configuración central. Todo lo ajustable se lee de variables de entorno (.env o secretos de GitHub)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

try:  # .env opcional en local
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("RADAR_DATA_DIR", ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"

# Ligas: código football-data.co.uk, clave de The Odds API e id de API-Football
LEAGUES: dict[str, dict] = {
    "laliga": {"nombre": "LaLiga", "fd": "SP1", "odds": "soccer_spain_la_liga", "apif": 140},
    "premier": {"nombre": "Premier League", "fd": "E0", "odds": "soccer_epl", "apif": 39},
    "seriea": {"nombre": "Serie A", "fd": "I1", "odds": "soccer_italy_serie_a", "apif": 135},
    "bundesliga": {"nombre": "Bundesliga", "fd": "D1", "odds": "soccer_germany_bundesliga", "apif": 78},
    "ligue1": {"nombre": "Ligue 1", "fd": "F1", "odds": "soccer_france_ligue_one", "apif": 61},
    # Champions: sin histórico gratuito en football-data → solo comparación de mercado
    "champions": {"nombre": "Champions League", "fd": None, "odds": "soccer_uefa_champs_league", "apif": 2},
    "europa": {"nombre": "Europa League", "fd": None, "odds": "soccer_uefa_europa_league", "apif": 3},
    "conference": {"nombre": "Conference League", "fd": None, "odds": "soccer_uefa_europa_conference_league", "apif": 848},
}
FD_TO_LEAGUE = {v["fd"]: k for k, v in LEAGUES.items() if v["fd"]}


def _f(name: str, default: float) -> float:
    return float(os.getenv(name, default))


def _i(name: str, default: int) -> int:
    return int(os.getenv(name, default))


@dataclass
class Settings:
    # Claves
    odds_api_key: str = field(default_factory=lambda: os.getenv("ODDS_API_KEY", ""))
    apifootball_key: str = field(default_factory=lambda: os.getenv("APIFOOTBALL_KEY", ""))
    telegram_token: str = field(default_factory=lambda: os.getenv("TELEGRAM_BOT_TOKEN", ""))
    telegram_chat_id: str = field(default_factory=lambda: os.getenv("TELEGRAM_CHAT_ID", ""))

    # The Odds API
    odds_regions: str = field(default_factory=lambda: os.getenv("ODDS_REGIONS", "eu"))
    # Si se rellena (máx. 10 casas = coste de 1 región), sustituye a regions. Ej: "bet365,williamhill,betfair_ex_eu"
    odds_bookmakers: str = field(default_factory=lambda: os.getenv("ODDS_BOOKMAKERS", ""))
    odds_markets: str = field(default_factory=lambda: os.getenv("ODDS_MARKETS", "h2h,totals"))
    odds_lookahead_h: int = field(default_factory=lambda: _i("ODDS_LOOKAHEAD_H", 72))
    odds_reserve: int = field(default_factory=lambda: _i("ODDS_RESERVE", 20))  # créditos que nunca se gastan

    # Modelo
    seasons_history: int = field(default_factory=lambda: _i("SEASONS_HISTORY", 4))
    time_decay_xi: float = field(default_factory=lambda: _f("TIME_DECAY_XI", 0.0019))  # vida media ~1 año
    model_weight: float = field(default_factory=lambda: _f("MODEL_WEIGHT", 0.30))  # peso inicial modelo vs mercado
    injury_impact: float = field(default_factory=lambda: _f("INJURY_IMPACT", 0.02))  # % de goles por baja
    injury_cap: float = field(default_factory=lambda: _f("INJURY_CAP", 0.12))

    # Filtros de apuestas de valor
    min_ev: float = field(default_factory=lambda: _f("MIN_EV", 0.04))
    min_prob: float = field(default_factory=lambda: _f("MIN_PROB", 0.25))
    max_odds: float = field(default_factory=lambda: _f("MAX_ODDS", 6.0))
    min_books: int = field(default_factory=lambda: _i("MIN_BOOKS", 2))
    alert_ev: float = field(default_factory=lambda: _f("ALERT_EV", 0.06))  # umbral para avisar por Telegram

    # Gestión de banca
    bankroll: float = field(default_factory=lambda: _f("BANKROLL", 100.0))
    kelly_fraction: float = field(default_factory=lambda: _f("KELLY_FRACTION", 0.25))
    max_stake_pct: float = field(default_factory=lambda: _f("MAX_STAKE_PCT", 0.03))

    # Combinadas
    max_combo_legs: int = field(default_factory=lambda: _i("MAX_COMBO_LEGS", 4))
    preferred_book: str = field(default_factory=lambda: os.getenv("PREFERRED_BOOK", "bet365"))
    # Casas donde puedes apostar (vacío = todas salvo las de NO_BET_BOOKS). Ej: "bet365" o "bet365,bwin,williamhill"
    bet_books: str = field(default_factory=lambda: os.getenv("BET_BOOKS", ""))
    # Casas que solo sirven de referencia (Pinnacle no opera en España)
    no_bet_books: str = field(default_factory=lambda: os.getenv("NO_BET_BOOKS", "pinnacle"))

    def can_bet(self, book: str) -> bool:
        allowed = {b.strip() for b in self.bet_books.split(",") if b.strip()}
        if allowed:
            return book in allowed
        return book not in {b.strip() for b in self.no_bet_books.split(",") if b.strip()}

    db_path: Path = field(default_factory=lambda: Path(os.getenv("RADAR_DB", DATA_DIR / "radar.db")))


def season_code(d: date, offset: int = 0) -> str:
    """Código de temporada de football-data ('2627' para 2026/27). offset=-1 → temporada anterior."""
    start = d.year if d.month >= 7 else d.year - 1
    start += offset
    return f"{start % 100:02d}{(start + 1) % 100:02d}"


def season_start_year(d: date) -> int:
    return d.year if d.month >= 7 else d.year - 1
