"""Configuración central: deportes, ligas y parámetros (todo ajustable por variables de entorno)."""
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

# ---------------------------------------------------------------- deportes
SPORTS = {
    "football": {"nombre": "Fútbol", "icono": "⚽", "unidad": "goles"},
    "tennis": {"nombre": "Tenis", "icono": "🎾", "unidad": "juegos"},
    "basketball": {"nombre": "Baloncesto", "icono": "🏀", "unidad": "puntos"},
    "nfl": {"nombre": "NFL", "icono": "🏈", "unidad": "puntos"},
}

# Fútbol. fd = código football-data.co.uk (histórico + cuotas bet365 gratis). odds = clave The Odds API.
# oddspapi = palabras para encontrar el torneo en OddsPapi. tier 1 = primeras divisiones top.
FOOTBALL = {
    "laliga": {"nombre": "LaLiga", "pais": "🇪🇸", "fd": "SP1", "odds": "soccer_spain_la_liga", "apif": 140, "tier": 1},
    "premier": {"nombre": "Premier League", "pais": "🏴", "fd": "E0", "odds": "soccer_epl", "apif": 39, "tier": 1},
    "seriea": {"nombre": "Serie A", "pais": "🇮🇹", "fd": "I1", "odds": "soccer_italy_serie_a", "apif": 135, "tier": 1},
    "bundesliga": {"nombre": "Bundesliga", "pais": "🇩🇪", "fd": "D1", "odds": "soccer_germany_bundesliga", "apif": 78, "tier": 1},
    "ligue1": {"nombre": "Ligue 1", "pais": "🇫🇷", "fd": "F1", "odds": "soccer_france_ligue_one", "apif": 61, "tier": 1},
    "laliga2": {"nombre": "LaLiga Hypermotion", "pais": "🇪🇸", "fd": "SP2", "odds": "soccer_spain_segunda_division", "apif": 141, "tier": 2},
    "championship": {"nombre": "Championship", "pais": "🏴", "fd": "E1", "odds": "soccer_efl_champ", "apif": 40, "tier": 2},
    "serieb": {"nombre": "Serie B", "pais": "🇮🇹", "fd": "I2", "odds": "soccer_italy_serie_b", "apif": 136, "tier": 2},
    "bundesliga2": {"nombre": "2. Bundesliga", "pais": "🇩🇪", "fd": "D2", "odds": "soccer_germany_bundesliga2", "apif": 79, "tier": 2},
    "ligue2": {"nombre": "Ligue 2", "pais": "🇫🇷", "fd": "F2", "odds": "soccer_france_ligue_two", "apif": 62, "tier": 2},
    "eredivisie": {"nombre": "Eredivisie", "pais": "🇳🇱", "fd": "N1", "odds": "soccer_netherlands_eredivisie", "apif": 88, "tier": 2},
    "portugal": {"nombre": "Liga Portugal", "pais": "🇵🇹", "fd": "P1", "odds": "soccer_portugal_primeira_liga", "apif": 94, "tier": 2},
    "belgica": {"nombre": "Pro League", "pais": "🇧🇪", "fd": "B1", "odds": "soccer_belgium_first_div", "apif": 144, "tier": 2},
    "turquia": {"nombre": "Süper Lig", "pais": "🇹🇷", "fd": "T1", "odds": "soccer_turkey_super_league", "apif": 203, "tier": 2},
    "grecia": {"nombre": "Super League", "pais": "🇬🇷", "fd": "G1", "odds": "soccer_greece_super_league", "apif": 197, "tier": 2},
    "escocia": {"nombre": "Premiership", "pais": "🏴", "fd": "SC0", "odds": "soccer_spl", "apif": 179, "tier": 2},
    # Competiciones europeas: sin histórico gratis → solo mercado
    "champions": {"nombre": "Champions League", "pais": "🇪🇺", "fd": None, "odds": "soccer_uefa_champs_league", "apif": 2, "tier": 1,
                  "oddspapi": "champions league"},
    "europa": {"nombre": "Europa League", "pais": "🇪🇺", "fd": None, "odds": "soccer_uefa_europa_league", "apif": 3, "tier": 1,
               "oddspapi": "europa league"},
    "conference": {"nombre": "Conference League", "pais": "🇪🇺", "fd": None, "odds": "soccer_uefa_europa_conference_league",
                   "apif": 848, "tier": 2, "oddspapi": "conference league"},
}

# Otros deportes con liga fija. espn = ruta de la API pública de ESPN (histórico para el modelo Elo)
OTHER = {
    "nba": {"sport": "basketball", "nombre": "NBA", "pais": "🇺🇸", "odds": "basketball_nba", "espn": "basketball/nba",
            "oddspapi": "nba"},
    "euroleague": {"sport": "basketball", "nombre": "Euroliga", "pais": "🇪🇺", "odds": "basketball_euroleague", "espn": None,
                   "oddspapi": "euroleague"},
    "nfl": {"sport": "nfl", "nombre": "NFL", "pais": "🇺🇸", "odds": "americanfootball_nfl", "espn": "football/nfl",
            "oddspapi": "nfl"},
}

# Mercados que se piden a The Odds API por deporte (cada mercado cuesta 1 crédito por región)
ODDS_MARKETS = {"football": "h2h,totals", "basketball": "h2h,spreads,totals", "nfl": "h2h,spreads,totals", "tennis": "h2h"}

FD_TO_LEAGUE = {v["fd"]: k for k, v in FOOTBALL.items() if v["fd"]}
LEAGUES = FOOTBALL  # compatibilidad con código antiguo


def sport_of(league: str) -> str:
    if league in FOOTBALL:
        return "football"
    if league in OTHER:
        return OTHER[league]["sport"]
    if league.startswith("tennis_"):
        return "tennis"
    return "football"


def league_name(league: str) -> str:
    if league in FOOTBALL:
        return FOOTBALL[league]["nombre"]
    if league in OTHER:
        return OTHER[league]["nombre"]
    if league.startswith("tennis_"):
        parts = league.split("_")[1:]
        tour = parts[0].upper()
        rest = " ".join(p.capitalize() for p in parts[1:])
        return f"{tour} {rest}".replace("Us Open", "US Open")
    return league


def league_flag(league: str) -> str:
    if league in FOOTBALL:
        return FOOTBALL[league]["pais"]
    if league in OTHER:
        return OTHER[league]["pais"]
    return ""


def _f(name: str, default: float) -> float:
    return float(os.getenv(name) or default)


def _i(name: str, default: int) -> int:
    return int(os.getenv(name) or default)


def _b(name: str, default: bool) -> bool:
    v = os.getenv(name)
    return default if v in (None, "") else v.strip().lower() in ("1", "true", "si", "sí", "yes")


@dataclass
class Settings:
    # Claves
    odds_api_key: str = field(default_factory=lambda: os.getenv("ODDS_API_KEY", ""))
    oddspapi_key: str = field(default_factory=lambda: os.getenv("ODDSPAPI_KEY", ""))
    apifootball_key: str = field(default_factory=lambda: os.getenv("APIFOOTBALL_KEY", ""))
    telegram_token: str = field(default_factory=lambda: os.getenv("TELEGRAM_BOT_TOKEN", ""))
    telegram_chat_id: str = field(default_factory=lambda: os.getenv("TELEGRAM_CHAT_ID", ""))

    # Deportes activos (coma): football,tennis,basketball,nfl
    sports: str = field(default_factory=lambda: os.getenv("SPORTS", "football,tennis,basketball,nfl"))

    # The Odds API (consenso del mercado; NO incluye bet365 España)
    odds_regions: str = field(default_factory=lambda: os.getenv("ODDS_REGIONS", "eu"))
    odds_lookahead_h: int = field(default_factory=lambda: _i("ODDS_LOOKAHEAD_H", 48))
    odds_reserve: int = field(default_factory=lambda: _i("ODDS_RESERVE", 20))
    odds_min_refresh_h: float = field(default_factory=lambda: _f("ODDS_MIN_REFRESH_H", 8))
    # Gastar créditos también en ligas de fútbol nacionales (ya tienen bet365 gratis en football-data)
    odds_api_football: bool = field(default_factory=lambda: _b("ODDS_API_FOOTBALL", False))

    # OddsPapi (cuotas reales de bet365 para tenis, NBA, NFL y copas europeas). Plan gratis: 250 peticiones/mes
    oddspapi_monthly: int = field(default_factory=lambda: _i("ODDSPAPI_MONTHLY", 250))
    oddspapi_reserve: int = field(default_factory=lambda: _i("ODDSPAPI_RESERVE", 10))

    # Modelo
    seasons_history: int = field(default_factory=lambda: _i("SEASONS_HISTORY", 4))
    time_decay_xi: float = field(default_factory=lambda: _f("TIME_DECAY_XI", 0.0019))
    injury_impact: float = field(default_factory=lambda: _f("INJURY_IMPACT", 0.02))
    injury_cap: float = field(default_factory=lambda: _f("INJURY_CAP", 0.12))

    # Casa donde se apuesta
    book: str = field(default_factory=lambda: os.getenv("BOOK", "bet365"))

    # Jugadas de valor (web)
    min_ev: float = field(default_factory=lambda: _f("MIN_EV", 0.03))
    min_prob: float = field(default_factory=lambda: _f("MIN_PROB", 0.25))
    max_odds: float = field(default_factory=lambda: _f("MAX_ODDS", 6.0))
    min_books: int = field(default_factory=lambda: _i("MIN_BOOKS", 3))

    # Filtro élite (Telegram)
    alert_ev: float = field(default_factory=lambda: _f("ALERT_EV", 0.05))
    alert_min_prob: float = field(default_factory=lambda: _f("ALERT_MIN_PROB", 0.33))
    alert_odds_min: float = field(default_factory=lambda: _f("ALERT_ODDS_MIN", 1.30))
    alert_odds_max: float = field(default_factory=lambda: _f("ALERT_ODDS_MAX", 4.0))
    alert_max_day: int = field(default_factory=lambda: _i("ALERT_MAX_DAY", 5))
    alert_combo_ev: float = field(default_factory=lambda: _f("ALERT_COMBO_EV", 0.08))
    daily_digest: bool = field(default_factory=lambda: _b("DAILY_DIGEST", True))

    # Gestión de banca
    bankroll: float = field(default_factory=lambda: _f("BANKROLL", 100.0))
    kelly_fraction: float = field(default_factory=lambda: _f("KELLY_FRACTION", 0.25))
    max_stake_pct: float = field(default_factory=lambda: _f("MAX_STAKE_PCT", 0.03))

    max_combo_legs: int = field(default_factory=lambda: _i("MAX_COMBO_LEGS", 4))

    db_path: Path = field(default_factory=lambda: Path(os.getenv("RADAR_DB", DATA_DIR / "radar.db")))

    @property
    def active_sports(self) -> set[str]:
        return {x.strip() for x in self.sports.split(",") if x.strip()}

    def can_bet(self, book: str) -> bool:
        return book == self.book


# Peso inicial del modelo frente al mercado (lo corrigen el backtest y los resultados reales)
DEFAULT_WEIGHT = {"football": 0.15, "tennis": 0.30, "basketball": 0.20, "nfl": 0.20}


def season_code(d: date, offset: int = 0) -> str:
    """Código de temporada de football-data ('2627' para 2026/27). offset=-1 → temporada anterior."""
    start = d.year if d.month >= 7 else d.year - 1
    start += offset
    return f"{start % 100:02d}{(start + 1) % 100:02d}"


def season_start_year(d: date) -> int:
    return d.year if d.month >= 7 else d.year - 1
