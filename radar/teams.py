"""Unifica nombres de equipos entre fuentes. El nombre canónico es el de football-data.co.uk."""
from __future__ import annotations

import difflib
import re
import unicodedata

# Nombre en otras fuentes (The Odds API, API-Football) → nombre en football-data.co.uk
ALIASES = {
    # Inglaterra
    "manchester united": "Man United", "manchester city": "Man City", "wolverhampton wanderers": "Wolves",
    "nottingham forest": "Nott'm Forest", "newcastle united": "Newcastle", "tottenham hotspur": "Tottenham",
    "brighton and hove albion": "Brighton", "west ham united": "West Ham", "leeds united": "Leeds",
    "leicester city": "Leicester", "ipswich town": "Ipswich", "afc bournemouth": "Bournemouth",
    "luton town": "Luton", "sheffield united": "Sheffield United", "west bromwich albion": "West Brom",
    "sunderland afc": "Sunderland",
    # España
    "atletico madrid": "Ath Madrid", "atletico de madrid": "Ath Madrid", "athletic bilbao": "Ath Bilbao",
    "athletic club": "Ath Bilbao", "real sociedad": "Sociedad", "celta vigo": "Celta", "rc celta": "Celta",
    "rayo vallecano": "Vallecano", "real betis": "Betis", "espanyol": "Espanol", "rcd espanyol": "Espanol",
    "deportivo alaves": "Alaves", "alaves": "Alaves", "ca osasuna": "Osasuna", "real oviedo": "Oviedo",
    "elche cf": "Elche", "real valladolid": "Valladolid", "leganes": "Leganes", "cd leganes": "Leganes",
    "ud las palmas": "Las Palmas", "rcd mallorca": "Mallorca", "girona fc": "Girona", "levante ud": "Levante",
    "sevilla fc": "Sevilla", "villarreal cf": "Villarreal", "getafe cf": "Getafe", "valencia cf": "Valencia",
    "racing santander": "Santander", "real zaragoza": "Zaragoza", "deportivo la coruna": "La Coruna",
    # Alemania
    "borussia dortmund": "Dortmund", "borussia monchengladbach": "M'gladbach", "bayer leverkusen": "Leverkusen",
    "eintracht frankfurt": "Ein Frankfurt", "1. fc koln": "FC Koln", "fc koln": "FC Koln", "fc cologne": "FC Koln",
    "mainz 05": "Mainz", "fsv mainz 05": "Mainz", "1. fsv mainz 05": "Mainz", "vfb stuttgart": "Stuttgart",
    "tsg hoffenheim": "Hoffenheim", "1899 hoffenheim": "Hoffenheim", "union berlin": "Union Berlin",
    "1. fc union berlin": "Union Berlin", "fc st. pauli": "St Pauli", "st. pauli": "St Pauli",
    "werder bremen": "Werder Bremen", "sv werder bremen": "Werder Bremen", "hamburger sv": "Hamburg",
    "1. fc heidenheim": "Heidenheim", "fc heidenheim": "Heidenheim", "vfl wolfsburg": "Wolfsburg",
    "sc freiburg": "Freiburg", "fc augsburg": "Augsburg", "rb leipzig": "RB Leipzig", "bayern munich": "Bayern Munich",
    "fc bayern munchen": "Bayern Munich", "vfl bochum": "Bochum", "holstein kiel": "Holstein Kiel",
    "fc schalke 04": "Schalke 04", "hertha berlin": "Hertha",
    # Italia
    "inter milan": "Inter", "internazionale": "Inter", "ac milan": "Milan", "as roma": "Roma", "hellas verona": "Verona",
    "atalanta bc": "Atalanta", "ss lazio": "Lazio", "ssc napoli": "Napoli", "acf fiorentina": "Fiorentina",
    "torino fc": "Torino", "genoa cfc": "Genoa", "us lecce": "Lecce", "cagliari calcio": "Cagliari",
    "udinese calcio": "Udinese", "us sassuolo": "Sassuolo", "parma calcio": "Parma", "como 1907": "Como",
    "ac pisa": "Pisa", "us cremonese": "Cremonese",
    # Francia
    "paris saint germain": "Paris SG", "paris saint-germain": "Paris SG", "psg": "Paris SG", "saint etienne": "St Etienne",
    "as saint-etienne": "St Etienne", "as monaco": "Monaco", "olympique lyonnais": "Lyon", "olympique marseille": "Marseille",
    "olympique de marseille": "Marseille", "stade rennais": "Rennes", "rc lens": "Lens", "stade brestois 29": "Brest",
    "ogc nice": "Nice", "losc lille": "Lille", "lille osc": "Lille", "rc strasbourg": "Strasbourg",
    "rc strasbourg alsace": "Strasbourg", "fc nantes": "Nantes", "montpellier hsc": "Montpellier", "stade de reims": "Reims",
    "toulouse fc": "Toulouse", "angers sco": "Angers", "aj auxerre": "Auxerre", "fc lorient": "Lorient", "fc metz": "Metz",
    "le havre ac": "Le Havre",
}

_STOP = {"fc", "cf", "ac", "afc", "sc", "ssc", "as", "rc", "cd", "ud", "sd", "club", "calcio", "1.", "the"}


def norm(name: str) -> str:
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z0-9.' -]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _core(name: str) -> str:
    return " ".join(t for t in norm(name).replace("-", " ").split() if t not in _STOP)


def match_team(raw: str, known: list[str], cutoff: float = 0.72) -> str | None:
    """Devuelve el nombre canónico (de la lista `known`) para `raw`, o None si no hay coincidencia fiable."""
    if raw in known:
        return raw
    n = norm(raw)
    alias = ALIASES.get(n)
    if alias and (not known or alias in known):
        return alias
    by_core = {_core(k): k for k in known}
    core = _core(raw)
    if core in by_core:
        return by_core[core]
    # coincidencia por inclusión ("Brentford FC" ↔ "Brentford")
    for c, k in by_core.items():
        if c and (c in core.split() or core in c.split()) and len(c) >= 4:
            return k
    best = difflib.get_close_matches(core, list(by_core), n=1, cutoff=cutoff)
    return by_core[best[0]] if best else (alias if alias else None)


# ------------------------------------------------------------------ NBA / NFL
US_FIX = {"la clippers": "clippers", "los angeles clippers": "clippers", "trail blazers": "blazers"}


def us_key(name: str) -> str:
    """En NBA y NFL el apodo (última palabra) es único: 'Los Angeles Lakers' → 'lakers'."""
    n = norm(name)
    for k, v in US_FIX.items():
        if n.endswith(k):
            return v
    return n.split()[-1] if n else n


def match_us(raw: str, known: list[str]) -> str | None:
    k = us_key(raw)
    for t in known:
        if us_key(t) == k:
            return t
    return None


# ------------------------------------------------------------------ tenis
def player_key(name: str) -> str:
    """Clave común para 'Carlos Alcaraz' (The Odds API / OddsPapi) y 'Alcaraz C.' (tennis-data):
    último apellido + inicial del nombre → 'alcaraz c'."""
    s = norm(name).replace("-", " ").replace(".", ". ").strip()
    toks = s.split()
    if not toks:
        return ""
    if toks[-1].endswith(".") or (len(toks[-1]) <= 2 and len(toks) > 1):
        # formato tennis-data: 'De Minaur A.' / 'Cerundolo J.M.'
        initials = [t for t in toks if t.endswith(".") or len(t.rstrip(".")) == 1]
        surname = [t for t in toks if t not in initials]
        first = initials[0].rstrip(".")[:1] if initials else ""
        return f"{surname[-1] if surname else toks[0]} {first}".strip()
    # formato 'Nombre Apellido(s)'
    if len(toks) == 1:
        return toks[0]
    if "," in name:  # 'Alcaraz, Carlos'
        last, first = [norm(x) for x in name.split(",", 1)]
        return f"{last.split()[-1]} {first[:1]}"
    return f"{toks[-1]} {toks[0][:1]}"


def match_player(raw: str, known_by_key: dict[str, str]) -> str | None:
    return known_by_key.get(player_key(raw))
