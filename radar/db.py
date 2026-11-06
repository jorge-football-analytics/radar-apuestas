"""Base de datos SQLite: un único fichero que viaja con el repo (GitHub Actions lo vuelve a subir tras cada ejecución)."""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS matches (
    id TEXT PRIMARY KEY,           -- liga|YYYY-MM-DD|local|visitante (nombres canónicos)
    league TEXT, kickoff TEXT, home TEXT, away TEXT,
    hg INTEGER, ag INTEGER,
    status TEXT DEFAULT 'pendiente',
    sport TEXT DEFAULT 'football'
);
CREATE TABLE IF NOT EXISTS odds (
    match_id TEXT, bookmaker TEXT, market TEXT, selection TEXT, point REAL,
    price REAL, first_price REAL, first_seen TEXT, updated TEXT, source TEXT,
    PRIMARY KEY (match_id, bookmaker, market, selection, point)
);
CREATE TABLE IF NOT EXISTS predictions (
    match_id TEXT, market TEXT, selection TEXT, point REAL,
    p_model REAL, p_market REAL, p_final REAL, updated TEXT, won INTEGER,
    PRIMARY KEY (match_id, market, selection, point)
);
-- Jugadas. tier: 'valor' (cuota bet365 confirmada con EV+), 'verificar' (falta la cuota de bet365),
-- 'pronostico' (lo más probable/rentable del partido aunque no tenga ventaja). won: 1, 0, -1 (nula)
CREATE TABLE IF NOT EXISTS picks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created TEXT, match_id TEXT, league TEXT, kickoff TEXT, home TEXT, away TEXT,
    market TEXT, selection TEXT, point REAL, label TEXT,
    p_final REAL, p_model REAL, p_market REAL,
    odds REAL, bookmaker TEXT, b365_odds REAL, ev REAL, stake REAL, confidence TEXT, reasons TEXT,
    alerted INTEGER DEFAULT 0, won INTEGER, profit REAL, closing_odds REAL, clv REAL,
    sport TEXT DEFAULT 'football', risk TEXT, tier TEXT DEFAULT 'valor', elite INTEGER DEFAULT 0,
    verified INTEGER DEFAULT 1, min_odds REAL, score REAL, updated TEXT,
    UNIQUE (match_id, market, selection, point)
);
CREATE TABLE IF NOT EXISTS combos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created TEXT, risk TEXT, bookmaker TEXT, legs TEXT, odds REAL, prob REAL, ev REAL, stake REAL,
    signature TEXT UNIQUE, alerted INTEGER DEFAULT 0, won INTEGER, profit REAL,
    sports TEXT, first_kickoff TEXT, tier TEXT DEFAULT 'valor'
);
CREATE TABLE IF NOT EXISTS team_map (source TEXT, league TEXT, raw TEXT, canonical TEXT,
    PRIMARY KEY (source, league, raw));
CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT);
-- Detalle del modelo por partido (goles/puntos esperados, Elo, superficie...) para la ficha del partido
CREATE TABLE IF NOT EXISTS match_info (match_id TEXT PRIMARY KEY, info TEXT, updated TEXT);
CREATE TABLE IF NOT EXISTS runs (ts TEXT, step TEXT, detail TEXT);
"""

# Columnas añadidas en la v2 (para migrar bases de datos antiguas sin perder nada)
MIGRATIONS = {
    "matches": {"sport": "TEXT DEFAULT 'football'"},
    "picks": {"sport": "TEXT DEFAULT 'football'", "risk": "TEXT", "tier": "TEXT DEFAULT 'valor'",
              "elite": "INTEGER DEFAULT 0", "verified": "INTEGER DEFAULT 1", "min_odds": "REAL", "score": "REAL",
              "updated": "TEXT"},
    "combos": {"sports": "TEXT", "first_kickoff": "TEXT", "tier": "TEXT DEFAULT 'valor'"},
}


class DB:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self._migrate()

    def _migrate(self) -> None:
        for table, cols in MIGRATIONS.items():
            have = {r[1] for r in self.conn.execute(f"PRAGMA table_info({table})")}
            for col, decl in cols.items():
                if col not in have:
                    self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {decl}")
        if self.get("schema_version", 1) < 2:
            # Las jugadas antiguas eran de cualquier casa: se conservan como históricas
            self.conn.execute("UPDATE picks SET risk = CASE WHEN p_final >= 0.6 THEN 'bajo' WHEN p_final >= 0.4 "
                              "THEN 'medio' ELSE 'alto' END WHERE risk IS NULL")
            self.conn.execute("UPDATE combos SET first_kickoff = created WHERE first_kickoff IS NULL")
            for key in ("weights", "weights_backtest"):  # '1x2' → 'football|1x2'
                w = self.get(key)
                if isinstance(w, dict):
                    self.set(key, {(k if "|" in k else f"football|{k}"): v for k, v in w.items()})
            self.set("schema_version", 2)
        self.conn.commit()

    @contextmanager
    def tx(self):
        try:
            yield self.conn
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    def execute(self, sql: str, params: tuple | dict = ()):
        cur = self.conn.execute(sql, params)
        self.conn.commit()
        return cur

    def df(self, sql: str, params: tuple | dict = ()) -> pd.DataFrame:
        return pd.read_sql_query(sql, self.conn, params=params)

    def get(self, key: str, default=None):
        row = self.conn.execute("SELECT value FROM kv WHERE key=?", (key,)).fetchone()
        return json.loads(row["value"]) if row else default

    def set(self, key: str, value) -> None:
        self.execute("INSERT OR REPLACE INTO kv (key, value) VALUES (?, ?)", (key, json.dumps(value)))

    def log(self, ts: str, step: str, detail: str) -> None:
        self.execute("INSERT INTO runs VALUES (?, ?, ?)", (ts, step, detail))
        self.execute("DELETE FROM runs WHERE rowid NOT IN (SELECT rowid FROM runs ORDER BY rowid DESC LIMIT 2000)")

    def close(self):
        self.conn.close()
