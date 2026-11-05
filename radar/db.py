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
    hg INTEGER, ag INTEGER,        -- resultado cuando se conoce
    status TEXT DEFAULT 'pendiente'
);
CREATE TABLE IF NOT EXISTS odds (
    match_id TEXT, bookmaker TEXT, market TEXT, selection TEXT, point REAL,
    price REAL, first_price REAL, first_seen TEXT, updated TEXT, source TEXT,
    PRIMARY KEY (match_id, bookmaker, market, selection, point)
);
-- Última predicción antes del partido por selección (sirve para calibrar el modelo)
CREATE TABLE IF NOT EXISTS predictions (
    match_id TEXT, market TEXT, selection TEXT, point REAL,
    p_model REAL, p_market REAL, p_final REAL, updated TEXT, won INTEGER,
    PRIMARY KEY (match_id, market, selection, point)
);
-- Apuestas de valor detectadas (se guardan con la cuota del momento del aviso)
CREATE TABLE IF NOT EXISTS picks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created TEXT, match_id TEXT, league TEXT, kickoff TEXT, home TEXT, away TEXT,
    market TEXT, selection TEXT, point REAL, label TEXT,
    p_final REAL, p_model REAL, p_market REAL,
    odds REAL, bookmaker TEXT, b365_odds REAL, ev REAL, stake REAL, confidence TEXT, reasons TEXT,
    alerted INTEGER DEFAULT 0, won INTEGER, profit REAL, closing_odds REAL, clv REAL,
    UNIQUE (match_id, market, selection, point)
);
CREATE TABLE IF NOT EXISTS combos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created TEXT, risk TEXT, bookmaker TEXT, legs TEXT, odds REAL, prob REAL, ev REAL, stake REAL,
    signature TEXT UNIQUE, alerted INTEGER DEFAULT 0, won INTEGER, profit REAL
);
CREATE TABLE IF NOT EXISTS team_map (source TEXT, league TEXT, raw TEXT, canonical TEXT,
    PRIMARY KEY (source, league, raw));
CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS runs (ts TEXT, step TEXT, detail TEXT);
"""


class DB:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)

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

    # --- clave-valor para parámetros aprendidos, presupuesto de créditos, etc.
    def get(self, key: str, default=None):
        row = self.conn.execute("SELECT value FROM kv WHERE key=?", (key,)).fetchone()
        return json.loads(row["value"]) if row else default

    def set(self, key: str, value) -> None:
        self.execute("INSERT OR REPLACE INTO kv (key, value) VALUES (?, ?)", (key, json.dumps(value)))

    def log(self, ts: str, step: str, detail: str) -> None:
        self.execute("INSERT INTO runs VALUES (?, ?, ?)", (ts, step, detail))
        # conservar solo los últimos 2000 registros
        self.execute("DELETE FROM runs WHERE rowid NOT IN (SELECT rowid FROM runs ORDER BY rowid DESC LIMIT 2000)")

    def close(self):
        self.conn.close()
