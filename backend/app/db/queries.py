import sqlite3
from pathlib import Path
from datetime import datetime

DB_PATH = Path("data/metrics/queries.db")

def get_conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_PATH)

def init_db():
    conn = get_conn()
    conn.execute("""
    CREATE TABLE IF NOT EXISTS queries (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        query TEXT,
        latency REAL,
        result_count INTEGER,
        created_at TEXT
    )
    """)
    conn.commit()
    conn.close()

def log_query(query: str, latency: float, result_count: int):
    conn = get_conn()
    conn.execute(
        "INSERT INTO queries (query, latency, result_count, created_at) VALUES (?, ?, ?, ?)",
        (query, latency, result_count, datetime.utcnow().isoformat())
    )
    conn.commit()
    conn.close()