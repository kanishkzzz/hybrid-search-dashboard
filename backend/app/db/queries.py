import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Any

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


def query_summary(limit: int = 10) -> dict[str, Any]:
    conn = get_conn()
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute("SELECT query, latency, result_count FROM queries").fetchall()
    finally:
        conn.close()

    latencies_ms = sorted(float(row["latency"]) * 1000.0 for row in rows)
    queries = [str(row["query"]) for row in rows if row["query"]]
    zero_result_queries = [
        str(row["query"])
        for row in rows
        if row["query"] and int(row["result_count"] or 0) == 0
    ]

    def percentile(values: list[float], p: float) -> float | None:
        if not values:
            return None
        index = int(round((len(values) - 1) * p))
        return values[max(0, min(index, len(values) - 1))]

    def top_counts(values: list[str]) -> list[dict[str, Any]]:
        counts: dict[str, int] = {}
        for value in values:
            counts[value] = counts.get(value, 0) + 1
        return [
            {"query": query, "count": count}
            for query, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]
        ]

    return {
        "request_volume": len(rows),
        "p50_latency_ms": percentile(latencies_ms, 0.50),
        "p95_latency_ms": percentile(latencies_ms, 0.95),
        "top_queries": top_counts(queries),
        "zero_result_queries": top_counts(zero_result_queries),
    }
