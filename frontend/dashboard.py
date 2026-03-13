from __future__ import annotations

import csv
import os
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any

import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
DEFAULT_SQLITE_PATHS = [
    Path(os.getenv("QUERY_LOG_DB", "")) if os.getenv("QUERY_LOG_DB") else None,
    Path("data/logs/query_logs.db"),
    Path("data/query_logs.db"),
    Path("backend/data/query_logs.db"),
]
DEFAULT_ERROR_LOG_PATHS = [
    Path("data/logs/errors.log"),
    Path("logs/errors.log"),
    Path("backend/logs/errors.log"),
]
METRICS_CSV_PATH = Path("data/metrics/experiments.csv")


def _existing_path(candidates: list[Path | None]) -> Path | None:
    for path in candidates:
        if path and path.exists():
            return path
    return None


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def render_search_page() -> None:
    st.header("Search")
    query = st.text_input("Query", placeholder="Type a search query")
    alpha = st.slider("Alpha", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
    top_k = st.number_input("Top K", min_value=1, max_value=100, value=10, step=1)

    if st.button("Search", type="primary"):
        if not query.strip():
            st.warning("Please enter a query.")
            return

        payload = {"query": query, "top_k": int(top_k), "alpha": float(alpha)}
        try:
            response = requests.post(f"{API_BASE_URL}/search", json=payload, timeout=15)
            response.raise_for_status()
            body = response.json()
            results = body.get("results", [])

            if not results:
                st.info("No results returned.")
                return

            rows = [
                {
                    "doc_id": row.get("doc_id"),
                    "bm25_score": row.get("bm25_score"),
                    "vector_score": row.get("vector_score"),
                    "hybrid_score": row.get("hybrid_score"),
                }
                for row in results
            ]
            st.dataframe(rows, use_container_width=True)
        except requests.RequestException as exc:
            st.error(f"Failed to call search API: {exc}")


def _load_query_logs(db_path: Path) -> list[dict[str, Any]]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        tables = [row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        for table in tables:
            rows = conn.execute(f"SELECT * FROM {table}").fetchall()
            if not rows:
                continue
            return [dict(row) for row in rows]
        return []
    finally:
        conn.close()


def _extract_kpis(logs: list[dict[str, Any]]) -> dict[str, Any]:
    latency_values: list[float] = []
    queries: list[str] = []
    zero_result_queries: list[str] = []

    for row in logs:
        query = str(row.get("query") or row.get("query_text") or "").strip()
        if query:
            queries.append(query)

        latency = (
            row.get("latency_ms")
            or row.get("response_time_ms")
            or row.get("duration_ms")
            or row.get("latency")
        )
        parsed_latency = _safe_float(latency)
        if parsed_latency is not None:
            latency_values.append(parsed_latency)

        result_count = row.get("result_count") or row.get("results_count") or row.get("num_results")
        parsed_result_count = _safe_float(result_count)
        if query and parsed_result_count is not None and parsed_result_count == 0:
            zero_result_queries.append(query)

    latency_values.sort()

    def percentile(vals: list[float], p: float) -> float | None:
        if not vals:
            return None
        index = int(round((len(vals) - 1) * p))
        return vals[max(0, min(index, len(vals) - 1))]

    return {
        "p50_latency": percentile(latency_values, 0.50),
        "p95_latency": percentile(latency_values, 0.95),
        "request_volume": len(logs),
        "top_queries": Counter(queries).most_common(10),
        "zero_result_queries": Counter(zero_result_queries).most_common(10),
    }


def render_kpi_page() -> None:
    st.header("KPI")
    db_path = _existing_path(DEFAULT_SQLITE_PATHS)
    if db_path is None:
        st.info("No SQLite query log database found.")
        return

    try:
        logs = _load_query_logs(db_path)
    except sqlite3.Error as exc:
        st.error(f"Failed to read SQLite logs: {exc}")
        return

    if not logs:
        st.info("No query logs available.")
        return

    kpis = _extract_kpis(logs)
    col1, col2, col3 = st.columns(3)
    col1.metric("p50 latency (ms)", f"{kpis['p50_latency']:.2f}" if kpis["p50_latency"] is not None else "N/A")
    col2.metric("p95 latency (ms)", f"{kpis['p95_latency']:.2f}" if kpis["p95_latency"] is not None else "N/A")
    col3.metric("request volume", str(kpis["request_volume"]))

    st.subheader("Top queries")
    st.dataframe([{"query": q, "count": c} for q, c in kpis["top_queries"]], use_container_width=True)

    st.subheader("Zero-result queries")
    st.dataframe([{"query": q, "count": c} for q, c in kpis["zero_result_queries"]], use_container_width=True)


def _load_experiments(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = [dict(row) for row in reader]
    return rows


def render_evaluation_page() -> None:
    st.header("Evaluation")
    rows = _load_experiments(METRICS_CSV_PATH)
    if not rows:
        st.info("No experiment metrics found.")
        return

    trend_points = []
    for row in rows:
        ts = row.get("timestamp") or ""
        ndcg = _safe_float(row.get("ndcg@10"))
        if ndcg is None:
            continue
        trend_points.append({"timestamp": ts, "ndcg@10": ndcg})

    st.subheader("nDCG@10 trend")
    if trend_points:
        st.line_chart({"ndcg@10": [point["ndcg@10"] for point in trend_points]})
        st.caption("Points ordered by row sequence in experiments.csv")
    else:
        st.info("No valid nDCG@10 values found.")

    st.subheader("Experiments")
    st.dataframe(rows, use_container_width=True)


def render_debug_page() -> None:
    st.header("Debug")
    error_log_path = _existing_path(DEFAULT_ERROR_LOG_PATHS)
    if error_log_path is None:
        st.info("No error log file found.")
        return

    st.write(f"Showing logs from: `{error_log_path}`")
    content = error_log_path.read_text(encoding="utf-8", errors="replace")
    lines = content.splitlines()[-200:]
    st.code("\n".join(lines) if lines else "(empty log file)", language="text")


def main() -> None:
    st.set_page_config(page_title="Hybrid Search Dashboard", layout="wide")
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Search", "KPI", "Evaluation", "Debug"])

    st.sidebar.caption(f"API: {API_BASE_URL}")
    st.title("Hybrid Search Dashboard")

    if page == "Search":
        render_search_page()
    elif page == "KPI":
        render_kpi_page()
    elif page == "Evaluation":
        render_evaluation_page()
    else:
        render_debug_page()


if __name__ == "__main__":
    main()
