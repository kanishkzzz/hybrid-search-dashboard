from __future__ import annotations

import json
import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, TYPE_CHECKING

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from app.search.hybrid import HybridSearch

DEFAULT_DOCS_PATH = Path(__file__).resolve().parents[3] / "data" / "processed" / "docs.jsonl"
DEFAULT_QUERY_LOG_DB_PATH = Path(__file__).resolve().parents[3] / "data" / "logs" / "query_logs.db"


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=10, ge=1)
    alpha: float = Field(default=0.5, ge=0.0, le=1.0)


class QueryLogger:
    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._lock = Lock()
        self._initialize_db()

    def _initialize_db(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS query_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query_text TEXT NOT NULL,
                    latency_ms REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    result_count INTEGER NOT NULL
                )
                """
            )
            conn.commit()

    def log_search(self, query_text: str, latency_ms: float, result_count: int) -> None:
        with self._lock:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO query_logs (query_text, latency_ms, timestamp, result_count)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        query_text,
                        float(latency_ms),
                        datetime.now(timezone.utc).isoformat(),
                        int(result_count),
                    ),
                )
                conn.commit()


class SearchService:
    def __init__(self) -> None:
        self._lock = Lock()
        self._hybrid_search: "HybridSearch" | None = None
        self._documents_count: int = 0

    @property
    def documents_count(self) -> int:
        return self._documents_count

    def _load_documents(self) -> list[dict[str, str]]:
        configured_path = os.getenv("DOCS_JSONL_PATH")
        docs_path = Path(configured_path) if configured_path else DEFAULT_DOCS_PATH

        if not docs_path.exists():
            return []

        documents: list[dict[str, str]] = []
        with docs_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                doc_id = payload.get("doc_id")
                text = payload.get("text", "")
                if not doc_id:
                    continue
                documents.append({"doc_id": str(doc_id), "text": str(text)})

        return documents

    def get_or_create(self) -> "HybridSearch":
        if self._hybrid_search is not None:
            return self._hybrid_search

        with self._lock:
            if self._hybrid_search is not None:
                return self._hybrid_search

            documents = self._load_documents()
            self._documents_count = len(documents)

            from app.search.bm25 import BM25Index
            from app.search.hybrid import HybridSearch
            from app.search.vector_index import VectorIndex

            bm25_index = BM25Index()
            vector_index = VectorIndex()

            if documents:
                bm25_index.build(documents)
                vector_index.build(documents)

            self._hybrid_search = HybridSearch(bm25_index=bm25_index, vector_index=vector_index)
            return self._hybrid_search


app = FastAPI(title="Hybrid Search API", version="1.0")
_service = SearchService()
_query_logger = QueryLogger(Path(os.getenv("QUERY_LOG_DB", str(DEFAULT_QUERY_LOG_DB_PATH))))

_metrics: dict[str, Any] = {
    "request_count": 0,
    "latency_ms": {"total": 0.0, "avg": 0.0, "min": None, "max": None},
}
_metrics_lock = Lock()


@app.middleware("http")
async def metrics_middleware(request, call_next):  # type: ignore[no-untyped-def]
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    with _metrics_lock:
        _metrics["request_count"] += 1
        latency = _metrics["latency_ms"]
        latency["total"] += elapsed_ms
        latency["avg"] = latency["total"] / _metrics["request_count"]
        latency["min"] = elapsed_ms if latency["min"] is None else min(latency["min"], elapsed_ms)
        latency["max"] = elapsed_ms if latency["max"] is None else max(latency["max"], elapsed_ms)

    return response


@app.on_event("startup")
def startup() -> None:
    # Warm service container lazily (indexes are built on first /search call).
    _service.get_or_create()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": "1.0"}


@app.post("/search")
def search(payload: SearchRequest) -> dict[str, Any]:
    start = time.perf_counter()
    result_count = 0

    try:
        hybrid = _service.get_or_create()
        if _service.documents_count == 0:
            return {"query": payload.query, "results": []}

        try:
            results = hybrid.search(query=payload.query, top_k=payload.top_k, alpha=payload.alpha)
        except ValueError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        result_count = len(results)
        return {"query": payload.query, "results": results}
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        _query_logger.log_search(query_text=payload.query, latency_ms=elapsed_ms, result_count=result_count)


@app.get("/metrics")
def metrics() -> dict[str, Any]:
    with _metrics_lock:
        return {
            "request_count": _metrics["request_count"],
            "latency_ms": {
                "avg": _metrics["latency_ms"]["avg"],
                "min": _metrics["latency_ms"]["min"],
                "max": _metrics["latency_ms"]["max"],
            },
        }
