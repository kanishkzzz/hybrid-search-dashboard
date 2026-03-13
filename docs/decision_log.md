# Technical Decision Log

This document records major design decisions made during development.

---

## 1. Hybrid Search (BM25 + Vector Search)

Decision:
Use hybrid retrieval combining BM25 and vector similarity.

Reason:
BM25 handles exact keyword matching while vector search captures semantic meaning.

Tradeoff:
Running two retrieval pipelines increases complexity but improves retrieval quality.

---

## 2. FAISS for Vector Search

Decision:
Use FAISS as the vector index.

Reason:
- fast nearest neighbor search
- CPU friendly
- easy integration with Python

Alternative considered:
ChromaDB / Elasticsearch

---

## 3. FastAPI for Backend API

Decision:
Use FastAPI to expose the search service.

Reason:
- high performance
- automatic OpenAPI documentation
- simple integration with Python services

---

## 4. Streamlit for Dashboard

Decision:
Use Streamlit to build the monitoring dashboard.

Reason:
- minimal frontend code
- fast prototyping
- easy data visualization

Alternative considered:
React dashboard (rejected due to time complexity).

---

## 5. SQLite for Query Logging

Decision:
Store query logs in SQLite.

Reason:
- lightweight
- no external database setup required
- sufficient for local analytics

---

## 6. Pytest for Testing

Decision:
Use Pytest for unit tests.

Reason:
- simple test structure
- good ecosystem
- widely used in Python projects