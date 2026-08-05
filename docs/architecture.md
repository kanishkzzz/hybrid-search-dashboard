# Hybrid Search Dashboard - System Architecture

## Overview

This project is a local hybrid search system. It ingests raw text files, turns them into searchable chunks, builds lexical and semantic indexes, exposes search through FastAPI, and displays search and metrics in Streamlit.

Hybrid retrieval combines two signals:

- BM25 lexical search for exact keyword matching.
- FAISS vector search for semantic similarity using SentenceTransformers.

## High-Level Flow

```text
data/raw/*.txt, *.md
        |
        v
backend/app/ingest
        |
        v
data/processed/docs.jsonl
        |
        +--> BM25 index ----------+
        |                         |
        +--> FAISS vector index --+--> HybridSearch
                                      |
                                      v
                              FastAPI backend
                                      |
                                      v
                              Streamlit dashboard
```

## Main Components

### Ingestion

Location: `backend/app/ingest/`

The ingestion pipeline reads `.txt` and `.md` files from `data/raw`, normalizes whitespace, chunks longer files, and writes JSONL records to `data/processed/docs.jsonl`.

Each record includes:

- `doc_id`
- `parent_id`
- `chunk_index`
- `title`
- `text`
- `source`
- `created_at`

Single-chunk files keep the same stable `doc_id` as the source file hash. Multi-chunk files use `parent_id:chunk_index`.

### BM25 Search

Location: `backend/app/search/bm25.py`

BM25 provides lexical retrieval. Text is lowercased and tokenized with a simple alphanumeric analyzer before ranking with `rank_bm25`.

The BM25 artifact can be saved to:

```text
data/indexes/bm25.pkl
```

### Vector Search

Location: `backend/app/search/vector_index.py`

Vector search embeds chunk text with `all-MiniLM-L6-v2`, normalizes embeddings, and stores them in a FAISS inner-product index.

Artifacts are saved to:

```text
data/indexes/faiss.index
data/indexes/vector_documents.json
```

### Hybrid Search

Location: `backend/app/search/hybrid.py`

Hybrid search runs both BM25 and vector retrieval, normalizes each score stream, unions the candidate document IDs, and computes:

```text
hybrid_score = alpha * vector_score + (1 - alpha) * bm25_score
```

`alpha` is the semantic/vector weight:

```text
0.0 = pure BM25
0.5 = balanced hybrid
1.0 = pure vector search
```

Results include document metadata and a snippet, not only scores.

### FastAPI Backend

Location: `backend/app/api/main.py`

Endpoints:

```text
GET  /
GET  /health
POST /search
GET  /metrics
```

The API loads `data/processed/docs.jsonl`, reuses fresh persisted indexes when available, and rebuilds indexes when documents are newer than the index artifacts.

Search requests are logged to SQLite through `backend/app/db/queries.py`.

### Metrics

Query logs are stored in:

```text
data/metrics/queries.db
```

Offline evaluation metrics are appended to:

```text
data/metrics/experiments.csv
```

The API `/metrics` endpoint includes in-memory request latency plus query-log summaries.

### Evaluation

Location: `backend/app/eval/evaluate.py`

The evaluation harness loads:

```text
data/eval/queries.jsonl
data/eval/qrels.json
```

It computes:

- `nDCG@10`
- `Recall@10`
- `MRR@10`

### Streamlit Dashboard

Location: `frontend/dashboard.py`

Pages:

- Search: sends queries to FastAPI and displays title, source, snippet, and scores.
- KPI: reads query logs from `data/metrics/queries.db`.
- Evaluation: displays experiment metrics from `data/metrics/experiments.csv`.
- Debug: shows local error logs if present.

## Startup

Unix-like environments:

```bash
./up.sh
```

Windows PowerShell:

```powershell
.\up.ps1
```
