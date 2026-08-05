# Hybrid Search Dashboard

Build, query, evaluate, and monitor a local hybrid search system that combines lexical BM25 retrieval with semantic FAISS vector search.

## What It Does

Hybrid Search Dashboard turns local `.txt` and `.md` files into searchable chunks, builds both keyword and embedding indexes, exposes retrieval through FastAPI, and gives you a Streamlit dashboard for searching and inspecting metrics.

It is designed as a compact retrieval lab: small enough to understand end to end, but structured like a real search service.

## Features

- Hybrid ranking with BM25 plus SentenceTransformer embeddings.
- FAISS vector index with persisted artifacts.
- Chunked ingestion for longer documents.
- FastAPI search API with health and metrics endpoints.
- Streamlit dashboard for search, KPIs, evaluation, and debug views.
- SQLite query logging.
- Offline evaluation with `nDCG@10`, `Recall@10`, and `MRR@10`.
- Cross-platform startup scripts for PowerShell and Unix-like shells.

## Architecture

```text
data/raw/*.txt, *.md
        |
        v
Ingestion and chunking
        |
        v
data/processed/docs.jsonl
        |
        +--> BM25 lexical index
        |
        +--> FAISS semantic index
                  |
                  v
            HybridSearch
                  |
                  v
            FastAPI backend
                  |
                  v
          Streamlit dashboard
```

The hybrid score is:

```text
hybrid_score = alpha * vector_score + (1 - alpha) * bm25_score
```

`alpha` controls semantic weight:

| Alpha | Behavior |
| --- | --- |
| `0.0` | Pure BM25 keyword search |
| `0.5` | Balanced hybrid search |
| `1.0` | Pure vector search |

## Tech Stack

| Layer | Technology |
| --- | --- |
| API | FastAPI |
| Dashboard | Streamlit |
| Lexical search | rank-bm25 |
| Vector search | FAISS |
| Embeddings | SentenceTransformers |
| Metrics | SQLite, CSV |
| Tests | Pytest |

## Quick Start

### Windows PowerShell

```powershell
.\up.ps1
```

### macOS/Linux/Git Bash

```bash
./up.sh
```

The startup scripts install dependencies, ingest raw documents, start the FastAPI backend, and launch the Streamlit dashboard.

## Manual Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Ingest documents:

```powershell
python -m backend.app.ingest --input data/raw --out data/processed
```

Start the API:

```powershell
uvicorn backend.app.api.main:app --reload
```

Start the dashboard in another terminal:

```powershell
streamlit run frontend/dashboard.py
```

Open:

- API docs: `http://localhost:8000/docs`
- Dashboard: `http://localhost:8501`

## API Usage

Health check:

```bash
curl http://localhost:8000/health
```

Search:

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query":"hybrid search","top_k":5,"alpha":0.5}'
```

Example response shape:

```json
{
  "query": "hybrid search",
  "results": [
    {
      "doc_id": "8de31ba49541240a",
      "title": "Hybrid search combines lexical search and semantic search.",
      "source": "file3.md",
      "snippet": "Hybrid search combines lexical search and semantic search.",
      "bm25_score": 1.0,
      "vector_score": 1.0,
      "hybrid_score": 1.0
    }
  ]
}
```

Metrics:

```bash
curl http://localhost:8000/metrics
```

## Evaluation

Evaluation inputs live in:

```text
data/eval/queries.jsonl
data/eval/qrels.json
```

Run evaluation:

```powershell
python -m backend.app.eval.evaluate `
  --queries data/eval/queries.jsonl `
  --qrels data/eval/qrels.json `
  --alpha 0.5 `
  --top-k 10
```

Results are appended to:

```text
data/metrics/experiments.csv
```

## Project Layout

```text
backend/
  app/
    api/          FastAPI application and request handling
    db/           SQLite query logging and metric summaries
    eval/         Offline retrieval evaluation
    ingest/       Raw document ingestion and chunking
    search/       BM25, FAISS, and hybrid ranking
  tests/          Unit tests

data/
  raw/            Source documents
  processed/      Generated JSONL corpus
  indexes/        Generated BM25 and FAISS artifacts
  metrics/        Query logs and experiment metrics
  eval/           Evaluation queries and relevance labels

frontend/
  dashboard.py    Streamlit UI

docs/
  architecture.md Detailed system architecture
```

## Generated Artifacts

The app generates these files during ingestion and API startup:

```text
data/processed/docs.jsonl
data/indexes/bm25.pkl
data/indexes/faiss.index
data/indexes/vector_documents.json
data/metrics/queries.db
```

Index artifacts are rebuilt when `docs.jsonl` is newer than the saved indexes.

## Testing

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Expected result:

```text
5 passed
```

## Configuration

Environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `DOCS_JSONL_PATH` | `data/processed/docs.jsonl` | Corpus loaded by the API |
| `API_BASE_URL` | `http://localhost:8000` | Backend URL used by Streamlit |
| `QUERY_LOG_DB` | `data/metrics/queries.db` | SQLite query log path for dashboard KPIs |

## Troubleshooting

If the dashboard cannot reach the API, confirm FastAPI is running at `http://localhost:8000`.

If KPI data is empty, run at least one search first. Query logs are created after `/search` requests.

If startup is slow on the first run, SentenceTransformers may be downloading or loading the embedding model. Later runs reuse saved index artifacts when they are fresh.
