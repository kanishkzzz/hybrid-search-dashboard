# Hybrid Search Dashboard — System Architecture

## Overview

This project implements a **Hybrid Search System** that combines **lexical search (BM25)** with **semantic vector search**.
The system ingests raw documents, builds search indices, exposes a search API, evaluates retrieval quality, and provides a monitoring dashboard.

Hybrid search improves retrieval quality by combining exact keyword matching with semantic similarity.

---

# High-Level Architecture

```
Raw Documents
      │
      ▼
Ingestion Pipeline
      │
      ▼
Processed Documents (docs.jsonl)
      │
      ├───────────────┐
      │               │
      ▼               ▼
BM25 Index        Vector Index (FAISS)
      │               │
      └─────── Hybrid Search Engine ───────┘
                      │
                      ▼
                FastAPI Backend
                      │
                      ▼
               Streamlit Dashboard
```

---

# System Components

## 1. Document Ingestion

Location:

```
backend/app/ingest/
```

Purpose:

* Read `.txt` and `.md` files from `data/raw`
* Normalize whitespace and text formatting
* Generate document IDs
* Convert documents into structured format

Output:

```
data/processed/docs.jsonl
```

Each document contains:

* `doc_id`
* `title`
* `text`
* `source`
* `created_at`

---

# 2. BM25 Index (Lexical Search)

Location:

```
backend/app/search/bm25.py
```

Purpose:

* Perform **keyword-based retrieval**
* Score documents using **BM25 ranking**

BM25 considers:

* term frequency
* inverse document frequency
* document length normalization

Strength:

* excellent for exact keyword matching

Limitation:

* cannot understand semantic meaning.

---

# 3. Vector Index (Semantic Search)

Location:

```
backend/app/search/vector_index.py
```

Purpose:

* Perform **semantic similarity search**

Pipeline:

1. Encode documents using **SentenceTransformers**
2. Generate dense embeddings
3. Store embeddings in a **FAISS index**
4. Perform nearest neighbor search

Advantages:

* retrieves semantically similar documents
* handles paraphrased queries

Limitation:

* weaker for exact keyword matching.

---

# 4. Hybrid Search Engine

Location:

```
backend/app/search/hybrid.py
```

Purpose:

Combine BM25 and vector search scores.

Formula:

```
hybrid_score = alpha * vector_score + (1 - alpha) * bm25_score
```

Where:

* `alpha` controls weighting between semantic and lexical search.

Example:

| alpha | Behavior           |
| ----- | ------------------ |
| 0.0   | pure BM25          |
| 1.0   | pure vector search |
| 0.5   | balanced hybrid    |

---

# 5. FastAPI Search Service

Location:

```
backend/app/api/main.py
```

Provides REST API endpoints.

Endpoints:

```
GET  /health
POST /search
GET  /metrics
```

Responsibilities:

* receive search queries
* execute hybrid search
* return ranked results
* log search metrics.

---

# 6. Query Logging (Metrics)

Location:

```
backend/app/db/queries.py
```

Stores query analytics in:

```
data/metrics/queries.db
```

Stored fields:

* query
* latency
* result count
* timestamp

Used for dashboard KPIs.

---

# 7. Evaluation Harness

Location:

```
backend/app/eval/evaluate.py
```

Measures search quality using IR metrics.

Metrics:

* **nDCG@10**
* **Recall@10**
* **MRR@10**

Results stored in:

```
data/metrics/experiments.csv
```

---

# 8. Streamlit Dashboard

Location:

```
frontend/dashboard.py
```

Provides a user interface for:

Search Page

* run hybrid queries
* tune alpha parameter

KPI Page

* request volume
* p50 latency
* p95 latency
* top queries
* zero-result queries

Evaluation Page

* visualize experiment metrics

Debug Page

* view logs and errors.

---

# Technology Stack

| Component       | Technology           |
| --------------- | -------------------- |
| Backend API     | FastAPI              |
| Vector Search   | FAISS                |
| Embeddings      | SentenceTransformers |
| Lexical Search  | BM25                 |
| Dashboard       | Streamlit            |
| Metrics Storage | SQLite               |
| Testing         | Pytest               |

---

# Data Flow Summary

```
Raw Files
   ↓
Ingestion
   ↓
docs.jsonl
   ↓
BM25 + Vector Indices
   ↓
Hybrid Search
   ↓
FastAPI API
   ↓
Streamlit Dashboard
```

---

# Deployment

The entire system can be started using:

```
./up.sh
```

This script:

1. creates virtual environment
2. installs dependencies
3. runs ingestion
4. starts FastAPI server
5. launches Streamlit dashboard.
