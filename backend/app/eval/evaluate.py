from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_DOCS_PATH = Path(__file__).resolve().parents[3] / "data" / "processed" / "docs.jsonl"
DEFAULT_METRICS_PATH = Path(__file__).resolve().parents[3] / "data" / "metrics" / "experiments.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate hybrid search metrics.")
    parser.add_argument("--queries", required=True, help="Path to queries JSONL file")
    parser.add_argument("--qrels", required=True, help="Path to qrels JSON file")
    parser.add_argument("--docs", default=str(DEFAULT_DOCS_PATH), help="Path to docs JSONL corpus")
    parser.add_argument("--alpha", type=float, default=0.5, help="Hybrid alpha weight (0..1)")
    parser.add_argument("--top-k", type=int, default=10, help="Top-k cutoff for retrieval/metrics")
    parser.add_argument(
        "--out",
        default=str(DEFAULT_METRICS_PATH),
        help="CSV path for appending experiment metrics",
    )
    return parser.parse_args()


def load_queries(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            query_id = str(payload.get("query_id") or payload.get("qid") or payload.get("id") or "")
            query_text = str(payload.get("query") or payload.get("text") or "")
            if not query_id or not query_text:
                continue
            rows.append({"query_id": query_id, "query": query_text})
    return rows


def load_qrels(path: Path) -> dict[str, dict[str, float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    qrels: dict[str, dict[str, float]] = {}

    for query_id, labels in payload.items():
        if isinstance(labels, dict):
            qrels[str(query_id)] = {str(doc_id): float(rel) for doc_id, rel in labels.items()}
        elif isinstance(labels, list):
            qrels[str(query_id)] = {str(doc_id): 1.0 for doc_id in labels}
        else:
            qrels[str(query_id)] = {}

    return qrels


def load_documents(path: Path) -> list[dict[str, str]]:
    docs: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            doc_id = payload.get("doc_id")
            text = payload.get("text", "")
            if not doc_id:
                continue
            docs.append({"doc_id": str(doc_id), "text": str(text)})
    return docs


def dcg_at_k(results: list[str], rels: dict[str, float], k: int) -> float:
    score = 0.0
    for rank, doc_id in enumerate(results[:k], start=1):
        rel = rels.get(doc_id, 0.0)
        gain = (2**rel - 1) / math.log2(rank + 1)
        score += gain
    return score


def ndcg_at_k(results: list[str], rels: dict[str, float], k: int) -> float:
    actual = dcg_at_k(results, rels, k)
    ideal_docs = sorted(rels.items(), key=lambda item: item[1], reverse=True)
    ideal_ids = [doc_id for doc_id, _ in ideal_docs]
    ideal = dcg_at_k(ideal_ids, rels, k)
    if ideal == 0:
        return 0.0
    return actual / ideal


def recall_at_k(results: list[str], rels: dict[str, float], k: int) -> float:
    relevant_docs = {doc_id for doc_id, rel in rels.items() if rel > 0}
    if not relevant_docs:
        return 0.0
    retrieved = set(results[:k])
    return len(retrieved & relevant_docs) / len(relevant_docs)


def mrr_at_k(results: list[str], rels: dict[str, float], k: int) -> float:
    for rank, doc_id in enumerate(results[:k], start=1):
        if rels.get(doc_id, 0.0) > 0:
            return 1.0 / rank
    return 0.0


def build_hybrid_index(documents: list[dict[str, str]]):
    from backend.app.search.bm25 import BM25Index
    from backend.app.search.hybrid import HybridSearch
    from backend.app.search.vector_index import VectorIndex

    bm25_index = BM25Index()
    vector_index = VectorIndex()
    bm25_index.build(documents)
    vector_index.build(documents)
    return HybridSearch(bm25_index=bm25_index, vector_index=vector_index)


def evaluate(
    queries: list[dict[str, str]],
    qrels: dict[str, dict[str, float]],
    hybrid_search,
    alpha: float,
    top_k: int,
) -> dict[str, float]:
    ndcg_scores: list[float] = []
    recall_scores: list[float] = []
    mrr_scores: list[float] = []

    for item in queries:
        query_id = item["query_id"]
        query_text = item["query"]
        labels = qrels.get(query_id, {})

        results = hybrid_search.search(query=query_text, top_k=top_k, alpha=alpha)
        ranked_doc_ids = [str(row["doc_id"]) for row in results]

        ndcg_scores.append(ndcg_at_k(ranked_doc_ids, labels, top_k))
        recall_scores.append(recall_at_k(ranked_doc_ids, labels, top_k))
        mrr_scores.append(mrr_at_k(ranked_doc_ids, labels, top_k))

    if not queries:
        return {"ndcg@10": 0.0, "recall@10": 0.0, "mrr@10": 0.0}

    return {
        "ndcg@10": sum(ndcg_scores) / len(ndcg_scores),
        "recall@10": sum(recall_scores) / len(recall_scores),
        "mrr@10": sum(mrr_scores) / len(mrr_scores),
    }


def append_metrics(path: Path, alpha: float, metrics: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()

    with path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["timestamp", "alpha", "ndcg@10", "recall@10", "mrr@10"],
        )
        if write_header:
            writer.writeheader()

        writer.writerow(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "alpha": alpha,
                "ndcg@10": metrics["ndcg@10"],
                "recall@10": metrics["recall@10"],
                "mrr@10": metrics["mrr@10"],
            }
        )


def main() -> None:
    args = parse_args()

    if not 0.0 <= args.alpha <= 1.0:
        raise SystemExit("--alpha must be in [0, 1]")
    if args.top_k <= 0:
        raise SystemExit("--top-k must be > 0")

    queries = load_queries(Path(args.queries))
    qrels = load_qrels(Path(args.qrels))
    documents = load_documents(Path(args.docs))

    if not documents:
        raise SystemExit(f"No documents loaded from: {args.docs}")

    hybrid_search = build_hybrid_index(documents)
    metrics = evaluate(queries=queries, qrels=qrels, hybrid_search=hybrid_search, alpha=args.alpha, top_k=args.top_k)

    append_metrics(path=Path(args.out), alpha=args.alpha, metrics=metrics)
    print(json.dumps(metrics))


if __name__ == "__main__":
    main()
