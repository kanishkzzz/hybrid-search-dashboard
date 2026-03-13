from __future__ import annotations


class HybridSearch:
    def __init__(self, bm25_index, vector_index) -> None:
        self._bm25_index = bm25_index
        self._vector_index = vector_index

    @staticmethod
    def _min_max_normalize(scores_by_doc_id: dict[str, float]) -> dict[str, float]:
        if not scores_by_doc_id:
            return {}

        min_score = min(scores_by_doc_id.values())
        max_score = max(scores_by_doc_id.values())

        if max_score == min_score:
            return {doc_id: 1.0 for doc_id in scores_by_doc_id}

        return {
            doc_id: (score - min_score) / (max_score - min_score)
            for doc_id, score in scores_by_doc_id.items()
        }

    def search(self, query: str, top_k: int, alpha: float) -> list[dict[str, float | str]]:
        bm25_results = self._bm25_index.query(query, top_k)
        vector_results = self._vector_index.query(query, top_k)

        bm25_scores = {row["doc_id"]: float(row["bm25_score"]) for row in bm25_results}
        vector_scores = {row["doc_id"]: float(row["vector_score"]) for row in vector_results}

        bm25_scores = self._min_max_normalize(bm25_scores)
        vector_scores = self._min_max_normalize(vector_scores)

        all_doc_ids = sorted(set(bm25_scores) | set(vector_scores))

        combined: list[dict[str, float | str]] = []
        for doc_id in all_doc_ids:
            bm25_score = bm25_scores.get(doc_id, 0.0)
            vector_score = vector_scores.get(doc_id, 0.0)
            hybrid_score = alpha * bm25_score + (1 - alpha) * vector_score
            combined.append(
                {
                    "doc_id": doc_id,
                    "bm25_score": bm25_score,
                    "vector_score": vector_score,
                    "hybrid_score": hybrid_score,
                }
            )

        combined.sort(key=lambda row: (-row["hybrid_score"], row["doc_id"]))
        return combined[:top_k]
