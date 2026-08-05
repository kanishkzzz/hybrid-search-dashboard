from __future__ import annotations

import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi


class BM25Index:
    def __init__(self) -> None:
        self._documents: list[dict[str, str]] = []
        self._tokenized_corpus: list[list[str]] = []
        self._bm25: BM25Okapi | None = None

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    def build(self, documents: list[dict[str, str]]) -> None:
        self._documents = documents
        self._tokenized_corpus = [self._tokenize(doc.get("text", "")) for doc in documents]
        self._bm25 = BM25Okapi(self._tokenized_corpus)

    def query(self, query_text: str, top_k: int) -> list[dict[str, float | str]]:
        if self._bm25 is None:
            raise ValueError("BM25 index is not built. Call build(documents) first.")
        if top_k <= 0:
            return []

        query_tokens = self._tokenize(query_text)
        scores = self._bm25.get_scores(query_tokens)

        ranked = sorted(
            zip(self._documents, scores),
            key=lambda pair: pair[1],
            reverse=True,
        )

        results: list[dict[str, float | str]] = []
        for doc, score in ranked[:top_k]:
            results.append({"doc_id": doc["doc_id"], "bm25_score": float(score)})

        return results

    def get_document(self, doc_id: str) -> dict[str, str] | None:
        return next((doc for doc in self._documents if doc.get("doc_id") == doc_id), None)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump(
                {
                    "documents": self._documents,
                    "tokenized_corpus": self._tokenized_corpus,
                    "bm25": self._bm25,
                },
                fh,
            )

    @classmethod
    def load(cls, path: Path) -> "BM25Index":
        with path.open("rb") as fh:
            payload = pickle.load(fh)

        index = cls()
        index._documents = payload["documents"]
        index._tokenized_corpus = payload["tokenized_corpus"]
        index._bm25 = payload["bm25"]
        return index
