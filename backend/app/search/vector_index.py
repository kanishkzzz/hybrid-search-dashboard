from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class VectorIndex:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self._model_name = model_name
        self._model: SentenceTransformer | None = None
        self._documents: list[dict[str, str]] = []
        self._index: faiss.IndexFlatIP | None = None

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self._model_name)
        return self._model

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return vectors / norms

    def build(self, documents: list[dict[str, str]]) -> None:
        self._documents = documents
        texts = [doc.get("text", "") for doc in documents]

        model = self._get_model()
        embeddings = model.encode(texts, convert_to_numpy=True).astype("float32")
        embeddings = self._normalize(embeddings)

        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        self._index = index

    def query(self, query_text: str, top_k: int) -> list[dict[str, Any]]:
        if self._index is None:
            raise ValueError("Vector index is not built. Call build(documents) first.")
        if top_k <= 0:
            return []

        model = self._get_model()
        query_embedding = model.encode([query_text], convert_to_numpy=True).astype("float32")
        query_embedding = self._normalize(query_embedding)

        k = min(top_k, len(self._documents))
        scores, indices = self._index.search(query_embedding, k)

        results: list[dict[str, Any]] = []
        for doc_idx, score in zip(indices[0], scores[0]):
            if doc_idx < 0:
                continue
            results.append(
                {
                    "doc_id": self._documents[int(doc_idx)]["doc_id"],
                    "vector_score": float(score),
                }
            )

        return results

    def get_document(self, doc_id: str) -> dict[str, str] | None:
        return next((doc for doc in self._documents if doc.get("doc_id") == doc_id), None)

    def save(self, index_path: Path, documents_path: Path) -> None:
        if self._index is None:
            raise ValueError("Vector index is not built. Call build(documents) first.")

        index_path.parent.mkdir(parents=True, exist_ok=True)
        documents_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(index_path))
        documents_path.write_text(
            json.dumps(
                {
                    "model_name": self._model_name,
                    "documents": self._documents,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, index_path: Path, documents_path: Path) -> "VectorIndex":
        payload = json.loads(documents_path.read_text(encoding="utf-8"))
        index = cls(model_name=payload.get("model_name", "all-MiniLM-L6-v2"))
        index._documents = payload["documents"]
        index._index = faiss.read_index(str(index_path))
        return index
