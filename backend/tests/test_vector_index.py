import numpy as np

from backend.app.search.vector_index import VectorIndex


class FakeSentenceTransformer:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def encode(self, texts, convert_to_numpy=True):
        mapping = {
            "apple banana": [1.0, 0.0, 0.0],
            "banana orange": [0.0, 1.0, 0.0],
            "apple fruit": [0.9, 0.1, 0.0],
            "apple": [1.0, 0.0, 0.0],
        }
        vectors = [mapping[text] for text in texts]
        return np.array(vectors, dtype=np.float32)


def test_vector_index_retrieves_best_match(monkeypatch) -> None:
    monkeypatch.setattr("app.search.vector_index.SentenceTransformer", FakeSentenceTransformer)

    documents = [
        {"doc_id": "doc-1", "text": "apple banana"},
        {"doc_id": "doc-2", "text": "banana orange"},
        {"doc_id": "doc-3", "text": "apple fruit"},
    ]

    index = VectorIndex()
    index.build(documents)

    results = index.query("apple", top_k=3)

    assert [result["doc_id"] for result in results] == ["doc-1", "doc-3", "doc-2"]
    assert results[0]["vector_score"] >= results[1]["vector_score"] >= results[2]["vector_score"]
