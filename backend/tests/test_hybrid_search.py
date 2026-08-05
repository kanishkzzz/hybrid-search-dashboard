from backend.app.search.hybrid import HybridSearch


class StubBM25Index:
    def query(self, query: str, top_k: int):
        return [
            {"doc_id": "doc-a", "bm25_score": 2.0},
            {"doc_id": "doc-b", "bm25_score": 1.0},
        ][:top_k]

    def get_document(self, doc_id: str):
        return {
            "doc_id": doc_id,
            "title": f"Title {doc_id}",
            "source": f"{doc_id}.txt",
            "text": f"Text for {doc_id}",
        }


class StubVectorIndex:
    def query(self, query: str, top_k: int):
        return [
            {"doc_id": "doc-b", "vector_score": 0.9},
            {"doc_id": "doc-a", "vector_score": 0.2},
        ][:top_k]

    def get_document(self, doc_id: str):
        return None


def test_hybrid_search_combines_bm25_and_vector_scores() -> None:
    hybrid = HybridSearch(bm25_index=StubBM25Index(), vector_index=StubVectorIndex())

    results = hybrid.search(query="anything", top_k=2, alpha=0.5)

    assert [row["doc_id"] for row in results] == ["doc-a", "doc-b"]

    doc_a = next(row for row in results if row["doc_id"] == "doc-a")
    doc_b = next(row for row in results if row["doc_id"] == "doc-b")

    assert doc_a["bm25_score"] == 1.0
    assert doc_a["vector_score"] == 0.0
    assert doc_a["hybrid_score"] == 0.5

    assert doc_b["bm25_score"] == 0.0
    assert doc_b["vector_score"] == 1.0
    assert doc_b["hybrid_score"] == 0.5


def test_hybrid_search_alpha_weights_sources() -> None:
    hybrid = HybridSearch(bm25_index=StubBM25Index(), vector_index=StubVectorIndex())

    results = hybrid.search(query="anything", top_k=2, alpha=0.8)

    assert [row["doc_id"] for row in results] == ["doc-b", "doc-a"]
    assert results[0]["hybrid_score"] > results[1]["hybrid_score"]


def test_hybrid_search_includes_document_metadata() -> None:
    hybrid = HybridSearch(bm25_index=StubBM25Index(), vector_index=StubVectorIndex())

    results = hybrid.search(query="anything", top_k=2, alpha=0.5)

    assert "title" in results[0]
    assert "source" in results[0]
    assert "snippet" in results[0]
