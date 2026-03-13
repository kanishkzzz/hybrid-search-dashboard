from app.search.bm25 import BM25Index


def test_bm25_ranking_order_with_three_documents() -> None:
    documents = [
        {"doc_id": "doc-1", "text": "apple banana apple"},
        {"doc_id": "doc-2", "text": "banana orange"},
        {"doc_id": "doc-3", "text": "apple"},
    ]

    index = BM25Index()
    index.build(documents)

    results = index.query("apple banana", top_k=3)

    doc_ids = [result["doc_id"] for result in results]

    assert "doc-1" in doc_ids
    assert "doc-2" in doc_ids
    assert "doc-3" in doc_ids
    assert all("bm25_score" in result for result in results)
    assert results[0]["bm25_score"] >= results[1]["bm25_score"] >= results[2]["bm25_score"]
