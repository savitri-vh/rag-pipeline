"""
End-to-end smoke test: real document -> chunk -> real transformer embedding
-> FAISS index -> query -> retrieval (no LLM call, since no API key is set
in CI, so generation uses the extractive fallback).

Requires network access to download the embedding model from the Hugging
Face Hub on first run — this is expected to work in GitHub Actions (unlike
some sandboxed/offline dev environments).
"""

from rag.pipeline import RAGPipeline


def test_ingest_and_query_returns_relevant_passage():
    pipeline = RAGPipeline(allow_hashing_fallback=False)

    pipeline.ingest_raw(
        "doc_a",
        "Retrieval-Augmented Generation combines a retriever with a language "
        "model so answers are grounded in retrieved passages instead of "
        "relying only on what the model memorized during training.",
    )
    pipeline.ingest_raw(
        "doc_b",
        "The capital of France is Paris, a city known for the Eiffel Tower "
        "and the Louvre museum.",
    )

    result = pipeline.query("What does RAG combine to ground its answers?", top_k=2)

    assert result.passages, "expected at least one retrieved passage"
    top_passage = result.passages[0]
    assert top_passage.chunk.doc_id == "doc_a"
    assert "Sources:" in str(result)
