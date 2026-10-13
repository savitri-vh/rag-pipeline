"""Unit tests for the offline hashing embedder fallback — no network needed."""

import numpy as np

from rag.hashing_embedder import HashingEmbedder


def test_embed_returns_correct_shape():
    embedder = HashingEmbedder(dim=128)
    vectors = embedder.embed(["hello world", "another sentence here"])
    assert vectors.shape == (2, 128)


def test_embeddings_are_normalized():
    embedder = HashingEmbedder(dim=64)
    vectors = embedder.embed(["some text to embed"])
    norm = np.linalg.norm(vectors[0])
    assert abs(norm - 1.0) < 1e-5


def test_identical_text_gives_identical_embedding():
    embedder = HashingEmbedder(dim=64)
    v1 = embedder.embed_query("the quick brown fox")
    v2 = embedder.embed_query("the quick brown fox")
    assert np.allclose(v1, v2)


def test_similar_text_scores_higher_than_dissimilar():
    embedder = HashingEmbedder(dim=256)
    query = embedder.embed_query("Retrieval-Augmented Generation reduces hallucination")
    close = embedder.embed_query("RAG reduces hallucination in generation")
    far = embedder.embed_query("The weather today is sunny and warm")

    sim_close = float(np.dot(query, close))
    sim_far = float(np.dot(query, far))
    assert sim_close > sim_far
