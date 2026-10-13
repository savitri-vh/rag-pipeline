"""Unit tests for the chunker — pure logic, no network/model needed."""

from rag.chunker import chunk_document
from rag.loader import Document


def test_chunk_document_respects_approx_size():
    text = " ".join(f"This is sentence number {i}." for i in range(100))
    doc = Document(doc_id="test.txt", text=text)
    chunks = chunk_document(doc, chunk_size=50, overlap=10)

    assert len(chunks) > 1
    for chunk in chunks:
        word_count = len(chunk.text.split())
        # allow slack since we flush on whole sentences, not mid-sentence
        assert word_count <= 80


def test_chunk_document_preserves_doc_id_and_position():
    doc = Document(doc_id="mydoc.txt", text="Sentence one. Sentence two. Sentence three.")
    chunks = chunk_document(doc, chunk_size=5, overlap=1)

    assert all(c.doc_id == "mydoc.txt" for c in chunks)
    positions = [c.position for c in chunks]
    assert positions == sorted(positions)
    assert positions == list(range(len(chunks)))


def test_chunk_document_overlap_shares_words():
    # Multiple short sentences so the chunker has real split points.
    text = " ".join(f"Word{i} marks sentence {i}." for i in range(30))
    doc = Document(doc_id="overlap.txt", text=text)
    chunks = chunk_document(doc, chunk_size=20, overlap=5)

    assert len(chunks) >= 2
    tail_of_first = set(chunks[0].text.split()[-5:])
    head_of_second = set(chunks[1].text.split()[:10])
    assert tail_of_first & head_of_second


def test_empty_document_produces_no_chunks():
    doc = Document(doc_id="empty.txt", text="   ")
    assert chunk_document(doc) == []
