"""
Step 2: Chunk documents into smaller passages.

Uses a sliding-window, token-approximate splitter on whitespace with
configurable chunk size and overlap. Overlap preserves context across
chunk boundaries so an answer-bearing sentence split across two chunks
is still retrievable. Each chunk keeps a back-reference to its source
document and position, which is what lets us cite "doc.pdf, chunk 3"
later.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

from .loader import Document


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    position: int             # index of this chunk within its source doc
    metadata: dict = field(default_factory=dict)


def _split_sentences(text: str) -> List[str]:
    # Lightweight sentence splitter; avoids pulling in a heavy NLP dependency.
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s for s in sentences if s]


def chunk_document(
    doc: Document,
    chunk_size: int = 150,     # approx. words per chunk
    overlap: int = 30,         # approx. words shared between consecutive chunks
) -> List[Chunk]:
    """
    Sentence-aware sliding window: accumulate whole sentences until the
    word budget is hit, then start the next chunk `overlap` words back
    so context isn't lost at the seam.
    """
    sentences = _split_sentences(doc.text)
    if not sentences:
        return []

    chunks: List[Chunk] = []
    current: List[str] = []
    current_len = 0
    position = 0

    def flush():
        nonlocal current, current_len, position
        if not current:
            return
        chunk_text = " ".join(current).strip()
        chunks.append(
            Chunk(
                chunk_id=f"{doc.doc_id}::chunk{position}",
                doc_id=doc.doc_id,
                text=chunk_text,
                position=position,
                metadata={**doc.metadata},
            )
        )
        position += 1

    for sentence in sentences:
        words = sentence.split()
        if current_len + len(words) > chunk_size and current:
            flush()
            # carry the tail of the previous chunk forward for overlap
            tail_words = " ".join(current).split()[-overlap:] if overlap > 0 else []
            current = [" ".join(tail_words)] if tail_words else []
            current_len = len(tail_words)
        current.append(sentence)
        current_len += len(words)

    flush()
    return chunks


def chunk_documents(docs: List[Document], chunk_size: int = 150, overlap: int = 30) -> List[Chunk]:
    all_chunks: List[Chunk] = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc, chunk_size=chunk_size, overlap=overlap))
    return all_chunks
