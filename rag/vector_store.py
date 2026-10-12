"""
Step 4: Store embeddings in a vector database (FAISS).

Wraps a FAISS `IndexFlatIP` (exact inner-product search, equivalent to
cosine similarity since embeddings are L2-normalized). Keeps a parallel
Python list of Chunk metadata so a FAISS row index maps back to the
original chunk text, doc_id, and position for citations.

For large corpora, swap `IndexFlatIP` for an approximate index
(`IndexIVFFlat`, `IndexHNSWFlat`) — the interface below stays the same.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import List, Tuple

import numpy as np

from .chunker import Chunk


class VectorStore:
    def __init__(self, dim: int):
        import faiss

        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.chunks: List[Chunk] = []

    def add(self, embeddings: np.ndarray, chunks: List[Chunk]) -> None:
        assert embeddings.shape[0] == len(chunks), "embeddings/chunks length mismatch"
        assert embeddings.shape[1] == self.dim, "embedding dim mismatch"
        self.index.add(embeddings)
        self.chunks.extend(chunks)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[Chunk, float]]:
        q = query_embedding.reshape(1, -1).astype("float32")
        scores, indices = self.index.search(q, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((self.chunks[idx], float(score)))
        return results

    def save(self, dir_path: str) -> None:
        import faiss

        out = Path(dir_path)
        out.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(out / "index.faiss"))
        with open(out / "chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)

    @classmethod
    def load(cls, dir_path: str) -> "VectorStore":
        import faiss

        inp = Path(dir_path)
        index = faiss.read_index(str(inp / "index.faiss"))
        with open(inp / "chunks.pkl", "rb") as f:
            chunks = pickle.load(f)
        store = cls.__new__(cls)
        store.dim = index.d
        store.index = index
        store.chunks = chunks
        return store
