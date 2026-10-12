"""
Step 5 (retrieval half): embed the query, search the vector store,
return the top-k passages ranked by similarity.
"""

from __future__ import annotations

from typing import List

from .embedder import Embedder
from .generator import RetrievedPassage
from .vector_store import VectorStore


class Retriever:
    def __init__(self, embedder: Embedder, store: VectorStore):
        self.embedder = embedder
        self.store = store

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievedPassage]:
        query_vec = self.embedder.embed_query(query)
        hits = self.store.search(query_vec, top_k=top_k)
        return [
            RetrievedPassage(rank=i + 1, chunk=chunk, score=score)
            for i, (chunk, score) in enumerate(hits)
        ]
