"""
Step 3: Embed passages (and queries) using a transformer-based embedding model.

Defaults to `sentence-transformers/all-MiniLM-L6-v2` (small, fast, 384-dim,
good baseline quality). Swap `model_name` for a stronger model like
`BAAI/bge-base-en-v1.5` or `intfloat/e5-base-v2` if quality matters more
than latency.
"""

from __future__ import annotations

from typing import List

import numpy as np


class Embedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str | None = None):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        self.dim = self.model.get_sentence_embedding_dimension()

    def embed(self, texts: List[str], batch_size: int = 64, normalize: bool = True) -> np.ndarray:
        """Returns a (len(texts), dim) float32 array, L2-normalized by default
        so that cosine similarity == inner product (what FAISS's IndexFlatIP expects)."""
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
        )
        return embeddings.astype("float32")

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed([query])[0]
