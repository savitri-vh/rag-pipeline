"""
Network-free fallback embedder.

`Embedder` (embedder.py) downloads real transformer weights from the
Hugging Face Hub on first use — the correct choice whenever the
environment has normal internet access. This module exists only for
environments where that download is blocked (e.g. a sandboxed egress
policy with no huggingface.co access): it implements a deterministic,
dependency-free "embedding" using hashed character n-gram term
frequencies, so the rest of the pipeline (chunking, FAISS indexing,
retrieval, citation-grounded generation) can still be exercised and
tested end to end with zero network calls.

It is semantically much weaker than a real sentence embedding model
(no learned synonymy/paraphrase understanding) — swap back to
`Embedder` for anything beyond local testing.
"""

from __future__ import annotations

import hashlib
import re
from typing import List

import numpy as np


class HashingEmbedder:
    def __init__(self, dim: int = 384, ngram_range=(3, 5)):
        self.dim = dim
        self.ngram_range = ngram_range
        self.model_name = f"hashing-ngram-fallback(dim={dim})"

    def _ngrams(self, text: str) -> List[str]:
        text = re.sub(r"\s+", " ", text.lower()).strip()
        grams = []
        for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
            grams.extend(text[i : i + n] for i in range(len(text) - n + 1))
        # also include whole words for a bit of lexical signal
        grams.extend(text.split())
        return grams

    def _vector(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype="float32")
        for gram in self._ngrams(text):
            h = int(hashlib.md5(gram.encode("utf-8")).hexdigest(), 16)
            idx = h % self.dim
            sign = 1.0 if (h // self.dim) % 2 == 0 else -1.0
            vec[idx] += sign
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def embed(self, texts: List[str], batch_size: int = 64, normalize: bool = True) -> np.ndarray:
        return np.stack([self._vector(t) for t in texts]).astype("float32")

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed([query])[0]
