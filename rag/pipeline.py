"""
Top-level orchestration: wires together loader -> chunker -> embedder ->
vector_store -> retriever -> generator into a single `RAGPipeline` class
with two entry points:

    pipeline.ingest(paths)       # steps 1-4, builds the index
    pipeline.query("...")        # step 5-6, retrieve + generate + cite

Also supports persisting/reloading the FAISS index so ingestion only
needs to run once.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .chunker import chunk_documents
from .embedder import Embedder
from .generator import generate_answer, format_sources, RetrievedPassage
from .loader import Document, load_directory, load_pdf_file, load_text_file, load_raw_string
from .retriever import Retriever
from .vector_store import VectorStore


@dataclass
class RAGAnswer:
    query: str
    answer: str
    passages: List[RetrievedPassage]

    def __str__(self) -> str:
        return f"{self.answer}\n\n{format_sources(self.passages)}"


class RAGPipeline:
    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 150,
        overlap: int = 30,
        llm_model: str = "claude-sonnet-4-5",
        allow_hashing_fallback: bool = True,
    ):
        try:
            self.embedder = Embedder(model_name=embedding_model)
        except Exception as e:
            if not allow_hashing_fallback:
                raise
            print(
                f"[pipeline] could not load transformer embedder '{embedding_model}' ({e}); "
                "falling back to a local hashing embedder. Retrieval quality will be lower — "
                "this fallback exists for offline/no-network environments only."
            )
            from .hashing_embedder import HashingEmbedder

            self.embedder = HashingEmbedder(dim=384)
        self.store = VectorStore(dim=self.embedder.dim)
        self.retriever = Retriever(self.embedder, self.store)
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.llm_model = llm_model

    # ---- Steps 1-4: ingestion ----
    def ingest_documents(self, docs: List[Document]) -> int:
        chunks = chunk_documents(docs, chunk_size=self.chunk_size, overlap=self.overlap)
        if not chunks:
            return 0
        embeddings = self.embedder.embed([c.text for c in chunks])
        self.store.add(embeddings, chunks)
        return len(chunks)

    def ingest_directory(self, dir_path: str) -> int:
        docs = load_directory(dir_path)
        return self.ingest_documents(docs)

    def ingest_files(self, paths: List[str]) -> int:
        docs = []
        for path in paths:
            docs.append(load_pdf_file(path) if path.lower().endswith(".pdf") else load_text_file(path))
        return self.ingest_documents(docs)

    def ingest_raw(self, doc_id: str, text: str, metadata: dict | None = None) -> int:
        return self.ingest_documents([load_raw_string(doc_id, text, metadata)])

    # ---- Steps 5-6: query ----
    def query(self, question: str, top_k: int = 5) -> RAGAnswer:
        passages = self.retriever.retrieve(question, top_k=top_k)
        answer = generate_answer(question, passages, model=self.llm_model)
        return RAGAnswer(query=question, answer=answer, passages=passages)

    # ---- persistence ----
    def save(self, dir_path: str) -> None:
        self.store.save(dir_path)

    def load_store(self, dir_path: str) -> None:
        self.store = VectorStore.load(dir_path)
        self.retriever = Retriever(self.embedder, self.store)
