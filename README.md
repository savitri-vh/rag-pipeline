# RAG Pipeline (from scratch)

A minimal, modular Retrieval-Augmented Generation pipeline built with
`sentence-transformers` + `FAISS` + the Anthropic API. No LangChain /
LlamaIndex — every stage is plain Python so you can see and modify exactly
what's happening.

## Pipeline stages

| Step | Module | What it does |
|---|---|---|
| 1. Load | `rag/loader.py` | Reads `.txt`/`.md`/`.pdf` files (or raw strings) into `Document` objects |
| 2. Chunk | `rag/chunker.py` | Sentence-aware sliding-window splitter with configurable size + overlap |
| 3. Embed | `rag/embedder.py` | Encodes chunks with a transformer embedding model (`all-MiniLM-L6-v2` by default) |
| 4. Store | `rag/vector_store.py` | FAISS `IndexFlatIP` (cosine similarity via normalized vectors) + chunk metadata |
| 5. Retrieve | `rag/retriever.py` | Embeds the query, does top-k nearest-neighbor search |
| 6. Generate | `rag/generator.py` | Builds a numbered-passage prompt, calls Claude, returns a cited answer (or an extractive fallback with no API key) |
| orchestration | `rag/pipeline.py` | `RAGPipeline` ties it all together: `.ingest_directory()` + `.query()` |

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-...   # optional — without it you get extractive fallback answers
```

## Run the demo

```bash
python main.py                              # ingests ./data, runs 3 sample questions
python main.py --data_dir /path/to/docs      # ingest your own files
python main.py --query "What is RAG?"        # single question
python main.py --interactive                 # REPL
```

## Use it as a library

```python
from rag.pipeline import RAGPipeline

pipeline = RAGPipeline()
pipeline.ingest_directory("data")            # steps 1-4
# or: pipeline.ingest_files(["doc1.pdf", "notes.txt"])
# or: pipeline.ingest_raw("scrape_1", "some scraped text...")

result = pipeline.query("What is RAG?", top_k=4)   # steps 5-6
print(result)          # answer + "Sources:" block
print(result.answer)   # just the generated text
print(result.passages) # list[RetrievedPassage] with chunk + similarity score

# Persist the index so you don't have to re-embed every run
pipeline.save("index/")
pipeline2 = RAGPipeline()
pipeline2.load_store("index/")
```

## Design notes

- **Embeddings are L2-normalized**, so FAISS's `IndexFlatIP` (inner product)
  is equivalent to cosine similarity — simple and exact. For corpora beyond
  ~1M chunks, swap in `faiss.IndexIVFFlat` or `IndexHNSWFlat` for approximate
  search; the `VectorStore` interface (`add`/`search`/`save`/`load`) doesn't
  change.
- **Chunking is sentence-aware**, not a hard character cutoff, so chunks
  don't end mid-sentence. Overlap (default 30 words) keeps context that
  spans a chunk boundary retrievable.
- **Citations are enforced via the prompt**, not by post-processing: the
  system prompt requires every claim to end with `[n]`, and each passage is
  numbered in the context block. `format_sources()` then prints what `[n]`
  maps to (doc id, chunk position, similarity score).
- **Works without an API key**: `generator.py` falls back to an extractive
  mode that surfaces the raw top passages with citations, so you can
  validate retrieval quality independent of the LLM.

## Tuning knobs

- `chunk_size` / `overlap` in `RAGPipeline(...)` — larger chunks keep more
  context per passage but dilute embedding specificity; more overlap reduces
  boundary loss at the cost of redundant storage.
- `top_k` in `pipeline.query(...)` — more passages improve recall but add
  noise/cost to the generation prompt.
- `embedding_model` — swap to `BAAI/bge-base-en-v1.5` or `intfloat/e5-base-v2`
  for higher retrieval quality at higher latency/memory cost.
