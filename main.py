"""
End-to-end demo / CLI for the RAG pipeline.

Usage:
    python main.py                          # ingest ./data, run sample queries
    python main.py --data_dir path/to/docs  # ingest a custom directory
    python main.py --query "your question"  # ask one question after ingesting
    python main.py --interactive            # drop into a REPL after ingesting

Set ANTHROPIC_API_KEY in the environment to get real generated, synthesized
answers. Without it, the pipeline still runs and returns an extractive
fallback answer built directly from the retrieved passages, so you can
sanity-check retrieval quality offline.
"""

from __future__ import annotations

import argparse
import sys

from rag.pipeline import RAGPipeline

SAMPLE_QUESTIONS = [
    "What is Retrieval-Augmented Generation and why does it reduce hallucination?",
    "Who founded Anthropic and what is its flagship product?",
    "What are the four main stages of a RAG pipeline?",
]


def main():
    parser = argparse.ArgumentParser(description="Run the RAG pipeline end to end.")
    parser.add_argument("--data_dir", default="data", help="Directory of .txt/.md/.pdf files to ingest")
    parser.add_argument("--top_k", type=int, default=4, help="Number of passages to retrieve per query")
    parser.add_argument("--query", default=None, help="Ask a single question instead of running samples")
    parser.add_argument("--interactive", action="store_true", help="Drop into a REPL after ingesting")
    args = parser.parse_args()

    print(f"[1-4] Loading + chunking + embedding + indexing documents from '{args.data_dir}' ...")
    pipeline = RAGPipeline()
    num_chunks = pipeline.ingest_directory(args.data_dir)
    print(f"      Indexed {num_chunks} chunks.\n")

    if num_chunks == 0:
        print("No documents were ingested. Add .txt/.md/.pdf files to the data directory and retry.")
        sys.exit(1)

    def ask(question: str):
        print(f"Q: {question}")
        result = pipeline.query(question, top_k=args.top_k)
        print(f"\n{result}\n")
        print("-" * 80)

    if args.query:
        ask(args.query)
    elif args.interactive:
        print("Interactive mode. Type a question, or 'quit' to exit.\n")
        while True:
            try:
                q = input("Q: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q or q.lower() in {"quit", "exit"}:
                break
            result = pipeline.query(q, top_k=args.top_k)
            print(f"\n{result}\n")
    else:
        for q in SAMPLE_QUESTIONS:
            ask(q)


if __name__ == "__main__":
    main()
