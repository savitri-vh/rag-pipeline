"""
Streamlit demo UI for the RAG pipeline.

Run locally:
    streamlit run app.py

Deploy for free on Streamlit Community Cloud (share.streamlit.io):
    1. Push this file to the root of your GitHub repo (same level as main.py).
    2. On share.streamlit.io, create a new app pointing at this repo/branch,
       with "Main file path" = app.py.
    3. (Optional) In the app's Settings -> Secrets, add:
           ANTHROPIC_API_KEY = "sk-ant-..."
       to get real generated answers instead of the extractive fallback.
"""

from __future__ import annotations

import os
import tempfile

import streamlit as st

from rag.pipeline import RAGPipeline

st.set_page_config(page_title="RAG Pipeline Demo", page_icon="🔎", layout="wide")


@st.cache_resource(show_spinner=False)
def _load_pipeline(data_dir: str) -> RAGPipeline:
    """Build the pipeline once per session and cache it (embedding the
    sample corpus is the slow part, so we don't want to redo it on every
    question)."""
    pipeline = RAGPipeline()
    pipeline.ingest_directory(data_dir)
    return pipeline


def _ensure_pipeline() -> RAGPipeline:
    if "pipeline" not in st.session_state:
        with st.spinner("Loading embedding model and indexing sample documents..."):
            st.session_state.pipeline = _load_pipeline("data")
            st.session_state.indexed_files = sorted(os.listdir("data"))
    return st.session_state.pipeline


st.title("🔎 RAG Pipeline — from scratch")
st.caption(
    "A minimal Retrieval-Augmented Generation pipeline: sentence-transformers "
    "+ FAISS + the Anthropic API, built without LangChain/LlamaIndex so every "
    "stage is visible. [View source on GitHub](https://github.com/savitri-vh/rag-pipeline)"
)

pipeline = _ensure_pipeline()

has_api_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
if not has_api_key:
    st.info(
        "No `ANTHROPIC_API_KEY` is set, so answers below use the pipeline's "
        "**extractive fallback** (retrieved passages, no LLM call) rather than "
        "a generated, synthesized answer. Retrieval still runs at full quality.",
        icon="ℹ️",
    )

with st.sidebar:
    st.header("Corpus")
    st.write("Currently indexed:")
    for name in st.session_state.get("indexed_files", []):
        st.write(f"- `{name}`")

    st.divider()
    st.subheader("Add your own document")
    uploaded = st.file_uploader("Upload a .txt or .md file", type=["txt", "md"])
    if uploaded is not None and st.button("Ingest this file"):
        text = uploaded.read().decode("utf-8", errors="ignore")
        with st.spinner("Chunking + embedding + indexing..."):
            n = pipeline.ingest_raw(uploaded.name, text)
        st.success(f"Indexed {n} new chunks from {uploaded.name}.")
        st.session_state.indexed_files.append(uploaded.name)

    st.divider()
    top_k = st.slider("Passages to retrieve (top_k)", min_value=1, max_value=8, value=4)

question = st.text_input(
    "Ask a question about the indexed documents",
    placeholder="What is Retrieval-Augmented Generation?",
)
ask = st.button("Ask", type="primary")

if ask and question.strip():
    with st.spinner("Retrieving passages and generating an answer..."):
        result = pipeline.query(question, top_k=top_k)

    st.subheader("Answer")
    st.write(result.answer)

    st.subheader("Sources")
    for p in result.passages:
        with st.expander(f"[{p.rank}] {p.chunk.doc_id} — chunk {p.chunk.position} (score={p.score:.3f})"):
            st.write(p.chunk.text)
elif ask:
    st.warning("Type a question first.")

st.divider()
with st.expander("Sample questions to try"):
    st.markdown(
        "- What is Retrieval-Augmented Generation and why does it reduce hallucination?\n"
        "- Who founded Anthropic and what is its flagship product?\n"
        "- What are the four main stages of a RAG pipeline?"
    )