"""
Step 5/6: Feed retrieved passages + query into an LLM and return a
grounded answer with citations.

Builds a prompt that:
  - numbers each retrieved passage [1], [2], ...
  - instructs the model to answer ONLY from those passages
  - instructs it to cite passage numbers inline after each claim
  - instructs it to say so if the passages don't contain the answer

Uses the Anthropic API when ANTHROPIC_API_KEY is set. Falls back to a
deterministic, extractive "answer" (no LLM call) so the pipeline still
runs end-to-end offline / without an API key — useful for testing the
retrieval quality in isolation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Tuple

from .chunker import Chunk

SYSTEM_PROMPT = (
    "You are a precise research assistant. Answer the user's question using ONLY "
    "the numbered passages provided below. Every factual sentence in your answer "
    "must end with a citation like [1] or [1][3] referencing the passage number(s) "
    "it came from. If the passages do not contain enough information to answer, "
    "say so plainly instead of guessing."
)


@dataclass
class RetrievedPassage:
    rank: int
    chunk: Chunk
    score: float


def build_prompt(query: str, passages: List[RetrievedPassage]) -> str:
    passage_block = "\n\n".join(
        f"[{p.rank}] (source: {p.chunk.doc_id}, chunk {p.chunk.position})\n{p.chunk.text}"
        for p in passages
    )
    return (
        f"Passages:\n{passage_block}\n\n"
        f"Question: {query}\n\n"
        "Answer the question using only the passages above, with inline citations "
        "like [1] after each claim."
    )


def _extractive_fallback(query: str, passages: List[RetrievedPassage]) -> str:
    """No-LLM fallback: surfaces the top passage(s) verbatim-ish with citations,
    so the pipeline is runnable without any API key."""
    if not passages:
        return "No relevant passages were found for this query."
    lines = [f"Based on the retrieved passages, here is what's available on \"{query}\":\n"]
    for p in passages:
        snippet = p.chunk.text.strip()
        if len(snippet) > 400:
            snippet = snippet[:400].rsplit(" ", 1)[0] + "..."
        lines.append(f"- {snippet} [{p.rank}]")
    lines.append(
        "\n(No LLM configured — this is an extractive fallback. Set ANTHROPIC_API_KEY "
        "to generate a synthesized, cited answer instead.)"
    )
    return "\n".join(lines)


def generate_answer(
    query: str,
    passages: List[RetrievedPassage],
    model: str = "claude-sonnet-4-5",
    max_tokens: int = 1024,
) -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return _extractive_fallback(query, passages)

    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    prompt = build_prompt(query, passages)
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(block.text for block in response.content if block.type == "text")


def format_sources(passages: List[RetrievedPassage]) -> str:
    lines = ["Sources:"]
    for p in passages:
        lines.append(f"[{p.rank}] {p.chunk.doc_id} (chunk {p.chunk.position}, score={p.score:.3f})")
    return "\n".join(lines)
