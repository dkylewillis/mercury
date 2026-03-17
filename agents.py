"""
Two-agent pipeline for Mercury: query expansion and question answering.

Query expansion transforms a user question into search keyphrases optimised
for semantic / embedding search.  The answering agent uses retrieved chunks
to produce a grounded answer.

Models are independently configurable via environment variables:

    MERCURY_QUERY_MODEL   Model for query expansion  (default: claude-3-5-haiku-latest)
    MERCURY_ANSWER_MODEL  Model for answering        (default: claude-3-5-haiku-latest)

Both require ANTHROPIC_API_KEY to be set.
"""

import os
from typing import List

from dotenv import load_dotenv
import anthropic

from models import QueryResult

load_dotenv()

QUERY_MODEL = os.environ.get("MERCURY_QUERY_MODEL", "claude-3-haiku-20240307")
ANSWER_MODEL = os.environ.get("MERCURY_ANSWER_MODEL", "claude-sonnet-4-20250514")

_QUERY_SYSTEM = """\
You are a query expansion specialist for a document search system that uses
semantic / embedding similarity (sentence-transformers).

Task: Transform the user's question into 6-12 effective search keyphrases that
will retrieve the most relevant chunks from a vector database.

Rules:
- Output ONLY comma-separated keyphrases. No explanations, no numbering.
- Include synonyms, abbreviations, and related technical terms.
- Rephrase the question from multiple angles to maximise recall.
- Keep each keyphrase concise (1-5 words).
"""

_ANSWER_SYSTEM = """\
You are a precise question-answering assistant. You answer the user's question
using ONLY the provided search results. Do not use any outside knowledge.

Rules:
- Reference sources using [N] notation where N is the source number.
- If the search results do not contain enough information, say so explicitly.
- Be concise and accurate.
"""


def _client() -> anthropic.Anthropic:
    return anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env


def expand_query(question: str) -> str:
    """Return comma-separated search keyphrases for *question*."""
    client = _client()
    message = client.messages.create(
        model=QUERY_MODEL,
        max_tokens=150,
        system=[
            {
                "type": "text",
                "text": _QUERY_SYSTEM,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": question}],
    )
    return message.content[0].text.strip()


def _format_context(results: List[QueryResult]) -> str:
    """Build a numbered context block from search results."""
    parts: list[str] = []
    for i, r in enumerate(results, 1):
        meta = f"[Source {i}] Document: {r.chunk.document_name}"
        if r.chunk.page_number is not None:
            meta += f", Page {r.chunk.page_number}"
        if r.chunk.headings:
            meta += f" | Headings: {' > '.join(r.chunk.headings)}"
        text = r.chunk.text
        if r.context:
            text = "\n".join(c.text for c in r.context)
        parts.append(f"{meta}\n{text}")
    return "\n\n---\n\n".join(parts)


def answer_question(question: str, results: List[QueryResult]) -> str:
    """Answer *question* using only the provided search *results*."""
    context = _format_context(results)
    client = _client()
    message = client.messages.create(
        model=ANSWER_MODEL,
        max_tokens=1024,
        system=[
            {
                "type": "text",
                "text": _ANSWER_SYSTEM,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Search results:\n\n{context}",
                        "cache_control": {"type": "ephemeral"},
                    },
                    {
                        "type": "text",
                        "text": f"Question: {question}",
                    },
                ],
            }
        ],
    )
    return message.content[0].text.strip()
