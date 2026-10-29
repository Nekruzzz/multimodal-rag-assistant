"""Answer generation.

* ``extractive`` (default): no model needed. Returns the best-matching passages.
* ``anthropic``: uses a Claude chat model through ``langchain-anthropic``.
  Needs ``ANTHROPIC_API_KEY`` and ``pip install langchain-anthropic``.
"""
from __future__ import annotations

import os

from langchain_core.output_parsers import StrOutputParser

from .prompts import RAG_PROMPT, format_context


def generate_answer(question: str, hits, provider: str = "extractive") -> str:
    if not hits:
        return "I don't know: nothing relevant was retrieved."
    if provider == "extractive":
        best = hits[0]
        return f"Most relevant passage [1] ({best.source}): {best.text}"
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        model = ChatAnthropic(model=os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5"), temperature=0)
        chain = RAG_PROMPT | model | StrOutputParser()
        return chain.invoke({"context": format_context(hits), "question": question})
    raise ValueError(f"Unknown provider {provider!r}; use 'extractive' or 'anthropic'")
