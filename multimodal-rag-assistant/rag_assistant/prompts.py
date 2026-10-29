"""Prompt templates. Keeping them in one file makes prompt changes easy to review."""
from langchain_core.prompts import ChatPromptTemplate

RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You answer questions using ONLY the numbered context passages.\n"
            "Rules:\n"
            "1. If the passages do not contain the answer, say you don't know.\n"
            "2. Cite passages like [1] or [2] after each claim.\n"
            "3. Passages marked (image) describe a picture; mention the image file when relevant.\n"
            "4. Be concise: at most four sentences.",
        ),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ]
)


def format_context(hits) -> str:
    lines = []
    for i, h in enumerate(hits, start=1):
        kind = " (image)" if h.is_image else ""
        lines.append(f"[{i}] {h.source}{kind}: {h.text}")
    return "\n".join(lines)
