"""Command line interface.

Examples:
    python -m rag_assistant.cli "When should I irrigate?" --data data
    python -m rag_assistant.cli "Show soil moisture over time" --backend llamaindex
"""
from __future__ import annotations

import argparse

from .pipeline import RAGPipeline


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Ask questions over text files and captioned images.")
    p.add_argument("question")
    p.add_argument("--data", default="data", help="folder with .md/.txt files and images")
    p.add_argument("--backend", choices=["langchain", "llamaindex"], default="langchain")
    p.add_argument("--embeddings", default="hashing",
                   help="'hashing' (offline) or a Hugging Face model, e.g. sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--llm", choices=["extractive", "anthropic"], default="extractive")
    p.add_argument("--caption-missing", action="store_true",
                   help="caption images that have no .txt sidecar using a Hugging Face model")
    p.add_argument("-k", type=int, default=4, help="passages to retrieve")
    a = p.parse_args(argv)

    pipe = RAGPipeline(a.backend, a.embeddings, a.llm, a.k)
    n = pipe.index(a.data, a.caption_missing)
    answer = pipe.ask(a.question)

    print(f"Indexed {n} chunks with the {a.backend} backend.\n")
    print("Answer:", answer.text, "\n")
    print("Sources:")
    for i, h in enumerate(answer.hits, start=1):
        print(f"  [{i}] {h.source}{' (image)' if h.is_image else ''}  score={h.score:.3f}")
    if answer.images:
        print("Related images:", ", ".join(answer.images))


if __name__ == "__main__":
    main()
