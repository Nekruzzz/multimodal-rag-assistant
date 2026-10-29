"""The end-to-end RAG pipeline: ingest -> index -> retrieve -> answer."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .backends import Hit, make_backend
from .embeddings import get_embeddings
from .ingest import load_corpus
from .llm import generate_answer


@dataclass
class Answer:
    text: str
    hits: list[Hit]

    @property
    def images(self) -> list[str]:
        return [h.metadata["image_path"] for h in self.hits if h.is_image]


class RAGPipeline:
    def __init__(self, backend: str = "langchain", embedding_model: str = "hashing",
                 provider: str = "extractive", k: int = 4):
        self.backend = make_backend(backend, get_embeddings(embedding_model))
        self.provider = provider
        self.k = k

    def index(self, corpus_dir: str | Path, caption_missing: bool = False) -> int:
        docs = load_corpus(corpus_dir, caption_missing=caption_missing)
        if not docs:
            raise ValueError(f"No documents found in {corpus_dir}")
        self.backend.build(docs)
        return len(docs)

    def retrieve(self, question: str) -> list[Hit]:
        return self.backend.search(question, self.k)

    def ask(self, question: str) -> Answer:
        hits = self.retrieve(question)
        return Answer(generate_answer(question, hits, self.provider), hits)
