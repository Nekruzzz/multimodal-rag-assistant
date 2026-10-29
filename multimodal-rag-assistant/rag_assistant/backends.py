"""Two interchangeable vector-search backends.

* ``LangChainFaissBackend``: LangChain + a FAISS vector store.
* ``LlamaIndexBackend``: LlamaIndex ``VectorStoreIndex`` (in-memory vector store).

Both take LangChain ``Document`` objects and return ``Hit`` records, so the rest
of the app does not care which one is used.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


@dataclass
class Hit:
    text: str
    score: float  # cosine similarity, higher is better
    metadata: dict = field(default_factory=dict)

    @property
    def source(self) -> str:
        return self.metadata.get("source", "unknown")

    @property
    def is_image(self) -> bool:
        return self.metadata.get("type") == "image"


class Backend(Protocol):
    def build(self, docs: list[Document]) -> None: ...
    def search(self, query: str, k: int) -> list[Hit]: ...


class LangChainFaissBackend:
    def __init__(self, embeddings: Embeddings):
        self.embeddings = embeddings
        self.store = None

    def build(self, docs: list[Document]) -> None:
        from langchain_community.vectorstores import FAISS

        self.store = FAISS.from_documents(docs, self.embeddings)

    def search(self, query: str, k: int = 4) -> list[Hit]:
        results = self.store.similarity_search_with_score(query, k=k)
        # FAISS returns squared L2 distance. For unit-length vectors that equals 2 - 2*cosine.
        return [Hit(d.page_content, 1.0 - float(dist) / 2.0, dict(d.metadata)) for d, dist in results]


class LlamaIndexBackend:
    def __init__(self, embeddings: Embeddings):
        self.embeddings = embeddings
        self.index = None

    def _embed_model(self):
        from llama_index.core.embeddings import BaseEmbedding

        lc = self.embeddings

        class _Wrapper(BaseEmbedding):
            def _get_query_embedding(self, query: str) -> list[float]:
                return lc.embed_query(query)

            async def _aget_query_embedding(self, query: str) -> list[float]:
                return lc.embed_query(query)

            def _get_text_embedding(self, text: str) -> list[float]:
                return lc.embed_documents([text])[0]

        return _Wrapper()

    def build(self, docs: list[Document]) -> None:
        from llama_index.core import Document as LIDocument
        from llama_index.core import VectorStoreIndex

        li_docs = [LIDocument(text=d.page_content, metadata=dict(d.metadata)) for d in docs]
        self.index = VectorStoreIndex.from_documents(li_docs, embed_model=self._embed_model())

    def search(self, query: str, k: int = 4) -> list[Hit]:
        retriever = self.index.as_retriever(similarity_top_k=k)
        return [
            Hit(n.node.get_content(), float(n.score or 0.0), dict(n.node.metadata))
            for n in retriever.retrieve(query)
        ]


def make_backend(name: str, embeddings: Embeddings) -> Backend:
    if name == "langchain":
        return LangChainFaissBackend(embeddings)
    if name == "llamaindex":
        return LlamaIndexBackend(embeddings)
    raise ValueError(f"Unknown backend {name!r}; use 'langchain' or 'llamaindex'")
