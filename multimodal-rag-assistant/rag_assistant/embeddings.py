"""Embedding models behind one LangChain-compatible interface.

* ``HashingEmbeddings``: deterministic bag-of-words hashing. No downloads, no GPU.
  Good for tests, demos and CI. It matches on shared words, not on meaning.
* ``get_embeddings("sentence-transformers/all-MiniLM-L6-v2")``: a real semantic
  model from the Hugging Face Hub (needs internet on first use).
"""
from __future__ import annotations

import hashlib
import math
import re

from langchain_core.embeddings import Embeddings

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how", "in", "is",
    "it", "of", "on", "or", "that", "the", "to", "what", "when", "which", "with",
}
_TOKEN = re.compile(r"[a-z0-9]+")


def _stem(tok: str) -> str:
    """Very light suffix stripping so 'probes'/'probe' and 'installed'/'install' match."""
    for suffix in ("ing", "ed", "es", "s"):
        if len(tok) > len(suffix) + 3 and tok.endswith(suffix):
            return tok[: -len(suffix)]
    return tok


class HashingEmbeddings(Embeddings):
    def __init__(self, dim: int = 512):
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for tok in _TOKEN.findall(text.lower()):
            if tok in _STOPWORDS:
                continue
            tok = _stem(tok)
            h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
            sign = 1.0 if (h >> 1) & 1 else -1.0
            vec[h % self.dim] += sign
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def get_embeddings(name: str = "hashing") -> Embeddings:
    """Return ``HashingEmbeddings`` or a Hugging Face sentence-transformers model."""
    if name == "hashing":
        return HashingEmbeddings()
    from langchain_community.embeddings import HuggingFaceEmbeddings  # needs sentence-transformers

    return HuggingFaceEmbeddings(model_name=name, encode_kwargs={"normalize_embeddings": True})
