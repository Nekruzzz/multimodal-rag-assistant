"""Load text files and images into LangChain ``Document`` objects.

Images are made searchable through their captions. A caption comes from a
sidecar file (``chart.png`` -> ``chart.png.txt``) or, if none exists and
``caption_missing=True``, from a Hugging Face image-to-text model.
"""
from __future__ import annotations

import warnings
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

TEXT_EXTS = {".md", ".txt"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def caption_image(path: Path, model: str = "Salesforce/blip-image-captioning-base") -> str:
    """Caption an image with a Hugging Face model (downloads the model on first use)."""
    from transformers import pipeline

    captioner = pipeline("image-to-text", model=model)
    return captioner(str(path))[0]["generated_text"]


def load_text_documents(root: Path, chunk_size: int = 500, chunk_overlap: int = 60) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs: list[Document] = []
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in TEXT_EXTS:
            continue
        if Path(path.stem).suffix.lower() in IMAGE_EXTS:
            continue  # "chart.png.txt" is an image caption; load_image_documents handles it
        text = path.read_text(encoding="utf-8")
        base = Document(page_content=text, metadata={"source": str(path.relative_to(root)), "type": "text"})
        docs.extend(splitter.split_documents([base]))
    return docs


def load_image_documents(root: Path, caption_missing: bool = False) -> list[Document]:
    docs: list[Document] = []
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in IMAGE_EXTS:
            continue
        sidecar = path.with_name(path.name + ".txt")
        if sidecar.exists():
            caption = sidecar.read_text(encoding="utf-8").strip()
        elif caption_missing:
            caption = caption_image(path)
        else:
            warnings.warn(f"No caption for {path}; skipping (add {sidecar.name} or enable captioning)")
            continue
        docs.append(
            Document(
                page_content=caption,
                metadata={"source": str(path.relative_to(root)), "type": "image", "image_path": str(path)},
            )
        )
    return docs


def load_corpus(root: str | Path, caption_missing: bool = False) -> list[Document]:
    root = Path(root)
    return load_text_documents(root) + load_image_documents(root, caption_missing)
