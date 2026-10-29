# Multimodal RAG Assistant

A small retrieval-augmented generation (RAG) app that answers questions over **text files and images**. Images become searchable through their captions, and the answer points back to the source passages and image files.

It is a learning project: a compact, tested implementation of the main RAG building blocks, with two interchangeable retrieval backends so you can compare them.

## What it uses

| Piece | Technology |
|---|---|
| Orchestration, prompts, text splitting | **LangChain** (`langchain-core`, `langchain-text-splitters`) |
| Alternative index and retriever | **LlamaIndex** (`VectorStoreIndex`) |
| Vector database | **FAISS** (via LangChain) and LlamaIndex's in-memory vector store |
| Embeddings | Offline hashing embedder (default) or any **Hugging Face** sentence-transformers model |
| Image captions | Sidecar `.txt` files, or a Hugging Face image-to-text model (BLIP) |
| LLM answers | Offline extractive mode (default) or Claude through `langchain-anthropic` |
| Prompt engineering | One cited-answer prompt in `rag_assistant/prompts.py` |

## How it works

```
data/*.md, data/images/*.png (+ .png.txt captions)
        │  ingest.py: split text into chunks, turn captions into image documents
        ▼
   embeddings.py ──► backends.py (LangChain+FAISS  or  LlamaIndex) ──► top-k hits
                                                                          │
                                       prompts.py + llm.py ◄──────────────┘
                                                │
                                        answer + sources + image paths
```

## Quick start

```bash
pip install -r requirements.txt
python scripts/make_sample_images.py          # creates the two demo images
python -m rag_assistant.cli "When should I start irrigation?" --backend langchain
python -m rag_assistant.cli "Show soil moisture over time" --backend llamaindex
python -m pytest -q
```

Use real models (needs internet the first time):

```bash
pip install sentence-transformers langchain-anthropic
export ANTHROPIC_API_KEY=...
python -m rag_assistant.cli "How do I spot leaf blight?" \
    --embeddings sentence-transformers/all-MiniLM-L6-v2 --llm anthropic
```

## What is tested

`python -m pytest -q` runs 12 tests covering ingestion (text chunks and captioned images), embedding behaviour, retrieval of the right document on **both** backends, image results with file paths, and prompt construction. They use the offline hashing embedder, so no downloads or API keys are needed.

## Limitations

- The default hashing embedder matches **words, not meaning**. It keeps tests and demos fast and offline, but retrieval quality is far below a real embedding model. A first version of the tests failed on a question that used "deep" where the document said "depths", which is exactly this weakness.
- "Multimodal" here means images are retrieved through their **captions**. It does not embed pixels directly (for example with CLIP).
- The Hugging Face embedding and captioning paths and the Claude answer mode are included but were **not run** while building this project, because the build environment had no access to model downloads or API keys.
- The index is rebuilt in memory on every run; there is no persistence. The sample corpus is tiny (3 documents, 2 images).
- `langchain-community` is being sunset by its maintainers. A natural next step is moving to the standalone `langchain-community` replacements such as a dedicated FAISS integration package.

## Project layout

```
rag_assistant/   embeddings, ingest, backends, prompts, llm, pipeline, cli
data/            sample documents and captioned images
scripts/         sample image generator
tests/           pytest suite
```

## About how this was built

This project was written with AI assistance (Claude) as a learning exercise, then tested locally.
