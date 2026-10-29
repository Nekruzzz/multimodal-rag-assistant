from pathlib import Path

import pytest

from rag_assistant import RAGPipeline
from rag_assistant.embeddings import HashingEmbeddings
from rag_assistant.ingest import load_corpus
from rag_assistant.llm import generate_answer
from rag_assistant.prompts import RAG_PROMPT, format_context

DATA = Path(__file__).resolve().parent.parent / "data"
BACKENDS = ["langchain", "llamaindex"]


def test_corpus_contains_text_chunks_and_captioned_images():
    docs = load_corpus(DATA)
    kinds = [d.metadata["type"] for d in docs]
    assert kinds.count("image") == 2
    assert kinds.count("text") >= 3
    assert not any(d.metadata["source"].endswith(".png.txt") for d in docs)  # sidecars are not text docs


def test_hashing_embeddings_are_unit_length_and_deterministic():
    e = HashingEmbeddings()
    a, b = e.embed_query("soil moisture"), e.embed_query("soil moisture")
    assert a == b
    assert sum(x * x for x in a) == pytest.approx(1.0)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "question, expected_source",
    [
        ("At what soil moisture should irrigation start?", "irrigation.md"),
        ("What do blight lesions on leaves look like?", "crop_disease.md"),
        ("At what depth should soil moisture probes be installed?", "sensors.md"),
    ],
)
def test_retrieval_finds_the_right_document(backend, question, expected_source):
    pipe = RAGPipeline(backend=backend, k=3)
    pipe.index(DATA)
    sources = [h.source for h in pipe.retrieve(question)]
    assert expected_source in sources[:2]


@pytest.mark.parametrize("backend", BACKENDS)
def test_image_results_are_returned_with_paths(backend):
    pipe = RAGPipeline(backend=backend, k=4)
    pipe.index(DATA)
    answer = pipe.ask("chart of soil moisture over seven days dropping below threshold")
    assert any(p.endswith("soil_moisture_week.png") for p in answer.images)


def test_prompt_numbers_context_and_marks_images():
    pipe = RAGPipeline(k=3)
    pipe.index(DATA)
    hits = pipe.retrieve("leaf blight lesions yellow halo")
    ctx = format_context(hits)
    assert ctx.startswith("[1] ")
    rendered = RAG_PROMPT.format_messages(context=ctx, question="What is blight?")
    assert "ONLY the numbered context" in rendered[0].content
    assert "What is blight?" in rendered[1].content


def test_no_hits_means_dont_know():
    assert "don't know" in generate_answer("anything", []).lower()
