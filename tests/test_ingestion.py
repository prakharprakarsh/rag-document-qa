"""Loading and chunking."""

import pytest
from langchain_core.documents import Document


def _tiktoken_available() -> bool:
    try:
        import tiktoken

        tiktoken.get_encoding("gpt2")
        return True
    except Exception:
        return False


needs_tiktoken = pytest.mark.skipif(not _tiktoken_available(), reason="tiktoken vocab not downloadable offline")

from src.ingestion.chunker import chunk_documents, compare_strategies, get_splitter
from src.ingestion.loader import load_document

TEXT = "\n\n".join(
    f"Section {i}. The rental contract requires a deposit of {i} months. " * 8 for i in range(1, 9)
)


@pytest.fixture
def doc():
    return [Document(page_content=TEXT, metadata={"source": "contract.txt"})]


def test_txt_loader_keeps_text(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("hello world\nsecond line")
    docs = load_document(path)
    assert docs and "second line" in docs[0].page_content


def test_unknown_extension_rejected(tmp_path):
    path = tmp_path / "x.xyz"
    path.write_text("data")
    with pytest.raises(ValueError):
        load_document(path)


@pytest.mark.parametrize("strategy", ["recursive", "character", pytest.param("token", marks=needs_tiktoken)])
def test_chunks_respect_size_and_carry_metadata(doc, strategy):
    chunks = chunk_documents(doc, strategy=strategy, chunk_size=300, chunk_overlap=30)
    assert len(chunks) > 1
    assert all(c.metadata["source"] == "contract.txt" for c in chunks)
    assert [c.metadata["chunk_id"] for c in chunks] == list(range(len(chunks)))
    if strategy == "recursive":
        assert max(len(c.page_content) for c in chunks) <= 300


def test_compare_strategies_reports_all_three(doc):
    stats = compare_strategies(doc, chunk_size=300, strategies=("recursive", "character"))
    assert set(stats) == {"recursive", "character"}
    assert all(s["num_chunks"] > 0 for s in stats.values())


def test_unknown_strategy():
    with pytest.raises(ValueError):
        get_splitter("semantic")
