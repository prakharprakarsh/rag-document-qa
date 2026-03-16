"""Text chunking strategies with comparison support."""

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter,
)
from langchain_core.documents import Document

import config


def get_splitter(
    strategy: str = "recursive",
    chunk_size: int = config.CHUNK_SIZE,
    chunk_overlap: int = config.CHUNK_OVERLAP,
):
    """Return a text splitter based on the chosen strategy.

    Strategies:
        - "recursive" (default): Splits on paragraphs, then sentences, then words.
          Best for most documents.
        - "character": Splits on a single separator (double newline).
          Good for well-structured documents.
        - "token": Splits based on token count.
          Best when you need precise token-level control.
    """
    splitters = {
        "recursive": RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        ),
        "character": CharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separator="\n\n",
        ),
        "token": TokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        ),
    }
    if strategy not in splitters:
        raise ValueError(f"Unknown strategy '{strategy}'. Choose from: {list(splitters.keys())}")
    return splitters[strategy]


def chunk_documents(
    documents: list[Document],
    strategy: str = "recursive",
    chunk_size: int = config.CHUNK_SIZE,
    chunk_overlap: int = config.CHUNK_OVERLAP,
) -> list[Document]:
    """Split documents into chunks using the specified strategy."""
    splitter = get_splitter(strategy, chunk_size, chunk_overlap)
    chunks = splitter.split_documents(documents)

    # Add chunk metadata
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
        chunk.metadata["chunk_strategy"] = strategy
        chunk.metadata["chunk_size"] = chunk_size

    return chunks


def compare_strategies(
    documents: list[Document],
    chunk_size: int = config.CHUNK_SIZE,
) -> dict[str, dict]:
    """Compare different chunking strategies — useful for the README."""
    results = {}
    for strategy in ["recursive", "character", "token"]:
        chunks = chunk_documents(documents, strategy=strategy, chunk_size=chunk_size)
        lengths = [len(c.page_content) for c in chunks]
        results[strategy] = {
            "num_chunks": len(chunks),
            "avg_length": sum(lengths) / len(lengths) if lengths else 0,
            "min_length": min(lengths) if lengths else 0,
            "max_length": max(lengths) if lengths else 0,
        }
    return results