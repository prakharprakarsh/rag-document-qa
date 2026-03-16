"""Embedding model wrapper using Sentence Transformers."""

from langchain_huggingface import HuggingFaceEmbeddings
import config


def get_embedding_model() -> HuggingFaceEmbeddings:
    """Initialize and return the embedding model.

    Uses sentence-transformers/all-MiniLM-L6-v2 by default.
    This is a small, fast model that produces 384-dim embeddings.
    Good balance of speed and quality for a portfolio project.
    """
    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )