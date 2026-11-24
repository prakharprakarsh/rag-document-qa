"""Embedding model wrapper using Sentence Transformers."""

import config


def get_embedding_model():
    """Initialize and return the embedding model.

    all-MiniLM-L6-v2 by default: 384-dim, small enough to run on CPU in a free
    HF Space. Imported lazily so tests don't need torch.
    """
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )