"""ChromaDB vector store management."""

from langchain_core.documents import Document

import config
from src.ingestion.embedder import get_embedding_model


def get_vector_store():
    """Get or create the ChromaDB vector store."""
    from langchain_community.vectorstores import Chroma

    embedding_model = get_embedding_model()
    return Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=embedding_model,
        persist_directory=str(config.CHROMA_DIR),
    )


def add_documents(documents: list[Document]):
    """Add documents to the vector store."""
    from langchain_community.vectorstores import Chroma

    embedding_model = get_embedding_model()
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        collection_name=config.COLLECTION_NAME,
        persist_directory=str(config.CHROMA_DIR),
    )
    print(f"Added {len(documents)} documents to vector store.")
    return vector_store


def similarity_search(query: str, k: int = config.TOP_K) -> list[Document]:
    """Search the vector store for similar documents."""
    vector_store = get_vector_store()
    results = vector_store.similarity_search_with_relevance_scores(query, k=k)
    # Attach scores to metadata
    docs = []
    for doc, score in results:
        doc.metadata["relevance_score"] = round(score, 4)
        docs.append(doc)
    return docs


def all_documents() -> list[Document]:
    """Every stored chunk, used to build the BM25 side of hybrid search."""
    data = get_vector_store().get(include=["documents", "metadatas"])
    return [Document(page_content=d, metadata=m or {}) for d, m in zip(data["documents"], data["metadatas"])]


def clear_vector_store():
    """Delete all documents from the vector store."""
    import chromadb

    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    try:
        client.delete_collection(config.COLLECTION_NAME)
        print("Vector store cleared.")
    except ValueError:
        print("Collection does not exist.")