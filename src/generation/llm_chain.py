"""LLM chain for generating answers from retrieved context."""

import config
from src.generation.prompts import RAG_PROMPT
from src.retrieval.vector_store import similarity_search
from src.retrieval.hybrid_search import HybridSearcher
from langchain_core.documents import Document


def get_llm():
    """Initialize the LLM based on configuration."""
    if config.LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model_name=config.LLM_MODEL_NAME or "gpt-4o-mini",
            temperature=config.LLM_TEMPERATURE,
            max_tokens=config.LLM_MAX_TOKENS,
            api_key=config.OPENAI_API_KEY,
        )
    else:
        from langchain_huggingface import HuggingFaceEndpoint
        return HuggingFaceEndpoint(
            repo_id=config.LLM_MODEL_NAME,
            task="text-generation",
            temperature=config.LLM_TEMPERATURE,
            max_new_tokens=config.LLM_MAX_TOKENS,
            huggingfacehub_api_token=config.HUGGINGFACEHUB_API_TOKEN,
        )


def format_context(documents: list[Document]) -> str:
    """Format retrieved documents into a context string."""
    context_parts = []
    for i, doc in enumerate(documents, 1):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "N/A")
        context_parts.append(
            f"[Source {i}: {source}, Page {page}]\n{doc.page_content}"
        )
    return "\n\n---\n\n".join(context_parts)


def ask_question(
    question: str,
    search_type: str = "hybrid",
    hybrid_searcher: HybridSearcher | None = None,
    k: int = config.TOP_K,
) -> dict:
    """Ask a question and get an answer with sources.

    Returns:
        dict with keys: answer, sources, context_documents
    """
    # Retrieve relevant documents
    if search_type == "hybrid" and hybrid_searcher is not None:
        retrieved_docs = hybrid_searcher.search(question, k=k)
    else:
        retrieved_docs = similarity_search(question, k=k)

    # Format context
    context = format_context(retrieved_docs)

    # Generate answer
    llm = get_llm()
    prompt = RAG_PROMPT.format(context=context, question=question)
    answer = llm.invoke(prompt)

    # Handle different return types
    if hasattr(answer, "content"):
        answer_text = answer.content
    else:
        answer_text = str(answer)

    # Extract sources
    sources = []
    for doc in retrieved_docs:
        sources.append({
            "source": doc.metadata.get("source", "Unknown"),
            "page": doc.metadata.get("page", "N/A"),
            "relevance_score": doc.metadata.get("relevance_score", 0),
            "chunk_preview": doc.page_content[:200] + "...",
        })

    return {
        "answer": answer_text,
        "sources": sources,
        "context_documents": retrieved_docs,
    }