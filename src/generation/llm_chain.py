"""LLM chain for generating answers from retrieved context."""

import requests
import config
from src.generation.prompts import RAG_PROMPT
from src.retrieval.vector_store import similarity_search
from src.retrieval.hybrid_search import HybridSearcher
from langchain_core.documents import Document


def get_llm_response(prompt: str) -> str:
    """Get a response from the LLM."""
    if config.LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model_name=config.LLM_MODEL_NAME or "gpt-4o-mini",
            temperature=config.LLM_TEMPERATURE,
            max_tokens=config.LLM_MAX_TOKENS,
            api_key=config.OPENAI_API_KEY,
        )
        answer = llm.invoke(prompt)
        return answer.content if hasattr(answer, "content") else str(answer)
    else:
        headers = {
            "Authorization": f"Bearer {config.HUGGINGFACEHUB_API_TOKEN}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": "mistralai/Mistral-7B-Instruct-v0.3",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": config.LLM_MAX_TOKENS,
            "temperature": max(config.LLM_TEMPERATURE, 0.1),
        }
        try:
            response = requests.post(
                "https://router.huggingface.co/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=120,
            )
            if response.status_code == 503:
                return "The model is loading, please try again in 30 seconds."
            if response.status_code != 200:
                return f"Model returned status {response.status_code}: {response.text[:200]}"
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except Exception as e:
            return f"Could not get response from model: {str(e)}"


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


def get_llm():
    """Placeholder for compatibility."""
    return None


def ask_question(
    question: str,
    search_type: str = "hybrid",
    hybrid_searcher: HybridSearcher | None = None,
    k: int = config.TOP_K,
) -> dict:
    """Ask a question and get an answer with sources."""
    if search_type == "hybrid" and hybrid_searcher is not None:
        retrieved_docs = hybrid_searcher.search(question, k=k)
    else:
        retrieved_docs = similarity_search(question, k=k)

    context = format_context(retrieved_docs)
    prompt = RAG_PROMPT.format(context=context, question=question)
    answer_text = get_llm_response(prompt)

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