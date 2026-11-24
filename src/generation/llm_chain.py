"""LLM chain for generating answers from retrieved context."""

import os
import re

import requests

import config
from src.generation.prompts import RAG_PROMPT
from src.retrieval.vector_store import all_documents, similarity_search
from src.retrieval.hybrid_search import HybridSearcher
from langchain_core.documents import Document


def get_llm_response(prompt: str) -> str:
    """Get a response from the LLM."""
    if config.LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model_name=os.getenv("LLM_MODEL_NAME", "gpt-4o-mini"),
            temperature=config.LLM_TEMPERATURE,
            max_tokens=config.LLM_MAX_TOKENS,
            api_key=config.OPENAI_API_KEY,
        )
        answer = llm.invoke(prompt)
        return answer.content if hasattr(answer, "content") else str(answer)
    return _hf_router_response(prompt)


def _hf_router_response(prompt: str) -> str:
    """Call the HF router, falling back to the next model if one is unavailable."""
    headers = {
        "Authorization": f"Bearer {config.HUGGINGFACEHUB_API_TOKEN}",
        "Content-Type": "application/json",
    }
    last_error = "no models configured"
    for model in config.HF_MODELS:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
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
        except requests.RequestException as e:
            last_error = f"{model}: {e}"
            continue
        if response.status_code == 200:
            content = response.json()["choices"][0]["message"].get("content") or ""
            # some reasoning models put their chain of thought in <think> tags
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.S).strip()
            if content:
                return content
            last_error = f"{model} returned an empty answer"
            continue
        if response.status_code in (401, 403):
            # bad or missing token: trying other models won't help
            return f"Model returned status {response.status_code}: check the HUGGINGFACEHUB_API_TOKEN secret."
        # 400/404/410: model not served or retired; 429/5xx: busy. Try the next one.
        last_error = f"{model} returned {response.status_code}: {response.text[:200]}"
    return f"Could not get a response from any model. Last error: {last_error}"


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
    """Ask a question and get an answer with sources."""
    if search_type == "hybrid":
        if hybrid_searcher is None:
            docs = all_documents()
            hybrid_searcher = HybridSearcher(docs) if docs else None
        retrieved_docs = hybrid_searcher.search(question, k=k) if hybrid_searcher else []
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
            "relevance_score": doc.metadata.get("hybrid_score", doc.metadata.get("relevance_score", 0)),
            "chunk_preview": doc.page_content[:200] + "...",
        })

    return {
        "answer": answer_text,
        "sources": sources,
        "context_documents": retrieved_docs,
    }