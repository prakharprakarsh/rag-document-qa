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
    providers = []
    if config.GROQ_API_KEY:
        providers.append(("Groq", "https://api.groq.com/openai/v1/chat/completions", config.GROQ_API_KEY, config.GROQ_MODELS))
    if config.HUGGINGFACEHUB_API_TOKEN:
        providers.append(("Hugging Face", "https://router.huggingface.co/v1/chat/completions", config.HUGGINGFACEHUB_API_TOKEN, config.HF_MODELS))
    if not providers:
        raise LLMUnavailable("no API key set (GROQ_API_KEY or HUGGINGFACEHUB_API_TOKEN)")
    errors = []
    for name, url, key, models in providers:
        try:
            return _chat_completion(url, key, models, prompt)
        except LLMUnavailable as e:
            errors.append(f"{name}: {e}")
    raise LLMUnavailable("; ".join(errors))


class LLMUnavailable(RuntimeError):
    """No configured provider could answer (quota, auth, retired models, network)."""


def _chat_completion(url: str, api_key: str, models: list[str], prompt: str) -> str:
    """OpenAI-style chat completion, trying each model in turn."""
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    last_error = "no models configured"
    for model in models:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": config.LLM_MAX_TOKENS,
            "temperature": max(config.LLM_TEMPERATURE, 0.1),
        }
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=120)
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
        if response.status_code in (401, 402, 403):
            # bad key or no credits: other models on this provider will fail the same way
            raise LLMUnavailable(f"HTTP {response.status_code} ({response.text[:120]})")
        # 400/404/410: model not served or retired; 429/5xx: busy. Try the next one.
        last_error = f"{model} returned {response.status_code}: {response.text[:120]}"
    raise LLMUnavailable(last_error)


def extractive_answer(documents: list[Document], reason: str) -> str:
    """Fallback when no LLM is reachable: show the best-matching passages."""
    if not documents:
        return "No relevant passages found in the uploaded documents."
    lines = [f"_The language model is unavailable ({reason}), so here are the most relevant passages instead:_", ""]
    for i, doc in enumerate(documents[:3], 1):
        src = doc.metadata.get("source", "document")
        page = doc.metadata.get("page", "")
        where = f"{src}, p.{page}" if page not in ("", None, "N/A") else src
        text = " ".join(doc.page_content.split())
        lines.append(f"**{i}. {where}** — {text[:400]}{'…' if len(text) > 400 else ''}")
    return "\n\n".join(lines)


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
    try:
        answer_text = get_llm_response(prompt)
    except LLMUnavailable as e:
        answer_text = extractive_answer(retrieved_docs, str(e)[:160])

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