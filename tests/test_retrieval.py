"""Hybrid search and answer assembly, with the vector store and LLM stubbed out."""

from langchain_core.documents import Document

from src.generation import llm_chain
from src.retrieval import hybrid_search
from src.retrieval.hybrid_search import HybridSearcher

DOCS = [
    Document(page_content="The deposit is two months of rent, paid before the key handover.", metadata={"source": "a.pdf", "page": 1}),
    Document(page_content="Pets are allowed with written permission from the landlord.", metadata={"source": "a.pdf", "page": 2}),
    Document(page_content="Service costs cover cleaning of shared stairwells.", metadata={"source": "b.pdf", "page": 1}),
    Document(page_content="Rent is indexed every July using the CPI.", metadata={"source": "b.pdf", "page": 3}),
]


def fake_semantic(scores):
    def _search(query, k=5):
        out = []
        for i, s in sorted(scores.items(), key=lambda x: -x[1])[:k]:
            d = DOCS[i].model_copy(deep=True)
            d.metadata["relevance_score"] = s
            out.append(d)
        return out

    return _search


def test_alpha_zero_is_pure_bm25(monkeypatch):
    monkeypatch.setattr(hybrid_search, "similarity_search", fake_semantic({2: 0.99}))
    top = HybridSearcher(DOCS, alpha=0.0).search("deposit rent", k=1)[0]
    assert "deposit" in top.page_content


def test_alpha_one_is_pure_semantic(monkeypatch):
    monkeypatch.setattr(hybrid_search, "similarity_search", fake_semantic({2: 0.99, 0: 0.1}))
    top = HybridSearcher(DOCS, alpha=1.0).search("deposit rent", k=1)[0]
    assert "stairwells" in top.page_content


def test_blend_uses_both_signals(monkeypatch):
    # doc 3 is decent on both, doc 0 only on keywords, doc 2 only semantically
    monkeypatch.setattr(hybrid_search, "similarity_search", fake_semantic({3: 0.8, 2: 0.9, 0: 0.1}))
    results = HybridSearcher(DOCS, alpha=0.5).search("rent indexed", k=4)
    assert "indexed" in results[0].page_content
    for r in results:
        m = r.metadata
        assert abs(m["hybrid_score"] - round(0.5 * m["semantic_score"] + 0.5 * m["bm25_score"], 4)) < 1e-3


def test_semantic_candidates_wider_than_k(monkeypatch):
    seen = {}

    def spy(query, k=5):
        seen["k"] = k
        return []

    monkeypatch.setattr(hybrid_search, "similarity_search", spy)
    HybridSearcher(DOCS).search("rent", k=3)
    assert seen["k"] >= 12


def test_ask_question_hybrid_builds_searcher_from_store(monkeypatch):
    monkeypatch.setattr(llm_chain, "all_documents", lambda: DOCS)
    monkeypatch.setattr(hybrid_search, "similarity_search", fake_semantic({0: 0.9}))
    captured = {}
    monkeypatch.setattr(llm_chain, "get_llm_response", lambda prompt: captured.setdefault("p", prompt) and "Two months.")
    out = llm_chain.ask_question("How big is the deposit?", search_type="hybrid", k=2)
    assert out["answer"] == "Two months."
    assert out["sources"][0]["source"] == "a.pdf"
    assert "[Source 1: a.pdf, Page 1]" in captured["p"]


def test_ask_question_with_empty_store(monkeypatch):
    monkeypatch.setattr(llm_chain, "all_documents", lambda: [])
    monkeypatch.setattr(llm_chain, "get_llm_response", lambda prompt: "I don't know.")
    out = llm_chain.ask_question("anything", search_type="hybrid")
    assert out["sources"] == []


def test_tokenizer_strips_punctuation():
    assert hybrid_search.tokenize("Rent, deposit. (CPI)") == ["rent", "deposit", "cpi"]


class _Resp:
    def __init__(self, status, content=None, text=""):
        self.status_code, self._content, self.text = status, content, text

    def json(self):
        return {"choices": [{"message": {"content": self._content}}]}


def test_router_falls_back_when_model_is_retired(monkeypatch):
    calls = []
    replies = iter([_Resp(410, text="deprecated"), _Resp(200, "<think>hmm</think>Two months.")])

    def fake_post(url, headers, json, timeout):
        calls.append(json["model"])
        return next(replies)

    monkeypatch.setattr(llm_chain.config, "HF_MODELS", ["old/model", "new/model"])
    monkeypatch.setattr(llm_chain.requests, "post", fake_post)
    assert llm_chain._hf_router_response("q") == "Two months."
    assert calls == ["old/model", "new/model"]


def test_router_stops_on_auth_error(monkeypatch):
    calls = []

    def fake_post(url, headers, json, timeout):
        calls.append(json["model"])
        return _Resp(401, text="unauthorized")

    monkeypatch.setattr(llm_chain.config, "HF_MODELS", ["a", "b"])
    monkeypatch.setattr(llm_chain.requests, "post", fake_post)
    assert "HUGGINGFACEHUB_API_TOKEN" in llm_chain._hf_router_response("q")
    assert calls == ["a"]
