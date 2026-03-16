"""Hybrid search combining BM25 (keyword) and semantic (vector) search."""

from rank_bm25 import BM25Okapi
from langchain_core.documents import Document

import config
from src.retrieval.vector_store import similarity_search


class HybridSearcher:
    """Combines keyword-based (BM25) and semantic (vector) search.

    alpha = 1.0 → pure semantic search
    alpha = 0.0 → pure keyword search
    alpha = 0.5 → equal blend (default)
    """

    def __init__(self, documents: list[Document], alpha: float = config.HYBRID_ALPHA):
        self.documents = documents
        self.alpha = alpha

        # Build BM25 index
        tokenized = [doc.page_content.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized)

    def search(self, query: str, k: int = config.TOP_K) -> list[Document]:
        """Run hybrid search and return top-k results."""
        # BM25 keyword scores
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)

        # Normalize BM25 scores to 0-1
        max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1
        bm25_normalized = {i: s / max_bm25 for i, s in enumerate(bm25_scores)}

        # Semantic search scores
        semantic_results = similarity_search(query, k=k)
        semantic_map = {}
        for doc in semantic_results:
            for i, orig_doc in enumerate(self.documents):
                if doc.page_content == orig_doc.page_content:
                    semantic_map[i] = doc.metadata.get("relevance_score", 0)
                    break

        # Combine scores
        all_indices = set(bm25_normalized.keys()) | set(semantic_map.keys())
        combined = {}
        for idx in all_indices:
            bm25_score = bm25_normalized.get(idx, 0)
            semantic_score = semantic_map.get(idx, 0)
            combined[idx] = (self.alpha * semantic_score) + ((1 - self.alpha) * bm25_score)

        # Sort by combined score and return top-k
        top_indices = sorted(combined, key=combined.get, reverse=True)[:k]
        results = []
        for idx in top_indices:
            doc = self.documents[idx].copy()
            doc.metadata["hybrid_score"] = round(combined[idx], 4)
            doc.metadata["bm25_score"] = round(bm25_normalized.get(idx, 0), 4)
            doc.metadata["semantic_score"] = round(semantic_map.get(idx, 0), 4)
            results.append(doc)

        return results