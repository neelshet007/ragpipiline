"""
Hybrid Retrieval Engine Component
Combines Dense Vector Search (Qdrant) and Sparse Lexical Search (BM25)
using Reciprocal Rank Fusion (RRF) and Weighted Score Fusion.
"""

import time
from typing import List, Dict, Any, Optional
from backend.app.retrieval.dense import QdrantDenseRetriever
from backend.app.retrieval.bm25 import BM25Retriever

class HybridRetriever:
    """
    Hybrid Retriever performing Reciprocal Rank Fusion (RRF) & Score Combination
    over Qdrant Dense Vector Search and BM25 Sparse Lexical Search.
    """

    def __init__(self, dense_retriever: QdrantDenseRetriever, bm25_retriever: BM25Retriever):
        self.dense_retriever = dense_retriever
        self.bm25_retriever = bm25_retriever

    def reciprocal_rank_fusion(
        self,
        dense_results: List[Dict[str, Any]],
        sparse_results: List[Dict[str, Any]],
        k: int = 60,
        top_k: int = 5,
        sparse_weight: float = 1.5
    ) -> List[Dict[str, Any]]:
        """
        Calculates Reciprocal Rank Fusion (RRF) score:
        RRF(d) = sum( weight / (k + rank(d)) ) for each retriever list.
        Sparse BM25 is weighted higher (1.5x) to ensure exact keyword matches
        take precedence over noisy fallback vector approximations.
        """
        rrf_scores: Dict[str, float] = {}
        item_payloads: Dict[str, Dict[str, Any]] = {}
        bm25_ranks: Dict[str, int] = {}

        # Process Sparse Ranks first (primary keyword relevance)
        for rank, item in enumerate(sparse_results, start=1):
            chunk_id = item.get("chunk_id") or item.get("document_id") or str(item.get("chunk_text"))
            item_payloads[chunk_id] = item
            bm25_ranks[chunk_id] = rank
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (sparse_weight / (k + rank))

        # Process Dense Ranks
        for rank, item in enumerate(dense_results, start=1):
            chunk_id = item.get("chunk_id") or item.get("document_id") or str(item.get("chunk_text"))
            if chunk_id not in item_payloads:
                item_payloads[chunk_id] = item
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (1.0 / (k + rank))

        # Sort by RRF score descending, breaking ties with BM25 rank
        sorted_ids = sorted(
            rrf_scores.keys(),
            key=lambda cid: (rrf_scores[cid], -bm25_ranks.get(cid, 999)),
            reverse=True
        )[:top_k]

        fused_results = []
        for rank, cid in enumerate(sorted_ids, start=1):
            payload = dict(item_payloads[cid])
            payload["fusion_score"] = round(rrf_scores[cid], 5)
            payload["hybrid_rank"] = rank
            payload["fusion_type"] = "rrf"
            fused_results.append(payload)

        return fused_results

    def convex_score_fusion(
        self,
        dense_results: List[Dict[str, Any]],
        sparse_results: List[Dict[str, Any]],
        alpha: float = 0.5,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Calculates Convex Combination of Min-Max Normalized Scores:
        Score(d) = alpha * Dense_Norm(d) + (1 - alpha) * Sparse_Norm(d)
        """
        # Min-max normalize dense scores
        d_scores = [d.get("score", 0.0) for d in dense_results]
        max_d, min_d = (max(d_scores), min(d_scores)) if d_scores else (1.0, 0.0)
        range_d = max(max_d - min_d, 1e-6)

        # Min-max normalize sparse scores
        s_scores = [s.get("score", 0.0) for s in sparse_results]
        max_s, min_s = (max(s_scores), min(s_scores)) if s_scores else (1.0, 0.0)
        range_s = max(max_s - min_s, 1e-6)

        comb_scores: Dict[str, float] = {}
        item_payloads: Dict[str, Dict[str, Any]] = {}

        for item in dense_results:
            cid = item.get("chunk_id") or item.get("document_id") or str(item.get("chunk_text"))
            item_payloads[cid] = item
            norm_sc = (item.get("score", 0.0) - min_d) / range_d
            comb_scores[cid] = comb_scores.get(cid, 0.0) + (alpha * norm_sc)

        for item in sparse_results:
            cid = item.get("chunk_id") or item.get("document_id") or str(item.get("chunk_text"))
            if cid not in item_payloads:
                item_payloads[cid] = item
            norm_sc = (item.get("score", 0.0) - min_s) / range_s
            comb_scores[cid] = comb_scores.get(cid, 0.0) + ((1.0 - alpha) * norm_sc)

        sorted_ids = sorted(comb_scores.keys(), key=lambda cid: comb_scores[cid], reverse=True)[:top_k]

        fused_results = []
        for rank, cid in enumerate(sorted_ids, start=1):
            payload = dict(item_payloads[cid])
            payload["fusion_score"] = round(comb_scores[cid], 5)
            payload["hybrid_rank"] = rank
            payload["fusion_type"] = "convex"
            fused_results.append(payload)

        return fused_results

    def search(
        self,
        query_text: str,
        query_vector: List[float],
        top_k: int = 5,
        mode: str = "rrf",
        alpha: float = 0.5,
        lang_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Executes hybrid retrieval using both dense vector and sparse lexical search.
        Attaches retrieval diagnostics: bm25_hit (True if BM25 found keyword matches).
        """
        t0 = time.perf_counter()

        # Dense Vector Search
        dense_results = self.dense_retriever.search(
            query_vector=query_vector,
            top_k=top_k * 2,
            lang_filter=lang_filter
        )

        # Sparse Lexical BM25 Search
        sparse_results = self.bm25_retriever.search(
            query=query_text,
            top_k=top_k * 2,
            lang_filter=lang_filter
        )

        # Diagnostic: did BM25 find any keyword match?
        bm25_hit = len(sparse_results) > 0
        bm25_max_score = max((r.get("score", 0.0) for r in sparse_results), default=0.0)

        # Fusion Algorithm
        if mode == "convex":
            fused = self.convex_score_fusion(dense_results, sparse_results, alpha=alpha, top_k=top_k)
        else: # Default RRF
            fused = self.reciprocal_rank_fusion(dense_results, sparse_results, k=60, top_k=top_k)

        latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        for item in fused:
            item["total_hybrid_latency_ms"] = latency_ms
            item["bm25_hit"] = bm25_hit
            item["bm25_max_score"] = round(bm25_max_score, 4)

        return fused
