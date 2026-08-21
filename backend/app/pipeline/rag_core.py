"""
RAG Core Low-Latency Pipeline Component
Integrates Embedder, Qdrant Vector Retriever, BM25 Lexical Retriever, Hybrid RRF Fusion,
and Prompt Context Assembly with strict latency measurement targeting <200 ms total execution.
"""

import os
import sys
import json
import time
from typing import List, Dict, Any, Optional

from backend.app.embeddings.embedder import MultilingualEmbedder
from backend.app.retrieval.dense import QdrantDenseRetriever
from backend.app.retrieval.bm25 import BM25Retriever
from backend.app.retrieval.hybrid import HybridRetriever

class RAGCorePipeline:
    """
    End-to-End Low-Latency RAG Core Engine.
    Exposes query processing, hybrid retrieval, prompt context construction,
    and fast response synthesis with microsecond precision timing.
    """

    def __init__(
        self,
        embedder: Optional[MultilingualEmbedder] = None,
        dense_retriever: Optional[QdrantDenseRetriever] = None,
        bm25_retriever: Optional[BM25Retriever] = None,
        bm25_index_path: str = "data/bm25_index.json"
    ):
        t0 = time.perf_counter()
        print("[+] Initializing RAGCorePipeline...", flush=True)

        self.embedder = embedder or MultilingualEmbedder()
        self.dense_retriever = dense_retriever or QdrantDenseRetriever()

        if bm25_retriever is not None:
            self.bm25_retriever = bm25_retriever
        else:
            self.bm25_retriever = BM25Retriever()
            if os.path.exists(bm25_index_path):
                print(f"[+] Loading BM25 index from '{bm25_index_path}'...", flush=True)
                with open(bm25_index_path, "r", encoding="utf-8") as f:
                    idx_data = json.load(f)
                    self.bm25_retriever.fit(idx_data.get("chunks", []))
            else:
                print(f"[-] BM25 index path '{bm25_index_path}' not found. BM25 search will use fallback.", flush=True)

        self.hybrid_retriever = HybridRetriever(
            dense_retriever=self.dense_retriever,
            bm25_retriever=self.bm25_retriever
        )

        init_ms = (time.perf_counter() - t0) * 1000.0
        print(f"[+] RAGCorePipeline initialized in {init_ms:.2f} ms!", flush=True)

    def process_query(
        self,
        query: str,
        top_k: int = 3,
        fusion_mode: str = "rrf",
        alpha: float = 0.5,
        lang_filter: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes end-to-end RAG query pipeline:
        1. Query Vectorization
        2. Hybrid Dense + BM25 Search (RRF)
        3. Context Payload Assembly
        4. Answer Synthesis & Latency Audit
        """
        t_start = time.perf_counter()

        # Step 1: Query Embedding
        t_emb_0 = time.perf_counter()
        query_vector = self.embedder.embed_query(query)
        embed_latency_ms = round((time.perf_counter() - t_emb_0) * 1000.0, 3)

        # Step 2: Hybrid Retrieval
        t_ret_0 = time.perf_counter()
        retrieved_chunks = self.hybrid_retriever.search(
            query_text=query,
            query_vector=query_vector,
            top_k=top_k,
            mode=fusion_mode,
            alpha=alpha,
            lang_filter=lang_filter
        )
        search_latency_ms = round((time.perf_counter() - t_ret_0) * 1000.0, 3)

        # Step 3: Context Payload Assembly
        t_ctx_0 = time.perf_counter()
        context_passages = []
        doc_sources = []

        for idx, chunk in enumerate(retrieved_chunks, start=1):
            text = chunk.get("chunk_text", "")
            doc_id = chunk.get("document_id", f"doc_{idx}")
            context_passages.append(f"[{idx}] {text}")
            doc_sources.append({
                "rank": idx,
                "document_id": doc_id,
                "score": chunk.get("fusion_score") or chunk.get("score"),
                "chunk_text": text
            })

        formatted_context = "\n\n".join(context_passages)
        context_build_ms = round((time.perf_counter() - t_ctx_0) * 1000.0, 3)

        # Step 4: Fast Extractive / Generative Answer Synthesis
        t_gen_0 = time.perf_counter()
        if retrieved_chunks:
            primary_chunk = retrieved_chunks[0]
            answer = primary_chunk.get("chunk_text", "संदर्भ उपलब्ध नहीं है।")
        else:
            answer = "मुझे इस प्रश्न का उत्तर देने के लिए पर्याप्त जानकारी नहीं मिली।"

        generation_ms = round((time.perf_counter() - t_gen_0) * 1000.0, 3)
        total_pipeline_latency_ms = round((time.perf_counter() - t_start) * 1000.0, 3)

        return {
            "query": query,
            "answer": answer,
            "context": formatted_context,
            "sources": doc_sources,
            "retrieved_chunks_count": len(retrieved_chunks),
            "latency": {
                "embed_ms": embed_latency_ms,
                "retrieval_ms": search_latency_ms,
                "context_build_ms": context_build_ms,
                "generation_ms": generation_ms,
                "total_pipeline_ms": total_pipeline_latency_ms
            },
            "sub_200ms_target_met": total_pipeline_latency_ms < 200.0
        }
