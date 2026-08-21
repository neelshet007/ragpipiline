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
from backend.app.guardrails.input_guard import InputGuardrail
from backend.app.guardrails.output_guard import OutputGuardrail

class RAGCorePipeline:
    """
    End-to-End Low-Latency RAG Core Engine with Security Guardrails & Refusal Architecture.
    """

    def __init__(
        self,
        embedder: Optional[MultilingualEmbedder] = None,
        dense_retriever: Optional[QdrantDenseRetriever] = None,
        bm25_retriever: Optional[BM25Retriever] = None,
        bm25_index_path: str = "data/bm25_index.json"
    ):
        t0 = time.perf_counter()
        print("[+] Initializing RAGCorePipeline with Guardrails...", flush=True)

        self.embedder = embedder or MultilingualEmbedder()
        self.dense_retriever = dense_retriever or QdrantDenseRetriever()

        if bm25_retriever is not None:
            self.bm25_retriever = bm25_retriever
        else:
            self.bm25_retriever = BM25Retriever()
            if os.path.exists(bm25_index_path):
                with open(bm25_index_path, "r", encoding="utf-8") as f:
                    idx_data = json.load(f)
                    self.bm25_retriever.fit(idx_data.get("chunks", []))

        self.hybrid_retriever = HybridRetriever(
            dense_retriever=self.dense_retriever,
            bm25_retriever=self.bm25_retriever
        )
        self.input_guard = InputGuardrail()
        self.output_guard = OutputGuardrail()

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
        t_start = time.perf_counter()

        # Input Guardrail Check
        input_check = self.input_guard.validate_query(query)
        if not input_check["is_safe"]:
            total_pipeline_ms = round((time.perf_counter() - t_start) * 1000.0, 3)
            return {
                "query": query,
                "answer": input_check["refusal_message"],
                "context": "",
                "sources": [],
                "retrieved_chunks_count": 0,
                "guardrail": {
                    "input_check": input_check,
                    "output_check": None,
                    "refused": True,
                    "reason": input_check["reason"]
                },
                "latency": {
                    "embed_ms": 0.0,
                    "retrieval_ms": 0.0,
                    "context_build_ms": 0.0,
                    "generation_ms": 0.0,
                    "guardrail_ms": input_check["check_latency_ms"],
                    "total_pipeline_ms": total_pipeline_ms
                },
                "sub_200ms_target_met": total_pipeline_ms < 200.0
            }

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

        # Step 4: Answer Synthesis
        t_gen_0 = time.perf_counter()
        raw_answer = retrieved_chunks[0].get("chunk_text", "") if retrieved_chunks else ""
        generation_ms = round((time.perf_counter() - t_gen_0) * 1000.0, 3)

        # Step 5: Output Guardrail Check
        output_check = self.output_guard.validate_output(
            query=query,
            retrieved_sources=doc_sources,
            generated_answer=raw_answer
        )

        total_pipeline_latency_ms = round((time.perf_counter() - t_start) * 1000.0, 3)

        return {
            "query": query,
            "answer": output_check["final_answer"],
            "context": formatted_context,
            "sources": doc_sources if output_check["is_grounded"] else [],
            "retrieved_chunks_count": len(retrieved_chunks),
            "guardrail": {
                "input_check": input_check,
                "output_check": output_check,
                "refused": not output_check["is_grounded"],
                "reason": output_check["reason"]
            },
            "latency": {
                "embed_ms": embed_latency_ms,
                "retrieval_ms": search_latency_ms,
                "context_build_ms": context_build_ms,
                "generation_ms": generation_ms,
                "guardrail_ms": input_check["check_latency_ms"] + output_check["guard_latency_ms"],
                "total_pipeline_ms": total_pipeline_latency_ms
            },
            "sub_200ms_target_met": total_pipeline_latency_ms < 200.0
        }

