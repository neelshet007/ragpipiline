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

def detect_query_language(text: str) -> str:
    """Detects script / language of input query text."""
    import re
    if not text:
        return "hi"
    if re.search(r'[\u0A80-\u0AFF]', text):
        return "gu" # Gujarati
    if re.search(r'[\u0980-\u09FF]', text):
        return "bn" # Bengali
    if re.search(r'[\u0B80-\u0BFF]', text):
        return "ta" # Tamil
    if re.search(r'[\u0C00-\u0C7F]', text):
        return "te" # Telugu
    if re.search(r'[\u0C80-\u0CFF]', text):
        return "kn" # Kannada
    if re.search(r'[\u0D00-\u0D7F]', text):
        return "ml" # Malayalam
    if re.search(r'[\u0A00-\u0A7F]', text):
        return "pa" # Punjabi
    if re.search(r'[\u0900-\u097F]', text):
        return "hi" # Hindi / Marathi / Devanagari
    if re.search(r'[a-zA-Z]', text):
        return "en" # English / Latin
    return "hi"


from functools import lru_cache

@lru_cache(maxsize=4096)
def _cached_translate(text: str, source_lang: str, target_lang: str) -> str:
    """Cached translation worker."""
    if not text or source_lang == target_lang:
        return text
    try:
        from deep_translator import GoogleTranslator
        return GoogleTranslator(source=source_lang, target=target_lang).translate(text)
    except Exception:
        return text


def translate_text(text: str, source_lang: str, target_lang: str) -> str:
    """Translates text between languages safely with high-speed LRU memory cache."""
    return _cached_translate(text, source_lang, target_lang)


class RAGCorePipeline:
    """
    End-to-End Low-Latency RAG Core Engine with Multilingual Language Detection,
    Cross-Lingual Retrieval, Security Guardrails & Refusal Architecture.
    """

    def __init__(
        self,
        embedder: Optional[MultilingualEmbedder] = None,
        dense_retriever: Optional[QdrantDenseRetriever] = None,
        bm25_retriever: Optional[BM25Retriever] = None,
        bm25_index_path: str = "data/bm25_index.json"
    ):
        t0 = time.perf_counter()
        print("[+] Initializing RAGCorePipeline with Guardrails & Multilingual Engine...", flush=True)

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

        # Step 0: Language Detection & Cross-Lingual Translation
        detected_lang = detect_query_language(query)
        search_query = query
        if detected_lang != "hi":
            search_query = translate_text(query, source_lang=detected_lang, target_lang="hi")

        # Input Guardrail Check
        input_check = self.input_guard.validate_query(query)
        if not input_check["is_safe"]:
            total_pipeline_ms = round((time.perf_counter() - t_start) * 1000.0, 3)
            refusal = input_check["refusal_message"]
            if detected_lang != "hi":
                refusal = translate_text(refusal, source_lang="hi", target_lang=detected_lang)
            return {
                "query": query,
                "detected_language": detected_lang,
                "answer": refusal,
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

        # Step 1: Query Embedding on Hindi Search Query
        t_emb_0 = time.perf_counter()
        query_vector = self.embedder.embed_query(search_query)
        embed_latency_ms = round((time.perf_counter() - t_emb_0) * 1000.0, 3)

        # Step 2: Hybrid Retrieval
        t_ret_0 = time.perf_counter()
        retrieved_chunks = self.hybrid_retriever.search(
            query_text=search_query,
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
                "chunk_text": text,
                "bm25_hit": chunk.get("bm25_hit", True),
                "bm25_max_score": chunk.get("bm25_max_score", 0.0)
            })

        formatted_context = "\n\n".join(context_passages)
        context_build_ms = round((time.perf_counter() - t_ctx_0) * 1000.0, 3)

        # Step 4: Answer Synthesis
        t_gen_0 = time.perf_counter()
        raw_answer = retrieved_chunks[0].get("chunk_text", "") if retrieved_chunks else ""

        # Step 5: Output Guardrail Check
        output_check = self.output_guard.validate_output(
            query=search_query,
            retrieved_sources=doc_sources,
            generated_answer=raw_answer
        )

        final_answer = output_check["final_answer"]

        # Step 6: Target Language Synthesis (Translate answer back to user language)
        if detected_lang != "hi" and output_check["is_grounded"]:
            final_answer = translate_text(raw_answer, source_lang="hi", target_lang=detected_lang)
        elif detected_lang != "hi" and not output_check["is_grounded"]:
            final_answer = (
                f"Sorry, no relevant information was found in our indexed database for '{query}'. "
                f"Please try asking about topics present in the index (e.g. corporations, checking accounts, Stanley Cup, primary teeth)."
            )

        generation_ms = round((time.perf_counter() - t_gen_0) * 1000.0, 3)
        total_pipeline_latency_ms = round((time.perf_counter() - t_start) * 1000.0, 3)

        return {
            "query": query,
            "detected_language": detected_lang,
            "search_query_hi": search_query,
            "answer": final_answer,
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

