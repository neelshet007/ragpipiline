"""
Ingestion Chunking Pipeline & Strategy Experiment Harness
Generates chunks across all 4 strategies (Sentence, Sliding Window, Semantic, Metadata-Aware)
and measures chunk count, length distribution, index size, and processing latency.
"""

import os
import sys
import json
import time
import argparse
from typing import List, Dict, Any

# Adjust import path to include backend/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.app.chunking import (
    SentenceChunker,
    SlidingWindowChunker,
    SemanticChunker,
    MetadataAwareChunker,
)

def run_chunking_experiment(input_path: str, output_dir: str = "data", reports_dir: str = "benchmarks/reports"):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    print("=" * 65, flush=True)
    print(f"RUNNING CHUNKING EXPERIMENT HARNESS: '{input_path}'", flush=True)
    print("=" * 65, flush=True)

    if not os.path.exists(input_path):
        print(f"[-] Preprocessed input file '{input_path}' not found! Run preprocess.py first.")
        sys.exit(1)

    docs = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                docs.append(json.loads(line))

    print(f"[+] Loaded {len(docs):,} preprocessed documents for chunking evaluation.\n")

    strategies = {
        "sentence": SentenceChunker(max_tokens=200),
        "sliding_window": SlidingWindowChunker(chunk_size=150, overlap=35),
        "semantic": SemanticChunker(target_chunk_size=200, similarity_threshold=0.5),
        "metadata_aware": MetadataAwareChunker(max_tokens=200),
    }

    results_report = {}

    for name, chunker in strategies.items():
        print(f"[+] Evaluating Strategy: '{name}'...", flush=True)
        t0 = time.perf_counter()
        
        all_chunks = []
        for doc in docs:
            doc_chunks = chunker.chunk_document(doc)
            all_chunks.extend(doc_chunks)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # Compute chunk length stats
        token_counts = [c.get("token_count", 0) for c in all_chunks]
        avg_tokens = sum(token_counts) / max(1, len(token_counts))
        min_tokens = min(token_counts) if token_counts else 0
        max_tokens = max(token_counts) if token_counts else 0

        # Output JSONL
        out_file = os.path.join(output_dir, f"chunks_{name}.jsonl")
        with open(out_file, "w", encoding="utf-8") as f:
            for chunk in all_chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

        file_size_mb = os.path.getsize(out_file) / (1024 * 1024)

        report_item = {
            "strategy": name,
            "total_documents": len(docs),
            "total_chunks": len(all_chunks),
            "avg_tokens_per_chunk": round(avg_tokens, 2),
            "min_tokens_per_chunk": min_tokens,
            "max_tokens_per_chunk": max_tokens,
            "processing_latency_ms": round(elapsed_ms, 2),
            "latency_per_doc_ms": round(elapsed_ms / max(1, len(docs)), 3),
            "estimated_file_size_mb": round(file_size_mb, 2),
            "output_file": out_file
        }

        results_report[name] = report_item
        print(f"    - Chunks Produced  : {len(all_chunks):,}")
        print(f"    - Avg Token Length : {avg_tokens:.1f} tokens (min: {min_tokens}, max: {max_tokens})")
        print(f"    - Ingestion Latency: {elapsed_ms:.2f} ms ({elapsed_ms/len(docs):.3f} ms/doc)")
        print(f"    - Saved File Size  : {file_size_mb:.2f} MB -> '{out_file}'\n")

    report_path = os.path.join(reports_dir, "chunking_experiment.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results_report, f, indent=2, ensure_ascii=False)

    print("=" * 65)
    print(f"[+] CHUNKING EXPERIMENT COMPLETE! Report saved to '{report_path}'")
    print("=" * 65)
    return results_report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run chunking strategy experiment")
    parser.add_argument("--input", type=str, default="data/preprocessed_docs_hi.jsonl", help="Input preprocessed docs jsonl")
    args = parser.parse_args()
    run_chunking_experiment(input_path=args.input)
