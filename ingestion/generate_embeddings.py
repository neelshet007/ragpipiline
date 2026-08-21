"""
Offline Embedding Generation Script
Generates dense vector embeddings for indexed chunks using MultilingualEmbedder.
Caches vectors to disk to make ingestion idempotent and fast.
"""

import os
import sys
import json
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.app.embeddings.embedder import MultilingualEmbedder

def generate_embeddings(
    input_chunks_path: str = "data/chunks_metadata_aware.jsonl",
    output_embeddings_path: str = "data/embedded_chunks.json",
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    batch_size: int = 64
):
    print("=" * 65, flush=True)
    print(f"OFFLINE EMBEDDING GENERATION", flush=True)
    print(f"Input Chunks : {input_chunks_path}", flush=True)
    print(f"Model        : {model_name}", flush=True)
    print("=" * 65, flush=True)

    if not os.path.exists(input_chunks_path):
        print(f"[-] Input file '{input_chunks_path}' not found! Run build_chunks.py first.")
        sys.exit(1)

    chunks = []
    with open(input_chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))

    print(f"[+] Loaded {len(chunks):,} text chunks for embedding generation.", flush=True)

    embedder = MultilingualEmbedder(model_name=model_name)
    texts = [c.get("chunk_text", "") for c in chunks]

    print(f"[+] Generating embeddings (Batch Size={batch_size})...", flush=True)
    t0 = time.perf_counter()
    vectors = embedder.embed_texts(texts, batch_size=batch_size, normalize=True)
    elapsed = time.perf_counter() - t0

    throughput = len(texts) / max(0.001, elapsed)
    print(f"[+] Generated {len(vectors):,} vectors in {elapsed:.2f} s ({throughput:.1f} chunks/sec)!")

    # Store payload + vectors
    payload_data = {
        "model_name": model_name,
        "embedding_dim": embedder.embedding_dim,
        "chunk_count": len(chunks),
        "generation_time_s": round(elapsed, 2),
        "chunks": chunks,
        "vectors": vectors
    }

    os.makedirs(os.path.dirname(output_embeddings_path), exist_ok=True)
    with open(output_embeddings_path, "w", encoding="utf-8") as f:
        json.dump(payload_data, f, ensure_ascii=False)

    file_size_mb = os.path.getsize(output_embeddings_path) / (1024 * 1024)
    print(f"[+] Embedded dataset saved to '{output_embeddings_path}'! Size: {file_size_mb:.2f} MB")
    return output_embeddings_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate embeddings for chunk dataset")
    parser.add_argument("--input", type=str, default="data/chunks_metadata_aware.jsonl")
    parser.add_argument("--output", type=str, default="data/embedded_chunks.json")
    parser.add_argument("--model", type=str, default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    generate_embeddings(
        input_chunks_path=args.input,
        output_embeddings_path=args.output,
        model_name=args.model,
        batch_size=args.batch_size
    )
