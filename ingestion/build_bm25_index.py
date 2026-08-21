"""
BM25 Serialized Index Building Script
Reads chunks, fits BM25Retriever, and saves index metadata to disk.
"""

import os
import sys
import json
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from backend.app.retrieval.bm25 import BM25Retriever

def build_bm25_index(
    input_chunks_path: str = "data/chunks_metadata_aware.jsonl",
    output_bm25_path: str = "data/bm25_index.json"
):
    print("=" * 65, flush=True)
    print("BUILDING BM25 SPARSE LEXICAL INDEX", flush=True)
    print("=" * 65, flush=True)

    if not os.path.exists(input_chunks_path):
        raise FileNotFoundError(f"Input chunks file '{input_chunks_path}' not found!")

    chunks = []
    with open(input_chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))

    print(f"[+] Loaded {len(chunks):,} document chunks from '{input_chunks_path}'.")

    bm25 = BM25Retriever()
    t0 = time.perf_counter()
    bm25.fit(chunks)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    index_payload = {
        "doc_count": bm25.doc_count,
        "avg_doc_len": bm25.avg_doc_len,
        "idf": bm25.idf,
        "chunks": chunks
    }

    with open(output_bm25_path, "w", encoding="utf-8") as f:
        json.dump(index_payload, f, ensure_ascii=False, indent=2)

    print(f"\n[+] BM25 index built in {elapsed_ms:.2f} ms and saved to '{output_bm25_path}'!")

    # Perform test verification query
    test_query = "कॉर्पोरेशन क्या है?"
    test_hits = bm25.search(test_query, top_k=3)
    print(f"\n[+] Verification BM25 search results for '{test_query}':")
    for rank, hit in enumerate(test_hits, start=1):
        print(f"    Rank {rank} [Score: {hit.get('score'):.4f}]: '{hit.get('chunk_text')[:70]}...'")

    print("\n=" * 65)
    print("[+] BM25 INDEX BUILDING SUCCEEDED!")
    print("=" * 65)

if __name__ == "__main__":
    build_bm25_index()
