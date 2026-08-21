"""
Qdrant Indexing Pipeline Script
Loads embedded chunks, initializes Qdrant vector collection, builds payload indexes,
and populates vector store safely and idempotently.
"""

import os
import sys
import json
import time
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from backend.app.retrieval.dense import QdrantDenseRetriever
from backend.app.embeddings.embedder import MultilingualEmbedder

def build_qdrant_index(
    embeddings_file: str = "data/embedded_chunks.json",
    collection_name: str = "msmarco_xi_hi",
    recreate: bool = True
):
    print("=" * 65, flush=True)
    print(f"BUILDING QDRANT VECTOR INDEX: Collection '{collection_name}'", flush=True)
    print("=" * 65, flush=True)

    if not os.path.exists(embeddings_file):
        print(f"[-] Embedded chunks file '{embeddings_file}' not found! Generating embeddings first...")
        from ingestion.generate_embeddings import generate_embeddings
        embeddings_file = generate_embeddings()

    with open(embeddings_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    chunks = data.get("chunks", [])
    vectors = data.get("vectors", [])
    dim = data.get("embedding_dim", 384)

    print(f"[+] Loaded {len(chunks):,} chunks and vectors (dim={dim}).")

    retriever = QdrantDenseRetriever(collection_name=collection_name, vector_size=dim)
    retriever.create_collection(recreate=recreate)

    t0 = time.perf_counter()
    retriever.index_chunks(chunks=chunks, embeddings=vectors, batch_size=250)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    print(f"\n[+] Qdrant indexing completed in {elapsed_ms:.2f} ms!")

    # Perform sample verification search
    embedder = MultilingualEmbedder()
    test_query = "कॉर्पोरेशन क्या है?"
    print(f"\n[+] Verification Vector Search for test query: '{test_query}'...")
    
    q_vec = embedder.embed_query(test_query)
    search_hits = retriever.search(query_vector=q_vec, top_k=3)

    print(f"[+] Verification Top Results ({search_hits[0].get('search_latency_ms')} ms retrieval latency):")
    for rank, hit in enumerate(search_hits, start=1):
        print(f"    Rank {rank} [Score: {hit.get('score'):.4f}]: '{hit.get('chunk_text')[:70]}...'")

    print("\n=" * 65)
    print("[+] QDRANT VECTOR INDEX BUILDING SUCCEEDED!")
    print("=" * 65)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Qdrant vector index")
    parser.add_argument("--embeddings", type=str, default="data/embedded_chunks.json")
    parser.add_argument("--collection", type=str, default="msmarco_xi_hi")
    parser.add_argument("--recreate", action="store_true", default=True)
    args = parser.parse_args()

    build_qdrant_index(
        embeddings_file=args.embeddings,
        collection_name=args.collection,
        recreate=args.recreate
    )
