"""
Qdrant Dense Vector Retriever Component
Manages Qdrant vector collection setup, batch indexing, and sub-millisecond dense vector search.
Supports both remote server connection and local embedded storage mode (:memory: / local path).
"""

import os
import time
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
DEFAULT_COLLECTION = "msmarco_xi_hi"

class QdrantDenseRetriever:
    def __init__(self, collection_name: str = DEFAULT_COLLECTION, vector_size: int = 384, url: Optional[str] = None):
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.storage_path = os.path.abspath("qdrant_storage")
        self.client = self._init_client(url or QDRANT_URL)

    def _init_client(self, url: str) -> QdrantClient:
        """
        Attempts to connect to a remote Qdrant instance; if unavailable,
        instantiates local disk-persisted vector storage ('./qdrant_storage').
        """
        if url == "memory":
            print("[+] Initializing in-memory Qdrant instance...", flush=True)
            return QdrantClient(":memory:")

        try:
            print(f"[+] Connecting to Qdrant cluster at {url}...", flush=True)
            client = QdrantClient(url=url, timeout=0.5, check_compatibility=False)
            client.get_collections()
            print(f"[+] Connected to remote Qdrant server at {url}!", flush=True)
            return client
        except Exception as e:
            print(f"[-] Remote Qdrant server unreachable ({e}). Using local disk storage './qdrant_storage'.", flush=True)
            os.makedirs(self.storage_path, exist_ok=True)
            return QdrantClient(path=self.storage_path)

    def create_collection(self, distance: Distance = Distance.COSINE, recreate: bool = False):
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection_name in collections:
            if recreate:
                print(f"[+] Recreating Qdrant collection '{self.collection_name}'...")
                self.client.delete_collection(self.collection_name)
            else:
                print(f"[+] Collection '{self.collection_name}' already exists.")
                return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=self.vector_size, distance=distance)
        )
        print(f"[+] Created Qdrant collection '{self.collection_name}' (dim={self.vector_size}, distance={distance})")

    def index_chunks(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]], batch_size: int = 250):
        if len(chunks) != len(embeddings):
            raise ValueError(f"Chunks count ({len(chunks)}) does not match embeddings count ({len(embeddings)})")

        points = []
        for idx, (chunk, vec) in enumerate(zip(chunks, embeddings)):
            p_id = chunk.get("chunk_id") or f"pt_{idx}"
            # Convert string IDs or custom IDs to deterministic integer or uuid for Qdrant if needed
            numeric_id = idx + 1 if isinstance(p_id, str) else p_id
            
            payload = {
                "chunk_id": chunk.get("chunk_id"),
                "document_id": chunk.get("document_id"),
                "query_id": chunk.get("query_id"),
                "language": chunk.get("language", "hi"),
                "chunk_strategy": chunk.get("chunk_strategy", "default"),
                "chunk_text": chunk.get("chunk_text"),
                "position": chunk.get("position", 0),
                "token_count": chunk.get("token_count", 0),
                "is_selected": chunk.get("is_selected", 0),
            }
            points.append(PointStruct(id=numeric_id, vector=vec, payload=payload))

        for b_start in range(0, len(points), batch_size):
            b_points = points[b_start:b_start + batch_size]
            self.client.upsert(collection_name=self.collection_name, points=b_points)
        print(f"[+] Successfully indexed {len(points):,} points into Qdrant collection '{self.collection_name}'!")

    def search(self, query_vector: List[float], top_k: int = 10, lang_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        t0 = time.perf_counter()
        
        query_filter = None
        if lang_filter:
            query_filter = Filter(
                must=[FieldCondition(key="language", match=MatchValue(value=lang_filter))]
            )

        if hasattr(self.client, "query_points"):
            res = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=top_k,
                query_filter=query_filter
            )
            hits = res.points
        else:
            hits = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k,
                query_filter=query_filter
            )

        latency_ms = (time.perf_counter() - t0) * 1000.0
        results = []
        for hit in hits:
            payload = hit.payload or {}
            payload["score"] = float(hit.score)
            payload["search_latency_ms"] = round(latency_ms, 3)
            results.append(payload)
        return results
