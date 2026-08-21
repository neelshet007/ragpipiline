"""
Unit Tests for BM25 Sparse Search & Hybrid RRF Retrieval Fusion
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.app.embeddings.embedder import MultilingualEmbedder
from backend.app.retrieval.dense import QdrantDenseRetriever
from backend.app.retrieval.bm25 import BM25Retriever
from backend.app.retrieval.hybrid import HybridRetriever

@pytest.fixture
def sample_chunks():
    return [
        {
            "chunk_id": "c101",
            "document_id": "doc_101",
            "query_id": 1,
            "language": "hi",
            "chunk_strategy": "metadata_aware",
            "chunk_text": "कॉर्पोरेशन एक कानूनी संस्था है जो अपने मालिकों से अलग होती है।",
            "is_selected": 1,
        },
        {
            "chunk_id": "c102",
            "document_id": "doc_102",
            "query_id": 2,
            "language": "hi",
            "chunk_strategy": "metadata_aware",
            "chunk_text": "भारत की राजधानी नई दिल्ली है। ताज महल आगरा शहर में स्थित है।",
            "is_selected": 0,
        },
        {
            "chunk_id": "c103",
            "document_id": "doc_103",
            "query_id": 3,
            "language": "hi",
            "chunk_strategy": "metadata_aware",
            "chunk_text": "कंपनी अधिनियम के तहत पंजीकृत निगम या कॉर्पोरेशन को शेयरधारकों का समर्थन प्राप्त होता है।",
            "is_selected": 1,
        }
    ]

def test_bm25_retrieval(sample_chunks):
    bm25 = BM25Retriever()
    bm25.fit(sample_chunks)

    results = bm25.search(query="कॉर्पोरेशन कानून शेयरधारक", top_k=2)
    assert len(results) >= 1
    assert "score" in results[0]
    assert results[0]["chunk_id"] in ["c101", "c103"]

def test_hybrid_rrf_fusion(sample_chunks):
    embedder = MultilingualEmbedder()
    vecs = embedder.embed_texts([c["chunk_text"] for c in sample_chunks])

    dense = QdrantDenseRetriever(collection_name="test_hybrid", vector_size=384, url="memory")
    dense.create_collection(recreate=True)
    dense.index_chunks(sample_chunks, vecs)

    bm25 = BM25Retriever()
    bm25.fit(sample_chunks)

    hybrid = HybridRetriever(dense_retriever=dense, bm25_retriever=bm25)

    test_q = "कॉर्पोरेशन शेयरधारक कानून"
    q_vec = embedder.embed_query(test_q)

    rrf_hits = hybrid.search(query_text=test_q, query_vector=q_vec, top_k=2, mode="rrf")
    assert len(rrf_hits) > 0
    assert rrf_hits[0]["fusion_type"] == "rrf"
    assert "fusion_score" in rrf_hits[0]

def test_hybrid_convex_fusion(sample_chunks):
    embedder = MultilingualEmbedder()
    vecs = embedder.embed_texts([c["chunk_text"] for c in sample_chunks])

    dense = QdrantDenseRetriever(collection_name="test_convex", vector_size=384, url="memory")
    dense.create_collection(recreate=True)
    dense.index_chunks(sample_chunks, vecs)

    bm25 = BM25Retriever()
    bm25.fit(sample_chunks)

    hybrid = HybridRetriever(dense_retriever=dense, bm25_retriever=bm25)

    test_q = "दिल्ली ताज महल"
    q_vec = embedder.embed_query(test_q)

    convex_hits = hybrid.search(query_text=test_q, query_vector=q_vec, top_k=2, mode="convex", alpha=0.6)
    assert len(convex_hits) > 0
    assert convex_hits[0]["fusion_type"] == "convex"
    assert "fusion_score" in convex_hits[0]
