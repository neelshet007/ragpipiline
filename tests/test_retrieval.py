"""
Unit Tests for Multilingual Embeddings and Dense Vector Search (Qdrant)
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.app.embeddings.embedder import MultilingualEmbedder
from backend.app.retrieval.dense import QdrantDenseRetriever

def test_embedder_dimension_and_output():
    embedder = MultilingualEmbedder()
    assert embedder.embedding_dim > 0

    texts = ["कॉर्पोरेशन क्या है?", "What is a corporation?"]
    vecs = embedder.embed_texts(texts, normalize=True)
    assert len(vecs) == 2
    assert len(vecs[0]) == embedder.embedding_dim

def test_qdrant_retriever_local_index_and_search():
    retriever = QdrantDenseRetriever(collection_name="test_collection", vector_size=384, url="memory")
    retriever.create_collection(recreate=True)

    dummy_chunks = [
        {
            "chunk_id": "c1",
            "document_id": "doc1",
            "query_id": 1,
            "language": "hi",
            "chunk_strategy": "metadata_aware",
            "chunk_text": "कॉर्पोरेशन एक कानूनी इकाई है।",
            "is_selected": 1,
        },
        {
            "chunk_id": "c2",
            "document_id": "doc2",
            "query_id": 2,
            "language": "hi",
            "chunk_strategy": "metadata_aware",
            "chunk_text": "ताज महल आगरा में स्थित है।",
            "is_selected": 0,
        }
    ]

    embedder = MultilingualEmbedder()
    texts = [c["chunk_text"] for c in dummy_chunks]
    vecs = embedder.embed_texts(texts)

    retriever.index_chunks(dummy_chunks, vecs)

    q_vec = embedder.embed_query("कॉर्पोरेशन क्या होती है?")
    hits = retriever.search(q_vec, top_k=2)

    assert len(hits) >= 1
    assert "score" in hits[0]
    assert hits[0]["chunk_id"] == "c1"
