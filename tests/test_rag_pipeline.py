"""
Unit Tests for RAG Core Pipeline End-to-End Execution and Latency Guarantees
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.app.pipeline.rag_core import RAGCorePipeline

@pytest.fixture(scope="module")
def rag_pipeline():
    return RAGCorePipeline()

def test_rag_pipeline_end_to_end_query(rag_pipeline):
    query = "कॉर्पोरेशन का अर्थ क्या है?"
    result = rag_pipeline.process_query(query, top_k=3)

    assert result["query"] == query
    assert "answer" in result
    assert "sources" in result
    assert len(result["sources"]) <= 3
    assert "latency" in result
    assert result["latency"]["total_pipeline_ms"] > 0

def test_rag_pipeline_sub_200ms_latency(rag_pipeline):
    query = "ताज महल कहाँ स्थित है?"
    result = rag_pipeline.process_query(query, top_k=3)

    assert result["sub_200ms_target_met"] is True
    assert result["latency"]["total_pipeline_ms"] < 200.0
