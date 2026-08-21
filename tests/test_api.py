"""
Unit & Integration Tests for FastAPI Server Endpoints
"""

import sys
import os
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.app.main import app

client = TestClient(app)

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert "HH Goa 2026" in response.text


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "collection_name" in data

def test_rest_query_endpoint():
    payload = {
        "query": "भारत की राजधानी क्या है?",
        "top_k": 2,
        "fusion_mode": "rrf"
    }
    response = client.post("/api/v1/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == payload["query"]
    assert "answer" in data
    assert "sources" in data
    assert "latency" in data
    assert data["latency"]["total_pipeline_ms"] > 0

def test_rest_voice_endpoint():
    payload = {
        "text_transcript": "ताज महल कहाँ स्थित है?",
        "language": "hi",
        "top_k": 2
    }
    response = client.post("/api/v1/voice", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "audio_base64" in data

def test_benchmarks_endpoint():
    response = client.get("/api/v1/benchmarks")
    assert response.status_code == 200
    data = response.json()
    assert "latency" in data or "chunking" in data
