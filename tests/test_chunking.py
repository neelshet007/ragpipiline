"""
Unit Tests for Intelligent Chunking Strategies
Validates sentence, sliding window, semantic, and metadata-aware chunkers.
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.app.chunking import (
    SentenceChunker,
    SlidingWindowChunker,
    SemanticChunker,
    MetadataAwareChunker,
)

SAMPLE_DOC = {
    "document_id": "doc_test_101_0",
    "query_id": 101,
    "query_type": "DESCRIPTION",
    "query_text": "भारत का संविधान क्या है?",
    "answer_text": "भारत का संविधान भारत का सर्वोच्च विधान है।",
    "passage_text": "भारत का संविधान भारत का सर्वोच्च विधान है। यह 26 नवंबर 1949 को पारित हुआ। 26 जनवरी 1950 को इसे लागू किया गया। यह दुनिया का सबसे बड़ा लिखित संविधान है।",
    "is_selected": 1,
    "language": "hi",
}

def test_sentence_chunker():
    chunker = SentenceChunker(max_tokens=20)
    chunks = chunker.chunk_document(SAMPLE_DOC)
    assert len(chunks) >= 1
    assert chunks[0]["chunk_strategy"] == "sentence"
    assert "document_id" in chunks[0]
    assert chunks[0]["token_count"] > 0

def test_sliding_window_chunker():
    chunker = SlidingWindowChunker(chunk_size=15, overlap=5)
    chunks = chunker.chunk_document(SAMPLE_DOC)
    assert len(chunks) >= 1
    assert chunks[0]["chunk_strategy"] == "sliding_window"
    assert "stride" in chunks[0]

def test_semantic_chunker():
    chunker = SemanticChunker(target_chunk_size=25, similarity_threshold=0.3)
    chunks = chunker.chunk_document(SAMPLE_DOC)
    assert len(chunks) >= 1
    assert chunks[0]["chunk_strategy"] == "semantic"
    assert "semantic_cohesion_score" in chunks[0]

def test_metadata_aware_chunker():
    chunker = MetadataAwareChunker(max_tokens=30)
    chunks = chunker.chunk_document(SAMPLE_DOC)
    assert len(chunks) >= 1
    assert chunks[0]["chunk_strategy"] == "metadata_aware"
    assert chunks[0]["query_id"] == 101
    assert chunks[0]["query_text"] == "भारत का संविधान क्या है?"
    assert chunks[0]["is_selected"] == 1
