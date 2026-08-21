"""
Semantic Chunker Strategy
Groups semantically related sentences using embedding boundary distance detection.
Designed for offline ingestion processing.
"""

import re
import numpy as np
from typing import List, Dict, Any

INDIC_SENTENCE_DELIMITERS = re.compile(r'(?<=[।?!.\n])\s+')

class SemanticChunker:
    def __init__(self, target_chunk_size: int = 200, similarity_threshold: float = 0.5):
        self.target_chunk_size = target_chunk_size
        self.similarity_threshold = similarity_threshold

    def _split_sentences(self, text: str) -> List[str]:
        if not text:
            return []
        return [s.strip() for s in INDIC_SENTENCE_DELIMITERS.split(text) if s.strip()]

    def _ngram_similarity(self, s1: str, s2: str) -> float:
        """Lightweight character n-gram cosine similarity for sentence transition boundary detection."""
        def get_ngrams(s, n=3):
            return set(s[i:i+n] for i in range(len(s)-n+1))
        
        g1 = get_ngrams(s1)
        g2 = get_ngrams(s2)
        if not g1 or not g2:
            return 0.0
        intersection = len(g1.intersection(g2))
        union = len(g1.union(g2))
        return intersection / float(union) if union > 0 else 0.0

    def chunk_document(self, doc: Dict[str, Any]) -> List[Dict[str, Any]]:
        text = doc.get("passage_text", "")
        sentences = self._split_sentences(text)
        if not sentences:
            return []

        if len(sentences) == 1:
            return [{
                "chunk_id": f"{doc.get('document_id', 'doc')}_sem_0",
                "document_id": doc.get("document_id"),
                "query_id": doc.get("query_id"),
                "language": doc.get("language", "hi"),
                "chunk_strategy": "semantic",
                "chunk_text": sentences[0],
                "position": 0,
                "token_count": len(sentences[0].split()),
                "semantic_cohesion_score": 1.0,
                "is_selected": doc.get("is_selected", 0)
            }]

        chunks = []
        current_group = [sentences[0]]
        current_tokens = len(sentences[0].split())
        chunk_index = 0

        for i in range(1, len(sentences)):
            prev_s = sentences[i-1]
            curr_s = sentences[i]
            curr_tokens = len(curr_s.split())

            sim = self._ngram_similarity(prev_s, curr_s)

            # If semantic similarity dips or max tokens exceeded, seal current chunk
            if (sim < self.similarity_threshold or current_tokens + curr_tokens > self.target_chunk_size) and current_tokens >= 25:
                chunk_text = " ".join(current_group)
                chunks.append({
                    "chunk_id": f"{doc.get('document_id', 'doc')}_sem_{chunk_index}",
                    "document_id": doc.get("document_id"),
                    "query_id": doc.get("query_id"),
                    "language": doc.get("language", "hi"),
                    "chunk_strategy": "semantic",
                    "chunk_text": chunk_text,
                    "position": chunk_index,
                    "token_count": len(chunk_text.split()),
                    "semantic_cohesion_score": round(sim, 3),
                    "is_selected": doc.get("is_selected", 0)
                })
                chunk_index += 1
                current_group = [curr_s]
                current_tokens = curr_tokens
            else:
                current_group.append(curr_s)
                current_tokens += curr_tokens

        if current_group:
            chunk_text = " ".join(current_group)
            chunks.append({
                "chunk_id": f"{doc.get('document_id', 'doc')}_sem_{chunk_index}",
                "document_id": doc.get("document_id"),
                "query_id": doc.get("query_id"),
                "language": doc.get("language", "hi"),
                "chunk_strategy": "semantic",
                "chunk_text": chunk_text,
                "position": chunk_index,
                "token_count": len(chunk_text.split()),
                "semantic_cohesion_score": 1.0,
                "is_selected": doc.get("is_selected", 0)
            })

        return chunks
