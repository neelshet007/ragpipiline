"""
Sentence-Based Chunker Strategy
Splits text on sentence boundaries (including Indic Purna Viram '।' and standard delimiters).
Groups sentences to respect max token/word limits while maintaining strict sentence integrity.
"""

import re
from typing import List, Dict, Any

INDIC_SENTENCE_DELIMITERS = re.compile(r'(?<=[।?!.\n])\s+')

class SentenceChunker:
    def __init__(self, max_tokens: int = 250, min_tokens: int = 20):
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens

    def split_sentences(self, text: str) -> List[str]:
        """Split text into sentences handling Indic and English sentence boundaries."""
        if not text:
            return []
        raw_sentences = INDIC_SENTENCE_DELIMITERS.split(text)
        sentences = [s.strip() for s in raw_sentences if s.strip()]
        return sentences

    def chunk_document(self, doc: Dict[str, Any]) -> List[Dict[str, Any]]:
        text = doc.get("passage_text", "")
        sentences = self.split_sentences(text)
        if not sentences:
            return []

        chunks = []
        current_chunk_sentences = []
        current_token_count = 0
        chunk_index = 0

        for s in sentences:
            s_tokens = len(s.split())
            if current_token_count + s_tokens > self.max_tokens and current_chunk_sentences:
                chunk_text = " ".join(current_chunk_sentences)
                chunks.append({
                    "chunk_id": f"{doc.get('document_id', 'doc')}_s_{chunk_index}",
                    "document_id": doc.get("document_id"),
                    "query_id": doc.get("query_id"),
                    "language": doc.get("language", "hi"),
                    "chunk_strategy": "sentence",
                    "chunk_text": chunk_text,
                    "position": chunk_index,
                    "token_count": len(chunk_text.split()),
                    "sentence_count": len(current_chunk_sentences),
                    "is_selected": doc.get("is_selected", 0)
                })
                chunk_index += 1
                current_chunk_sentences = [s]
                current_token_count = s_tokens
            else:
                current_chunk_sentences.append(s)
                current_token_count += s_tokens

        if current_chunk_sentences:
            chunk_text = " ".join(current_chunk_sentences)
            chunks.append({
                "chunk_id": f"{doc.get('document_id', 'doc')}_s_{chunk_index}",
                "document_id": doc.get("document_id"),
                "query_id": doc.get("query_id"),
                "language": doc.get("language", "hi"),
                "chunk_strategy": "sentence",
                "chunk_text": chunk_text,
                "position": chunk_index,
                "token_count": len(chunk_text.split()),
                "sentence_count": len(current_chunk_sentences),
                "is_selected": doc.get("is_selected", 0)
            })

        return chunks
