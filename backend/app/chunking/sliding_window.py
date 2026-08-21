"""
Sliding Window Chunker Strategy
Configurable chunk size and token overlap for fixed-stride sliding window text segmentation.
"""

from typing import List, Dict, Any

class SlidingWindowChunker:
    def __init__(self, chunk_size: int = 150, overlap: int = 35):
        self.chunk_size = chunk_size
        self.overlap = max(0, min(overlap, chunk_size - 1))

    def chunk_document(self, doc: Dict[str, Any]) -> List[Dict[str, Any]]:
        text = doc.get("passage_text", "")
        words = text.split()
        if not words:
            return []

        chunks = []
        chunk_index = 0
        step = max(1, self.chunk_size - self.overlap)
        
        i = 0
        while i < len(words):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = " ".join(chunk_words)
            
            chunks.append({
                "chunk_id": f"{doc.get('document_id', 'doc')}_sw_{chunk_index}",
                "document_id": doc.get("document_id"),
                "query_id": doc.get("query_id"),
                "language": doc.get("language", "hi"),
                "chunk_strategy": "sliding_window",
                "chunk_text": chunk_text,
                "position": chunk_index,
                "token_count": len(chunk_words),
                "stride": step,
                "is_selected": doc.get("is_selected", 0)
            })
            
            chunk_index += 1
            if i + self.chunk_size >= len(words):
                break
            i += step

        return chunks
