"""
Metadata-Aware Chunker Strategy
Preserves and injects rich structural and domain metadata into each chunk payload.
"""

from typing import List, Dict, Any
from .sentence import SentenceChunker

class MetadataAwareChunker:
    def __init__(self, max_tokens: int = 200):
        self.base_chunker = SentenceChunker(max_tokens=max_tokens)

    def chunk_document(self, doc: Dict[str, Any]) -> List[Dict[str, Any]]:
        base_chunks = self.base_chunker.chunk_document(doc)
        enriched_chunks = []

        total_chunks = len(base_chunks)
        for idx, c in enumerate(base_chunks):
            chunk_text = c.get("chunk_text", "")
            
            # Enriched metadata payload
            enriched = {
                "chunk_id": f"{doc.get('document_id', 'doc')}_meta_{idx}",
                "document_id": doc.get("document_id"),
                "query_id": doc.get("query_id"),
                "query_type": doc.get("query_type", "UNKNOWN"),
                "query_text": doc.get("query_text", ""),
                "answer_text": doc.get("answer_text", ""),
                "language": doc.get("language", "hi"),
                "chunk_strategy": "metadata_aware",
                "chunk_text": chunk_text,
                "position": idx,
                "total_chunks_in_doc": total_chunks,
                "token_count": len(chunk_text.split()),
                "char_count": len(chunk_text),
                "is_selected": doc.get("is_selected", 0),
                "has_query_context": bool(doc.get("query_text"))
            }
            enriched_chunks.append(enriched)

        return enriched_chunks
