"""
Pydantic API Request/Response Schemas
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class QueryRequest(BaseModel):
    query: str = Field(..., description="Target search question in Indic or English")
    top_k: int = Field(default=3, ge=1, le=10, description="Number of passages to retrieve")
    fusion_mode: str = Field(default="rrf", description="Hybrid fusion strategy: 'rrf' or 'convex'")
    alpha: float = Field(default=0.5, ge=0.0, le=1.0, description="Dense vs Sparse weight for convex mode")
    lang_filter: Optional[str] = Field(default=None, description="Language filter e.g. 'hi'")

class VoiceQueryRequest(BaseModel):
    audio_base64: Optional[str] = Field(default=None, description="Base64 encoded audio stream")
    text_transcript: Optional[str] = Field(default=None, description="Web Speech API transcript text")
    language: str = Field(default="hi", description="Spoken language code")
    top_k: int = Field(default=3, ge=1, le=10)

class SourceDoc(BaseModel):
    rank: int
    document_id: str
    score: Optional[float] = None
    chunk_text: str

class LatencyMetrics(BaseModel):
    embed_ms: float
    retrieval_ms: float
    context_build_ms: float
    generation_ms: float
    guardrail_ms: Optional[float] = 0.0
    stt_ms: Optional[float] = 0.0
    tts_ms: Optional[float] = 0.0
    total_pipeline_ms: float

class RAGResponse(BaseModel):
    query: str
    answer: str
    context: str
    sources: List[SourceDoc]
    retrieved_chunks_count: int
    guardrail: Dict[str, Any]
    latency: LatencyMetrics
    sub_200ms_target_met: bool
    audio_base64: Optional[str] = None
    audio_format: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    collection_name: str
    bm25_docs: int
    qdrant_status: str
    timestamp: str
