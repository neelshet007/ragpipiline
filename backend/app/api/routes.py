"""
FastAPI REST Routes Component
Exposes text query, voice query, benchmark reports, and health status endpoints.
"""

import os
import json
import time
from fastapi import APIRouter, HTTPException, Depends
from backend.app.api.models import QueryRequest, VoiceQueryRequest, RAGResponse, HealthResponse
from backend.app.pipeline.rag_core import RAGCorePipeline
from backend.app.voice.stt import SpeechToTextEngine
from backend.app.voice.tts import TextToSpeechEngine

router = APIRouter()

# Global Singleton Instances
_pipeline: RAGCorePipeline = None
_stt_engine: SpeechToTextEngine = None
_tts_engine: TextToSpeechEngine = None

def get_rag_pipeline() -> RAGCorePipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGCorePipeline()
    return _pipeline

def get_stt_engine() -> SpeechToTextEngine:
    global _stt_engine
    if _stt_engine is None:
        _stt_engine = SpeechToTextEngine()
    return _stt_engine

def get_tts_engine() -> TextToSpeechEngine:
    global _tts_engine
    if _tts_engine is None:
        _tts_engine = TextToSpeechEngine()
    return _tts_engine

@router.get("/health", response_model=HealthResponse)
def health_check(pipeline: RAGCorePipeline = Depends(get_rag_pipeline)):
    return {
        "status": "healthy",
        "service": "Voice-Enabled RAG System (HH Goa 2026)",
        "version": "1.0.0",
        "collection_name": pipeline.dense_retriever.collection_name,
        "bm25_docs": pipeline.bm25_retriever.doc_count,
        "qdrant_status": "connected",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

@router.post("/api/v1/query", response_model=RAGResponse)
def query_rag_pipeline(
    req: QueryRequest,
    pipeline: RAGCorePipeline = Depends(get_rag_pipeline),
    tts_engine: TextToSpeechEngine = Depends(get_tts_engine)
):
    """
    Executes REST text query against RAG Core Engine and returns response with latency audit.
    """
    try:
        rag_res = pipeline.process_query(
            query=req.query,
            top_k=req.top_k,
            fusion_mode=req.fusion_mode,
            alpha=req.alpha,
            lang_filter=req.lang_filter
        )

        # Generate audio output for answer synthesis
        tts_res = tts_engine.synthesize_speech(rag_res["answer"], language=req.lang_filter or "hi")
        rag_res["latency"]["tts_ms"] = tts_res["tts_latency_ms"]
        rag_res["audio_base64"] = tts_res["audio_base64"]
        rag_res["audio_format"] = tts_res["audio_format"]

        return rag_res
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG query execution failed: {str(e)}")

@router.post("/api/v1/voice", response_model=RAGResponse)
def voice_rag_pipeline(
    req: VoiceQueryRequest,
    pipeline: RAGCorePipeline = Depends(get_rag_pipeline),
    stt_engine: SpeechToTextEngine = Depends(get_stt_engine),
    tts_engine: TextToSpeechEngine = Depends(get_tts_engine)
):
    """
    Executes Voice RAG query pipeline: Audio/Transcript -> STT -> RAG -> TTS -> Audio Response.
    """
    t_start = time.perf_counter()

    # Step 1: STT Processing
    stt_res = stt_engine.process_speech_input(
        text_transcript=req.text_transcript,
        audio_base64=req.audio_base64,
        language=req.language
    )

    query_text = stt_res["query_text"]
    if not query_text:
        query_text = "क्या आप फिर से बोल सकते हैं?"

    # Step 2: RAG Pipeline Execution
    rag_res = pipeline.process_query(
        query=query_text,
        top_k=req.top_k,
        lang_filter=req.language
    )

    # Step 3: TTS Synthesis
    tts_res = tts_engine.synthesize_speech(rag_res["answer"], language=req.language)

    # Attach STT & TTS timing metrics
    rag_res["latency"]["stt_ms"] = stt_res["stt_latency_ms"]
    rag_res["latency"]["tts_ms"] = tts_res["tts_latency_ms"]
    rag_res["latency"]["total_pipeline_ms"] = round((time.perf_counter() - t_start) * 1000.0, 3)
    rag_res["sub_200ms_target_met"] = rag_res["latency"]["total_pipeline_ms"] < 200.0
    rag_res["audio_base64"] = tts_res["audio_base64"]
    rag_res["audio_format"] = tts_res["audio_format"]

    return rag_res

@router.get("/api/v1/benchmarks")
def get_benchmark_report():
    """
    Returns latest benchmark reports for latency and chunking strategies.
    """
    reports = {}
    lat_path = "benchmarks/reports/latency_report.json"
    chunk_path = "benchmarks/reports/chunking_experiment.json"

    if os.path.exists(lat_path):
        with open(lat_path, "r", encoding="utf-8") as f:
            reports["latency"] = json.load(f)

    if os.path.exists(chunk_path):
        with open(chunk_path, "r", encoding="utf-8") as f:
            reports["chunking"] = json.load(f)

    return reports
