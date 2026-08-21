"""
FastAPI WebSocket Handler Component
Supports bi-directional real-time audio and text streaming for live voice RAG interaction.
"""

import json
import time
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from backend.app.pipeline.rag_core import RAGCorePipeline
from backend.app.voice.stt import SpeechToTextEngine
from backend.app.voice.tts import TextToSpeechEngine

ws_router = APIRouter()

@ws_router.websocket("/ws/rag")
async def websocket_rag_endpoint(websocket: WebSocket):
    """
    WebSocket Streaming Endpoint:
    Receives JSON frame: {"type": "text"|"audio", "payload": "...", "language": "hi"}
    Emits real-time stream frames with incremental status & audio response.
    """
    await websocket.accept()
    print("[+] WebSocket connection established on /ws/rag")

    pipeline = RAGCorePipeline()
    stt_engine = SpeechToTextEngine()
    tts_engine = TextToSpeechEngine()

    try:
        while True:
            raw_data = await websocket.receive_text()
            t_start = time.perf_counter()

            try:
                msg = json.loads(raw_data)
            except Exception:
                await websocket.send_json({"error": "invalid_json_payload"})
                continue

            frame_type = msg.get("type", "text")
            payload = msg.get("payload", "")
            language = msg.get("language", "hi")
            top_k = msg.get("top_k", 3)

            await websocket.send_json({
                "status": "processing",
                "frame_type": frame_type,
                "message": "query_received"
            })

            # STT
            if frame_type == "audio":
                stt_res = stt_engine.process_speech_input(audio_base64=payload, language=language)
                query_text = stt_res["query_text"]
            else:
                stt_res = {"stt_latency_ms": 0.0}
                query_text = payload

            # RAG Pipeline
            await websocket.send_json({
                "status": "retrieving",
                "query_text": query_text
            })

            rag_res = pipeline.process_query(query=query_text, top_k=top_k, lang_filter=language)

            # TTS Synthesis
            tts_res = tts_engine.synthesize_speech(rag_res["answer"], language=language)

            total_ms = round((time.perf_counter() - t_start) * 1000.0, 3)
            rag_res["latency"]["stt_ms"] = stt_res["stt_latency_ms"]
            rag_res["latency"]["tts_ms"] = tts_res["tts_latency_ms"]
            rag_res["latency"]["total_pipeline_ms"] = total_ms
            rag_res["audio_base64"] = tts_res["audio_base64"]
            rag_res["status"] = "completed"

            await websocket.send_json(rag_res)

    except WebSocketDisconnect:
        print("[-] WebSocket client disconnected from /ws/rag")
    except Exception as e:
        print(f"[-] WebSocket error: {e}")
        try:
            await websocket.send_json({"status": "error", "error": str(e)})
        except Exception:
            pass
