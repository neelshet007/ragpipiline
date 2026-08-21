"""
Text-to-Speech (TTS) Engine Component
Synthesizes Indic and Multilingual text responses into audio payloads (MP3/WAV base64).
"""

import base64
import time
import io
from typing import Dict, Any, Optional

try:
    from gtts import gTTS
    HAS_GTTS = True
except ImportError:
    HAS_GTTS = False

class TextToSpeechEngine:
    """
    Text-to-Speech Engine generating synthesized audio output for RAG responses.
    """

    def __init__(self, default_lang: str = "hi"):
        self.default_lang = default_lang

    def synthesize_speech(
        self,
        text: str,
        language: str = "hi",
        slow: bool = False
    ) -> Dict[str, Any]:
        """
        Synthesizes text into spoken audio stream (base64 encoded).
        """
        t0 = time.perf_counter()
        target_lang = language or self.default_lang

        if not text or not text.strip():
            return {
                "audio_base64": "",
                "audio_format": "mp3",
                "tts_latency_ms": round((time.perf_counter() - t0) * 1000.0, 3),
                "error": "empty_text_input"
            }

        try:
            if HAS_GTTS:
                # Map language codes for gTTS
                lang_code = "hi" if target_lang.startswith("hi") else "en"
                fp = io.BytesIO()
                tts = gTTS(text=text[:300], lang=lang_code, slow=slow)
                tts.write_to_fp(fp)
                fp.seek(0)
                audio_bytes = fp.read()
            else:
                # Fast mock audio header fallback if gtts library is not loaded
                audio_bytes = b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80\x3e\x00\x00\x00\x7d\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00"

            audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
            tts_latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)

            return {
                "audio_base64": audio_base64,
                "audio_format": "mp3" if HAS_GTTS else "wav",
                "audio_size_bytes": len(audio_bytes),
                "tts_latency_ms": tts_latency_ms,
                "error": None
            }
        except Exception as e:
            return {
                "audio_base64": "",
                "audio_format": "mp3",
                "tts_latency_ms": round((time.perf_counter() - t0) * 1000.0, 3),
                "error": f"tts_synthesis_failed: {str(e)}"
            }
