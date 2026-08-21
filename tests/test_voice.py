"""
Unit Tests for Speech-to-Text (STT) and Text-to-Speech (TTS) Modules
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.app.voice.stt import SpeechToTextEngine
from backend.app.voice.tts import TextToSpeechEngine

def test_stt_web_speech_transcript():
    stt = SpeechToTextEngine()
    res = stt.process_speech_input(text_transcript="भारत की राजधानी क्या है?", language="hi")
    assert res["query_text"] == "भारत की राजधानी क्या है?"
    assert res["input_type"] == "web_speech_api"
    assert res["stt_latency_ms"] >= 0.0

def test_stt_audio_base64():
    stt = SpeechToTextEngine()
    dummy_b64 = "UklGRiQAAABXQVZFZm10IBAAAAABAAEARKwAAIhYAQACABAAZGF0YQAAAAA="
    res = stt.process_speech_input(audio_base64=dummy_b64, language="hi")
    assert len(res["query_text"]) > 0
    assert res["input_type"] == "audio_base64"

def test_tts_synthesis():
    tts = TextToSpeechEngine()
    res = tts.synthesize_speech(text="नमस्ते, आप कैसे हैं?", language="hi")
    assert len(res["audio_base64"]) > 0
    assert res["tts_latency_ms"] >= 0.0
    assert res["error"] is None
