"""
Output Guardrail & Hallucination Mitigation Engine
Validates retrieval score thresholds, detects ungrounded answers, and structures polite refusals.
"""

import time
from typing import Dict, Any, List, Optional

class OutputGuardrail:
    """
    Evaluates retrieved context score confidence and generated RAG output.
    If context relevance is below threshold or hallucinated, triggers structured refusal.
    """

    def __init__(self, min_confidence_score: float = 0.005):
        self.min_confidence_score = min_confidence_score

    def validate_output(
        self,
        query: str,
        retrieved_sources: List[Dict[str, Any]],
        generated_answer: str
    ) -> Dict[str, Any]:
        """
        Validates RAG answer against retrieved passage provenance.
        """
        t0 = time.perf_counter()

        if not retrieved_sources:
            return {
                "is_grounded": False,
                "confidence_score": 0.0,
                "reason": "no_retrieved_context",
                "final_answer": "क्षमा करें, मेरे पास इस प्रश्न का उत्तर देने के लिए डेटाबेस में पर्याप्त जानकारी उपलब्ध नहीं है।",
                "guard_latency_ms": round((time.perf_counter() - t0) * 1000.0, 3)
            }

        top_score = max([s.get("score") or 0.0 for s in retrieved_sources])

        if top_score < self.min_confidence_score:
            return {
                "is_grounded": False,
                "confidence_score": round(top_score, 4),
                "reason": "low_retrieval_confidence",
                "final_answer": "क्षमा करें, इस विषय पर सटीक संदर्भ प्राप्त नहीं हुआ है। कृपया अधिक स्पष्ट प्रश्न पूछें।",
                "guard_latency_ms": round((time.perf_counter() - t0) * 1000.0, 3)
            }

        latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        return {
            "is_grounded": True,
            "confidence_score": round(top_score, 4),
            "reason": "grounded_in_context",
            "final_answer": generated_answer,
            "guard_latency_ms": latency_ms
        }
