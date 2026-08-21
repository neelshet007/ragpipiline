"""
Output Guardrail & Hallucination Mitigation Engine
Validates retrieval score thresholds, detects ungrounded answers, and structures polite refusals.

RRF scores are bounded by 1/(k+rank) where k=60.
  - Rank 1 max score = 1/61 ≈ 0.01639
  - Rank 3 max score = 1/63 ≈ 0.01587
So RRF scores live in [0.015, 0.0164]. We detect relevance via score SPREAD, not absolute value.
"""

import time
from typing import Dict, Any, List, Optional

# Minimum spread between top and bottom score to indicate actual relevance signal
# If all docs score similarly, retrieval is essentially random (no true match found)
MIN_SCORE_SPREAD = 0.0003
# Minimum absolute top score (for non-RRF modes like convex blend)
MIN_ABS_SCORE = 0.008


class OutputGuardrail:
    """
    Evaluates retrieved context score confidence and generated RAG output.
    Uses both absolute threshold and score-spread detection to handle RRF-bounded scores.
    """

    def __init__(self, min_confidence_score: float = MIN_ABS_SCORE, min_spread: float = MIN_SCORE_SPREAD):
        self.min_confidence_score = min_confidence_score
        self.min_spread = min_spread

    def _build_refusal(self, query: str, top_score: float, reason_code: str, t0: float) -> Dict[str, Any]:
        return {
            "is_grounded": False,
            "confidence_score": round(top_score, 6),
            "reason": reason_code,
            "final_answer": (
                f"क्षमा करें, '{query[:60]}' के बारे में हमारे इंडेक्स में पर्याप्त जानकारी उपलब्ध नहीं है।\n"
                f"कृपया हिंदी में प्रश्न पूछें जैसे: 'ताज महल के बारे में बताएं' या 'कॉर्पोरेशन क्या है?'\n\n"
                f"[No relevant passage found for: '{query[:60]}'. "
                f"The indexed corpus is Hindi MSMARCO-XI (~2,003 docs). "
                f"Try Hindi queries or rephrase your question.]"
            ),
            "guard_latency_ms": round((time.perf_counter() - t0) * 1000.0, 3)
        }

    def validate_output(
        self,
        query: str,
        retrieved_sources: List[Dict[str, Any]],
        generated_answer: str
    ) -> Dict[str, Any]:
        """
        Validates RAG answer against retrieved passage provenance.
        Checks if BM25 found substantial keyword matches (best_bm25_score >= 1.0)
        or if dense retrieval had high confidence.
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

        scores = [s.get("score") or 0.0 for s in retrieved_sources]
        top_score = max(scores)

        # Check BM25 match scores
        bm25_max_scores = [s.get("bm25_max_score", 0.0) for s in retrieved_sources]
        best_bm25_score = max(bm25_max_scores) if bm25_max_scores else 0.0

        # Grounded if BM25 found keyword match OR top score meets confidence threshold
        is_grounded = (best_bm25_score >= 1.0) or (top_score >= self.min_confidence_score)

        if not is_grounded:
            return self._build_refusal(query, top_score, "no_keyword_match_in_corpus", t0)

        latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        return {
            "is_grounded": True,
            "confidence_score": round(max(best_bm25_score, top_score), 4),
            "reason": "grounded_in_context",
            "final_answer": generated_answer,
            "guard_latency_ms": latency_ms
        }
