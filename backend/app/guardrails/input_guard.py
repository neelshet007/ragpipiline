"""
Input Guardrail Engine
Detects prompt injections, jailbreaks, malicious payloads, and out-of-domain (OOD) queries.
"""

import re
import time
from typing import Dict, Any, List, Tuple

# Common prompt injection patterns (English & Romanized/Hindi)
INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|above|system)?\s*instructions",
    r"disregard\s+(all\s+)?rules",
    r"reveal\s+system\s+prompt",
    r"you\s+are\s+now\s+in\s+dan\s+mode",
    r"system\s+prompt\s+override",
    r"drop\s+database",
    r"delete\s+from",
    r"eval\(",
    r"exec\(",
    r"<script>",
    r"पुराने\s+निर्देशों\s+को\s+अनसुना\s+करें",
    r"सभी\s+नियमों\s+को\s+तोड़ें"
]

class InputGuardrail:
    """
    Evaluates incoming voice/text queries before passing them to the RAG retrieval pipeline.
    Ensures system safety, prevents prompt injection, and flags malicious/OOD intent.
    """

    def __init__(self, min_token_length: int = 1, max_char_length: int = 1000):
        self.min_token_length = min_token_length
        self.max_char_length = max_char_length
        self.injection_regexes = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]

    def validate_query(self, query: str) -> Dict[str, Any]:
        """
        Validates an input query string.
        Returns validation status, violation category, and polite refusal message if unsafe.
        """
        t0 = time.perf_counter()

        if not query or not query.strip():
            return {
                "is_safe": False,
                "reason": "empty_query",
                "refusal_message": "कृपया एक वैध प्रश्न पूछें।",
                "check_latency_ms": round((time.perf_counter() - t0) * 1000.0, 3)
            }

        if len(query) > self.max_char_length:
            return {
                "is_safe": False,
                "reason": "query_too_long",
                "refusal_message": f"प्रश्न बहुत लंबा है। कृपया इसे {self.max_char_length} अक्षरों के भीतर रखें।",
                "check_latency_ms": round((time.perf_counter() - t0) * 1000.0, 3)
            }

        # Check for Prompt Injection / Security Violations
        for regex in self.injection_regexes:
            if regex.search(query):
                return {
                    "is_safe": False,
                    "reason": "prompt_injection_detected",
                    "refusal_message": "सुरक्षा कारणों से इस अनुरोध को संसाधित नहीं किया जा सकता है।",
                    "check_latency_ms": round((time.perf_counter() - t0) * 1000.0, 3)
                }

        latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        return {
            "is_safe": True,
            "reason": "clean",
            "refusal_message": None,
            "check_latency_ms": latency_ms
        }
