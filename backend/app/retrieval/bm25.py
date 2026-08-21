"""
BM25 Sparse Lexical Retriever Component
Implements Okapi BM25 indexer with Indic text tokenization and fast keyword searching.
"""

import math
import time
import re
from collections import Counter
from typing import List, Dict, Any, Optional

# Common Indic & English Stopwords to prevent spurious lexical matches
STOPWORDS = {
    # Hindi Stopwords
    "क्या", "है", "हैं", "था", "थी", "थे", "का", "की", "के", "में", "से", "पर", "को", "और", "या",
    "यह", "वह", "ये", "वे", "जो", "एक", "तो", "भी", "नहीं", "होता", "होती", "होते", "करने", "करता",
    "करती", "करते", "किया", "गया", "गई", "गए", "लिए", "अपने", "अपनी", "अपना", "इस", "उस", "इन",
    "उन", "कहा", "कहते", "सकता", "सकती", "सकते", "पास", "हुए", "हुआ", "हुई", "द्वारा", "तक",
    # English Stopwords
    "what", "is", "are", "was", "were", "the", "a", "an", "in", "on", "of", "to", "for",
    "and", "or", "how", "why", "where", "when", "which", "who", "whom", "this", "that",
    "these", "those", "it", "its", "do", "does", "did", "have", "has", "had", "be", "been"
}


class BM25Retriever:
    """
    Okapi BM25 Sparse Lexical Retriever.
    Supports Devanagari/Indic tokenization, term frequency indexing, and sub-millisecond search.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_count = 0
        self.avg_doc_len = 0.0
        self.doc_lengths: List[int] = []
        self.doc_payloads: List[Dict[str, Any]] = []
        self.doc_term_freqs: List[Counter] = []
        self.doc_freqs: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}

    def _tokenize(self, text: str, filter_stopwords: bool = False) -> List[str]:
        """
        Tokenizes Devanagari and Latin text into lowercase word tokens,
        removing punctuation and isolated symbols.
        """
        if not text:
            return []
        # Retain Devanagari script (\u0900-\u097F) and alphanumeric characters
        tokens = re.findall(r"[\u0900-\u097F\w]+", text.lower())
        if filter_stopwords:
            content_tokens = [t for t in tokens if t not in STOPWORDS]
            return content_tokens if content_tokens else tokens
        return tokens

    def fit(self, chunks: List[Dict[str, Any]]):
        """
        Builds the BM25 index over a collection of document chunks.
        """
        t0 = time.perf_counter()
        self.doc_payloads = chunks
        self.doc_count = len(chunks)
        self.doc_lengths = []
        self.doc_term_freqs = []
        self.doc_freqs = {}

        total_length = 0
        for chunk in chunks:
            text = chunk.get("chunk_text", "")
            tokens = self._tokenize(text)
            doc_len = len(tokens)
            self.doc_lengths.append(doc_len)
            total_length += doc_len

            tf = Counter(tokens)
            self.doc_term_freqs.append(tf)

            for token in tf.keys():
                self.doc_freqs[token] = self.doc_freqs.get(token, 0) + 1

        self.avg_doc_len = (total_length / max(self.doc_count, 1)) if self.doc_count > 0 else 0.0

        # Compute IDF values: log(1 + (N - df + 0.5) / (df + 0.5))
        self.idf = {}
        for token, df in self.doc_freqs.items():
            idf_val = math.log(1.0 + (self.doc_count - df + 0.5) / (df + 0.5))
            self.idf[token] = max(idf_val, 1e-4)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        print(f"[+] BM25 Index built over {self.doc_count:,} documents in {elapsed_ms:.2f} ms (Avg len: {self.avg_doc_len:.1f} tokens)", flush=True)

    def search(self, query: str, top_k: int = 10, lang_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Executes Okapi BM25 search for a given query string.
        """
        t0 = time.perf_counter()
        query_tokens = self._tokenize(query, filter_stopwords=True)
        if not query_tokens or self.doc_count == 0:
            return []

        scores = [0.0] * self.doc_count

        for q_token in query_tokens:
            if q_token not in self.idf:
                continue
            token_idf = self.idf[q_token]

            for doc_idx, tf_counter in enumerate(self.doc_term_freqs):
                if lang_filter:
                    doc_lang = self.doc_payloads[doc_idx].get("language", "hi")
                    if doc_lang != lang_filter:
                        continue

                tf = tf_counter.get(q_token, 0)
                if tf == 0:
                    continue

                doc_len = self.doc_lengths[doc_idx]
                denom = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / max(self.avg_doc_len, 1e-4)))
                numerator = tf * (self.k1 + 1.0)
                score = token_idf * (numerator / denom)
                scores[doc_idx] += score

        # Sort document indices by BM25 score descending
        top_indices = sorted(range(self.doc_count), key=lambda i: scores[i], reverse=True)[:top_k]

        latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)

        results = []
        for rank, idx in enumerate(top_indices, start=1):
            sc = scores[idx]
            if sc <= 0.0:
                continue
            payload = dict(self.doc_payloads[idx])
            payload["score"] = round(float(sc), 4)
            payload["bm25_rank"] = rank
            payload["search_latency_ms"] = latency_ms
            results.append(payload)

        return results
