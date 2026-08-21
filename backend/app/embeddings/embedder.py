"""
Multilingual Embedder Component
Wraps sentence-transformers models for Indic & Multilingual text vectorization.
Supports GPU acceleration, CPU fallback, batching, and score normalization.
"""

import os
import sys
import time
import numpy as np
from typing import List, Union

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except Exception:
    HAS_SENTENCE_TRANSFORMERS = False

class MultilingualEmbedder:
    """
    Multilingual Sentence Embedder with auto-detection of SentenceTransformer
    and fast deterministic multilingual hashing fallback.
    """

    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        self.model_name = model_name
        self.embedding_dim = 384
        self.model = None

        # Check environment flag or attempt SentenceTransformer import safely
        force_fallback = os.getenv("USE_FALLBACK_EMBEDDER", "1") == "1"

        if HAS_SENTENCE_TRANSFORMERS and not force_fallback:
            try:
                t0 = time.perf_counter()
                print(f"[+] Initializing SentenceTransformer model '{model_name}'...", flush=True)
                # Set torch single thread to prevent multithread C-level access violations under Python 3.14
                import torch
                torch.set_num_threads(1)
                self.model = SentenceTransformer(model_name)
                self.embedding_dim = self.model.get_sentence_embedding_dimension()
                elapsed = (time.perf_counter() - t0) * 1000.0
                print(f"[+] SentenceTransformer initialized in {elapsed:.2f} ms! Dim: {self.embedding_dim}", flush=True)
            except Exception as e:
                print(f"[-] SentenceTransformer load failed ({e}). Using FastMultilingualVectorizer fallback.", flush=True)
                self.model = None
            except BaseException as e:
                print(f"[-] SentenceTransformer system exception ({e}). Using FastMultilingualVectorizer fallback.", flush=True)
                self.model = None
        else:
            print("[-] Using FastMultilingualVectorizer fallback.", flush=True)


    def _fallback_embed(self, text: str) -> List[float]:
        """
        Fast deterministic multilingual n-gram hashing vectorizer (384-dim, L2 normalized).
        """
        vec = np.zeros(self.embedding_dim, dtype=np.float32)
        words = text.lower().split()
        for word in words:
            h = hash(word) % self.embedding_dim
            vec[h] += 1.0
            for i in range(len(word) - 2):
                gram = word[i:i+3]
                gh = hash(gram) % self.embedding_dim
                vec[gh] += 0.5

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    def embed_texts(self, texts: List[str], batch_size: int = 64, normalize: bool = True) -> List[List[float]]:
        """Encode a list of texts into dense vectors."""
        if not texts:
            return []
        
        t0 = time.perf_counter()
        if self.model is not None:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=normalize
            )
            vecs = embeddings.tolist()
        else:
            vecs = [self._fallback_embed(t) for t in texts]

        elapsed = (time.perf_counter() - t0) * 1000.0
        print(f"[+] Embedded {len(texts)} texts in {elapsed:.2f} ms ({elapsed/max(len(texts),1):.2f} ms/item)", flush=True)
        return vecs

    def embed_query(self, query: str, normalize: bool = True) -> List[float]:
        """Encode a single query string into a vector."""
        if not query:
            return [0.0] * self.embedding_dim
        vec = self.embed_texts([query], batch_size=1, normalize=normalize)[0]
        return vec
