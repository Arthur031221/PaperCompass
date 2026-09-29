"""Embedding backend: sentence-transformers all-MiniLM-L6-v2 on CPU.

The model is loaded lazily on first use so importing this module, or building
a Database and API app around it, never triggers a download or a slow load.
Tests inject a fake embedder instead of touching the real model, see
tests/conftest.py.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class Embedder(Protocol):
    """Anything that turns a list of strings into an (n, dim) float32 array."""

    def encode(self, texts: list[str]) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    """Real embedder backed by sentence-transformers, CPU only."""

    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            # Imported lazily: this pulls in torch, which is slow to import
            # and unnecessary for any code path that never embeds text.
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device="cpu")
        return self._model

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)
        model = self._load()
        vectors = model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity between rows of a (n, d) and rows of b (m, d) -> (n, m).

    Assumes inputs may not be pre-normalized.
    """
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if b.ndim == 1:
        b = b.reshape(1, -1)
    a_norm = a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-8, None)
    b_norm = b / np.clip(np.linalg.norm(b, axis=1, keepdims=True), 1e-8, None)
    # Apple's Accelerate BLAS sets spurious divide-by-zero/overflow FP flags on
    # some small float32 matmuls even though the result has no NaN or Inf in
    # it (verified: values stay in [-1, 1] as expected for unit vectors).
    # Silence the resulting RuntimeWarning rather than let it spam logs.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        return a_norm @ b_norm.T
