"""Ембедінги: перетворення тексту на вектор.

Модель локальна (fastembed, ONNX на CPU): без ключів і без плати за токени,
працює офлайн, той самий вектор для того самого тексту. Мультимовна
MiniLM-L12 важить 220 МБ і дає 384-вимірні вектори. Вона слабша за великі
моделі на кшталт multilingual-e5-large (2.2 ГБ), тому пошук у проєкті
гібридний: вектори доповнює BM25 по ключових словах.
"""

from __future__ import annotations

import os

import numpy as np

MODEL_NAME = os.getenv(
    "EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
_model = None


def get_model():
    global _model
    if _model is None:
        from fastembed import TextEmbedding

        _model = TextEmbedding(
            MODEL_NAME, cache_dir=os.getenv("FASTEMBED_CACHE", ".fastembed_cache")
        )
    return _model


def embed_texts(texts: list[str], batch_size: int = 64) -> np.ndarray:
    """Повертає матрицю N×dim з нормованими векторами (довжина 1)."""
    vecs = np.array(
        list(get_model().embed(texts, batch_size=batch_size)), dtype=np.float32
    )
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.clip(norms, 1e-9, None)


def embed_query(text: str) -> np.ndarray:
    return embed_texts([text])[0]
