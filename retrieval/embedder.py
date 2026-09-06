"""Thin wrapper around a small sentence-transformers model.

Kept isolated so the rest of the codebase (index building, fusion, API)
depends on a narrow encode() interface rather than sentence-transformers
directly.
"""
from __future__ import annotations

import numpy as np

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # 22M params, 384-dim


class Embedder:
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: list[str], batch_size: int = 64, show_progress_bar: bool = False) -> np.ndarray:
        """Returns L2-normalized float32 embeddings, shape (len(texts), dim).

        L2-normalizing at encode time means inner product == cosine
        similarity, so the FAISS index can use a plain IndexFlatIP.
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings.astype(np.float32)
