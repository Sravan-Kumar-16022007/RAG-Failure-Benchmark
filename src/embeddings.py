"""Embedding utilities using sentence-transformers."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Default model name if downloading from Hugging Face
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


class EmbeddingModel:
    """Wrapper around sentence-transformers for consistent encoding."""

    def __init__(self, local_model_path: Path | None = None, device: str = "cpu"):
        """Initialize the embedding model.
        
        Args:
            local_model_path: Path to a locally saved model directory. 
                              If None, downloads from Hugging Face.
            device: 'cpu' or 'cuda'. Defaults to 'cpu' for local laptop inference.
        """
        if local_model_path and local_model_path.exists():
            logger.info("Loading local embedding model from: %s", local_model_path)
            self.model = SentenceTransformer(str(local_model_path), device=device)
            self.model_name = local_model_path.name
        else:
            logger.info("Downloading/loading embedding model: %s", DEFAULT_MODEL_NAME)
            self.model = SentenceTransformer(DEFAULT_MODEL_NAME, device=device)
            self.model_name = DEFAULT_MODEL_NAME

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """Encode a list of texts into normalized embeddings."""
        logger.info("Encoding %d texts...", len(texts))
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,  # Required for Cosine Similarity via Inner Product
        )
        return np.array(embeddings, dtype=np.float32)