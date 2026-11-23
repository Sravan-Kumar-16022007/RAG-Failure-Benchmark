"""FAISS-based retrieval engine with local persistence."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import faiss
import numpy as np

logger = logging.getLogger(__name__)


class FAISSRetriever:
    """Simple FAISS index for cosine similarity search with save/load capabilities."""

    def __init__(self, embedding_dim: int):
        # IndexFlatIP with normalized vectors = Cosine Similarity
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.doc_ids: list[str] = []

    def add(self, embeddings: np.ndarray, doc_ids: list[str]) -> None:
        """Add embeddings and their corresponding document IDs to the index."""
        if embeddings.shape[0] != len(doc_ids):
            raise ValueError("Number of embeddings must match number of doc_ids")
        self.index.add(embeddings)
        self.doc_ids.extend(doc_ids)
        logger.info("Added %d vectors to FAISS index", embeddings.shape[0])

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[tuple[str, float]]:
        """Search for top-k similar documents. Returns list of (doc_id, score)."""
        if self.index.ntotal == 0:
            raise RuntimeError("FAISS index is empty")
        
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        scores, indices = self.index.search(query_embedding, top_k)
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx != -1:  # FAISS returns -1 if not enough vectors
                results.append((self.doc_ids[idx], float(score)))
        return results

    def save(self, index_path: Path, id_map_path: Path) -> None:
        """Save the FAISS index and the document ID mapping to disk."""
        index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_path))
        
        with id_map_path.open("w", encoding="utf-8") as fh:
            json.dump(self.doc_ids, fh)
            
        logger.info("Saved FAISS index to %s and ID map to %s", index_path, id_map_path)

    @classmethod
    def load(cls, index_path: Path, id_map_path: Path) -> FAISSRetriever:
        """Load a pre-built FAISS index and ID mapping from disk."""
        if not index_path.exists() or not id_map_path.exists():
            raise FileNotFoundError(f"Artifacts not found at {index_path} or {id_map_path}")
            
        index = faiss.read_index(str(index_path))
        retriever = cls(embedding_dim=index.d)
        retriever.index = index
        
        with id_map_path.open("r", encoding="utf-8") as fh:
            retriever.doc_ids = json.load(fh)
            
        logger.info("Loaded FAISS index with %d vectors from %s", index.ntotal, index_path)
        return retriever