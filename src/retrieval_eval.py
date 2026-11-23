"""Run retrieval evaluation on the normalized dataset using pre-built FAISS artifacts."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from .config import load_config, get_paths
from .embeddings import EmbeddingModel
from .metrics import evaluate_retrieval
from .retrieval import FAISSRetriever

logger = logging.getLogger(__name__)


def load_dataset(jsonl_path: Path) -> List[Dict[str, Any]]:
    """Load JSONL dataset."""
    records = []
    with jsonl_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            records.append(json.loads(line))
    return records


def run_evaluation(dataset_path: Path, artifact_dir: Path, output_path: Path) -> None:
    """Main evaluation pipeline using local artifacts."""
    logger.info("Loading dataset from %s", dataset_path)
    records = load_dataset(dataset_path)
    logger.info("Loaded %d records", len(records))

    # 1. Initialize Embedding Model from local artifacts
    local_model_path = artifact_dir / "model"
    embedder = EmbeddingModel(local_model_path=local_model_path, device="cpu")

    # 2. Load Pre-built FAISS Index
    index_path = artifact_dir / "index.faiss"
    id_map_path = artifact_dir / "id_map.json"
    retriever = FAISSRetriever.load(index_path, id_map_path)

    # 3. Run Retrieval for each query
    all_retrieved = []
    all_relevant = []
    top_k = 10

    logger.info("Running retrieval for %d queries...", len(records))
    for rec in records:
        query = rec["question"]
        relevant_doc_id = rec["document_id"]
        
        # Encode ONLY the query (very fast on CPU)
        query_embedding = embedder.encode([query])[0]
        
        # Search the pre-built index
        results = retriever.search(query_embedding, top_k=top_k)
        retrieved_doc_ids = [doc_id for doc_id, _ in results]
        
        all_retrieved.append(retrieved_doc_ids)
        all_relevant.append(relevant_doc_id)

    # 4. Calculate Metrics
    metrics = evaluate_retrieval(all_retrieved, all_relevant, k_values=[1, 5, 10])
    
    print("\n" + "="*40)
    print("🎯 Retrieval Evaluation Results")
    print("="*40)
    for metric, value in metrics.items():
        print(f"{metric:<12}: {value:.4f}")
    print("="*40 + "\n")

    # 5. Save Results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    logger.info("Saved metrics to %s", output_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = load_config()
    paths = get_paths(cfg)
    
    dataset_path = paths.processed / "rag_failurebench_full.jsonl"
    artifact_dir = paths.root / cfg["artifacts"]["dir"]
    output_path = paths.reports / "retrieval_metrics.json"
    
    run_evaluation(dataset_path, artifact_dir, output_path)