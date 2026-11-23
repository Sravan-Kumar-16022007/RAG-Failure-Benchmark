"""Retrieval evaluation metrics."""
from __future__ import annotations

from typing import Dict, List


def recall_at_k(retrieved_doc_ids: List[str], relevant_doc_id: str, k: int) -> float:
    """Calculate Recall@k: 1 if relevant doc is in top-k, else 0."""
    top_k_docs = retrieved_doc_ids[:k]
    return 1.0 if relevant_doc_id in top_k_docs else 0.0


def precision_at_k(retrieved_doc_ids: List[str], relevant_doc_id: str, k: int) -> float:
    """Calculate Precision@k: 1/k if relevant doc is in top-k, else 0."""
    top_k_docs = retrieved_doc_ids[:k]
    if relevant_doc_id in top_k_docs:
        return 1.0 / k
    return 0.0


def evaluate_retrieval(
    all_retrieved: List[List[str]],  # List of lists of doc_ids for each query
    all_relevant: List[str],          # List of relevant doc_ids
    k_values: List[int] = [1, 5, 10],
) -> Dict[str, float]:
    """Calculate average Recall@k and Precision@k across all queries."""
    if len(all_retrieved) != len(all_relevant):
        raise ValueError("Number of retrieved lists must match number of relevant docs")
    
    metrics = {}
    for k in k_values:
        recalls = [recall_at_k(retrieved, relevant, k) for retrieved, relevant in zip(all_retrieved, all_relevant)]
        precisions = [precision_at_k(retrieved, relevant, k) for retrieved, relevant in zip(all_retrieved, all_relevant)]
        metrics[f"Recall@{k}"] = sum(recalls) / len(recalls)
        metrics[f"Precision@{k}"] = sum(precisions) / len(precisions)
    
    return metrics
