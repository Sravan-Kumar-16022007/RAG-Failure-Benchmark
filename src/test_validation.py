"""Tests for schema + dataset validation. Run with: pytest -q"""
from __future__ import annotations

import copy
from pathlib import Path

import pytest

from src.schemas import RAGExample
from src.validation import check_split_leakage, validate_dataset, validate_example

SAMPLE_FILE = Path(__file__).resolve().parents[1] / "data" / "processed" / "sample_rag_failurebench.jsonl"


def make_valid_example() -> dict:
    return {
        "example_id": "rag_000099",
        "domain": "technology",
        "question_type": "single_hop",
        "question": "What is a vector database?",
        "document_id": "doc_0099",
        "document_title": "Synthetic Vector DB Primer",
        "source_url": None,
        "source_license": "synthetic",
        "ground_truth_answer": "It stores vector embeddings and supports similarity search.",
        "required_facts": ["stores vector embeddings", "supports similarity search"],
        "retrieved_contexts": [
            {
                "context_id": "ctx_001",
                "document_id": "doc_0099",
                "rank": 1,
                "retrieval_score": 0.9,
                "text": "A vector database stores embedding vectors and supports similarity search.",
                "is_relevant": True,
            }
        ],
        "generated_answer": "It stores embeddings and supports similarity search. [ctx_001]",
        "answer_citations": ["ctx_001"],
        "citation_spans": [
            {
                "claim": "It stores embeddings and supports similarity search.",
                "citation_context_ids": ["ctx_001"],
                "is_supported": True,
                "evidence_explanation": "Directly stated in the context.",
            }
        ],
        "primary_failure_label": "no_failure",
        "secondary_failure_labels": [],
        "error_explanation": "Answer is correct, grounded, and cited.",
        "context_precision": None,
        "context_recall": None,
        "faithfulness_score": None,
        "answer_relevance_score": None,
        "answer_correctness_score": None,
        "citation_correctness_score": None,
        "severity": "low",
        "annotation_method": "synthetic_rule",
        "annotator_id": None,
        "split": "train",
    }


def test_valid_example_passes():
    assert validate_example(make_valid_example()) == []


def test_invalid_primary_label_rejected():
    raw = make_valid_example()
    raw["primary_failure_label"] = "made_up_label"
    assert validate_example(raw)


def test_primary_label_in_secondary_rejected():
    raw = make_valid_example()
    raw["primary_failure_label"] = "hallucination"
    raw["secondary_failure_labels"] = ["hallucination"]
    assert validate_example(raw)


def test_unknown_citation_rejected():
    raw = make_valid_example()
    raw["answer_citations"] = ["ctx_999"]
    assert validate_example(raw)


def test_score_out_of_range_rejected():
    raw = make_valid_example()
    raw["faithfulness_score"] = 1.7
    assert validate_example(raw)


def test_empty_question_rejected():
    raw = make_valid_example()
    raw["question"] = "   "  # Whitespace-only is now caught by the new _no_whitespace_only validator
    assert validate_example(raw)


def test_duplicate_context_id_rejected():
    raw = make_valid_example()
    raw["retrieved_contexts"].append(copy.deepcopy(raw["retrieved_contexts"][0]))
    assert validate_example(raw)


def test_split_leakage_detected():
    ex_a = RAGExample.model_validate(make_valid_example())
    ex_b = RAGExample.model_validate({**make_valid_example(), "example_id": "rag_000098", "split": "test"})
    assert check_split_leakage([ex_a, ex_b]) == ["doc_0099"]


@pytest.mark.skipif(not SAMPLE_FILE.exists(), reason="sample dataset not present")
def test_committed_sample_file_validates():
    report = validate_dataset(SAMPLE_FILE)
    assert report.total == 10
    assert report.invalid == 0, report.summary()
    assert not report.leakage_doc_ids, report.summary()
    assert set(report.label_distribution) <= {
        "hallucination", "retrieval_failure", "contradiction",
        "wrong_citation", "reasoning_error", "partial_answer", "no_failure",
    }