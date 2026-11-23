"""Pydantic v2 schema for RAG-FailureBench examples.

Single source of truth for the dataset contract. All pipeline code and all
dataset files must validate against RAGExample before being written to disk.
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator


class FailureLabel(str, Enum):
    """The seven canonical failure labels (see docs/annotation_guide.md)."""

    HALLUCINATION = "hallucination"
    RETRIEVAL_FAILURE = "retrieval_failure"
    CONTRADICTION = "contradiction"
    WRONG_CITATION = "wrong_citation"
    REASONING_ERROR = "reasoning_error"
    PARTIAL_ANSWER = "partial_answer"
    NO_FAILURE = "no_failure"


SEVERITY_LEVELS = ("low", "medium", "high", "critical")
ANNOTATION_METHODS = ("human", "llm_judge", "synthetic_rule", "hybrid")
SPLITS = ("train", "validation", "test")
QUESTION_TYPES = ("single_hop", "multi_hop", "comparison", "unanswerable")


class RetrievedContext(BaseModel):
    context_id: str = Field(..., pattern=r"^ctx_\d{3,}$")
    document_id: str = Field(..., pattern=r"^doc_\d{4}$")
    rank: int = Field(..., ge=1)
    retrieval_score: float | None = Field(None, ge=0.0, le=1.0)
    text: str = Field(..., min_length=1)
    is_relevant: bool


class CitationSpan(BaseModel):
    claim: str = Field(..., min_length=1)
    # Allow empty list: a claim might have no citations (e.g., in retrieval failures)
    citation_context_ids: list[str] = Field(default_factory=list)
    is_supported: bool
    evidence_explanation: str | None = None


class RAGExample(BaseModel):
    """One labeled RAG benchmark example."""

    example_id: str = Field(..., pattern=r"^rag_\d{6}$")
    domain: str = Field(..., min_length=1)
    question_type: str
    question: str = Field(..., min_length=1)
    document_id: str = Field(..., pattern=r"^doc_\d{4}$")
    document_title: str = Field(..., min_length=1)
    source_url: str | None = None
    source_license: str = Field("synthetic", min_length=1)
    ground_truth_answer: str = Field(..., min_length=1)
    required_facts: list[str] = Field(..., min_length=1)

    retrieved_contexts: list[RetrievedContext] = Field(..., min_length=1)
    generated_answer: str = Field(..., min_length=1)
    answer_citations: list[str] = Field(default_factory=list)
    citation_spans: list[CitationSpan] = Field(default_factory=list)

    primary_failure_label: FailureLabel
    secondary_failure_labels: list[FailureLabel] = Field(default_factory=list)
    error_explanation: str = Field(..., min_length=1)

    context_precision: float | None = Field(None, ge=0.0, le=1.0)
    context_recall: float | None = Field(None, ge=0.0, le=1.0)
    faithfulness_score: float | None = Field(None, ge=0.0, le=1.0)
    answer_relevance_score: float | None = Field(None, ge=0.0, le=1.0)
    answer_correctness_score: float | None = Field(None, ge=0.0, le=1.0)
    citation_correctness_score: float | None = Field(None, ge=0.0, le=1.0)

    severity: str
    annotation_method: str
    annotator_id: str | None = None
    split: str

    @field_validator("question_type")
    @classmethod
    def _valid_question_type(cls, v: str) -> str:
        if v not in QUESTION_TYPES:
            raise ValueError(f"question_type must be one of {QUESTION_TYPES}, got {v!r}")
        return v

    @field_validator("severity")
    @classmethod
    def _valid_severity(cls, v: str) -> str:
        if v not in SEVERITY_LEVELS:
            raise ValueError(f"severity must be one of {SEVERITY_LEVELS}, got {v!r}")
        return v

    @field_validator("annotation_method")
    @classmethod
    def _valid_annotation_method(cls, v: str) -> str:
        if v not in ANNOTATION_METHODS:
            raise ValueError(f"annotation_method must be one of {ANNOTATION_METHODS}, got {v!r}")
        return v

    @field_validator("split")
    @classmethod
    def _valid_split(cls, v: str) -> str:
        if v not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {v!r}")
        return v

    @field_validator("required_facts")
    @classmethod
    def _facts_non_empty(cls, v: list[str]) -> list[str]:
        for fact in v:
            if not fact or not fact.strip():
                raise ValueError("required_facts entries must be non-empty strings")
        return v

    @field_validator("question", "ground_truth_answer", "generated_answer", "error_explanation")
    @classmethod
    def _no_whitespace_only(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("field cannot be empty or whitespace only")
        return v

    @model_validator(mode="after")
    def _check_secondary_labels(self) -> RAGExample:
        seen = set()
        for label in self.secondary_failure_labels:
            if label == self.primary_failure_label:
                raise ValueError(
                    f"primary label {self.primary_failure_label.value!r} must not appear in secondary labels"
                )
            if label in seen:
                raise ValueError(f"duplicate secondary label {label.value!r}")
            seen.add(label)
        return self

    @model_validator(mode="after")
    def _check_context_and_citation_integrity(self) -> RAGExample:
        ctx_ids = [c.context_id for c in self.retrieved_contexts]
        if len(ctx_ids) != len(set(ctx_ids)):
            raise ValueError("duplicate context_id in retrieved_contexts")
        ranks = [c.rank for c in self.retrieved_contexts]
        if len(ranks) != len(set(ranks)):
            raise ValueError("duplicate rank in retrieved_contexts")

        ctx_id_set = set(ctx_ids)
        for cid in self.answer_citations:
            if cid not in ctx_id_set:
                raise ValueError(f"citation {cid!r} does not exist in retrieved_contexts")
        for span in self.citation_spans:
            for cid in span.citation_context_ids:
                if cid not in ctx_id_set:
                    raise ValueError(f"citation_span references unknown context {cid!r}")
        return self