"""Dataset validation: schema checks, uniqueness, citation integrity, and
train/validation/test leakage by document_id.

Usage:
    python -m src.validation --path data/processed/sample_rag_failurebench.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .schemas import RAGExample

logger = logging.getLogger(__name__)


@dataclass
class DatasetValidationReport:
    total: int = 0
    valid: int = 0
    invalid: int = 0
    duplicate_ids: list[str] = field(default_factory=list)
    leakage_doc_ids: list[str] = field(default_factory=list)
    errors: list[tuple[int, str, list[str]]] = field(default_factory=list)  # (line, id, msgs)
    label_distribution: dict[str, int] = field(default_factory=dict)
    split_distribution: dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.invalid == 0 and not self.duplicate_ids and not self.leakage_doc_ids

    def summary(self) -> str:
        lines = [
            f"total={self.total} valid={self.valid} invalid={self.invalid}",
            f"duplicate_ids={self.duplicate_ids}",
            f"leakage_doc_ids={self.leakage_doc_ids}",
            f"label_distribution={dict(self.label_distribution)}",
            f"split_distribution={dict(self.split_distribution)}",
        ]
        for line_no, ex_id, msgs in self.errors:
            lines.append(f"  line {line_no} [{ex_id}]: " + " | ".join(msgs))
        return "\n".join(lines)


def load_jsonl(path: Path) -> list[tuple[int, dict[str, Any]]]:
    """Read a JSONL file, returning (line_number, record) pairs. Raises on bad JSON."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    rows: list[tuple[int, dict[str, Any]]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, raw_line in enumerate(fh, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} of {path}: {exc}") from exc
            if not isinstance(obj, dict):
                raise TypeError(f"Line {line_no} of {path} is not a JSON object")
            rows.append((line_no, obj))
    return rows


def validate_example(raw: dict[str, Any]) -> list[str]:
    """Validate one record against RAGExample. Returns a list of error messages."""
    try:
        RAGExample.model_validate(raw)
        return []
    except ValidationError as exc:
        return [str(err["msg"]) for err in exc.errors()]


def check_split_leakage(examples: list[RAGExample]) -> list[str]:
    """Return document_ids that appear in more than one split."""
    doc_splits: dict[str, set] = {}
    for ex in examples:
        doc_splits.setdefault(ex.document_id, set()).add(ex.split)
    return sorted(doc for doc, splits in doc_splits.items() if len(splits) > 1)


def validate_dataset(path: Path) -> DatasetValidationReport:
    """Validate an entire JSONL dataset file. Never mutates the file."""
    report = DatasetValidationReport()
    rows = load_jsonl(path)
    report.total = len(rows)

    seen_ids: set = set()
    parsed: list[RAGExample] = []
    for line_no, raw in rows:
        ex_id = str(raw.get("example_id", "<missing-id>"))
        errors = validate_example(raw)
        if ex_id in seen_ids:
            errors.append(f"duplicate example_id {ex_id}")
        seen_ids.add(ex_id)
        if errors:
            report.invalid += 1
            report.errors.append((line_no, ex_id, errors))
        else:
            report.valid += 1
            parsed.append(RAGExample.model_validate(raw))

    report.duplicate_ids = [i for i, c in Counter(
        str(raw.get("example_id")) for _, raw in rows).items() if c > 1]
    report.leakage_doc_ids = check_split_leakage(parsed)
    report.label_distribution = dict(Counter(ex.primary_failure_label.value for ex in parsed))
    report.split_distribution = dict(Counter(ex.split for ex in parsed))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a RAG-FailureBench JSONL dataset.")
    parser.add_argument("--path", required=True, type=Path, help="Path to JSONL dataset file")
    parser.add_argument("--strict", action="store_true", help="Fail on leakage as well as invalid rows")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    report = validate_dataset(args.path)
    print(report.summary())
    if report.ok:
        print("RESULT: PASS")
        return 0
    if report.invalid == 0 and args.strict:
        print("RESULT: FAIL (strict mode: leakage detected)")
        return 1
    print("RESULT: FAIL")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())