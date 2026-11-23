"""Evaluate the failure classification baseline."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from .classification import FailureClassifier
from .config import load_config, get_paths

logger = logging.getLogger(__name__)


def load_split(dataset_path: Path, split: str) -> List[Dict[str, Any]]:
    """Load records belonging to a specific split."""
    records = []
    with dataset_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            if rec.get("split") == split:
                records.append(rec)
    return records


def plot_confusion_matrix(y_true: List[str], y_pred: List[str], classes: List[str], save_path: Path) -> None:
    """Generate and save a confusion matrix chart."""
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title("Confusion Matrix: RAG Failure Classification")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info("Saved confusion matrix to %s", save_path)


def run_classification_eval(dataset_path: Path, model_save_path: Path, chart_save_path: Path) -> None:
    """Main classification evaluation pipeline."""
    logger.info("Loading train and test splits...")
    train_records = load_split(dataset_path, "train")
    test_records = load_split(dataset_path, "test")
    
    logger.info("Train size: %d, Test size: %d", len(train_records), len(test_records))

    # 1. Train
    clf = FailureClassifier()
    clf.fit(train_records)
    clf.save(model_save_path)

    # 2. Predict
    logger.info("Predicting on test set...")
    y_true = [rec["primary_failure_label"] for rec in test_records]
    y_pred = clf.predict(test_records)

    # 3. Evaluate
    classes = sorted(list(set(y_true)))
    report = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
    
    print("\n" + "="*50)
    print("🎯 Classification Evaluation Results (Test Set)")
    print("="*50)
    print(classification_report(y_true, y_pred, target_names=classes))
    print("="*50 + "\n")

    # 4. Save Report
    report_path = chart_save_path.parent / "classification_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
        
    # 5. Plot Confusion Matrix
    plot_confusion_matrix(y_true, y_pred, classes, chart_save_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = load_config()
    paths = get_paths(cfg)
    
    # Note: For a true end-to-end test, we should use the generated dataset.
    # For now, we evaluate on the base dataset to establish the baseline.
    dataset_path = paths.processed / "rag_failurebench_full.jsonl"
    model_save_path = paths.predictions / "failure_classifier.joblib"
    chart_save_path = paths.charts / "confusion_matrix.png"
    
    run_classification_eval(dataset_path, model_save_path, chart_save_path)