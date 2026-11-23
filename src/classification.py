"""Baseline failure classification model."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


class FailureClassifier:
    """TF-IDF + Logistic Regression baseline for RAG failure classification."""

    def __init__(self):
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                stop_words="english",
                lowercase=True
            )),
            ("clf", LogisticRegression(
                max_iter=1000,
                class_weight="balanced",  # Important for imbalanced datasets
                random_state=42
            ))
        ])
        self.is_trained = False

    def prepare_text(self, record: Dict[str, Any]) -> str:
        """Concatenate relevant fields into a single text string for classification."""
        question = record.get("question", "")
        context = " ".join([ctx.get("text", "") for ctx in record.get("retrieved_contexts", [])])
        answer = record.get("generated_answer", "")
        return f"Q: {question} C: {context} A: {answer}"

    def fit(self, records: List[Dict[str, Any]]) -> None:
        """Train the classifier on a list of records."""
        logger.info("Preparing training data...")
        X = [self.prepare_text(rec) for rec in records]
        y = [rec["primary_failure_label"] for rec in records]
        
        logger.info("Training classifier on %d samples...", len(X))
        self.pipeline.fit(X, y)
        self.is_trained = True
        logger.info("Training complete.")

    def predict(self, records: List[Dict[str, Any]]) -> List[str]:
        """Predict failure labels for a list of records."""
        if not self.is_trained:
            raise RuntimeError("Classifier must be trained before predicting.")
        X = [self.prepare_text(rec) for rec in records]
        return self.pipeline.predict(X).tolist()

    def predict_proba(self, records: List[Dict[str, Any]]) -> List[Dict[str, float]]:
        """Get prediction probabilities for each class."""
        if not self.is_trained:
            raise RuntimeError("Classifier must be trained before predicting.")
        X = [self.prepare_text(rec) for rec in records]
        probs = self.pipeline.predict_proba(X)
        classes = self.pipeline.classes_
        
        results = []
        for prob_row in probs:
            results.append({cls: float(p) for cls, p in zip(classes, prob_row)})
        return results

    def save(self, path: Path) -> None:
        """Save the trained pipeline to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, path)
        logger.info("Saved classifier to %s", path)

    @classmethod
    def load(cls, path: Path) -> "FailureClassifier":
        """Load a trained pipeline from disk."""
        clf = cls()
        clf.pipeline = joblib.load(path)
        clf.is_trained = True
        logger.info("Loaded classifier from %s", path)
        return clf