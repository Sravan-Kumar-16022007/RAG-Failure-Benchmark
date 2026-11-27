"""Demo server: static files + live pipeline API for the landing page.

Endpoints:
    GET  /api/dataset   -> {"records": [...]}
    GET  /api/metrics   -> retrieval + classification metrics
    POST /api/classify  -> live failure classification of {question, answer, context?, satisfaction?}

Usage:
    python app/server.py
    python app/server.py --port 8080 --no-open
"""
from __future__ import annotations

import argparse
import json
import threading
import webbrowser
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLASSIFIER_PATH = ROOT / "outputs" / "predictions" / "failure_classifier.joblib"

DATASET_CANDIDATES = [
    ROOT / "data" / "interim" / "rag_failurebench_generated_limitNone.jsonl",
    ROOT / "data" / "processed" / "rag_failurebench_full.jsonl",
]

LABEL_DISPLAY = {
    "no_failure": "Completely Successful (No Failure)",
    "hallucination": "Hallucination",
    "retrieval_failure": "Retrieval Failure",
    "contradiction": "Contradiction",
    "wrong_citation": "Wrong Citation",
    "reasoning_error": "Reasoning Error",
    "partial_answer": "Partial Answer",
}


def load_records() -> tuple[list[dict], str | None]:
    for cand in DATASET_CANDIDATES:
        if cand.exists():
            recs = [json.loads(line) for line in cand.read_text(encoding="utf-8").splitlines() if line.strip()]
            if recs:
                return recs, cand.name
    return [], None


def load_metrics() -> dict:
    metrics: dict = {}
    ret_path = ROOT / "outputs" / "reports" / "retrieval_metrics.json"
    if ret_path.exists():
        metrics["retrieval"] = json.loads(ret_path.read_text(encoding="utf-8"))
    rep_path = ROOT / "outputs" / "charts" / "classification_report.json"
    if rep_path.exists():
        rep = json.loads(rep_path.read_text(encoding="utf-8"))
        metrics["accuracy"] = rep.get("accuracy")
        metrics["macroF1"] = (rep.get("macro avg") or {}).get("f1-score")
        metrics["perClass"] = {
            key: {"p": v.get("precision"), "r": v.get("recall"), "f1": v.get("f1-score"), "support": v.get("support")}
            for key, v in rep.items()
            if key not in ("accuracy", "macro avg", "weighted avg")
        }
    return metrics


def load_classifier():
    try:
        import joblib
        if CLASSIFIER_PATH.exists():
            return joblib.load(CLASSIFIER_PATH)
    except Exception as exc:
        print(f"[server] WARNING: could not load classifier: {exc}")
    return None


CACHE = {"records": load_records(), "metrics": load_metrics(), "clf": load_classifier()}


class Handler(SimpleHTTPRequestHandler):
    extensions_map = {
        **SimpleHTTPRequestHandler.extensions_map,
        ".jsonl": "text/plain; charset=utf-8",
        ".md": "text/plain; charset=utf-8",
    }

    def do_GET(self) -> None:
        if self.path.startswith("/api/dataset"):
            recs, source = CACHE["records"]
            self._send_json({"records": recs, "source": source})
        elif self.path.startswith("/api/metrics"):
            self._send_json(CACHE["metrics"])
        else:
            super().do_GET()

    def do_POST(self) -> None:
        if self.path.startswith("/api/classify"):
            self._classify()
        else:
            self._send_json({"error": "Not found"}, 404)

    def _classify(self) -> None:
        clf = CACHE["clf"]
        if clf is None:
            self._send_json({"error": "Classifier not loaded. Run: python -m src.classification_eval"}, 503)
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length) or b"{}")
        except Exception:
            self._send_json({"error": "Invalid JSON body"}, 400)
            return

        question = str(payload.get("question", "")).strip()
        answer = str(payload.get("answer", "")).strip()
        context = str(payload.get("context", "")).strip()
        raw_sat = payload.get("satisfaction")
        try:
            satisfaction = float(raw_sat) if raw_sat is not None else None
        except (TypeError, ValueError):
            satisfaction = None

        if not question or not answer:
            self._send_json({"error": "Both 'question' and 'answer' are required."}, 400)
            return

        # Same text recipe the classifier was trained with (src/classification.py)
        text = f"Q: {question} C: {context} A: {answer}"
        probs = clf.predict_proba([text])[0]
        prob_map = {cls: float(p) for cls, p in zip(clf.classes_, probs)}
        label = max(prob_map, key=prob_map.get)
        confidence = prob_map[label]

        result = {
            "label": label,
            "label_display": LABEL_DISPLAY.get(label, label),
            "confidence": round(confidence, 4),
            "probabilities": {k: round(v, 4) for k, v in sorted(prob_map.items(), key=lambda kv: -kv[1])},
            "satisfaction": satisfaction,
            "note": "",
        }
        if satisfaction is not None:
            if satisfaction >= 70 and label != "no_failure":
                result["note"] = "You are satisfied but the model flags a failure — worth a manual review."
            elif satisfaction <= 40 and label == "no_failure":
                result["note"] = "You are unsatisfied but the model sees no failure — check completeness / required facts."
        self._send_json(result)

    def _send_json(self, obj: dict, status: int = 200) -> None:
        data = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, fmt: str, *args) -> None:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve RAG-FailureBench landing page with live data.")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--page", default="/app/web/index.html")
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()

    handler = partial(Handler, directory=str(ROOT))
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    recs, source = CACHE["records"]
    print(f"Records loaded    : {len(recs)} ({source or 'none — page will use seeded preview'})")
    print(f"Metrics loaded    : {list(CACHE['metrics'].keys()) or 'none — page will use embedded results'}")
    print(f"Classifier loaded : {'YES' if CACHE['clf'] is not None else 'NO — run python -m src.classification_eval'}")
    url = f"http://127.0.0.1:{args.port}{args.page}"
    print(f"Demo URL          : {url}")
    if not args.no_open:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
        server.shutdown()


if __name__ == "__main__":
    main()