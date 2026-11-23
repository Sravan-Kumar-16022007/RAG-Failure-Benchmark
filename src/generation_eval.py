"""Run generation and citation validation on the dataset."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from .config import load_config, get_paths
from .generation import generate_with_citations
from .citation_utils import extract_citations_from_text, validate_claim_support

logger = logging.getLogger(__name__)


def run_generation_eval(dataset_path: Path, output_path: Path, limit: int | None = None) -> None:
    """Main generation and citation evaluation pipeline."""
    logger.info("Loading dataset from %s", dataset_path)
    records = []
    with dataset_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            records.append(json.loads(line))
    
    if limit:
        records = records[:limit]
        logger.info("Running in LIMITED mode: processing only %d records", limit)
    else:
        logger.info("Loaded %d records for full processing", len(records))

    evaluated_records = []
    
    for i, rec in enumerate(records):
        logger.info("Processing %d/%d: %s", i + 1, len(records), rec["example_id"])
        
        question = rec["question"]
        contexts = rec["retrieved_contexts"]
        
        # 1. Generate answer with citations
        gen_result = generate_with_citations(question, contexts)
        raw_answer = gen_result.get("answer", "")
        claims_data = gen_result.get("claims", [])
        
        # 2. Extract and clean citations
        inline_citations = extract_citations_from_text(raw_answer)
        clean_answer = raw_answer # Keep inline citations in the main answer as per schema
        
        # 3. Build citation spans
        citation_spans = []
        all_context_text = " ".join([ctx["text"] for ctx in contexts])
        
        if claims_data:
            for claim_obj in claims_data:
                claim_text = claim_obj.get("claim", "")
                cited_ids = claim_obj.get("citation_context_ids", [])
                
                # Find the specific context text for these IDs
                supporting_texts = [
                    ctx["text"] for ctx in contexts if ctx["context_id"] in cited_ids
                ]
                combined_support_text = " ".join(supporting_texts) if supporting_texts else all_context_text
                
                is_supported, explanation = validate_claim_support(claim_text, combined_support_text)
                
                citation_spans.append({
                    "claim": claim_text,
                    "citation_context_ids": cited_ids,
                    "is_supported": is_supported,
                    "evidence_explanation": explanation,
                })
        else:
            # Fallback if LLM didn't output structured claims
            citation_spans.append({
                "claim": clean_answer,
                "citation_context_ids": inline_citations,
                "is_supported": False,
                "evidence_explanation": "LLM failed to structure claims; baseline validation marked as unsupported.",
            })

        # 4. Update record
        rec["generated_answer"] = clean_answer
        rec["answer_citations"] = inline_citations
        rec["citation_spans"] = citation_spans
        rec["annotation_method"] = "hybrid"  # LLM generation + heuristic validation
        
        evaluated_records.append(rec)

    # 5. Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for rec in evaluated_records:
            fh.write(json.dumps(rec) + "\n")
            
    logger.info("Saved %d evaluated records to %s", len(evaluated_records), output_path)
    print(f"\n✅ Generation complete! Saved to {output_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Run RAG generation and citation evaluation.")
    parser.add_argument("--limit", type=int, default=None, help="Number of records to process (default: all)")
    args = parser.parse_args()
    
    cfg = load_config()
    paths = get_paths(cfg)
    
    # We read from the processed full dataset, but we will write to a new interim file
    dataset_path = paths.processed / "rag_failurebench_full.jsonl"
    output_path = paths.interim / f"rag_failurebench_generated_limit{args.limit}.jsonl"
    
    run_generation_eval(dataset_path, output_path, limit=args.limit)