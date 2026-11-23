"""Citation extraction and baseline validation utilities."""
from __future__ import annotations

import re
from typing import Any, Dict, List


def extract_citations_from_text(text: str) -> List[str]:
    """Extract all [ctx_XXX] patterns from a text string."""
    return list(set(re.findall(r'\[ctx_\d+\]', text)))


def clean_citation_markers(text: str) -> str:
    """Remove [ctx_XXX] markers from text for cleaner display."""
    return re.sub(r'\s*\[ctx_\d+\]\s*', ' ', text).strip()


def validate_claim_support(claim: str, context_text: str, threshold: float = 0.3) -> tuple[bool, str]:
    """
    Baseline validation: Checks if the claim has sufficient keyword overlap with the context.
    Returns (is_supported, explanation).
    """
    # Simple tokenization and lowercasing
    claim_words = set(re.findall(r'\b\w+\b', claim.lower()))
    context_words = set(re.findall(r'\b\w+\b', context_text.lower()))
    
    # Remove common stop words (simplified)
    stop_words = {'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'to', 'of', 'in', 'for', 'with'}
    claim_words = claim_words - stop_words
    context_words = context_words - stop_words
    
    if not claim_words:
        return True, "Claim is too short to evaluate."
    
    # Jaccard similarity
    intersection = claim_words.intersection(context_words)
    similarity = len(intersection) / len(claim_words)
    
    if similarity >= threshold:
        return True, f"High keyword overlap ({similarity:.2f}) with context."
    else:
        return False, f"Low keyword overlap ({similarity:.2f}). Claim may be hallucinated or unsupported."