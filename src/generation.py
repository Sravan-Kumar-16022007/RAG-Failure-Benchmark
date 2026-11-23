"""LLM Generation utilities using Ollama."""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict

import requests

logger = logging.getLogger(__name__)

DEFAULT_OLLAMA_URL = "http://localhost:11434/api/generate"
# Updated to the fast, CPU-friendly Llama 3.2 3B model
DEFAULT_MODEL = "llama3.2:3b"


def generate_with_citations(
    question: str,
    contexts: list[Dict[str, Any]],
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_OLLAMA_URL,
) -> Dict[str, Any]:
    """Generate an answer with structured citations using Ollama."""
    context_text = "\n\n".join([f"[{ctx['context_id']}] {ctx['text']}" for ctx in contexts])
    
    # Llama 3.2 responds very well to this explicit JSON structure prompt
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a precise RAG assistant. Answer the user's question using ONLY the provided context.
You MUST cite your sources by appending the context ID in brackets, like [ctx_001], immediately after the relevant claim.
Respond ONLY with a valid JSON object. Do not include markdown formatting like ```json.
<|eot_id|><|start_header_id|>user<|end_header_id|>
Context:
{context_text}

Question: {question}

Respond with this exact JSON structure:
{{
  "answer": "Your concise answer with inline citations like [ctx_001].",
  "claims": [
    {{
      "claim": "A specific factual statement from your answer.",
      "citation_context_ids": ["ctx_001"]
    }}
  ]
}}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 512,  # Limit output length to prevent rambling
        }
    }

    try:
        response = requests.post(base_url, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        response_text = result.get("response", "")
        
        # Extract JSON block using regex (highly robust for Llama models)
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group(0))
            return parsed
        else:
            logger.warning("No JSON object found in response. Raw: %s", response_text[:100])
            return {"answer": response_text.strip(), "claims": []}
            
    except requests.exceptions.RequestException as e:
        logger.error("Ollama request failed: %s", e)
        return {"answer": "Error generating response.", "claims": []}
    except json.JSONDecodeError as e:
        logger.error("Failed to parse Ollama JSON response: %s. Raw: %s", e, response_text[:100])
        return {"answer": "Error parsing response.", "claims": []}