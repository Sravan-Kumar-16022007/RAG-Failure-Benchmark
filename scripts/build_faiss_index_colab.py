"""
Standalone script to build FAISS index and save embedding model.
Designed to be run in Google Colab.
"""
import json
import os
import zipfile
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# --- Configuration ---
DATASET_FILE = "rag_failurebench_full.jsonl"  # Must be uploaded to Colab root
OUTPUT_DIR = Path("./artifacts")
MODEL_NAME = "all-MiniLM-L6-v2"

def main():
    print("🚀 Starting FAISS Index Build...")
    
    # 2. Load Dataset
    if not Path(DATASET_FILE).exists():
        raise FileNotFoundError(f"Please upload {DATASET_FILE} to the Colab root directory.")
    
    print(f"📂 Loading dataset from {DATASET_FILE}...")
    records = []
    with open(DATASET_FILE, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    print(f"   Loaded {len(records)} records.")

    # 3. Extract Contexts and Document IDs
    print("🔍 Extracting contexts...")
    texts = []
    doc_ids = []
    for rec in records:
        # We use the first context from each record as the document representation
        ctx = rec["retrieved_contexts"][0]
        texts.append(ctx["text"])
        doc_ids.append(rec["document_id"])
    
    print(f"   Extracted {len(texts)} contexts.")

    # 4. Load Model and Encode
    print(f"🤖 Loading model: {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    
    print("🧠 Encoding contexts (this may take a minute)...")
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype=np.float32)
    print(f"   Embeddings shape: {embeddings.shape}")

    # 5. Build FAISS Index
    print("🏗️ Building FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner Product = Cosine Sim for normalized vectors
    index.add(embeddings)
    print(f"   Index built with {index.ntotal} vectors.")

    # 6. Save Artifacts
    print(f"💾 Saving artifacts to {OUTPUT_DIR}...")
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Save Model
    model_path = OUTPUT_DIR / "model"
    model.save(str(model_path))
    print(f"   Model saved to {model_path}")
    
    # Save Index
    index_path = OUTPUT_DIR / "index.faiss"
    faiss.write_index(index, str(index_path))
    print(f"   Index saved to {index_path}")
    
    # Save ID Map
    id_map_path = OUTPUT_DIR / "id_map.json"
    with open(id_map_path, "w", encoding="utf-8") as f:
        json.dump(doc_ids, f)
    print(f"   ID map saved to {id_map_path}")

    # 7. Zip for Download
    print("📦 Zipping artifacts for download...")
    zip_path = "artifacts.zip"
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(OUTPUT_DIR):
            for file in files:
                file_path = Path(root) / file
                zipf.write(file_path, file_path.relative_to(Path(".")))
    print(f"✅ Done! Download '{zip_path}' from the Colab file browser.")

if __name__ == "__main__":
    main()