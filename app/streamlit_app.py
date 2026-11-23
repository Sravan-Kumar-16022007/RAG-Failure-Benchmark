"""RAG-FailureBench Streamlit Dashboard.
Matches the landing page design with interactive dataset exploration and evaluation metrics.
"""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

# --- Page Config ---
st.set_page_config(
    page_title="RAG-FailureBench",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- Paths ---
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "rag_failurebench_full.jsonl"
METRICS_PATH = ROOT / "outputs" / "reports" / "retrieval_metrics.json"
CLASS_REPORT_PATH = ROOT / "outputs" / "charts" / "classification_report.json"
CM_PATH = ROOT / "outputs" / "charts" / "confusion_matrix.png"

# --- Data Loading ---
@st.cache_data
def load_data():
    records = []
    if DATA_PATH.exists():
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            for line in f:
                records.append(json.loads(line))
    return pd.DataFrame(records)

df = load_data()

# --- Custom CSS for professional look ---
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: 700; color: #1f77b4; margin-bottom: 0.5rem; }
    .sub-header { font-size: 1.2rem; color: #666; margin-bottom: 2rem; }
    .stats-bar { background-color: #f0f2f6; padding: 1rem; border-radius: 8px; text-align: center; font-weight: 500; margin: 1rem 0; }
    .taxonomy-card { background-color: #fff; border: 1px solid #e0e0e0; border-radius: 8px; padding: 1rem; margin-bottom: 1rem; }
    .taxonomy-title { font-weight: 600; font-size: 1.1rem; margin-bottom: 0.5rem; }
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown('<div class="main-header">🛡️ Build Robust RAG Systems:</div>', unsafe_allow_html=True)
st.markdown('<div class="main-header">Analyze & Benchmark AI Failures</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">A comprehensive dataset and evaluation pipeline for identifying and classifying Retrieval-Augmented Generation errors, curated for the Kaggle community.</div>', unsafe_allow_html=True)

col1, col2 = st.columns([1, 1])
with col1:
    st.link_button("📊 Explore Dataset on Kaggle", "https://www.kaggle.com/datasets/yogeshm01/rag-error-analysis-dataset-for-llm-evaluation", use_container_width=True)
with col2:
    st.link_button("📖 View Docs & Annotation Guide", "https://github.com", use_container_width=True)

# --- Stats Bar ---
st.markdown(f'<div class="stats-bar">Dataset Stats: 7 Failure Taxonomies | {df["document_id"].nunique()} Source Documents | {len(df)} Samples | JSONL & CSV Format | Pydantic Validated</div>', unsafe_allow_html=True)

# --- Tabs ---
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Failure Taxonomy", "Dataset Schema", "Project Structure", "Resources"])

# --- Tab 1: Overview & Failure Diff ---
with tab1:
    st.subheader("🔍 Failure Diff Viewer")
    st.markdown("Select an example to inspect the retrieval context, LLM output, and detected failure labels.")
    
    example_ids = df['example_id'].tolist()
    selected_id = st.selectbox("Select Example ID", example_ids, index=0)
    rec = df[df['example_id'] == selected_id].iloc[0]
    
    col_q, col_c = st.columns([1, 2])
    with col_q:
        st.markdown("**Query**")
        st.info(rec['question'])
    with col_c:
        st.markdown("**Retrieved Context**")
        st.text(rec['retrieved_contexts'][0]['text'])
        
    st.markdown("**LLM Output**")
    st.code(rec['generated_answer'], language="text")
    
    st.markdown("**Detected Labels**")
    label_cols = st.columns(3)
    with label_cols[0]:
        st.metric("Primary Label", rec['primary_failure_label'])
    with label_cols[1]:
        sec = ", ".join(rec['secondary_failure_labels']) if rec['secondary_failure_labels'] else "None"
        st.metric("Secondary Labels", sec)
    with label_cols[2]:
        st.metric("Severity", rec['severity'])
        
    st.markdown("---")
    st.subheader("📈 Evaluation Metrics")
    
    met_col1, met_col2 = st.columns(2)
    with met_col1:
        if METRICS_PATH.exists():
            with open(METRICS_PATH) as f:
                metrics = json.load(f)
            st.markdown("**Retrieval Metrics**")
            st.json(metrics)
        else:
            st.warning("Retrieval metrics not found. Run `src.retrieval_eval` first.")
            
    with met_col2:
        if CM_PATH.exists():
            st.markdown("**Classification Confusion Matrix**")
            st.image(str(CM_PATH), use_container_width=True)
        else:
            st.warning("Confusion matrix not found. Run `src.classification_eval` first.")

# --- Tab 2: Failure Taxonomy ---
with tab2:
    st.subheader("🏷️ 7 Failure Taxonomies")
    
    taxonomies = {
        "Hallucination": "The generated answer includes a factual claim that is not supported by the retrieved context.",
        "Retrieval Failure": "The retrieved context does not contain sufficient relevant evidence to answer the question correctly.",
        "Contradiction": "The generated answer directly conflicts with the retrieved context or verified ground truth.",
        "Wrong Citation": "A citation is present but does not support the associated answer claim.",
        "Reasoning Error": "The relevant evidence exists in retrieved contexts, but the model makes an incorrect calculation, inference, comparison, ordering, or multi-hop conclusion.",
        "Partial Answer": "The answer is relevant and partly correct but misses one or more required facts.",
        "No Failure": "The answer is correct, relevant, complete enough, grounded in context, and properly cited when citations are required."
    }
    
    cols = st.columns(4)
    for i, (label, desc) in enumerate(taxonomies.items()):
        with cols[i % 4]:
            count = len(df[df['primary_failure_label'] == label.lower().replace(' ', '_')])
            st.markdown(f'<div class="taxonomy-card"><div class="taxonomy-title">{label}</div><div>{desc}</div><br><b>Count: {count}</b></div>', unsafe_allow_html=True)

# --- Tab 3: Dataset Schema ---
with tab3:
    st.subheader("📋 Dataset Schema")
    st.markdown("Every example in the dataset follows this strict Pydantic-validated structure:")
    schema_example = {
        "example_id": "rag_000001",
        "domain": "technology",
        "question_type": "single_hop",
        "question": "What is a vector database?",
        "document_id": "doc_0001",
        "document_title": "Introduction to Vector Databases",
        "source_url": None,
        "source_license": "CC BY 4.0",
        "ground_truth_answer": "A vector database stores vector embeddings and supports similarity search.",
        "required_facts": ["stores vector embeddings", "supports similarity search"],
        "retrieved_contexts": [{"context_id": "ctx_001", "document_id": "doc_0001", "rank": 1, "retrieval_score": 0.92, "text": "...", "is_relevant": True}],
        "generated_answer": "A vector database stores embeddings and enables similarity search. [ctx_001]",
        "answer_citations": ["ctx_001"],
        "citation_spans": [{"claim": "...", "citation_context_ids": ["ctx_001"], "is_supported": True, "evidence_explanation": "..."}],
        "primary_failure_label": "no_failure",
        "secondary_failure_labels": [],
        "error_explanation": "The answer is correct, complete, grounded, and correctly cited.",
        "severity": "low",
        "annotation_method": "hybrid",
        "split": "train"
    }
    st.json(schema_example)

# --- Tab 4: Project Structure ---
with tab4:
    st.subheader("📁 Project Structure")
    st.code("""
rag-failure-bench/
├── app/
│   └── streamlit_app.py          # This dashboard
├── src/
│   ├── schemas.py                # Pydantic data models
│   ├── validation.py             # Dataset validation
│   ├── normalize.py              # Kaggle CSV → JSONL
│   ├── retrieval.py              # FAISS index & search
│   ├── embeddings.py             # Sentence-transformers
│   ├── generation.py             # Ollama LLM generation
│   ├── citation_utils.py         # Citation extraction & validation
│   ├── classification.py         # TF-IDF + Logistic Regression
│   └── config.py                 # Configuration loader
├── data/
│   ├── raw/                      # Original Kaggle CSV
│   ├── processed/                # Normalized JSONL dataset
│   └── interim/                  # Generated answers
├── outputs/
│   ├── charts/                   # Confusion matrix, reports
│   └── reports/                  # Retrieval metrics
├── artifacts/                    # Pre-built FAISS index & model (from Colab)
├── configs/
│   └── config.yaml
└── requirements.txt
""", language="text")

# --- Tab 5: Resources ---
with tab5:
    st.subheader("🔗 Resources")
    st.markdown("""
    - **[Kaggle Dataset](https://www.kaggle.com/datasets/yogeshm01/rag-error-analysis-dataset-for-llm-evaluation)**: Original 1,200-example benchmark.
    - **[GitHub Repository](#)**: Source code and documentation.
    - **[Dataset Card](#)**: Detailed documentation of the dataset schema, collection method, and limitations.
    - **[Annotation Guide](#)**: Guidelines for human annotators.
    """)
    
    st.markdown("---")
    st.markdown("**Collaborator:** Sravan Kumar (CSE-AIML)")
    st.markdown("**Teammate:** RAG-FailureBench Lead AI")
    st.markdown("*Built with Python, FAISS, Sentence-Transformers, Ollama, and Streamlit.*")