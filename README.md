# RAG-FailureBench

**A reproducible benchmark for evaluating Retrieval-Augmented Generation (RAG) failures.**

RAG-FailureBench provides a normalized dataset, Pydantic-validated schema, leakage-aware data splits, retrieval evaluation, and failure-classification baselines. It is designed to make RAG error analysis consistent, inspectable, and comparable.

[Kaggle Dataset](https://www.kaggle.com/datasets/yogeshm01/rag-error-analysis-dataset-for-llm-evaluation) · [Streamlit Dashboard](app/streamlit_app.py)

## Overview

The project transforms RAG evaluation examples into a structured JSONL benchmark with:

- Seven canonical failure categories
- Query, document, retrieved-context, and citation metadata
- Ground-truth answers and required facts
- Train, validation, and test splits separated by document ID
- Retrieval, classification, generation, and citation evaluation utilities
- A Streamlit interface for dataset exploration

## Failure Taxonomy

| Label | Description |
|---|---|
| `hallucination` | The answer introduces factual claims unsupported by the retrieved context |
| `retrieval_failure` | The retrieved context lacks sufficient relevant evidence |
| `contradiction` | The answer conflicts with the context or ground truth |
| `wrong_citation` | A citation does not support its associated claim |
| `reasoning_error` | The model draws an incorrect inference or conclusion from relevant evidence |
| `partial_answer` | The response is relevant but omits required facts |
| `no_failure` | The response is relevant, complete, grounded, and appropriately cited |

## Dataset

The source dataset is [RAG Error Analysis Dataset for LLM Evaluation](https://www.kaggle.com/datasets/yogeshm01/rag-error-analysis-dataset-for-llm-evaluation) by **Yogesh Mishra**, published on Kaggle.

- Source size: 1,200 examples
- License: CC BY 4.0
- Attribution: Yogesh Mishra / Kaggle dataset
- Normalized format: JSONL
- Validation: Pydantic models and citation-integrity checks

Because the source license requires attribution, retain this reference whenever redistributing or adapting the benchmark data.

## Installation

Python 3.10 or later is recommended.

### Windows PowerShell

```powershell
git clone https://github.com/Sravan-Kumar-16022007/RAG-Failure-Benchmark.git
cd RAG-Failure-Benchmark
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### macOS or Linux

```bash
git clone https://github.com/Sravan-Kumar-16022007/RAG-Failure-Benchmark.git
cd RAG-Failure-Benchmark
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Kaggle Authentication

The data ingestion script expects a Kaggle API token.

1. Create an API token on your [Kaggle account settings](https://www.kaggle.com/settings) page.
2. Create a `.env` file in the repository root:

```dotenv
KAGGLE_API_TOKEN=your_token_here
```

3. Download and normalize the dataset:

```bash
python -m src.ingest
python -m src.normalize
python -m src.validation --path data/processed/rag_failurebench_full.jsonl
```

The normalized records are written to `data/processed/rag_failurebench_full.jsonl`.

## Evaluation

Run the core tests:

```bash
pytest -q
```

Run the available evaluation modules after preparing the required data and artifacts:

```bash
python -m src.retrieval_eval
python -m src.classification_eval
python -m src.generation_eval --limit 50
```

Retrieval evaluation requires the embedding model and FAISS artifacts under `artifacts/`. FAISS artifact generation is available in `scripts/build_faiss_index_colab.py`.

Generation evaluation requires an Ollama service with the configured model, as specified in `configs/config.yaml`.

## Dashboard

```bash
python -m pip install streamlit
streamlit run app/streamlit_app.py
```

The dashboard uses the normalized dataset and generated evaluation reports when available.

## Project Structure

```text
RAG-Failure-Benchmark/
├── app/                 # Streamlit dashboard
├── configs/             # Dataset, retrieval, and generation configuration
├── data/
│   ├── raw/             # Downloaded source data
│   ├── processed/       # Normalized JSONL benchmark
│   └── interim/         # Generated and intermediate records
├── scripts/             # Artifact-building utilities
├── src/
│   ├── schemas.py       # Pydantic benchmark models
│   ├── ingest.py        # Kaggle data download
│   ├── normalize.py     # Source-to-benchmark normalization
│   ├── validation.py    # Schema, split, and leakage validation
│   ├── retrieval.py     # FAISS retrieval
│   ├── classification.py
│   └── generation.py
├── artifacts/           # Locally generated model and index artifacts
└── requirements.txt
```

## Reproducibility

- Data splits use `random_state=42`.
- Document IDs prevent records with the same document from appearing across splits.
- Canonical labels are derived from the Kaggle source's failure categories.
- Configuration is centralized in `configs/config.yaml`.
- Generated artifacts and raw data are excluded from Git by `.gitignore`.

## Contributing

Contributions are welcome. Please open an issue or pull request that includes:

- A clear description of the proposed change
- Relevant tests or validation output
- Any updates required by the dataset schema or configuration

## License

The source benchmark data is licensed under **CC BY 4.0** and requires attribution to **Yogesh Mishra** and the Kaggle dataset. Confirm the licensing terms for this repository and any derived distributions before reuse.