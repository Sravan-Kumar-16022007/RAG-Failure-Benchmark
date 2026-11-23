# RAG-FailureBench: RAG Error Analysis Dataset for LLM Evaluation

A reproducible dataset + evaluation pipeline for identifying and classifying
Retrieval-Augmented Generation (RAG) failures across seven canonical labels:
hallucination, retrieval_failure, contradiction, wrong_citation,
reasoning_error, partial_answer, no_failure.

Data source (Milestone 2+): Kaggle `yogeshm01/rag-error-analysis-dataset-for-llm-evaluation`
(1,200 examples, CC BY 4.0 — attribution required).

## Setup (Windows PowerShell)

```powershell
mkdir rag-failure-bench; cd rag-failure-bench
code .
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt