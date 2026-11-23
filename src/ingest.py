"""Ingest raw data from Kaggle.

Supports the new Kaggle single-token API (KGAT_...).
Requires KAGGLE_API_TOKEN to be set in the .env file.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

from .config import PROJECT_ROOT, load_config

KAGGLE_DATASET_SLUG = "yogeshm01/rag-error-analysis-dataset-for-llm-evaluation"


def _ensure_kaggle_auth(token: str) -> None:
    """Write the token to ~/.kaggle/access_token so the Kaggle CLI and Python API both see it."""
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(parents=True, exist_ok=True)
    token_path = kaggle_dir / "access_token"
    token_path.write_text(token, encoding="utf-8")
    # Kaggle CLI requires 600 permissions on Unix; on Windows this is a no-op but harmless.
    try:
        os.chmod(token_path, 0o600)
    except OSError:
        pass
    print(f"[ingest] Wrote Kaggle token to {token_path}")


def download_dataset(raw_dir: Path) -> Path:
    """Download the Kaggle dataset into raw_dir. Returns path to the CSV."""
    load_dotenv()
    token = os.getenv("KAGGLE_API_TOKEN")
    if not token:
        raise OSError(
            "KAGGLE_API_TOKEN is not set in .env. "
            "Get a token from https://www.kaggle.com/settings -> API -> Create New Token"
        )

    raw_dir.mkdir(parents=True, exist_ok=True)
    csv_path = raw_dir / "rag_failure_cases.csv"

    if csv_path.exists() and csv_path.stat().st_size > 0:
        print(f"[ingest] CSV already exists at {csv_path}, skipping download.")
        return csv_path

    _ensure_kaggle_auth(token)

    # Use the Kaggle CLI via subprocess — it's the most reliable method and
    # automatically picks up ~/.kaggle/access_token.
    print(f"[ingest] Downloading {KAGGLE_DATASET_SLUG} to {raw_dir}...")
    try:
        result = subprocess.run(
            [
                sys.executable, "-m", "kaggle",
                "datasets", "download",
                "-d", KAGGLE_DATASET_SLUG,
                "-p", str(raw_dir),
                "--unzip",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        print(result.stdout)
    except subprocess.CalledProcessError as exc:
        print("[ingest] Kaggle CLI failed:", exc.stderr)
        raise RuntimeError("Kaggle download failed. Check your token and network.") from exc

    if not csv_path.exists():
        raise FileNotFoundError(f"Download completed but {csv_path} was not found in {raw_dir}.")
    print(f"[ingest] Download complete: {csv_path}")
    return csv_path


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = load_config()
    paths_cfg = cfg["paths"]
    raw_dir = PROJECT_ROOT / paths_cfg["data_raw"]
    download_dataset(raw_dir)
    