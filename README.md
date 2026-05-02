# Math QA LLM (CSE 151B)

LLM pipeline for math problems (free-form and MCQ), inference, and CSV submission.

## Quick start

1. **Environment (recommended):** from repo root, `conda env create -f environment.yml`, then `conda activate cse151b-math-qa`, then `python scripts/register_jupyter_kernel.py`. Select kernel **Python (cse151b-math-qa)**. For Linux/WSL + vLLM, use `environment-vllm.yml` instead. Alternatives: **`context/ENVIRONMENT_SETUP.md`** and `pip install -r requirements.txt` in a venv.
2. Add competition files under `data/raw/` (`public.jsonl`, `private.jsonl`) and `data/external/` (`sample_submission.csv`) — these are the **only** data files for the task.
3. Explore data: `notebooks/01_eda.ipynb` (skeleton).
4. Inference: `notebooks/02_inference.ipynb` (starter flow; paths will move to `configs/default.yaml` — see `context/STATUS.md`).
5. Scoring helpers: `judger.py`, `utils.py` (root; notebook uses `sys.path` to import).

Documentation and project memory live in **`context/`** (layout: `context/structure.md`).

## Layout

| Area | Role |
|------|------|
| `configs/` | Run configuration |
| `context/` | Docs, assumptions, decisions, status |
| `data/raw/` | Competition JSONL (not committed) |
| `data/external/` | Sample submission and similar |
| `notebooks/` | EDA + inference |
| `src/` | Reusable library (scaffold) |
| `scripts/` | CLI (e.g. submission CSV) |
| `artifacts/` | Run logs and submission CSVs |
