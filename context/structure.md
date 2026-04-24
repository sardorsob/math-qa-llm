# Repository structure

Skeleton index — fill as the project grows.

**Competition reference:** [151B_SP26_Competition](https://github.com/brooksniu/151B_SP26_Competition) — starter template, dataset narrative, loading patterns. Task definition and schema are summarized in `PROJECT_BRIEF.md` and `DATASETS.md`.

**Environment:** `requirements.txt`, `environment.yml`, and step-by-step setup (venv, kernel, GPU, Hugging Face cache) — `ENVIRONMENT_SETUP.md`.

## Layout

- `configs/` — run configuration (`default.yaml`)
- `context/` — documentation and project memory (this file, rules, briefs)
- `data/raw/` — competition JSONL (gitignored; use `.gitkeep` for empty dir)
- `data/external/` — sample submission and other external references
- `notebooks/` — EDA and inference notebooks
- `src/` — reusable library code
- `scripts/` — CLI utilities (e.g. submission CSV)
- `artifacts/logs/runs/` — per-run JSONL and metadata (gitignored contents)
- `artifacts/submissions/` — CSVs for leaderboard upload

## Artifact index

| Run / artifact | Path | Notes |
|----------------|------|-------|
| — | — | — |

## Results log

| Date | Notebook / script | Config | Notes |
|------|-------------------|--------|-------|
| — | — | — | — |
