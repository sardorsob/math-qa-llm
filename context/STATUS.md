# Status

## Done

- [x] Repository folder scaffold (`structure.md`)
- [x] Context updated for competition task, schema, and submission format (`PROJECT_BRIEF.md`, `DATASETS.md`)
- [x] `requirements.txt`, `environment.yml`, and `ENVIRONMENT_SETUP.md` (venv, kernel, GPU/HF model cache)

## In progress (planning)

- [ ] Align `notebooks/02_inference.ipynb` with full pipeline (see **Notebook gaps** below) — not started in code yet
- [ ] Implement `configs/default.yaml` + `src/utils/paths.py` and wire notebooks/scripts to them
- [ ] Implement `src/` loaders, inference helpers, and `scripts/generate_submission.py`
- [ ] Download competition files into `data/raw/` and optional sample CSV into `data/external/`

## Notebook gaps (`02_inference.ipynb` vs competition)

These are **planning notes** only; the notebook is unchanged until implementation.

| Gap | Why it matters |
|-----|----------------|
| **Subset-only generation** | Starter often runs on a tiny slice (e.g. first 5 rows). Competition requires **all** private `id`s in the submission CSV. |
| **No private-set run** | Need a path that loads `private.jsonl`, skips local scoring (no labels), and still saves **per-id responses**. |
| **No submission CSV step** | Need a dedicated step or script: results JSONL (or in-memory records) → **`id,response`** CSV with correct escaping. |
| **Hardcoded paths** | Should read paths from config + `paths.py` so public/private and output dirs stay consistent. |
| **Output location** | Prefer run JSONL under `artifacts/logs/runs/` (and CSV under `artifacts/submissions/`) instead of ad hoc `results/`. |
| **Venv activation cell** | `source .venv/...` is a no-op inside Jupyter; real fix is selecting the project kernel / interpreter. |

Optional niceties for later: `N_QUESTIONS = None` meaning “all”; progress bar over full split; toggle `SAVE_EVAL` / `USE_PUBLIC` for dev vs submission runs.

## Next steps — your setup checklist

1. **Clone or open** [151B_SP26_Competition](https://github.com/brooksniu/151B_SP26_Competition) if you need the canonical starter notebook or file checksums; keep **this** repo as your working tree.
2. **Download** `public.jsonl`, `private.jsonl`, and `sample_submission.csv` (if provided) into `data/raw/` and `data/external/` per `structure.md`.
3. **Kernel:** Create/select a venv (e.g. `.venv`) and register a Jupyter kernel; do not rely on `!source` in the notebook for package visibility.
4. **Fill `configs/default.yaml`** with model id, GPU, paths to `data/raw/*.jsonl`, and sampling limits once you lock choices.
5. **Implement** `src/utils/paths.py` (repo root + dirs) and load YAML from the notebook / scripts.
6. **EDA:** Run `notebooks/01_eda.ipynb` after implementing loaders — confirm MCQ vs free-form counts and `[ANS]` counts.
7. **Inference notebook:** Extend per the table above — full public eval path, full private inference path, JSONL export, then CSV via `scripts/generate_submission.py` (or equivalent cells).
8. **Spot-check CSV:** Open in Excel or pandas — columns `id`, `response`; no broken rows; every private `id` present.
9. **Log the run** in `structure.md` (artifact index / results log) when you have a real baseline.

## Next (docs)

- [ ] Record first modeling decisions in `DECISIONS.md`
- [ ] Add measured dataset stats to `DATASETS.md`
