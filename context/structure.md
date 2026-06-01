# Repository structure

**Competition reference:** [151B_SP26_Competition](https://github.com/brooksniu/151B_SP26_Competition) — starter template, dataset narrative, loading patterns. Task definition and schema are summarized in `PROJECT_BRIEF.md` and `DATASETS.md`.

**Environment:** `environment.yml` (full GPU stack; Windows-friendly), `environment-vllm.yml` (adds vLLM on Linux/WSL), `requirements.txt` (venv), `scripts/register_jupyter_kernel.py`, and `ENVIRONMENT_SETUP.md`.

**Inference notebook (implemented):** `notebooks/02_inference.ipynb` — see `STATUS.md` for the full change list. High level: required `Qwen/Qwen3-4B-Thinking-2507`, repo-relative paths, public/private `DATA_MODE`, adaptive Transformers-based generation on DSMLP, public scoring, private JSONL save, and quoted CSV export.

## Layout

- `context/` — documentation and project memory (this file, rules, briefs)
- `data/raw/` — competition JSONL (gitignored; use `.gitkeep` for empty dir)
- `data/external/` — sample submission and other external references
- `notebooks/` — EDA and the numbered pipeline notebooks (inference, QLoRA, GRPO, verifier, private submission)
- `artifacts/logs/runs/` — per-run JSONL and metadata (gitignored contents)
- `artifacts/submissions/` — CSVs for leaderboard upload

**Note:** `configs/`, `src/`, and most of `scripts/` were removed in the 2026-05 cleanup pass. These directories contained only scaffold stubs that were never wired into any notebook. Notebooks are self-contained; utility code lives inside notebooks until there is a clear need to extract it.

**Exception (2026-05-31):** `scripts/` was repopulated with three real, used scripts for the deadline submission:

- `scripts/jsonl_to_csv.py` — standalone converter from `private_qlora_v1_merged_checkpoint.jsonl` to the competition CSV. Mirrors Section 7 of `06_private_submission.ipynb` so the CSV can be regenerated without loading the model or running the notebook.
- `scripts/recover_truncated_answers.py` — v1 heuristic recovery. Appends `\boxed{X}` to responses where the model ran out of tokens before formalizing its answer. Tiered confidence regex extraction. Coverage jump 52% → 99.5%.
- `scripts/recover_truncated_answers_v2.py` — v2 recovery with MCQ option-count validation, multi-answer delimiter detection, repeated-value confirmation, and a `--compare-with v1.csv` mode for side-by-side diff.

These exist because re-running inference with bigger `MAX_TOKENS` (the proper fix) would have taken 1-2 more days of GPU time, which the deadline did not allow. See `DECISIONS.md` and `ITERATIONS.md` for the full reasoning.

### Notebooks overview

| Notebook | Purpose | Key settings |
|----------|---------|-------------|
| `01_eda.ipynb` | Exploratory data analysis | Skeleton — not yet implemented |
| `02_inference.ipynb` | Public evaluation + adaptive inference | Phase 1/2 thinking budgets, IS_COLAB, Transformers load on DSMLP |
| `03_qlora_finetune.ipynb` | QLoRA supervised fine-tuning | MAX_SEQ_LENGTH=8192, LR=5e-5, RUN_MERGE=True |
| `04_grpo_train.ipynb` | GRPO reinforcement learning | G=8, MAX_COMPLETION_LEN=4096, BETA=0.1, RUN_MERGE=True |
| `05_train_ebm_verifier.ipynb` | Train the verifier head used for candidate reranking | Uses public inference artifacts from notebook 02 |
| `06_private_submission.ipynb` | Private inference → submission CSV | Same inference settings as 02, DATA_MODE=private |

## Current methodology notes (2026-05-26)

- The repo’s **current** execution path is **DSMLP + Hugging Face Transformers**.
- vLLM is still documented because it explains earlier design choices, but it is not the primary path new work should assume.
- The notebook numbering now matches the conceptual order of the pipeline: verifier before private submission.

### Pros and cons of the current structure

**Pros**

- The directory order now reflects the actual dependency order.
- Docs and notebooks are less likely to contradict each other about which backend is active.
- New contributors can follow the numbered workflow without guessing.

**Cons**

- Historical vLLM notes still exist, so some context sections remain longer than the current workflow alone would require.
- The pipeline still depends on artifacts flowing from notebook 02 into later stages, so stale outputs remain a practical risk.

## Artifact index

| Run / artifact | Path | Notes |
|----------------|------|-------|
| Starter / dev JSONL | `artifacts/logs/runs/starter_results.jsonl` | Default `OUTPUT_PATH` in `02_inference.ipynb` when `SAVE_EVAL=True` |
| Phase 2 public batch JSONL | `artifacts/logs/runs/phase2_public_batch_results.jsonl` | Current validation target: 50 random public rows, structured prompts, self-consistency off |
| Adaptive public v1 snapshot CSV | `artifacts/submissions/adaptive_public_v1_snapshot_20260506_105419.csv` | Public split snapshot from interrupted run; **not valid for leaderboard** |
| Adaptive v2 run outputs | `artifacts/logs/runs/adaptive_{DATA_MODE}_v2_results.jsonl` / `_checkpoint.jsonl` | Current notebook naming. With default `DATA_MODE="public"` this is public; private must become `adaptive_private_v2_*`. |
| Public inference results | `artifacts/logs/runs/adaptive_public_v2_results.jsonl` / `_checkpoint.jsonl` | Output of running notebook 02 on the public set; consumed by notebook 03 (QLoRA rejection sampling) and notebook 05 (EBM training) |
| Private inference results | `artifacts/logs/runs/private_qlora_v1_merged_checkpoint.jsonl` / `_results.jsonl` | Output of running notebook 06 on the private set; 943 rows with full reasoning + boxed answers |
| Recovery v1 output | `artifacts/logs/runs/private_qlora_v1_merged_recovered_results.jsonl` | Same 943 rows but with `\boxed{X}` heuristically appended to truncated responses |
| Recovery v2 output | `artifacts/logs/runs/private_qlora_v1_merged_recovered_v2_results.jsonl` | Same but with MCQ option validation + multi-answer delimiter detection + repeated-value confirmation |
| Submission CSV (raw) | `artifacts/submissions/submission_qlora_v1_merged_YYYY-MM-DD.csv` | From `scripts/jsonl_to_csv.py` or Section 7 of notebook 06; ~52% boxed coverage if recovery not applied |
| Submission CSV (recovery v1) | `artifacts/submissions/submission_qlora_v1_merged_YYYY-MM-DD_recovered.csv` | After running `scripts/recover_truncated_answers.py`; 99.5% boxed coverage; Kaggle baseline |
| Submission CSV (recovery v2) | `artifacts/submissions/submission_qlora_v1_merged_YYYY-MM-DD_recovered_v2.csv` | After running `scripts/recover_truncated_answers_v2.py`; 99.8% boxed coverage; latest submission |

## Results log

| Date | Notebook / script | Config | Notes |
|------|-------------------|--------|-------|
| 2026-04-23 | `02_inference` §1–9 | vLLM, `N_QUESTIONS=10`, random seed 42, `MAX_TOKENS=8192` | First smoke: 40% overall (4/10); see `ITERATIONS.md` |
| 2026-05-01 | `02_inference` Phase 2 implementation | Required 4B model, structured prompts, current `N_QUESTIONS=50`, `USE_SELF_CONSISTENCY=False` | Implementation checkpoint; score pending. See `ITERATIONS.md` for what changed and why |
| 2026-05-03 | `02_inference` adaptive public v1 | 50 public rows, 3-phase adaptive retry | 42% overall; MCQ improved, free-form remained low. See `ITERATIONS.md` |
| 2026-05-06 | `02_inference` interrupted full public v1 | Accidentally left `DATA_MODE="public"`; 1126 public rows | Interrupted during Phase 3; snapshot CSV is public-only and should not be submitted |
| 2026-05-06 | `02_inference` current source audit | `adaptive_{DATA_MODE}_v2`, two phases only, `DATA_MODE="public"`, `N_QUESTIONS=None` | Private run still pending; must switch to `DATA_MODE="private"` and verify 943 rows |
| 2026-05-13 | Full pipeline optimization pass | All notebooks, judger.py; see DECISIONS.md and ITERATIONS.md | 15+ bug fixes and hyperparameter changes; requires A100 execution for training notebooks |
| 2026-05-26 | Notebook readability cleanup | `02` through `06`; no algorithm changes | Removed machine-looking divider comments, simplified padded status prints, and cleaned notebook code layout while preserving behavior |
| 2026-05-26 | Methodology clarification + notebook renumbering | Context docs + notebooks `05/06` | Clarified DSMLP + Transformers as the active methodology and renumbered verifier/private notebooks to match the real execution order |

