# Status

## Done

- [x] Repository scaffold (`structure.md`)
- [x] Competition task + schema docs (`PROJECT_BRIEF.md`, `DATASETS.md`)
- [x] Environment: `environment.yml`, `environment-vllm.yml`, `requirements.txt`, `scripts/register_jupyter_kernel.py`, `ENVIRONMENT_SETUP.md`
- [x] **`notebooks/02_inference.ipynb` — major implementation pass**
  - Repo-root resolution (`judger.py` parent), `sys.path`, paths under `data/raw/`, outputs under `artifacts/logs/runs/`
  - Optional **`TEST_RANDOM_SUBSET` + `RANDOM_SEED`** when `N_QUESTIONS` is an int (smoke tests across the file)
  - **`N_QUESTIONS = None`** = full file for eventual production runs
  - Transformers path: **`tokenizer.padding_side = "left"`**, **`enable_thinking=True`** in `apply_chat_template`, **`MAX_TOKENS = 8192`** (headroom for Qwen3-Thinking before `\boxed{}`), **`min_p=0.0`**, **`pad_token_id=tokenizer.eos_token_id`**
  - **Sequential generation** with `tqdm` (one question at a time): visible progress; interrupt leaves partial **`responses`** list
  - **Scoring:** guards if `responses` missing or length mismatch; tqdm with `id=` postfix; public-only (`answer` required)
  - vLLM import optional (`USE_VLLM`); default workflow is **Transformers + bitsandbytes** (Windows-friendly)
  - Install cell: `sys.executable`-based pip hints; kernel note (no reliance on `!source`)
- [x] Section 3 dataset markdown: public vs private fields, `[ANS]`, answer shapes
- [x] **Iteration 1 (first eval)** — metrics + config + per-row notes in **`ITERATIONS.md`**; artifact `artifacts/logs/runs/starter_results.jsonl` (10 rows); summary row in `structure.md`
- [x] **Phase 2 notebook plan implemented and corrected for competition rules** — structured math/MCQ prompts, adaptive retry generation, self-consistency / majority vote on uncertain questions, truncation/uncertainty diagnostics, checkpointing, and Section 10 CSV export. The model is locked back to the required **`Qwen/Qwen3-4B-Thinking-2507`** after a temporary invalid 8B model experiment was rejected.
- [x] **Current notebook audit completed (2026-05-06)** — `notebooks/02_inference.ipynb` is now a two-phase `adaptive_{DATA_MODE}_v2` implementation, not the earlier three-phase `v1` plan.

## In progress

- [ ] **Current notebook default is still public** — `DATA_MODE="public"`, `N_QUESTIONS=None`, `RUN_NAME="adaptive_public_v2"`. This will run all 1126 public rows unless changed before generation.
- [ ] Full **`private.jsonl`** inference + **`SAVE_EVAL=False`** + submission CSV (`id`, `response`)
- [x] Notebook Section 10 CSV export cell added (`id,response`, `csv.QUOTE_ALL`)
- [ ] Wire **`configs/default.yaml`** + `src/utils/paths.py` (optional consolidation; notebook currently self-contained)

## Current notebook implementation audit (2026-05-06)

This reflects the source cells on disk, not stale notebook output.

| Area | Current source state |
|------|----------------------|
| Required model | `MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"` |
| Dataset toggle | `DATA_MODE = "public"` by default |
| Run size | `N_QUESTIONS = None`, so the selected full split is used |
| Run naming | `RUN_NAME = f"adaptive_{DATA_MODE}_v2"` |
| Phase 1 | All missing rows in one vLLM batch; `thinking_budget=1024`, `max_tokens=2048`, `N=1`, temperature `0.6` |
| Phase 2 | All Phase 1 uncertain rows in one flattened vLLM batch; `thinking_budget=4096`, `max_tokens=6144`, `PHASE2_N_SAMPLES=3`, temperature `0.65`, repetition penalty `1.05` |
| Phase 3 | Removed from current code. Older docs/results mention `adaptive_public_v1` Phase 3, but the notebook source now stops after Phase 2. |
| vLLM settings | `gpu_memory_utilization=0.78`, `max_model_len=8192`, `max_num_seqs=4`, `max_num_batched_tokens=4096` |
| Checkpoint | `artifacts/logs/runs/adaptive_public_v2_checkpoint.jsonl` while `DATA_MODE="public"` |
| Output JSONL | `artifacts/logs/runs/adaptive_public_v2_results.jsonl` after Section 9 save |
| Submission CSV | Section 10 writes `artifacts/submissions/submission_YYYY-MM-DD.csv` from `OUTPUT_PATH` |

Critical warning: the last long run accidentally used the public split because `DATA_MODE` remained `"public"`. A public CSV has 1126 rows and is not a valid leaderboard upload. Private must show 943 loaded rows before generation.

## Environment status

| Environment | State | Used for |
|-------------|-------|----------|
| Windows — Conda (`cse151b-math-qa`) | ✅ working | Exploration, quick edits, Transformers fallback |
| WSL2 Ubuntu — pip venv (`cse151b-venv`) | ✅ installed | **Production inference via vLLM** |

WSL2 setup: Ubuntu installed via `wsl --install`; Python venv created; full pip stack installed including `vllm`, `transformers`, `torch`, `bitsandbytes`. GPU verified visible via `torch.cuda.get_device_name(0)`. See `ENVIRONMENT_SETUP.md` for full reproducible steps.

## Remaining gaps (competition-complete pipeline)

| Gap | Notes |
|-----|--------|
| **vLLM smoke test** | ✅ Done for random 10 — see `ITERATIONS.md`; watch truncation before `\boxed{}` on long chains |
| **Private run + CSV** | Notebook can now write quoted CSV; still needs actual private run with `DATA_MODE="private"` and 943 loaded rows |
| **Config centralization** | Paths / `MAX_TOKENS` duplicated in notebook vs YAML |
| **EDA notebook** | `01_eda.ipynb` still skeleton until loaders land in `src/` |
| **Notebook stale outputs** | Old notebook outputs may still show 10-question results; trust source cells and rerun config → dataset → generation → scoring in order |
| **Notebook markdown drift** | Some markdown/output text still describes older v1 settings (Phase 3, `N=4/8`, or 8B wording). Source code cells are the current authority until notebook markdown is cleaned. |
| **Checkpoint granularity** | Current v2 writes checkpoints after full Phase 1 and full Phase 2, not after every question; an interrupt during a large batched phase may lose in-phase work. |

## Next steps

1. Before any private run, edit Section 2 to **`DATA_MODE="private"`** and keep **`N_QUESTIONS=None`**.
2. Rerun Section 2 and Section 3 and confirm the dataset cell prints **943** loaded/running rows.
3. Confirm `RUN_NAME` changes to `adaptive_private_v2`, so the checkpoint/output do not reuse public artifacts.
4. Run model load and generation. Skip scoring/summary for private because there is no `answer` field.
5. Run Section 9 save and Section 10 CSV. Confirm the CSV has **943 data rows plus header** before upload.

## Docs hygiene

- [x] Implementation decisions logged in `DECISIONS.md` (see table)
- [x] Observed public split stats in `DATASETS.md` (approximate; re-verify after data refresh)
