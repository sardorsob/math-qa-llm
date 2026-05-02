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
- [x] **Phase 2 notebook plan implemented and corrected for competition rules** — structured math/MCQ prompts, adaptive multi-pass generation, optional self-consistency / majority vote on uncertain questions, truncation/uncertainty diagnostics, checkpointing, and Section 10 CSV export. The model is locked back to the required **`Qwen/Qwen3-4B-Thinking-2507`** after a temporary invalid 8B model experiment was rejected.

## In progress

- [ ] **WSL2 + vLLM adaptive validation run** — current notebook is set for `DATA_MODE="public"`, `N_QUESTIONS=50`, adaptive Phase 1/2/3, checkpoint `adaptive_public_v1_checkpoint.jsonl`, output `adaptive_public_v1_results.jsonl`
- [ ] Full **`private.jsonl`** inference + **`SAVE_EVAL=False`** + submission CSV (`id`, `response`)
- [x] Notebook Section 10 CSV export cell added (`id,response`, `csv.QUOTE_ALL`)
- [ ] Wire **`configs/default.yaml`** + `src/utils/paths.py` (optional consolidation; notebook currently self-contained)

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
| **Private run + CSV** | Notebook can now write quoted CSV; still needs actual private run |
| **Config centralization** | Paths / `MAX_TOKENS` duplicated in notebook vs YAML |
| **EDA notebook** | `01_eda.ipynb` still skeleton until loaders land in `src/` |
| **Notebook stale outputs** | Old notebook outputs may still show 10-question results; trust source cells and rerun config → dataset → generation → scoring in order |

## Next steps

1. Run the **50-question public adaptive validation batch**. Confirm the dataset cell prints `random sample of 50`, adaptive generation prints phase counts, and scoring prints `Scoring 50 responses`.
2. Append the Phase 2 public result to `ITERATIONS.md` and `structure.md`.
3. If `thinking_budget` is unsupported or Phase 1 is still too slow, reduce `PHASE1_MAX_TOKENS` or skip Phase 2/3 for private.
4. For a first leaderboard upload, set **`DATA_MODE="private"`**, **`N_QUESTIONS=None`**, skip scoring/summary, then run save + CSV for a 943-row submission.

## Docs hygiene

- [x] Implementation decisions logged in `DECISIONS.md` (see table)
- [x] Observed public split stats in `DATASETS.md` (approximate; re-verify after data refresh)
