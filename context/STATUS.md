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

---

## Pipeline optimization pass — 2026-05

### What was changed

A full audit of bugs, hyperparameters, and training decisions identified 15+ improvements across all notebooks. Changes are now in the main project files (previously in a git worktree).

#### Bug fixes (no retraining required)

| Fix | File | Impact |
|-----|------|--------|
| Zero-division guard in numeric judging | `judger.py` lines 756, 769 | Correct answers with gold=0 were silently marked wrong |
| `max_model_len` 8192 → 16384 | `notebooks/02_inference.ipynb`, `notebooks/05_private_submission.ipynb` | 38% of outputs were truncated — zero chance of correct answer |
| `_batch_tok` 16384 → 32768 on Colab | same notebooks | Double throughput on A100 |
| `PHASE1_THINKING_BUDGET` 1024 → 4096 | same notebooks | Small 4B model needs more thinking tokens per Qwen3 paper |
| `PHASE1_MAX_TOKENS` 2048 → 6144 | same notebooks | Output cap was less than thinking budget — contradiction |
| `PHASE2_N_SAMPLES` 3 → 8 | same notebooks | Better majority vote on uncertain questions |
| Assert order fixed in notebook 05 | `notebooks/05_private_submission.ipynb` | Assert fired before IS_COLAB block, always failed on GCP |
| `TEST_RANDOM_SUBSET=False`, `N_QUESTIONS=None` | all notebooks | Production-ready defaults |

#### Training fixes (require retraining on A100)

| Fix | File | Impact |
|-----|------|--------|
| `MAX_SEQ_LENGTH` 1024 → 8192 | `notebooks/03_qlora_finetune.ipynb` | Every training example was truncated before `</think>` or `\boxed{}` |
| `LEARNING_RATE` 2e-4 → 5e-5 | same | Prevented catastrophic forgetting of Qwen3 reasoning pattern |
| `NUMINA_SUBSET` 20K → 5K | same | Format mismatch (no `<think>` tags in NuminaMath) dominated training |
| `EPOCHS` 3 → 2 | same | Less overfitting on small format-mismatched corpus |
| `RUN_MERGE=True` | notebooks 03 and 04 | Models now auto-saved for downstream use |
| `G` 2 → 8 in GRPO | `notebooks/04_grpo_train.ipynb` | With G=2 most steps had zero gradient; G=8 ensures mixed rewards |
| `MAX_COMPLETION_LEN` 1024 → 4096 | same | Rollouts were truncated → reward=0 → model penalized for thinking |
| `MAX_PROMPT_LENGTH` 512 → 1024 | same | Many problems were truncated mid-statement |
| `LEARNING_RATE` 1e-7 → 5e-7 | same | 1e-7 produced near-zero parameter movement in ~200 steps |
| `BETA` 0.04 → 0.1 | same | Low KL penalty allowed reward hacking on tiny 50-question training set |

#### Infrastructure changes

- Removed all scaffold Python files: `src/`, `scripts/create_*.py`, `scripts/generate_submission.py`, `scripts/register_jupyter_kernel.py`, `configs/default.yaml`, all patch scripts
- Added `IS_COLAB` auto-detection to all notebooks (checks `google.colab` in `sys.modules`)
- Added `DRIVE_BASE` path logic for Google Drive model persistence on Colab/GCP
- Added `colab_setup` cells to notebooks 02, 03, 04, 05 for package installation on Colab

### Deployment environment

| Environment | State | Used for |
|-------------|-------|----------|
| Windows — Conda (`cse151b-math-qa`) | ✅ working | Local exploration only, Transformers fallback |
| WSL2 Ubuntu — pip venv (`cse151b-venv`) | ✅ installed | Local inference fallback via vLLM |
| **Vertex AI Workbench (GCP A100 40GB)** | ✅ planned | **Primary platform for training + inference** |

GCP credits ($300) are accessed via `console.cloud.google.com` → Vertex AI → Workbench. A100 40GB instance costs ~$2.50/hr. Full training + inference session estimated at $21 for 8–10 hours.

### Expected outcomes after A100 training run

| Stage | Expected accuracy |
|-------|-------------------|
| Bug fixes only (no retraining) | ~47–52% |
| + QLoRA fixes | ≥42% baseline preserved |
| + GRPO overhaul (G=8, longer completions) | **60–75%** target |
| Best-case (POLARIS-equivalent G=8, ~700 steps) | up to 79% |

Research basis: DAPO (ByteDance/Tsinghua), Dr. GRPO, POLARIS-4B paper, Best-of-Majority, Qwen3 technical report, inference scaling law papers.

### In progress / remaining gaps

- [ ] **Run on A100** — all changes are in files, need execution on Vertex AI Workbench
- [ ] **Public eval after training** — run notebook 02 with merged GRPO model to verify improvement
- [ ] **Private submission** — run notebook 05 after public eval confirms improvement
- [ ] Notebook 01 EDA is still a skeleton

---

## DSMLP vLLM → Transformers migration — 2026-05-16

### Why this happened

vLLM 0.21.0 (the only Python 3.13 wheel on PyPI) was compiled against CUDA 13. DSMLP's `sp26-cuda128` container ships CUDA 12.8. The binary `vllm/_C.abi3.so` requires ELF-versioned symbol `libcudart.so.13` which does not exist in CUDA 12.8. `patchelf` is not available on DSMLP. Every workaround attempted (stub symlink, `LD_LIBRARY_PATH`, `tilelang` stub) was confirmed broken. The only correct solution is to use HuggingFace Transformers `model.generate()` instead.

### What changed in the notebooks

#### notebooks/02_inference.ipynb

| Cell | Change | Reason |
|------|--------|--------|
| `b9a459bf` (Config) | `VLLM_AVAILABLE=False`, `USE_VLLM=False`, vLLM try/except commented out with `# VLLM:` prefix | vLLM broken on DSMLP CUDA 12.8 |
| `b9a459bf` (Config) | Added `CHUNK_SIZE=6` | Controls sequences per `model.generate()` call; tuned for A30 24GB BF16 KV budget |
| `b9a459bf` (Config) | `PHASE1_THINKING_BUDGET` 4096→1024, `PHASE1_MAX_TOKENS` 6144→4096, `PHASE2_MAX_TOKENS` 6144→5120, `PHASE2_N_SAMPLES` 8→3 | Token budget reduced for Transformers path speed; N=3 sufficient for majority vote without vLLM batching efficiency |
| `ad824e4c` (Helpers) | `make_sampling_params()` now returns a plain dict instead of `SamplingParams` object | HF Transformers `model.generate(**kwargs)` accepts a dict, not a vLLM object |
| `ad824e4c` (Helpers) | Added `generate_batch()` function | Core replacement for vLLM's `llm.generate()` — handles chunked batching, left-padding, prompt slicing, EOS-based finish reason detection, and CUDA cache clearing |
| `4b1492d0` (Model load) | Replaced entire vLLM `LLM(...)` block with `AutoModelForCausalLM.from_pretrained(...)` | Transformers path is now primary; VRAM auto-detection: ≥20GB → BF16+SDPA, <20GB → NF4 fallback |
| `4b1492d0` (Model load) | `attn_implementation="flash_attention_2"` → `"sdpa"`, `torch_dtype=` → `dtype=` | `flash_attn` not installed on DSMLP; SDPA is PyTorch built-in. `torch_dtype` deprecated in newer Transformers. |
| `4b1492d0` (Model load) | `tokenizer.padding_side = "left"` added | Critical for correct batched decoder-only generation — without this, padding on the right shifts the generation context and produces garbage |
| `4915f406` (Generation) | Phase 1 loop refactored: `for chunk in chunks → generate_batch() → write_checkpoint()` | Per-chunk crash-safe checkpointing replaces whole-phase checkpoint; blue tqdm bar replaces `\r` print |
| `4915f406` (Generation) | Phase 2 loop: per-question `generate_batch()` + per-question `write_checkpoint()` | Each retry is saved immediately; pod death loses at most the current in-progress question |
| All markdown cells | Rewritten to reflect actual Transformers path, removed Phase 3 references, relabeled active vs disabled sections | Titles were describing the old vLLM flow |

#### notebooks/05_private_submission.ipynb

Same model load and generation changes as notebook 02. Additionally:
- `_fix_vllm_cuda()` function deleted (dead code — was the CUDA stub workaround that never worked)
- 7 new markdown section headers added (notebook previously had none between code cells)

#### notebooks/03_qlora_finetune.ipynb

| Change | Value | Reason |
|--------|-------|--------|
| `NUMINA_SUBSET` | 5,000 → **15,000** | A30 24GB confirmed; more training data fits in the compute budget |
| `MAX_SEQ_LENGTH` | kept at 4096 | Safe for A30 24GB with NF4 + grad_accum=8; not changed |

#### notebooks/04_grpo_train.ipynb

| Change | Value | Reason |
|--------|-------|--------|
| `G` | already 4 (set in prior commit) | A30 24GB: G=8 OOMs during rollout phase; G=4 is safe |
| `MAX_COMPLETION_LEN` | already 2048 | No further changes needed |

### Current execution environment (DSMLP)

| Item | Value |
|------|-------|
| GPU | NVIDIA A30 — 24 GB VRAM |
| Container | `sp26-cuda128` (CUDA 12.8, Python 3.13) |
| PyTorch | 2.11.0+cu128 (pre-installed in `/opt/conda`) |
| vLLM | **disabled** (ELF symbol incompatibility with CUDA 12.8) |
| Inference | HF Transformers `model.generate()`, BF16+SDPA, CHUNK_SIZE=6 |
| Estimated Phase 1 throughput | ~270 tok/sec → ~2.5 hrs for 1126 questions |
| Estimated full run (Phase 1+2) | ~4–5 hrs (fits 6-hr pod) |

### Remaining gaps

| Gap | Notes |
|-----|-------|
| **Run notebook 02 on DSMLP overnight** | `DATA_MODE="public"`, `N_QUESTIONS=None`, Phase 1+2 to build rejection-sampling targets for QLoRA |
| **Run notebook 03 (QLoRA)** | After public inference; needs `adaptive_public_v2_results.jsonl` for rejection sampling |
| **Run notebook 04 (GRPO)** | Needs 12-hr pod: `export K8S_TIMEOUT_SECONDS=43200` before launch |
| **Run notebook 05 (private submission)** | After GRPO merge; auto-selects best merged model |
| **Re-enable vLLM** | Only possible on CUDA 13+ system; would restore N=8 majority vote and full batching speed |
