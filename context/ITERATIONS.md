# Experiment iterations

Append-only log of notable runs. Link artifacts under `artifacts/logs/runs/` and note notebook settings used.

---

## Iteration 1 — first end-to-end smoke (public eval)

**Date:** 2026-04-23 (session; confirm local run date if needed)

### Summary metrics (notebook §8)

| Split | Correct | Total | Accuracy |
|-------|---------|-------|------------|
| MCQ | 2 | 4 | 50.00% |
| Free-form | 2 | 6 | 33.33% |
| **Overall** | **4** | **10** | **40.00%** |

### Configuration (from `notebooks/02_inference.ipynb`)

| Setting | Value |
|---------|--------|
| Model | `Qwen/Qwen3-4B-Thinking-2507` |
| Backend | vLLM (`USE_VLLM = True`), Transformers load/generate cells left commented |
| `N_QUESTIONS` | `10` |
| `TEST_RANDOM_SUBSET` | `True` |
| `RANDOM_SEED` | `42` |
| `MAX_TOKENS` (`max_tokens` / `max_new_tokens`) | `8192` |
| Generation | vLLM batched prompts + `enable_thinking=True` in `apply_chat_template` |
| Scoring | `SAVE_EVAL = True`, public `DATA_PATH`, `Judger` + MCQ letter extract |

### Artifact

| File | Role |
|------|------|
| `artifacts/logs/runs/starter_results.jsonl` | 10 lines; fields `id`, `is_mcq`, `gold`, `response`, `correct` |

### Per-row outcomes (from saved JSONL, line order)

| `id` | Type | Correct | Notes |
|------|------|---------|--------|
| 228 | MCQ | No | Gold `I`; long bisection / derivative reasoning; wrong letter |
| 51 | MCQ | Yes | Gold `A` |
| 563 | MCQ | No | Gold `C`; trace very long—check truncation before final `\boxed{}` |
| 501 | FF | No | Gold multi-part (`-6`, `6.5`, `A`); partial credit not applied |
| 457 | FF | No | Gold `85.9436692696235°`; model ~`85.94366925`—likely tolerance / formatting |
| 285 | FF | No | Gold four parts (`9`, `16`, `27`, `26`); multi-part |
| 209 | FF | No | CI numeric interval; judger mismatch |
| 1116 | MCQ | Yes | Gold `A` |
| 178 | FF | Yes | Gold four dollar amounts |
| 864 | FF | Yes | Gold `9` |

### Observations

- First **full pipeline** run: **generate (vLLM) → score → save JSONL** on a **random 10-question** slice of public data.
- **MCQ id 563:** response appears **cut off** before a final `\boxed{…}`; aligns with **KV / length or output cap** under vLLM settings—worth checking **finish reason** or **max_tokens vs prompt length** on the next pass.
- **FF** mistakes mix **format**, **multi-answer**, and **numeric tolerance**—baseline for later prompt or decoding tweaks.

### Next ideas (not done)

- Re-run same **10 ids** after any config change to measure delta, or move to `N_QUESTIONS = None` for full public accuracy.
- Log **vLLM** `max_model_len` / `gpu_memory_utilization` in this file when you lock them.
- Add **private** run + CSV when `scripts/generate_submission.py` is ready.

---

## Phase 2 implementation state — prompt upgrade + submission path

**Date:** 2026-05-01

This is an implementation checkpoint, not a completed scored run yet.

### What changed in `notebooks/02_inference.ipynb`

| Area | Current state |
|------|---------------|
| Required model | `MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"` |
| Prompting | Structured **UNDERSTAND → PLAN → SOLVE → VERIFY → ANSWER** prompts for free-form and MCQ |
| Multi-answer handling | `build_prompt()` counts `[ANS]`; if more than one placeholder, it appends an explicit order-preserving comma-separated `\boxed{}` hint |
| MCQ formatting | Prompt asks for reasoning first, then final line only `\boxed{X}` |
| Validation batch | `DATA_PATH = public.jsonl`, `N_QUESTIONS = 50`, `TEST_RANDOM_SUBSET = True`, `RANDOM_SEED = 42` |
| Current output path | `artifacts/logs/runs/phase2_public_batch_results.jsonl` |
| Self-consistency | Code exists (`extract_boxed`, `majority_vote`, `N_SAMPLES`) but current validation setting is `USE_SELF_CONSISTENCY = False` |
| Truncation diagnostic | Generation cell counts vLLM finish reasons (`stop` vs `length`) |
| Submission export | Section 10 writes quoted `id,response` CSV under `artifacts/submissions/` |

### What was tried / corrected

- A larger-model upgrade was considered (`Qwen3-8B` / `Qwen3-8B-Thinking`) based on expected math gains, but it is **not allowed** for this competition because final responses must come from **`Qwen/Qwen3-4B-Thinking-2507`**.
- `Qwen/Qwen3-8B-Thinking` also failed at load time with Hugging Face `401 Unauthorized` / invalid model identifier, so the notebook was reverted to the required 4B thinking model.
- Full public evaluation plus `N_SAMPLES=8` self-consistency was deemed too slow after observing roughly **8 minutes per question** in the active user run. That would make full public/private runs take days.
- Current practical plan is therefore: validate prompt changes on **50 public questions**, then run the **943-row private set** without self-consistency for the first leaderboard submission.

### Run-order warning

Notebook outputs can be stale. After changing `N_QUESTIONS`, `DATA_PATH`, or `USE_SELF_CONSISTENCY`, rerun cells in this order:

1. Config
2. Dataset load
3. Prompt construction
4. Self-consistency config
5. vLLM model load if model/load settings changed
6. Generate responses
7. Score + summary only for public runs
8. Save JSONL
9. Generate CSV

Before scoring, confirm:

```python
print("N_QUESTIONS:", N_QUESTIONS)
print("len(data_run):", len(data_run))
print("len(responses):", len(responses))
```

For the current public validation batch these should be `50`, `50`, and `50`.

### Next result to log

Append a new scored iteration after the 50-question public validation run completes:

| Field | Expected value to fill |
|-------|------------------------|
| Artifact | `artifacts/logs/runs/phase2_public_batch_results.jsonl` |
| Config | required 4B thinking model, structured prompts, `N_QUESTIONS=50`, `USE_SELF_CONSISTENCY=False` |
| Metrics | MCQ / free-form / overall from notebook Section 8 |
| Diagnostic | finish reason counts from Section 6 |

---

## Adaptive inference implementation checkpoint

**Date:** 2026-05-01

This is an implementation checkpoint, not a completed scored run yet.

### What changed in `notebooks/02_inference.ipynb`

| Area | Current state |
|------|---------------|
| Required model | Still locked to `Qwen/Qwen3-4B-Thinking-2507` |
| Data toggle | `DATA_MODE = "public"` for scoring or `"private"` for leaderboard CSV |
| Validation size | `N_QUESTIONS = 50` by default; set `None` for full public/private runs |
| Phase 1 | All questions, `thinking_budget=1024`, `max_tokens=2048`, `N=1` |
| Phase 2 | Uncertain questions only, retry prefix, `thinking_budget=4096`, `max_tokens=6144`, `N=4` majority vote |
| Phase 3 | Still-uncertain questions only, metacognitive prefix, `thinking_budget=None`, `max_tokens=8192`, `N=8`, temperature `0.7` |
| Uncertainty logic | Retry if finish reason is `length`, no `\boxed{}` is found, answer section is too short, or multi-sample consensus is weak |
| Checkpointing | Writes/resumes `artifacts/logs/runs/adaptive_{DATA_MODE}_v1_checkpoint.jsonl` |
| Final output | Writes `adaptive_{DATA_MODE}_v1_results.jsonl` with phase metadata; Section 10 still writes quoted `id,response` CSV |

### Important run notes

- The notebook attempts to pass Qwen3's native `thinking_budget` into `tokenizer.apply_chat_template`.
- If the installed tokenizer does not support that kwarg, the helper falls back to a prompt-level budget hint and relies on the phase `max_tokens` cap.
- For final leaderboard work, use `DATA_MODE="private"` and `N_QUESTIONS=None`; skip scoring and summary because `private.jsonl` has no answers.

---

## Iteration 2 — Phase 2 structured prompts, 50-question public eval (scored)

**Date:** 2026-05-01

### Summary metrics (notebook §8)

| Split | Correct | Total | Accuracy |
|-------|---------|-------|----------|
| MCQ | 8 | 20 | 40.00% |
| Free-form | 11 | 30 | 36.67% |
| **Overall** | **19** | **50** | **38.00%** |

### Configuration (from `notebooks/02_inference.ipynb`)

| Setting | Value |
|---------|-------|
| Model | `Qwen/Qwen3-4B-Thinking-2507` |
| Backend | vLLM (`USE_VLLM = True`) |
| `N_QUESTIONS` | `50` |
| `TEST_RANDOM_SUBSET` | `True` |
| `RANDOM_SEED` | `42` |
| `MAX_TOKENS` | `8192` |
| Prompting | Structured **UNDERSTAND → PLAN → SOLVE → VERIFY → ANSWER** (free-form and MCQ) |
| Multi-answer handling | `build_prompt()` counts `[ANS]`; appends comma-separated `\boxed{}` hint when > 1 placeholder |
| MCQ formatting | Reasoning first, then final line `\boxed{X}` only |
| Self-consistency | `USE_SELF_CONSISTENCY = False` |
| Scoring | `SAVE_EVAL = True`, public `DATA_PATH`, `Judger` + MCQ letter extract |

### Artifact

| File | Role |
|------|------|
| `artifacts/logs/runs/phase2_public_batch_results.jsonl` | 50 lines; fields `id`, `is_mcq`, `gold`, `response`, `correct` |

### Observations

- First **scored run** with the structured prompt framework on a **50-question** public subset.
- **MCQ accuracy dropped** from 50% (Iteration 1, N=10) to 40% (N=50), likely reflecting the larger and more representative sample rather than prompt regression.
- **Free-form accuracy improved slightly** from 33.33% to 36.67%, suggesting the structured prompt's multi-answer formatting and simplification rules provide a marginal benefit.
- **Overall 38%** serves as the prompt-only baseline for comparison with the adaptive multi-phase system.
- No self-consistency or adaptive retries — single-pass generation only.

### Comparison with Iteration 1

| Metric | Iteration 1 (N=10) | Iteration 2 (N=50) | Delta |
|--------|---------------------|---------------------|-------|
| MCQ | 50.00% (2/4) | 40.00% (8/20) | −10.00 pp |
| Free-form | 33.33% (2/6) | 36.67% (11/30) | +3.34 pp |
| Overall | 40.00% (4/10) | 38.00% (19/50) | −2.00 pp |

Note: Iteration 1 used a 10-question random subset; Iteration 2 uses 50. The larger sample gives a more reliable accuracy estimate.

---

## Iteration 3 — adaptive multi-phase inference, 50-question public eval (scored)

**Date:** 2026-05-03

### Summary metrics (notebook §8)

| Split | Correct | Total | Accuracy |
|-------|---------|-------|----------|
| MCQ | 11 | 20 | 55.00% |
| Free-form | 10 | 30 | 33.33% |
| **Overall** | **21** | **50** | **42.00%** |

### Configuration (from `notebooks/02_inference.ipynb`)

| Setting | Value |
|---------|-------|
| Model | `Qwen/Qwen3-4B-Thinking-2507` |
| Backend | vLLM (`USE_VLLM = True`) |
| `N_QUESTIONS` | `50` |
| `TEST_RANDOM_SUBSET` | `True` |
| `RANDOM_SEED` | `42` |
| `DATA_MODE` | `"public"` |
| Phase 1 | All questions; `thinking_budget=1024`, `max_tokens=2048`, `N=1` |
| Phase 2 | Uncertain only; `thinking_budget=4096`, `max_tokens=6144`, `N=4`, majority vote |
| Phase 3 | Still-uncertain only; `thinking_budget=None`, `max_tokens=8192`, `N=8`, temperature `0.7` |
| Uncertainty triggers | Finish reason `length`, no `\boxed{}`, answer section too short, weak consensus |
| Checkpointing | `artifacts/logs/runs/adaptive_public_v1_checkpoint.jsonl` |

### Phase usage

| Phase | Count | Percentage |
|-------|-------|------------|
| Phase 1 (resolved) | 7 | 14% |
| Phase 2 (escalated) | 17 | 34% |
| Phase 3 (max budget) | 26 | 52% |
| **Still uncertain after all phases** | **23** | **46%** |

### Finish-reason diagnostics

| Finish Reason | Count | Percentage |
|---------------|-------|------------|
| `stop` | 31 | 62% |
| `length` (truncated) | 19 | 38% |

### Artifact

| File | Role |
|------|------|
| `artifacts/logs/runs/adaptive_public_v1_results.jsonl` | 50 lines; includes `phase_used`, `finish_reason`, `consensus_count`, `n_samples` |
| `artifacts/logs/runs/adaptive_public_v1_checkpoint.jsonl` | Checkpoint for resume |

### Observations

- **Overall accuracy improved to 42%**, a **+4 pp gain** over the Phase 2 prompt-only baseline (38%) and **+2 pp** over the initial 10-question smoke test (40%).
- **MCQ accuracy jumped to 55%** (+15 pp over Iteration 2), the largest single improvement across all iterations. The adaptive retry mechanism appears particularly effective for MCQ, where the model's initial reasoning is often on the right track but needs additional thinking budget to arrive at the correct letter.
- **Free-form accuracy remained flat at 33.33%**, slightly below Iteration 2's 36.67%. Free-form problems present persistent challenges related to multi-answer formatting, numeric precision, and answer diversity that adaptive retries alone do not resolve.
- **52% of questions escalated to Phase 3**, indicating that the majority of the 50-question sample is genuinely difficult for the 4B model under constrained thinking budgets.
- **38% of final responses were truncated** (finish reason `length`), even at Phase 3's maximum `max_tokens=8192`. Truncation remains a significant source of answer loss and should be a priority for the next iteration.
- **46% of questions remained uncertain** after all three phases, suggesting a ceiling on what inference-time methods can achieve without model improvement (e.g., fine-tuning).

### Comparison across all iterations

| Iteration | Config | MCQ | Free-form | Overall |
|-----------|--------|-----|-----------|---------|
| 1 — starter baseline | vLLM, generic prompt, N=10 | 50.00% (2/4) | 33.33% (2/6) | 40.00% (4/10) |
| 2 — structured prompts | vLLM, UNDERSTAND/PLAN/SOLVE/VERIFY/ANSWER, N=50 | 40.00% (8/20) | 36.67% (11/30) | 38.00% (19/50) |
| 3 — adaptive multi-phase | vLLM, 3-phase adaptive + majority vote, N=50 | **55.00% (11/20)** | 33.33% (10/30) | **42.00% (21/50)** |

### Next ideas (not done)

- Increase `max_tokens` beyond 8192 (e.g., 16384) for Phase 3 to reduce the 38% truncation rate.
- Add prompt-level conciseness instructions to reduce reasoning verbosity without sacrificing answer quality.
- Investigate free-form failures to distinguish between reasoning errors and formatting/extraction failures.
- Run full private inference (`DATA_MODE="private"`, `N_QUESTIONS=None`) for the first leaderboard submission.
- Consider QLoRA fine-tuning on external math corpora if prompt-only improvements plateau.

---

## Full public run mistake — interrupted adaptive v1

**Date:** 2026-05-06

This was not a leaderboard run. It is logged here because it explains lost runtime and the current code simplification.

### What happened

- The notebook was intended to move toward a private submission, but `DATA_MODE` remained `"public"`.
- The active checkpoint was `artifacts/logs/runs/adaptive_public_v1_checkpoint.jsonl`.
- The row count was **1126**, which matches the public split, not the private split.
- A snapshot CSV was created at `artifacts/submissions/adaptive_public_v1_snapshot_20260506_105419.csv`, but it has public ids/responses and must **not** be submitted to the leaderboard.

### Last observed checkpoint counts before stopping

| Count | Value |
|-------|-------|
| Rows with at least one response | 1126 |
| Phase 1 only | 256 |
| Phase 2 done | 806 |
| Phase 3 done | 64 |
| Still needing Phase 3 | 412 |
| Phase 3 resolved | 12 |
| Phase 3 still uncertain | 52 |

### Outcome

- The run was interrupted because Phase 3 would take too long.
- GPU/vLLM stopped after interrupt (`nvidia-smi` dropped from near-full VRAM/utilization to idle-level memory/utilization).
- The key process lesson is that **private submission requires verifying `DATA_MODE="private"` and 943 rows before generation starts**.

---

## Current implementation audit — adaptive v2 source state

**Date:** 2026-05-06

This is a source-code audit of `notebooks/02_inference.ipynb`, not a scored run.

### Current Section 2 configuration

| Setting | Current value |
|---------|---------------|
| `MODEL_ID` | `Qwen/Qwen3-4B-Thinking-2507` |
| `DATA_MODE` | `"public"` |
| `DATA_PATH` | `PUBLIC_PATH if DATA_MODE == "public" else PRIVATE_PATH` |
| `N_QUESTIONS` | `None` |
| `RUN_NAME` | `adaptive_{DATA_MODE}_v2` |
| `OUTPUT_PATH` | `artifacts/logs/runs/{RUN_NAME}_results.jsonl` |
| `CHECKPOINT_PATH` | `artifacts/logs/runs/{RUN_NAME}_checkpoint.jsonl` |

Because the current default is `DATA_MODE="public"` and `N_QUESTIONS=None`, running Section 6 without editing config will process all **1126 public rows** again.

### Current generation design

| Area | Current code |
|------|--------------|
| Phase 1 | One flattened/batched vLLM call for all rows missing from checkpoint |
| Phase 1 params | `thinking_budget=1024`, `max_tokens=2048`, temperature `0.6`, `N=1` |
| Phase 2 input | Rows still marked `uncertain` after Phase 1 and not already phase 2 |
| Phase 2 params | `thinking_budget=4096`, `max_tokens=6144`, `PHASE2_N_SAMPLES=3`, temperature `0.65`, repetition penalty `1.05` |
| Phase 2 batching | Builds `len(uncertain) * PHASE2_N_SAMPLES` prompts, runs one flattened `llm.generate`, then groups samples per question for majority vote |
| Phase 3 | Removed from current source. Older `adaptive_public_v1` artifacts/docs mention Phase 3, but current `v2` stops after Phase 2. |
| Checkpointing | Writes once after Phase 1 and once after Phase 2, not after every question |
| Final `responses` | Built from `response_records` in `data_run` order |

### Current uncertainty / voting logic

- A response is uncertain if finish reason contains `length`, no `\boxed{}` is extractable, or answer-only text after removing `<think>...</think>` is shorter than 30 characters.
- Phase 2 chooses the most common extracted boxed answer.
- If there is a tie, the longest trace is used.
- If the majority count is below `ceil(N/2)`, the chosen result remains marked uncertain.

### Current vLLM load settings

| Setting | Current value |
|---------|---------------|
| Quantization | `bitsandbytes` with `load_format="bitsandbytes"` |
| `gpu_memory_utilization` | `0.78` |
| `max_model_len` | `8192` |
| `max_num_seqs` | `4` |
| `max_num_batched_tokens` | `4096` |

### Required private run procedure

Before the next generation run, change and verify:

```python
DATA_MODE = "private"
N_QUESTIONS = None
```

Then rerun config and dataset cells. The dataset cell must print **943** rows. Only after that should Section 5 model load and Section 6 generation run.

---

## Pipeline optimization session — 2026-05

**Date:** 2026-05-13 / 2026-05-14

This is an implementation checkpoint, not a scored run. Changes are ready; execution requires A100 on Vertex AI Workbench.

### Research basis

All changes are grounded in published results:
- **DAPO** (ByteDance/Tsinghua): Clip-Higher, Dynamic Sampling, Token-Level Policy Gradient
- **Dr. GRPO**: KL penalty calibration for small training sets
- **POLARIS-4B**: Qwen3-4B reaches 79% AIME with G=8+ GRPO in ~700 RL steps
- **Best-of-Majority**: N=8 sampling with majority vote vs N=3
- **Qwen3 technical report**: 4B models gain +15–20% from 0→4K thinking budget

### Root causes identified (why 42% was the ceiling)

| Category | Issue | Effect |
|----------|-------|--------|
| Bug | `judger.py` divides by gold when gold=0 | Correct answers silently marked wrong |
| Bug | `max_model_len=8192` too small | 38% of outputs truncated, no `\boxed{}` possible |
| Bug | `PHASE1_THINKING_BUDGET=1024` | 4B model insufficient thinking; wrong answers sent to Phase 2 |
| Bug | `PHASE1_MAX_TOKENS=2048 < thinking_budget` | Output cap less than thinking budget — contradiction |
| Training | `MAX_SEQ_LENGTH=1024` in QLoRA | Every training example truncated before `</think>` or `\boxed{}` |
| Training | `LEARNING_RATE=2e-4` in QLoRA | Catastrophic forgetting of Qwen3 reasoning pattern |
| Training | `G=2` in GRPO | Most batches had zero gradient; no learning signal |
| Training | `MAX_COMPLETION_LEN=1024` in GRPO | Rollouts truncated → reward=0 → model penalized for thinking |
| Training | `LEARNING_RATE=1e-7` in GRPO | Near-zero parameter movement in ~200 total steps |
| Training | `BETA=0.04` in GRPO | Reward hacking on 50-question training set |

### All changes made

**Inference notebooks (02, 06):**
- `max_model_len`: 8192 → **16384**
- `_batch_tok` (Colab): 16384 → **32768**
- `PHASE1_THINKING_BUDGET`: 1024 → **4096**
- `PHASE1_MAX_TOKENS`: 2048 → **6144**
- `PHASE2_N_SAMPLES`: 3 → **8**
- `TEST_RANDOM_SUBSET`: True → **False** (production default)
- `N_QUESTIONS`: now **None** by default
- `IS_COLAB` auto-detection + `colab_setup` cell added
- Assert order fixed in the private submission notebook (now `06_private_submission.ipynb`; at the time this note referred to notebook 05)

**QLoRA notebook (03):**
- `MAX_SEQ_LENGTH`: 1024 → **8192**
- `LEARNING_RATE`: 2e-4 → **5e-5**
- `NUMINA_SUBSET`: 20,000 → **5,000**
- `EPOCHS`: 3 → **2**
- `RUN_MERGE`: False → **True**
- `IS_COLAB` auto-detection + Drive path logic added

**GRPO notebook (04):**
- `G`: 2 → **8**
- `MAX_COMPLETION_LEN`: 1024 → **4096**
- `MAX_PROMPT_LENGTH`: 512 → **1024**
- `LEARNING_RATE`: 1e-7 → **5e-7**
- `BETA`: 0.04 → **0.1**
- `RUN_MERGE`: False → **True**
- `IS_COLAB` auto-detection + Drive path logic added

**`judger.py`:**
- Line 756: division guard for gold=0
- Line 769: division guard for gold=0

**Repository cleanup:**
- Deleted: `src/`, `scripts/create_*.py`, `scripts/generate_submission.py`, `scripts/register_jupyter_kernel.py`, `configs/default.yaml`, all patch scripts (`patch_nb02.py`, `patch_notebook.py`, `patch_tokens.py`)

### Expected accuracy after A100 run

| Checkpoint | Expected accuracy |
|------------|-------------------|
| Baseline (current) | 42% (MCQ 55%, free-form 33%) |
| Inference fixes only | ~47–52% |
| + QLoRA fine-tuning | ≥42% (should preserve or improve) |
| + GRPO (G=8, 700 steps) | **60–75%** target |
| Best-case (POLARIS-equivalent) | up to 79% |

### Next scored run to log

After executing on A100, append results here:

| Metric | Value |
|--------|-------|
| Date | TBD |
| Platform | Vertex AI Workbench A100 40GB |
| Config | Merged GRPO model, Phase 1 budget=4096, max_model_len=16384, N_SAMPLES=8 |
| MCQ accuracy | TBD |
| Free-form accuracy | TBD |
| Overall accuracy | TBD |
| Truncation rate | TBD (target: <5%) |

---

## DSMLP vLLM → Transformers migration — 2026-05-16

**Type:** Implementation change checkpoint (not a scored run — execution pending on DSMLP overnight)

### Root cause: why vLLM had to be dropped

| Layer | Finding |
|-------|---------|
| vLLM version | 0.21.0 — only Python 3.13 wheel available on PyPI |
| Compiled against | CUDA 13 |
| DSMLP container | `sp26-cuda128` — CUDA 12.8 |
| Failure mode | `vllm/_C.abi3.so` requires ELF-versioned symbol `LIBCUDART_13.0`, which does not exist in `libcudart.so.12` |
| Error message | `version 'libcudart.so.13' not found (required by vllm/_C.abi3.so)` |
| Workarounds tried | Symlink `libcudart.so.13 → libcudart.so.12` (wrong — version tag is in ELF symbol table, not filename); tilelang `libcudart_stub.so` (satisfies dlopen but not ELF symbol version check); `LD_LIBRARY_PATH` patch (same reason); manual `patchelf --replace-needed` (patchelf not available) |
| Confirmed dead end | All workarounds fail at ELF symbol version verification, not at path resolution |

### What was implemented

**Core inference replacement:** vLLM's `llm.generate(prompts, params)` → HF Transformers `model.generate(**inputs, **gen_kwargs)` via a new `generate_batch()` helper.

Key implementation details:

```
generate_batch(prompts, gen_kwargs, chunk_size=CHUNK_SIZE):
  for each chunk of CHUNK_SIZE prompts:
    tokenize with left-padding (max_length=16384)
    run model.generate() under torch.no_grad()
    slice off prompt prefix tokens
    detect finish_reason by checking last token == eos_token_id
    decode and return {"text", "finish_reason"}
    del tensors + torch.cuda.empty_cache()
```

Left-padding is critical: decoder-only models must see the same positional encodings for all sequences in a batch. Right-padding would shift the context window and corrupt generation.

Finish reason detection: vLLM returned a string `"stop"` or `"length"`. Transformers does not. The equivalent: if the last generated token is not `eos_token_id`, the model ran out of budget → `"length"`. If the last token IS `eos_token_id` → `"stop"`. This feeds directly into `is_uncertain()` which checks for `"length"` in the finish reason.

**Checkpointing upgrade:** Changed from whole-phase checkpoints (one write after all of Phase 1, one after all of Phase 2) to per-chunk in Phase 1 (every 6 questions) and per-question in Phase 2. This was not strictly necessary for vLLM because vLLM completed each phase in seconds/minutes. With Transformers, Phase 1 takes hours — losing it to a pod timeout would cost the full 6-hour allocation.

**Flash Attention 2 → SDPA:** `flash_attn` is not installed in the DSMLP venv. `attn_implementation="flash_attention_2"` raises `ImportError` on model load. PyTorch 2.x ships Scaled Dot-Product Attention as a built-in that uses fused kernels (equivalent to FA2 in practice on A30). Changed to `attn_implementation="sdpa"`. Also fixed `torch_dtype=` → `dtype=` (deprecated parameter name in newer Transformers).

**tqdm progress bar:** Phase 1 previously used `print(..., end="\r")` — a carriage-return overwrite that is invisible in Jupyter output cells. Replaced with `tqdm(total=N, unit="q")` + `pbar.update(len(batch))`, identical to Phase 2's existing bar.

### Parameter changes vs prior optimization pass

| Parameter | Prior (A100/vLLM plan) | DSMLP (A30/Transformers) | Reason for change |
|-----------|------------------------|--------------------------|-------------------|
| `USE_VLLM` | `True` | `False` | vLLM broken on DSMLP CUDA 12.8 |
| `attn_implementation` | `"flash_attention_2"` | `"sdpa"` | `flash_attn` not installed |
| `torch_dtype` → `dtype` | `torch_dtype=bfloat16` | `dtype=bfloat16` | Deprecation fix |
| `CHUNK_SIZE` | N/A (vLLM native batching) | `6` | A30 24GB BF16 KV cache budget |
| `PHASE1_THINKING_BUDGET` | 4096 (A100 plan) | 1024 | Transformers slower; 4× speedup |
| `PHASE1_MAX_TOKENS` | 6144 (A100 plan) | 4096 | Fewer tokens → faster Phase 1 |
| `PHASE2_MAX_TOKENS` | 6144 (A100 plan) | 5120 | Minor reduction for Transformers |
| `PHASE2_N_SAMPLES` | 8 (A100+vLLM plan) | 3 | vLLM handled 8 concurrently; Transformers would take 8× longer |
| `NUMINA_SUBSET` | 5,000 (format-mismatch fix) | 15,000 | A30 24GB confirmed; more data fits |
| `G` (GRPO) | 8 (A100 plan) | 4 (already set) | A30 24GB: G=8 OOMs during rollout |
| Checkpoint frequency | After whole phase | After every chunk (P1) / every question (P2) | Pod timeout recovery |
| Phase 1 progress indicator | `print(..., end="\r")` | `tqdm` bar | `\r` invisible in Jupyter |

### Architecture of the new generation pipeline

```
Phase 1: Fast full sweep
  missing_phase1 = all questions not in checkpoint
  tqdm(total=len(missing_phase1))
  for chunk in range(0, len(missing_phase1), CHUNK_SIZE=6):
    batch = missing_phase1[chunk:chunk+6]
    prompts = [build_chat_prompt(item, thinking_budget=1024) for item in batch]
    outputs = generate_batch(prompts, phase1_params)  # model.generate(), left-padded
    save each result to response_records
    write_checkpoint(CHECKPOINT_PATH, response_records)  ← every 6 questions
    pbar.update(6)

Phase 2: Targeted retry of uncertain questions
  phase1_uncertain = questions where response_records[id].uncertain == True
  for item in tqdm(phase1_uncertain):
    prompt = build_chat_prompt(item, thinking_budget=4096, prefix=RETRY_PREFIX)
    outputs = generate_batch([prompt] * PHASE2_N_SAMPLES=3, phase2_params)
    chosen = choose_best_sample(outputs)  ← majority vote
    response_records[item.id] = chosen
    write_checkpoint(CHECKPOINT_PATH, response_records)  ← every question
```

### Runtime estimates on DSMLP A30 24GB

| Phase | Questions | Avg tokens | Estimated time |
|-------|-----------|------------|----------------|
| Phase 1 | 1,126 | ~1,500 | ~2.3 hrs |
| Phase 2 (est. 25% uncertain) | ~282 × 3 samples | ~4,500 | ~2.5 hrs |
| **Total** | — | — | **~4.8 hrs** (fits 6-hr pod) |

### Notebook state after migration

| Notebook | Status |
|----------|--------|
| `02_inference.ipynb` | ✅ Transformers path active, SDPA, per-chunk checkpoint, tqdm bar, markdown updated |
| `03_qlora_finetune.ipynb` | ✅ NUMINA_SUBSET=15K, MAX_SEQ_LENGTH=4096 (unchanged), section headers accurate |
| `04_grpo_train.ipynb` | ✅ G=4 (A30 safe), Transformers only (no vLLM was ever in this notebook), headers accurate |
| `06_private_submission.ipynb` | ✅ Same Transformers migration as 02, _fix_vllm_cuda deleted, 7 section headers added |

### Next scored run to log (DSMLP overnight)

| Metric | Expected |
|--------|----------|
| Date | TBD (next DSMLP pod) |
| Platform | DSMLP A30 24GB, sp26-cuda128, Transformers BF16+SDPA |
| Config | Base Qwen3-4B-Thinking-2507, CHUNK_SIZE=6, Phase1 budget=1024, Phase2 N=3 |
| MCQ accuracy | TBD (baseline ~42% before training) |
| Free-form accuracy | TBD |
| Overall accuracy | TBD |
| Truncation rate | TBD (target: <15% given shorter budgets vs A100 plan) |
| Phase 2 uncertain rate | TBD |

---

## Methodology clarification and notebook renumbering — 2026-05-26

**Type:** Documentation + structure checkpoint (no intended algorithm change)

### What changed

| Area | Change |
|------|--------|
| Current backend description | The context docs now explicitly treat **Hugging Face Transformers on DSMLP** as the active methodology, not just one option among several |
| Historical vLLM notes | Retained, but reframed as historical / fallback context rather than the default operating path |
| Notebook order | `06_train_ebm_verifier.ipynb` → `05_train_ebm_verifier.ipynb`; `05_private_submission.ipynb` → `06_private_submission.ipynb` |
| Notebook readability | Comments and status prints across notebooks 02–06 were cleaned up to look more human-maintained without changing the underlying pipeline logic |

### Why this matters

- The old numbering made the verifier look like it happened after private submission, which was backwards conceptually.
- The repo had accurate migration history, but the “current methodology” was still too easy to misread as “vLLM if possible, Transformers otherwise.”
- A new reader should now see the intended flow more clearly: public inference first, then optional training/reranking improvements, then verifier, then private submission.

### Pros and cons

**Pros**

- The repo structure now matches the actual dependency order.
- Contributors are less likely to optimize for the wrong backend or run notebooks out of order.
- The context docs are more honest about what is current versus what is historical.

**Cons**

- Historical references need more careful wording because notebook numbers changed.
- Some old notes remain verbose because they preserve the reasoning behind past pivots, even when the current path is simpler.

---

## 2026-05-31 — Truncation crisis and heuristic recovery

### What this iteration solved

Phase 1+2 inference completed on the private set, but post-inference analysis revealed a serious problem: **47.9% (452/943) of responses had no `\boxed{}` answer** because the model was generating long thinking traces that hit `PHASE1_MAX_TOKENS=4096` (or Phase 2's 5120) before reaching the final boxed answer. The grader's extraction returns empty for those — guaranteed wrong.

### The diagnostic

```
Phase 1 (max=4096) truncation: 32.5% empty boxed (190/584)
Phase 2 (max=5120) truncation: 73.0% empty boxed (262/359)  ← worse, because Phase 2 questions are HARDER
MCQ truncation:                 73.7% empty boxed (221/300)
Free-form truncation:           35.9% empty boxed (231/643)

All 452 empty responses had finish_reason="length" — every one was truncated.
```

This explained the Kaggle score (~45%): even with the strong responses being mostly correct, half the answers were unscored.

### Two paths considered

1. **Re-run inference with bigger `MAX_TOKENS` (8192+)** — solves the root cause but requires 1-2 more days of GPU time. Cluster was jammed; deadline was hours away.
2. **Heuristic post-processing recovery** — scan the truncated reasoning for answer signals, append `\boxed{X}` based on regex extraction. ~10 sec runtime, no GPU. Trade-off: heuristic guesses are weaker than letting the model finish, but available immediately.

Picked option 2 given the deadline pressure.

### What was built

| Script | Role |
|---|---|
| `scripts/jsonl_to_csv.py` | Standalone converter that mirrors Section 7 of `06_private_submission.ipynb` so the CSV can be regenerated at any time without re-running the notebook. |
| `scripts/recover_truncated_answers.py` | v1: confidence-tiered regex (high → med → last_standalone → last_number) plus MCQ letter / free-form number extraction. Coverage 52% → 99.5%. |
| `scripts/recover_truncated_answers_v2.py` | v2: adds "should be X" patterns, non-anchored `= X`, MCQ option-count validator (rejects out-of-range letters), multi-answer delimiter detection (a) X b) Y), repeated-value confirmation. `--compare-with` flag for side-by-side v1/v2 diff. |

### Results

- v1 lifted Kaggle from ~45% to ~52% (+7%)
- v2 generated 102 high-confidence recoveries (vs v1's 40), zero regressions vs v1
- v2 has some questionable cases in `delimiter_multi` and `repeated_value` tiers but no v1-correct answer was replaced with worse

### Why it matters

This iteration confirmed that **token budget is the single biggest lever for Qwen3-4B-Thinking accuracy**, far bigger than sample count, EBM verifier, or prompt engineering. The decision matrix for future projects: set `PHASE1_MAX_TOKENS ≥ 8192` from the start even at the cost of longer wall-clock time, because losing 50% of responses to truncation is catastrophic in a way that more samples can never recover from.

The recovery script remains useful as a safety net for any truncated inference run.

