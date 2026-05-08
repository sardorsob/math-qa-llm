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
