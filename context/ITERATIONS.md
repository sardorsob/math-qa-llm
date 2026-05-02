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
