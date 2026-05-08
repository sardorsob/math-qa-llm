# Project brief

**Course / competition:** CSE 151B Spring 2026 — mathematical reasoning (free-form and multiple-choice).

**Official starter reference:** [github.com/brooksniu/151B_SP26_Competition](https://github.com/brooksniu/151B_SP26_Competition) — dataset description, starter notebook patterns, and data loading expectations.

## What we are building

An **inference** pipeline (starter: pretrained **`Qwen/Qwen3-4B-Thinking-2507`** via Transformers and/or vLLM — see `notebooks/02_inference.ipynb`) that:

1. Uses **only** the provided files: `public.jsonl`, `private.jsonl`, and `sample_submission.csv` (no extra competition data).
2. Loads **public** JSONL for development and **private** JSONL for submission.
3. Runs the model so each problem gets a **full raw output** (chain-of-thought / thinking traces included), not only a parsed final answer.
4. Produces a **CSV** that matches the competition format for **every** private `id`.

This is **not** “training a new LLM from scratch” in the starter flow; weights come from Hugging Face and are **downloaded automatically** on first use (never committed to git). Optional finetuning would be extra work beyond the baseline notebook.

## Hard competition constraints

- **Final inference model:** must be **`Qwen/Qwen3-4B-Thinking-2507`**. Alternative base models are not allowed for final responses.
- **Allowed:** prompt engineering, chain-of-thought prompting, self-consistency, progressive prompting, supervised fine-tuning, LoRA/QLoRA, DPO/GRPO/RL-style model-intrinsic methods.
- **Not allowed at inference:** external model/API calls, calculators, code interpreters, SymPy/Python tool use, retrieval, or other tool-augmented generation.
- **Submission split:** the upload must contain predictions for **private** problems, not public validation problems.

## Success criteria

- **Submission validity:** One CSV row per `id` in `private.jsonl`; header `id,response`; `response` is the **complete** model output used at inference time.
- **Expected private row count:** current competition UI says the submission should have **943 rows plus header**.
- **Local quality:** On the public set, scores from `judger.py` (and related utilities) reflect how well extracted answers match ground truth — see competition rules for exact metrics.
- **Format correctness:** Free-form items need **all** sub-answers correct when evaluated; MCQ needs the selected letter to match ground truth (evaluation extracts answers from the response trace).

## Current implementation snapshot

As of 2026-05-06, `notebooks/02_inference.ipynb` is a self-contained vLLM inference notebook with an adaptive two-phase retry design:

- Required model: **`Qwen/Qwen3-4B-Thinking-2507`**
- Current default split: **`DATA_MODE = "public"`**
- Current default size: **`N_QUESTIONS = None`**, meaning all rows in the selected split
- Current run name: **`adaptive_{DATA_MODE}_v2`**
- Prompting: structured **UNDERSTAND / PLAN / SOLVE / VERIFY / ANSWER**
- Phase 1: all selected rows, `thinking_budget=1024`, `max_tokens=2048`, one sample
- Phase 2: only uncertain rows, `thinking_budget=4096`, `max_tokens=6144`, `PHASE2_N_SAMPLES=3`, majority vote
- Phase 3: removed from current source after the earlier full-public `v1` run proved too slow
- Submission export: notebook Section 10 writes a quoted CSV with `id,response`

For a private upload, switch **`DATA_MODE = "private"`**, keep **`N_QUESTIONS = None`**, rerun config + dataset cells, confirm **943 rows**, skip scoring/summary, then run save + CSV.

Important: a CSV produced from the public split has **1126** rows and is not a valid leaderboard submission.

## Deliverables

- Reproducible code under `src/`, `scripts/`, `notebooks/`.
- Run artifacts (JSONL logs, configs) under `artifacts/` where appropriate.
- Final upload file under `artifacts/submissions/` (or equivalent), with RFC-4180-style quoting so commas, newlines, and quotes inside `response` do not break rows.

## Submission CSV (competition rules)

| Column | Content |
|--------|---------|
| `id` | Integer, must match `private.jsonl` |
| `response` | Full model trace — reasoning **and** final answer presentation (e.g. `\boxed{...}` patterns as produced by the model) |

**Important:** Evaluators parse the final answer from this trace. The field must not be stripped down to only the extracted answer string.

**Quoting:** Use standard CSV double-quote rules; escape embedded `"` as `""`.

**License (competition data):** CC BY-NC-SA 4.0 (per release notes).

## Related docs

- `DATASETS.md` — field schema, formats, examples.
- `STATUS.md` — what’s implemented vs still open (inference notebook, submission path).
- `DECISIONS.md` — rationale for generation defaults, env split, Transformers vs vLLM default.
- `ENVIRONMENT_SETUP.md` — conda/venv, HF token, Qwen3-Thinking generation notes.
