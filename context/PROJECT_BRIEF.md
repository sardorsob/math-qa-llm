# Project brief

**Course / competition:** CSE 151B Spring 2026 — mathematical reasoning (free-form and multiple-choice).

**Official starter reference:** [github.com/brooksniu/151B_SP26_Competition](https://github.com/brooksniu/151B_SP26_Competition) — dataset description, starter notebook patterns, and data loading expectations.

## What we are building

An **inference** pipeline (starter: pretrained **Qwen3-4B-Thinking** via Transformers and/or vLLM — see `notebooks/02_inference.ipynb`) that:

1. Uses **only** the provided files: `public.jsonl`, `private.jsonl`, and `sample_submission.csv` (no extra competition data).
2. Loads **public** JSONL for development and **private** JSONL for submission.
3. Runs the model so each problem gets a **full raw output** (chain-of-thought / thinking traces included), not only a parsed final answer.
4. Produces a **CSV** that matches the competition format for **every** private `id`.

This is **not** “training a new LLM from scratch” in the starter flow; weights come from Hugging Face and are **downloaded automatically** on first use (never committed to git). Optional finetuning would be extra work beyond the baseline notebook.

## Success criteria

- **Submission validity:** One CSV row per `id` in `private.jsonl`; header `id,response`; `response` is the **complete** model output used at inference time.
- **Local quality:** On the public set, scores from `judger.py` (and related utilities) reflect how well extracted answers match ground truth — see competition rules for exact metrics.
- **Format correctness:** Free-form items need **all** sub-answers correct when evaluated; MCQ needs the selected letter to match ground truth (evaluation extracts answers from the response trace).

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
- `STATUS.md` — current checklist and planned notebook / pipeline steps.
