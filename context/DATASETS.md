# Datasets

Source: CSE 151B Spring 2026 competition materials. Starter assets and narrative also live at [151B_SP26_Competition](https://github.com/brooksniu/151B_SP26_Competition).

Data is **JSONL**: one JSON object per line. Problems span mathematical domains and difficulty (high school through graduate) and mix **response formats**.

## Files you need (only assets provided for the competition)

There are **no other official dataset files** beyond:

| Asset | Typical location | Split | Ground truth | Purpose |
|-------|------------------|-------|----------------|---------|
| `public.jsonl` | e.g. `data/raw/public.jsonl` | Public | Yes | Development, local scoring |
| `private.jsonl` | e.g. `data/raw/private.jsonl` | Private | **No** | Leaderboard; one CSV row per `id` |
| `sample_submission.csv` | e.g. `data/external/sample_submission.csv` | — | — | Example `id,response` format |

Point your notebook/config at wherever you store these paths; the schema is the same regardless of folder names.

## Fields

| Field | Presence | Meaning |
|-------|----------|---------|
| `id` | Always | Unique integer identifier for the problem |
| `question` | Always | Problem statement in **LaTeX**. Free-form items use `[ANS]` placeholders where answers belong |
| `answer` | Public only | Ground truth: **list of strings** for free-form (one string per `[ANS]`), or a **single capital letter** (e.g. `"C"`) for multiple-choice |
| `options` | MCQ only | List of candidate choices in LaTeX |

### Question formats

**Free-form:** The model must produce one or more numerical or symbolic answers matching each `[ANS]`. One question may require a single answer or multiple. **Grading:** every sub-answer must be correct for the item to count as correct.

**Multiple-choice:** The model must select the correct option. Ground truth is one letter.

## Example shapes

**Free-form (single answer):**

```json
{"question": "Here is an expression with negative exponents.\n$\\frac{1}{(-8)^{-3}}=$ [ANS]\nEvaluate the expression.", "answer": ["-512"], "id": 4}
```

**Free-form (multiple `[ANS]`):**

```json
{"question": "If $f(x)=4x^2+x+2$, find the following:\n(a) $f(3)=$ [ANS]\n(b) $f(-3)=$ [ANS]\n(c) $f(-2)=$ [ANS]", "answer": ["41", "35", "16"], "id": 2}
```

**Multiple-choice** (abbreviated):

```json
{"question": "Given $u(x, y) = x^3 + ...$ find $f(z)$ ...", "options": ["$$...$$", "$$...$$"], "answer": "C", "id": 1}
```

**Private row** (illustrative — no `answer`):

- Expect at least `id` and `question`; MCQ rows include `options`.

## Splits and distribution

- **Public:** Labels provided; use for validation and pipeline debugging.
- **Private:** Same broad distribution as public (difficulty, domains, formats) but **answers withheld**.

## Stats

Re-count after any official data refresh. **Observed once** from a local `public.jsonl` load (dev machine):

| Metric | Value |
|--------|--------|
| Public line count | 1126 |
| Public MCQ / free-form | 375 MCQ, 751 free-form |
| Private line count | 943 observed locally / expected by competition upload UI |

## Current workflow notes

- `public.jsonl` is for validation only. A CSV generated from the public split is **not** a valid leaderboard submission.
- `N_QUESTIONS = None` means “all rows in the currently selected split,” not necessarily private rows. With `DATA_MODE="public"` this is 1126 public rows; with `DATA_MODE="private"` this is 943 private rows.
- Current notebook source default is `DATA_MODE="public"` and `N_QUESTIONS=None`, so it will process all public rows unless changed.
- For final submission set `DATA_MODE="private"` and `N_QUESTIONS=None`; then skip scoring because private rows have no `answer`.
- The generated CSV must contain **all 943 private ids** with columns exactly `id,response`.
- Before spending GPU time, rerun the dataset cell and confirm it prints **943** loaded/running rows for private.
