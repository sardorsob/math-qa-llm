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

Fill after loading local copies (counts, MCQ vs free-form, placeholder counts):

| Metric | Value |
|--------|--------|
| Public line count | TBD |
| Private line count | TBD |
| MCQ vs free-form | TBD |
