#!/usr/bin/env python3
"""Heuristic answer recovery v2 — smarter extraction with validation.

What's new vs v1 (recover_truncated_answers.py):

  1. "should be X" patterns added to both MCQ_HIGH_CONF and FF_HIGH_CONF.
  2. Non-anchored "= X" pattern (v1 required end-of-string; v2 catches mid-text).
  3. MCQ OPTION-COUNT VALIDATOR — rejects extracted letters outside the question's
     actual option range (e.g. "G" when question only has A-E).
  4. Multi-answer free-form delimiter detection — looks for "a)", "b)", "1.", "2.",
     "first:", "second:" markers BEFORE falling back to "last N numbers".
  5. Repeated-value confirmation tier for free-form — if a number appears 3+ times
     in the tail with no nearby negation, use it.

Comparison mode (`--compare-with v1.csv`) prints a side-by-side diff so you can
sanity-check before submitting.

Writes:
  - artifacts/logs/runs/private_qlora_v1_merged_recovered_v2_results.jsonl
  - artifacts/submissions/submission_qlora_v1_merged_<YYYY-MM-DD>_recovered_v2.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from datetime import date
from pathlib import Path


# ── Paths ─────────────────────────────────────────────────────────────────────
def repo_root() -> Path:
    p = Path(__file__).resolve()
    for d in (p, *p.parents):
        if (d / "judger.py").is_file():
            return d
    return p.parent


REPO_ROOT       = repo_root()
MODEL_LABEL     = "qlora_v1_merged"
RUN_NAME        = f"private_{MODEL_LABEL}"

PRIVATE_PATH    = REPO_ROOT / "data" / "raw" / "private.jsonl"
CHECKPOINT_PATH = REPO_ROOT / "artifacts" / "logs" / "runs" / f"{RUN_NAME}_checkpoint.jsonl"
OUTPUT_JSONL    = REPO_ROOT / "artifacts" / "logs" / "runs" / f"{RUN_NAME}_recovered_v2_results.jsonl"
SUBMISSION_CSV  = (
    REPO_ROOT / "artifacts" / "submissions"
    / f"submission_{MODEL_LABEL}_{date.today().isoformat()}_recovered_v2.csv"
)


# ── \boxed{} extraction (handles nested braces) ───────────────────────────────
def extract_boxed(text: str) -> str:
    if not text:
        return ""
    needle = r"\boxed{"
    matches = []
    i = 0
    while i < len(text):
        idx = text.find(needle, i)
        if idx == -1:
            break
        j, depth, start = idx + len(needle), 1, idx + len(needle)
        while j < len(text) and depth:
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    matches.append(text[start:j])
                    break
            j += 1
        i = idx + 1
    return matches[-1].strip() if matches else ""


def ascii_safe(s: str) -> str:
    return s.encode("ascii", errors="replace").decode("ascii")


# ── MCQ extraction: confidence-ranked patterns ────────────────────────────────
# NEW in v2: added "should be X" pattern
MCQ_HIGH_CONF = [
    r"the\s+answer\s+is\s+\*?\*?([A-J])\b",
    r"answer\s*[:\-]\s*\*?\*?([A-J])\b",
    r"correct\s+answer\s+is\s+\*?\*?([A-J])\b",
    r"correct\s+choice\s+is\s+\*?\*?([A-J])\b",
    r"final\s+answer\s*[:\-]?\s*\*?\*?([A-J])\b",
    r"answer\s+should\s+be\s+\*?\*?([A-J])\b",
    r"answer\s+would\s+be\s+\*?\*?([A-J])\b",
    r"should\s+be\s+\*?\*?([A-J])\b",                       # NEW v2
    r"choose\s+\*?\*?([A-J])\b",
    r"pick\s+\*?\*?([A-J])\b",
    r"option\s+\*?\*?([A-J])\b\s+is\s+correct",
    r"\*?\*?([A-J])\b\s+is\s+the\s+(?:correct|right)\s+answer",
    r"so\s+the\s+answer\s+is\s+\*?\*?([A-J])\b",
]

MCQ_MED_CONF = [
    r"i\s+(?:think|believe|guess)\s+(?:it'?s\s+)?\*?\*?([A-J])\b",
    r"(?:most\s+likely|probably|likely)\s+\*?\*?([A-J])\b",
    r"i'?ll\s+go\s+with\s+\*?\*?([A-J])\b",
    r"go\s+with\s+\*?\*?([A-J])\b",
    r"my\s+best\s+guess\s+is\s+\*?\*?([A-J])\b",
    r"leaning\s+(?:toward|towards)\s+\*?\*?([A-J])\b",
    r"option\s+\*?\*?([A-J])\b",
    r"looks\s+like\s+\*?\*?([A-J])\b",
    r"answer\s+is\s+(?:likely|probably)\s+\*?\*?([A-J])\b",
]


def extract_mcq_letter(response: str, max_letter: str = "J") -> tuple[str, str]:
    """Return (letter, source) or ('', '') if nothing found.

    NEW v2: validates letter against `max_letter` (computed from question.options length).
    """
    tail = response[-1500:] if len(response) > 1500 else response

    valid_letters = "ABCDEFGHIJ"[: ord(max_letter) - ord("A") + 1]

    # High-confidence patterns
    for pat in MCQ_HIGH_CONF:
        matches = list(re.finditer(pat, tail, re.IGNORECASE))
        for m in reversed(matches):
            letter = m.group(1).upper()
            if letter in valid_letters:
                return letter, "high"
            # if letter is out of range (e.g. "G" when only A-E), continue searching

    # Medium-confidence patterns
    for pat in MCQ_MED_CONF:
        matches = list(re.finditer(pat, tail, re.IGNORECASE))
        for m in reversed(matches):
            letter = m.group(1).upper()
            if letter in valid_letters:
                return letter, "med"

    # Fallback: last standalone capital letter in valid range
    standalone_pat = r"(?:\*{0,2}|\(|\s|^)([A-J])(?:\*{0,2}|\)|\s|\.|,|$)"
    matches = list(re.finditer(standalone_pat, tail))
    if matches:
        # Filter to valid range first
        valid_matches = [m for m in matches if m.group(1) in valid_letters]
        if valid_matches:
            # Prefer matches in the very last 500 chars
            late_matches = [m for m in valid_matches if m.start() >= len(tail) - 500]
            if late_matches:
                return late_matches[-1].group(1), "last_standalone_late"
            return valid_matches[-1].group(1), "last_standalone"

    return "", ""


# ── Free-form extraction ─────────────────────────────────────────────────────
NUM_TOKEN = (
    r"(?:-?\d+(?:\.\d+)?(?:/\d+)?"
    r"|-?\\?[a-zA-Z](?:\^?\d+)?"
    r"|\\[a-zA-Z]+(?:\{[^{}]*\})?"
    r"|-?\d+[a-zA-Z]+"
    r"|-?\d+/\d+\\?[a-zA-Z]*"
    r")"
)

ENGLISH_BLACKLIST = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "should",
    "could", "may", "might", "must", "shall", "can",
    "this", "that", "these", "those", "which", "what", "who",
    "where", "when", "why", "how", "and", "or", "but", "not", "no",
    "for", "of", "to", "in", "on", "at", "by", "with", "from", "as",
    "if", "so", "such", "very", "just", "only",
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
    "my", "your", "his", "its", "our", "their",
    "answer", "solution", "result", "value", "number", "letter", "option",
    "choice", "correct", "right", "wrong", "say", "think", "guess",
}


def is_valid_freeform(val: str) -> bool:
    if not val or len(val) > 40:
        return False
    if val.lower() in ENGLISH_BLACKLIST:
        return False
    if not (any(c.isdigit() for c in val) or "\\" in val or len(val) <= 2):
        return False
    return True


# v1 patterns + v2 additions
# NEW v2: "equals X" without end-of-string anchor
FF_HIGH_CONF = [
    rf"the\s+answer\s+is\s+\$?\\?({NUM_TOKEN})\$?",
    rf"answer\s*[:\-]\s*\$?\\?({NUM_TOKEN})\$?",
    rf"final\s+answer\s*[:\-]?\s*\$?\\?({NUM_TOKEN})\$?",
    rf"so\s+(?:the\s+answer|result)\s+is\s+\$?\\?({NUM_TOKEN})\$?",
    rf"therefore\s*[,:]?\s*\$?\\?({NUM_TOKEN})\$?",
    rf"thus\s*[,:]?\s*\$?\\?({NUM_TOKEN})\$?",
    rf"=\s*\$?\\?({NUM_TOKEN})\$?\s*$",                                    # v1: end-anchored
    rf"=\s*\$?\\?({NUM_TOKEN})\$?(?=[\s.,)]|$)",                           # NEW v2: not end-anchored
    rf"equals\s+\$?\\?({NUM_TOKEN})\$?",
    rf"result\s+is\s+\$?\\?({NUM_TOKEN})\$?",
    rf"answer\s+should\s+be\s+\$?\\?({NUM_TOKEN})\$?",
    rf"should\s+be\s+\$?\\?({NUM_TOKEN})\$?",                              # NEW v2
]

FF_MED_CONF = [
    rf"i\s+(?:get|got|obtain|find|conclude)\s+\$?\\?({NUM_TOKEN})\$?",
    rf"(?:most\s+likely|probably|likely)\s+\$?\\?({NUM_TOKEN})\$?",
    rf"my\s+best\s+(?:guess|estimate)\s+is\s+\$?\\?({NUM_TOKEN})\$?",
    rf"value\s+is\s+\$?\\?({NUM_TOKEN})\$?",
]


# NEW v2: repeated-value confirmation
def repeated_value_extract(tail: str) -> tuple[str, str]:
    """Find a number that appears 3+ times in the tail and isn't negated."""
    NEG_WORDS = ("doesn't", "doesnt", "isn't", "isnt", "not", "wrong", "no", "fail")

    # Find all numeric tokens with their positions
    pattern = r"\b(-?\d+(?:\.\d+)?(?:/\d+)?)\b"
    matches = list(re.finditer(pattern, tail))
    if not matches:
        return "", ""

    # Count occurrences
    counts = Counter(m.group(1) for m in matches)
    candidates = [(v, c) for v, c in counts.items() if c >= 3]

    if not candidates:
        return "", ""

    # Among candidates, find the one whose LATEST occurrence is not preceded by negation
    candidates.sort(key=lambda x: -x[1])  # by count, descending

    for val, count in candidates:
        # Skip values that look like dates, IDs, or way out of range
        try:
            num = float(val.split("/")[0])
            if abs(num) > 1e9 or (abs(num) > 9999 and "." not in val):
                continue
        except ValueError:
            continue

        # Get last position of this value
        last_match = None
        for m in reversed(matches):
            if m.group(1) == val:
                last_match = m
                break
        if not last_match:
            continue

        # Check for negation in the 30 chars before
        lookback_start = max(0, last_match.start() - 30)
        lookback = tail[lookback_start : last_match.start()].lower()
        if any(neg in lookback for neg in NEG_WORDS):
            continue  # negated, skip

        if is_valid_freeform(val):
            return val, "repeated_value"

    return "", ""


def extract_freeform_value(response: str) -> tuple[str, str]:
    tail = response[-1500:] if len(response) > 1500 else response

    for pat in FF_HIGH_CONF:
        matches = list(re.finditer(pat, tail, re.IGNORECASE))
        for m in reversed(matches):
            val = m.group(1).strip()
            if is_valid_freeform(val):
                return val, "high"

    for pat in FF_MED_CONF:
        matches = list(re.finditer(pat, tail, re.IGNORECASE))
        for m in reversed(matches):
            val = m.group(1).strip()
            if is_valid_freeform(val):
                return val, "med"

    # NEW v2: repeated-value confirmation, slotted above last-number fallback
    val, source = repeated_value_extract(tail)
    if val:
        return val, source

    # Final fallback: last number
    last_500 = tail[-500:]
    nums = re.findall(r"-?\d+(?:\.\d+)?(?:/\d+)?", last_500)
    if nums:
        return nums[-1], "last_number"

    return "", ""


# NEW v2: multi-answer free-form with delimiter detection
def extract_multi_answer_with_delimiters(response: str, n_ans: int) -> tuple[str, str]:
    """Look for explicit a)/b)/1./2./first:/second: markers in the tail.

    Returns (joined_values, source) or ('', '') if no delimiter pattern found.
    """
    tail = response[-2000:] if len(response) > 2000 else response

    # Pattern set 1: lettered "a) X", "b) Y"
    letter_pattern = r"(?:^|\n|\s)([a-h])\)\s*([^\n,;]{1,30})"
    letter_matches = list(re.finditer(letter_pattern, tail, re.IGNORECASE))

    # Pattern set 2: numbered "1. X", "2. Y" or "1) X", "2) Y"
    num_pattern = r"(?:^|\n|\s)([1-9])[.)]\s*([^\n,;]{1,30})"
    num_matches = list(re.finditer(num_pattern, tail))

    # Pattern set 3: ordinal "first: X", "second: Y"
    ordinal_words = ["first", "second", "third", "fourth", "fifth"]
    ord_pattern = (
        r"(?:^|\n|\s)(" + "|".join(ordinal_words) + r")\s*[:\-]\s*([^\n,;]{1,30})"
    )
    ord_matches = list(re.finditer(ord_pattern, tail, re.IGNORECASE))

    best_matches = None
    if len(letter_matches) >= n_ans:
        best_matches = letter_matches
    elif len(num_matches) >= n_ans:
        best_matches = num_matches
    elif len(ord_matches) >= n_ans:
        best_matches = ord_matches

    if not best_matches:
        return "", ""

    # Take the last n_ans matches (the model's final commitment)
    selected = best_matches[-n_ans:]
    values = []
    for m in selected:
        raw = m.group(2).strip()
        # Extract a number-like token from the raw value
        num_match = re.search(NUM_TOKEN, raw)
        if num_match:
            v = num_match.group(0).strip()
            if is_valid_freeform(v):
                values.append(v)
            else:
                # try to find any number
                fallback = re.search(r"-?\d+(?:\.\d+)?(?:/\d+)?", raw)
                if fallback:
                    values.append(fallback.group(0))
                else:
                    return "", ""  # delimiter found but value not extractable
        else:
            return "", ""

    if len(values) == n_ans:
        return ", ".join(values), "delimiter_multi"

    return "", ""


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--compare-with",
        help="Path to v1 CSV. Prints side-by-side diff of recoveries.",
        type=Path,
        default=None,
    )
    args = parser.parse_args()

    print(f"REPO_ROOT       : {REPO_ROOT}")
    print(f"CHECKPOINT_PATH : {CHECKPOINT_PATH}")
    print(f"OUTPUT_JSONL    : {OUTPUT_JSONL}")
    print(f"SUBMISSION_CSV  : {SUBMISSION_CSV}")
    print()

    # Load private dataset (gives options list per question)
    private_lookup = {}
    with open(PRIVATE_PATH, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            opts = item.get("options")
            private_lookup[str(item["id"])] = {
                "options":   opts,
                "is_mcq":    bool(opts),
                "n_options": len(opts) if opts else 0,
                "question":  item.get("question", ""),
            }

    # Load checkpoint
    records = {}
    with open(CHECKPOINT_PATH, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            records[str(r["id"])] = r

    print(f"Loaded {len(records)} checkpoint records")
    print(f"Loaded {len(private_lookup)} private questions")
    print()

    stats = Counter()
    source_breakdown = Counter()
    examples: dict[str, list] = {}

    updated = []
    for item_id, info in private_lookup.items():
        rec = records.get(item_id)
        if not rec:
            stats["missing_from_checkpoint"] += 1
            continue

        response = rec.get("response", "")
        existing = extract_boxed(response)

        if existing:
            stats["already_had_boxed"] += 1
            new_response = response
            recovery_source = "had_boxed"
            recovered_value = existing
        else:
            stats["needs_recovery"] += 1

            if info["is_mcq"]:
                # Validate against actual option count
                n_opts = info["n_options"]
                max_letter = chr(65 + n_opts - 1) if n_opts > 0 else "J"
                value, source = extract_mcq_letter(response, max_letter=max_letter)
            else:
                n_ans = info["question"].count("[ANS]") or 1
                if n_ans > 1:
                    # NEW v2: try delimiter detection first
                    value, source = extract_multi_answer_with_delimiters(response, n_ans)
                    if not value:
                        # Fall back to v1's "last N numbers" behavior
                        tail = response[-1500:]
                        nums = re.findall(r"-?\d+(?:\.\d+)?(?:/\d+)?", tail)
                        if len(nums) >= n_ans:
                            value = ", ".join(nums[-n_ans:])
                            source = "last_numbers_multi"
                        else:
                            value, source = extract_freeform_value(response)
                else:
                    value, source = extract_freeform_value(response)

            if value:
                new_response = response.rstrip() + f"\n\n\\boxed{{{value}}}"
                recovery_source = source
                recovered_value = value
                stats[f"recovered_{source}"] += 1
                source_breakdown[source] += 1
                examples.setdefault(source, []).append(
                    (item_id, info["is_mcq"], value, response[-300:])
                )
            else:
                new_response = response
                recovery_source = "no_recovery"
                recovered_value = ""
                stats["no_recovery_possible"] += 1
                examples.setdefault("no_recovery", []).append(
                    (item_id, info["is_mcq"], "", response[-300:])
                )

        updated.append({
            "id":                rec["id"],
            "is_mcq":           info["is_mcq"],
            "response":          new_response,
            "phase_used":        rec.get("phase_used"),
            "uncertain":         rec.get("uncertain"),
            "finish_reason":     rec.get("finish_reason"),
            "consensus_count":   rec.get("consensus_count"),
            "n_samples":         rec.get("n_samples"),
            "recovery_source":   recovery_source,
            "recovered_value":   recovered_value,
            "model":             MODEL_LABEL,
        })

    # Stats
    print("=== RECOVERY STATS ===")
    for k in sorted(stats.keys()):
        print(f"  {k}: {stats[k]}")
    print()
    print("=== RECOVERY SOURCE BREAKDOWN ===")
    for k in sorted(source_breakdown.keys()):
        print(f"  {k}: {source_breakdown[k]}")
    print()

    # Examples
    for cat in sorted(examples.keys()):
        ex = examples[cat]
        if ex:
            print(f"=== EXAMPLES: {cat} (first 2) ===")
            for item_id, is_mcq, val, snippet in ex[:2]:
                kind = "MCQ" if is_mcq else "FF"
                print(f"  id={item_id} ({kind}) -> recovered='{ascii_safe(val)}'")
                snippet_ascii = ascii_safe(snippet.strip())[-200:]
                print(f"    last 200 chars: ...{snippet_ascii}")
            print()

    # Write JSONL
    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    updated_sorted = sorted(updated, key=lambda r: int(r["id"]))
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for rec in updated_sorted:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Wrote JSONL: {OUTPUT_JSONL}")

    # Write CSV
    SUBMISSION_CSV.parent.mkdir(parents=True, exist_ok=True)
    csv_rows = sorted(
        [{"id": r["id"], "response": r["response"]} for r in updated],
        key=lambda r: r["id"],
    )
    with open(SUBMISSION_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "response"], quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"Wrote CSV  : {SUBMISSION_CSV} ({len(csv_rows)} rows)")

    # Final coverage
    print()
    print("=== POST-RECOVERY COVERAGE ===")
    final_has_boxed = sum(1 for r in updated if extract_boxed(r["response"]))
    final_mcq_has = sum(1 for r in updated if r["is_mcq"] and extract_boxed(r["response"]))
    final_mcq_no  = sum(1 for r in updated if r["is_mcq"] and not extract_boxed(r["response"]))
    final_ff_has  = sum(1 for r in updated if not r["is_mcq"] and extract_boxed(r["response"]))
    final_ff_no   = sum(1 for r in updated if not r["is_mcq"] and not extract_boxed(r["response"]))
    n = len(updated)
    print(f"Total has \\boxed{{}}: {final_has_boxed}/{n} ({100*final_has_boxed/n:.1f}%)")
    print(f"  MCQ:       has={final_mcq_has}  no={final_mcq_no}")
    print(f"  Free-form: has={final_ff_has}  no={final_ff_no}")

    # Comparison mode
    if args.compare_with:
        compare_with_v1(args.compare_with, updated)


def compare_with_v1(v1_csv_path: Path, v2_updated: list[dict]) -> None:
    print()
    print(f"=== COMPARISON WITH v1: {v1_csv_path.name} ===")

    if not v1_csv_path.is_file():
        print(f"  ERROR: file not found")
        return

    v1_lookup = {}
    with open(v1_csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            v1_lookup[str(row["id"])] = row["response"]

    same      = 0
    v2_added  = 0  # v1 had no boxed, v2 does
    v1_added  = 0  # v1 had boxed, v2 doesn't (regression)
    disagree  = 0  # both have boxed but different content
    disagreements = []

    for rec in v2_updated:
        v1_resp = v1_lookup.get(str(rec["id"]), "")
        v2_resp = rec["response"]
        v1_box = extract_boxed(v1_resp)
        v2_box = extract_boxed(v2_resp)

        if v1_box == v2_box and v1_resp == v2_resp:
            same += 1
        elif not v1_box and v2_box:
            v2_added += 1
        elif v1_box and not v2_box:
            v1_added += 1
        elif v1_box != v2_box:
            disagree += 1
            if len(disagreements) < 10:
                disagreements.append((rec["id"], rec["is_mcq"], v1_box, v2_box, rec.get("recovery_source")))

    print(f"  v1 == v2 (identical):                {same}")
    print(f"  v2 added new \\boxed{{}}:              {v2_added}")
    print(f"  v1 had \\boxed{{}} but v2 lost it:     {v1_added}  (regression!)")
    print(f"  Both have \\boxed{{}}, content differs: {disagree}")
    print()

    if disagreements:
        print(f"  SAMPLE DISAGREEMENTS (first 10):")
        for item_id, is_mcq, v1box, v2box, source in disagreements:
            kind = "MCQ" if is_mcq else "FF"
            print(f"    id={item_id} ({kind}, src={source})")
            print(f"      v1: '{ascii_safe(v1box)[:50]}'")
            print(f"      v2: '{ascii_safe(v2box)[:50]}'")


if __name__ == "__main__":
    main()
