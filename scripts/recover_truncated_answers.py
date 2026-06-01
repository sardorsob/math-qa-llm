#!/usr/bin/env python3
"""Heuristic answer recovery from truncated responses.

Many private-set responses got cut off by max_new_tokens before the model could
write its final \\boxed{...}. The reasoning often still contains a strong signal
about the intended answer — e.g. "I think the answer is C", "= 42", "looks like
option D", etc. This script scans the last portion of each response for those
signals and appends \\boxed{X} where one would have likely gone.

Strategy:
  - Responses that ALREADY have \\boxed{}: untouched.
  - Responses with NO \\boxed{}: look for confidence cues, by priority:
       1. "the answer is X"  / "answer: X"  / "= X"            (highest conf)
       2. "I'll choose X" / "I pick X" / "I think X"
       3. "most likely X" / "probably X" / "likely X" / "correct answer is X"
       4. (MCQ only) last standalone letter A-J in the final 1500 chars
       5. (Free-form) last number-like token in the final 1500 chars
  - If nothing extractable, leave response unchanged (still no \\boxed{}).

Writes:
  - artifacts/logs/runs/private_qlora_v1_merged_results.jsonl  (with recovery flags)
  - artifacts/submissions/submission_qlora_v1_merged_<YYYY-MM-DD>_recovered.csv
"""

from __future__ import annotations

import csv
import json
import re
import statistics
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
OUTPUT_JSONL    = REPO_ROOT / "artifacts" / "logs" / "runs" / f"{RUN_NAME}_recovered_results.jsonl"
SUBMISSION_CSV  = (
    REPO_ROOT / "artifacts" / "submissions"
    / f"submission_{MODEL_LABEL}_{date.today().isoformat()}_recovered.csv"
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


# ── MCQ extraction: confidence-ranked patterns ────────────────────────────────
MCQ_HIGH_CONF = [
    r"the\s+answer\s+is\s+\*?\*?([A-J])\b",
    r"answer\s*[:\-]\s*\*?\*?([A-J])\b",
    r"correct\s+answer\s+is\s+\*?\*?([A-J])\b",
    r"correct\s+choice\s+is\s+\*?\*?([A-J])\b",
    r"final\s+answer\s*[:\-]?\s*\*?\*?([A-J])\b",
    r"answer\s+should\s+be\s+\*?\*?([A-J])\b",
    r"answer\s+would\s+be\s+\*?\*?([A-J])\b",
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


def extract_mcq_letter(response: str) -> tuple[str, str]:
    """Return (letter, source) or ('', '') if nothing found.

    Source labels: 'high', 'med', 'last_standalone'.
    Looks in the LAST 1500 chars (where the conclusion usually lives).
    """
    tail = response[-1500:] if len(response) > 1500 else response
    tail_lower = tail.lower()

    # High-confidence patterns
    for pat in MCQ_HIGH_CONF:
        for m in reversed(list(re.finditer(pat, tail_lower, re.IGNORECASE))):
            # Re-find in original-case to get the actual letter
            start_pos = m.start()
            actual = re.search(pat, tail[start_pos:start_pos + m.end() - m.start() + 2], re.IGNORECASE)
            if actual:
                letter = actual.group(1).upper()
                if letter in "ABCDEFGHIJ":
                    return letter, "high"

    # Medium-confidence patterns
    for pat in MCQ_MED_CONF:
        matches = list(re.finditer(pat, tail, re.IGNORECASE))
        if matches:
            # Take the LAST one — usually the model's final commitment
            letter = matches[-1].group(1).upper()
            if letter in "ABCDEFGHIJ":
                return letter, "med"

    # Fallback: last standalone capital A-J letter in the tail
    # (e.g. "**B**" or " B " or "(C)" or "C." patterns)
    standalone_pat = r"(?:\*{0,2}|\(|\s|^)([A-J])(?:\*{0,2}|\)|\s|\.|,|$)"
    matches = list(re.finditer(standalone_pat, tail))
    if matches:
        # Prefer matches in the very last 500 chars
        last_500 = tail[-500:]
        late_matches = [m for m in matches if m.start() >= len(tail) - 500]
        if late_matches:
            return late_matches[-1].group(1), "last_standalone_late"
        return matches[-1].group(1), "last_standalone"

    return "", ""


# ── Free-form extraction: confidence-ranked patterns ──────────────────────────
# A "number-like" token: integer, decimal, fraction, LaTeX expression, variable
# Strictly excludes common English words like "the", "answer", "is".
NUM_TOKEN = (
    r"(?:-?\d+(?:\.\d+)?(?:/\d+)?"           # number/decimal/fraction: 42, -3.14, 1/2
    r"|-?\\?[a-zA-Z](?:\^?\d+)?"             # variable: x, y, x^2
    r"|\\[a-zA-Z]+(?:\{[^{}]*\})?"           # LaTeX: \pi, \sqrt{2}
    r"|-?\d+[a-zA-Z]+"                        # 2x, 3pi
    r"|-?\d+/\d+\\?[a-zA-Z]*"                 # fractional w/ var
    r")"
)

# Common English words that the regex might accidentally pick up. If our recovered
# value matches one of these, treat as a non-recovery and try the next pattern.
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
    """Reject obvious English words and overly long tokens."""
    if not val or len(val) > 40:
        return False
    if val.lower() in ENGLISH_BLACKLIST:
        return False
    # Must contain at least one digit OR a backslash (LaTeX) OR be a short symbol
    if not (any(c.isdigit() for c in val) or "\\" in val or len(val) <= 2):
        return False
    return True

FF_HIGH_CONF = [
    rf"the\s+answer\s+is\s+\$?\\?({NUM_TOKEN})\$?",
    rf"answer\s*[:\-]\s*\$?\\?({NUM_TOKEN})\$?",
    rf"final\s+answer\s*[:\-]?\s*\$?\\?({NUM_TOKEN})\$?",
    rf"so\s+(?:the\s+answer|result)\s+is\s+\$?\\?({NUM_TOKEN})\$?",
    rf"therefore\s*[,:]?\s*\$?\\?({NUM_TOKEN})\$?",
    rf"thus\s*[,:]?\s*\$?\\?({NUM_TOKEN})\$?",
    rf"=\s*\$?\\?({NUM_TOKEN})\$?\s*$",
    rf"equals\s+\$?\\?({NUM_TOKEN})\$?",
    rf"result\s+is\s+\$?\\?({NUM_TOKEN})\$?",
    rf"answer\s+should\s+be\s+\$?\\?({NUM_TOKEN})\$?",
]

FF_MED_CONF = [
    rf"i\s+(?:get|got|obtain|find|conclude)\s+\$?\\?({NUM_TOKEN})\$?",
    rf"(?:most\s+likely|probably|likely)\s+\$?\\?({NUM_TOKEN})\$?",
    rf"my\s+best\s+(?:guess|estimate)\s+is\s+\$?\\?({NUM_TOKEN})\$?",
    rf"value\s+is\s+\$?\\?({NUM_TOKEN})\$?",
]


def extract_freeform_value(response: str) -> tuple[str, str]:
    """Return (value, source) or ('', '') if nothing found.

    Looks in the LAST 1500 chars.
    """
    tail = response[-1500:] if len(response) > 1500 else response

    for pat in FF_HIGH_CONF:
        matches = list(re.finditer(pat, tail, re.IGNORECASE))
        # Iterate from LAST match backward — first valid wins (latest commitment)
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

    # Fallback: last number in the last 500 chars
    last_500 = tail[-500:]
    nums = re.findall(r"-?\d+(?:\.\d+)?(?:/\d+)?", last_500)
    if nums:
        return nums[-1], "last_number"

    return "", ""


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    print(f"REPO_ROOT       : {REPO_ROOT}")
    print(f"CHECKPOINT_PATH : {CHECKPOINT_PATH}")
    print(f"OUTPUT_JSONL    : {OUTPUT_JSONL}")
    print(f"SUBMISSION_CSV  : {SUBMISSION_CSV}")
    print()

    # Load private to know question type
    private_lookup = {}
    with open(PRIVATE_PATH, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            private_lookup[str(item["id"])] = {
                "options":   item.get("options"),
                "is_mcq":    bool(item.get("options")),
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

    # Process each record
    stats = Counter()
    source_breakdown = Counter()
    examples = {"high": [], "med": [], "last_standalone": [], "last_number": [], "no_recovery": []}

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
            new_response = response  # untouched
            recovery_source = "had_boxed"
            recovered_value = existing
        else:
            # Need recovery
            stats["needs_recovery"] += 1
            if info["is_mcq"]:
                value, source = extract_mcq_letter(response)
            else:
                # Free-form: how many [ANS] placeholders?
                n_ans = info["question"].count("[ANS]") or 1
                if n_ans > 1:
                    # Multi-answer free-form is harder. Try to find n_ans numbers.
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
                # Append \boxed{...} at the end of the original response
                new_response = response.rstrip() + f"\n\n\\boxed{{{value}}}"
                recovery_source = source
                recovered_value = value
                stats[f"recovered_{source}"] += 1
                source_breakdown[source] += 1
                # Save first 3 examples per category
                if len(examples[source]) < 3 if source in examples else False:
                    examples[source].append((item_id, info["is_mcq"], value, response[-300:]))
            else:
                new_response = response  # leave as-is
                recovery_source = "no_recovery"
                recovered_value = ""
                stats["no_recovery_possible"] += 1
                if len(examples["no_recovery"]) < 3:
                    examples["no_recovery"].append((item_id, info["is_mcq"], "", response[-300:]))

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
    print("=== RECOVERY SOURCE BREAKDOWN (only for recovered ones) ===")
    for k in sorted(source_breakdown.keys()):
        print(f"  {k}: {source_breakdown[k]}")
    print()

    # Show examples (strip non-ASCII from snippets so Windows console doesn't choke)
    def ascii_safe(s: str) -> str:
        return s.encode("ascii", errors="replace").decode("ascii")

    for cat in ("high", "med", "last_standalone", "last_standalone_late", "last_number", "last_numbers_multi", "no_recovery"):
        ex = examples.get(cat, [])
        if ex:
            print(f"=== EXAMPLES: {cat} ===")
            for item_id, is_mcq, val, snippet in ex[:2]:
                kind = "MCQ" if is_mcq else "FF"
                print(f"  id={item_id} ({kind}) -> recovered='{ascii_safe(val)}'")
                snippet_ascii = ascii_safe(snippet.strip())[-200:]
                print(f"    last 200 chars: ...{snippet_ascii}")
            print()

    # Write the new JSONL
    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    # Sort by id for the JSONL
    updated_sorted = sorted(updated, key=lambda r: int(r["id"]))
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for rec in updated_sorted:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Wrote JSONL: {OUTPUT_JSONL}")

    # Write the CSV (id, response, sorted by id, QUOTE_ALL)
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

    # Final coverage check
    print()
    print("=== POST-RECOVERY COVERAGE ===")
    final_has_boxed = 0
    final_no_boxed = 0
    final_mcq_has = 0
    final_mcq_no = 0
    final_ff_has = 0
    final_ff_no = 0
    for rec in updated:
        has = bool(extract_boxed(rec["response"]))
        if rec["is_mcq"]:
            if has:
                final_mcq_has += 1
            else:
                final_mcq_no += 1
        else:
            if has:
                final_ff_has += 1
            else:
                final_ff_no += 1
        if has:
            final_has_boxed += 1
        else:
            final_no_boxed += 1
    n = len(updated)
    print(f"Total has \\boxed{{}}: {final_has_boxed}/{n} ({100*final_has_boxed/n:.1f}%)")
    print(f"Total NO  \\boxed{{}}: {final_no_boxed}/{n} ({100*final_no_boxed/n:.1f}%)")
    print(f"  MCQ:       has={final_mcq_has}  no={final_mcq_no}  ({100*final_mcq_no/(final_mcq_has+final_mcq_no):.1f}% empty)")
    print(f"  Free-form: has={final_ff_has}  no={final_ff_no}  ({100*final_ff_no/(final_ff_has+final_ff_no):.1f}% empty)")


if __name__ == "__main__":
    main()
