#!/usr/bin/env python3
"""Convert the private inference checkpoint JSONL to the competition submission CSV.

Mirrors the logic of cell `1db185c9` (Section 7) of
`notebooks/06_private_submission.ipynb` so you can run it standalone without
spinning up the full notebook + GPU.

Reads
    artifacts/logs/runs/private_qlora_v1_merged_checkpoint.jsonl
Writes
    artifacts/logs/runs/private_qlora_v1_merged_results.jsonl  (metadata-rich)
    artifacts/submissions/submission_qlora_v1_merged_<YYYY-MM-DD>.csv
        Two columns: id,response   QUOTE_ALL   sorted by id ascending
"""

from __future__ import annotations

import csv
import json
from datetime import date
from pathlib import Path


def repo_root() -> Path:
    """Walk upwards to find the dir containing judger.py (the project root)."""
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
OUTPUT_PATH     = REPO_ROOT / "artifacts" / "logs" / "runs" / f"{RUN_NAME}_results.jsonl"
SUBMISSION_PATH = (
    REPO_ROOT / "artifacts" / "submissions"
    / f"submission_{MODEL_LABEL}_{date.today().isoformat()}.csv"
)

print(f"REPO_ROOT       : {REPO_ROOT}")
print(f"PRIVATE_PATH    : {PRIVATE_PATH}  | exists: {PRIVATE_PATH.is_file()}")
print(f"CHECKPOINT_PATH : {CHECKPOINT_PATH}  | exists: {CHECKPOINT_PATH.is_file()}")
print(f"OUTPUT_PATH     : {OUTPUT_PATH}")
print(f"SUBMISSION_PATH : {SUBMISSION_PATH}")

# ── Load private dataset (defines the canonical question list) ────────────────
with open(PRIVATE_PATH, encoding="utf-8") as f:
    data_run = [json.loads(line) for line in f]
print(f"Loaded {len(data_run)} private questions")

# ── Load checkpoint records (id → dict with response, phase_used, etc.) ───────
response_records: dict[str, dict] = {}
with open(CHECKPOINT_PATH, encoding="utf-8") as f:
    for line in f:
        rec = json.loads(line)
        response_records[str(rec["id"])] = rec
print(f"Loaded {len(response_records)} checkpoint records")

# ── Sanity check: every question must have a response ────────────────────────
missing_ids = [item["id"] for item in data_run if str(item["id"]) not in response_records]
if missing_ids:
    raise RuntimeError(
        f"Missing responses for {len(missing_ids)} id(s). First few: {missing_ids[:10]}"
    )

# ── Write the metadata-rich JSONL (mirrors the notebook Section 7 logic) ─────
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    for item in data_run:
        meta = response_records[str(item["id"])]
        f.write(json.dumps({
            "id":              item["id"],
            "is_mcq":          bool(item.get("options")),
            "response":        meta["response"],
            "phase_used":      meta.get("phase_used"),
            "uncertain":       meta.get("uncertain"),
            "finish_reason":   meta.get("finish_reason"),
            "consensus_count": meta.get("consensus_count"),
            "n_samples":       meta.get("n_samples"),
            "model":           MODEL_LABEL,
        }, ensure_ascii=False) + "\n")
print(f"Wrote results JSONL: {OUTPUT_PATH}")

# ── Write the competition CSV (id, response — QUOTE_ALL, sorted by id) ───────
SUBMISSION_PATH.parent.mkdir(parents=True, exist_ok=True)

rows = sorted(
    [
        {"id": item["id"], "response": response_records[str(item["id"])]["response"]}
        for item in data_run
    ],
    key=lambda r: r["id"],
)

with open(SUBMISSION_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["id", "response"], quoting=csv.QUOTE_ALL)
    writer.writeheader()
    writer.writerows(rows)

# ── Summary stats so you can sanity-check before uploading ───────────────────
from collections import Counter

phase_counts    = Counter(response_records[str(item["id"])].get("phase_used") for item in data_run)
uncertain_count = sum(bool(response_records[str(item["id"])].get("uncertain")) for item in data_run)
finish_counts   = Counter(str(response_records[str(item["id"])].get("finish_reason")) for item in data_run)

print()
print(f"CSV  : {SUBMISSION_PATH}  ({len(rows)} rows)")
print(f"JSONL: {OUTPUT_PATH}")
print()
print("Summary:")
print(f"  phase1   : {phase_counts.get(1, 0)}")
print(f"  phase2   : {phase_counts.get(2, 0)}")
print(f"  phase3   : {phase_counts.get(3, 0)}")
print(f"  uncertain: {uncertain_count}")
print(f"  finish_reasons: {dict(finish_counts)}")
