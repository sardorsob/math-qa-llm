"""
Restore Phase 1 / Phase 2 token budgets to original values in 02_inference.ipynb.

Why this is now safe:
  - Phase 3 is removed (was 70%+ of original runtime)
  - Phase 2 batching bug is fixed (was 5-10x slower due to per-question generate() calls)
  - With those two fixes, the original token counts are affordable again.

Changes vs current patched values:
  PHASE1_THINKING_BUDGET  512  -> 1024
  PHASE1_MAX_TOKENS      1024  -> 2048
  PHASE2_THINKING_BUDGET 1024  -> 4096
  PHASE2_MAX_TOKENS      2048  -> 6144
  MAX_TOKENS             2048  -> 6144  (kept in sync with Phase 2)
"""
import json
from pathlib import Path

NB_PATH = Path(r"C:/Users/sardo/OneDrive/Desktop/Classes/projects/math-qa-llm/notebooks/02_inference.ipynb")

with open(NB_PATH, encoding="utf-8") as f:
    nb = json.load(f)

TARGET_ID = "b9a459bf"

for i, cell in enumerate(nb["cells"]):
    if cell.get("id") != TARGET_ID:
        continue

    src = "".join(cell["source"])

    # Phase 1
    src = src.replace(
        "PHASE1_THINKING_BUDGET = 512   # ~6x faster than unbounded; expect ~8 sec / question",
        "PHASE1_THINKING_BUDGET = 1024  # soft limit on <think> block tokens",
    )
    src = src.replace(
        "PHASE1_MAX_TOKENS      = 1024  # thinking ~700 + answer ~300",
        "PHASE1_MAX_TOKENS      = 2048  # thinking ~1500 + answer ~500",
    )

    # Phase 2
    src = src.replace(
        "PHASE2_THINKING_BUDGET    = 1024  # more reasoning room for hard questions",
        "PHASE2_THINKING_BUDGET    = 4096  # more reasoning room for hard questions",
    )
    src = src.replace(
        "PHASE2_MAX_TOKENS         = 2048  # thinking ~1400 + answer ~600",
        "PHASE2_MAX_TOKENS         = 6144  # thinking ~5500 + answer ~600",
    )

    # MAX_TOKENS sentinel (kept in sync with Phase 2)
    src = src.replace(
        "MAX_TOKENS = PHASE2_MAX_TOKENS",
        "MAX_TOKENS = PHASE2_MAX_TOKENS  # 6144 — in sync with Phase 2",
    )

    # Convert back to source list format
    lines = src.split("\n")
    new_source = [l + "\n" for l in lines[:-1]] + [lines[-1]]
    if new_source and new_source[-1] == "":
        new_source.pop()

    nb["cells"][i]["source"] = new_source
    nb["cells"][i]["outputs"] = []
    nb["cells"][i]["execution_count"] = None
    print(f"Patched cell {TARGET_ID}")
    break
else:
    print(f"ERROR: cell {TARGET_ID} not found")

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print(f"Saved: {NB_PATH}")
