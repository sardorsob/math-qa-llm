#!/usr/bin/env bash
# fetch_papers.sh — download every paper listed in README.md into ./pdfs/
#
# The Cowork sandbox can't reach arxiv.org (egress proxy allowlist), but your
# local machine can. Run this script locally:
#
#   cd research/papers
#   bash fetch_papers.sh
#
# Re-runs are safe: curl -C - resumes partial downloads; already-complete
# files are skipped.

set -euo pipefail

OUT_DIR="pdfs"
mkdir -p "$OUT_DIR"

# Tier: arxiv_id : filename
papers=(
  # Tier 1 — inference-time techniques
  "2201.11903:01_cot_prompting.pdf"
  "2203.11171:02_self_consistency.pdf"
  "2408.03314:03_scaling_test_time_compute.pdf"
  "2211.12588:04_program_of_thoughts.pdf"
  "2309.17452:05_tora.pdf"
  "2305.20050:06_lets_verify_step_by_step.pdf"

  # Tier 2 — model / checkpoint papers
  "2501.12948:07_deepseek_r1.pdf"
  "2409.12122:08_qwen2.5_math.pdf"
  "2505.09388:09_qwen3.pdf"
  "2504.21233:10_phi4_mini_reasoning.pdf"
  "2504.21318:11_phi4_reasoning.pdf"

  # Tier 3 — training data
  "2309.12284:12_metamath.pdf"
  "2309.05653:13_mammoth_mathinstruct.pdf"
  "2410.01560:14_openmathinstruct2.pdf"

  # Tier 4 — systems & efficiency
  "2106.09685:15_lora.pdf"
  "2305.14314:16_qlora.pdf"
  "2208.07339:17_llm_int8_bitsandbytes.pdf"
  "2306.00978:18_awq.pdf"
  "2309.06180:19_vllm_pagedattention.pdf"

  # Optional / supporting
  "2305.17306:20_cot_hub.pdf"
  "2305.10601:21_tree_of_thoughts.pdf"
  "2103.03874:22_math_dataset.pdf"
  "2110.14168:23_gsm8k.pdf"
  "2211.14275:24_process_outcome_feedback.pdf"
)

# NuminaMath writeup is hosted on GitHub, not arXiv — fetched separately below.
NUMINA_URL="https://raw.githubusercontent.com/project-numina/aimo-progress-prize/main/report/numina_dataset.pdf"
NUMINA_DEST="$OUT_DIR/12b_numinamath_aimo_prize.pdf"

echo "Downloading ${#papers[@]} arXiv papers + 1 GitHub-hosted paper into $OUT_DIR/ ..."
for entry in "${papers[@]}"; do
  id="${entry%%:*}"
  name="${entry##*:}"
  dest="$OUT_DIR/$name"
  url="https://arxiv.org/pdf/${id}"

  if [[ -s "$dest" ]]; then
    echo "  [skip]   $name (already present)"
    continue
  fi

  echo "  [fetch]  $name  <-  $url"
  # -L follows redirects, -C - resumes, --retry for transient arxiv rate-limiting
  curl -fsSL -C - --retry 3 --retry-delay 2 \
       -A "Mozilla/5.0 (math-qa-llm reading-list fetcher)" \
       -o "$dest" "$url"
done

if [[ -s "$NUMINA_DEST" ]]; then
  echo "  [skip]   $(basename "$NUMINA_DEST") (already present)"
else
  echo "  [fetch]  $(basename "$NUMINA_DEST")  <-  $NUMINA_URL"
  curl -fsSL -C - --retry 3 --retry-delay 2 \
       -A "Mozilla/5.0 (math-qa-llm reading-list fetcher)" \
       -o "$NUMINA_DEST" "$NUMINA_URL" || \
       echo "  [warn]   Numina PDF not fetched (repo may have moved); see README for manual link."
fi

echo ""
echo "Done. Files in $OUT_DIR/:"
ls -lh "$OUT_DIR"
