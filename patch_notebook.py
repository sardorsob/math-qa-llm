"""
Patch 02_inference.ipynb:
  - Cell [7]  config    : slash tokens, remove Phase 3, add Phase 2 hyperparams
  - Cell [13] helpers   : add repetition_penalty to make_sampling_params,
                          add suffix param to build_chat_prompt
  - Cell [14] model     : reduce max_num_seqs / max_num_batched_tokens
  - Cell [19] generation: remove Phase 3, fix Phase 2 batching bug,
                          one progress bar per phase (not per question)

Fix applied vs previous version: outer strings now use r\'\'\'...\'\'\'
so inner triple-double-quote docstrings don't prematurely terminate them.
"""
import json
from pathlib import Path


NB_PATH = Path(r"C:/Users/sardo/OneDrive/Desktop/Classes/projects/math-qa-llm/notebooks/02_inference.ipynb")


def to_source(code: str) -> list:
    """Convert a plain multi-line string into notebook source list format."""
    lines = code.split("\n")
    out = []
    for i, line in enumerate(lines):
        out.append(line + "\n" if i < len(lines) - 1 else line)
    # Drop a trailing empty entry that results from a code string ending with \n
    if out and out[-1] == "":
        out.pop()
    return out


# ── New cell sources ──────────────────────────────────────────────────────────
# NOTE: outer delimiters are r\'\'\'...\'\'\'  so that inner  """docstrings"""
#       do NOT terminate the outer string.

CELL_7 = r'''import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Optional


def repo_root() -> Path:
    """Directory that contains `judger.py` (works from repo root or `notebooks/`)."""
    p = Path.cwd().resolve()
    for d in (p, *p.parents):
        if (d / "judger.py").is_file():
            return d
    return p


REPO_ROOT = repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ── Configuration ─────────────────────────────────────────────────────────────
# Competition rule: final inference must use this exact model, no alternatives.
MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"
GPU_ID   = "0"

PUBLIC_PATH  = REPO_ROOT / "data" / "raw" / "public.jsonl"
PRIVATE_PATH = REPO_ROOT / "data" / "raw" / "private.jsonl"

# "public"  → runs scoring after generation (requires answer labels in the file).
# "private" → skips scoring; use to generate the leaderboard submission CSV.
DATA_MODE = "public"
DATA_PATH = PUBLIC_PATH if DATA_MODE == "public" else PRIVATE_PATH

# None = all questions in the file.
# Set to 50 to run a fast accuracy check before committing to a full overnight run.
N_QUESTIONS        = 50
TEST_RANDOM_SUBSET = True
RANDOM_SEED        = 42

# ── Run naming / artifact paths ────────────────────────────────────────────────
RUN_NAME        = f"adaptive_{DATA_MODE}_v2"
OUTPUT_PATH     = REPO_ROOT / "artifacts" / "logs" / "runs" / f"{RUN_NAME}_results.jsonl"
CHECKPOINT_PATH = REPO_ROOT / "artifacts" / "logs" / "runs" / f"{RUN_NAME}_checkpoint.jsonl"

# ── Adaptive 2-phase generation config ────────────────────────────────────────
# Phase 1 — fast pass over ALL questions, single sample each.
#   thinking_budget : soft token budget for the <think> block (Qwen3 native feature).
#                     Falls back to a plain-text hint if the tokenizer does not support it.
#   max_tokens      : hard cap on TOTAL output tokens (thinking + answer combined).
PHASE1_THINKING_BUDGET = 512   # ~6x faster than unbounded; expect ~8 sec / question
PHASE1_MAX_TOKENS      = 1024  # thinking ~700 + answer ~300
PHASE1_N_SAMPLES       = 1     # single sample — speed is the priority here

# Phase 2 — single batched retry of uncertain questions only.
#   All uncertain prompts are submitted as ONE llm.generate() call (not a per-question loop).
PHASE2_THINKING_BUDGET    = 1024  # more reasoning room for hard questions
PHASE2_MAX_TOKENS         = 2048  # thinking ~1400 + answer ~600
PHASE2_N_SAMPLES          = 2     # 2 samples -> majority vote
PHASE2_TEMPERATURE        = 0.65  # slightly higher than Phase 1 -> more diverse retry paths
PHASE2_REPETITION_PENALTY = 1.05  # discourages repeating the same wrong reasoning path

# Phase 3 has been removed — it was the main cause of the 5000-minute timeout.
# Hyperparameter tuning on Phase 2 (temperature + repetition_penalty) compensates.

# MAX_TOKENS is referenced by the scoring and save cells — keep in sync with Phase 2.
MAX_TOKENS = PHASE2_MAX_TOKENS

os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID

from transformers import AutoTokenizer

try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    LLM            = None  # type: ignore[misc, assignment]
    SamplingParams = None  # type: ignore[misc, assignment]
    VLLM_AVAILABLE = False

from tqdm.auto import tqdm

USE_VLLM = True

print("REPO_ROOT   :", REPO_ROOT)
print("DATA_MODE   :", DATA_MODE)
print("DATA_PATH   :", DATA_PATH, "| exists:", DATA_PATH.is_file())
print("N_QUESTIONS :", N_QUESTIONS)
print("RUN_NAME    :", RUN_NAME)
print("CHECKPOINT  :", CHECKPOINT_PATH)
print("vLLM        :", VLLM_AVAILABLE)'''

# ─────────────────────────────────────────────────────────────────────────────

CELL_13 = r'''# ── Adaptive generation helpers ────────────────────────────────────────────────
from collections import Counter
from math import ceil


def extract_boxed(text: str) -> str:
    """Extract last \\boxed{...} content from a response (nested braces supported)."""
    matches = []
    needle  = r"\boxed{"
    i = 0
    while i < len(text):
        idx = text.find(needle, i)
        if idx == -1:
            break
        j     = idx + len(needle)
        depth = 1
        start = j
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


def majority_vote(answers: list) -> str:
    """Return the most common answer string from a list."""
    if not answers:
        return ""
    return Counter(answers).most_common(1)[0][0]


def strip_thinking(text: str) -> str:
    """Remove explicit <think>...</think> blocks when present."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def is_uncertain(response: str, finish_reason: str = "") -> bool:
    """Return True if this response should be retried with more compute."""
    if "length" in str(finish_reason).lower():
        return True   # truncated — never reached \boxed{}
    if not extract_boxed(response):
        return True   # no answer found
    answer_only = strip_thinking(response)
    if len(answer_only) < 30:
        return True   # vacuous answer section
    return False


def choose_best_sample(samples: list, finish_reasons: list) -> dict:
    """Choose the sample whose boxed answer wins majority; tie-break by longer trace."""
    extracted = [extract_boxed(s) for s in samples]
    nonempty  = [e for e in extracted if e]

    if nonempty:
        counts       = Counter(nonempty)
        top_count    = counts.most_common(1)[0][1]
        tied_answers = {ans for ans, cnt in counts.items() if cnt == top_count}
        candidates   = [i for i, ans in enumerate(extracted) if ans in tied_answers]
        best_idx     = max(candidates, key=lambda i: len(samples[i]))
        best_answer  = extracted[best_idx]
    else:
        top_count   = 0
        best_idx    = 0
        best_answer = ""

    threshold = ceil(len(samples) / 2)
    uncertain = (
        is_uncertain(samples[best_idx], finish_reasons[best_idx])
        or top_count < threshold
    )
    return {
        "response":        samples[best_idx],
        "answer":          best_answer,
        "finish_reason":   finish_reasons[best_idx],
        "consensus_count": top_count,
        "n_samples":       len(samples),
        "uncertain":       uncertain,
    }


def make_sampling_params(max_tokens: int, temperature: float = 0.6,
                          repetition_penalty: float = 1.0):
    """Create vLLM SamplingParams.

    repetition_penalty > 1.0 discourages the model from repeating the same
    wrong reasoning path — useful for Phase 2 retries.
    """
    return SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
        presence_penalty=0.0,
        repetition_penalty=repetition_penalty,
    )


def build_chat_prompt(item: dict, thinking_budget=None,
                       prefix: str = "", suffix: str = "") -> str:
    """Render a chat prompt string ready for llm.generate().

    prefix  — prepended to the user message (e.g. a retry instruction).
    suffix  — appended  to the user message (e.g. an MCQ verification reminder).
    Falls back to a plain-text budget hint when the tokenizer does not support
    the thinking_budget kwarg.
    """
    system, user = build_prompt(item["question"], item.get("options"))
    if prefix:
        user = prefix + user
    if suffix:
        user = user + suffix

    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]
    kwargs = dict(tokenize=False, add_generation_prompt=True, enable_thinking=True)
    if thinking_budget is not None:
        kwargs["thinking_budget"] = thinking_budget

    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError as exc:
        if thinking_budget is None:
            raise
        # Tokenizer does not support thinking_budget kwarg — use a prompt hint instead.
        hint = (
            f"Use at most about {thinking_budget} thinking tokens. "
            "Be concise but do not skip necessary arithmetic.\n\n"
        )
        messages[1]["content"] = hint + messages[1]["content"]
        kwargs.pop("thinking_budget", None)
        print(f"[thinking_budget not supported by tokenizer; using prompt-hint fallback ({exc})]")
        return tokenizer.apply_chat_template(messages, **kwargs)


def load_checkpoint(path) -> dict:
    """Load an existing checkpoint JSONL; returns {} if the file does not exist."""
    if not path.exists():
        return {}
    records = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            records[str(rec["id"])] = rec
    print(f"Loaded checkpoint: {len(records)} record(s) from {path}")
    return records


def write_checkpoint(path, records: dict) -> None:
    """Write all response records to a checkpoint JSONL (sorted by id)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in sorted(records.values(), key=lambda r: int(r["id"])):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")'''

# ─────────────────────────────────────────────────────────────────────────────

CELL_14 = r'''tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

llm = LLM(
    model=MODEL_ID,
    quantization="bitsandbytes",
    load_format="bitsandbytes",
    gpu_memory_utilization=0.78,   # RTX 3070 8 GB — leaves ~1.7 GB for WSL2 overhead
    max_model_len=8192,            # generous; long questions need prompt + answer room
    trust_remote_code=True,
    max_num_seqs=4,                # reduced from 8 — matches PHASE2_N_SAMPLES; saves VRAM
    max_num_batched_tokens=4096,   # reduced from 8192 — matches new max_tokens ceiling
)

sampling_params = make_sampling_params(MAX_TOKENS, temperature=0.6)

print("Model loaded.")'''

# ─────────────────────────────────────────────────────────────────────────────

CELL_19 = r'''# ── 2-Phase Adaptive Generation ───────────────────────────────────────────────
# Phase 1 : one llm.generate() call for ALL questions  -> one vLLM bar total.
# Phase 2 : one llm.generate() call for ALL uncertain  -> one vLLM bar total.
# Checkpoints written after each complete phase (not per question).
# Phase 3 removed — was the primary cause of the 5000-minute timeout.

response_records = load_checkpoint(CHECKPOINT_PATH)

# ── PHASE 1 ───────────────────────────────────────────────────────────────────
phase1_params  = make_sampling_params(PHASE1_MAX_TOKENS, temperature=0.6)
missing_phase1 = [item for item in data_run if str(item["id"]) not in response_records]

print(f"\n{'='*60}")
print(f"PHASE 1  |  {len(missing_phase1)} to generate  "
      f"({len(data_run) - len(missing_phase1)} already checkpointed)")
print(f"{'='*60}")

if missing_phase1:
    phase1_prompts = [
        build_chat_prompt(item, thinking_budget=PHASE1_THINKING_BUDGET)
        for item in missing_phase1
    ]
    # Single batched call — vLLM shows ONE progress bar for the entire phase.
    phase1_outputs = llm.generate(phase1_prompts, phase1_params)

    for item, out in zip(missing_phase1, phase1_outputs):
        response      = out.outputs[0].text.strip()
        finish_reason = str(out.outputs[0].finish_reason)
        response_records[str(item["id"])] = {
            "id":              item["id"],
            "phase_used":      1,
            "response":        response,
            "answer":          extract_boxed(response),
            "finish_reason":   finish_reason,
            "uncertain":       is_uncertain(response, finish_reason),
            "n_samples":       1,
            "consensus_count": 1,
        }

    write_checkpoint(CHECKPOINT_PATH, response_records)

_p1_uncertain = sum(
    1 for item in data_run
    if response_records.get(str(item["id"]), {}).get("uncertain")
)
print(f"\nPhase 1 done.  {_p1_uncertain} / {len(data_run)} uncertain  ->  entering Phase 2.")

# ── PHASE 2 ───────────────────────────────────────────────────────────────────
phase2_params = make_sampling_params(
    PHASE2_MAX_TOKENS,
    temperature=PHASE2_TEMPERATURE,
    repetition_penalty=PHASE2_REPETITION_PENALTY,
)

phase1_uncertain = [
    item for item in data_run
    if response_records.get(str(item["id"]), {}).get("uncertain")
    and int(response_records.get(str(item["id"]), {}).get("phase_used", 0)) < 2
]

RETRY_PREFIX = "Previous attempt was unclear. Solve this again carefully from scratch:\n\n"
MCQ_VERIFY_SUFFIX = (
    "\n\nAfter finding your answer, check each option against the problem conditions. "
    "Eliminate any letter that clearly fails. "
    "Then on the very last line write ONLY \\boxed{X}."
)

total_phase2_prompts = len(phase1_uncertain) * PHASE2_N_SAMPLES
print(f"\n{'='*60}")
print(f"PHASE 2  |  {len(phase1_uncertain)} uncertain  x  {PHASE2_N_SAMPLES} samples  "
      f"=  {total_phase2_prompts} prompts total")
print(f"{'='*60}")

if phase1_uncertain:
    # Build a flat list: PHASE2_N_SAMPLES copies per question, all in one list.
    # This lets vLLM process everything in a single batched call — one progress bar.
    phase2_prompts = []
    for item in phase1_uncertain:
        suffix      = MCQ_VERIFY_SUFFIX if item.get("options") else ""
        prompt_text = build_chat_prompt(
            item,
            thinking_budget=PHASE2_THINKING_BUDGET,
            prefix=RETRY_PREFIX,
            suffix=suffix,
        )
        for _ in range(PHASE2_N_SAMPLES):
            phase2_prompts.append(prompt_text)

    # Single batched call — vLLM shows ONE progress bar for the entire phase.
    phase2_outputs_flat = llm.generate(phase2_prompts, phase2_params)

    # Reconstruct per-question sample groups and pick best via majority vote.
    for q_idx, item in enumerate(phase1_uncertain):
        start          = q_idx * PHASE2_N_SAMPLES
        end            = start + PHASE2_N_SAMPLES
        samples        = [phase2_outputs_flat[j].outputs[0].text.strip()       for j in range(start, end)]
        finish_reasons = [str(phase2_outputs_flat[j].outputs[0].finish_reason) for j in range(start, end)]
        chosen         = choose_best_sample(samples, finish_reasons)
        chosen.update({"id": item["id"], "phase_used": 2})
        response_records[str(item["id"])] = chosen

    write_checkpoint(CHECKPOINT_PATH, response_records)

_p2_recovered = sum(
    1 for item in phase1_uncertain
    if not response_records.get(str(item["id"]), {}).get("uncertain")
)
print(f"\nPhase 2 done.  {_p2_recovered} / {len(phase1_uncertain)} uncertain questions recovered.")

# ── ASSEMBLE FINAL RESPONSES ──────────────────────────────────────────────────
missing_ids = [item["id"] for item in data_run if str(item["id"]) not in response_records]
if missing_ids:
    raise RuntimeError(
        f"Missing responses for {len(missing_ids)} id(s): {missing_ids[:10]}..."
    )

responses = [response_records[str(item["id"])]["response"] for item in data_run]

# ── Summary ───────────────────────────────────────────────────────────────────
phase_counts    = Counter(response_records[str(item["id"])].get("phase_used") for item in data_run)
uncertain_count = sum(bool(response_records[str(item["id"])].get("uncertain"))  for item in data_run)
finish_counts   = Counter(str(response_records[str(item["id"])].get("finish_reason")) for item in data_run)
length_hits     = (finish_counts.get("RequestStatus.FINISH_REASON_LENGTH", 0)
                   + finish_counts.get("length", 0))

print(f"\n{'='*60}")
print("GENERATION COMPLETE")
print(f"{'='*60}")
print(f"  Questions          : {len(data_run)}")
print(f"  Responses          : {len(responses)}")
print(f"  Phase 1 only       : {phase_counts.get(1, 0)}")
print(f"  Recovered Phase 2  : {phase_counts.get(2, 0)}")
print(f"  Still uncertain    : {uncertain_count}  (response saved, low confidence)")
print(f"  Truncated (length) : {length_hits}  (consider raising PHASE2_MAX_TOKENS if high)")
print(f"  Checkpoint saved   : {CHECKPOINT_PATH}")

print("\nSample outputs (first 3):")
for i in range(min(3, len(responses))):
    rec = response_records[str(data_run[i]["id"])]
    ans = rec.get("answer", "") or "(no boxed answer)"
    print(f"\n  [{i}] id={data_run[i].get('id')}  phase={rec.get('phase_used')}  "
          f"answer={ans[:80]}")
    print(f"       {strip_thinking(responses[i])[:200]}")'''


# ── Patch notebook ─────────────────────────────────────────────────────────────

TARGET = {
    "b9a459bf": CELL_7,
    "ad824e4c": CELL_13,
    "4b1492d0": CELL_14,
    "4915f406": CELL_19,
}

with open(NB_PATH, encoding="utf-8") as f:
    nb = json.load(f)

patched = []
for i, cell in enumerate(nb["cells"]):
    cid = cell.get("id", "")
    if cid in TARGET:
        nb["cells"][i] = dict(cell)
        nb["cells"][i]["source"] = to_source(TARGET[cid])
        nb["cells"][i]["outputs"] = []
        nb["cells"][i]["execution_count"] = None
        patched.append(cid)

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"Patched {len(patched)} cell(s): {patched}")
print(f"Notebook saved: {NB_PATH}")
if len(patched) < len(TARGET):
    missing = set(TARGET) - set(patched)
    print(f"WARNING: these cell IDs were NOT found in the notebook: {missing}")
