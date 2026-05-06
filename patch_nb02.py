"""
Patch notebooks/02_inference.ipynb:
- N_QUESTIONS = None  (full public run)
- Clean generation cell (remove banner prints)
- Clean helpers cell (remove verbose docstrings / divider comments)
- Clean prompts cell (remove duplicate comment, remove verify prints)
- Clean summary cell (remove banner prints)
- Clean data cell (remove MCQ/free-form preview prints)
"""
import json
from pathlib import Path


def repo_root() -> Path:
    p = Path(__file__).resolve().parent
    if (p / "judger.py").is_file():
        return p
    for d in (Path.cwd(), *Path.cwd().parents):
        if (d / "judger.py").is_file():
            return d
    return p


NB_PATH = repo_root() / "notebooks" / "02_inference.ipynb"

with open(NB_PATH, encoding="utf-8") as f:
    nb = json.load(f)


def set_source(cell: dict, new_code: str) -> None:
    new_code = new_code.lstrip("\n")
    lines = new_code.split("\n")
    source = [line + "\n" for line in lines[:-1]] + [lines[-1]]
    if source and source[-1] == "":
        source.pop()
    cell["source"] = source
    cell["outputs"] = []
    cell["execution_count"] = None


# ─── Locate cells by id ───────────────────────────────────────────────────────
cells_by_id = {c.get("id"): c for c in nb["cells"]}

# ─── Cell b9a459bf — Config ───────────────────────────────────────────────────
CONFIG_CODE = r'''import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Optional


def repo_root() -> Path:
    p = Path.cwd().resolve()
    for d in (p, *p.parents):
        if (d / "judger.py").is_file():
            return d
    return p


REPO_ROOT = repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"
GPU_ID   = "0"

PUBLIC_PATH  = REPO_ROOT / "data" / "raw" / "public.jsonl"
PRIVATE_PATH = REPO_ROOT / "data" / "raw" / "private.jsonl"

DATA_MODE = "public"
DATA_PATH = PUBLIC_PATH if DATA_MODE == "public" else PRIVATE_PATH

N_QUESTIONS        = None   # None = all questions; set to 50 for a quick smoke test
TEST_RANDOM_SUBSET = True
RANDOM_SEED        = 42

RUN_NAME        = f"adaptive_{DATA_MODE}_v2"
OUTPUT_PATH     = REPO_ROOT / "artifacts" / "logs" / "runs" / f"{RUN_NAME}_results.jsonl"
CHECKPOINT_PATH = REPO_ROOT / "artifacts" / "logs" / "runs" / f"{RUN_NAME}_checkpoint.jsonl"

PHASE1_THINKING_BUDGET    = 1024
PHASE1_MAX_TOKENS         = 2048
PHASE1_N_SAMPLES          = 1

PHASE2_THINKING_BUDGET    = 4096
PHASE2_MAX_TOKENS         = 6144
PHASE2_N_SAMPLES          = 2
PHASE2_TEMPERATURE        = 0.65
PHASE2_REPETITION_PENALTY = 1.05

MAX_TOKENS = PHASE2_MAX_TOKENS

os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID

from transformers import AutoTokenizer

try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    LLM            = None
    SamplingParams = None
    VLLM_AVAILABLE = False

from tqdm.auto import tqdm

USE_VLLM = True

print("REPO_ROOT  :", REPO_ROOT)
print("DATA_MODE  :", DATA_MODE)
print("DATA_PATH  :", DATA_PATH, "| exists:", DATA_PATH.is_file())
print("N_QUESTIONS:", N_QUESTIONS)
print("RUN_NAME   :", RUN_NAME)
print("vLLM       :", VLLM_AVAILABLE)
'''

# ─── Cell 23ee5092 — Data loading ────────────────────────────────────────────
DATA_CODE = r'''with open(DATA_PATH, encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

n_mcq  = sum(bool(d.get("options")) for d in data)
n_free = len(data) - n_mcq
print(f"Loaded {len(data)} questions  ({n_mcq} MCQ, {n_free} free-form)")

if N_QUESTIONS is None:
    data_run = data
else:
    k = min(int(N_QUESTIONS), len(data))
    if TEST_RANDOM_SUBSET:
        rng      = random.Random(RANDOM_SEED)
        data_run = rng.sample(data, k=k)
    else:
        data_run = data[:k]

_nrun_mcq = sum(bool(d.get("options")) for d in data_run)
print(f"Running on {len(data_run)} questions  ({_nrun_mcq} MCQ, {len(data_run)-_nrun_mcq} free-form)")
'''

# ─── Cell 4e5169ac — Prompts ─────────────────────────────────────────────────
PROMPTS_CODE = r'''SYSTEM_PROMPT_MATH = (
    "You are an expert mathematician with deep knowledge of all areas of mathematics, "
    "from algebra and calculus to number theory and combinatorics. "
    "This problem is very important to my career - please think carefully and be precise.\n\n"
    "Solve using this structured approach:\n"
    "1. UNDERSTAND: Identify what is given and what you need to find.\n"
    "2. PLAN: Write down the key equations, formulas, or theorems you will use.\n"
    "3. SOLVE: Work through each step carefully. Compute intermediate results explicitly. "
    "Pay special attention to arithmetic - do not skip steps.\n"
    "4. VERIFY: Check that your answer satisfies all conditions in the problem. "
    "Check units, sign, and order of magnitude.\n"
    "5. ANSWER: Put your final answer in \\boxed{}.\n\n"
    "Additional rules:\n"
    "- If the problem has multiple blanks ([ANS] placeholders), put ALL answers "
    "comma-separated in ONE \\boxed{} in the order they appear. "
    "Example: \\boxed{3, -7, 42}.\n"
    "- Simplify all fractions and radical expressions completely.\n"
    "- You'd better be sure of your answer."
)

SYSTEM_PROMPT_MCQ = (
    "You are an expert mathematician with deep knowledge of all areas of mathematics. "
    "This problem is very important to my career - please think carefully and be precise.\n\n"
    "Solve using this structured approach:\n"
    "1. UNDERSTAND: Read the problem and all answer choices carefully.\n"
    "2. PLAN: Identify the relevant concepts, formulas, or theorems that apply.\n"
    "3. SOLVE: Work through the problem step by step. Compute intermediate results "
    "explicitly - do not skip arithmetic steps.\n"
    "4. ELIMINATE: Cross out answer choices that are clearly wrong.\n"
    "5. VERIFY: Confirm your chosen answer is consistent with every condition in the problem.\n"
    "6. ANSWER: On the very last line of your response, write ONLY \\boxed{X} "
    "where X is the letter of the correct answer (A-J). "
    "Do not write any text after \\boxed{}.\n\n"
    "You'd better be sure of your answer."
)


def build_prompt(question: str, options: Optional[list]) -> tuple[str, str]:
    if options:
        labels    = [chr(65 + i) for i in range(len(options))]
        opts_text = "\n".join(f"{lbl}. {opt.strip()}" for lbl, opt in zip(labels, options))
        return SYSTEM_PROMPT_MCQ, f"{question}\n\nOptions:\n{opts_text}"
    n_ans = question.count("[ANS]")
    if n_ans > 1:
        hint = (
            f"\n\n[Note: This problem has {n_ans} answers. "
            f"Put all {n_ans} answers comma-separated in ONE \\boxed{{}} "
            f"in the order they appear in the question.]"
        )
        return SYSTEM_PROMPT_MATH, question + hint
    return SYSTEM_PROMPT_MATH, question
'''

# ─── Cell ad824e4c — Helpers ──────────────────────────────────────────────────
HELPERS_CODE = r'''from collections import Counter
from math import ceil


def extract_boxed(text: str) -> str:
    """Extract last \boxed{...} content (nested braces supported)."""
    matches = []
    needle  = r"\boxed{"
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


def strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def is_uncertain(response: str, finish_reason: str = "") -> bool:
    if "length" in str(finish_reason).lower():
        return True
    if not extract_boxed(response):
        return True
    if len(strip_thinking(response)) < 30:
        return True
    return False


def choose_best_sample(samples: list, finish_reasons: list) -> dict:
    """Majority vote; tie-break by longest trace."""
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
        top_count, best_idx, best_answer = 0, 0, ""
    uncertain = (
        is_uncertain(samples[best_idx], finish_reasons[best_idx])
        or top_count < ceil(len(samples) / 2)
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
    return SamplingParams(
        max_tokens=max_tokens, temperature=temperature,
        top_p=0.95, top_k=20, min_p=0.0,
        presence_penalty=0.0, repetition_penalty=repetition_penalty,
    )


def build_chat_prompt(item: dict, thinking_budget=None,
                       prefix: str = "", suffix: str = "") -> str:
    """Render a prompt string for llm.generate(). Falls back to plain-text budget hint."""
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
        hint = (
            f"Use at most about {thinking_budget} thinking tokens. "
            "Be concise but do not skip necessary arithmetic.\n\n"
        )
        messages[1]["content"] = hint + messages[1]["content"]
        kwargs.pop("thinking_budget", None)
        return tokenizer.apply_chat_template(messages, **kwargs)


def load_checkpoint(path) -> dict:
    if not path.exists():
        return {}
    records = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            records[str(rec["id"])] = rec
    print(f"Checkpoint: {len(records)} records from {path.name}")
    return records


def write_checkpoint(path, records: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in sorted(records.values(), key=lambda r: int(r["id"])):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
'''

# ─── Cell 4915f406 — Generation ──────────────────────────────────────────────
GENERATE_CODE = r'''response_records = load_checkpoint(CHECKPOINT_PATH)

# Phase 1 — single batched call for all questions
phase1_params  = make_sampling_params(PHASE1_MAX_TOKENS, temperature=0.6)
missing_phase1 = [item for item in data_run if str(item["id"]) not in response_records]

if missing_phase1:
    print(f"Phase 1: {len(missing_phase1)} questions "
          f"({len(data_run)-len(missing_phase1)} checkpointed)")
    phase1_prompts = [
        build_chat_prompt(item, thinking_budget=PHASE1_THINKING_BUDGET)
        for item in missing_phase1
    ]
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

p1_uncertain = sum(1 for item in data_run
                   if response_records.get(str(item["id"]), {}).get("uncertain"))
print(f"Phase 1 done: {p1_uncertain}/{len(data_run)} uncertain")

# Phase 2 — single batched call for all uncertain questions
phase2_params    = make_sampling_params(
    PHASE2_MAX_TOKENS, temperature=PHASE2_TEMPERATURE,
    repetition_penalty=PHASE2_REPETITION_PENALTY,
)
phase1_uncertain = [
    item for item in data_run
    if response_records.get(str(item["id"]), {}).get("uncertain")
    and int(response_records.get(str(item["id"]), {}).get("phase_used", 0)) < 2
]

RETRY_PREFIX      = "Previous attempt was unclear. Solve this again carefully from scratch:\n\n"
MCQ_VERIFY_SUFFIX = (
    "\n\nAfter finding your answer, check each option against the problem conditions. "
    "Eliminate any letter that clearly fails. "
    "Then on the very last line write ONLY \\boxed{X}."
)

if phase1_uncertain:
    print(f"Phase 2: {len(phase1_uncertain)} uncertain x {PHASE2_N_SAMPLES} = "
          f"{len(phase1_uncertain)*PHASE2_N_SAMPLES} prompts")
    phase2_prompts = []
    for item in phase1_uncertain:
        suffix      = MCQ_VERIFY_SUFFIX if item.get("options") else ""
        prompt_text = build_chat_prompt(
            item, thinking_budget=PHASE2_THINKING_BUDGET,
            prefix=RETRY_PREFIX, suffix=suffix,
        )
        for _ in range(PHASE2_N_SAMPLES):
            phase2_prompts.append(prompt_text)

    phase2_outputs_flat = llm.generate(phase2_prompts, phase2_params)

    for q_idx, item in enumerate(phase1_uncertain):
        start          = q_idx * PHASE2_N_SAMPLES
        end            = start + PHASE2_N_SAMPLES
        samples        = [phase2_outputs_flat[j].outputs[0].text.strip()       for j in range(start, end)]
        finish_reasons = [str(phase2_outputs_flat[j].outputs[0].finish_reason) for j in range(start, end)]
        chosen         = choose_best_sample(samples, finish_reasons)
        chosen.update({"id": item["id"], "phase_used": 2})
        response_records[str(item["id"])] = chosen

    write_checkpoint(CHECKPOINT_PATH, response_records)

missing_ids = [item["id"] for item in data_run if str(item["id"]) not in response_records]
if missing_ids:
    raise RuntimeError(f"Missing responses for {len(missing_ids)} id(s): {missing_ids[:10]}")

responses = [response_records[str(item["id"])]["response"] for item in data_run]

phase_counts    = Counter(response_records[str(item["id"])].get("phase_used") for item in data_run)
uncertain_count = sum(bool(response_records[str(item["id"])].get("uncertain"))  for item in data_run)
finish_counts   = Counter(str(response_records[str(item["id"])].get("finish_reason")) for item in data_run)
length_hits     = (finish_counts.get("RequestStatus.FINISH_REASON_LENGTH", 0)
                   + finish_counts.get("length", 0))

print(f"Done: {len(responses)} responses | "
      f"phase1={phase_counts.get(1,0)} phase2={phase_counts.get(2,0)} "
      f"uncertain={uncertain_count} truncated={length_hits}")
'''

# ─── Cell 52f6d9ed — Summary ──────────────────────────────────────────────────
SUMMARY_CODE = r'''mcq_res  = [r for r in results if r["is_mcq"]]
free_res = [r for r in results if not r["is_mcq"]]


def acc(subset):
    return sum(r["correct"] for r in subset) / len(subset) * 100 if subset else 0.0


print(f"MCQ       : {sum(r['correct'] for r in mcq_res):4d} / {len(mcq_res):4d}  ({acc(mcq_res):.2f}%)")
print(f"Free-form : {sum(r['correct'] for r in free_res):4d} / {len(free_res):4d}  ({acc(free_res):.2f}%)")
print(f"Overall   : {sum(r['correct'] for r in results):4d} / {len(results):4d}  ({acc(results):.2f}%)")
'''

CSV_CODE = r'''import csv
from datetime import date

SUBMISSION_INPUT  = OUTPUT_PATH
SUBMISSION_OUTPUT = (
    REPO_ROOT / "artifacts" / "submissions"
    / f"submission_{date.today().isoformat()}.csv"
)
SUBMISSION_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

rows = []
with open(SUBMISSION_INPUT, encoding="utf-8") as f:
    for line in f:
        rec = json.loads(line)
        rows.append({"id": rec["id"], "response": rec["response"]})

rows.sort(key=lambda r: r["id"])

with open(SUBMISSION_OUTPUT, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["id", "response"], quoting=csv.QUOTE_ALL)
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote {len(rows)} rows -> {SUBMISSION_OUTPUT}")
'''

patches = {
    "b9a459bf": CONFIG_CODE,
    "23ee5092": DATA_CODE,
    "4e5169ac": PROMPTS_CODE,
    "ad824e4c": HELPERS_CODE,
    "4915f406": GENERATE_CODE,
    "52f6d9ed": SUMMARY_CODE,
    "5d0ec498": CSV_CODE,
}

patched = 0
for cell in nb["cells"]:
    cid = cell.get("id")
    if cid in patches:
        set_source(cell, patches[cid])
        patched += 1
        print(f"Patched cell {cid}")

if patched != len(patches):
    missing = set(patches) - {c.get("id") for c in nb["cells"]}
    print(f"WARNING: {len(missing)} cell(s) not found: {missing}")

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"Saved: {NB_PATH}  ({patched}/{len(patches)} cells patched)")
