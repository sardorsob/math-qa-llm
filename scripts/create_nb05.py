"""Generate notebooks/05_private_submission.ipynb."""
import json
import uuid
from pathlib import Path


def repo_root() -> Path:
    p = Path(__file__).resolve().parent.parent
    if (p / "judger.py").is_file():
        return p
    for d in (Path.cwd(), *Path.cwd().parents):
        if (d / "judger.py").is_file():
            return d
    return p


REPO = repo_root()


def cid() -> str:
    return uuid.uuid4().hex[:8]


def src(code: str) -> list:
    code = code.lstrip("\n")
    lines = code.split("\n")
    out = [line + "\n" for line in lines[:-1]] + [lines[-1]]
    if out and out[-1] == "":
        out.pop()
    return out


def code_cell(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "id": cid(),
            "metadata": {}, "outputs": [], "source": src(text)}


def md_cell(text: str) -> dict:
    return {"cell_type": "markdown", "id": cid(), "metadata": {}, "source": src(text)}


def notebook(cells: list) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python (cse151b WSL)",
                           "language": "python", "name": "cse151b-wsl"},
            "language_info": {"name": "python", "version": "3.11.0"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


CELL_INTRO = md_cell("""\
# 05 — Private Set Submission

Run 2-phase adaptive inference on the private test set and produce the submission CSV.

**Pipeline order:**
1. `02_inference.ipynb` — full public inference (`N_QUESTIONS=None`)
2. `03_qlora_finetune.ipynb` — QLoRA SFT → merge adapter
3. `04_grpo_train.ipynb` — GRPO RL → merge adapter
4. **This notebook** → `artifacts/submissions/submission_*.csv`

The notebook auto-selects the best merged model: GRPO > QLoRA > base.
No scoring section — the private set has no ground-truth labels.
""")

CELL_CONFIG = code_cell('''\
import json, os, re, sys
from pathlib import Path


def repo_root() -> Path:
    p = Path.cwd().resolve()
    for d in (p, *p.parents):
        if (d / "judger.py").is_file():
            return d
    return p


REPO_ROOT = repo_root()
sys.path.insert(0, str(REPO_ROOT))

GPU_ID = "0"

_GRPO_MERGED  = REPO_ROOT / "artifacts" / "models" / "grpo_v1_merged"
_QLORA_MERGED = REPO_ROOT / "artifacts" / "models" / "qlora_v1_merged"
_BASE_MODEL   = "Qwen/Qwen3-4B-Thinking-2507"

if (_GRPO_MERGED / "config.json").is_file():
    MODEL_ID, MODEL_LABEL = str(_GRPO_MERGED), "grpo_v1_merged"
elif (_QLORA_MERGED / "config.json").is_file():
    MODEL_ID, MODEL_LABEL = str(_QLORA_MERGED), "qlora_v1_merged"
else:
    MODEL_ID, MODEL_LABEL = _BASE_MODEL, "base_model"

print(f"Model: {MODEL_LABEL}")

PRIVATE_PATH = REPO_ROOT / "data" / "raw" / "private.jsonl"
assert PRIVATE_PATH.is_file(), f"private.jsonl not found: {PRIVATE_PATH}"

RUN_NAME        = f"private_{MODEL_LABEL}"
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
    LLM = SamplingParams = None
    VLLM_AVAILABLE = False

from tqdm.auto import tqdm

print(f"REPO_ROOT   : {REPO_ROOT}")
print(f"PRIVATE_PATH: {PRIVATE_PATH}")
print(f"CHECKPOINT  : {CHECKPOINT_PATH}")
print(f"vLLM        : {VLLM_AVAILABLE}")
''')

CELL_DATA = code_cell('''\
with open(PRIVATE_PATH, encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

data_run = data  # always all private questions

n_mcq = sum(bool(d.get("options")) for d in data)
print(f"Loaded {len(data)} private questions ({n_mcq} MCQ, {len(data)-n_mcq} free-form)")
''')

CELL_PROMPTS = code_cell('''\
from typing import Optional

SYSTEM_PROMPT_MATH = (
    "You are an expert mathematician with deep knowledge of all areas of mathematics, "
    "from algebra and calculus to number theory and combinatorics. "
    "This problem is very important to my career - please think carefully and be precise.\\n\\n"
    "Solve using this structured approach:\\n"
    "1. UNDERSTAND: Identify what is given and what you need to find.\\n"
    "2. PLAN: Write down the key equations, formulas, or theorems you will use.\\n"
    "3. SOLVE: Work through each step carefully. Compute intermediate results explicitly. "
    "Pay special attention to arithmetic - do not skip steps.\\n"
    "4. VERIFY: Check that your answer satisfies all conditions in the problem. "
    "Check units, sign, and order of magnitude.\\n"
    "5. ANSWER: Put your final answer in \\\\boxed{}.\\n\\n"
    "Additional rules:\\n"
    "- If the problem has multiple blanks ([ANS] placeholders), put ALL answers "
    "comma-separated in ONE \\\\boxed{} in the order they appear. "
    "Example: \\\\boxed{3, -7, 42}.\\n"
    "- Simplify all fractions and radical expressions completely.\\n"
    "- You\'d better be sure of your answer."
)

SYSTEM_PROMPT_MCQ = (
    "You are an expert mathematician with deep knowledge of all areas of mathematics. "
    "This problem is very important to my career - please think carefully and be precise.\\n\\n"
    "Solve using this structured approach:\\n"
    "1. UNDERSTAND: Read the problem and all answer choices carefully.\\n"
    "2. PLAN: Identify the relevant concepts, formulas, or theorems that apply.\\n"
    "3. SOLVE: Work through the problem step by step. Compute intermediate results "
    "explicitly - do not skip arithmetic steps.\\n"
    "4. ELIMINATE: Cross out answer choices that are clearly wrong.\\n"
    "5. VERIFY: Confirm your chosen answer is consistent with every condition in the problem.\\n"
    "6. ANSWER: On the very last line of your response, write ONLY \\\\boxed{X} "
    "where X is the letter of the correct answer (A-J). "
    "Do not write any text after \\\\boxed{}.\\n\\n"
    "You\'d better be sure of your answer."
)


def build_prompt(question: str, options: Optional[list]) -> tuple:
    if options:
        labels   = [chr(65 + i) for i in range(len(options))]
        opts_txt = "\\n".join(f"{l}. {o.strip()}" for l, o in zip(labels, options))
        return SYSTEM_PROMPT_MCQ, f"{question}\\n\\nOptions:\\n{opts_txt}"
    n_ans = question.count("[ANS]")
    if n_ans > 1:
        hint = (
            f"\\n\\n[Note: This problem has {n_ans} answers. "
            f"Put all {n_ans} answers comma-separated in ONE \\\\boxed{{}} "
            f"in the order they appear in the question.]"
        )
        return SYSTEM_PROMPT_MATH, question + hint
    return SYSTEM_PROMPT_MATH, question
''')

CELL_HELPERS = code_cell('''\
from collections import Counter
from math import ceil


def extract_boxed(text: str) -> str:
    matches, needle, i = [], r"\\boxed{", 0
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
    extracted = [extract_boxed(s) for s in samples]
    nonempty  = [e for e in extracted if e]
    if nonempty:
        counts    = Counter(nonempty)
        top_count = counts.most_common(1)[0][1]
        tied      = {a for a, c in counts.items() if c == top_count}
        candidates = [i for i, a in enumerate(extracted) if a in tied]
        best_idx  = max(candidates, key=lambda i: len(samples[i]))
        best_ans  = extracted[best_idx]
    else:
        top_count, best_idx, best_ans = 0, 0, ""
    uncertain = (is_uncertain(samples[best_idx], finish_reasons[best_idx])
                 or top_count < ceil(len(samples) / 2))
    return {"response": samples[best_idx], "answer": best_ans,
            "finish_reason": finish_reasons[best_idx], "consensus_count": top_count,
            "n_samples": len(samples), "uncertain": uncertain}


def make_sampling_params(max_tokens: int, temperature: float = 0.6,
                          repetition_penalty: float = 1.0):
    return SamplingParams(max_tokens=max_tokens, temperature=temperature,
                          top_p=0.95, top_k=20, min_p=0.0,
                          presence_penalty=0.0, repetition_penalty=repetition_penalty)


def build_chat_prompt(item: dict, thinking_budget=None,
                       prefix: str = "", suffix: str = "") -> str:
    system, user = build_prompt(item["question"], item.get("options"))
    if prefix:
        user = prefix + user
    if suffix:
        user = user + suffix
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    kwargs = dict(tokenize=False, add_generation_prompt=True, enable_thinking=True)
    if thinking_budget is not None:
        kwargs["thinking_budget"] = thinking_budget
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError as exc:
        if thinking_budget is None:
            raise
        hint = (f"Use at most about {thinking_budget} thinking tokens. "
                "Be concise but do not skip necessary arithmetic.\\n\\n")
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
    print(f"Checkpoint: {len(records)} records")
    return records


def write_checkpoint(path, records: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in sorted(records.values(), key=lambda r: int(r["id"])):
            f.write(json.dumps(rec, ensure_ascii=False) + "\\n")
''')

CELL_MODEL = code_cell('''\
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

llm = LLM(
    model=MODEL_ID,
    quantization="bitsandbytes",
    load_format="bitsandbytes",
    gpu_memory_utilization=0.78,
    max_model_len=8192,
    trust_remote_code=True,
    max_num_seqs=4,
    max_num_batched_tokens=4096,
)

sampling_params = make_sampling_params(MAX_TOKENS, temperature=0.6)
print(f"Model loaded: {MODEL_LABEL}")
''')

CELL_GENERATE = code_cell('''\
response_records = load_checkpoint(CHECKPOINT_PATH)

# Phase 1 — single batched call for all questions
phase1_params  = make_sampling_params(PHASE1_MAX_TOKENS, temperature=0.6)
missing_phase1 = [item for item in data_run if str(item["id"]) not in response_records]

if missing_phase1:
    print(f"Phase 1: {len(missing_phase1)} questions")
    phase1_prompts = [build_chat_prompt(item, thinking_budget=PHASE1_THINKING_BUDGET)
                      for item in missing_phase1]
    phase1_outputs = llm.generate(phase1_prompts, phase1_params)

    for item, out in zip(missing_phase1, phase1_outputs):
        response      = out.outputs[0].text.strip()
        finish_reason = str(out.outputs[0].finish_reason)
        response_records[str(item["id"])] = {
            "id": item["id"], "phase_used": 1, "response": response,
            "answer": extract_boxed(response), "finish_reason": finish_reason,
            "uncertain": is_uncertain(response, finish_reason),
            "n_samples": 1, "consensus_count": 1,
        }
    write_checkpoint(CHECKPOINT_PATH, response_records)

p1_uncertain = sum(1 for item in data_run
                   if response_records.get(str(item["id"]), {}).get("uncertain"))
print(f"Phase 1 done: {p1_uncertain}/{len(data_run)} uncertain")

# Phase 2 — single batched call for all uncertain questions
phase2_params    = make_sampling_params(PHASE2_MAX_TOKENS, temperature=PHASE2_TEMPERATURE,
                                        repetition_penalty=PHASE2_REPETITION_PENALTY)
phase1_uncertain = [item for item in data_run
                    if response_records.get(str(item["id"]), {}).get("uncertain")
                    and int(response_records.get(str(item["id"]), {}).get("phase_used", 0)) < 2]

RETRY_PREFIX      = "Previous attempt was unclear. Solve this again carefully from scratch:\\n\\n"
MCQ_VERIFY_SUFFIX = (
    "\\n\\nAfter finding your answer, check each option against the problem conditions. "
    "Eliminate any letter that clearly fails. "
    "Then on the very last line write ONLY \\\\boxed{X}."
)

if phase1_uncertain:
    print(f"Phase 2: {len(phase1_uncertain)} uncertain x {PHASE2_N_SAMPLES} = "
          f"{len(phase1_uncertain)*PHASE2_N_SAMPLES} prompts")
    phase2_prompts = []
    for item in phase1_uncertain:
        suffix = MCQ_VERIFY_SUFFIX if item.get("options") else ""
        p = build_chat_prompt(item, thinking_budget=PHASE2_THINKING_BUDGET,
                              prefix=RETRY_PREFIX, suffix=suffix)
        for _ in range(PHASE2_N_SAMPLES):
            phase2_prompts.append(p)

    phase2_flat = llm.generate(phase2_prompts, phase2_params)

    for q_idx, item in enumerate(phase1_uncertain):
        start = q_idx * PHASE2_N_SAMPLES
        end   = start + PHASE2_N_SAMPLES
        samples        = [phase2_flat[j].outputs[0].text.strip()       for j in range(start, end)]
        finish_reasons = [str(phase2_flat[j].outputs[0].finish_reason) for j in range(start, end)]
        chosen         = choose_best_sample(samples, finish_reasons)
        chosen.update({"id": item["id"], "phase_used": 2})
        response_records[str(item["id"])] = chosen

    write_checkpoint(CHECKPOINT_PATH, response_records)

missing_ids = [item["id"] for item in data_run if str(item["id"]) not in response_records]
if missing_ids:
    raise RuntimeError(f"Missing responses for {len(missing_ids)} id(s): {missing_ids[:10]}")

responses    = [response_records[str(item["id"])]["response"] for item in data_run]
phase_counts = Counter(response_records[str(item["id"])].get("phase_used") for item in data_run)
uncertain    = sum(bool(response_records[str(item["id"])].get("uncertain")) for item in data_run)

print(f"Done: {len(responses)} responses | "
      f"phase1={phase_counts.get(1,0)} phase2={phase_counts.get(2,0)} uncertain={uncertain}")
''')

CELL_SAVE = code_cell('''\
import csv
from datetime import date

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    for item, response in zip(data_run, responses):
        meta = response_records.get(str(item["id"]), {})
        f.write(json.dumps({
            "id":              item["id"],
            "is_mcq":          bool(item.get("options")),
            "response":        response,
            "phase_used":      meta.get("phase_used"),
            "uncertain":       meta.get("uncertain"),
            "finish_reason":   meta.get("finish_reason"),
            "consensus_count": meta.get("consensus_count"),
            "n_samples":       meta.get("n_samples"),
            "model":           MODEL_LABEL,
        }, ensure_ascii=False) + "\\n")

SUBMISSION_PATH = (
    REPO_ROOT / "artifacts" / "submissions"
    / f"submission_{MODEL_LABEL}_{date.today().isoformat()}.csv"
)
SUBMISSION_PATH.parent.mkdir(parents=True, exist_ok=True)

rows = sorted(
    [{"id": item["id"], "response": response_records[str(item["id"])]["response"]}
     for item in data_run],
    key=lambda r: r["id"],
)

with open(SUBMISSION_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["id", "response"], quoting=csv.QUOTE_ALL)
    writer.writeheader()
    writer.writerows(rows)

print(f"JSONL : {OUTPUT_PATH}")
print(f"CSV   : {SUBMISSION_PATH}  ({len(rows)} rows)")
''')

cells = [CELL_INTRO, CELL_CONFIG, CELL_DATA, CELL_PROMPTS, CELL_HELPERS,
         CELL_MODEL, CELL_GENERATE, CELL_SAVE]

out = REPO / "notebooks" / "05_private_submission.ipynb"
with open(out, "w", encoding="utf-8") as f:
    json.dump(notebook(cells), f, ensure_ascii=False, indent=1)
print(f"Created: {out}  ({len(cells)} cells)")
