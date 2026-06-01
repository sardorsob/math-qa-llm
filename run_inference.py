#!/usr/bin/env python3
"""
Single entry point for CSE 151B private-set inference.

This script loads the merged QLoRA model and EBM verifier head from Hugging
Face Hub, runs the private-set pipeline end-to-end, applies the same v1-style
truncated-answer recovery used in the submission workflow, and writes the
final competition CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


HF_MODEL_ID = "SardorSob/qwen3-4b-thinking-math-qa-qlora"
EBM_SUBFOLDER = "ebm"
MODEL_LABEL = "qlora_v1_merged"

# Improved verification defaults. These are more generous than the original
# deadline run because graders should have more compute available.
PHASE1_THINKING_BUDGET = 4096
PHASE1_MAX_TOKENS = 5120
PHASE1_TEMPERATURE = 0.6

PHASE2_THINKING_BUDGET = 8192
PHASE2_MAX_TOKENS = 10240
PHASE2_N_SAMPLES = 4
PHASE2_TEMPERATURE = 0.65
PHASE2_REPETITION_PENALTY = 1.05

EBM_SCORE_BATCH = 2
EBM_SCORE_MAX_LEN = 3072
USE_EBM = True


SYSTEM_PROMPT_MATH = (
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

RETRY_PREFIX = "Previous attempt was unclear. Solve this again carefully from scratch:\n\n"
MCQ_VERIFY_SUFFIX = (
    "\n\nAfter finding your answer, check each option against the problem conditions. "
    "Eliminate any letter that clearly fails. "
    "Then on the very last line write ONLY \\boxed{X}."
)


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
        j = idx + len(needle)
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


def build_prompt(question: str, options: Optional[list]) -> tuple[str, str]:
    if options:
        labels = [chr(65 + i) for i in range(len(options))]
        opts_txt = "\n".join(f"{label}. {opt.strip()}" for label, opt in zip(labels, options))
        return SYSTEM_PROMPT_MCQ, f"{question}\n\nOptions:\n{opts_txt}"
    n_ans = question.count("[ANS]")
    if n_ans > 1:
        hint = (
            f"\n\n[Note: This problem has {n_ans} answers. "
            f"Put all {n_ans} answers comma-separated in ONE \\boxed{{}} "
            f"in the order they appear in the question.]"
        )
        return SYSTEM_PROMPT_MATH, question + hint
    return SYSTEM_PROMPT_MATH, question


def build_chat_prompt(
    item: dict,
    tokenizer,
    thinking_budget: int | None = None,
    prefix: str = "",
    suffix: str = "",
) -> str:
    system, user = build_prompt(item["question"], item.get("options"))
    if prefix:
        user = prefix + user
    if suffix:
        user = user + suffix
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    kwargs = dict(tokenize=False, add_generation_prompt=True, enable_thinking=True)
    if thinking_budget is not None:
        kwargs["thinking_budget"] = thinking_budget
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        if thinking_budget is None:
            raise
        hint = (
            f"Use at most about {thinking_budget} thinking tokens. "
            "Be concise but do not skip necessary arithmetic.\n\n"
        )
        messages[1]["content"] = hint + messages[1]["content"]
        kwargs.pop("thinking_budget", None)
        return tokenizer.apply_chat_template(messages, **kwargs)


def make_sampling_params(
    max_tokens: int,
    tokenizer,
    temperature: float = 0.6,
    repetition_penalty: float = 1.0,
) -> dict:
    return dict(
        max_new_tokens=max_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=0.95,
        top_k=20,
        repetition_penalty=repetition_penalty,
        pad_token_id=tokenizer.eos_token_id,
    )


class VerifierHead(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.head = nn.Linear(hidden_size, 1, bias=True)

    def score(self, base_model, input_ids, attention_mask) -> torch.Tensor:
        backbone = base_model.model if hasattr(base_model, "model") else base_model
        with torch.no_grad():
            hidden = backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            ).last_hidden_state
        seq_lengths = attention_mask.sum(dim=1) - 1
        last_tok = hidden[torch.arange(hidden.size(0), device=hidden.device), seq_lengths]
        return self.head(last_tok).squeeze(-1)


def score_candidates_ebm(item, candidates, verifier_head, tokenizer, llm) -> list[float]:
    system = "You are evaluating whether this mathematical solution is correct and complete."
    user = item["question"]
    if item.get("options"):
        labels = [chr(65 + i) for i in range(len(item["options"]))]
        user += "\n\nOptions:\n" + "\n".join(
            f"{label}. {opt.strip()}" for label, opt in zip(labels, item["options"])
        )

    texts = []
    for cand in candidates:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": cand},
        ]
        texts.append(tokenizer.apply_chat_template(messages, tokenize=False))

    saved_side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    scores: list[float] = []
    try:
        for start in range(0, len(texts), EBM_SCORE_BATCH):
            mini = texts[start : start + EBM_SCORE_BATCH]
            enc = tokenizer(
                mini,
                padding=True,
                truncation=True,
                max_length=EBM_SCORE_MAX_LEN,
                return_tensors="pt",
            ).to(llm.device)
            try:
                batch_scores = verifier_head.score(llm, enc["input_ids"], enc["attention_mask"])
                scores.extend(batch_scores.float().tolist())
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                for i in range(len(mini)):
                    single = verifier_head.score(
                        llm,
                        enc["input_ids"][i : i + 1],
                        enc["attention_mask"][i : i + 1],
                    )
                    scores.append(single.float().item())
                    torch.cuda.empty_cache()
            del enc
            torch.cuda.empty_cache()
    finally:
        tokenizer.padding_side = saved_side
    return scores


def choose_best_sample(samples, finish_reasons) -> dict:
    extracted = [extract_boxed(s) for s in samples]
    nonempty = [e for e in extracted if e]
    if nonempty:
        counts = Counter(nonempty)
        top_count = counts.most_common(1)[0][1]
        tied = {ans for ans, count in counts.items() if count == top_count}
        candidates = [i for i, ans in enumerate(extracted) if ans in tied]
        best_idx = max(candidates, key=lambda i: len(samples[i]))
        best_answer = extracted[best_idx]
    else:
        best_idx = 0
        best_answer = ""
    return {
        "response": samples[best_idx],
        "answer": best_answer,
        "finish_reason": finish_reasons[best_idx],
        "n_samples": len(samples),
    }


def choose_best_sample_ebm(item, samples, finish_reasons, verifier_head, tokenizer, llm) -> dict:
    if verifier_head is None:
        return choose_best_sample(samples, finish_reasons)
    try:
        scores = score_candidates_ebm(item, samples, verifier_head, tokenizer, llm)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return choose_best_sample(samples, finish_reasons)
    best_idx = scores.index(max(scores))
    best_answer = extract_boxed(samples[best_idx])
    if not best_answer and any(extract_boxed(s) for s in samples):
        return choose_best_sample(samples, finish_reasons)
    return {
        "response": samples[best_idx],
        "answer": best_answer,
        "finish_reason": finish_reasons[best_idx],
        "n_samples": len(samples),
    }


def generate_batch(prompts, gen_kwargs, tokenizer, llm, chunk_size: int) -> list[dict]:
    results = []
    for start in range(0, len(prompts), chunk_size):
        chunk = prompts[start : start + chunk_size]
        inputs = tokenizer(
            chunk,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=16384,
        ).to(llm.device)
        prompt_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            out_ids = llm.generate(**inputs, **gen_kwargs)
        new_ids = out_ids[:, prompt_len:]
        for seq in new_ids:
            last_tok = seq[-1].item()
            finish = "length" if last_tok != tokenizer.eos_token_id else "stop"
            text = tokenizer.decode(seq, skip_special_tokens=True).strip()
            results.append({"text": text, "finish_reason": finish})
        del inputs, out_ids, new_ids
        torch.cuda.empty_cache()
    return results


# v1-style truncated-answer recovery. This keeps the simpler, deadline-era
# boxing pass instead of the later v2 recovery refinements.
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


def is_valid_freeform(val: str) -> bool:
    if not val or len(val) > 40:
        return False
    if val.lower() in ENGLISH_BLACKLIST:
        return False
    if not (any(c.isdigit() for c in val) or "\\" in val or len(val) <= 2):
        return False
    return True


def recover_mcq_letter(response: str) -> str:
    tail = response[-1500:] if len(response) > 1500 else response
    tail_lower = tail.lower()

    for pat in MCQ_HIGH_CONF:
        for match in reversed(list(re.finditer(pat, tail_lower, re.IGNORECASE))):
            start_pos = match.start()
            actual = re.search(
                pat,
                tail[start_pos : start_pos + match.end() - match.start() + 2],
                re.IGNORECASE,
            )
            if actual:
                letter = actual.group(1).upper()
                if letter in "ABCDEFGHIJ":
                    return letter

    for pat in MCQ_MED_CONF:
        matches = list(re.finditer(pat, tail, re.IGNORECASE))
        if matches:
            letter = matches[-1].group(1).upper()
            if letter in "ABCDEFGHIJ":
                return letter

    standalone_pat = r"(?:\*{0,2}|\(|\s|^)([A-J])(?:\*{0,2}|\)|\s|\.|,|$)"
    matches = list(re.finditer(standalone_pat, tail))
    if matches:
        late_matches = [m for m in matches if m.start() >= len(tail) - 500]
        if late_matches:
            return late_matches[-1].group(1)
        return matches[-1].group(1)
    return ""


def recover_freeform_value(response: str) -> str:
    tail = response[-1500:] if len(response) > 1500 else response

    for pat in FF_HIGH_CONF:
        matches = list(re.finditer(pat, tail, re.IGNORECASE))
        for match in reversed(matches):
            val = match.group(1).strip()
            if is_valid_freeform(val):
                return val

    for pat in FF_MED_CONF:
        matches = list(re.finditer(pat, tail, re.IGNORECASE))
        for match in reversed(matches):
            val = match.group(1).strip()
            if is_valid_freeform(val):
                return val

    last_500 = tail[-500:]
    nums = re.findall(r"-?\d+(?:\.\d+)?(?:/\d+)?", last_500)
    if nums:
        return nums[-1]
    return ""


def recover_multi_answer(response: str, n_ans: int) -> str:
    tail = response[-1500:] if len(response) > 1500 else response
    nums = re.findall(r"-?\d+(?:\.\d+)?(?:/\d+)?", tail)
    if len(nums) >= n_ans:
        return ", ".join(nums[-n_ans:])
    return ""


def apply_recovery(records: dict, private_lookup: dict) -> int:
    recovered = 0
    for item_id, info in private_lookup.items():
        rec = records.get(item_id)
        if not rec:
            continue
        if extract_boxed(rec["response"]):
            continue

        if info["is_mcq"]:
            value = recover_mcq_letter(rec["response"])
        else:
            n_ans = info["question"].count("[ANS]") or 1
            if n_ans > 1:
                value = recover_multi_answer(rec["response"], n_ans)
                if not value:
                    value = recover_freeform_value(rec["response"])
            else:
                value = recover_freeform_value(rec["response"])

        if value:
            rec["response"] = rec["response"].rstrip() + f"\n\n\\boxed{{{value}}}"
            recovered += 1
    return recovered


def load_model_and_tokenizer():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "run_inference.py requires a CUDA-capable GPU. "
            "This pipeline is not designed for CPU-only execution."
        )

    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)

    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    vram_gb = (
        torch.cuda.get_device_properties(0).total_memory / 1e9
        if torch.cuda.is_available()
        else 0
    )
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"GPU: {gpu_name} | VRAM: {vram_gb:.1f} GB")

    common_kwargs = dict(
        device_map={"": 0},
        trust_remote_code=True,
        attn_implementation="sdpa",
    )

    if vram_gb >= 23:
        try:
            llm = AutoModelForCausalLM.from_pretrained(
                HF_MODEL_ID,
                dtype=torch.bfloat16,
                **common_kwargs,
            )
            label = "BF16"
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print("BF16 OOM, falling back to NF4")
            llm = AutoModelForCausalLM.from_pretrained(
                HF_MODEL_ID,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                ),
                **common_kwargs,
            )
            label = "NF4"
    else:
        print(f"VRAM {vram_gb:.1f} GB is tight; using NF4 quantization")
        llm = AutoModelForCausalLM.from_pretrained(
            HF_MODEL_ID,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            ),
            **common_kwargs,
        )
        label = "NF4"

    llm.eval()
    print(f"Model loaded ({label}) | VRAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    return llm, tokenizer, vram_gb


def load_ebm_head(llm):
    if not USE_EBM:
        return None
    try:
        head_path = hf_hub_download(
            repo_id=HF_MODEL_ID,
            filename=f"{EBM_SUBFOLDER}/verifier_head.pt",
        )
    except Exception as exc:
        print(f"Could not download EBM head ({exc}); falling back to majority vote")
        return None

    head = VerifierHead(llm.config.hidden_size).to(llm.device).to(torch.bfloat16)
    head.load_state_dict(torch.load(head_path, map_location=llm.device))
    head.eval()
    print(f"EBM verifier head loaded (Phase 2 reranks {PHASE2_N_SAMPLES} candidates)")
    return head


def auto_tune_batch_sizes(vram_gb: float) -> tuple[int, int]:
    if vram_gb >= 70:
        return 4, 16
    if vram_gb >= 38:
        return 2, 8
    if vram_gb >= 22:
        return 1, 4
    return 1, 2


def _write_checkpoint(path: Path, records: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for rec in sorted(records.values(), key=lambda row: int(row["id"])):
            handle.write(json.dumps(rec, ensure_ascii=False) + "\n")


def run_inference(
    private_jsonl: str | Path = "data/raw/private.jsonl",
    output_csv: str | Path = "submission.csv",
    checkpoint_path: str | Path | None = None,
):
    """Run the private-set pipeline and return the written CSV path."""
    private_jsonl = Path(private_jsonl)
    output_csv = Path(output_csv)
    checkpoint = Path(checkpoint_path) if checkpoint_path else output_csv.with_suffix(".checkpoint.jsonl")

    assert private_jsonl.is_file(), f"Private dataset not found: {private_jsonl}"

    with open(private_jsonl, encoding="utf-8") as handle:
        data = [json.loads(line) for line in handle]
    print(f"Loaded {len(data)} questions from {private_jsonl}")

    private_lookup = {
        str(item["id"]): {
            "options": item.get("options"),
            "is_mcq": bool(item.get("options")),
            "question": item.get("question", ""),
        }
        for item in data
    }

    records: dict[str, dict] = {}
    if checkpoint.exists():
        with open(checkpoint, encoding="utf-8") as handle:
            for line in handle:
                rec = json.loads(line)
                records[str(rec["id"])] = rec
        print(f"Resuming from checkpoint: {len(records)} questions already done")

    llm, tokenizer, vram_gb = load_model_and_tokenizer()
    verifier_head = load_ebm_head(llm)
    phase2_batch_questions, chunk_size = auto_tune_batch_sizes(vram_gb)
    print(
        f"Auto-tuned batching: PHASE2_BATCH_QUESTIONS={phase2_batch_questions}, "
        f"CHUNK_SIZE={chunk_size}"
    )

    phase1_params = make_sampling_params(
        PHASE1_MAX_TOKENS,
        tokenizer=tokenizer,
        temperature=PHASE1_TEMPERATURE,
    )
    missing_phase1 = [item for item in data if str(item["id"]) not in records]
    print(f"Phase 1: {len(missing_phase1)} questions to generate")

    with tqdm(total=len(missing_phase1), desc="Phase 1", unit="q") as pbar:
        for start in range(0, len(missing_phase1), chunk_size):
            batch = missing_phase1[start : start + chunk_size]
            prompts = [
                build_chat_prompt(item, tokenizer, thinking_budget=PHASE1_THINKING_BUDGET)
                for item in batch
            ]
            outputs = generate_batch(prompts, phase1_params, tokenizer, llm, chunk_size=len(batch))
            for item, out in zip(batch, outputs):
                records[str(item["id"])] = {
                    "id": item["id"],
                    "phase_used": 1,
                    "response": out["text"],
                    "finish_reason": out["finish_reason"],
                    "uncertain": is_uncertain(out["text"], out["finish_reason"]),
                }
            _write_checkpoint(checkpoint, records)
            pbar.update(len(batch))

    p1_uncertain = sum(1 for item in data if records[str(item["id"])].get("uncertain"))
    print(f"Phase 1 done: {p1_uncertain}/{len(data)} uncertain")

    phase2_params = make_sampling_params(
        PHASE2_MAX_TOKENS,
        tokenizer=tokenizer,
        temperature=PHASE2_TEMPERATURE,
        repetition_penalty=PHASE2_REPETITION_PENALTY,
    )
    uncertain_items = [
        item
        for item in data
        if records[str(item["id"])].get("uncertain")
        and int(records[str(item["id"])].get("phase_used", 0)) < 2
    ]
    print(
        f"Phase 2: {len(uncertain_items)} uncertain x {PHASE2_N_SAMPLES} samples "
        f"(batched {phase2_batch_questions} questions per call)"
    )

    with tqdm(total=len(uncertain_items), desc="Phase 2", unit="q") as pbar:
        for start in range(0, len(uncertain_items), phase2_batch_questions):
            batch_items = uncertain_items[start : start + phase2_batch_questions]
            all_prompts = []
            for item in batch_items:
                suffix = MCQ_VERIFY_SUFFIX if item.get("options") else ""
                prompt = build_chat_prompt(
                    item,
                    tokenizer,
                    thinking_budget=PHASE2_THINKING_BUDGET,
                    prefix=RETRY_PREFIX,
                    suffix=suffix,
                )
                all_prompts.extend([prompt] * PHASE2_N_SAMPLES)

            outputs = generate_batch(
                all_prompts,
                phase2_params,
                tokenizer,
                llm,
                chunk_size=len(all_prompts),
            )

            for q_idx, item in enumerate(batch_items):
                left = q_idx * PHASE2_N_SAMPLES
                right = left + PHASE2_N_SAMPLES
                samples = [outputs[i]["text"] for i in range(left, right)]
                finish_reasons = [outputs[i]["finish_reason"] for i in range(left, right)]
                chosen = choose_best_sample_ebm(
                    item,
                    samples,
                    finish_reasons,
                    verifier_head,
                    tokenizer,
                    llm,
                )
                chosen.update({"id": item["id"], "phase_used": 2})
                chosen["uncertain"] = is_uncertain(chosen["response"], chosen["finish_reason"])
                records[str(item["id"])] = chosen

            _write_checkpoint(checkpoint, records)
            pbar.update(len(batch_items))

    rec_count = apply_recovery(records, private_lookup)
    print(f"Recovery: appended boxed answers to {rec_count} truncated responses")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(
        [{"id": int(rec_id), "response": rec["response"]} for rec_id, rec in records.items()],
        key=lambda row: row["id"],
    )
    with open(output_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "response"], quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)

    has_boxed = sum(1 for rec in records.values() if extract_boxed(rec["response"]))
    print()
    print("Submission CSV written")
    print(f"  Path: {output_csv}")
    print(f"  Rows: {len(rows)}")
    print(f"  Responses with boxed answers: {has_boxed}/{len(records)}")
    print(f"  Checkpoint: {checkpoint}")
    return output_csv


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CSE 151B Math QA private-set inference pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--private-jsonl",
        type=str,
        default="data/raw/private.jsonl",
        help="Path to the private dataset JSONL.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="submission.csv",
        help="Where to write the final submission CSV.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional checkpoint JSONL path. Defaults to <output-csv>.checkpoint.jsonl",
    )
    args = parser.parse_args()

    run_inference(
        private_jsonl=args.private_jsonl,
        output_csv=args.output_csv,
        checkpoint_path=args.checkpoint,
    )
