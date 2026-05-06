"""Generate notebooks/03_qlora_finetune.ipynb and notebooks/04_grpo_train.ipynb."""
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
NB_DIR = REPO / "notebooks"
NB_DIR.mkdir(parents=True, exist_ok=True)


def cid() -> str:
    return uuid.uuid4().hex[:8]


def src(code: str) -> list:
    code = code.lstrip("\n")
    lines = code.split("\n")
    out = [line + "\n" for line in lines[:-1]] + [lines[-1]]
    if out and out[-1] == "":
        out.pop()
    return out


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cid(),
        "metadata": {},
        "outputs": [],
        "source": src(text),
    }


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": cid(),
        "metadata": {},
        "source": src(text),
    }


def notebook(cells: list) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python (cse151b WSL)",
                "language": "python",
                "name": "cse151b-wsl",
            },
            "language_info": {"name": "python", "version": "3.11.0"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Shared prompt strings (identical to 02_inference.ipynb)
# ─────────────────────────────────────────────────────────────────────────────

SHARED_PROMPTS = r'''
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
    "- You\'d better be sure of your answer."
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
    "You\'d better be sure of your answer."
)


def build_prompt(question: str, options) -> tuple:
    if options:
        labels   = [chr(65 + i) for i in range(len(options))]
        opts_txt = "\n".join(f"{l}. {o.strip()}" for l, o in zip(labels, options))
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
'''.strip()


# ─────────────────────────────────────────────────────────────────────────────
#  NOTEBOOK 03 — QLoRA Fine-Tuning
# ─────────────────────────────────────────────────────────────────────────────

NB03_CELLS = [

md("""# 03 — QLoRA Fine-Tuning

Fine-tune `Qwen/Qwen3-4B-Thinking-2507` with QLoRA (NF4 + LoRA rank-64) on the
public competition set plus a 20K subset of NuminaMath-CoT.

**Rejection-sampling SFT:** if a prior inference run exists
(`adaptive_public_v2_results.jsonl`), the model's own correct completions are used
as training targets. Incorrect or missing completions fall back to the gold answer.
NuminaMath-CoT examples provide additional rich CoT signal across a wider problem
distribution.

**Run order:** 02 (full public inference) → this notebook → 04 (GRPO) → 05 (private).
"""),

code('''import json, os, re, sys
from pathlib import Path


def repo_root() -> Path:
    p = Path.cwd().resolve()
    for d in (p, *p.parents):
        if (d / "judger.py").is_file():
            return d
    return p


REPO_ROOT = repo_root()
sys.path.insert(0, str(REPO_ROOT))

MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"
GPU_ID   = "0"

LORA_RANK    = 64
LORA_ALPHA   = 128
LORA_DROPOUT = 0.05
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

LEARNING_RATE  = 2e-4
BATCH_SIZE     = 1
GRAD_ACCUM     = 8
EPOCHS         = 3
MAX_SEQ_LENGTH = 1024
WARMUP_STEPS   = 50
NUMINA_SUBSET  = 20_000

PUBLIC_PATH  = REPO_ROOT / "data" / "raw" / "public.jsonl"
PREV_RESULTS = REPO_ROOT / "artifacts" / "logs" / "runs" / "adaptive_public_v2_results.jsonl"
ADAPTER_OUT  = REPO_ROOT / "artifacts" / "models" / "qlora_v1"

os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID

print(f"REPO_ROOT    : {REPO_ROOT}")
print(f"public       : {PUBLIC_PATH.is_file()} | prev_results: {PREV_RESULTS.is_file()}")
print(f"adapter_out  : {ADAPTER_OUT}")'''),

code('''import subprocess, sys

for dep in ["peft", "trl>=0.12.0", "datasets", "accelerate"]:
    pkg = dep.split(">=")[0].split("==")[0]
    try:
        __import__(pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", dep])'''),

code('''import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
from datasets import Dataset, load_dataset

print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU : {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")'''),

md("## 1. Build Training Data"),

code('''with open(PUBLIC_PATH, encoding="utf-8") as f:
    public_data = [json.loads(line) for line in f]

prev_results: dict[str, dict] = {}
if PREV_RESULTS.is_file():
    with open(PREV_RESULTS, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            prev_results[str(rec["id"])] = rec

def gold_to_boxed(item: dict) -> str:
    gold = item.get("answer")
    if isinstance(gold, list):
        return "\\\\boxed{" + ", ".join(str(g) for g in gold) + "}"
    return "\\\\boxed{" + str(gold) + "}"

train_items = []
for item in public_data:
    res = prev_results.get(str(item["id"]))
    if res and res.get("correct") and res.get("response"):
        train_items.append({"item": item, "target": res["response"], "src": "model"})
    elif item.get("answer") is not None:
        train_items.append({"item": item, "target": gold_to_boxed(item), "src": "gold"})

from_model = sum(1 for t in train_items if t["src"] == "model")
print(f"Public: {len(train_items)} examples ({from_model} model correct, {len(train_items)-from_model} gold)")'''),

md("## 2. Load Tokenizer & Build Dataset"),

code(SHARED_PROMPTS + '''


tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.pad_token    = tokenizer.eos_token
tokenizer.padding_side = "right"


def format_public(t: dict) -> str:
    system, user = build_prompt(t["item"]["question"], t["item"].get("options"))
    msgs = [
        {"role": "system",    "content": system},
        {"role": "user",      "content": user},
        {"role": "assistant", "content": t["target"]},
    ]
    return tokenizer.apply_chat_template(msgs, tokenize=False)


def format_numina(ex: dict) -> str:
    msgs = [
        {"role": "system",    "content": SYSTEM_PROMPT_MATH},
        {"role": "user",      "content": ex["problem"]},
        {"role": "assistant", "content": ex["solution"]},
    ]
    return tokenizer.apply_chat_template(msgs, tokenize=False)


public_texts = [format_public(t) for t in train_items]

print(f"Downloading NuminaMath-CoT {NUMINA_SUBSET:,} samples...")
nds = load_dataset("AI-MO/NuminaMath-CoT", split="train")
nds = nds.shuffle(seed=42).select(range(NUMINA_SUBSET))
numina_texts = [format_numina(ex) for ex in nds]

all_texts = public_texts + numina_texts
dataset   = Dataset.from_dict({"text": all_texts})
print(f"Dataset: {len(dataset)} examples ({len(public_texts)} public + {len(numina_texts)} NuminaMath)")'''),

md("## 3. Load Model with QLoRA"),

code('''bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
model.config.use_cache = False

lora_cfg = LoraConfig(
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=LORA_TARGETS,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()

alloc = torch.cuda.memory_allocated() / 1e9
print(f"VRAM allocated: {alloc:.2f} GB")'''),

md("## 4. Train"),

code('''RESPONSE_TEMPLATE = "<|im_start|>assistant\\n"

try:
    collator     = DataCollatorForCompletionOnlyLM(RESPONSE_TEMPLATE, tokenizer=tokenizer)
    use_collator = True
except Exception as e:
    collator, use_collator = None, False
    print(f"Collator unavailable ({e}) — loss on full sequence.")

sft_config = SFTConfig(
    output_dir=str(ADAPTER_OUT / "checkpoints"),
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    gradient_checkpointing=True,
    learning_rate=LEARNING_RATE,
    lr_scheduler_type="cosine",
    warmup_steps=WARMUP_STEPS,
    bf16=True,
    fp16=False,
    optim="paged_adamw_8bit",
    logging_steps=20,
    save_steps=200,
    save_total_limit=2,
    report_to="none",
    max_seq_length=MAX_SEQ_LENGTH,
    dataset_text_field="text",
    packing=False,
)

trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=dataset,
    tokenizer=tokenizer,
    data_collator=collator if use_collator else None,
)

print(f"Training {len(dataset)} examples, {EPOCHS} epochs, effective batch {BATCH_SIZE * GRAD_ACCUM}")
trainer.train()
print("Training complete.")'''),

md("## 5. Save Adapter"),

code('''ADAPTER_OUT.mkdir(parents=True, exist_ok=True)
model.save_pretrained(str(ADAPTER_OUT))
tokenizer.save_pretrained(str(ADAPTER_OUT))
print(f"Adapter saved: {ADAPTER_OUT}")
print("Next: run 04_grpo_train.ipynb, or set RUN_MERGE=True below for vLLM inference.")'''),

md("## 6. Merge Adapter (set RUN_MERGE=True when done with GRPO, or to skip GRPO)"),

code('''MERGE_OUTPUT = REPO_ROOT / "artifacts" / "models" / "qlora_v1_merged"
RUN_MERGE    = False

if RUN_MERGE:
    merged = model.merge_and_unload()
    MERGE_OUTPUT.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(MERGE_OUTPUT), safe_serialization=True)
    tokenizer.save_pretrained(str(MERGE_OUTPUT))
    print(f"Merged model saved: {MERGE_OUTPUT}")
    print(f"Set MODEL_ID = r\'{MERGE_OUTPUT}\' in 05_private_submission.ipynb")
else:
    print("Merge skipped (RUN_MERGE=False).")'''),

]  # end NB03_CELLS


# ─────────────────────────────────────────────────────────────────────────────
#  NOTEBOOK 04 — GRPO Reinforcement Learning
# ─────────────────────────────────────────────────────────────────────────────

NB04_CELLS = [

md("""# 04 — GRPO Reinforcement Learning

Fine-tune with **Group Relative Policy Optimization** using `judger.auto_judge()`
as a binary verifiable reward — no separate reward model needed.

For each training step: generate G completions per question, score each with the
judger, compute group-normalized advantages, apply policy gradient + KL penalty.

**Prerequisites:** run `02_inference.ipynb` on the full public set, then optionally
`03_qlora_finetune.ipynb`. Set `ADAPTER_PATH` below accordingly.
"""),

code('''import json, os, re, sys, random
from pathlib import Path


def repo_root() -> Path:
    p = Path.cwd().resolve()
    for d in (p, *p.parents):
        if (d / "judger.py").is_file():
            return d
    return p


REPO_ROOT = repo_root()
sys.path.insert(0, str(REPO_ROOT))

MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"
GPU_ID   = "0"

ADAPTER_PATH = REPO_ROOT / "artifacts" / "models" / "qlora_v1"

LEARNING_RATE       = 1e-7
BATCH_SIZE          = 1
GRAD_ACCUM          = 4
EPOCHS              = 2
G                   = 2       # rollouts per question; reduce further if VRAM OOM
BETA                = 0.04
MAX_PROMPT_LENGTH   = 512
MAX_COMPLETION_LEN  = 1024
ROLLOUT_TEMPERATURE = 0.7

LORA_RANK    = 32
LORA_ALPHA   = 64
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

TRAIN_SPLIT = 0.8
RANDOM_SEED = 42

PUBLIC_PATH = REPO_ROOT / "data" / "raw" / "public.jsonl"
GRPO_OUT    = REPO_ROOT / "artifacts" / "models" / "grpo_v1"

os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID

print(f"REPO_ROOT    : {REPO_ROOT}")
print(f"ADAPTER_PATH : {ADAPTER_PATH} | exists: {Path(ADAPTER_PATH).is_dir()}")
print(f"GRPO output  : {GRPO_OUT}")'''),

code('''import subprocess, sys

for dep in ["peft", "trl>=0.12.0", "datasets", "accelerate"]:
    pkg = dep.split(">=")[0].split("==")[0]
    try:
        __import__(pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", dep])'''),

code('''import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from trl import GRPOTrainer, GRPOConfig
from datasets import Dataset
from judger import Judger

judger_obj = Judger(strict_extract=False)

print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU : {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")'''),

md("## 1. Load & Split Data"),

code(SHARED_PROMPTS + '''


tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.pad_token    = tokenizer.eos_token
tokenizer.padding_side = "left"

with open(PUBLIC_PATH, encoding="utf-8") as f:
    all_data = [json.loads(line) for line in f]

rng = random.Random(RANDOM_SEED)
rng.shuffle(all_data)
n_train    = int(len(all_data) * TRAIN_SPLIT)
train_data = all_data[:n_train]
val_data   = all_data[n_train:]
print(f"Total: {len(all_data)} | Train: {len(train_data)} | Val: {len(val_data)}")


def make_dataset_row(item: dict) -> dict:
    """Convert public.jsonl item to GRPOTrainer row with a \'prompt\' column."""
    system, user = build_prompt(item["question"], item.get("options"))
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    try:
        prompt_str = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True
        )
    except TypeError:
        prompt_str = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
    return {
        "prompt":       prompt_str,
        "gold_answer":  json.dumps(item["answer"]),
        "options_json": json.dumps(item.get("options") or []),
        "item_id":      item["id"],
    }


train_dataset = Dataset.from_list([make_dataset_row(it) for it in train_data])
val_dataset   = Dataset.from_list([make_dataset_row(it) for it in val_data])'''),

md("## 2. Reward Function"),

code('''def extract_letter(text: str) -> str:
    m = re.search(r"\\\\boxed\\{([A-Za-z])\\}", text)
    if m:
        return m.group(1).upper()
    matches = re.findall(r"\\b([A-Z])\\b", text.upper())
    return matches[-1] if matches else ""


def math_reward_fn(completions: list, gold_answer: list, options_json: list, **_) -> list:
    """Binary reward: 1.0 correct, 0.0 wrong. Called per micro-batch by GRPOTrainer."""
    rewards = []
    for pred, gold_str, opts_str in zip(completions, gold_answer, options_json):
        try:
            gold = json.loads(gold_str)
            opts = json.loads(opts_str)
            if isinstance(gold, str):
                correct = (extract_letter(pred) == gold.strip().upper())
            else:
                gold_list = gold if isinstance(gold, list) else [gold]
                correct   = judger_obj.auto_judge(
                    pred=pred, gold=gold_list, options=[[]] * len(gold_list)
                )
            rewards.append(1.0 if correct else 0.0)
        except Exception:
            rewards.append(0.0)
    return rewards


_r = math_reward_fn(
    ["<think>\\n1+1=2\\n</think>\\n\\\\boxed{2}", "<think>\\n</think>\\n\\\\boxed{3}"],
    [json.dumps(["2"]), json.dumps(["2"])],
    [json.dumps([]), json.dumps([])],
)
assert _r == [1.0, 0.0], f"Reward sanity check failed: {_r}"
print("Reward function OK.")'''),

md("## 3. Load Model"),

code('''bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

base = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
base.config.use_cache = False

adapter_dir = Path(ADAPTER_PATH)
if adapter_dir.is_dir() and (adapter_dir / "adapter_config.json").is_file():
    model = PeftModel.from_pretrained(base, str(adapter_dir), is_trainable=True)
    print(f"Loaded QLoRA adapter: {adapter_dir}")
else:
    lora_cfg = LoraConfig(
        r=LORA_RANK, lora_alpha=LORA_ALPHA, lora_dropout=0.05,
        target_modules=LORA_TARGETS, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(base, lora_cfg)
    print("No adapter found — starting from base model with fresh LoRA.")

model.print_trainable_parameters()
print(f"VRAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB")'''),

md("## 4. Train"),

code('''grpo_config = GRPOConfig(
    output_dir=str(GRPO_OUT / "checkpoints"),
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LEARNING_RATE,
    lr_scheduler_type="cosine",
    bf16=True,
    fp16=False,
    optim="paged_adamw_8bit",
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,
    report_to="none",
    num_generations=G,
    max_prompt_length=MAX_PROMPT_LENGTH,
    max_completion_length=MAX_COMPLETION_LEN,
    beta=BETA,
    temperature=ROLLOUT_TEMPERATURE,
    top_p=0.95,
    top_k=20,
)

trainer = GRPOTrainer(
    model=model,
    reward_funcs=[math_reward_fn],
    args=grpo_config,
    train_dataset=train_dataset,
    processing_class=tokenizer,
)

print(f"GRPO: {len(train_dataset)} train | G={G} | lr={LEARNING_RATE} | {EPOCHS} epochs")
trainer.train()
print("GRPO training complete.")'''),

md("## 5. Quick Validation"),

code('''EVAL_N = 10
model.eval()

correct = 0
for i, row in enumerate(val_dataset.select(range(min(EVAL_N, len(val_dataset))))):
    inputs = tokenizer(
        row["prompt"], return_tensors="pt", truncation=True, max_length=MAX_PROMPT_LENGTH
    ).to(model.device)
    with torch.no_grad():
        out_ids = model.generate(
            **inputs, max_new_tokens=512, temperature=0.6, do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    pred = tokenizer.decode(out_ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    gold = json.loads(row["gold_answer"])
    opts = json.loads(row["options_json"])
    try:
        if isinstance(gold, str):
            ok = (extract_letter(pred) == gold.strip().upper())
        else:
            gold_list = gold if isinstance(gold, list) else [gold]
            ok = judger_obj.auto_judge(pred=pred, gold=gold_list, options=[[]] * len(gold_list))
    except Exception:
        ok = False
    correct += int(ok)

print(f"Val accuracy (first {EVAL_N}): {correct}/{EVAL_N}")'''),

md("## 6. Save Adapter"),

code('''GRPO_OUT.mkdir(parents=True, exist_ok=True)
model.save_pretrained(str(GRPO_OUT))
tokenizer.save_pretrained(str(GRPO_OUT))
print(f"GRPO adapter saved: {GRPO_OUT}")
print("Set RUN_MERGE=True below, then run 05_private_submission.ipynb.")'''),

md("## 7. Merge Adapter"),

code('''MERGE_OUTPUT = REPO_ROOT / "artifacts" / "models" / "grpo_v1_merged"
RUN_MERGE    = False

if RUN_MERGE:
    merged = model.merge_and_unload()
    MERGE_OUTPUT.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(MERGE_OUTPUT), safe_serialization=True)
    tokenizer.save_pretrained(str(MERGE_OUTPUT))
    print(f"Merged model saved: {MERGE_OUTPUT}")
else:
    print("Merge skipped (RUN_MERGE=False).")'''),

]  # end NB04_CELLS


# ─────────────────────────────────────────────────────────────────────────────
#  Write notebooks
# ─────────────────────────────────────────────────────────────────────────────

NB03_PATH = NB_DIR / "03_qlora_finetune.ipynb"
NB04_PATH = NB_DIR / "04_grpo_train.ipynb"

with open(NB03_PATH, "w", encoding="utf-8") as f:
    json.dump(notebook(NB03_CELLS), f, ensure_ascii=False, indent=1)
print(f"Created: {NB03_PATH}")

with open(NB04_PATH, "w", encoding="utf-8") as f:
    json.dump(notebook(NB04_CELLS), f, ensure_ascii=False, indent=1)
print(f"Created: {NB04_PATH}")
