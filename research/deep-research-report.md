# Single-GPU Pipeline for the Math Reasoning Competition

## Executive summary

The highest-ROI strategy for this competition is not to start by training a custom model from scratch. It is to build a **tight inference-and-evaluation pipeline** around a strong small open-weight math reasoner, align your output format to the starter evaluator, add **adaptive best-of-N** sampling, and only then consider lightweight fine-tuning. The starter repository is explicitly built around notebook-based inference and public-set scoring, and its evaluator is unusually forgiving on symbolic equivalence but very sensitive to final-answer extraction format. That makes **format control, normalization, and candidate selection** disproportionately important. citeturn38view0turn37view0turn13search0turn19search0

If you want one primary submission model on a single 24GB GPU, my first choice is **DeepSeek-R1-0528-Qwen3-8B**. If you want the best efficiency-per-GPU-dollar, my second anchor model is **Phi-4-mini-reasoning**. If you want a tool-first branch for arithmetic-heavy or symbolic questions, the most directly relevant small model is **NuminaMath-7B-TIR**, with **Qwen2.5-Math-7B-Instruct** as the safest mature fallback for CoT/TIR prompting. Those choices are all much closer to a competition-winning pipeline than a generic 7B instruct model. citeturn32view0turn29view2turn29view1turn36view0turn7view1

The practical build order is straightforward. First, reproduce the starter notebook and scorer exactly. Second, implement dataset parsing and strict answer-shape controls around `[ANS]` placeholders and MCQ letters. Third, force every completion to end with a **single final boxed line** that the provided judger can extract cleanly. Fourth, add adaptive self-consistency and answer voting. Fifth, add a small Python/SymPy tool branch for questions where computation is the main bottleneck. Only after that should you spend time on LoRA, reward modeling, or search-heavy tree methods. citeturn38view0turn37view0turn13search0turn15search2turn15search3

## What the starter repo and evaluator imply

The starter repository on entity["company","GitHub","code hosting platform"] is minimal by design. Its `README` says the main entry point is `starter_code_cse151b_comp.ipynb`, and that the notebook covers environment setup, inference with **Qwen3-4B-Thinking (INT8)**, and scoring against the public dataset. The repo contents are intentionally small: the notebook, `judger.py`, `utils.py`, `data/public.jsonl`, and a `results/` directory. That means the competition baseline is already pointing you toward an **inference-first** workflow rather than a large training stack. citeturn38view0

The deepest strategic insight is in `judger.py`. The evaluator normalizes a large amount of LaTeX and punctuation noise, maps variants such as `\dfrac` and `\tfrac` to `\frac`, strips `\left`, `\right`, some text wrappers, and normalizes answer strings before comparison. It also extracts **the last contiguous group of `\boxed{...}` answers**, joins multiple final boxes for multi-answer problems, and only falls back to weaker heuristics such as “last explicit answer,” “last LaTeX formula,” or “last number” when extraction is not clean. For symbolic answers, it uses SymPy-style normalization and random-value checking with `num_samples = 100`, `num_times = 3`, and default precision `1e-8`. In practice, this means that **your prompt should be engineered for the extractor, not for human elegance**. The safest completion is one whose very last line is the complete final answer in boxed form, with exactly the right number of boxes. citeturn37view0

Your competition brief adds the rest of the rules that matter operationally: data arrive as JSONL, free-form questions use `[ANS]` placeholders, multiple-choice questions include `options`, the public set includes answers for development, the private set does not, and the submission must be a CSV with `id,response`, where `response` is the **full raw model trace**, not just the extracted final answer. Given those rules, the correct development protocol is to create your own **public-dev / public-holdout** split inside the public set, tune prompts and budgets on public-dev, and reserve public-holdout for final decision-making before you run the private set. Because the private set is distributionally similar, leakage from repeated prompt tuning on the entire public set is one of the easiest ways to fool yourself.

A robust parser can be very small:

```python
from dataclasses import dataclass

@dataclass
class ProblemSpec:
    id: int
    kind: str          # "mcq" or "freeform"
    n_answers: int
    question: str
    options: list[str] | None

def parse_example(ex: dict) -> ProblemSpec:
    is_mcq = bool(ex.get("options"))
    n_answers = 1 if is_mcq else ex["question"].count("[ANS]")
    return ProblemSpec(
        id=int(ex["id"]),
        kind="mcq" if is_mcq else "freeform",
        n_answers=n_answers,
        question=ex["question"],
        options=ex.get("options"),
    )
```

The CSV writer should be equally strict. Because your `response` field may contain commas, line breaks, and quotes, the safest route is to let the standard `csv` module handle quoting and doubling of embedded quotation marks for you. The Python docs and PEP 305 both describe the relevant behavior: `QUOTE_ALL` forces quoting, and `doublequote=True` writes internal `"` characters as `""`. citeturn39search0turn39search7

```python
import csv

def write_submission(rows, out_path: str) -> None:
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["id", "response"],
            quoting=csv.QUOTE_ALL,
            doublequote=True,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
```

## Model choices for a single 24GB GPU

For this competition, I would separate models into two groups: **submission-ready reasoners** and **fine-tuning bases**. Submission-ready reasoners matter more at the start, because you can get large gains without training. Fine-tuning bases matter later, once your extraction, routing, and sampling are already stable.

| Recommendation | Model | License | Why it is relevant | 24GB fit | Source |
|---|---|---|---|---|---|
| Best overall single-model choice | **DeepSeek-R1-0528-Qwen3-8B** | MIT | Strongest ≤8B-class open-weight option in this set for hard reasoning; official card reports **86.0 on AIME 2024**, ahead of Qwen3-8B and even above Phi-4-reasoning-plus on that benchmark, but with long reasoning traces and higher latency. | Prefer 4-bit or 8-bit for best-of-N; BF16 can work but leaves less KV-cache headroom. | citeturn32view0turn20search0turn21search0 |
| Best efficiency / secondary ensemble model | **Phi-4-mini-reasoning** | MIT | Very strong for its size: model card reports **57.5 AIME** and **94.6 MATH-500** at only **3.8B** parameters, explicitly targeted at memory/compute-constrained environments. | Easiest high-quality model to run comfortably on 24GB; good for parallel candidate generation. | citeturn29view2turn20search0 |
| Strong mature baseline | **DeepSeek-R1-Distill-Qwen-7B** | Apache 2.0 | Still one of the most reliable public 7B-class reasoning distills; model card reports **55.5 AIME 2024** and **92.8 MATH-500**. | 4-bit and 8-bit both practical; good “safe baseline” if the 0528 model is too verbose. | citeturn7view1turn20search0 |
| Best tool-first small solver | **NuminaMath-7B-TIR** | Apache 2.0 | Explicitly trained for tool-integrated reasoning; model card says it won the first AIMO progress prize and reports **68.1 MATH** and **10/30 AIME 2024 maj@64** among 7B-class models. | Best as a routed branch, not your only model. | citeturn29view1turn28search2 |
| Best mature math-focused checkpoint for further adaptation | **Qwen2.5-Math-7B-Instruct** | Apache 2.0 | Supports both CoT and TIR; official card reports **85.3 on MATH using TIR** and gives directly relevant boxed-answer prompting examples. | Excellent QLoRA target if you later format-match to the competition. | citeturn36view0turn20search0 |
| Best open math SFT alternative if you want a more “classical” base | **OpenMath2-Llama3.1-8B** | Llama 3.1 community license | Strong open 8B math specialist; model card reports **67.8 MATH** and **76.1 majority@256**, clearly above vanilla Llama-3.1-8B-Instruct on math. | Feasible in 4-bit; attractive if you want less verbose traces than R1-style models. | citeturn7view3turn20search0 |
| Strong but noncommercial-only option | **AceMath-7B-Instruct** | CC BY-NC 4.0 | Official card says it improves over Qwen2.5-Math-7B-Instruct on average pass@1 and comes close to the much larger 72B version. | Worth testing only if the competition rules allow noncommercial weights. | citeturn34view0turn34view1 |
| General-purpose current fallback | **Qwen3-8B** | Apache 2.0 | Not math-specific, but the current generation supports an explicit “thinking mode,” provides official math prompting recommendations, and avoids the age of older 7B instruct baselines. | Good all-around fallback; less specialized than the math-tuned options above. | citeturn30view1turn30view2turn30view3 |

The two most useful “big-model ideas distilled into small models” are already available to you. One is the **DeepSeek** distillation line, where the model card explicitly argues that smaller dense models can inherit strong reasoning behavior from a much larger teacher. The other is **Phi-4-mini-reasoning**, which shows that careful reasoning-focused distillation can make a 3.8B model surprisingly competitive on math. On a single 24GB GPU, those two facts matter more than chasing a 70B checkpoint you cannot comfortably serve. citeturn7view1turn29view2turn32view0

If you later decide to fine-tune, use external math corpora that already match reasoning style rather than overfitting the public split. The most useful public sources for that are **OpenMathInstruct-2**, **NuminaMath-CoT**, **NuminaMath-TIR**, **MetaMathQA**, and **MathInstruct**. Together they cover broad competition-style questions, CoT traces, and tool-integrated traces, which is exactly the mixture your competition format rewards. citeturn7view3turn28search3turn28search2turn24search6turn24search7

Quantization is essential to making these choices practical. The easiest experimentation path is **bitsandbytes 4-bit NF4** or 8-bit loading; the strongest serving-oriented option is often **AWQ**. The underlying tradeoff is straightforward: BF16 preserves quality best, 8-bit is the safest compression, and 4-bit is the sweet spot for a 24GB card when you want best-of-N, bigger context, or a secondary verifier/reranker. For serving multiple requests with shared prompt prefixes, **vLLM** and **SGLang** matter because they directly attack KV-cache inefficiency and prefix reuse. citeturn20search0turn20search4turn21search0turn22search0turn21search2

## Prompting, sampling, extraction, and normalization

The prompt strategy should be built around one fact: the provided scorer is optimized to find a **clean final answer segment**, not to read a pretty solution. Your prompts should therefore enforce two simultaneously true properties. First, the model is allowed to reason in detail because your submission must keep the raw trace. Second, the model must end with a deterministic, extractor-friendly final line. The strongest general pattern is: **reason however you like, but terminate with exactly the required number of boxed answers and then stop**. This is also aligned with current official boxed-answer recommendations in the Qwen3, Qwen2.5-Math, and OpenMath2 model materials. citeturn37view0turn30view1turn36view0turn7view3

A strong generic template for free-form questions is:

```text
SYSTEM:
You are solving competition math problems.
Reason step by step.
Count the number of [ANS] placeholders in the question.

Rules:
- If there is 1 [ANS], end with exactly one boxed final answer.
- If there are k [ANS] placeholders, end with exactly k boxed final answers in the same order as the placeholders.
- Do not put any extra boxes after the final answer line.
- If you use intermediate calculations, keep them above the final line.

Required final format:
FINAL:
\boxed{answer_1}
or
FINAL:
\boxed{answer_1}, \boxed{answer_2}, ..., \boxed{answer_k}

USER:
{question}
```

A strong generic template for multiple-choice questions is:

```text
SYSTEM:
You are solving a multiple-choice math problem.
Reason step by step and eliminate wrong options if helpful.
Your final line must contain only one boxed capital letter from the available choices.

Required final format:
FINAL:
\boxed{C}

USER:
Question:
{question}

Options:
A. {option_a}
B. {option_b}
C. {option_c}
D. {option_d}
...
```

For models that were trained with explicit reasoning tags or two-section outputs, keep that native structure, but still **append the boxed final line last**. For example, Phi-4-reasoning-plus explicitly expects a thought section and solution section, while DeepSeek-R1-0528 and Qwen3 both publish sampling guidance tailored to reasoning mode. In all cases, the last line matters more than the exact scratchpad style. citeturn33view0turn32view0turn30view1

The normalization rules should mimic the starter judger rather than invent a clean-room evaluator. The most important ones are these: count `[ANS]` to determine expected answer arity; for multiple answers, split only on commas **outside** brackets or parentheses; uppercase single-letter MCQ answers; normalize LaTeX wrappers and common aliases such as `\dfrac` → `\frac`; strip spurious punctuation; and treat algebraically equivalent expressions as correct whenever SymPy says the normalized difference is zero or numerically indistinguishable. The reason to mirror the starter logic is simple: if your local selection logic differs from the official scorer, you optimize for the wrong objective. citeturn37view0turn23search1turn23search3turn23search5turn23search11

The decoding policy should be **adaptive**, not fixed. Self-consistency is one of the most robust improvements in reasoning tasks, and more recent work on inference-time compute shows that adaptive allocation beats naive constant-budget sampling. Combined with the current model cards, that leads to a practical policy: use one sample first; if answer extraction fails, candidate answers disagree, or a verifier flags low confidence, escalate to `N=4`; only use `N=8` or higher on hard cases. For Qwen3 thinking mode, the official guidance specifically says not to use greedy decoding and recommends around `temperature=0.6`, `top_p=0.95`, and `top_k=20`. DeepSeek-R1-0528 uses `temperature=0.6` in its own recommendations. Phi-4-reasoning-plus recommends a hotter sampled regime: `temperature=0.8`, `top_k=50`, `top_p=0.95`, `do_sample=True`. citeturn13search0turn19search0turn30view1turn32view0turn33view0

A practical policy table is:

| Setting | Free-form | Multiple-choice | Why |
|---|---|---|---|
| First pass | 1 sample | 1 sample | Cheap baseline; many easy public items will already solve. |
| Escalation trigger | parse failure, malformed box count, disagreement, or tool-eligible question | disagreement or weak rationale consistency | Drives compute only when likely useful. citeturn19search0turn37view0 |
| Second pass | 4 samples, majority vote on normalized answer tuple | 4 samples, majority vote on uppercase letter | Standard self-consistency regime. citeturn13search0 |
| Hard-case pass | 8–16 samples, optional verifier rerank | 8 samples if top-2 letters tie | Worth it only on the subset of hard problems. citeturn19search0 |
| Decoding defaults | Qwen/DeepSeek-style: `T≈0.6`, `top_p≈0.95` | same, but cap max tokens more tightly | Matches current reasoning-model guidance. citeturn30view1turn32view0 |

For answer selection, I would use **answer-level voting first, solution-level reranking second**. In other words, normalize every candidate to the exact answer tuple you think the official scorer will see, vote on that normalized tuple, and only use a reranker when the vote is inconclusive. This is cheaper and usually more robust than picking the most eloquent full solution trace.

## Tool-augmented solving and safe verification

Tool use is worth it here, but only if you keep it tight. The most relevant papers and model cards all point the same way: separating reasoning from computation helps on numerical and symbolic tasks, and explicitly tool-integrated math models can outperform plain CoT on hard arithmetic, algebraic manipulation, and intermediate-program problems. That is the core idea behind **Program of Thoughts**, **ToRA**, **Qwen2.5-Math TIR**, and **NuminaMath-TIR**. citeturn15search3turn15search2turn36view0turn29view1

The right implementation on one 24GB GPU is **not** a free-form agent loop. It is a narrow branch with a restricted Python sandbox for only the question classes that benefit the most: arithmetic-heavy free-form tasks, equation solving, polynomial manipulation, simple combinatorics, matrix operations, and expression simplification. I would not use the tool branch on every question, because many geometry or pure proof-style items are bottlenecked by reasoning, not arithmetic, and a tool loop adds latency.

A good router is heuristic and cheap: if the question contains many numerals, explicit expressions, words like “evaluate,” “solve the system,” “roots,” “determinant,” “eigenvalues,” or if the model itself emits executable intermediate code, send it to the tool branch. Otherwise keep it in plain reasoning mode. That router can be upgraded later, but it is plenty good enough for a first competitive system.

The safe verification stack should be conservative:

1. **No network and no filesystem writes** inside the tool sandbox.
2. **Short timeouts** on every execution.
3. **Whitelisted imports only**, ideally `math`, `fractions`, and `sympy`.
4. **Intermediate result checks** only; the final answer still comes from the language model trace.
5. **Post-hoc normalization** of the final answer with the same parser you use for voting.
6. **Fallback to no-tool mode** when code errors or times out.

For symbolic verification, use SymPy carefully. The official SymPy docs are clear that `simplify()` is broad but not perfectly well-defined, and that symbolic equality is not the same thing as Python assignment or structural equality. The Hugging Face evaluation guidebook also notes a practical problem: math parsing with SymPy and LaTeX is not perfect, even on benchmark ground truths. So the right architecture is **tiered verification**: first try clean string normalization, then parse-latex equivalence when the expression is simple, then optional numeric substitution checks, and finally fall back to human-readable normalized strings if parsing fails. citeturn23search1turn23search3turn23search5turn23search11turn23search2turn37view0

A tiny verifier skeleton looks like this:

```python
import re
from sympy import simplify
from sympy.parsing.latex import parse_latex

def normalize_answer_text(s: str) -> str:
    s = s.strip()
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.strip("$ ").strip()
    return s

def symbolic_equiv(a: str, b: str) -> bool:
    a = normalize_answer_text(a)
    b = normalize_answer_text(b)
    if a == b:
        return True
    try:
        return simplify(parse_latex(a) - parse_latex(b)) == 0
    except Exception:
        return False
```

This should not replace the official scorer. It should only help you choose among candidate generations and catch obvious formatting failures before you write the final CSV.

## Reference pipeline and implementation plan

```mermaid
flowchart TD
    A[Load JSONL problem] --> B[Detect format]
    B -->|MCQ| C[MCQ prompt builder]
    B -->|Free-form| D[Count [ANS] placeholders]
    D --> E[Free-form prompt builder]

    C --> F[Primary model generation]
    E --> F

    F --> G[Extract final boxed answer(s)]
    G --> H[Normalize answer tuple]

    F --> I{Tool-eligible?}
    I -->|Yes| J[Python/SymPy tool branch]
    I -->|No| K[No-tool branch]

    J --> L[Tool-assisted candidate generation]
    K --> M[Plain CoT candidate generation]

    L --> N[Candidate pool]
    M --> N

    N --> O[Adaptive best-of-N sampling]
    O --> P[Answer-level vote]
    P --> Q{Confident?}
    Q -->|Yes| R[Keep full winning response trace]
    Q -->|No| S[Verifier / reranker / escalate N]

    S --> T[Select final response]
    R --> U[CSV writer]
    T --> U

    U --> V[id,response submission.csv]
```

The implementation order should be ruthless about ROI:

1. **Reproduce the starter notebook and scorer** exactly, using the public split as your first regression target. Do not touch the modeling stack until you can round-trip the repo and understand where the scorer is permissive versus brittle. citeturn38view0turn37view0

2. **Implement parsing and answer-shape control**. Detect MCQ versus free-form, count `[ANS]`, and reject or regenerate samples whose final boxed answer count does not match the expected count. This is the easiest accuracy gain in the whole project.

3. **Implement strict final-answer prompting** with boxed answers last. The starter judger strongly rewards clean final boxes, and current math model cards explicitly recommend boxed formatting. citeturn37view0turn30view1turn36view0turn7view3

4. **Benchmark three models only** on your public-dev split: DeepSeek-R1-0528-Qwen3-8B, Phi-4-mini-reasoning, and Qwen2.5-Math-7B-Instruct or NuminaMath-7B-TIR. If you test ten models up front, you waste time comparing mediocre pipelines rather than strong ones. citeturn32view0turn29view2turn36view0turn29view1

5. **Add adaptive self-consistency**. Start with answer voting before any reranker. Most of the benefit arrives there, and the implementation burden is tiny compared with verifier training or tree search. citeturn13search0turn19search0

6. **Add a narrow tool branch** with a restricted Python sandbox and SymPy verification. Route only the questions that plausibly benefit from execution. citeturn15search2turn15search3turn29view1turn36view0

7. **Only then consider QLoRA / SFT**, and only on external public math corpora or format-matching synthetic data. Do not make public-split overfitting your first move. Recent small-model reasoning progress has come from careful post-training and inference-time scaling, not from naive benchmark-specific tuning. citeturn7view1turn29view2turn25search1turn24search2turn24search3

If you want a concise “implement first” list, it is this: **clean boxed outputs**, **multi-answer counting**, **adaptive best-of-N**, **tool branch**, **only then fine-tuning**.

## Compute budget and technique ranking

A 24GB GPU is enough, but only if you design around it. The safest mental model is this: **BF16 preserves quality, 4-bit buys you budget**. A 3.8B model is easy; a 7–8B model is comfortable in 4-bit and often manageable in 8-bit; a 14B model is plausible only with quantization and tighter batch/context limits. That follows directly from model size and the documented 8-bit/4-bit quantization paths in bitsandbytes and AWQ. citeturn29view2turn30view3turn32view0turn20search0turn20search4turn21search0

A practical rule-of-thumb table is:

| Model class | BF16 weights | 8-bit weights | 4-bit weights | Practical implication on 24GB |
|---|---:|---:|---:|---|
| 3.8B | ~7.6GB | ~3.8GB | ~1.9GB raw | Easy headroom for long outputs or parallel candidates |
| 7–8B | ~14–16GB | ~7–8GB | ~3.5–4GB raw | Sweet spot for competition inference |
| 14B | ~28GB | ~14GB | ~7GB raw | Needs 4-bit and disciplined batching / KV usage |

These are weight-only approximations. Real serving requires extra space for KV cache, runtime overhead, activations, and batch scheduling. That is exactly why **vLLM** and **SGLang** matter: they reduce KV waste and exploit shared prefixes, which is especially useful when you are running many candidates from the same prompt template. citeturn22search0turn21search2turn20search0turn21search0

The highest-value techniques rank roughly as follows:

| Technique | Expected accuracy gain | Implementation difficulty | Compute cost | Recommendation |
|---|---|---|---|---|
| Strong small reasoning checkpoint | High | Low | Medium | Implement first |
| Strict boxed final answers + answer-count control | High | Low | Low | Implement first |
| Adaptive self-consistency / best-of-N | High | Low–Medium | Medium–High | Implement immediately after baseline |
| Tool branch for computation-heavy items | Medium–High | Medium | Medium | Implement after voting |
| Cross-model ensemble | Medium | Medium | High | Add only if public-holdout justifies it |
| QLoRA on external math corpora | Medium | Medium–High | Medium | Good later-stage addition |
| Outcome reward model / PRM reranking | Medium | High | Medium–High | Useful, but lower ROI than voting early on |
| Tree search / MCTS / PRM-guided search | Potentially high | Very high | Very high | Deprioritize on one 24GB GPU |

This ranking is consistent with the literature. Self-consistency gives large gains for reasoning; inference-time compute helps when allocated adaptively; tool-integrated reasoning helps when computation is the bottleneck; and heavy tree-search or process-reward machinery can work, but they are much less attractive if your actual constraint is one consumer-grade GPU and limited engineering time. citeturn13search0turn19search0turn15search2turn15search3turn13search2turn19search2

The important negative recommendation is just as strong as the positive one: **do not begin with multi-GPU serving, giant 70B checkpoints, or elaborate MCTS frameworks**. Those ideas are interesting academically, but they are not compute-optimal under your stated constraints.

## Git workflow and repo setup

If your goal is simply to work independently while preserving an easy path to upstream updates, a **fork** is the safest default. The official GitHub docs describe a fork as a new repository that shares the original repo’s code and visibility settings, and they explicitly document adding an `upstream` remote so you can sync changes from the original repository into your fork. citeturn10search2turn10search1turn10search0turn10search6

If you are happy to keep the repo public or the competition allows fork-based work, use:

```bash
gh repo fork brooksniu/151B_SP26_Competition --clone=true --remote=true
cd 151B_SP26_Competition

# verify remotes
git remote -v

# if needed, ensure upstream exists
git remote add upstream https://github.com/brooksniu/151B_SP26_Competition.git

# pull future upstream updates
git fetch upstream
git checkout main
git merge upstream/main
```

If you want a **fresh private repo** that is not technically a GitHub fork, you can still preserve upstream cleanly by cloning the starter repo, renaming its original remote to `upstream`, and then attaching your own private repository as `origin`. This is a perfectly valid Git workflow; it just does not get GitHub’s native “fork” UI and sync mechanics. The GitHub docs for remotes and forks, together with the Git book’s remote commands, cover exactly this pattern. citeturn10search1turn10search14

```bash
git clone https://github.com/brooksniu/151B_SP26_Competition.git my-math-reasoning-competition
cd my-math-reasoning-competition

# rename original remote to upstream
git remote rename origin upstream

# create your own empty private repo first, then:
git remote add origin git@github.com:YOUR-USERNAME/my-math-reasoning-competition.git

# push your working copy
git push -u origin main
```

My recommendation is simple. If you care about **easy upstream syncing**, use a fork. If you care about **privacy or cleaner ownership**, create a fresh private repository but keep `upstream` pointing at the starter repo. In both cases, preserve upstream from day one. That prevents painful repository surgery later and makes starter updates trivial to absorb.