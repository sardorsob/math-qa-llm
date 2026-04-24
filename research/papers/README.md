# Reading list — math reasoning pipeline (CSE 151B SP26)

> **Why this file exists instead of PDFs.** The Cowork sandbox where this was prepared has an outbound-network allowlist that blocks `arxiv.org` and `huggingface.co`, so the PDFs could not be fetched here. Everything below is a curated, annotated list of the papers referenced (explicitly or implicitly) in `../deep-research-report.md`, with direct arXiv links. On your own machine, either click each link, or run the helper script in this folder:
>
> ```bash
> bash fetch_papers.sh          # downloads every PDF below into ./pdfs/
> ```
>
> The list is ordered the way it should be *read* for this competition: highest-leverage ideas first, then models, then training data, then systems/efficiency background.

Each entry has:
- **Paper** — title, authors (abbreviated), arXiv ID
- **Why it matters here** — specifically for your Qwen3-4B-Thinking inference pipeline / potential finetune
- **Read for** — the sections worth your time; skip everything else

---

## Tier 1 — Read these first (inference-time techniques that move the score)

These are the highest-ROI papers for your pipeline *today*, before any finetuning. They directly justify the "adaptive best-of-N + strict boxed output + narrow tool branch" recipe in the deep research report.

### 1. Chain-of-Thought Prompting Elicits Reasoning in LLMs
- **Authors:** Wei et al., Google Brain, 2022
- **arXiv:** [2201.11903](https://arxiv.org/abs/2201.11903) · PDF: <https://arxiv.org/pdf/2201.11903>
- **Why it matters here:** The foundational result that simply asking a model to "think step by step" unlocks reasoning gains. Every reasoning model you're choosing between (Qwen3-Thinking, DeepSeek-R1, Phi-4-reasoning) is a CoT descendant. Sets the baseline you're trying to beat with sampling + tools.
- **Read for:** §3 (prompting format), §5 (results on GSM8K/MATH-style benchmarks), §6 (robustness to prompt variations).

### 2. Self-Consistency Improves Chain-of-Thought Reasoning
- **Authors:** Wang et al., Google Research, 2022
- **arXiv:** [2203.11171](https://arxiv.org/abs/2203.11171) · PDF: <https://arxiv.org/pdf/2203.11171>
- **Why it matters here:** The paper behind the "majority-vote on normalized answers" step in your pipeline. Shows that sampling N CoT completions and voting on the *final answer* (not the full reasoning trace) beats greedy decoding by a large margin on math benchmarks — often as much as a 10–20 point lift on GSM8K-class tasks. This is the single most important inference-time trick you will implement.
- **Read for:** §2.1 (the voting procedure — implement exactly this), §3 (sample-size vs. accuracy curves; motivates your adaptive N schedule), §4.4 (robustness to decoding temperature).

### 3. Scaling LLM Test-Time Compute Optimally Can Be More Effective Than Scaling Model Parameters
- **Authors:** Snell, Lee, Xu, Kumar (Google DeepMind / UC Berkeley), 2024
- **arXiv:** [2408.03314](https://arxiv.org/abs/2408.03314) · PDF: <https://arxiv.org/pdf/2408.03314>
- **Why it matters here:** The formal argument for the "adaptive, not fixed N" rule in your deep-research report. Shows that allocating compute *per problem* (easier problems get fewer samples, harder ones get more) Pareto-dominates fixed best-of-N at equal total budget. Directly supports the tiered `N=1 → N=4 → N=8` escalation in your sampling policy.
- **Read for:** §2 (compute-optimal policies), §4 (difficulty-aware allocation), §5 (verifier-vs-sampling trade-off — relevant if you consider a reranker).

### 4. Program of Thoughts (PoT) Prompting
- **Authors:** Chen, Ma, Wang, Cohen (2022)
- **arXiv:** [2211.12588](https://arxiv.org/abs/2211.12588) · PDF: <https://arxiv.org/pdf/2211.12588>
- **Why it matters here:** Justifies the "narrow Python/SymPy tool branch" recommendation. PoT decouples reasoning (the LLM writes a program) from computation (the program runs), and shows consistent gains on arithmetic-heavy MATH / AQuA problems. Your router from the deep-research report ("numeric-heavy → tool branch") is a direct application.
- **Read for:** §3 (the prompting template — reuse near-verbatim), §4 on GSM8K/SVAMP/MultiArith (the gains are on exactly the problem classes your router should catch).

### 5. ToRA: Tool-Integrated Reasoning Agent for Mathematical Problem Solving
- **Authors:** Gou, Shao et al., Microsoft Research / Tsinghua, 2024
- **arXiv:** [2309.17452](https://arxiv.org/abs/2309.17452) · PDF: <https://arxiv.org/pdf/2309.17452>
- **Why it matters here:** Tightens PoT into an interleaved "think → call Python → observe → think" loop, and releases the ToRA models (7B, 13B, 70B). This is the template Qwen2.5-Math-TIR and NuminaMath-TIR both descend from. Gives you the concrete interleaving format and output-parsing scheme for your tool branch.
- **Read for:** §3 (interleaved format and stop-token handling), §4.3 (ablations showing where tool use hurts — skip the tool branch on those question classes), App. for the `<code>...</code>` fencing convention.

### 6. Let's Verify Step by Step (Process Reward Models, PRM800K)
- **Authors:** Lightman, Kosaraju, Burda et al., OpenAI, 2023
- **arXiv:** [2305.20050](https://arxiv.org/abs/2305.20050) · PDF: <https://arxiv.org/pdf/2305.20050>
- **Why it matters here:** The canonical reference for **process reward models** vs. outcome reward models. Your deep-research report ranks PRM reranking below answer-voting for this competition, but if you later want to squeeze more out of best-of-N, this is the paper that tells you exactly how to build a step-level verifier and the trade-offs vs. ORM.
- **Read for:** §3 (PRM training objective), §4.3 (why process supervision wins on MATH), §6 (weak-to-strong generalization — useful if you want to bootstrap a verifier without human annotation).

---

## Tier 2 — Model / checkpoint papers (decide which base to run or finetune)

These map to the "submission-ready reasoners" table in the deep-research report. Read the relevant one(s) for whichever base model you pick.

### 7. DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning
- **Authors:** DeepSeek-AI, 2025
- **arXiv:** [2501.12948](https://arxiv.org/abs/2501.12948) · PDF: <https://arxiv.org/pdf/2501.12948>
- **Why it matters here:** The paper behind DeepSeek-R1-Distill-Qwen-7B and DeepSeek-R1-0528-Qwen3-8B — both top-table candidates for your pipeline. Shows the GRPO-based RL recipe that produces long thinking traces, and documents the distillation pipeline that yields the small checkpoints you'd actually run on 24 GB.
- **Read for:** §2.2 (GRPO), §3 (distillation recipe — the template for the 7B/8B models you'd serve), §4 (benchmarks — gives you the "AIME 86.0" numbers in the deep-research report), Appendix on sampling guidance (`T=0.6`).

### 8. Qwen2.5-Math Technical Report: Toward Mathematical Expert Model via Self-Improvement
- **Authors:** An Yang, Beichen Zhang et al., Alibaba Qwen Team, 2024
- **arXiv:** [2409.12122](https://arxiv.org/abs/2409.12122) · PDF: <https://arxiv.org/pdf/2409.12122>
- **Why it matters here:** Documents Qwen2.5-Math-7B-Instruct — your "safest mature fallback." Contains the official **boxed-answer prompting examples** you should mirror for format compliance with the starter judger. Also describes their CoT-vs-TIR split, which is the same dichotomy you'll route with.
- **Read for:** §3 (CoT + TIR prompt templates — copy these), §4 (self-improvement loop via reward-model sampling — sketches a SFT curriculum if you later go that route), benchmark tables.

### 9. Qwen3 Technical Report
- **Authors:** Qwen Team, 2025
- **arXiv:** [2505.09388](https://arxiv.org/abs/2505.09388) · PDF: <https://arxiv.org/pdf/2505.09388>
- **Why it matters here:** Documents the Qwen3 series including the "Thinking" variants. The starter notebook uses **Qwen3-4B-Thinking** — this is *your* model's technical report. Contains the official decoding recommendations (`T=0.6, top_p=0.95, top_k=20`) cited in the deep-research report, plus the thinking/non-thinking mode toggle you'll use in prompts.
- **Read for:** Sections on Thinking mode, decoding guidance, and long-context handling. The MoE / 235B sections are not relevant for your 24 GB setup — skip.

### 10. Phi-4-Mini-Reasoning Technical Report
- **Authors:** Microsoft, 2025
- **arXiv:** [2504.21233](https://arxiv.org/abs/2504.21233) · PDF: <https://arxiv.org/pdf/2504.21233>
- **Why it matters here:** Your "best efficiency / secondary ensemble model." 3.8 B parameters, 57.5 AIME / 94.6 MATH-500, MIT license, fits BF16 comfortably on 24 GB with room for best-of-N. If you cross-model ensemble, this is probably the second model you'd pair with DeepSeek-R1-Qwen-8B.
- **Read for:** §3 (distillation curriculum), §4 (MATH / AIME numbers for calibration vs. your public-set scores), prompting appendix.

### 11. Phi-4-Reasoning Technical Report (same release family, Phi-4-reasoning-plus)
- **Authors:** Microsoft, 2025
- **arXiv:** [2504.21318](https://arxiv.org/abs/2504.21318) · PDF: <https://arxiv.org/pdf/2504.21318>
- **Why it matters here:** The larger (14 B) sibling. Relevant because the paper documents the explicit "thought section + solution section" output structure you need to respect when prompting — which affects how you splice a final `\boxed{…}` line onto the end.
- **Read for:** §3.2 (output format), decoding recommendations (`T=0.8, top_k=50, top_p=0.95`).

### 12. NuminaMath / NuminaMath-TIR (AIMO Progress Prize writeup)
- **Paper:** "NuminaMath: The largest public dataset in AI4Maths with 860k pairs of competition math problems and solutions," Jia Li, Edward Beeching, Lewis Tunstall et al., 2024
- **Link:** <https://github.com/project-numina/aimo-progress-prize/blob/main/report/numina_dataset.pdf>  (canonical writeup; no arXiv ID)
- **HF page:** <https://huggingface.co/AI-MO/NuminaMath-7B-TIR>
- **Why it matters here:** Your "best tool-first small solver." 7B, trained on tool-integrated competition math traces, won the first AIMO Progress Prize. Use this model (and/or the dataset) whenever your router sends a problem to the tool branch. Documents the exact `<code>`/output fencing you need to parse.
- **Read for:** Dataset composition (what kinds of problems tool-style prompting handles well), training recipe (SFT on tool traces), and the inference format.

---

## Tier 3 — Training data papers (read only if you go past baseline and do SFT/QLoRA)

If the deep-research report's recommendation holds and you *don't* finetune until late, these are for the final phase.

### 13. MetaMath: Bootstrap Your Own Mathematical Questions for Large Language Models
- **Authors:** Yu, Jiang, Shi, Yu et al., 2023
- **arXiv:** [2309.12284](https://arxiv.org/abs/2309.12284) · PDF: <https://arxiv.org/pdf/2309.12284>
- **Why it matters here:** Introduces MetaMathQA, one of the key augmentation datasets cited in the deep-research report. Their rewording / backward-reasoning augmentations are cheap and well-studied — a good source of extra SFT data that won't overfit the public split.
- **Read for:** §3 (the five augmentation operators — rephrase, FOBAR, self-verification, etc.), §4.3 (ablations showing which augmentations help which benchmark).

### 14. MAmmoTH: Building Math Generalist Models through Hybrid Instruction Tuning (MathInstruct)
- **Authors:** Yue, Qu, Zhang et al., 2023
- **arXiv:** [2309.05653](https://arxiv.org/abs/2309.05653) · PDF: <https://arxiv.org/pdf/2309.05653>
- **Why it matters here:** Introduces MathInstruct, a hybrid CoT+PoT instruction dataset. The "hybrid" aspect is exactly what you want for the competition because some problems need tool traces and others are pure reasoning — training on a hybrid mix roughly mirrors your inference-time routing.
- **Read for:** §3 (dataset construction), §5.1 (CoT vs. PoT ablations).

### 15. OpenMathInstruct-2: Accelerating AI for Math with Massive Open-Source Instruction Data
- **Authors:** NVIDIA (Toshniwal et al.), 2024
- **arXiv:** [2410.01560](https://arxiv.org/abs/2410.01560) · PDF: <https://arxiv.org/pdf/2410.01560>
- **Why it matters here:** Produces OpenMath2-Llama3.1-8B, a strong open math SFT checkpoint cited in the deep-research report. The 14M-example dataset is a large, permissively-licensed SFT corpus — if you do QLoRA, this is the cleanest modern source.
- **Read for:** §3 (data-generation pipeline using Llama-3.1-405B as teacher), §4 (scaling-law-style plots showing how much data is "enough"; saves you from over-training).

---

## Tier 4 — Systems & efficiency (needed to actually fit this on one 24 GB GPU)

Skim these; don't read cover-to-cover. You want working knowledge, not depth.

### 16. LoRA: Low-Rank Adaptation of Large Language Models
- **Authors:** Hu et al., Microsoft, 2021
- **arXiv:** [2106.09685](https://arxiv.org/abs/2106.09685) · PDF: <https://arxiv.org/pdf/2106.09685>
- **Why it matters here:** The parameter-efficient-fine-tuning backbone. If/when you finetune, you'll almost certainly do it via LoRA adapters.
- **Read for:** §4 (rank and target-module choices — `q_proj`, `v_proj` is the standard starting point; rank 16–32 is usually plenty).

### 17. QLoRA: Efficient Finetuning of Quantized LLMs
- **Authors:** Dettmers, Pagnoni, Holtzman, Zettlemoyer, 2023
- **arXiv:** [2305.14314](https://arxiv.org/abs/2305.14314) · PDF: <https://arxiv.org/pdf/2305.14314>
- **Why it matters here:** The combination the deep-research report literally names ("QLoRA target if you later format-match"). Makes it feasible to fine-tune 7–8 B models on a single 24 GB GPU with negligible quality loss vs. full-precision LoRA.
- **Read for:** §3 (NF4 quantization — you need to know what `load_in_4bit=True, bnb_4bit_quant_type="nf4"` actually does), §4 (double-quantization and memory budget).

### 18. LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale (bitsandbytes)
- **Authors:** Dettmers et al., 2022
- **arXiv:** [2208.07339](https://arxiv.org/abs/2208.07339) · PDF: <https://arxiv.org/pdf/2208.07339>
- **Why it matters here:** The 8-bit path in your quantization table. Read alongside QLoRA to understand *why* 8-bit is usually safer quality-wise than 4-bit, even though 4-bit buys more budget.
- **Read for:** §3 (outlier features — why naive int8 was broken before this paper), §4 (perplexity vs. bits trade-off).

### 19. AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration
- **Authors:** Lin et al., MIT / NVIDIA, 2023
- **arXiv:** [2306.00978](https://arxiv.org/abs/2306.00978) · PDF: <https://arxiv.org/pdf/2306.00978>
- **Why it matters here:** The deep-research report names AWQ as "the strongest serving-oriented option" — meaning: if you use vLLM with a quantized model, AWQ is usually the path that preserves quality best at inference speed. Short paper, quick read.
- **Read for:** §3 (the "salient weight" argument), §5 (vs. GPTQ and RTN).

### 20. Efficient Memory Management for LLM Serving with PagedAttention (vLLM)
- **Authors:** Kwon, Li, Zhuang et al., UC Berkeley, 2023
- **arXiv:** [2309.06180](https://arxiv.org/abs/2309.06180) · PDF: <https://arxiv.org/pdf/2309.06180>
- **Why it matters here:** Why you should serve via vLLM instead of raw `transformers.generate()` once you start doing best-of-N. Paged KV cache is the reason vLLM can batch 8 candidates from one prompt without OOM'ing on 24 GB.
- **Read for:** §3 (PagedAttention mechanics), §4.1 (prefix sharing — directly relevant because all your best-of-N candidates share the same long prompt prefix).

---

## Optional — deeper dives if you still have time after everything above

- **Chain-of-Thought Hub** — a useful benchmark-of-benchmarks for reasoning: [2305.17306](https://arxiv.org/abs/2305.17306)
- **Tree of Thoughts** — search over reasoning traces; worth knowing even if deprioritized for this project: [2305.10601](https://arxiv.org/abs/2305.10601)
- **MATH dataset paper** (Hendrycks et al., the benchmark almost everything is measured against): [2103.03874](https://arxiv.org/abs/2103.03874)
- **GSM8K paper** (Cobbe et al., the other canonical benchmark): [2110.14168](https://arxiv.org/abs/2110.14168)
- **Solving Math Word Problems with Process- and Outcome-based Feedback** — DeepMind's earlier companion to *Let's Verify Step by Step*: [2211.14275](https://arxiv.org/abs/2211.14275)

---

## Suggested reading order if time-constrained

If you only have an afternoon, read these four in order:

1. **Self-Consistency** (Wang 2022) — the best-of-N foundation.
2. **Scaling Test-Time Compute** (Snell 2024) — why adaptive > fixed N.
3. **Qwen3 Tech Report** (2025) — your actual base model's documentation.
4. **ToRA** (Gou 2024) — the tool-branch template.

Everything else is support for these four.
