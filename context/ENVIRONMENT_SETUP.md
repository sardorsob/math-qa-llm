# Environment setup

## Same page: what this project does with the LLM

The starter notebook (`notebooks/02_inference.ipynb`) loads a **pretrained** causal LM from the Hugging Face Hub (**Qwen3-4B-Thinking-2507**) and runs **inference** (generation + scoring on the public set). It does **not** walk through fine-tuning or training a new model from scratch. You *can* add finetuning later, but the competition baseline is: **frozen weights + prompting + decoding**.

## Data you need (only files provided)

These are the **only** competition datasets required:

| File | Role |
|------|------|
| `public.jsonl` | Labels present — local evaluation |
| `private.jsonl` | No labels — build submission |
| `sample_submission.csv` | Example CSV shape |

Keep them under the paths your notebook/config use (e.g. `data/raw/` and `data/external/`). No extra downloads for the dataset itself.

## Python environment

**Option A — venv + pip (simplest)**

```text
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -r requirements.txt
```

On macOS / Linux, activate with `source .venv/bin/activate`.

**Option B — Conda**

```text
conda env create -f environment.yml
conda activate cse151b-math-qa
```

If `file:requirements.txt` fails in your conda version, create an empty env with `python=3.11` and run `pip install -r requirements.txt` inside it.

**Suggested Python:** 3.11 (good compatibility with recent `torch`, `transformers`, and `vllm`).

## Jupyter kernel (so the notebook uses this env)

From the **activated** environment:

```text
python -m ipykernel install --user --name cse151b-math-qa --display-name "Python (cse151b-math-qa)"
```

In VS Code / Cursor: **Select Kernel** → **Python (cse151b-math-qa)**.  
Shell lines like `!source .venv/bin/activate` in a notebook **do not** switch the kernel’s Python; the kernel choice does.

## GPU, PyTorch, vLLM, and Windows

- **NVIDIA GPU + recent driver** is assumed for running the 4B model with quantization in reasonable time.
- **`requirements.txt`** includes `torch` and `vllM`. If `pip` installs a CPU-only `torch` or the wrong CUDA build, install the wheel that matches your system from [PyTorch Get Started](https://pytorch.org/get-started/locally/) first, then re-run `pip install -r requirements.txt` (or install remaining packages).
- **vLLM** is primarily supported on **Linux** (and often **WSL2** with GPU). On **native Windows**, `pip install vllm` may fail; use the notebook’s **Transformers + bitsandbytes** path on a GPU machine, or run on Linux / WSL2 / the course GPU host.

## Do I need to “download the model” or “deploy” it?

- **Download:** The first time you run `AutoModelForCausalLM.from_pretrained(...)` or construct a vLLM `LLM(...)`, libraries **download weights from Hugging Face** automatically (several GB). You do **not** check weights into GitHub.
- **Where it lives:** Default cache is usually under your user profile, e.g. `~/.cache/huggingface/hub` (or `%USERPROFILE%\.cache\huggingface\hub` on Windows). You can point `HF_HOME` / `HUGGINGFACE_HUB_CACHE` elsewhere if disk is tight.
- **Gated / private models:** If a repo ever requires accepting terms, log in once: `huggingface-cli login` (token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)).
- **Deploy:** For the course workflow you typically **run inference locally or on a provided GPU server** — no separate “deployment” step unless you choose to host an API yourself.

## Quick verification

```text
python -c "import torch, transformers, sympy, pandas; print('torch', torch.__version__)"
```

Optional (if vLLM installed):

```text
python -c "import vllm; print('vllm ok')"
```

## Related

- `requirements.txt` — package list  
- `environment.yml` — conda/mamba wrapper  
- `PROJECT_BRIEF.md` / `DATASETS.md` — task and data  
