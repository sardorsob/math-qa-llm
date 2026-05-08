# Environment setup

## Same page: what this project does with the LLM

The starter notebook (`notebooks/02_inference.ipynb`) loads the competition-required **pretrained** causal LM from the Hugging Face Hub (**`Qwen/Qwen3-4B-Thinking-2507`**) and runs **inference** (generation + scoring on the public set, or generation + CSV export on the private set). It does **not** walk through fine-tuning or training a new model from scratch. You *can* add finetuning later, but the current notebook flow is: **frozen required model + prompting + decoding**.

## Data you need (only files provided)

These are the **only** competition datasets required:

| File | Role |
|------|------|
| `public.jsonl` | Labels present — local evaluation |
| `private.jsonl` | No labels — build submission |
| `sample_submission.csv` | Example CSV shape |

Keep them under the paths your notebook/config use (e.g. `data/raw/` and `data/external/`). No extra downloads for the dataset itself.

## Python environment

**Option A — Conda / Mamba (recommended)**

From the **repo root**:

```text
conda env create -f environment.yml
conda activate cse151b-math-qa
python scripts/register_jupyter_kernel.py
```

This installs **Python 3.11**, **Jupyter + ipykernel**, the **scientific stack**, **PyTorch with CUDA 12.4** (from the `pytorch` + `nvidia` channels), and **pip** packages for **Transformers**, **bitsandbytes**, **judger** dependencies, etc. The display name **`Python (cse151b-math-qa)`** then appears in the kernel picker.

- **Linux / WSL2** and you want **vLLM** too: `conda env create -f environment-vllm.yml` (same env name; recreates the env — remove the old one first if it exists, or use a different name).
- If **CUDA solve** fails, edit `pytorch-cuda=12.4` → `12.1` in the YAML and retry.

**Option B — venv + pip**

```text
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -r requirements.txt
```

On macOS / Linux, activate with `source .venv/bin/activate`. You may need to install a **CUDA-enabled torch** wheel from [PyTorch](https://pytorch.org/get-started/locally/) before or after `requirements.txt`.

**Suggested Python:** 3.11.

## Jupyter kernel (so the notebook uses this env)

Conda path: run **`python scripts/register_jupyter_kernel.py`** once after `conda activate cse151b-math-qa`.

Manual equivalent:

```text
python -m ipykernel install --user --name cse151b-math-qa --display-name "Python (cse151b-math-qa)"
```

In VS Code / Cursor: **Select Kernel** → **Python (cse151b-math-qa)**.  
Shell lines like `!source .venv/bin/activate` in a notebook **do not** switch the kernel’s Python; the kernel choice does.

## Hugging Face Hub (optional but recommended)

- **Unauthenticated warnings:** Setting **`HF_TOKEN`** (or `huggingface-cli login`) raises rate limits and speeds downloads. See [token settings](https://huggingface.co/settings/tokens).
- **Windows symlink warning** in the HF cache: caching still works; enabling **Developer Mode** or using an admin shell can restore symlink behavior — not required if downloads complete. See [HF cache docs](https://huggingface.co/docs/huggingface_hub/how-to-cache).

## Inference defaults (Qwen3-Thinking) — see notebook

Aligned with `notebooks/02_inference.ipynb` as implemented:

- **Model**: **`Qwen/Qwen3-4B-Thinking-2507`**. Do not switch to 8B/14B/etc. for final submissions; the competition requires this model.
- **`MAX_TOKENS` (`max_new_tokens`)**: current source sets this to **`PHASE2_MAX_TOKENS = 6144`** for the v2 retry path. Earlier experiments used 8192 for maximum Phase 3, but Phase 3 was removed because it was too slow.
- **`enable_thinking=True`** passed to **`apply_chat_template`** (requires a recent **Transformers** build for this model).
- Sampling: Phase 1 uses **`temperature=0.6`**, **`top_p=0.95`**, **`top_k=20`**, **`min_p=0.0`**, **`repetition_penalty=1.0`**. Phase 2 uses temperature **`0.65`** and repetition penalty **`1.05`**.
- **Current notebook default**: `DATA_MODE="public"`, `N_QUESTIONS=None`, `RUN_NAME="adaptive_public_v2"`. Change to `DATA_MODE="private"` before leaderboard generation.
- **Self-consistency**: current v2 source uses retry-only majority voting with `PHASE2_N_SAMPLES=3`, not blanket self-consistency for every row.

## GPU, PyTorch, vLLM, and Windows

- **NVIDIA GPU + recent driver** is assumed for running the 4B model with quantization in reasonable time.
- **`requirements.txt`** can include **`torch`** and **`vllm`**. Conda users get **PyTorch** from **`environment.yml`**; plain pip may need the correct CUDA wheel from [PyTorch Get Started](https://pytorch.org/get-started/locally/).
- **vLLM** is primarily supported on **Linux** (and often **WSL2** with GPU). On **native Windows**, `pip install vllm` may fail; use the notebook’s **Transformers + bitsandbytes** path on a GPU machine, or run on Linux / WSL2 / the course GPU host.

---

## Production path — WSL2 + vLLM (recommended for desktop with NVIDIA GPU)

The Transformers path works everywhere but is **10–20× slower** than vLLM for a thinking model that generates thousands of tokens per question. On a desktop with a capable NVIDIA GPU, WSL2 unlocks vLLM and makes a full 1,000+ question run feasible in hours rather than days.

### Why WSL2 and not native Windows
vLLM’s PagedAttention kernel requires Linux. WSL2 exposes the Windows NVIDIA driver directly to a Linux environment — no separate driver install inside WSL2. The GPU is the same physical card; only the runtime changes.

### One-time WSL2 setup (Windows 10/11)

**In PowerShell (run as Administrator):**
```powershell
wsl --install
# Restart when prompted
```
After restart, open the Ubuntu app from the Start menu and complete the first-time user setup.

> **Note:** If you already have Docker Desktop installed, opening a terminal inside it will show a `docker-desktop` prompt — that is the wrong distro. Always open the **Ubuntu** app specifically, or run `wsl -d Ubuntu` from PowerShell.

### Python environment inside WSL2

```bash
# 1. Update package lists
sudo apt update && sudo apt upgrade -y

# 2. Install Python tooling
sudo apt install python3 python3-pip python3-venv -y

# 3. Create a dedicated venv for this project
python3 -m venv ~/cse151b-venv

# 4. Activate it (run this every new WSL session)
source ~/cse151b-venv/bin/activate
# Prompt should show: (cse151b-venv) ...

# 5. Install all dependencies (vLLM + ML stack — takes 5–15 min on first run)
pip install vllm transformers torch tqdm jupyter ipykernel \
    sympy numpy bitsandbytes "antlr4-python3-runtime==4.11.1"
```

### Verify GPU visibility
```bash
python3 -c "import torch; print(torch.cuda.get_device_name(0))"
# Expected: NVIDIA GeForce RTX <model>
```

### Register kernel and launch Jupyter

```bash
# Register this venv as a named Jupyter kernel
python3 -m ipykernel install --user --name cse151b-wsl --display-name "Python (cse151b WSL)"

# Launch Jupyter pointed at the project root (Windows path via /mnt/c/)
jupyter notebook --no-browser --port=8888 \
    "/mnt/c/<path-to-repo>/math-qa-llm"
```
Copy the printed `http://127.0.0.1:8888/?token=...` URL into your Windows browser. In the notebook, select kernel **"Python (cse151b WSL)"** from the top-right kernel picker.

### Notebook changes to activate vLLM

In `notebooks/02_inference.ipynb`, **Section 2 (Config)**:
```python
USE_VLLM   = True    # was False
MAX_TOKENS = 8192    # was 2048 — critical for thinking model
```

**Section 5 — current vLLM block** (tuned conservatively for 8 GB VRAM):
```python
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

llm = LLM(
    model=MODEL_ID,
    quantization="bitsandbytes",
    load_format="bitsandbytes",
    gpu_memory_utilization=0.78,   # leaves WSL2/8 GB headroom
    max_model_len=8192,
    trust_remote_code=True,
    max_num_seqs=4,                # conservative for 8 GB VRAM
    max_num_batched_tokens=4096,
)

sampling_params = SamplingParams(
    max_tokens=MAX_TOKENS,         # currently PHASE2_MAX_TOKENS = 6144
    temperature=0.6,               # Phase 2 uses 0.65
    top_p=0.95,
    top_k=20,
    min_p=0.0,
    repetition_penalty=1.0,
)
```

**Section 6 — use the vLLM generation cell** (already in the notebook, just uncomment it) and comment out the Transformers loop.

### Hardware / runtime notes (8 GB VRAM)

| Model | Quantization | VRAM used | Full run estimate |
|-------|-------------|-----------|-------------------|
| Required Qwen3-4B-Thinking | INT8 / bitsandbytes via vLLM | depends on cache + context | observed runtime can still be minutes/question with long thinking traces |

Because observed runtime was much slower than early estimates in the active notebook session, the current plan is **not** full public + Phase 3. The current notebook source uses a two-phase v2 retry path. For a leaderboard upload, first switch to `DATA_MODE="private"`, rerun the dataset cell, and confirm 943 rows before loading/generating.

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
- `environment.yml` / `environment-vllm.yml` — conda stacks  
- `PROJECT_BRIEF.md` / `DATASETS.md` — task and data  
- `STATUS.md` / `DECISIONS.md` — what’s implemented and why  
