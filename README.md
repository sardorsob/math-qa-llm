# Math QA LLM

End-to-end repository for the CSE 151B Spring 2026 math reasoning competition. The project fine-tunes Qwen3-4B-Thinking, reranks uncertain generations with a lightweight verifier head, and writes the final `id,response` submission CSV for the private set.

## What this repository contains

- A single grading entry point in [run_inference.py](C:/Users/sardo/OneDrive/Desktop/Classes/projects/math-qa-llm/run_inference.py)
- Notebook-based experimentation and training in [notebooks](C:/Users/sardo/OneDrive/Desktop/Classes/projects/math-qa-llm/notebooks)
- Submission and recovery utilities in [scripts](C:/Users/sardo/OneDrive/Desktop/Classes/projects/math-qa-llm/scripts)
- Project decisions, methodology notes, and experiment history in [context](C:/Users/sardo/OneDrive/Desktop/Classes/projects/math-qa-llm/context)

## Submission entry point

The repository exposes one function:

```python
from run_inference import run_inference

run_inference(
    private_jsonl="data/raw/private.jsonl",
    output_csv="submission.csv",
)
```

That function performs the full pipeline end to end:

1. Downloads the merged fine-tuned model from Hugging Face Hub.
2. Downloads the EBM verifier head from the same Hub repo.
3. Loads `private.jsonl`.
4. Runs multi-phase inference on the private set.
5. Applies post-processing to recover truncated answers into boxed form.
6. Writes the final submission CSV.

You can also run the same pipeline from the command line:

```bash
python run_inference.py --private-jsonl data/raw/private.jsonl --output-csv submission.csv
```

## Model weights

The default inference path loads weights directly from Hugging Face Hub, so no manual model download is required for grading.

- Main model repo: `SardorSob/qwen3-4b-thinking-math-qa-qlora`
- EBM verifier head: stored under the `ebm/` subdirectory of the same Hugging Face model repo

The script will automatically download:

- the merged QLoRA model
- tokenizer files
- generation config
- the verifier head checkpoint

If you want to run from local weights instead of Hugging Face Hub, keep the merged model under:

- `artifacts/models/qlora_v1_merged/`

and the verifier head under:

- `artifacts/models/ebm_verifier_v1/`

The current submission entry point is configured to use Hugging Face Hub by default.

## Dataset placement

Place the competition files here:

- `data/raw/public.jsonl`
- `data/raw/private.jsonl`
- `data/external/sample_submission.csv`

`run_inference()` only needs `data/raw/private.jsonl`.

## Hardware used

Primary development, notebook execution, and private-set generation work were run on DSMLP GPU pods.

- Main DSMLP target: NVIDIA A30 24 GB
- Additional debugging and memory testing: NVIDIA H100 PCIe MIG 1g.20gb

## Approximate inference time

Runtime depends heavily on GPU memory, backend choice, and how many questions escalate to the second phase.

Practical expectations from this repository:

- On a 24 GB class GPU such as an A30, private-set inference is an hours-scale run, roughly about 4 to 6 hours for the full 943-question private set, plus a short post-processing pass.
- Recovery and CSV writing are fast compared with generation, typically seconds to a few minutes.
- Larger GPUs with more memory should complete noticeably faster because the script can use larger batches and avoid the most aggressive memory compromises.

## Environment setup

Recommended Conda setup:

```bash
conda env create -f environment.yml
conda activate cse151b-math-qa
```

Alternative pip setup:

```bash
pip install -r requirements.txt
```

Important runtime dependencies include:

- `torch`
- `transformers`
- `accelerate`
- `bitsandbytes`
- `huggingface_hub`
- `safetensors`

See [ENVIRONMENT_SETUP.md](C:/Users/sardo/OneDrive/Desktop/Classes/projects/math-qa-llm/context/ENVIRONMENT_SETUP.md) for additional notes.

## Inference pipeline summary

The current submission entry point uses the following structure:

1. Phase 1 runs one first-pass generation for every private question.
2. Phase 2 retries only uncertain questions with more compute and multiple candidates.
3. The EBM verifier head reranks those second-phase candidates when available.
4. A v1-style truncated-answer recovery pass appends boxed answers where the model clearly signaled an answer but ran out of tokens before formatting it.

This keeps the full logic in one callable function rather than requiring manual notebook steps.

## Reproducibility notes

- The repository keeps the notebooks, recovered artifacts, and context notes that document the original deadline-time workflow.
- The grading entry point is [run_inference.py](C:/Users/sardo/OneDrive/Desktop/Classes/projects/math-qa-llm/run_inference.py). That is the file we intend graders to use.
- The script writes a checkpoint JSONL next to the output CSV so interrupted runs can resume.
- The private-set CSV format is exactly two columns: `id,response`.

## Repository layout

- [context](C:/Users/sardo/OneDrive/Desktop/Classes/projects/math-qa-llm/context) - project memory, assumptions, decisions, and status
- [notebooks](C:/Users/sardo/OneDrive/Desktop/Classes/projects/math-qa-llm/notebooks) - experimentation, training, verifier training, and private submission notebooks
- [scripts](C:/Users/sardo/OneDrive/Desktop/Classes/projects/math-qa-llm/scripts) - standalone recovery and CSV utilities
- [artifacts](C:/Users/sardo/OneDrive/Desktop/Classes/projects/math-qa-llm/artifacts) - models, logs, and submission outputs
- [judger.py](C:/Users/sardo/OneDrive/Desktop/Classes/projects/math-qa-llm/judger.py) - local public-set scoring helper
- [utils.py](C:/Users/sardo/OneDrive/Desktop/Classes/projects/math-qa-llm/utils.py) - shared utility helpers used by the original workflow

## Key files

- [run_inference.py](C:/Users/sardo/OneDrive/Desktop/Classes/projects/math-qa-llm/run_inference.py) - single end-to-end grading entry point
- [06_private_submission.ipynb](C:/Users/sardo/OneDrive/Desktop/Classes/projects/math-qa-llm/notebooks/06_private_submission.ipynb) - private inference notebook that informed the script
- [recover_truncated_answers.py](C:/Users/sardo/OneDrive/Desktop/Classes/projects/math-qa-llm/scripts/recover_truncated_answers.py) - original truncated-answer recovery logic
- [recover_truncated_answers_v2.py](C:/Users/sardo/OneDrive/Desktop/Classes/projects/math-qa-llm/scripts/recover_truncated_answers_v2.py) - later recovery refinement kept for reference

## Quick verification

To make sure the script is importable and syntactically valid:

```bash
python -m py_compile run_inference.py
```

To run the full private pipeline:

```bash
python run_inference.py --private-jsonl data/raw/private.jsonl --output-csv submission.csv
```
