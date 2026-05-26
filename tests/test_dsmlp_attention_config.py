import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = [
    REPO_ROOT / "notebooks" / "02_inference.ipynb",
    REPO_ROOT / "notebooks" / "06_private_submission.ipynb",
]


def read_code_cells(path: Path) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8-sig"))
    parts = []
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(source)
        parts.append(source)
    return "\n\n".join(parts)


class DsmlpAttentionConfigTests(unittest.TestCase):
    def test_notebooks_use_explicit_attention_backend_config(self):
        for notebook in NOTEBOOKS:
            code = read_code_cells(notebook)
            self.assertIn('ATTN_IMPLEMENTATION = "eager"', code, notebook.name)
            self.assertIn("attn_implementation=ATTN_IMPLEMENTATION", code, notebook.name)

    def test_notebooks_guard_sdpa_runtime_failures(self):
        for notebook in NOTEBOOKS:
            code = read_code_cells(notebook)
            self.assertIn("def looks_like_sdpa_runtime_error", code, notebook.name)
            self.assertIn("Retrying the batch one prompt at a time", code, notebook.name)


if __name__ == "__main__":
    unittest.main()
