"""Register this environment as a Jupyter kernel (run after `conda activate cse151b-math-qa`)."""

import subprocess
import sys

NAME = "cse151b-math-qa"
DISPLAY = "Python (cse151b-math-qa)"


def main() -> None:
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "ipykernel",
            "install",
            "--user",
            "--name",
            NAME,
            "--display-name",
            DISPLAY,
        ]
    )
    print(f"Registered Jupyter kernel: {DISPLAY} (internal name: {NAME})")


if __name__ == "__main__":
    main()
