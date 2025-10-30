#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(script: str) -> None:
    print(f"\n==== Running: {script} ====")
    cmd = [sys.executable, str(ROOT / "experiments" / script)]
    env = os.environ.copy()
    # Ensure project root is importable as module path (so `import src...` works)
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"] if "PYTHONPATH" in env and env["PYTHONPATH"] else "")
    subprocess.run(cmd, check=True, env=env)


def main() -> None:
    run("pillar1_linear_separable.py")
    run("pillar1_xor_nonseparable.py")
    run("pillar2_comparisons.py")
    run("pillar3_svm_compare.py")
    # Extended experiments (may take longer / require network for Olivetti):
    try:
        run("multiclass_perceptron_demo.py")
    except Exception as e:
        print(f"[WARN] multiclass_perceptron_demo failed: {e}")
    try:
        run("lda_faces_perceptron.py")
    except Exception as e:
        print(f"[WARN] lda_faces_perceptron failed: {e}")
    print("\nAll experiments completed. Figures and tables are in assets/.")


if __name__ == "__main__":
    main()
