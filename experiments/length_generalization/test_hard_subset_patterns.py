"""Wrapper that runs the root-level hard-subset pattern script."""

from pathlib import Path
import runpy


if __name__ == "__main__":
    script = Path(__file__).resolve().parents[2] / "test_hard_subset_patterns.py"
    runpy.run_path(str(script), run_name="__main__")
