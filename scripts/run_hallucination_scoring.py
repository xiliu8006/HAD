#!/usr/bin/env python3
from pathlib import Path
import runpy


if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("run_lvsm_hallucination.py")), run_name="__main__")
