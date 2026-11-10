#!/usr/bin/env python3
"""Absolute-path entry point for the conversion worker, independent of its cwd."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lerobot_annotation.__main__ import main

if __name__ == "__main__":
    main()
