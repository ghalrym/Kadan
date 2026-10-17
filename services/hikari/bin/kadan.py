#!/usr/bin/env python3
"""Compatibility entry point for existing Kadan developer commands."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bin.hikari import main


if __name__ == "__main__":
    main()
