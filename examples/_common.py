"""Shared plumbing for the examples: paths, style, and a tiny report helper.

Each example is a self-contained script that fits in a few minutes on a laptop,
writes its figure(s) to ``examples/output/`` and prints a short plain-text takeaway.
The figures follow the house style (viridis, no bold anywhere in a figure).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(parents=True, exist_ok=True)


def style():
    from benchmarks.plotstyle import apply_style

    return apply_style()


def takeaway(title: str, lines):
    bar = "=" * 74
    print(f"\n{bar}\n{title}\n{bar}")
    for line in lines:
        print(f"  {line}")
    print(bar, flush=True)
