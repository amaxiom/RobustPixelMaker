"""Merge every real-dataset result CSV into one cross-dataset comparison.

Reads all ``results/*_results.csv`` files that contain a ``dataset`` column (the
real-data runs) and prints, plus writes, a single table of score, coverage, stability
and border coverage per method per dataset. Use it after adding a dataset so the
scientific sets can be read side by side rather than one summary at a time.

Run:  py benchmarks/compare_real.py
Output: benchmarks/results/CROSS_DATASET.md
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from benchmarks.competitors import ordered_methods  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"
METRICS = ["score", "coverage", "stability", "border_coverage"]


def load_all() -> pd.DataFrame:
    frames = []
    for csv in sorted(RESULTS.glob("*_results.csv")):
        df = pd.read_csv(csv)
        if "dataset" in df.columns:
            frames.append(df)
    if not frames:
        raise SystemExit("no real-dataset result CSVs found; run a real benchmark first")
    return pd.concat(frames, ignore_index=True)


def main():
    df = load_all()
    order = ordered_methods(df["method"].unique())
    lines = ["# Cross-dataset comparison (real scientific image sets)", "",
             "One row per method per dataset. `score` is AUC (binary) or weighted "
             "one-vs-rest AUC (multiclass); `coverage` is the fraction of patches kept; "
             "`stability` is the mean pairwise Jaccard of selections across folds; "
             "`border_coverage` is the fraction of the outer patch ring retained, a "
             "background-rejection diagnostic for centred imagery (lower is better).", ""]

    for ds, sub in df.groupby("dataset"):
        agg = sub.groupby("method")[METRICS].mean().reindex(
            [m for m in order if m in sub["method"].unique()])
        lines.append(f"## {ds}  (task: {sub['task'].iloc[0]}, seeds: {sub['seed'].nunique()})")
        lines.append("")
        lines.append("| method | " + " | ".join(METRICS) + " |")
        lines.append("| --- | " + " | ".join(["---"] * len(METRICS)) + " |")
        for m, row in agg.iterrows():
            lines.append("| " + m + " | " + " | ".join(f"{row[c]:.3f}" for c in METRICS) + " |")
        lines.append("")

    # leanness-at-comparable-score view: how much does each method keep, and how much
    # of the uninformative border does it wrongly retain, averaged over datasets
    lines.append("## Averaged over all real datasets")
    lines.append("")
    agg = df.groupby("method")[METRICS].mean().reindex(
        [m for m in order if m in df["method"].unique()])
    lines.append("| method | " + " | ".join(METRICS) + " |")
    lines.append("| --- | " + " | ".join(["---"] * len(METRICS)) + " |")
    for m, row in agg.iterrows():
        lines.append("| " + m + " | " + " | ".join(f"{row[c]:.3f}" for c in METRICS) + " |")
    lines.append("")

    out = RESULTS / "CROSS_DATASET.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
