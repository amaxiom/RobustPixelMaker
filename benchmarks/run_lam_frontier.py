"""Sparsity-strength frontier for RobustPixelMaker.

The single-point benchmark fixes ``lam``, which conflates "the method loses score"
with "this particular sparsity strength was too aggressive". This script sweeps
``lam`` for both gate types and traces the score against coverage frontier, so the
operating point is a reported choice rather than a hidden default.

For each ``lam`` and gate: fit the mask on each training fold, refit the common Random
Forest on the selected patches, score the held-out fold, and record coverage and
cross-fold Jaccard stability. Competitor methods are drawn as reference points.

Run:  py benchmarks/run_lam_frontier.py [--dataset mnist-3v8] [--n-max 2000]
Outputs: benchmarks/results/lam_frontier_<dataset>.csv and .png
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from robustpixelmaker import PatchGrid, SoftMaskSelector, mean_pairwise_jaccard  # noqa: E402
from benchmarks.competitors import SELECTORS, score_selection  # noqa: E402
from benchmarks.datasets import LOADERS  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"
LAMS = [0.001, 0.003, 0.01, 0.03, 0.1]
REFERENCE = ["full", "anova", "rfe", "rf"]


def folds(F, y, task, seed, n_splits):
    from sklearn.model_selection import KFold, StratifiedKFold
    sp = (KFold if task == "regression" else StratifiedKFold)(
        n_splits=n_splits, shuffle=True, random_state=seed)
    return list(sp.split(F, y))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mnist-3v8")
    ap.add_argument("--n-max", type=int, default=2000)
    ap.add_argument("--patch", type=int, default=4)
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)

    X, y, meta = LOADERS[args.dataset](n_max=args.n_max)
    task = meta["task"]
    grid = PatchGrid.from_image_shape(X.shape, patch=args.patch)
    F = grid.pool(X)
    splits = folds(F, y, task, args.seed, args.folds)
    print(f"{args.dataset}: n={meta['n']} task={task} patches={grid.n_patches}", flush=True)

    rows = []
    for gate in ("sigmoid", "hardconcrete"):
        for lam in LAMS:
            t0 = time.perf_counter()
            sc, cov, sels = [], [], []
            for tr, te in splits:
                sel = SoftMaskSelector(task=task, patch=args.patch, lam=lam, gate=gate,
                                       n_iter=500, seed=args.seed).fit(X[tr], y[tr])
                s = sel.selected_patches_
                sc.append(score_selection(F[tr], y[tr], F[te], y[te], s, task, args.seed))
                cov.append(len(s) / grid.n_patches)
                sels.append(np.asarray(s, dtype=int))
            rows.append(dict(method=f"rpm-{gate}", gate=gate, lam=lam,
                             score=float(np.nanmean(sc)), coverage=float(np.mean(cov)),
                             stability=float(mean_pairwise_jaccard(sels)),
                             seconds=time.perf_counter() - t0))
            print(f"  {gate:13s} lam={lam:<6g} score={rows[-1]['score']:.4f} "
                  f"cov={rows[-1]['coverage']:.2f} stab={rows[-1]['stability']:.2f}", flush=True)

    for name in REFERENCE:
        t0 = time.perf_counter()
        sc, cov, sels = [], [], []
        for tr, te in splits:
            s = SELECTORS[name](X[tr], F[tr], y[tr], grid, task, args.seed)
            sc.append(score_selection(F[tr], y[tr], F[te], y[te], s, task, args.seed))
            cov.append(len(s) / grid.n_patches)
            sels.append(np.asarray(s, dtype=int))
        rows.append(dict(method=name, gate="reference", lam=np.nan,
                         score=float(np.nanmean(sc)), coverage=float(np.mean(cov)),
                         stability=float(mean_pairwise_jaccard(sels)),
                         seconds=time.perf_counter() - t0))
        print(f"  {name:13s} (reference)  score={rows[-1]['score']:.4f} "
              f"cov={rows[-1]['coverage']:.2f} stab={rows[-1]['stability']:.2f}", flush=True)

    df = pd.DataFrame(rows)
    csv = RESULTS / f"lam_frontier_{args.dataset}.csv"
    df.to_csv(csv, index=False)

    from benchmarks.plotstyle import apply_style
    plt = apply_style()

    vir = plt.cm.viridis(np.linspace(0.0, 0.9, 4))
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for i, gate in enumerate(("sigmoid", "hardconcrete")):
        d = df[df["gate"] == gate].sort_values("coverage")
        ax.plot(d["coverage"], d["score"], marker="o", color=vir[i], label=f"rpm ({gate})")
        for _, r in d.iterrows():
            ax.annotate(f"{r['lam']:g}", (r["coverage"], r["score"]),
                        textcoords="offset points", xytext=(5, -10), fontsize=7)
    ref = df[df["gate"] == "reference"]
    ax.scatter(ref["coverage"], ref["score"], color=vir[3], marker="s", s=70, label="reference", zorder=5)
    for _, r in ref.iterrows():
        ax.annotate(r["method"], (r["coverage"], r["score"]),
                    textcoords="offset points", xytext=(6, 5), fontsize=8)
    ax.set_xlabel("coverage (fraction of patches kept)")
    ax.set_ylabel("score (held-out)")
    ax.set_title(f"Score vs coverage frontier: {args.dataset}\n(labels on the curves are lam)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / f"lam_frontier_{args.dataset}.png", dpi=110)
    plt.close(fig)
    print(f"\nDONE. wrote {csv}", flush=True)


if __name__ == "__main__":
    main()
