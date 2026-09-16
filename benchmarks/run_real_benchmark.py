"""Real-data benchmark for RobustPixelMaker (scientific image sets).

Same protocol as the synthetic benchmark: every method selects among the SAME
patch-pooled features, a common Random Forest is refit on the selected patches and
scored on a held-out fold, and selection stability is the mean pairwise Jaccard across
folds. Recovery metrics are omitted because real data has no ground-truth informative
region; instead we report score, coverage, stability, and (for image sets with an
identifiable background) a border-coverage diagnostic showing how much of the
uninformative frame each method keeps.

Run:  py benchmarks/run_real_benchmark.py [--datasets mnist-3v8,mnist-all] [--quick]
Outputs: benchmarks/results/real_results.csv, REAL_SUMMARY.md, *.png
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

from robustpixelmaker import PatchGrid, mean_pairwise_jaccard  # noqa: E402
from benchmarks.competitors import (JUDGES, NO_SELECTION_NOTE, SELECTORS,  # noqa: E402
                                    mark_non_selectors, ordered_methods,
                                    score_panel, score_selection)
from benchmarks.datasets import LOADERS  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"


def border_patches(grid: PatchGrid, ring: int = 1) -> set:
    """Patch ids in the outer `ring` of the grid: background for centred imagery."""
    ids = set()
    for i in range(grid.n_py):
        for j in range(grid.n_px):
            if i < ring or j < ring or i >= grid.n_py - ring or j >= grid.n_px - ring:
                ids.add(i * grid.n_px + j)
    return ids


def evaluate(X, y, task, method, patch, seed, n_splits=5):
    from sklearn.model_selection import KFold, StratifiedKFold

    grid = PatchGrid.from_image_shape(X.shape, patch=patch)
    F = grid.pool(X)
    P = grid.n_patches
    border = border_patches(grid)
    selector = SELECTORS[method]

    if task == "regression":
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    else:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    panels, covs, sels, border_frac = [], [], [], []
    t0 = time.perf_counter()
    for tr, te in splitter.split(F, y):
        s = selector(X[tr], F[tr], y[tr], grid, task, seed)
        # one selection, scored by every judge: the selection is the expensive step, so
        # a panel costs a few classifier fits rather than a re-run per judge
        panels.append(score_panel(F[tr], y[tr], F[te], y[te], s, task, seed))
        covs.append(len(s) / P)
        sels.append(np.asarray(s, dtype=int))
        border_frac.append(len(set(int(i) for i in s) & border) / max(1, len(border)))
    seconds = time.perf_counter() - t0

    out = dict(coverage=float(np.mean(covs)),
               stability=float(mean_pairwise_jaccard(sels)),
               border_coverage=float(np.mean(border_frac)),
               seconds=seconds, n_patches=P)
    for j in list(JUDGES) + ["panel_mean"]:
        out[f"score_{j}"] = float(np.nanmean([p[j] for p in panels]))
    # `score` stays the random-forest judge, so every table and figure written before
    # the panel existed keeps meaning exactly what it meant
    out["score"] = out["score_rf"]
    return out


def _md_table(df: pd.DataFrame, floatfmt="{:.3f}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join([df.index.name or ""] + [str(c) for c in cols]) + " |",
             "| " + " | ".join(["---"] * (len(cols) + 1)) + " |"]
    for idx, row in df.iterrows():
        cells = [floatfmt.format(row[c]) if isinstance(row[c], (int, float, np.floating))
                 else str(row[c]) for c in cols]
        lines.append("| " + " | ".join([str(idx)] + cells) + " |")
    return "\n".join(lines)


def mask_figure(X, y, task, patch, seed, outdir, name):
    """Show the RPM mask over the mean image, so the selected region is visible."""
    from benchmarks.plotstyle import apply_style
    plt = apply_style()

    from robustpixelmaker import SoftMaskSelector

    grid = PatchGrid.from_image_shape(X.shape, patch=patch)
    sel = SoftMaskSelector(task=task, patch=patch, lam=0.03, gate="hardconcrete",
                           seed=seed).fit(X, y)
    mask_img = grid.expand(sel.mask_)

    fig, ax = plt.subplots(1, 3, figsize=(11, 3.6))
    ax[0].imshow(X.mean(axis=0), cmap="viridis")
    ax[0].set_title(f"{name}: mean image")
    im = ax[1].imshow(mask_img, cmap="viridis", vmin=0, vmax=1)
    ax[1].set_title(f"RPM mask (coverage {sel.coverage():.2f})")
    fig.colorbar(im, ax=ax[1], fraction=0.046)
    ax[2].imshow(X.mean(axis=0) * mask_img, cmap="viridis")
    ax[2].set_title("retained signal")
    for a in ax:
        a.set_xticks([])
        a.set_yticks([])
    fig.tight_layout()
    fig.savefig(outdir / f"mask_{name}.png", dpi=110)
    plt.close(fig)


def score_plot(df, outdir, prefix="real"):
    from benchmarks.plotstyle import apply_style
    plt = apply_style()

    methods = ordered_methods(df["method"].unique())
    vir = plt.cm.viridis(np.linspace(0.0, 0.9, len(methods)))
    colors = {m: vir[i] for i, m in enumerate(methods)}
    datasets = list(df["dataset"].unique())

    fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 4.6), squeeze=False)
    for k, ds in enumerate(datasets):
        a = axes[0][k]
        sub = df[df["dataset"] == ds].groupby("method")[["score", "coverage", "stability"]].mean()
        sub = sub.reindex([m for m in methods if m in sub.index])
        for m in sub.index:
            a.scatter(sub.loc[m, "coverage"], sub.loc[m, "score"],
                      s=60 + 260 * sub.loc[m, "stability"], color=colors[m])
            a.annotate(m, (sub.loc[m, "coverage"], sub.loc[m, "score"]),
                       textcoords="offset points", xytext=(7, 4), fontsize=9)
        a.set_xlabel("coverage (fraction of patches kept)")
        a.set_ylabel("score")
        a.set_title(f"{ds}\n(marker size = selection stability)")
    fig.tight_layout()
    fig.savefig(outdir / f"{prefix}_score_coverage.png", dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="mnist-3v8,mnist-all")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--patch", type=int, default=4)
    ap.add_argument("--n-max", type=int, default=None, help="cap on samples per dataset")
    ap.add_argument("--seeds", type=int, default=None, help="number of seeds")
    ap.add_argument("--out-prefix", default="real")
    ap.add_argument("--methods", default=None,
                    help="comma-separated subset of selectors (default: all)")
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    n_seeds = args.seeds if args.seeds is not None else (1 if args.quick else 3)
    seeds = list(range(n_seeds))
    names = [d.strip() for d in args.datasets.split(",") if d.strip()]
    n_max = args.n_max if args.n_max is not None else (1000 if args.quick else 4000)

    rows = []
    for name in names:
        X, y, meta = LOADERS[name](n_max=n_max)
        print(f"\n=== {name}: n={meta['n']} shape={meta['shape']} task={meta['task']} "
              f"classes={meta['classes']} ===", flush=True)
        wanted = ([m.strip() for m in args.methods.split(",")] if args.methods
                  else list(SELECTORS))
        for method in wanted:
            for seed in seeds:
                m = evaluate(X, y, meta["task"], method, args.patch, seed)
                rows.append(dict(dataset=name, task=meta["task"], method=method, seed=seed, **m))
                print(f"  {method:6s} seed={seed} score={m['score']:.4f} cov={m['coverage']:.2f} "
                      f"stab={m['stability']:.2f} border={m['border_coverage']:.2f} "
                      f"({m['seconds']:.1f}s)", flush=True)
        try:
            mask_figure(X, y, meta["task"], args.patch, 0, RESULTS, name)
        except Exception as exc:
            print(f"  mask figure failed: {exc}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / f"{args.out_prefix}_results.csv", index=False)
    try:
        score_plot(df, RESULTS, prefix=args.out_prefix)
    except Exception as exc:
        print(f"plotting failed: {exc}", flush=True)

    order = ordered_methods(df["method"].unique())
    lines = ["# Real-Data Benchmark Summary (scientific image sets)", "",
             "Score is AUC (binary) or accuracy-equivalent OVR AUC (multiclass); higher is "
             "better. `coverage` is the fraction of patches kept, `stability` the mean pairwise "
             "Jaccard of selections across folds, `border_coverage` the fraction of the outer "
             "patch ring retained (a background-rejection diagnostic for centred imagery).", "",
             NO_SELECTION_NOTE, ""]
    for ds in df["dataset"].unique():
        sub = df[df["dataset"] == ds].groupby("method")[
            ["score", "coverage", "stability", "border_coverage", "seconds"]].mean().reindex(order)
        sub = mark_non_selectors(sub)
        sub.index.name = f"{ds}"
        lines += [f"## {ds}", "", _md_table(sub), ""]
    lines += ["## Figures", ""]
    for ds in df["dataset"].unique():
        lines += [f"![RPM mask on {ds}](mask_{ds}.png)", ""]
    lines += [f"![Score vs coverage]({args.out_prefix}_score_coverage.png)", ""]
    (RESULTS / f"{args.out_prefix.upper()}_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nDONE. wrote {args.out_prefix}_results.csv and "
          f"{args.out_prefix.upper()}_SUMMARY.md in {RESULTS}", flush=True)


if __name__ == "__main__":
    main()
