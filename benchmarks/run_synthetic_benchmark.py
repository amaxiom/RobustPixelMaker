"""Comprehensive synthetic benchmark for RobustPixelMaker.

Sweeps sample size, image shape, noise, signal, task, and complexity (informative
region size and number of distractors), comparing RobustPixelMaker (`rpm`) against a
full-feature baseline and four standard selectors (anova, lasso, rfe, rf). Because the
data are synthetic, the ground-truth informative region is known, so besides the usual
predictive score, coverage, and cross-fold selection stability, we measure how well
each method RECOVERS the planted region (precision, recall, F1, IoU) and rejects the
equal-variance distractor.

Protocol per (scenario, method, seed): stratified/plain K-fold; in each fold the
selector is fit on the training patches only, a common Random Forest is refit on the
selected patches and scored on the held-out fold (leakage-safe), and recovery is
measured against the ground truth. Selection stability is the mean pairwise Jaccard of
the selected sets across folds (RobustModelMaker's stability metric).

Run:  py benchmarks/run_synthetic_benchmark.py [--quick]
Outputs: benchmarks/results/synthetic_results.csv, SYNTHETIC_SUMMARY.md, *.png
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from robustpixelmaker import PatchGrid, make_synthetic_images, mean_pairwise_jaccard  # noqa: E402
from benchmarks.competitors import (NO_SELECTION_NOTE, SELECTORS,  # noqa: E402
                                    mark_non_selectors, ordered_methods,
                                    recovery_metrics, score_selection)

RESULTS = Path(__file__).resolve().parent / "results"


def cell_for(shape):
    return max(2, min(shape) // 8)


DEFAULT = dict(n=400, shape=(32, 32), task="binary", signal=2.0, noise=1.0,
               region_side=2, distractor_side=2)


def scenarios():
    scen = []

    def add(axis, value, **over):
        p = dict(DEFAULT)
        p.update(over)
        p["cell"] = cell_for(p["shape"])
        scen.append(dict(axis=axis, value=str(value), label=f"{axis}={value}", params=p))

    for n in [100, 200, 400, 800]:
        add("n", n, n=n)
    for s in [(16, 16), (24, 24), (32, 32), (48, 48)]:
        add("shape", f"{s[0]}x{s[1]}", shape=s)
    for nz in [0.5, 1.0, 2.0, 4.0]:
        add("noise", nz, noise=nz)
    for sg in [1.0, 2.0, 3.0]:
        add("signal", sg, signal=sg)
    for t in ["binary", "regression"]:
        add("task", t, task=t)
    # complexity: informative region size and distractor count (larger frame to fit)
    for rs in [1, 2, 3]:
        add("region_side", rs, shape=(48, 48), region_side=rs)
    for ds in [0, 1, 2]:
        add("distractor_side", ds, shape=(48, 48), distractor_side=ds)
    return scen


def evaluate(params, method, seed, n_splits=5):
    from sklearn.model_selection import KFold, StratifiedKFold

    c = make_synthetic_images(seed=seed, **params)
    task = params["task"]
    grid = PatchGrid.from_image_shape(c.X.shape, patch=params["cell"])
    F = grid.pool(c.X)
    P = grid.n_patches
    info_p = np.unique(grid.pixel_patch[c.informative]) if c.informative.any() else np.array([], int)
    dist_p = np.unique(grid.pixel_patch[c.distractor]) if c.distractor.any() else np.array([], int)
    selector = SELECTORS[method]

    if task == "regression":
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    else:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    scores, covs, recs, sels = [], [], [], []
    t0 = time.perf_counter()
    for tr, te in splitter.split(F, c.y):
        s = selector(c.X[tr], F[tr], c.y[tr], grid, task, seed)
        scores.append(score_selection(F[tr], c.y[tr], F[te], c.y[te], s, task, seed))
        covs.append(len(s) / P)
        recs.append(recovery_metrics(s, info_p, dist_p, P))
        sels.append(np.asarray(s, dtype=int))
    seconds = time.perf_counter() - t0

    agg = dict(score=float(np.nanmean(scores)), coverage=float(np.mean(covs)),
               stability=float(mean_pairwise_jaccard(sels)), seconds=seconds, n_patches=P)
    for k in ("precision", "recall", "f1", "iou", "distractor_fp"):
        agg[k] = float(np.nanmean([r[k] for r in recs]))
    return agg


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _md_table(df: pd.DataFrame, floatfmt="{:.3f}") -> str:
    cols = list(df.columns)
    head = "| " + " | ".join([df.index.name or ""] + [str(c) for c in cols]) + " |"
    sep = "| " + " | ".join(["---"] * (len(cols) + 1)) + " |"
    lines = [head, sep]
    for idx, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            cells.append(floatfmt.format(v) if isinstance(v, (int, float, np.floating)) else str(v))
        lines.append("| " + " | ".join([str(idx)] + cells) + " |")
    return "\n".join(lines)


def make_plots(df, outdir, prefix="synthetic"):
    from benchmarks.plotstyle import apply_style
    plt = apply_style()

    methods = ordered_methods(df["method"].unique())
    vir = plt.cm.viridis(np.linspace(0.0, 0.9, len(methods)))
    colors = {m: vir[i] for i, m in enumerate(methods)}

    # 1) headline recovery F1 and stability by method
    g = df.groupby("method")[["f1", "stability", "iou", "distractor_fp"]].mean().reindex(methods)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].bar(g.index, g["f1"], color=[colors[m] for m in g.index])
    ax[0].set_title("Mean region-recovery F1 (higher better)")
    ax[0].set_ylim(0, 1)
    ax[1].bar(g.index, g["stability"], color=[colors[m] for m in g.index])
    ax[1].set_title("Mean selection stability, Jaccard (higher better)")
    ax[1].set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(outdir / f"{prefix}_recovery_stability.png", dpi=110)
    plt.close(fig)

    # 2) recovery F1 vs noise and vs n (robustness sweeps)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for name, axis, a in [("noise", "noise", ax[0]), ("n", "n", ax[1])]:
        sub = df[df["axis"] == axis]
        for m in methods:
            d = sub[sub["method"] == m].copy()
            if d.empty:
                continue
            d["xv"] = d["value"].astype(float)
            dd = d.groupby("xv")["f1"].mean().sort_index()
            a.plot(dd.index, dd.values, marker="o", label=m, color=colors[m])
        a.set_title(f"Recovery F1 vs {name}")
        a.set_xlabel(name)
        a.set_ylim(0, 1)
    for a in ax:
        if a.get_legend_handles_labels()[0]:
            a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / f"{prefix}_recovery_sweeps.png", dpi=110)
    plt.close(fig)

    # 3) score-coverage frontier (binary AUC scenarios)
    b = df[df["task"] == "binary"].groupby("method")[["score", "coverage", "stability"]].mean().reindex(methods)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for m in b.index:
        ax.scatter(b.loc[m, "coverage"], b.loc[m, "score"], s=140, color=colors[m])
        ax.annotate(m, (b.loc[m, "coverage"], b.loc[m, "score"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=9)
    ax.set_xlabel("coverage (fraction of patches kept, lower is leaner)")
    ax.set_ylabel("AUC (binary scenarios, higher better)")
    ax.set_title("Score vs coverage frontier")
    fig.tight_layout()
    fig.savefig(outdir / f"{prefix}_score_coverage_frontier.png", dpi=110)
    plt.close(fig)


def write_summary(df, outdir, prefix="synthetic"):
    lines = [f"# Synthetic Benchmark Summary ({prefix})", ""]
    lines.append(f"Rows: {len(df)}  |  methods: {sorted(df['method'].unique())}  |  "
                 f"scenarios: {df['scenario'].nunique()}  |  seeds: {sorted(df['seed'].unique())}")
    lines.append("")
    lines.append("Metrics: **f1/iou/precision/recall** = patch-level recovery of the known "
                 "informative region; **distractor_fp** = fraction of the non-predictive "
                 "distractor wrongly selected (lower better); **coverage** = fraction of patches "
                 "kept; **stability** = mean pairwise Jaccard of selections across folds; "
                 "**score** = AUC (binary, higher better) or RMSE (regression, lower better).")
    lines.append("")

    order = ordered_methods(df["method"].unique())

    lines.append("## Headline (mean over all scenarios and seeds)")
    lines.append("")
    head = df.groupby("method")[["f1", "iou", "precision", "recall", "distractor_fp",
                                  "coverage", "stability", "seconds"]].mean().reindex(order)
    head.index.name = "method"
    head = mark_non_selectors(head)
    lines.append(_md_table(head))
    lines.append("")
    lines.append(NO_SELECTION_NOTE)
    lines.append("")

    lines.append("## Binary score vs coverage vs stability (mean over binary scenarios/seeds)")
    lines.append("")
    b = df[df["task"] == "binary"].groupby("method")[["score", "coverage", "stability", "f1"]].mean().reindex(order)
    b.index.name = "method(AUC)"
    b = mark_non_selectors(b)
    lines.append(_md_table(b))
    lines.append("")

    for axis, title in [("noise", "Recovery F1 vs noise"), ("n", "Recovery F1 vs sample size"),
                        ("signal", "Recovery F1 vs signal"), ("region_side", "Recovery F1 vs region size")]:
        sub = df[df["axis"] == axis]
        if sub.empty:
            continue
        piv = sub.pivot_table(index="value", columns="method", values="f1", aggfunc="mean").reindex(columns=order)
        piv.index.name = axis
        lines.append(f"## {title}")
        lines.append("")
        lines.append(_md_table(piv))
        lines.append("")

    lines.append("## Figures")
    lines.append("")
    for png, cap in [(f"{prefix}_recovery_stability.png", "Recovery F1 and selection stability by method"),
                     (f"{prefix}_recovery_sweeps.png", "Recovery F1 vs noise and vs sample size"),
                     (f"{prefix}_score_coverage_frontier.png", "Score vs coverage frontier (binary)")]:
        lines.append(f"![{cap}]({png})")
        lines.append("")
    (outdir / f"{prefix.upper()}_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main(quick=False, axes=None, out_prefix="synthetic"):
    RESULTS.mkdir(parents=True, exist_ok=True)
    seeds = [0] if quick else [0, 1, 2]
    scen = scenarios()
    if axes:
        wanted = set(axes)
        scen = [s for s in scen if s["axis"] in wanted]
    if quick:
        scen = scen[:3]
    total = len(scen) * len(SELECTORS) * len(seeds)
    rows = []
    i = 0
    for s in scen:
        for method in SELECTORS:
            for seed in seeds:
                i += 1
                m = evaluate(s["params"], method, seed)
                rows.append(dict(axis=s["axis"], scenario=s["label"], value=s["value"],
                                 task=s["params"]["task"], method=method, seed=seed, **m))
                print(f"[{i}/{total}] {s['label']:22s} {method:6s} seed={seed} "
                      f"score={m['score']:.3f} f1={m['f1']:.2f} cov={m['coverage']:.2f} "
                      f"stab={m['stability']:.2f}", flush=True)
    df = pd.DataFrame(rows)
    csv_path = RESULTS / f"{out_prefix}_results.csv"
    df.to_csv(csv_path, index=False)
    try:
        make_plots(df, RESULTS, prefix=out_prefix)
    except Exception as exc:  # never lose the CSV over a plotting error
        print(f"plotting failed: {exc}", flush=True)
    write_summary(df, RESULTS, prefix=out_prefix)
    print(f"\nDONE. wrote {csv_path} and {out_prefix.upper()}_SUMMARY.md", flush=True)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--axes", default=None,
                    help="comma-separated sweep axes to run (default: all)")
    ap.add_argument("--out-prefix", default="synthetic")
    a = ap.parse_args()
    main(quick=a.quick,
         axes=[x.strip() for x in a.axes.split(",")] if a.axes else None,
         out_prefix=a.out_prefix)
