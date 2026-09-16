"""Timing and scaling benchmark, with the idleness precondition enforced.

Every other benchmark in this suite records a ``seconds`` column, and every one of those
numbers was measured while something else was running. They are fine for ordering methods
within a single run, and worthless as absolute costs. This script exists to produce
timings that can actually be quoted.

Three things make that possible.

**It refuses to run on a busy machine.** ``machine_state`` samples CPU utilisation and
enumerates competing interpreters; by default a busy machine aborts rather than producing
a number that looks authoritative and is not. ``--force`` overrides, and the override is
recorded in the output so a forced run can never be mistaken for a clean one.

**It reports best-of-k, not the mean.** Contention can only ever make a measurement
slower, so over repeats the minimum is the estimate closest to the true cost, while the
mean drifts with whatever else the scheduler did. The spread between minimum and median
is reported alongside, because a large gap is itself evidence the machine was not quiet.

**It records the conditions next to the numbers.** CPU count, utilisation, competing
process count and library versions go into the CSV and the summary, so a timing can be
interpreted, or discarded, months later without having to remember the day it was taken.

Axes measured: sample count, image side, patch size, bootstrap count, representation
count, and a per-method comparison against the competitors at a fixed problem size.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from machine_state import is_idle, snapshot                       # noqa: E402
from robustpixelmaker import (                                    # noqa: E402
    BootstrapMaskSelector,
    PatchGrid,
    RepresentationEnsembleSelector,
    SoftMaskSelector,
    make_synthetic_images,
)

RESULTS = HERE / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

# problem size held fixed while one axis varies
BASE = dict(n=400, side=32, patch=4, n_bootstrap=10, n_representations=4)


def timed(fn, repeats: int) -> dict:
    """Best-of-k timing, with the spread that says whether to believe it."""
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    ts = np.array(ts)
    return {"seconds": float(ts.min()),
            "median_seconds": float(np.median(ts)),
            "max_seconds": float(ts.max()),
            "spread_ratio": float(np.median(ts) / ts.min()) if ts.min() > 0 else float("nan"),
            "repeats": repeats}


def data(n, side, seed=0):
    c = make_synthetic_images(n=n, shape=(side, side), task="binary", seed=seed)
    return c.X, c.y


def fit_fns(X, y, patch, n_bootstrap, n_representations, n_iter):
    """The method under test, as a zero-argument callable each time."""
    common = dict(task="binary", patch=patch, lam=0.03, n_iter=n_iter, seed=0)
    return {
        "single": lambda: SoftMaskSelector(**common).fit(X, y),
        "bootstrap": lambda: BootstrapMaskSelector(n_bootstrap=n_bootstrap, **common).fit(X, y),
        "ensemble": lambda: RepresentationEnsembleSelector(
            n_bootstrap=max(2, n_bootstrap // 2),
            n_representations=n_representations, **common).fit(X, y),
    }


def competitor_fns(X, y, patch):
    """Competitors timed on the same pooled features, so the comparison is like for like."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.linear_model import LogisticRegression

    F = PatchGrid.from_image_shape(X.shape, patch=patch).pool(X)
    k = max(1, F.shape[1] // 10)
    return {
        "anova": lambda: SelectKBest(f_classif, k=k).fit(F, y),
        "lasso": lambda: LogisticRegression(penalty="l1", solver="saga", C=0.1,
                                            max_iter=300).fit(F, y),
        "rf": lambda: RandomForestClassifier(n_estimators=200, random_state=0).fit(F, y),
    }


def run(args):
    rows = []

    def record(axis, value, method, res, **extra):
        row = dict(axis=axis, value=value, method=method, **res, **extra)
        rows.append(row)
        print(f"  {axis:16s} {str(value):>6s}  {method:10s} "
              f"{res['seconds']:8.3f}s  (spread x{res['spread_ratio']:.2f})", flush=True)

    print("\n[1/6] sample count", flush=True)
    for n in [100, 200, 400, 800, 1600]:
        X, y = data(n, BASE["side"])
        for name, fn in fit_fns(X, y, BASE["patch"], BASE["n_bootstrap"],
                                BASE["n_representations"], args.n_iter).items():
            record("n_samples", n, name, timed(fn, args.repeats), n_patches=64)

    print("\n[2/6] image side", flush=True)
    # 32 is the smallest frame the synthetic generator's planted cells fit into; it
    # raises on anything smaller rather than silently shrinking the region, so the
    # axis starts there instead of at 16
    for side in [32, 64, 128, 256]:
        X, y = data(BASE["n"], side)
        P = PatchGrid.from_image_shape(X.shape, patch=BASE["patch"]).n_patches
        for name, fn in fit_fns(X, y, BASE["patch"], BASE["n_bootstrap"],
                                BASE["n_representations"], args.n_iter).items():
            record("image_side", side, name, timed(fn, args.repeats), n_patches=P)

    print("\n[3/6] patch size", flush=True)
    X, y = data(BASE["n"], 64)
    for patch in [2, 4, 8, 16]:
        P = PatchGrid.from_image_shape(X.shape, patch=patch).n_patches
        for name, fn in fit_fns(X, y, patch, BASE["n_bootstrap"],
                                BASE["n_representations"], args.n_iter).items():
            record("patch", patch, name, timed(fn, args.repeats), n_patches=P)

    print("\n[4/6] bootstrap count", flush=True)
    X, y = data(BASE["n"], BASE["side"])
    for b in [5, 10, 20, 40]:
        fn = fit_fns(X, y, BASE["patch"], b, BASE["n_representations"], args.n_iter)["bootstrap"]
        record("n_bootstrap", b, "bootstrap", timed(fn, args.repeats), n_patches=64)

    print("\n[5/6] representation count", flush=True)
    for r in [2, 4, 8]:
        fn = fit_fns(X, y, BASE["patch"], BASE["n_bootstrap"], r, args.n_iter)["ensemble"]
        record("n_representations", r, "ensemble", timed(fn, args.repeats), n_patches=64)

    print("\n[6/6] method comparison at a fixed size", flush=True)
    X, y = data(BASE["n"], BASE["side"])
    fns = dict(fit_fns(X, y, BASE["patch"], BASE["n_bootstrap"],
                       BASE["n_representations"], args.n_iter),
               **competitor_fns(X, y, BASE["patch"]))
    for name, fn in fns.items():
        record("method_comparison", BASE["n"], name, timed(fn, args.repeats), n_patches=64)

    return rows


def figures(df, plt):
    from plotstyle import series_colours

    axes = [("n_samples", "samples"), ("image_side", "image side (pixels)"),
            ("patch", "patch size"), ("n_bootstrap", "bootstrap resamples")]
    fig, ax = plt.subplots(1, len(axes), figsize=(4.0 * len(axes), 3.6))
    for a, (axis, label) in zip(ax, axes):
        sub = df[df.axis == axis]
        methods = list(dict.fromkeys(sub.method))
        for colour, m in zip(series_colours(len(methods)), methods):
            s = sub[sub.method == m].sort_values("value")
            a.plot(s.value, s.seconds, marker="o", color=colour, label=m)
        a.set_xlabel(label)
        a.set_ylabel("fit time (s), best of k")
        a.set_xscale("log", base=2)
        a.set_yscale("log")
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "timing_scaling.png", dpi=120)

    sub = df[df.axis == "method_comparison"].sort_values("seconds")
    fig2, ax2 = plt.subplots(figsize=(7, 3.8))
    ax2.barh(sub.method, sub.seconds, color=series_colours(len(sub)))
    ax2.set_xlabel("fit time (s), best of k, log scale")
    ax2.set_xscale("log")
    fig2.tight_layout()
    fig2.savefig(RESULTS / "timing_methods.png", dpi=120)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3,
                    help="timing repeats; the minimum is reported")
    ap.add_argument("--n-iter", type=int, default=300)
    ap.add_argument("--force", action="store_true",
                    help="run even when the machine is busy (recorded in the output)")
    ap.add_argument("--max-busy", type=float, default=25.0)
    args = ap.parse_args()

    before = snapshot(sample_seconds=4.0)
    ok, why = is_idle(before, max_busy=args.max_busy)
    print(f"machine check: {why}")
    if not ok and not args.force:
        print("\nREFUSING to measure timings on a busy machine, because the numbers "
              "would describe the load, not the code.\nWait for the machine to go quiet, "
              "or pass --force to record them as provisional.")
        return 2
    if not ok:
        print("WARNING: --force used; these timings are provisional, not quotable.")

    t0 = time.perf_counter()
    rows = run(args)
    after = snapshot(sample_seconds=4.0)
    total_min = (time.perf_counter() - t0) / 60

    import pandas as pd
    df = pd.DataFrame(rows)
    df["clean_run"] = bool(ok)
    df["cpu_busy_before"] = before["cpu_busy_percent"]
    df["cpu_busy_after"] = after["cpu_busy_percent"]
    df["competing_before"] = before["n_competing_python"]
    df.to_csv(RESULTS / "timing_results.csv", index=False)

    (RESULTS / "timing_machine.json").write_text(
        json.dumps({"before": before, "after": after, "clean_run": bool(ok),
                    "forced": bool(args.force and not ok),
                    "repeats": args.repeats, "n_iter": args.n_iter,
                    "total_minutes": total_min}, indent=2), encoding="utf-8")

    from plotstyle import apply_style
    figures(df, apply_style())

    worst = df.spread_ratio.max()
    tight = "(tight, consistent with a quiet machine)."
    wide = "(wide, evidence of interference during the run)."
    flag = "" if ok else "  <- provisional, not quotable"
    lines = [
        "# Timing and scaling results", "",
        f"Measured {before['timestamp']}, finished after {total_min:.1f} minutes.",
        "",
        f"- Machine: {before['cpu_count']} logical cores, "
        f"CPU {before['cpu_busy_percent']}% busy before and "
        f"{after['cpu_busy_percent']}% after, "
        f"{before['n_competing_python']} competing python processes at the start.",
        f"- Clean run (idleness precondition met): **{bool(ok)}**" + flag,
        f"- python {before['python']}, numpy {before['numpy']}, "
        f"scipy {before['scipy']}, scikit-learn {before['sklearn']}.",
        f"- Each figure is the best of {args.repeats} repeats. Largest median-to-minimum "
        f"spread seen: x{worst:.2f} " + (tight if worst < 1.15 else wide),
        "", "## Method comparison at n=400, 32x32, patch 4", "",
        "| method | best (s) | median (s) | spread |", "|---|---|---|---|",
    ]
    for _, r in df[df.axis == "method_comparison"].sort_values("seconds").iterrows():
        lines.append(f"| {r.method} | {r.seconds:.3f} | {r.median_seconds:.3f} "
                     f"| x{r.spread_ratio:.2f} |")
    lines += ["", "## Scaling", "",
              "Fit time against each axis, all methods, in `timing_scaling.png`. "
              "Raw numbers in `timing_results.csv`; machine conditions in "
              "`timing_machine.json`.", ""]
    for axis in ["n_samples", "image_side", "patch", "n_bootstrap", "n_representations"]:
        sub = df[df.axis == axis]
        if sub.empty:
            continue
        lines.append(f"- **{axis}**: {sub.value.min()} to {sub.value.max()}, "
                     f"fit time {sub.seconds.min():.2f}s to {sub.seconds.max():.2f}s.")
    (RESULTS / "TIMING_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\nwrote timing_results.csv, TIMING_SUMMARY.md, timing_scaling.png, "
          f"timing_methods.png  ({total_min:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
