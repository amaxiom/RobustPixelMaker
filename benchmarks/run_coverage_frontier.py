"""Matched-coverage comparison: can RPM beat random-forest importance on leanness?

The headline benchmark compares each selector at whatever coverage it happens to land
on, and that is not a fair test of leanness. Random-forest importance keeps every
feature scoring above the mean importance, which lands near a third of the patches by
arithmetic rather than by choice; RPM's `target_coverage` is an explicit dial. Comparing
0.33 against 0.45 therefore compares two different questions.

This script gives **every** method the same dial and sweeps it. At each target coverage
c, each selector returns its best ceil(c * P) patches:

  rpm       the stability-selection frequency map thresholded at the c-th quantile
  rf        the top ceil(c*P) feature importances
  anova     the top ceil(c*P) univariate F statistics
  mi        the top ceil(c*P) by mutual information (a nonlinear ranker, since the
            F test only sees linear dependence and RPM's advantage on the synthetic
            control was precision under nonlinearity)

Everything downstream is held fixed: same folds, same patch features, and a common
Random Forest refit on whatever was selected, so a difference between arms is a
difference in *which patches were chosen* and nothing else.

Two readings matter, and they are not the same question:

  * At matched coverage, whose score is higher?
  * At matched score, who needs fewer patches? (the leanness question actually asked)

Run:  py benchmarks/run_coverage_frontier.py [--datasets mnist-3v8,medmnist-blood]
Outputs: results/coverage_frontier.csv, COVERAGE_FRONTIER.md, coverage_frontier.png
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from robustpixelmaker import BootstrapMaskSelector, PatchGrid, mean_pairwise_jaccard  # noqa: E402
from competitors import score_selection                                               # noqa: E402
from datasets import LOADERS                                                          # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

TARGETS = [0.05, 0.10, 0.15, 0.25, 0.40]
DEFAULT_DATASETS = ["mnist-3v8", "medmnist-blood", "medmnist-pneumonia", "medmnist-breast"]


def top_k(scores, k):
    """Indices of the k largest scores, NaNs treated as worthless."""
    s = np.nan_to_num(np.asarray(scores, dtype=float), nan=-np.inf)
    return np.sort(np.argsort(s)[::-1][:k]).astype(int)


def folds(y, task, seed, n_splits):
    from sklearn.model_selection import KFold, StratifiedKFold
    sp = (KFold if task == "regression" else StratifiedKFold)(
        n_splits=n_splits, shuffle=True, random_state=seed)
    return list(sp.split(np.zeros(len(y)), y))


def score_with(downstream, F_tr, y_tr, F_te, y_te, sel, task, seed):
    """Score a selection with a chosen downstream model.

    The suite's default downstream model is a Random Forest, which quietly advantages
    one competitor: random-forest importance ranks features by their usefulness *to a
    random forest*, so selector and evaluator share a model family and an inductive
    bias. Swapping the downstream model is the control that separates "these are better
    patches" from "these patches suit the judge". `rf` here reproduces the suite default
    exactly, so the two protocols stay comparable.
    """
    if downstream == "rf":
        return score_selection(F_tr, y_tr, F_te, y_te, sel, task, seed)

    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import mean_squared_error, roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    if len(sel) == 0:
        return 0.5 if task != "regression" else float(np.sqrt(
            mean_squared_error(y_te, np.full_like(y_te, y_tr.mean(), dtype=float))))
    Xtr, Xte = F_tr[:, sel], F_te[:, sel]
    if task == "regression":
        pred = make_pipeline(StandardScaler(), Ridge()).fit(Xtr, y_tr).predict(Xte)
        return float(np.sqrt(mean_squared_error(y_te, pred)))
    if downstream == "logreg":
        est = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    elif downstream == "svm":
        est = make_pipeline(StandardScaler(), SVC(probability=True, random_state=seed))
    else:
        raise ValueError(f"unknown downstream model {downstream!r}")
    proba = est.fit(Xtr, y_tr).predict_proba(Xte)
    if task == "multiclass":
        return float(roc_auc_score(y_te, proba, multi_class="ovr", average="weighted"))
    return float(roc_auc_score(y_te, proba[:, 1]))


def select_at(method, X_tr, F_tr, y_tr, grid, task, k, seed):
    """Exactly k patches from `method`, so every arm sits at the same coverage."""
    if method == "rpm":
        sel = BootstrapMaskSelector(task=task, patch=grid.patch, lam=0.03,
                                    n_bootstrap=10, n_iter=350, seed=seed).fit(X_tr, y_tr)
        # the frequency map is a ranking; take its top k rather than a fixed tau, which
        # is the same operating-point choice the other arms are being given. top_patches
        # breaks the map's heavy ties on gate strength, which matters most at small k
        return sel.top_patches(k)
    if method == "rpm_mlp":
        # lam_scale="auto": at a fixed lam the MLP head leaves every gate open on
        # multiclass, and a flat mask ranks by index order rather than by data
        sel = BootstrapMaskSelector(task=task, patch=grid.patch, lam=0.03,
                                    n_bootstrap=10, n_iter=350, head="mlp",
                                    lam_scale="auto", seed=seed).fit(X_tr, y_tr)
        return sel.top_patches(k)
    if method == "rpm_pi":
        # the pre-fix behaviour, kept so the tie-breaking effect stays measurable
        sel = BootstrapMaskSelector(task=task, patch=grid.patch, lam=0.03,
                                    n_bootstrap=10, n_iter=350, seed=seed).fit(X_tr, y_tr)
        return top_k(sel.pi_, k)
    if method == "rf":
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        RF = RandomForestRegressor if task == "regression" else RandomForestClassifier
        est = RF(n_estimators=200, random_state=seed).fit(F_tr, y_tr)
        return top_k(est.feature_importances_, k)
    if method == "anova":
        from sklearn.feature_selection import f_classif, f_regression
        f = f_regression if task == "regression" else f_classif
        return top_k(f(F_tr, y_tr)[0], k)
    if method == "mi":
        from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
        f = mutual_info_regression if task == "regression" else mutual_info_classif
        return top_k(f(F_tr, y_tr, random_state=seed), k)
    raise ValueError(method)


def run_dataset(name, n_max, patch, k_folds, seed, methods, downstream):
    X, y, meta = LOADERS[name](n_max=n_max)
    task = "multiclass" if len(np.unique(y)) > 2 else "binary"
    grid = PatchGrid.from_image_shape(X.shape, patch=patch)
    F = grid.pool(X)
    P = grid.n_patches
    rows = []
    for target in TARGETS:
        k = max(1, int(round(target * P)))
        for method in methods:
            t0 = time.perf_counter()
            scores, sels = [], []
            for tr, te in folds(y, task, seed, k_folds):
                sel = select_at(method, X[tr], F[tr], y[tr], grid, task, k, seed)
                scores.append(score_with(downstream, F[tr], y[tr], F[te], y[te], sel, task, seed))
                sels.append(sel)
            rows.append(dict(
                dataset=name, method=method, target_coverage=target, downstream=downstream,
                k=k, n_patches=P, coverage=k / P,
                score=float(np.mean(scores)), score_sd=float(np.std(scores)),
                stability=float(mean_pairwise_jaccard(sels)),
                seconds=time.perf_counter() - t0))
            r = rows[-1]
            print(f"  {name:22s} cov={target:.2f} {method:6s} "
                  f"score={r['score']:.4f} stab={r['stability']:.2f} "
                  f"({r['seconds']:.0f}s)", flush=True)
    return rows


def figure(df, plt, prefix='coverage_frontier'):
    from plotstyle import series_colours
    dsets = list(dict.fromkeys(df.dataset))
    fig, ax = plt.subplots(1, len(dsets), figsize=(4.2 * len(dsets), 3.8))
    ax = np.atleast_1d(ax)
    methods = list(dict.fromkeys(df.method))
    colours = dict(zip(methods, series_colours(len(methods))))
    for a, d in zip(ax, dsets):
        sub = df[df.dataset == d]
        for m in methods:
            s = sub[sub.method == m].sort_values("coverage")
            a.plot(s.coverage, s.score, marker="o", color=colours[m], label=m)
        a.set_title(d, fontsize=10)
        a.set_xlabel("coverage (fraction of patches kept)")
        a.set_ylabel("held-out score")
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / f"{prefix}.png", dpi=120)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    ap.add_argument("--methods", default="rpm,rf,anova,mi")
    ap.add_argument("--n-max", type=int, default=3000)
    ap.add_argument("--patch", type=int, default=4)
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--downstream", default="rf", choices=["rf", "logreg", "svm"],
                    help="downstream model; rf is the suite default and shares a "
                         "model family with the rf selector")
    ap.add_argument("--out-prefix", default="coverage_frontier")
    args = ap.parse_args()

    methods = args.methods.split(",")
    rows = []
    for name in args.datasets.split(","):
        print(f"\n=== {name} ===", flush=True)
        rows += run_dataset(name, args.n_max, args.patch, args.folds, args.seed,
                            methods, args.downstream)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / f"{args.out_prefix}.csv", index=False)

    from plotstyle import apply_style
    figure(df, apply_style(), args.out_prefix)

    lines = ["# Matched-coverage frontier: RPM against the competitors", "",
             "Every arm is given the same dial and returns exactly the same number of "
             "patches, so a difference here is a difference in which patches were "
             "chosen. Downstream classifier, folds and features are identical.", ""]
    for d, g in df.groupby("dataset"):
        lines += [f"## {d}", "", "| coverage | " +
                  " | ".join(sorted(g.method.unique())) + " | best |",
                  "|---" * (len(g.method.unique()) + 2) + "|"]
        for target, gg in g.groupby("target_coverage"):
            by = gg.set_index("method").score
            best = by.idxmax()
            cells = " | ".join(f"{by[m]:.4f}" for m in sorted(by.index))
            lines.append(f"| {target:.2f} | {cells} | **{best}** |")
        wins = g.loc[g.groupby("target_coverage").score.idxmax()].method.value_counts()
        lines += ["", "Wins by coverage level: " +
                  ", ".join(f"{m} {n}" for m, n in wins.items()), ""]
    overall = df.loc[df.groupby(["dataset", "target_coverage"]).score.idxmax()]
    tally = overall.method.value_counts()
    lines += ["## Overall", "",
              "Operating points won: " + ", ".join(f"**{m}** {n}" for m, n in tally.items()),
              ""]
    (RESULTS / f"{args.out_prefix.upper()}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\nwrote coverage_frontier.csv, COVERAGE_FRONTIER.md, coverage_frontier.png")


if __name__ == "__main__":
    main()
