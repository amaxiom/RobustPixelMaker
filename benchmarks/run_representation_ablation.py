"""Experiment 1: the representation-axis ablation (RELATED_WORK.md sec.12).

This is the make-or-break test for the method's novelty claim. Learnable-mask feature
selection, bootstrap stability selection and ensembling-for-stability all have prior
art. What is unclaimed is treating the REPRESENTATION as a nuisance variable and
marginalising the selection over it. If marginalising does not measurably improve
selection stability over a single fixed representation, the contribution collapses
into prior art and should be reported as such.

The comparison, all under the same leakage-safe fold structure and the same total
budget where possible:

  single:<name>   one fixed representation, bootstrap over data resamples only. This is
                  the prior-art configuration (a Concrete-Autoencoder-style learned
                  mask with stability selection bolted on).
  marginalised    the double bootstrap over data resamples AND representations.

Reported per configuration: cross-fold selection stability (the quantity the claim is
about), held-out score, coverage, and on synthetic data the recovery of the known
region. For the marginalised run we also report how much the single-lens selections
disagreed with each other, which is the direct measurement of how much the answer
depended on the lens in the first place.

Run:  py benchmarks/run_representation_ablation.py [--datasets synthetic,medmnist-breast]
Outputs: benchmarks/results/representation_ablation.csv and REPRESENTATION_ABLATION.md
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

from robustpixelmaker import (  # noqa: E402
    BootstrapMaskSelector,
    PatchGrid,
    RepresentationEnsembleSelector,
    default_ensemble,
    make_synthetic_images,
    mean_pairwise_jaccard,
)
from benchmarks.competitors import recovery_metrics, score_selection  # noqa: E402
from benchmarks.datasets import LOADERS  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"


def load(name, n_max, seed):
    """Return X, y, task, and the ground-truth regions when they exist."""
    if name == "synthetic":
        c = make_synthetic_images(n=400, task="binary", noise=2.0, seed=seed)
        return c.X, c.y, "binary", c.informative, c.distractor
    X, y, meta = LOADERS[name](n_max=n_max)
    return X, y, meta["task"], None, None


def folds(F, y, task, seed, k):
    from sklearn.model_selection import KFold, StratifiedKFold
    sp = (KFold if task == "regression" else StratifiedKFold)(
        n_splits=k, shuffle=True, random_state=seed)
    return list(sp.split(F, y))


def evaluate(X, y, task, grid, F, splits, seed, mode, rep=None, n_reps=4,
             n_bootstrap=5, n_iter=300, info_p=None, dist_p=None):
    sc, cov, sels, agree, amb = [], [], [], [], []
    t0 = time.perf_counter()
    for tr, te in splits:
        if mode == "single":
            s = _single_rep_fit(X[tr], y[tr], task, grid, rep, n_bootstrap, n_iter, seed)
        else:
            s = RepresentationEnsembleSelector(
                task=task, patch=grid.patch, lam=0.03, n_bootstrap=n_bootstrap,
                n_representations=n_reps, tau=0.6, n_iter=n_iter, seed=seed).fit(X[tr], y[tr])
            agree.append(s.representation_agreement())
        p = np.asarray(s.selected_patches_, dtype=int)
        sc.append(score_selection(F[tr], y[tr], F[te], y[te], p, task, seed))
        cov.append(len(p) / grid.n_patches)
        sels.append(p)
        amb.append(s.ambiguous_fraction())
    out = dict(score=float(np.nanmean(sc)), coverage=float(np.mean(cov)),
               stability=mean_pairwise_jaccard(sels),
               ambiguous=float(np.nanmean(amb)),
               rep_agreement=float(np.nanmean(agree)) if agree else float("nan"),
               seconds=time.perf_counter() - t0)
    if info_p is not None:
        recs = [recovery_metrics(s, info_p, dist_p, grid.n_patches) for s in sels]
        out["f1"] = float(np.nanmean([r["f1"] for r in recs]))
    else:
        out["f1"] = float("nan")
    return out


class _RepBootstrap(BootstrapMaskSelector):
    """BootstrapMaskSelector pinned to one representation (the prior-art control)."""

    def __init__(self, representation, **kw):
        super().__init__(**kw)
        self._rep = representation

    def _inner_kwargs(self):
        kw = super()._inner_kwargs()
        kw["representation"] = self._rep
        return kw


def _single_rep_fit(Xtr, ytr, task, grid, rep, n_bootstrap, n_iter, seed):
    return _RepBootstrap(rep, task=task, patch=grid.patch, lam=0.03,
                         n_bootstrap=n_bootstrap, tau=0.6, n_iter=n_iter,
                         seed=seed).fit(Xtr, ytr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="synthetic,medmnist-breast")
    ap.add_argument("--n-max", type=int, default=2000)
    ap.add_argument("--patch", type=int, default=4)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--n-reps", type=int, default=4)
    ap.add_argument("--n-bootstrap", type=int, default=5)
    args = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)

    rows = []
    for name in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        for seed in range(args.seeds):
            X, y, task, info, dist = load(name, args.n_max, seed)
            grid = PatchGrid.from_image_shape(X.shape, patch=args.patch)
            F = grid.pool(X)
            splits = folds(F, y, task, seed, args.folds)
            info_p = None if info is None else np.unique(grid.pixel_patch[info])
            dist_p = None if dist is None else np.unique(grid.pixel_patch[dist])
            reps = default_ensemble(args.n_reps, seed=seed)
            print(f"\n=== {name} seed={seed}: n={len(y)} task={task} "
                  f"patches={grid.n_patches} ===", flush=True)

            for rep in reps:
                m = evaluate(X, y, task, grid, F, splits, seed, "single", rep=rep,
                             n_bootstrap=args.n_bootstrap, info_p=info_p, dist_p=dist_p)
                rows.append(dict(dataset=name, seed=seed,
                                 config=f"single:{getattr(rep,'name','rep')}", **m))
                print(f"  single:{getattr(rep,'name','rep'):<10s} score={m['score']:.4f} "
                      f"cov={m['coverage']:.2f} stab={m['stability']:.2f} "
                      f"f1={m['f1']:.2f} ({m['seconds']:.0f}s)", flush=True)

            m = evaluate(X, y, task, grid, F, splits, seed, "marginalised",
                         n_reps=args.n_reps, n_bootstrap=args.n_bootstrap,
                         info_p=info_p, dist_p=dist_p)
            rows.append(dict(dataset=name, seed=seed, config="marginalised", **m))
            print(f"  MARGINALISED      score={m['score']:.4f} cov={m['coverage']:.2f} "
                  f"stab={m['stability']:.2f} f1={m['f1']:.2f} "
                  f"rep_agree={m['rep_agreement']:.2f} ({m['seconds']:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "representation_ablation.csv", index=False)

    lines = ["# Representation-axis ablation (Experiment 1)", "",
             "The make-or-break test for the novelty claim. `single:<lens>` fixes one "
             "representation and bootstraps over data only (the prior-art "
             "configuration); `marginalised` is the double bootstrap over data AND "
             "representations. The quantity the claim is about is **stability**, the "
             "mean pairwise Jaccard of the selected region across outer folds. "
             "`rep_agreement` is how much the individual lenses agreed with each other: "
             "a low value means a single-lens answer was lens-specific.", ""]
    for ds, sub in df.groupby("dataset"):
        agg = sub.groupby("config")[["score", "coverage", "stability", "f1",
                                     "ambiguous", "rep_agreement", "seconds"]].mean()
        singles = agg.loc[[i for i in agg.index if i.startswith("single")]]
        lines += [f"## {ds}", "",
                  "| config | score | coverage | stability | recovery F1 |",
                  "| --- | --- | --- | --- | --- |"]
        for cfg, r in agg.iterrows():
            f1 = "n/a" if np.isnan(r["f1"]) else f"{r['f1']:.3f}"
            lines.append(f"| {cfg} | {r['score']:.4f} | {r['coverage']:.3f} | "
                         f"{r['stability']:.3f} | {f1} |")
        best = singles["stability"].max()
        mean_single = singles["stability"].mean()
        marg = agg.loc["marginalised", "stability"] if "marginalised" in agg.index else float("nan")
        lines += ["",
                  f"Best single lens stability {best:.3f}, mean single lens "
                  f"{mean_single:.3f}, marginalised **{marg:.3f}** "
                  f"(delta vs mean single {marg - mean_single:+.3f}, "
                  f"vs best single {marg - best:+.3f}).",
                  f"Lens agreement: {agg.loc['marginalised','rep_agreement']:.3f}"
                  if "marginalised" in agg.index else "", ""]
    (RESULTS / "REPRESENTATION_ABLATION.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nDONE. wrote representation_ablation.csv and REPRESENTATION_ABLATION.md", flush=True)


if __name__ == "__main__":
    main()
