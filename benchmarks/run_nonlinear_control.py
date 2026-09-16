"""Is RPM's synthetic advantage real, or is the control matched to its model class?

`make_synthetic_images` builds the label as `agg = S.sum(axis=1)` then `y = agg > 0`:
a LINEAR function of the informative cell signals. RPM's mask sits under a linear or
logistic head. So on the shipped control, RPM's inductive bias is correct by
construction while a random forest has to approximate an additive rule with trees.
That is exactly the setting where RPM reports its largest win (recovery F1 0.984
against 0.737), and it cannot be taken at face value until the label rule is varied.

This script keeps the planted region, the noise, the cells and the geometry identical
and changes only how the cell signals combine into the label:

  linear        y = sum(S) > 0                    the shipped control
  interaction   y = product of cell signs > 0     XOR-like, no additive main effect
  threshold     y = all cells above their median  conjunctive, needs every cell
  saturating    y = tanh(sum) with a dead zone    monotone but strongly nonlinear

Recovery F1 against the planted region is the criterion, because it is the only one
with a known answer and it is the metric the synthetic claim rests on. If RPM's margin
over RF survives the nonlinear rules, the advantage is about finding regions. If it
collapses, the advantage was about the control agreeing with RPM's head.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from competitors import recovery_metrics                      # noqa: E402
from robustpixelmaker import PatchGrid                        # noqa: E402
from robustpixelmaker.synthetic import _grid_coords           # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"
RULES = ("linear", "interaction", "threshold", "saturating")


def make_control(rule, n=400, shape=(32, 32), cell=4, region_side=2,
                 distractor_side=2, signal=2.0, noise=1.0, seed=0):
    """The shipped control's geometry exactly, with a swappable label rule."""
    rng = np.random.default_rng(seed)
    H, W = shape
    r0 = (H - region_side * cell) // 2
    c_info = (W // 2 - region_side * cell) // 2
    c_dist = W // 2 + (W // 2 - distractor_side * cell) // 2
    info = _grid_coords(r0, c_info, region_side, cell)
    dist = _grid_coords(r0, c_dist, distractor_side, cell)

    X = rng.normal(scale=noise, size=(n, H, W))
    S = rng.normal(size=(n, len(info)))
    D = rng.normal(size=(n, len(dist)))          # equal variance, non-predictive
    for k, (rr, cc) in enumerate(info):
        X[:, rr:rr + cell, cc:cc + cell] += S[:, k, None, None] * signal
    for k, (rr, cc) in enumerate(dist):
        X[:, rr:rr + cell, cc:cc + cell] += D[:, k, None, None] * signal

    if rule == "linear":
        y = (S.sum(axis=1) > 0).astype(int)
    elif rule == "interaction":
        y = (np.prod(np.sign(S), axis=1) > 0).astype(int)
    elif rule == "threshold":
        y = (S > np.median(S, axis=0)).all(axis=1).astype(int)
    elif rule == "saturating":
        a = np.tanh(S).sum(axis=1)
        y = (np.abs(a) > np.median(np.abs(a))).astype(int)
    else:
        raise ValueError(rule)

    info_mask = np.zeros((H, W), bool)
    dist_mask = np.zeros((H, W), bool)
    for rr, cc in info:
        info_mask[rr:rr + cell, cc:cc + cell] = True
    for rr, cc in dist:
        dist_mask[rr:rr + cell, cc:cc + cell] = True
    return X, y, info_mask, dist_mask


BANNER_PREC = "\n=== recovery PRECISION, and the chance rate it must beat ==="
BANNER_COV = "\n=== coverage (fraction of patches kept) ==="


def select(method, X, F, y, grid, seed, lam_scale="fixed"):
    """One selection. ``lam_scale`` applies to the MLP arm only, and matters.

    At a fixed lam an expressive head finds a use for every input, so the mask stops
    closing: on the interaction rule the MLP kept 47% of the image and scored recovery
    precision 0.082, which is exactly the chance rate for 4 informative patches of 49.
    That was first read as a fix and is an artefact of not pruning (FINDINGS sec.17).
    ``auto`` searches lam for a mask that actually closes, so it is the setting under
    which the interaction question can be asked at all. The linear arm is unaffected
    by the choice and is always run fixed, so it stays comparable across both runs.
    """
    if method in ("rpm_boot", "rpm_mlp"):
        from robustpixelmaker.selection import BootstrapMaskSelector
        mlp = method == "rpm_mlp"
        s = BootstrapMaskSelector(task="binary", patch=grid.patch, lam=0.03,
                                  n_iter=400, n_bootstrap=10, tau=0.6,
                                  gate="hardconcrete", head="mlp" if mlp else "linear",
                                  lam_scale=lam_scale if mlp else "fixed",
                                  seed=seed).fit(X, y)
        return s.selected_patches_
    if method == "rf":
        from sklearn.ensemble import RandomForestClassifier
        imp = RandomForestClassifier(n_estimators=200,
                                     random_state=seed).fit(F, y).feature_importances_
        return np.where(imp > imp.mean())[0].astype(int)
    raise ValueError(method)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--lam-scale", default="fixed", choices=["fixed", "auto"],
                    help="penalty scaling for the MLP arm; 'auto' searches lam so "
                         "the mask actually closes (FINDINGS sec.17, sec.18)")
    ap.add_argument("--out-prefix", default="nonlinear_control")
    args = ap.parse_args()

    rows = []
    for rule in RULES:
        for seed in range(args.seeds):
            X, y, info_mask, dist_mask = make_control(rule, n=args.n, seed=seed)
            if len(np.unique(y)) < 2:
                continue
            grid = PatchGrid.from_image_shape(X.shape, patch=4)
            F = grid.pool(X)
            info_p = np.unique(grid.pixel_patch[info_mask])
            dist_p = np.unique(grid.pixel_patch[dist_mask])
            for method in ("rpm_boot", "rpm_mlp", "rf"):
                sel = select(method, X, F, y, grid, seed, args.lam_scale)
                rec = recovery_metrics(sel, info_p, dist_p, grid.n_patches)
                rows.append(dict(rule=rule, seed=seed, method=method,
                                 lam_scale=args.lam_scale, balance=float(y.mean()),
                                 n_info=len(info_p), n_patches=grid.n_patches,
                                 chance=len(info_p) / grid.n_patches,
                                 coverage=len(sel) / grid.n_patches, **rec))
            a, b_, c_ = rows[-3], rows[-2], rows[-1]
            print(f"  {rule:12s} seed={seed}  linear F1={a['f1']:.3f}  "
                  f"mlp F1={b_['f1']:.3f}  rf F1={c_['f1']:.3f}  "
                  f"(balance {a['balance']:.2f})", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / f"{args.out_prefix}.csv", index=False)
    print(f"\n=== recovery F1 by label rule (MLP lam_scale={args.lam_scale}) ===")
    piv = df.pivot_table(index="rule", columns="method", values="f1").round(3)
    piv["mlp_gain"] = (piv.rpm_mlp - piv.rpm_boot).round(3)
    piv["mlp_vs_rf"] = (piv.rpm_mlp - piv.rf).round(3)
    print(piv.reindex(RULES).to_string())

    # F1 alone cannot separate recovery from refusing to prune: keeping every patch
    # scores recall 1.0 and precision equal to the chance rate. Precision read
    # against that rate is what distinguishes the two (FINDINGS sec.17).
    print(BANNER_PREC)
    prec = df.pivot_table(index="rule", columns="method", values="precision").round(3)
    # divide by the EXACT chance rate, not the rounded one shown in the column: with
    # 4 informative patches of 64 the rate is 0.0625, and dividing by a displayed
    # 0.062 reports 16.13x where the true figure is 16.00x
    chance = df.groupby("rule").chance.first()
    prec["mlp_over_chance"] = (prec.rpm_mlp / chance).round(2)
    prec["chance"] = chance.round(4)
    print(prec.reindex(RULES).to_string())

    print(BANNER_COV)
    print(df.pivot_table(index="rule", columns="method",
                         values="coverage").round(3).reindex(RULES).to_string())
    print(f"\nwrote {args.out_prefix}.csv")


if __name__ == "__main__":
    main()
