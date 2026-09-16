"""Side-by-side map of WHICH patches each method selects.

Aggregate metrics say how much each method keeps and how well it scores, but not
whether the kept region is a coherent structure or a scatter of individually ranked
patches. On scientific images that difference is the point: a spatially coherent
region is interpretable as a finding, a scatter is not.

For each method this plots the selected-patch map over the dataset mean image, and
reports coverage plus a contiguity score (the fraction of selected patches that have
at least one selected 4-neighbour).

Run:  py benchmarks/compare_selection_maps.py --dataset medmnist-blood
Output: benchmarks/results/selection_maps_<dataset>.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from robustpixelmaker import PatchGrid  # noqa: E402
from benchmarks.competitors import SELECTORS, ordered_methods  # noqa: E402
from benchmarks.datasets import LOADERS  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"


def contiguity(sel, grid) -> float:
    """Fraction of selected patches with at least one selected 4-neighbour."""
    if len(sel) < 2:
        return 0.0
    on = np.zeros((grid.n_py, grid.n_px), dtype=bool)
    for p in sel:
        on[int(p) // grid.n_px, int(p) % grid.n_px] = True
    nb = np.zeros_like(on)
    nb[1:, :] |= on[:-1, :]
    nb[:-1, :] |= on[1:, :]
    nb[:, 1:] |= on[:, :-1]
    nb[:, :-1] |= on[:, 1:]
    return float((on & nb).sum() / on.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="medmnist-blood")
    ap.add_argument("--n-max", type=int, default=3000)
    ap.add_argument("--patch", type=int, default=4)
    ap.add_argument("--methods", default="anova,rfe,rf,rpm,rpm_l0")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from benchmarks.plotstyle import apply_style
    plt = apply_style()

    X, y, meta = LOADERS[args.dataset](n_max=args.n_max)
    grid = PatchGrid.from_image_shape(X.shape, patch=args.patch)
    F = grid.pool(X)
    methods = ordered_methods([m.strip() for m in args.methods.split(",")])
    mean_img = X.mean(axis=0)

    fig, axes = plt.subplots(1, len(methods) + 1, figsize=(3.0 * (len(methods) + 1), 3.4))
    axes[0].imshow(mean_img, cmap="viridis")
    axes[0].set_title(f"{args.dataset}\nmean image", fontsize=9)
    axes[0].set_xticks([]); axes[0].set_yticks([])

    print(f"{args.dataset}: n={meta['n']} task={meta['task']} patches={grid.n_patches}")
    for a, m in zip(axes[1:], methods):
        sel = SELECTORS[m](X, F, y, grid, meta["task"], args.seed)
        keep = np.zeros(grid.n_patches)
        keep[np.asarray(sel, dtype=int)] = 1.0
        cov = len(sel) / grid.n_patches
        cont = contiguity(sel, grid)
        a.imshow(mean_img, cmap="viridis", alpha=0.45)
        a.imshow(grid.expand(keep), cmap="viridis", alpha=0.55, vmin=0, vmax=1)
        a.set_title(f"{m}\ncoverage {cov:.2f}, contiguity {cont:.2f}", fontsize=9)
        a.set_xticks([]); a.set_yticks([])
        print(f"  {m:7s} coverage={cov:.3f} contiguity={cont:.3f} n_selected={len(sel)}")

    fig.tight_layout()
    out = RESULTS / f"selection_maps_{args.dataset}.png"
    fig.savefig(out, dpi=115)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
