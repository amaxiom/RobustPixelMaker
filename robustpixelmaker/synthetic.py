"""Synthetic control: registered images with a KNOWN informative region.

Milestone-2 validation fixture and the substrate for the synthetic benchmark
(README.md, Why). Each image shares a common coordinate frame (registered). The
informative region is a ``region_side`` x ``region_side`` grid of independent cells,
each carrying its own latent, and the target depends on the SUM of those latents, so
every informative cell is jointly necessary (no single cell suffices). A distractor
region of equal-size cells carries independent latents that do NOT enter the target,
and the rest of the frame is noise.

This design is what makes the control discriminating: a correct selector must recover
ALL informative cells (each adds independent signal) while rejecting both the noise
and the equal-variance distractor. Sweeping ``n``, ``shape``, ``noise``, ``signal``,
``region_side`` and ``distractor_side`` covers size, shape, noise, and complexity.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SyntheticControl:
    X: np.ndarray            # (n, H, W)
    y: np.ndarray            # (n,)
    informative: np.ndarray  # (H, W) bool, ground-truth predictive region
    distractor: np.ndarray   # (H, W) bool, non-predictive equal-variance region
    task: str


def _grid_coords(top: int, left: int, side: int, cell: int):
    """Row-major top-left coords of a side x side grid of `cell`-sized blocks."""
    return [(top + i * cell, left + j * cell) for i in range(side) for j in range(side)]


def make_synthetic_images(
    n: int = 300,
    shape=(32, 32),
    task: str = "binary",
    signal: float = 2.0,
    noise: float = 1.0,
    cell: int = 4,
    region_side: int = 2,
    distractor_side: int = 2,
    seed: int = 0,
) -> SyntheticControl:
    """Generate a registered synthetic control (see module docstring).

    ``region_side``/``distractor_side`` are the grid side lengths; the informative
    region has ``region_side**2`` independent cells. Defaults reproduce the original
    2x2 informative + 2x2 distractor layout. Raises if the requested cells do not fit.
    """
    rng = np.random.default_rng(seed)
    H, W = shape
    r0 = H // 4
    c_info = W // 4
    c_dist = W // 4 + region_side * cell + cell   # one-cell gap after the info band

    info_coords = _grid_coords(r0, c_info, region_side, cell) if region_side > 0 else []
    dist_coords = _grid_coords(r0, c_dist, distractor_side, cell) if distractor_side > 0 else []
    for (rr, cc) in info_coords + dist_coords:
        if rr + cell > H or cc + cell > W:
            raise ValueError(
                f"cells do not fit in {shape} (cell={cell}, region_side={region_side}, "
                f"distractor_side={distractor_side}); use a larger shape or fewer/smaller cells"
            )

    S = rng.normal(size=(n, len(info_coords)))
    D = rng.normal(size=(n, len(dist_coords)))
    X = rng.normal(scale=noise, size=(n, H, W))
    informative = np.zeros((H, W), dtype=bool)
    distractor = np.zeros((H, W), dtype=bool)
    for k, (rr, cc) in enumerate(info_coords):
        X[:, rr:rr + cell, cc:cc + cell] += S[:, k, None, None] * signal
        informative[rr:rr + cell, cc:cc + cell] = True
    for k, (rr, cc) in enumerate(dist_coords):
        X[:, rr:rr + cell, cc:cc + cell] += D[:, k, None, None] * signal
        distractor[rr:rr + cell, cc:cc + cell] = True

    agg = S.sum(axis=1) if info_coords else np.zeros(n)
    if task == "regression":
        y = agg + rng.normal(scale=0.3, size=n)
    elif task == "binary":
        y = (agg > 0).astype(int) if info_coords else (rng.normal(size=n) > 0).astype(int)
    else:
        raise ValueError("synthetic control supports task in {'binary','regression'}")

    return SyntheticControl(X=X, y=y, informative=informative, distractor=distractor, task=task)
