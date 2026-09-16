"""Patch representations, and the ensemble over which selection is marginalised.

This is the machinery behind RobustPixelMaker's central claim (RELATED_WORK.md sec.12,
and the ablation that tests it in benchmarks/FINDINGS.md sec.9). The
objection to any pixel-selection method is that the answer depends on how the image was
represented: choose a different lens and different pixels look important. Rather than
picking one lens and defending it, RPM treats the representation as a **nuisance
variable** and integrates the selection over an ensemble of them, so what survives is
what is selected regardless of the lens.

The crucial invariant is that a representation changes *which patches rank highly*, but
never *the coordinate system of the answer*. Every representation here returns features
indexed ``(n_samples, n_patches, n_channels)``, the mask gates whole patches across
their channels, and the deliverable stays a set of pixels. This is exactly the
distinction drawn in RELATED_WORK.md between an evaluator lens (tolerable, and here
marginalised away) and a feature-space transform such as a radiomics collapse (fatal,
because it changes what the answer is indexed by).

Representations are deliberately numpy and scipy only, so the ensemble runs with no
torch dependency. ``RandomConvFeatures`` is the workhorse: random zero-mean filters
followed by a rectifier and patch pooling. Different seeds give genuinely different
lenses, which is the numpy-side analogue of "several convolutional networks at
different initialisations", and random convolutional features are a well established
strong representation in their own right. Learned convolutional backbones can later be
added as further ensemble members behind the same ``transform`` interface without any
change to the selection machinery.
"""
from __future__ import annotations

import numpy as np
from typing import Any
from numpy.typing import ArrayLike


class PatchMean:
    """Per-patch mean intensity: the plainest lens, and the Milestone-2 default."""

    name = "mean"

    def transform(self, X: ArrayLike, grid: Any) -> np.ndarray:
        return grid.pool(np.asarray(X, dtype=float))[:, :, None]


class PatchStats:
    """Per-patch mean, dispersion and local gradient energy (a texture-aware lens).

    The dispersion channel is computed in the shifted form, pooling the squared
    deviation from each patch's own mean, never as E[x^2] - E[x]^2. The textbook
    difference form loses the entire variance to catastrophic cancellation once the
    image sits at a large offset (unit-variance data at offset 1e8 cancels to noise)
    and overflows outright beyond |x| of about 1e154; the shifted form is exact at any
    offset the data itself can represent. Same defect family as the scale-invariance
    entries in the sibling library's records.
    """

    name = "stats"

    def transform(self, X: ArrayLike, grid: Any) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        mean = grid.pool(X)
        centred = X - mean[:, grid.pixel_patch]      # subtract each patch's own mean
        # max-normalise before squaring, then rescale: squaring raw deviations
        # overflows beyond |x| of about 1e154 even after centring
        scale = np.max(np.abs(centred), axis=(1, 2), keepdims=True)
        safe = np.where(scale == 0.0, 1.0, scale)
        std = scale.reshape(-1, 1) * np.sqrt(grid.pool((centred / safe) ** 2))
        gy = np.zeros_like(X)
        gx = np.zeros_like(X)
        gy[:, 1:, :] = np.diff(X, axis=1)
        gx[:, :, 1:] = np.diff(X, axis=2)
        grad = grid.pool(np.abs(gy) + np.abs(gx))
        return np.stack([mean, std, grad], axis=-1)


class GaussianScale:
    """Patch means of a blurred image: the same lens at a coarser spatial scale."""

    def __init__(self, sigma: float = 1.0):
        self.sigma = sigma
        self.name = f"blur{sigma:g}"

    def transform(self, X: ArrayLike, grid: Any) -> np.ndarray:
        from scipy.ndimage import gaussian_filter

        X = np.asarray(X, dtype=float)
        # blur within each image only, never across the batch axis
        sm = gaussian_filter(X, sigma=(0.0, self.sigma, self.sigma), mode="reflect")
        return grid.pool(sm)[:, :, None]


class RandomConvFeatures:
    """Random zero-mean convolutional filters, rectified and pooled per patch.

    Each seed yields a different lens. Filters are zero-meaned so a channel responds to
    local structure rather than to overall brightness, which would otherwise duplicate
    the plain mean channel.
    """

    def __init__(self, n_filters: int = 4, kernel: int = 3, seed: int = 0):
        self.n_filters = n_filters
        self.kernel = kernel
        self.seed = seed
        self.name = f"randconv{seed}"

    def _filters(self):
        rng = np.random.default_rng(self.seed)
        f = rng.normal(size=(self.n_filters, self.kernel, self.kernel))
        f -= f.mean(axis=(1, 2), keepdims=True)
        norm = np.sqrt((f ** 2).sum(axis=(1, 2), keepdims=True))
        return f / np.maximum(norm, 1e-8)

    def transform(self, X: ArrayLike, grid: Any) -> np.ndarray:
        from scipy.ndimage import convolve

        X = np.asarray(X, dtype=float)
        out = []
        for filt in self._filters():
            conv = convolve(X, filt[None, :, :], mode="reflect")
            out.append(grid.pool(np.maximum(conv, 0.0)))
        return np.stack(out, axis=-1)


def default_ensemble(n_representations: int = 4, seed: int = 0) -> list:
    """A deliberately diverse default ensemble.

    Diversity is the point: members that agree by construction would make the
    marginalisation vacuous. The set spans a raw intensity lens, a texture lens, a
    coarser spatial scale, and random convolutional lenses, so agreement between them
    is evidence rather than an artefact of shared design.
    """
    pool = [PatchMean(), PatchStats(), GaussianScale(sigma=1.0)]
    pool += [RandomConvFeatures(n_filters=4, seed=seed + 1000 + i) for i in range(8)]
    if n_representations > len(pool):
        pool += [RandomConvFeatures(n_filters=4, seed=seed + 2000 + i)
                 for i in range(n_representations - len(pool))]
    return pool[:n_representations]
