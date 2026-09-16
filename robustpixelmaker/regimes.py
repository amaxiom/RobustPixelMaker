"""Regime and granularity logic, plus the input-validation helper.

`registered=True` (the v1 regime) uses a single global mask shared across all images
in a common coordinate frame. Selection happens at a coarse granularity (patch grid
here; SLIC superpixels and atlas ROIs are later backends) to beat spatial
autocorrelation: neighbouring pixels are near-equivalent, so a pixel-level mask
never stabilises, whereas a patch/superpixel/ROI mask does. See RELATED_WORK.md sec.11.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def ensure_finite(arr: ArrayLike, name: str) -> np.ndarray:
    """Raise a clear error on NaN or infinite entries instead of computing garbage.

    RobustPixelMaker performs no imputation, so a NaN cannot mean anything here; left
    alone it propagates silently through pooling, standardisation and the mask fit and
    emerges as a confident-looking selection of nothing in particular. That is the
    guards-fail-toward-confidence defect family: "could not compute" must never be
    converted into a plausible number. Following the parent framework's policy,
    non-finite input is an explicit error naming the count, never a silent pass.
    """
    arr = np.asarray(arr, dtype=float)
    if not np.isfinite(arr).all():
        n_nan = int(np.isnan(arr).sum())
        n_inf = int(np.isinf(arr).sum())
        raise ValueError(
            f"{name} contains non-finite values ({n_nan} NaN, {n_inf} infinite). "
            "RobustPixelMaker does not impute; clean or remove these values first.")
    return arr


class PatchGrid:
    """A regular patch partition of a fixed (registered) 2-D coordinate frame.

    Provides pooling (image stack to per-patch means) and expansion (per-patch vector
    to pixel map). Pooling pads the frame with zeros up to a multiple of the patch
    size, sums each patch block by a reshape, and divides by the TRUE pixel count of
    each patch, so ragged edge patches average over exactly the pixels they contain
    and the padding contributes nothing.

    An earlier implementation materialised a dense (H*W, n_patches) pooling matrix.
    That was exact and fast on the 28x28 test fixtures but needed O(H*W*n_patches)
    memory, which reaches gigabytes on a single realistic micrograph (8.6 GB at
    512x512 with patch 8), a violation of the framework's memory constraint hidden in
    the one helper every estimator shares. This implementation is exact, allocates
    only a padded copy of the batch, and is covered by an equality test against a
    naive per-patch reference.
    """

    def __init__(self, height: int, width: int, patch: int = 4):
        if patch < 1:
            raise ValueError("patch must be >= 1")
        self.height = int(height)
        self.width = int(width)
        self.patch = int(patch)
        self.n_py = (self.height + patch - 1) // patch
        self.n_px = (self.width + patch - 1) // patch
        self.n_patches = self.n_py * self.n_px

        # pixel -> patch id map (height, width)
        ys = np.arange(self.height) // patch
        xs = np.arange(self.width) // patch
        self.pixel_patch = (ys[:, None] * self.n_px + xs[None, :]).astype(int)

        # true pixel count per patch (edge patches are smaller when patch does not
        # divide the frame)
        self._counts = np.bincount(self.pixel_patch.ravel(),
                                   minlength=self.n_patches).astype(float)
        self._pad_h = self.n_py * self.patch - self.height
        self._pad_w = self.n_px * self.patch - self.width

    @classmethod
    def from_image_shape(cls, shape, patch: int = 4) -> "PatchGrid":
        if len(shape) < 2:
            raise ValueError("need at least (H, W)")
        return cls(shape[-2], shape[-1], patch=patch)

    def pool(self, X: ArrayLike) -> np.ndarray:
        """(n, H, W) -> (n, n_patches) per-patch means."""
        X = np.asarray(X, dtype=float)
        if X.ndim != 3 or X.shape[1] != self.height or X.shape[2] != self.width:
            raise ValueError(
                f"image batch of shape {X.shape} does not match grid "
                f"(n, {self.height}, {self.width})")
        if self._pad_h or self._pad_w:
            X = np.pad(X, ((0, 0), (0, self._pad_h), (0, self._pad_w)))
        n = X.shape[0]
        sums = (X.reshape(n, self.n_py, self.patch, self.n_px, self.patch)
                 .sum(axis=(2, 4))
                 .reshape(n, self.n_patches))
        return sums / self._counts[None, :]

    def expand(self, patch_vec: ArrayLike) -> np.ndarray:
        """(n_patches,) -> (H, W) pixel map by nearest-patch assignment."""
        patch_vec = np.asarray(patch_vec)
        return patch_vec[self.pixel_patch]

    def patches_to_pixels(self, patch_ids: ArrayLike) -> np.ndarray:
        """Flat pixel indices covered by the given patch ids."""
        ids = sorted({int(p) for p in np.asarray(patch_ids).ravel()})
        mask = np.isin(self.pixel_patch.ravel(), ids)
        return np.where(mask)[0]
