"""Making a masked region ABSENT rather than fabricated (Milestone 3).

When the mask is applied after pooling, as in Milestones 2 and 4, nothing spatial ever
mixes kept and dropped pixels, so how the dropped region is represented does not matter.
It starts to matter as soon as the mask is applied at the INPUT and a convolutional lens
looks at the result, which is the path the representation ensemble introduced. A filter
straddling the boundary of a dropped region then sees whatever was written there:

  zero fill   invents a hard edge at the boundary and a constant black region. Both are
              out of distribution, and the filter responds strongly to the artificial
              edge, so the response says more about the mask than about the image. This
              is precisely the failure partial convolutions were introduced to fix.
  mean fill   removes the constant-region artefact but keeps a (softer) boundary edge.
  renormalise the convolution is computed over the KEPT pixels only and rescaled by how
              many there were, so a partially covered window reports the average
              response of the evidence that actually exists. A dropped region then reads
              as missing rather than as a fabricated value.

``renormalised_convolve`` implements the third, following the partial-convolution
construction of Liu et al.: numerator over kept pixels, denominator counting them, with
windows that see no kept pixel at all returning zero and being reported as invalid.
"""
from __future__ import annotations

import numpy as np
from typing import Any, Sequence
from numpy.typing import ArrayLike


def fill_zero(X: ArrayLike, mask_img: ArrayLike) -> np.ndarray:
    """Write zeros into the dropped region (the naive, out-of-distribution choice)."""
    m = np.asarray(mask_img, dtype=float)
    return np.asarray(X, dtype=float) * m[None, :, :]


def fill_mean(X: ArrayLike, mask_img: ArrayLike,
              baseline: ArrayLike | None = None) -> np.ndarray:
    """Write a neutral baseline into the dropped region, per-pixel training mean."""
    Xa = np.asarray(X, dtype=float)
    m = np.asarray(mask_img, dtype=float)
    b = Xa.mean(axis=0) if baseline is None else np.asarray(baseline, dtype=float)
    return Xa * m[None, :, :] + b[None, :, :] * (1.0 - m)[None, :, :]


def fill_blur(X: ArrayLike, mask_img: ArrayLike, sigma: float = 2.0) -> np.ndarray:
    """Write a blurred version of the image itself into the dropped region."""
    from scipy.ndimage import gaussian_filter

    Xa = np.asarray(X, dtype=float)
    m = np.asarray(mask_img, dtype=float)
    sm = gaussian_filter(Xa, sigma=(0.0, sigma, sigma), mode="reflect")
    return Xa * m[None, :, :] + sm * (1.0 - m)[None, :, :]


def renormalised_convolve(X, mask_img, filt):
    """Partial convolution that measures contrast against the surviving evidence only.

    ``mask_img`` is (H, W) with 1 for kept and 0 for dropped. Returns the response and
    the per-position validity (whether the window contained any kept pixel at all).

    The textbook partial convolution takes the numerator over kept pixels and rescales
    by how many there were. That is right for a filter measuring LEVEL (one whose
    weights sum to something nonzero, such as a local average), but not for a zero-mean
    CONTRAST filter: over a partial support its weights no longer sum to zero, so a
    perfectly flat region still produces a response at a mask boundary, a fabricated
    edge of exactly the kind the construction exists to prevent.

    Which correction applies is therefore decided by the filter itself. A contrast
    filter is additionally re-centred over the surviving support,

        out = [ sum_kept f_i x_i  -  xbar_kept * sum_kept f_i ] * (mass_full / mass_kept)

    with ``xbar_kept`` the mean of the kept pixels in the window, which is identically
    zero on a flat region. A level filter keeps the plain rescaling, so a local average
    still reports the average of the evidence that survived. Both reduce exactly to a
    plain convolution when the window is fully kept.

    The rescaling is by kept filter MASS, not kept pixel COUNT (fixed 2026-09-07), and
    the right mass differs by branch:

      * LEVEL filter: ``mass = sum f_i``, i.e. the observed filter weight. This is an
        identity, not a tuning choice. On constant data x the numerator is
        ``x * mass_kept``, so rescaling returns ``x * mass_full`` exactly.
      * CONTRAST filter: ``mass = sqrt(sum f_i^2)``, the L2 norm over the kept support.
        The signed weights cancel, so they cannot normalise anything, and the absolute
        mass is not the right substitute either: after re-centring the response is
        ``sum_kept f_i r_i`` over residuals, whose magnitude for exchangeable residuals
        goes as ``sqrt(sum_kept f^2)``.

    Count-based rescaling, how the textbook partial convolution is usually written and
    how this function was first implemented, is exact only for a UNIFORM kernel, where
    every pixel carries the same weight so count is proportional to mass. For any other
    kernel it charges a dropped pixel the average weight rather than its own. On a flat
    image beside a straight gap edge a uniform 7x7 kernel was exact, a Gaussian 7x7
    sigma 1.5 was 74% wrong and a Gaussian 9x9 sigma 1.0 was 99.9% wrong, fabricating
    exactly the edge this construction exists to prevent. The unit tests missed it for
    a release because every level-filter test used ``ones / 9``, the one kernel shape
    for which the old code was right.

    Both count and L1 mass go as N/c for a random kept subset, so both over-correct a
    contrast response by roughly a square; the L2 ratio does not. Measured over 8 seeds
    recovering the full-data response of a random zero-mean 5x5 filter on smooth data:
    L2 gives 79.9% RMS error against count's 93.2% under 30% scattered dropout, and
    57.3% against 75.2% under patch dropout. Note that a contrast response under heavy
    dropout is hard to recover at all, which those magnitudes reflect; the level branch
    is where the decisive gain is (2.2x on a Gaussian under scattered dropout).

    RobustSignalMaker fixed the level half of this in its
    ``validity.renormalised_convolve1d`` on 2026-09-01 and normalises its contrast
    branch by L1 mass. That difference is now deliberate on both sides rather than a
    defect on either. The L2 rule was ported to RSM on 2026-09-07 and scored on the
    criterion that governs THAT library, truth recovery of the selection rather than
    fidelity of the response, and it lost: paired over 10 seeds, two dropout regimes
    and two contrast lenses, L1 beat L2 21 to 9 with 10 ties. RSM's default lens set
    includes a Savitzky-Golay derivative, whose coefficients are smooth and
    antisymmetric rather than iid-like, so the exchangeable-residual argument for L2
    does not hold there; RPM's lens is random zero-mean kernels only, where it does.
    Each library keeps the rule its own measurement supports (RSM FINDINGS sec.12).

    Validity now also requires the window to hold enough filter mass to normalise by,
    not merely one kept pixel. A window whose only kept pixels sit where the kernel is
    ~0 carries no usable evidence, and dividing by that mass would manufacture a large
    response from nothing; such positions return 0.0 with ``valid`` False.
    """
    from scipy.ndimage import convolve

    X = np.asarray(X, dtype=float)
    m = np.asarray(mask_img, dtype=float)
    filt = np.asarray(filt, dtype=float)
    ones = np.ones_like(filt)
    m3 = np.broadcast_to(m, (1,) + m.shape).copy()

    num = convolve(X * m[None, :, :], filt[None, :, :], mode="reflect")
    count = convolve(m3, ones[None, :, :], mode="reflect")[0]
    has_any = count > 0
    # exact reciprocal where there is evidence; no epsilon, which would otherwise leave
    # a residual response of order eps on a flat region and blunt the property claimed
    inv = np.where(has_any, 1.0 / np.where(has_any, count, 1.0), 0.0)

    is_contrast = abs(float(filt.sum())) < 1e-8 * max(1.0, float(np.abs(filt).sum()))
    if is_contrast:
        weight_kept = convolve(m3, filt[None, :, :], mode="reflect")[0]
        sum_kept = convolve(X * m[None, :, :], ones[None, :, :], mode="reflect")
        xbar = sum_kept * inv[None, :, :]
        num = num - xbar * weight_kept[None, :, :]
        # A contrast filter's signed weights cancel, so they cannot normalise
        # anything, and the right substitute is NOT the absolute mass. After
        # re-centring the response is sum_kept f_i r_i over residuals r. For
        # exchangeable residuals that sum has magnitude proportional to
        # sqrt(sum_kept f^2), so the magnitude-preserving rescale is the L2
        # ratio. Both the count ratio and the L1 ratio go as N/c for a random
        # kept subset, which over-corrects by roughly a square. Measured over
        # 8 seeds recovering the full-data response of a random zero-mean 5x5
        # filter: L2 79.9% RMS error against count 93.2% under 30% scattered
        # dropout, and 57.3% against 75.2% under patch dropout.
        mass_kept = np.sqrt(np.maximum(
            convolve(m3, (filt ** 2)[None, :, :], mode="reflect")[0], 0.0))
        mass_full = float(np.sqrt((filt ** 2).sum()))
    else:
        # A level filter IS exact under weight rescaling, and this is an
        # identity rather than a calibration choice: on constant data x the
        # numerator is x * mass_kept, so x * mass_full comes back exactly.
        mass_kept = convolve(m3, filt[None, :, :], mode="reflect")[0]
        mass_full = float(filt.sum())

    # the mask is dimensionless and so is the mass, so the only floor here is the
    # float-subnormal boundary; no epsilon ever meets a data magnitude
    tiny = 1e-12 * max(abs(mass_full), 1.0)
    valid = has_any & (np.abs(mass_kept) > tiny)
    scale = np.where(valid, mass_full / np.where(valid, mass_kept, 1.0), 0.0)
    return num * scale[None, :, :], valid


def masked_conv_features(X: ArrayLike, mask_img: ArrayLike, filters: Sequence[Any],
                         grid: Any, renormalise: bool = True) -> np.ndarray:
    """Rectified convolutional patch features computed on a masked image.

    With ``renormalise`` the dropped region is treated as absent (partial convolution);
    without it the mask is applied by plain multiplication, which is the zero-fill
    behaviour and the thing to beat.
    """
    from scipy.ndimage import convolve

    X = np.asarray(X, dtype=float)
    m = np.asarray(mask_img, dtype=float)
    out = []
    for filt in filters:
        if renormalise:
            conv, _ = renormalised_convolve(X, m, filt)
        else:
            conv = convolve(X * m[None, :, :], filt[None, :, :], mode="reflect")
        out.append(grid.pool(np.maximum(conv, 0.0)))
    return np.stack(out, axis=-1)


def fabricated_edge(X_flat: ArrayLike, mask_img: ArrayLike, filters: Sequence[Any],
                    eps: float = 1e-12) -> dict:
    """How much fabricated response each strategy invents at the mask boundary.

    The measurement needs a case where the right answer is known. On a FLAT image a
    zero-mean contrast filter must report zero everywhere, so any response is an edge
    that the masking invented rather than something in the data. Reported as the largest
    absolute response anywhere, relative to the image level; zero is perfect.

    An earlier version of this diagnostic compared masked features against the FULL
    image instead, which was wrong: the full image includes the dropped region's
    contribution, so that comparison rewards a treatment for letting dropped data keep
    leaking in, and penalised renormalisation for correctly excluding it.
    """
    from scipy.ndimage import convolve

    X = np.asarray(X_flat, dtype=float)
    m = np.asarray(mask_img, dtype=float)
    level = float(np.abs(X).mean()) + eps

    def _plain(img):
        return max(float(np.abs(convolve(img, f[None], mode="reflect")).max()) for f in filters)

    out = {
        "zero": _plain(fill_zero(X, m)),
        "mean": _plain(fill_mean(X, m)),
        "blur": _plain(fill_blur(X, m)),
    }
    out["renorm"] = max(
        float(np.abs(renormalised_convolve(X, m, f)[0]).max()) for f in filters)
    return {k: v / level for k, v in out.items()}
