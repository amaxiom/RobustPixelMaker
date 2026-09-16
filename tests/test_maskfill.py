"""Milestone 3: making a dropped region absent rather than fabricated."""
import numpy as np
import pytest

from robustpixelmaker import PatchGrid, make_synthetic_images
from robustpixelmaker.maskfill import (
    fabricated_edge,
    fill_blur,
    fill_mean,
    fill_zero,
    masked_conv_features,
    renormalised_convolve,
)


def _setup(n=40, seed=0):
    c = make_synthetic_images(n=n, seed=seed)
    grid = PatchGrid.from_image_shape(c.X.shape, patch=4)
    H, W = c.X.shape[1:]
    mask = np.ones((H, W))
    mask[:, W // 2:] = 0.0          # drop the right half
    rng = np.random.default_rng(0)
    filts = rng.normal(size=(3, 3, 3))
    filts -= filts.mean(axis=(1, 2), keepdims=True)
    return c, grid, mask, filts


# --- the fills behave as described ----------------------------------------- #
def test_fill_zero_writes_zeros_and_keeps_the_rest():
    c, grid, mask, _ = _setup()
    out = fill_zero(c.X, mask)
    assert np.allclose(out[:, :, mask[0] == 0], 0.0)
    np.testing.assert_allclose(out[:, :, mask[0] == 1], c.X[:, :, mask[0] == 1])


def test_fill_mean_and_blur_preserve_the_kept_region():
    c, grid, mask, _ = _setup()
    keep = mask[0] == 1
    for out in (fill_mean(c.X, mask), fill_blur(c.X, mask)):
        np.testing.assert_allclose(out[:, :, keep], c.X[:, :, keep])
    # and they do NOT write zeros into the dropped region
    assert not np.allclose(fill_mean(c.X, mask)[:, :, ~keep], 0.0)


# --- the partial convolution ------------------------------------------------ #
def test_renormalised_convolve_ignores_dropped_pixels():
    """A fully kept window must match a plain convolution of the same data."""
    from scipy.ndimage import convolve

    c, grid, mask, filts = _setup()
    conv, valid = renormalised_convolve(c.X, np.ones_like(mask), filts[0])
    plain = convolve(c.X, filts[0][None], mode="reflect")
    np.testing.assert_allclose(conv, plain, atol=1e-8)
    assert valid.all()


def test_renormalised_convolve_marks_fully_dropped_windows_invalid():
    c, grid, mask, filts = _setup()
    zero_mask = np.zeros_like(mask)
    conv, valid = renormalised_convolve(c.X, zero_mask, filts[0])
    assert not valid.any()
    assert np.allclose(conv, 0.0)


def test_level_filter_reports_the_average_of_surviving_pixels():
    """A LEVEL filter (weights summing to nonzero) keeps the plain rescaling."""
    X = np.ones((1, 8, 8))
    m = np.ones((8, 8))
    m[:, 4:] = 0.0
    filt = np.ones((3, 3)) / 9.0
    conv, valid = renormalised_convolve(X, m, filt)
    # deep inside the kept region the answer is the unmasked one
    assert conv[0, 4, 1] == pytest.approx(1.0, abs=1e-6)
    # straddling the boundary it is still about 1, not pulled toward 0 by the drop
    assert conv[0, 4, 3] == pytest.approx(1.0, abs=1e-6)


# --- the claim Milestone 3 exists to test ----------------------------------- #
def test_renormalisation_invents_no_edge_on_a_flat_image():
    """The claim Milestone 3 exists to test, on a case with a known right answer.

    On a flat image a zero-mean contrast filter must report zero everywhere. Any
    response at the mask boundary is an edge the masking invented. Zero fill should be
    the worst offender and renormalisation should be essentially exact.
    """
    _, grid, mask, filts = _setup()
    flat = np.full((1, mask.shape[0], mask.shape[1]), 3.0)
    art = fabricated_edge(flat, mask, filts)
    assert art["renorm"] < 1e-12, art
    assert art["zero"] > 0.1, art
    assert art["renorm"] < art["zero"], art
    assert art["renorm"] < art["blur"], art


def test_renormalisation_reduces_to_plain_convolution_when_nothing_is_dropped():
    from scipy.ndimage import convolve

    c, grid, mask, filts = _setup()
    conv, valid = renormalised_convolve(c.X, np.ones_like(mask), filts[0])
    np.testing.assert_allclose(conv, convolve(c.X, filts[0][None], mode="reflect"), atol=1e-8)
    assert valid.all()


def test_masked_conv_features_shape_and_finiteness():
    c, grid, mask, filts = _setup()
    f = masked_conv_features(c.X, mask, filts, grid, renormalise=True)
    assert f.shape == (c.X.shape[0], grid.n_patches, len(filts))
    assert np.isfinite(f).all()


def test_contrast_and_level_filters_are_handled_differently():
    """The correction applied depends on what the filter measures, and must be automatic."""
    m = np.ones((8, 8)); m[:, 4:] = 0.0
    flat = np.full((1, 8, 8), 3.0)

    level = np.ones((3, 3)) / 9.0                       # sums to 1: measures level
    contrast = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], float)
    contrast -= contrast.mean()                          # sums to 0: measures contrast

    lvl, _ = renormalised_convolve(flat, m, level)
    con, _ = renormalised_convolve(flat, m, contrast)
    # on a flat image the level filter reports the level, the contrast filter reports zero
    assert lvl[0, 4, 3] == pytest.approx(3.0, abs=1e-9)
    assert abs(con[0, 4, 3]) < 1e-12


def test_mean_fill_invents_an_edge_when_the_baseline_does_not_match():
    """The discriminating case between renormalisation and a neutral fill.

    A flat image is too kind to mean fill: writing the mean into flat data creates no
    edge. The realistic case is a baseline that does not match the local content, which
    is the norm, since a per-pixel training mean is not this image's local value. Then
    the fill writes a step at the boundary and the contrast filter fires on it, whereas
    renormalisation never looks at the dropped side and reports nothing.
    """
    m = np.ones((8, 8)); m[:, 4:] = 0.0
    flat = np.full((1, 8, 8), 3.0)
    mismatched = np.full((8, 8), 9.0)          # baseline unlike the actual content
    contrast = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], float)
    contrast -= contrast.mean()

    from scipy.ndimage import convolve
    filled = fill_mean(flat, m, baseline=mismatched)
    edge_from_fill = float(np.abs(convolve(filled, contrast[None], mode="reflect")).max())
    renorm_edge = float(np.abs(renormalised_convolve(flat, m, contrast)[0]).max())

    assert edge_from_fill > 1.0, edge_from_fill      # the fill fabricated a step
    assert renorm_edge < 1e-12                        # renormalisation invented nothing


# --- the 2026-09-07 mass-rescaling fix -------------------------------------- #
# Every level-filter test above uses ``ones / 9``, the one kernel shape for which
# count-based and weight-based rescaling coincide. That is why the defect below
# survived a release: the suite was green throughout. These pin the general case.

def _gauss2d(n, sigma):
    a = np.arange(n) - (n - 1) / 2.0
    g = np.exp(-(a ** 2) / (2 * sigma ** 2))
    k = np.outer(g, g)
    return k / k.sum()


@pytest.mark.parametrize("kernel", [
    np.ones((7, 7)) / 49.0,      # uniform: the case the old code got right
    _gauss2d(7, 1.5),
    _gauss2d(9, 1.0),
    _gauss2d(5, 0.7),            # very peaked: count and weight diverge hardest
])
def test_level_filter_on_constant_data_returns_the_constant_for_any_kernel(kernel):
    """A LEVEL filter must report the level, whatever the kernel's shape.

    This is an identity, not a tolerance: on constant data the numerator is
    ``x * mass_kept``, so rescaling by ``mass_full / mass_kept`` returns ``x``
    exactly. Under the old count-based rescaling the Gaussian kernels were wrong
    by 74% and 99.9% of the level, fabricating an edge on a perfectly flat image
    beside a gap, which is the exact failure this module exists to prevent.
    """
    X = np.full((1, 21, 21), 5.0)
    mask = np.ones((21, 21))
    mask[:, :8] = 0.0                       # a straight gap edge, worst case
    out, valid = renormalised_convolve(X, mask, kernel)
    assert valid.any()
    np.testing.assert_allclose(out[0][valid], 5.0, atol=1e-12)


def test_contrast_filter_on_constant_data_is_exactly_zero_for_any_kernel():
    """The re-centring must hold for non-uniform contrast kernels too."""
    X = np.full((1, 21, 21), 5.0)
    mask = np.ones((21, 21))
    mask[:, :8] = 0.0
    rng = np.random.default_rng(0)
    for _ in range(4):
        f = rng.normal(size=(5, 5))
        f -= f.mean()
        out, valid = renormalised_convolve(X, mask, f)
        assert np.abs(out[0][valid]).max() < 1e-12


def test_rescaling_is_by_filter_mass_not_pixel_count():
    """Pin the denominator directly, so a revert to count-based fails loudly.

    A level filter's response on constant data equals ``mass_full / mass_kept``
    times ``x * mass_kept``. Choosing a kernel whose kept mass fraction differs
    sharply from its kept COUNT fraction separates the two rules: here the gap
    removes half the pixels but far more than half the Gaussian's weight.
    """
    kernel = _gauss2d(9, 1.0)
    X = np.full((1, 25, 25), 4.0)
    mask = np.ones((25, 25))
    mask[:, :13] = 0.0
    out, valid = renormalised_convolve(X, mask, kernel)

    kept_count_frac = 1.0 - 0.5                      # about half the window
    kept_mass_frac = kernel[:, 4:].sum() / kernel.sum()
    assert abs(kept_mass_frac - kept_count_frac) > 0.05, "kernel does not separate the rules"

    # the weight rule gives exactly 4.0; the count rule would not
    col = 13                                          # first fully observed column
    assert out[0, 12, col] == pytest.approx(4.0, abs=1e-12)


def test_contrast_branch_uses_the_l2_ratio():
    """A contrast response scales as sqrt(sum f^2), so the rescale is the L2 ratio.

    Count and absolute-mass ratios both go as N/c for a random kept subset and so
    over-correct by roughly a square. Checked here against a directly computed L2
    reference rather than against a remembered number.
    """
    from scipy.ndimage import convolve

    rng = np.random.default_rng(3)
    X = rng.normal(size=(1, 21, 21)) * 2.0
    f = rng.normal(size=(5, 5))
    f -= f.mean()
    mask = np.ones((21, 21))
    mask[rng.random((21, 21)) < 0.3] = 0.0

    out, valid = renormalised_convolve(X, mask, f)

    ones = np.ones_like(f)
    m3 = np.broadcast_to(mask, (1,) + mask.shape).copy()
    num = convolve(X * mask[None], f[None], mode="reflect")
    count = convolve(m3, ones[None], mode="reflect")[0]
    inv = np.where(count > 0, 1.0 / np.where(count > 0, count, 1.0), 0.0)
    wk = convolve(m3, f[None], mode="reflect")[0]
    sk = convolve(X * mask[None], ones[None], mode="reflect")
    num = num - (sk * inv[None]) * wk[None]
    kept_l2 = np.sqrt(np.maximum(convolve(m3, (f ** 2)[None], mode="reflect")[0], 0.0))
    ref = num * np.where(valid, np.sqrt((f ** 2).sum()) / np.where(valid, kept_l2, 1.0), 0.0)[None]

    np.testing.assert_allclose(out[0][valid], ref[0][valid], rtol=1e-10)


def test_a_window_with_evidence_but_no_filter_mass_is_invalid():
    """Kept pixels sitting where the kernel is ~0 carry no usable evidence.

    Validity used to be ``count > 0``, so such a window returned a number obtained
    by dividing a near-zero numerator by a near-zero mass. It must refuse instead.
    """
    kernel = np.zeros((5, 5))
    kernel[2, 2] = 1.0                       # all the weight at the centre
    X = np.ones((1, 9, 9))
    mask = np.zeros((9, 9))
    mask[0, 0] = 1.0                         # one kept pixel, nowhere near any centre

    out, valid = renormalised_convolve(X, mask, kernel)
    # the window centred far from the kept pixel has evidence in view but no mass
    assert not valid[4, 4]
    assert out[0, 4, 4] == 0.0
    assert np.isfinite(out).all()


def test_uniform_kernels_are_unchanged_by_the_fix():
    """Backward compatibility: for a uniform kernel count and weight coincide.

    Anything computed with ``ones / n`` before the fix must be bit-comparable
    after it, which is what makes this a fix rather than a change of definition.
    """
    from scipy.ndimage import convolve

    rng = np.random.default_rng(5)
    X = rng.normal(size=(1, 24, 24)) * 3.0 + 1.0
    mask = np.ones((24, 24))
    mask[rng.random((24, 24)) < 0.25] = 0.0
    kernel = np.ones((5, 5)) / 25.0

    out, valid = renormalised_convolve(X, mask, kernel)

    ones = np.ones_like(kernel)
    m3 = np.broadcast_to(mask, (1,) + mask.shape).copy()
    num = convolve(X * mask[None], kernel[None], mode="reflect")
    count = convolve(m3, ones[None], mode="reflect")[0]
    ok = count > 0
    old = num * np.where(ok, kernel.size / np.where(ok, count, 1.0), 0.0)[None]

    np.testing.assert_allclose(out[0][valid], old[0][valid], rtol=1e-12)


def test_masked_conv_features_without_renormalisation_is_the_zero_fill_path():
    """``renormalise=False`` is the baseline the module exists to beat.

    It applies the mask by plain multiplication, so a dropped pixel contributes a
    zero rather than being treated as absent, and the two paths must differ near
    a boundary while agreeing when nothing is dropped.
    """
    c, grid, mask, filts = _setup()
    on = masked_conv_features(c.X, mask, filts, grid, renormalise=True)
    off = masked_conv_features(c.X, mask, filts, grid, renormalise=False)
    assert on.shape == off.shape
    assert np.isfinite(off).all()
    assert not np.allclose(on, off)

    full = np.ones_like(mask)
    np.testing.assert_allclose(
        masked_conv_features(c.X, full, filts, grid, renormalise=True),
        masked_conv_features(c.X, full, filts, grid, renormalise=False),
        atol=1e-8)


def test_masked_convolution_feeds_no_selection_path():
    """Pin the scoping fact that licenses judging maskfill on response fidelity alone.

    `renormalised_convolve` is currently evaluated by how faithfully it reproduces the
    full-data response, which is defensible ONLY because no selection consumes it: the
    mask is applied after pooling, so nothing spatial ever mixes kept and dropped
    pixels. If that changes, the L1-versus-L2 normalisation choice has to be re-measured
    on selection quality instead, and this test is what will notice.
    """
    import pathlib

    import robustpixelmaker

    # locate the package by importing it, not by guessing that it sits beside tests/.
    # Running the suite against an INSTALLED copy is a release check, and the guess
    # makes this test fail there for a reason that has nothing to do with what it pins.
    pkg = pathlib.Path(robustpixelmaker.__file__).resolve().parent
    selection_path = ["masking.py", "selection.py", "representations.py",
                      "core.py", "nested_cv.py", "regimes.py"]
    offenders = [m for m in selection_path
                 if "maskfill" in (pkg / m).read_text(encoding="utf-8")]
    assert not offenders, (
        f"{offenders} now import maskfill, so masked convolution has entered the "
        "selection path. Re-measure the contrast normalisation on recovery F1 against "
        "the planted region, not on response fidelity; see CHANGELOG.md.")
