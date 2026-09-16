"""Deep bug sweep: regression tests for defects found by adversarial review.

Each test here encodes either a bug that was found and fixed, or an edge case the
sweep checked and pinned down so it cannot regress silently. The docstrings say which.
"""
import numpy as np
import pytest

from robustpixelmaker import (
    BootstrapMaskSelector,
    NestedCV,
    PatchGrid,
    RobustPixelMaker,
    SoftMaskSelector,
    make_synthetic_images,
)
from robustpixelmaker.nested_cv import align_proba


# --------------------------------------------------------------------------- #
# Bug 1 (fixed): multiclass OOF crash when a training fold misses a rare class
# --------------------------------------------------------------------------- #
def test_align_proba_places_fold_columns_into_global_order():
    proba = np.array([[0.7, 0.3], [0.2, 0.8]])
    out = align_proba(proba, fold_classes=[0, 2], classes=[0, 1, 2])
    np.testing.assert_allclose(out[:, 0], proba[:, 0])
    np.testing.assert_allclose(out[:, 1], 0.0)     # class 1 unseen by this fold
    np.testing.assert_allclose(out[:, 2], proba[:, 1])


def test_align_proba_rejects_inconsistent_shapes():
    with pytest.raises(ValueError, match="columns"):
        align_proba(np.ones((3, 2)), fold_classes=[0, 1, 2], classes=[0, 1, 2])
    with pytest.raises(ValueError, match="did not expose"):
        align_proba(np.ones((3, 2)), fold_classes=None, classes=[0, 1, 2])


def test_align_proba_passthrough_when_widths_match():
    proba = np.array([[0.5, 0.5]])
    np.testing.assert_array_equal(align_proba(proba, None, [0, 1]), proba)


def test_nested_cv_survives_a_rare_class():
    """The engine-level regression: a class rarer than k_outer used to crash the
    out-of-fold accumulation, because one training fold missed it and predict_proba
    came back one column short. This is routine in the small-sample regime."""
    c = make_synthetic_images(n=150, task="regression", seed=0)
    q = np.quantile(c.y, [1 / 3, 2 / 3])
    y = np.digitize(c.y, q)
    rare = np.where(y == 2)[0]
    y[rare[3:]] = 1                      # leave exactly 3 members of class 2
    assert (y == 2).sum() == 3
    res = NestedCV(k_outer=5, l_inner=3, n_iter=4, random_state=0).run(c.X, y)
    assert res.oof_predictions.shape == (150, 3)
    # OOF rows are probability vectors wherever any fold predicted them
    sums = res.oof_predictions[res.oof_count > 0].sum(axis=1)
    np.testing.assert_allclose(sums, 1.0, atol=1e-6)


# --------------------------------------------------------------------------- #
# Bug 2 (fixed): resample-fit failures were silently absorbed as zero votes
# --------------------------------------------------------------------------- #
def test_all_failed_fits_raise_instead_of_reporting_nothing_stable():
    class Broken(BootstrapMaskSelector):
        def _one_fit(self, X, y, idx, seed, n_patches):
            return None                   # simulate a systematic inner-fit failure

    c = make_synthetic_images(n=80, seed=0)
    with pytest.raises(RuntimeError, match="systematic"):
        Broken(task="binary", n_bootstrap=4, n_iter=50, seed=0).fit(c.X, c.y)


def test_partial_failures_are_counted_not_hidden():
    class HalfBroken(BootstrapMaskSelector):
        def _one_fit(self, X, y, idx, seed, n_patches):
            if seed % 2 == 0:
                return None
            return super()._one_fit(X, y, idx, seed, n_patches)

    c = make_synthetic_images(n=120, seed=0)
    sel = HalfBroken(task="binary", n_bootstrap=6, n_iter=100, seed=0).fit(c.X, c.y)
    rep = sel.stability_report()
    assert rep["n_failed"] == 3
    assert rep["n_fits"] == 3


# --------------------------------------------------------------------------- #
# Bug 3 (fixed): task='binary' silently mislabelled 3-class data
# --------------------------------------------------------------------------- #
def test_binary_task_with_three_classes_is_an_error_not_a_guess():
    c = make_synthetic_images(n=90, seed=0)
    y3 = np.arange(90) % 3
    with pytest.raises(ValueError, match="3 classes"):
        SoftMaskSelector(task="binary", n_iter=20).fit(c.X, y3)


# --------------------------------------------------------------------------- #
# Edge cases the sweep checked and pinned
# --------------------------------------------------------------------------- #
def test_non_square_and_non_divisible_images():
    """Patch grids must handle H != W and patch sizes that do not divide the image."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 18, 30))            # 18x30 with patch 4 -> ragged edge
    y = (X[:, 2:6, 2:6].mean(axis=(1, 2)) > 0).astype(int)
    grid = PatchGrid.from_image_shape(X.shape, patch=4)
    assert grid.n_patches == 5 * 8               # ceil(18/4) x ceil(30/4)
    f = grid.pool(X)
    assert f.shape == (40, 40)
    # partial edge patches average over their true (smaller) pixel count
    ones = grid.pool(np.ones((1, 18, 30)))
    np.testing.assert_allclose(ones, 1.0)
    # and the selector runs end to end on the ragged grid
    sel = SoftMaskSelector(task="binary", patch=4, n_iter=150, seed=0).fit(X, y)
    assert sel.predict(X[:5]).shape == (5,)


def test_single_class_binary_degenerates_gracefully():
    """A resample can lose a class entirely; the fit must not crash or divide by zero."""
    c = make_synthetic_images(n=60, seed=0)
    y = np.zeros(60, dtype=int)
    sel = SoftMaskSelector(task="binary", n_iter=50, seed=0).fit(c.X, y)
    assert set(np.unique(sel.predict(c.X[:10]))) <= {0}


def test_constant_feature_columns_do_not_produce_nans():
    """Zero-variance patches (a constant border) must standardise safely, the exact
    failure family recorded for the sibling library (division by zero variance)."""
    c = make_synthetic_images(n=80, seed=0)
    X = c.X.copy()
    X[:, :4, :] = 7.0                            # constant band across every image
    sel = SoftMaskSelector(task="binary", n_iter=100, seed=0).fit(X, c.y)
    assert np.isfinite(sel.mask_).all()
    assert np.isfinite(sel.predict_proba(X[:5])).all()


def test_grid_expand_and_pixel_roundtrip():
    grid = PatchGrid(12, 12, patch=4)
    vec = np.arange(grid.n_patches, dtype=float)
    img = grid.expand(vec)
    assert img.shape == (12, 12)
    # every pixel of patch p carries value p
    for p in range(grid.n_patches):
        pix = grid.patches_to_pixels([p])
        assert np.allclose(img.ravel()[pix], p)


def test_facade_multiclass_with_rare_class_end_to_end():
    """The facade composes engine + selector; the rare-class path must survive it."""
    c = make_synthetic_images(n=150, task="regression", seed=0)
    q = np.quantile(c.y, [1 / 3, 2 / 3])
    y = np.digitize(c.y, q)
    rare = np.where(y == 2)[0]
    y[rare[3:]] = 1
    rpm = RobustPixelMaker(selector="single", patch=4, n_iter=150, k_outer=5,
                           random_state=0).fit(c.X, y)
    assert rpm.task_ == "multiclass"
    assert rpm.predict_proba(c.X[:4]).shape == (4, 3)


def test_result_oof_probabilities_are_normalised():
    c = make_synthetic_images(n=120, task="binary", seed=0)
    res = NestedCV(k_outer=4, l_inner=3, n_iter=4, random_state=0).run(c.X, c.y)
    sums = res.oof_predictions[res.oof_count > 0].sum(axis=1)
    np.testing.assert_allclose(sums, 1.0, atol=1e-9)
