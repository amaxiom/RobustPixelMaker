"""Multiclass support and Hard-Concrete L0 gates (Milestone 2 extension)."""
import numpy as np
import pytest

from robustpixelmaker import PatchGrid, SoftMaskSelector, make_synthetic_images
from robustpixelmaker.masking import HC_GAMMA, HC_ZETA


def _iou(sel_pixels_flat, region_mask):
    sel = np.zeros(region_mask.size, dtype=bool)
    sel[sel_pixels_flat] = True
    reg = region_mask.ravel()
    union = np.logical_or(sel, reg).sum()
    return np.logical_and(sel, reg).sum() / union if union else 1.0


def _multiclass_control(n=480, seed=0):
    """Registered images whose class depends only on the informative region.

    Reuses the synthetic layout, then re-labels into 3 classes from the informative
    cells so the ground-truth region is unchanged and recovery stays measurable.
    """
    c = make_synthetic_images(n=n, task="regression", seed=seed)
    q = np.quantile(c.y, [1 / 3, 2 / 3])
    y = np.digitize(c.y, q)
    return c, y


# --- Hard-Concrete gate ---------------------------------------------------- #
def test_hardconcrete_recovers_region():
    c = make_synthetic_images(n=400, task="binary", seed=0)
    sel = SoftMaskSelector(task="binary", patch=4, lam=0.03,
                           gate="hardconcrete", seed=0).fit(c.X, c.y)
    assert _iou(sel.selected_pixels(), c.informative) > 0.9
    assert _iou(sel.selected_pixels(), c.distractor) < 0.05


def test_hardconcrete_mask_is_binary_at_test_time():
    """The deterministic gate must be exactly 0 or 1 for most patches (true L0)."""
    c = make_synthetic_images(n=300, task="binary", seed=0)
    sel = SoftMaskSelector(task="binary", gate="hardconcrete", lam=0.03, seed=0).fit(c.X, c.y)
    m = sel.mask_
    assert m.min() >= 0.0 and m.max() <= 1.0
    exact = np.mean(np.isclose(m, 0.0) | np.isclose(m, 1.0))
    assert exact > 0.9, f"only {exact:.2f} of gates are exactly 0/1"


def test_hardconcrete_stretch_constants_allow_exact_zero_and_one():
    # the stretch interval must extend beyond [0, 1] or gates can never close fully
    assert HC_GAMMA < 0.0 < 1.0 < HC_ZETA


def test_hardconcrete_reproducible():
    c = make_synthetic_images(n=300, seed=0)
    kw = dict(task="binary", gate="hardconcrete", lam=0.03, seed=0)
    np.testing.assert_array_equal(
        SoftMaskSelector(**kw).fit(c.X, c.y).mask_,
        SoftMaskSelector(**kw).fit(c.X, c.y).mask_,
    )


def test_hardconcrete_keeps_a_large_region_whole():
    """L0 penalises the gate COUNT, not magnitude, so a big true region survives."""
    c = make_synthetic_images(n=500, task="binary", shape=(48, 48), cell=6,
                              region_side=3, distractor_side=1, seed=0)
    sel = SoftMaskSelector(task="binary", patch=6, lam=0.03,
                           gate="hardconcrete", seed=0).fit(c.X, c.y)
    grid = PatchGrid.from_image_shape(c.X.shape, patch=6)
    info = set(np.unique(grid.pixel_patch[c.informative]).tolist())
    recall = len(set(sel.selected_patches_.tolist()) & info) / len(info)
    assert recall > 0.75, f"L0 gate dropped too much of the large region (recall {recall:.2f})"


# --- multiclass ------------------------------------------------------------ #
def test_multiclass_recovers_region_and_predicts():
    from sklearn.metrics import accuracy_score

    c, y = _multiclass_control(n=480, seed=0)
    sel = SoftMaskSelector(task="multiclass", patch=4, lam=0.03, seed=0).fit(c.X, y)
    assert _iou(sel.selected_pixels(), c.informative) > 0.5
    assert _iou(sel.selected_pixels(), c.distractor) < 0.2

    proba = sel.predict_proba(c.X)
    assert proba.shape == (len(y), 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-8)
    assert accuracy_score(y, sel.predict(c.X)) > 0.6  # 3 classes, chance is 0.33


def test_multiclass_labels_are_preserved():
    """predict must return the caller's original label values, not 0..K-1 indices."""
    c, y = _multiclass_control(n=300, seed=1)
    labels = np.array([10, 20, 30])[y]
    sel = SoftMaskSelector(task="multiclass", lam=0.03, seed=0).fit(c.X, labels)
    assert set(np.unique(sel.predict(c.X))).issubset({10, 20, 30})


def test_binary_with_nonzero_one_labels():
    """Binary labels such as {3, 8} (the MNIST subset) must work directly."""
    c = make_synthetic_images(n=300, task="binary", seed=0)
    labels = np.where(c.y == 1, 8, 3)
    sel = SoftMaskSelector(task="binary", lam=0.03, seed=0).fit(c.X, labels)
    assert set(np.unique(sel.predict(c.X))).issubset({3, 8})
    assert _iou(sel.selected_pixels(), c.informative) > 0.5


def test_invalid_task_and_gate_rejected():
    with pytest.raises(ValueError):
        SoftMaskSelector(task="ordinal")
    with pytest.raises(ValueError):
        SoftMaskSelector(gate="relu")


# --- spatial smoothness prior ---------------------------------------------- #
def test_tv_prior_runs_and_stays_localised():
    c = make_synthetic_images(n=300, task="binary", seed=0)
    sel = SoftMaskSelector(task="binary", lam=0.03, tv=0.01, seed=0).fit(c.X, c.y)
    assert _iou(sel.selected_pixels(), c.informative) > 0.4
