"""Milestone 2: synthetic control + soft-mask selection recovers the planted region."""
import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from robustpixelmaker import (
    NestedCV,
    PatchGrid,
    SoftMaskFoldEstimator,
    SoftMaskSelector,
    make_synthetic_images,
)


def _iou(sel_pixels_flat, region_mask):
    sel = np.zeros(region_mask.size, dtype=bool)
    sel[sel_pixels_flat] = True
    reg = region_mask.ravel()
    inter = np.logical_and(sel, reg).sum()
    union = np.logical_or(sel, reg).sum()
    return inter / union if union else 1.0


# --- the control is set up as intended ------------------------------------- #
def test_signal_is_localised_to_informative_region():
    c = make_synthetic_images(n=400, seed=0)
    grid = PatchGrid.from_image_shape(c.X.shape, patch=4)
    feats = grid.pool(c.X)  # (n, P) patch means

    info_patches = np.unique(grid.pixel_patch[c.informative])
    dist_patches = np.unique(grid.pixel_patch[c.distractor])

    # target depends on the SUM of the informative cells; the distractor sum does not
    auc_info = roc_auc_score(c.y, feats[:, info_patches].sum(axis=1))
    auc_dist = roc_auc_score(c.y, feats[:, dist_patches].sum(axis=1))
    assert max(auc_info, 1 - auc_info) > 0.85
    assert abs(auc_dist - 0.5) < 0.1


# --- the mask recovers the region ------------------------------------------ #
def test_softmask_recovers_region_binary():
    c = make_synthetic_images(n=400, task="binary", seed=0)
    sel = SoftMaskSelector(task="binary", patch=4, lam=0.03, seed=0).fit(c.X, c.y)

    iou = _iou(sel.selected_pixels(), c.informative)
    # overlap with the planted region is high, distractor is largely rejected
    assert iou > 0.5, f"IoU with informative region too low: {iou:.2f}"
    dist_overlap = _iou(sel.selected_pixels(), c.distractor)
    assert dist_overlap < 0.2, f"distractor wrongly selected: {dist_overlap:.2f}"
    # most of the frame is dropped
    assert sel.coverage() < 0.35
    # predictive performance retained on held-out data
    c_te = make_synthetic_images(n=200, task="binary", seed=99)
    auc = roc_auc_score(c_te.y, sel.predict_proba(c_te.X)[:, 1])
    assert auc > 0.8


def test_softmask_recovers_region_regression():
    c = make_synthetic_images(n=400, task="regression", seed=1)
    sel = SoftMaskSelector(task="regression", patch=4, lam=0.03, seed=0).fit(c.X, c.y)
    assert _iou(sel.selected_pixels(), c.informative) > 0.5
    assert _iou(sel.selected_pixels(), c.distractor) < 0.2


# --- sparsity knob behaves ------------------------------------------------- #
def test_higher_lambda_selects_fewer_patches():
    c = make_synthetic_images(n=400, task="binary", seed=0)
    low = SoftMaskSelector(task="binary", patch=4, lam=0.005, seed=0).fit(c.X, c.y)
    high = SoftMaskSelector(task="binary", patch=4, lam=0.15, seed=0).fit(c.X, c.y)
    assert high.coverage() <= low.coverage()


# --- reproducibility ------------------------------------------------------- #
def test_softmask_reproducible():
    c = make_synthetic_images(n=300, seed=0)
    m1 = SoftMaskSelector(seed=0).fit(c.X, c.y).mask_
    m2 = SoftMaskSelector(seed=0).fit(c.X, c.y).mask_
    np.testing.assert_array_equal(m1, m2)


# --- integrates with the M1 nested-CV engine ------------------------------- #
def test_softmask_fold_estimator_in_nested_cv():
    c = make_synthetic_images(n=300, task="binary", seed=0)
    factory = lambda: SoftMaskFoldEstimator(patch=4, lam=0.03, n_iter=300)
    res = NestedCV(estimator_factory=factory, k_outer=4, random_state=0).run(c.X, c.y)
    assert len(res.per_fold_scores) == 4
    assert res.mean_score > 0.7          # recovers signal under nested CV
    assert 0.0 <= res.selection_stability <= 1.0
    # reproducible end to end
    res2 = NestedCV(estimator_factory=factory, k_outer=4, random_state=0).run(c.X, c.y)
    np.testing.assert_array_equal(res.per_fold_scores, res2.per_fold_scores)
