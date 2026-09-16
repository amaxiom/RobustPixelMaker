"""Milestone 5: representations and the double bootstrap over data and lenses."""
import numpy as np
import pytest

from robustpixelmaker import (
    GaussianScale,
    PatchGrid,
    PatchMean,
    PatchStats,
    RandomConvFeatures,
    RepresentationEnsembleSelector,
    SoftMaskSelector,
    default_ensemble,
    make_synthetic_images,
)


def _iou(sel_pixels_flat, region_mask):
    sel = np.zeros(region_mask.size, dtype=bool)
    sel[sel_pixels_flat] = True
    reg = region_mask.ravel()
    union = np.logical_or(sel, reg).sum()
    return np.logical_and(sel, reg).sum() / union if union else 1.0


# --- the invariant the whole claim rests on -------------------------------- #
@pytest.mark.parametrize("rep", [PatchMean(), PatchStats(), GaussianScale(1.0),
                                 RandomConvFeatures(n_filters=3, seed=1)])
def test_every_representation_keeps_the_patch_index(rep):
    """A lens may change which patches rank highly, never what the answer indexes.

    Each representation must return (n, n_patches, channels) on the SAME grid, so the
    selection stays pixel-indexed. This is the line between an evaluator lens and a
    feature-space transform such as a radiomics collapse.
    """
    c = make_synthetic_images(n=40, seed=0)
    grid = PatchGrid.from_image_shape(c.X.shape, patch=4)
    f = rep.transform(c.X, grid)
    assert f.ndim == 3
    assert f.shape[0] == 40
    assert f.shape[1] == grid.n_patches
    assert f.shape[2] >= 1
    assert np.isfinite(f).all()


def test_representations_are_actually_different():
    """Members that agreed by construction would make the marginalisation vacuous."""
    c = make_synthetic_images(n=40, seed=0)
    grid = PatchGrid.from_image_shape(c.X.shape, patch=4)
    a = PatchMean().transform(c.X, grid)[:, :, 0]
    b = RandomConvFeatures(n_filters=3, seed=1).transform(c.X, grid)[:, :, 0]
    d = RandomConvFeatures(n_filters=3, seed=2).transform(c.X, grid)[:, :, 0]
    assert not np.allclose(a, b)
    assert not np.allclose(b, d)  # different seeds give different lenses


def test_random_conv_is_reproducible_for_a_seed():
    c = make_synthetic_images(n=30, seed=0)
    grid = PatchGrid.from_image_shape(c.X.shape, patch=4)
    r1 = RandomConvFeatures(n_filters=3, seed=7).transform(c.X, grid)
    r2 = RandomConvFeatures(n_filters=3, seed=7).transform(c.X, grid)
    np.testing.assert_array_equal(r1, r2)


def test_default_ensemble_is_diverse_and_sized():
    reps = default_ensemble(5, seed=0)
    assert len(reps) == 5
    assert len({getattr(r, "name", str(i)) for i, r in enumerate(reps)}) == 5


# --- multichannel mask fitting --------------------------------------------- #
def test_multichannel_representation_still_recovers_region():
    """The mask gates whole patches across channels, so recovery must survive."""
    c = make_synthetic_images(n=300, task="binary", seed=0)
    sel = SoftMaskSelector(task="binary", patch=4, lam=0.03,
                           representation=PatchStats(), seed=0).fit(c.X, c.y)
    assert sel._n_channels == 3
    assert _iou(sel.selected_pixels(), c.informative) > 0.5
    assert _iou(sel.selected_pixels(), c.distractor) < 0.2


def test_single_channel_path_unchanged_by_refactor():
    """PatchMean as an explicit representation must equal the built-in default."""
    c = make_synthetic_images(n=200, seed=0)
    a = SoftMaskSelector(task="binary", lam=0.03, seed=0).fit(c.X, c.y)
    b = SoftMaskSelector(task="binary", lam=0.03, representation=PatchMean(),
                         seed=0).fit(c.X, c.y)
    np.testing.assert_allclose(a.mask_, b.mask_)


def test_multichannel_predicts_and_scores():
    from sklearn.metrics import roc_auc_score

    c = make_synthetic_images(n=300, task="binary", seed=0)
    sel = SoftMaskSelector(task="binary", lam=0.03, representation=PatchStats(),
                           seed=0).fit(c.X, c.y)
    te = make_synthetic_images(n=150, task="binary", seed=99)
    assert roc_auc_score(te.y, sel.predict_proba(te.X)[:, 1]) > 0.75


# --- the double bootstrap --------------------------------------------------- #
def test_ensemble_marginalises_over_both_axes():
    c = make_synthetic_images(n=250, task="binary", seed=0)
    sel = RepresentationEnsembleSelector(
        task="binary", patch=4, n_representations=3, n_bootstrap=4,
        n_iter=250, seed=0).fit(c.X, c.y)
    assert sel.n_fits_ == 3 * 4                      # representations x resamples
    assert len(sel.pi_by_representation_) == 3
    assert sel.pi_.min() >= 0.0 and sel.pi_.max() <= 1.0
    assert _iou(sel.selected_pixels(), c.informative) > 0.5


def test_representation_agreement_is_reported():
    c = make_synthetic_images(n=250, task="binary", seed=0)
    sel = RepresentationEnsembleSelector(
        task="binary", n_representations=3, n_bootstrap=3, n_iter=250, seed=0).fit(c.X, c.y)
    agree = sel.representation_agreement()
    assert np.isnan(agree) or 0.0 <= agree <= 1.0
    rep = sel.stability_report()
    assert rep["n_representations"] == 3
    assert "representation_agreement" in rep


def test_ensemble_reproducible():
    c = make_synthetic_images(n=200, seed=0)
    kw = dict(task="binary", n_representations=2, n_bootstrap=3, n_iter=200, seed=0)
    np.testing.assert_array_equal(
        RepresentationEnsembleSelector(**kw).fit(c.X, c.y).pi_,
        RepresentationEnsembleSelector(**kw).fit(c.X, c.y).pi_,
    )


def test_ensemble_requires_at_least_one_representation():
    with pytest.raises(ValueError):
        RepresentationEnsembleSelector(task="binary", representations=[])
