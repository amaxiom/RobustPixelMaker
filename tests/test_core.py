"""The RobustPixelMaker facade: the family-standard public surface."""
import numpy as np
import pytest

from robustpixelmaker import RobustPixelMaker, make_synthetic_images, run_pipeline


def _small_kwargs(selector):
    """Configurations small enough for tests; the defaults are production-sized.

    The frequency selectors use ``target_coverage`` rather than a fixed threshold:
    with only a handful of test-sized resamples the frequency grid is coarse, and a
    fixed tau then keeps far too much. Targeting the operating point directly is the
    facade's own recommended answer to that, so the tests exercise it.
    """
    kw = dict(selector=selector, patch=4, lam=0.03, n_iter=300, k_outer=3,
              random_state=0)
    if selector in ("bootstrap", "ensemble"):
        kw.update(n_bootstrap=4, target_coverage=0.1)
    if selector == "ensemble":
        kw.update(n_representations=2)
    return kw


def _iou(sel_pixels_flat, region_mask):
    sel = np.zeros(region_mask.size, dtype=bool)
    sel[sel_pixels_flat] = True
    reg = region_mask.ravel()
    union = np.logical_or(sel, reg).sum()
    return np.logical_and(sel, reg).sum() / union if union else 1.0


# --- regime contract --------------------------------------------------------- #
def test_registered_false_is_a_documented_refusal():
    c = make_synthetic_images(n=60, seed=0)
    with pytest.raises(NotImplementedError, match="registered=False"):
        RobustPixelMaker(registered=False).fit(c.X, c.y)


def test_unknown_selector_rejected():
    with pytest.raises(ValueError):
        RobustPixelMaker(selector="magic")


# --- end to end, all three selector kinds ------------------------------------ #
@pytest.mark.parametrize("selector", ["single", "bootstrap", "ensemble"])
def test_fit_predict_and_summary(selector):
    c = make_synthetic_images(n=200, task="binary", seed=0)
    rpm = RobustPixelMaker(**_small_kwargs(selector)).fit(c.X, c.y)

    # honest nested-CV deliverables
    assert len(rpm.result_.per_fold_scores) == 3
    assert rpm.result_.mean_score > 0.7
    # the final selection localises onto the planted region
    assert _iou(rpm.selected_pixels(), c.informative) > 0.4
    assert 0.0 < rpm.coverage() < 0.5
    # prediction surface
    assert rpm.predict(c.X[:7]).shape == (7,)
    proba = rpm.predict_proba(c.X[:7])
    assert proba.shape == (7, 2)
    # summary carries both halves and names the selector
    s = rpm.summary()
    assert s["selector"] == selector
    assert "score_mean" in s and "coverage" in s
    # pi_map always renders, whatever the selector kind
    assert rpm.pi_map().shape == c.X.shape[1:]


def test_task_inference_matches_module():
    c = make_synthetic_images(n=150, task="regression", seed=1)
    rpm = RobustPixelMaker(**_small_kwargs("bootstrap")).fit(c.X, c.y)
    assert rpm.task_ == "regression"
    assert rpm.predict(c.X[:5]).dtype.kind == "f"


def test_reproducible_end_to_end():
    c = make_synthetic_images(n=150, seed=0)
    kw = _small_kwargs("bootstrap")
    a = RobustPixelMaker(**kw).fit(c.X, c.y)
    b = RobustPixelMaker(**kw).fit(c.X, c.y)
    np.testing.assert_array_equal(a.result_.per_fold_scores, b.result_.per_fold_scores)
    np.testing.assert_array_equal(a.pi_, b.pi_)
    np.testing.assert_array_equal(a.selected_patches_, b.selected_patches_)


def test_run_pipeline_is_the_functional_equivalent():
    c = make_synthetic_images(n=150, seed=0)
    rpm = run_pipeline(c.X, c.y, **_small_kwargs("bootstrap"))
    assert isinstance(rpm, RobustPixelMaker)
    assert rpm.result_.mean_score > 0.6


def test_save_exports_selection_artifacts(tmp_path):
    c = make_synthetic_images(n=150, seed=0)
    rpm = RobustPixelMaker(**_small_kwargs("bootstrap")).fit(c.X, c.y)
    out = rpm.save(tmp_path)
    assert (out / "rpm_selection.json").exists()
    assert (out / "rpm_pi_map.npy").exists()
    assert (out / "rpm_summary.json").exists()   # the nested-CV artefacts too
    import json
    sel = json.loads((out / "rpm_selection.json").read_text())
    assert sel["selected_patches"] == [int(p) for p in rpm.selected_patches_]
    assert "stability_report" in sel


def test_stability_report_present_for_frequency_selectors():
    c = make_synthetic_images(n=150, seed=0)
    rpm = RobustPixelMaker(**_small_kwargs("ensemble")).fit(c.X, c.y)
    rep = rpm.stability_report()
    assert "ambiguous_fraction" in rep and "representation_agreement" in rep
