"""Milestone 4: bootstrap stability selection, selection frequency, coverage targeting."""
import numpy as np
import pytest

from robustpixelmaker import (
    BootstrapMaskFoldEstimator,
    BootstrapMaskSelector,
    NestedCV,
    PatchGrid,
    complementary_pair,
    make_synthetic_images,
    mean_pairwise_jaccard,
    resample_indices,
)


def _iou(sel_pixels_flat, region_mask):
    sel = np.zeros(region_mask.size, dtype=bool)
    sel[sel_pixels_flat] = True
    reg = region_mask.ravel()
    union = np.logical_or(sel, reg).sum()
    return np.logical_and(sel, reg).sum() / union if union else 1.0


# --- resampling schemes ---------------------------------------------------- #
def test_bootstrap_resample_is_right_size_and_stratified():
    y = np.array([0] * 30 + [1] * 10)
    rng = np.random.default_rng(0)
    idx = resample_indices(len(y), "bootstrap", rng, y)
    assert len(idx) == len(y)
    # stratified: class proportions preserved exactly
    assert (y[idx] == 1).sum() == 10


def test_half_resample_is_half_size():
    y = np.array([0] * 30 + [1] * 10)
    idx = resample_indices(len(y), "half", np.random.default_rng(0), y)
    assert len(idx) == 20
    assert len(set(idx.tolist())) == len(idx)  # without replacement


def test_complementary_pairs_are_disjoint_and_cover():
    y = np.array([0] * 30 + [1] * 10)
    a, b = complementary_pair(len(y), np.random.default_rng(0), y)
    assert set(a.tolist()).isdisjoint(set(b.tolist()))
    assert sorted(np.concatenate([a, b]).tolist()) == list(range(len(y)))


def test_unknown_resample_mode_rejected():
    with pytest.raises(ValueError):
        resample_indices(10, "nonsense", np.random.default_rng(0))
    with pytest.raises(ValueError):
        BootstrapMaskSelector(subsample="nonsense")
    with pytest.raises(ValueError):
        BootstrapMaskSelector(tau=0.0)


# --- selection frequency --------------------------------------------------- #
def test_pi_is_a_valid_frequency_and_localises():
    c = make_synthetic_images(n=300, task="binary", seed=0)
    sel = BootstrapMaskSelector(task="binary", patch=4, n_bootstrap=8,
                                n_iter=300, seed=0).fit(c.X, c.y)
    assert sel.pi_.shape == (sel.grid.n_patches,)
    assert sel.pi_.min() >= 0.0 and sel.pi_.max() <= 1.0
    grid = PatchGrid.from_image_shape(c.X.shape, patch=4)
    info = np.unique(grid.pixel_patch[c.informative])
    dist = np.unique(grid.pixel_patch[c.distractor])
    # informative patches are selected far more often than distractor patches
    assert sel.pi_[info].mean() > 0.8
    assert sel.pi_[dist].mean() < 0.2


def test_pi_map_is_image_shaped():
    c = make_synthetic_images(n=200, seed=0)
    sel = BootstrapMaskSelector(task="binary", n_bootstrap=5, n_iter=200, seed=0).fit(c.X, c.y)
    assert sel.pi_map().shape == c.X.shape[1:]


def test_bootstrap_recovers_region():
    c = make_synthetic_images(n=300, task="binary", seed=0)
    sel = BootstrapMaskSelector(task="binary", patch=4, n_bootstrap=8,
                                n_iter=300, seed=0).fit(c.X, c.y)
    assert _iou(sel.selected_pixels(), c.informative) > 0.7
    assert _iou(sel.selected_pixels(), c.distractor) < 0.1


# --- threshold behaviour --------------------------------------------------- #
def test_higher_tau_selects_no_more_patches():
    c = make_synthetic_images(n=300, seed=0)
    kw = dict(task="binary", n_bootstrap=8, n_iter=300, seed=0)
    lo = BootstrapMaskSelector(tau=0.3, **kw).fit(c.X, c.y)
    hi = BootstrapMaskSelector(tau=0.9, **kw).fit(c.X, c.y)
    assert hi.coverage() <= lo.coverage()


def test_target_coverage_lands_on_the_nearest_reachable_operating_point():
    """The real contract, which the previous version of this test did not check.

    It was called ...hits_requested_operating_point and asserted only
    ``abs(coverage - target) < 0.15``. At target 0.1 that passes anything up to 0.25,
    a 150% miss, so a test whose name promised the target was hit in fact permitted
    missing it by more than the target itself. `examples/ex4` then relied on
    `target_coverage` to match two datasets, missed by a factor of two and reported
    the comparison as matched, with this test green throughout (FINDINGS sec.22).

    What the aggregator actually promises is the NEAREST reachable coverage, since
    `pi_` takes at most B+1 values and the reachable set is those retained fractions.
    That is exact and always checkable, so it is what is asserted here.
    """
    import warnings

    c = make_synthetic_images(n=300, seed=0)
    for target in (0.1, 0.3):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sel = BootstrapMaskSelector(task="binary", n_bootstrap=8, n_iter=300,
                                        target_coverage=target, seed=0).fit(c.X, c.y)
        pi = sel.pi_
        candidates = np.unique(np.concatenate([pi[pi > 0], [1.0 / 8]]))
        reachable = {float((pi >= t).mean()) for t in candidates}
        best = min(reachable, key=lambda cov: abs(cov - target))
        assert sel.coverage() == pytest.approx(best), (
            f"target {target}: landed on {sel.coverage():.4f}, but {best:.4f} was "
            f"reachable and nearer; reachable set {sorted(reachable)}")
        # and the verdict must agree with the arithmetic, not be decorative
        gap = abs(sel.coverage() - target)
        assert sel.coverage_target_met_ is bool(gap <= 0.25 * target)


def test_never_returns_empty_selection():
    """A degenerate aggregation must stay usable and flag itself, not select nothing."""
    c = make_synthetic_images(n=200, seed=0)
    sel = BootstrapMaskSelector(task="binary", n_bootstrap=4, n_iter=200,
                                tau=1.0, lam=5.0, seed=0).fit(c.X, c.y)
    assert len(sel.selected_patches_) >= 1
    assert sel.predict(c.X[:5]).shape == (5,)


# --- reproducibility and integration --------------------------------------- #
def test_bootstrap_reproducible():
    c = make_synthetic_images(n=200, seed=0)
    kw = dict(task="binary", n_bootstrap=5, n_iter=200, seed=0)
    np.testing.assert_array_equal(
        BootstrapMaskSelector(**kw).fit(c.X, c.y).pi_,
        BootstrapMaskSelector(**kw).fit(c.X, c.y).pi_,
    )


def test_complementary_pairs_mode_runs():
    c = make_synthetic_images(n=200, seed=0)
    sel = BootstrapMaskSelector(task="binary", n_bootstrap=6, subsample="complementary",
                                n_iter=200, seed=0).fit(c.X, c.y)
    assert sel.n_fits_ == 6
    assert sel.pi_.max() <= 1.0


def test_fold_estimator_in_nested_cv():
    c = make_synthetic_images(n=240, task="binary", seed=0)
    factory = lambda: BootstrapMaskFoldEstimator(patch=4, n_bootstrap=5, n_iter=250)
    res = NestedCV(estimator_factory=factory, k_outer=3, random_state=0).run(c.X, c.y)
    assert len(res.per_fold_scores) == 3
    assert res.mean_score > 0.7


# --- the metric hazard guard ----------------------------------------------- #
def test_empty_selection_does_not_report_perfect_stability():
    """An empty selection previously scored Jaccard 1.00, looking maximally stable."""
    assert np.isnan(mean_pairwise_jaccard([[], []]))
    assert np.isnan(mean_pairwise_jaccard([[1, 2], []]))
    assert mean_pairwise_jaccard([[1, 2], [1, 2]]) == 1.0


def test_min_size_guard_is_configurable():
    assert np.isnan(mean_pairwise_jaccard([[1], [1]], min_size=2))
    assert mean_pairwise_jaccard([[1], [1]], min_size=1) == 1.0


# --- bimodality diagnostic -------------------------------------------------- #
def test_ambiguous_fraction_is_low_when_signal_is_reproducible():
    """A clean planted region should give a bimodal pi, hence few undecided patches."""
    c = make_synthetic_images(n=300, task="binary", seed=0)
    sel = BootstrapMaskSelector(task="binary", patch=4, n_bootstrap=10,
                                n_iter=300, seed=0).fit(c.X, c.y)
    assert sel.ambiguous_fraction() < 0.25
    assert 0.0 <= sel.ambiguous_fraction() <= 1.0


def test_stability_report_shape():
    c = make_synthetic_images(n=200, seed=0)
    sel = BootstrapMaskSelector(task="binary", n_bootstrap=5, n_iter=200, seed=0).fit(c.X, c.y)
    rep = sel.stability_report()
    for key in ("n_fits", "tau", "coverage", "pi_mean", "ambiguous_fraction", "degenerate"):
        assert key in rep
    assert rep["n_fits"] == 5


# --- coverage confound in the stability metric ------------------------------ #
def test_raw_jaccard_rewards_keeping_almost_everything():
    """The confound: two near-complete sets overlap heavily whatever they select."""
    from robustpixelmaker import adjusted_jaccard, expected_jaccard

    P = 49
    big_a, big_b = list(range(0, 40)), list(range(5, 45))     # 82% coverage each
    small_a, small_b = [1, 2, 3], [2, 3, 4]                   # 6% coverage each
    assert mean_pairwise_jaccard([big_a, big_b]) > mean_pairwise_jaccard([small_a, small_b])
    # corrected for chance, the small pair is recognised as the more concordant one
    assert (adjusted_jaccard(small_a, small_b, P)
            > adjusted_jaccard(big_a, big_b, P))


def test_adjusted_jaccard_bounds():
    from robustpixelmaker import adjusted_jaccard, expected_jaccard

    P = 50
    assert adjusted_jaccard([1, 2, 3], [1, 2, 3], P) == pytest.approx(1.0)
    # disjoint sets score at or below zero (no better than chance)
    assert adjusted_jaccard([1, 2, 3], [10, 11, 12], P) <= 0.0
    assert 0.0 <= expected_jaccard(10, 10, P) <= 1.0


def test_mean_pairwise_jaccard_accepts_n_total():
    a, b = [1, 2, 3], [2, 3, 4]
    raw = mean_pairwise_jaccard([a, b])
    adj = mean_pairwise_jaccard([a, b], n_total=49)
    assert adj != raw and adj <= 1.0


def test_an_unreachable_target_coverage_is_reported_not_silently_missed():
    """A request that cannot be met must say so, not return something else quietly.

    `pi_` takes at most B+1 distinct values, so achievable coverages are quantised,
    and ties make the grid coarse. On blood at B=12, 24.5% of patches sit at pi_
    exactly 1.0, so no threshold yields 0.12 and a request for it returned 0.245.
    `examples/ex4` asked for matched coverage across two datasets on that basis and
    compared 0.245 against 0.122 while reporting the comparison as matched
    (FINDINGS sec.22).
    """
    import warnings

    from robustpixelmaker.selection import BootstrapMaskSelector
    from robustpixelmaker.synthetic import make_synthetic_images

    c = make_synthetic_images(n=200, seed=0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sel = BootstrapMaskSelector(task="binary", patch=8, n_bootstrap=4, n_iter=120,
                                    target_coverage=0.01, seed=0).fit(c.X, c.y)
    assert sel.coverage_target_met_ is False
    assert sel.coverage_realised_ > 0.01
    assert any("not reachable" in str(w.message) for w in caught)
    assert sel.stability_report()["coverage_target_met"] is False


def test_no_target_coverage_means_no_verdict_and_no_warning():
    """Absence of a request must read as None, not as a failed one."""
    import warnings

    from robustpixelmaker.selection import BootstrapMaskSelector
    from robustpixelmaker.synthetic import make_synthetic_images

    c = make_synthetic_images(n=200, seed=0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sel = BootstrapMaskSelector(task="binary", patch=8, n_bootstrap=4,
                                    n_iter=120, seed=0).fit(c.X, c.y)
    assert sel.coverage_target_met_ is None
    assert sel.stability_report()["coverage_target_met"] is None
    assert not any("not reachable" in str(w.message) for w in caught)


def test_a_reachable_target_coverage_is_reported_as_met():
    """The verdict must be able to come back positive, or it measures nothing."""
    from robustpixelmaker.selection import BootstrapMaskSelector
    from robustpixelmaker.synthetic import make_synthetic_images

    c = make_synthetic_images(n=200, seed=0)
    free = BootstrapMaskSelector(task="binary", patch=8, n_bootstrap=6, n_iter=150,
                                 seed=0).fit(c.X, c.y)
    reachable = free.coverage()          # a coverage the data demonstrably supports
    sel = BootstrapMaskSelector(task="binary", patch=8, n_bootstrap=6, n_iter=150,
                                target_coverage=reachable, seed=0).fit(c.X, c.y)
    assert sel.coverage_target_met_ is True
