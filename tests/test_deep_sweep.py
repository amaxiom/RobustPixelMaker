"""Deep sweep, round two: defect families known to recur in the sibling libraries.

This sweep hunted the specific families recorded for MissLearn (shared-helper blind
spots, scale-dependent thresholds, catastrophic cancellation, guards that fail toward
confidence, silent numpy traps, seed collisions) and found seven genuine defects in
RobustPixelMaker. Each test below either encodes a found-and-fixed bug (marked BUG) or
pins an adversarial property the sweep verified (marked PROPERTY).
"""
import numpy as np
import pytest

from robustpixelmaker import (
    BootstrapMaskSelector,
    PatchGrid,
    PatchStats,
    RepresentationEnsembleSelector,
    RobustPixelMaker,
    Seeds,
    SoftMaskSelector,
    ensure_finite,
    make_synthetic_images,
)


# --------------------------------------------------------------------------- #
# BUG 1: the shared pooling helper needed O(H*W*n_patches) memory
# --------------------------------------------------------------------------- #
def test_pooling_matches_naive_reference_exactly():
    """The rewritten padded-reshape pooling must equal a per-patch loop exactly,
    including ragged edge patches, since every estimator sits on this helper."""
    rng = np.random.default_rng(0)
    for H, W, patch in [(18, 30, 4), (16, 16, 4), (7, 11, 3), (5, 5, 5)]:
        X = rng.normal(size=(6, H, W))
        grid = PatchGrid(H, W, patch=patch)
        got = grid.pool(X)
        want = np.zeros_like(got)
        for p in range(grid.n_patches):
            pix = grid.patches_to_pixels([p])
            want[:, p] = X.reshape(6, -1)[:, pix].mean(axis=1)
        np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12)


def test_pooling_is_memory_safe_at_realistic_image_size():
    """512x512 at patch 8 used to require an 8.6 GB dense matrix just to build the
    grid; it must now run in ordinary memory (the padded copy of the batch only)."""
    grid = PatchGrid(512, 512, patch=8)          # formerly allocated ~8.6 GB here
    X = np.random.default_rng(0).normal(size=(4, 512, 512))
    f = grid.pool(X)
    assert f.shape == (4, 4096)
    assert np.isfinite(f).all()
    assert not hasattr(grid, "_pool_matrix")     # the dense matrix must be gone


def test_pool_rejects_wrong_shapes():
    grid = PatchGrid(8, 8, patch=4)
    with pytest.raises(ValueError, match="does not match"):
        grid.pool(np.zeros((2, 10, 10)))
    with pytest.raises(ValueError, match="does not match"):
        grid.pool(np.zeros((2, 64)))             # flat input is not an image batch


# --------------------------------------------------------------------------- #
# BUG 2: the constant-feature cutoff was absolute, so small-scale data was erased
# --------------------------------------------------------------------------- #
def test_selection_is_scale_invariant_across_extreme_scales():
    """Fitting a*X + b must select the SAME patches for any positive scale a and
    offset b, because standardisation removes both. The old absolute cutoff
    (sd < 1e-8) declared every feature constant at a = 1e-9 and below, silently
    fitting an intercept-only model on zeroed features. Tested at the extreme scales
    the sibling library's records prescribe."""
    c = make_synthetic_images(n=250, task="binary", seed=0)
    ref = SoftMaskSelector(task="binary", lam=0.03, n_iter=250, seed=0).fit(c.X, c.y)
    for a, b in [(1e-9, 0.0), (1e-200, 0.0), (1e6, 3.0), (1e300, 0.0), (1.0, 1e8)]:
        sel = SoftMaskSelector(task="binary", lam=0.03, n_iter=250, seed=0).fit(
            c.X * a + b, c.y)
        np.testing.assert_array_equal(sel.selected_patches_, ref.selected_patches_), \
            f"selection changed at scale a={a}, offset b={b}"


def test_constant_patches_standardise_to_exact_zero():
    """A genuinely constant patch must become exactly 0, not amplified noise."""
    c = make_synthetic_images(n=100, seed=0)
    X = c.X.copy()
    X[:, :4, :4] = 42.0                          # patch 0 constant everywhere
    sel = SoftMaskSelector(task="binary", n_iter=50, seed=0).fit(X, c.y)
    f = sel._standardise(sel._features(X))
    assert np.allclose(f[:, 0, :], 0.0)


# --------------------------------------------------------------------------- #
# BUG 3: PatchStats variance by E[x^2] - E[x]^2 cancelled catastrophically
# --------------------------------------------------------------------------- #
def test_patchstats_variance_survives_large_offsets_and_scales():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 16, 16))            # unit variance, zero offset
    grid = PatchGrid(16, 16, patch=4)
    truth = PatchStats().transform(X, grid)[:, :, 1]

    # at offset 1e8 the difference form lost the entire variance to cancellation
    shifted = PatchStats().transform(X + 1e8, grid)[:, :, 1]
    np.testing.assert_allclose(shifted, truth, rtol=1e-4)

    # beyond |x| ~ 1e154 the difference form overflowed to inf/nan outright
    scaled = PatchStats().transform(X * 1e200, grid)[:, :, 1]
    assert np.isfinite(scaled).all()
    np.testing.assert_allclose(scaled / 1e200, truth, rtol=1e-6)


# --------------------------------------------------------------------------- #
# BUG 4: predict_proba on regression fabricated probabilities
# --------------------------------------------------------------------------- #
def test_regression_predict_proba_refuses():
    c = make_synthetic_images(n=100, task="regression", seed=0)
    sel = SoftMaskSelector(task="regression", n_iter=50, seed=0).fit(c.X, c.y)
    with pytest.raises(ValueError, match="fabricated"):
        sel.predict_proba(c.X[:3])
    # and it refuses through the aggregators and facade too
    boot = BootstrapMaskSelector(task="regression", n_bootstrap=2, n_iter=50,
                                 seed=0).fit(c.X, c.y)
    with pytest.raises(ValueError, match="fabricated"):
        boot.predict_proba(c.X[:3])


# --------------------------------------------------------------------------- #
# BUG 5: NaN and Inf inputs propagated silently into confident selections
# --------------------------------------------------------------------------- #
def test_non_finite_input_is_an_explicit_error_everywhere():
    c = make_synthetic_images(n=80, seed=0)
    Xnan = c.X.copy(); Xnan[3, 5, 5] = np.nan
    Xinf = c.X.copy(); Xinf[0, 0, 0] = np.inf
    for bad in (Xnan, Xinf):
        with pytest.raises(ValueError, match="non-finite"):
            SoftMaskSelector(task="binary", n_iter=20).fit(bad, c.y)
        with pytest.raises(ValueError, match="non-finite"):
            BootstrapMaskSelector(task="binary", n_bootstrap=2, n_iter=20).fit(bad, c.y)
        with pytest.raises(ValueError, match="non-finite"):
            RepresentationEnsembleSelector(task="binary", n_bootstrap=2,
                                           n_representations=2, n_iter=20).fit(bad, c.y)
    # a NaN regression target is caught too
    ybad = c.y.astype(float); ybad[0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        SoftMaskSelector(task="regression", n_iter=20).fit(c.X, ybad)
    # the helper reports the counts it found
    with pytest.raises(ValueError, match="1 NaN, 1 infinite"):
        ensure_finite(np.array([1.0, np.nan, np.inf]), "X")


# --------------------------------------------------------------------------- #
# BUG 6: ensemble fit seeds collided for (resample, lens) pairs with equal sums
# --------------------------------------------------------------------------- #
def test_ensemble_task_seeds_are_unique():
    s = Seeds(base=0, n_bootstrap=8, n_representations=4)
    R, B = 4, 8
    seeds = [s.ensemble(r * B + b) for r in range(R) for b in range(B)]
    assert len(set(seeds)) == R * B              # the old additive scheme collided
    with pytest.raises(ValueError, match="task index"):
        s.ensemble(-1)


def test_ensemble_still_reproducible_after_seed_fix():
    c = make_synthetic_images(n=150, seed=0)
    kw = dict(task="binary", n_bootstrap=3, n_representations=2, n_iter=150, seed=0)
    np.testing.assert_array_equal(
        RepresentationEnsembleSelector(**kw).fit(c.X, c.y).pi_,
        RepresentationEnsembleSelector(**kw).fit(c.X, c.y).pi_,
    )


# --------------------------------------------------------------------------- #
# BUG 7: per-lens agreement was measured at the wrong threshold under
#        target_coverage
# --------------------------------------------------------------------------- #
def test_per_lens_sets_use_the_operating_threshold():
    c = make_synthetic_images(n=200, seed=0)
    sel = RepresentationEnsembleSelector(
        task="binary", n_bootstrap=4, n_representations=2, n_iter=200,
        target_coverage=0.1, seed=0).fit(c.X, c.y)
    # every per-lens set must be thresholded at the tau_ actually chosen
    for name, pi_r in sel.pi_by_representation_.items():
        expected = np.where(pi_r >= sel.tau_)[0]
        found = [set(s.tolist()) for s in sel._per_rep_sets]
        assert set(expected.tolist()) in found


# --------------------------------------------------------------------------- #
# PROPERTY: row-order invariance (the degenerate-columns family's second half)
# --------------------------------------------------------------------------- #
def test_single_fit_is_row_order_invariant():
    """Shuffling the training rows must not change the fitted mask: gradients are
    full-batch sums, and standardisation is order-free. The sibling library's records
    include order-dependent coefficients as a real defect family."""
    c = make_synthetic_images(n=200, seed=0)
    perm = np.random.default_rng(1).permutation(200)
    a = SoftMaskSelector(task="binary", lam=0.03, n_iter=250, seed=0).fit(c.X, c.y)
    b = SoftMaskSelector(task="binary", lam=0.03, n_iter=250, seed=0).fit(
        c.X[perm], c.y[perm])
    np.testing.assert_allclose(a.mask_, b.mask_, atol=1e-10)


# --------------------------------------------------------------------------- #
# PROPERTY: representation lenses are affine-equivariant, so selection through
# them is scale invariant too
# --------------------------------------------------------------------------- #
def test_lens_selection_is_scale_invariant():
    c = make_synthetic_images(n=200, task="binary", seed=0)
    for rep in (PatchStats(),):
        ref = SoftMaskSelector(task="binary", lam=0.03, n_iter=200,
                               representation=rep, seed=0).fit(c.X, c.y)
        two = SoftMaskSelector(task="binary", lam=0.03, n_iter=200,
                               representation=rep, seed=0).fit(c.X * 1e-7 + 5.0, c.y)
        np.testing.assert_array_equal(two.selected_patches_, ref.selected_patches_)


# --------------------------------------------------------------------------- #
# PROPERTY: the numpy truthiness trap in the regression target scale
# --------------------------------------------------------------------------- #
def test_constant_regression_target_does_not_divide_by_zero():
    c = make_synthetic_images(n=80, seed=0)
    y = np.full(80, 3.14)
    sel = SoftMaskSelector(task="regression", n_iter=50, seed=0).fit(c.X, y)
    pred = sel.predict(c.X[:5])
    assert np.isfinite(pred).all()
    np.testing.assert_allclose(pred, 3.14, atol=1e-6)


# --------------------------------------------------------------------------- #
# PROPERTY: the facade propagates every guard
# --------------------------------------------------------------------------- #
def test_facade_propagates_the_guards():
    c = make_synthetic_images(n=80, seed=0)
    Xnan = c.X.copy(); Xnan[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        RobustPixelMaker(selector="single", n_iter=20, k_outer=3).fit(Xnan, c.y)


# --------------------------------------------------------------------------- #
# BUG 8: an explicit facade task that disagreed with the engine's inference
# silently fitted the two halves of the result for different tasks
# --------------------------------------------------------------------------- #
def test_facade_task_mismatch_is_an_error():
    c = make_synthetic_images(n=80, task="regression", seed=0)
    with pytest.raises(ValueError, match="looks like 'regression'"):
        RobustPixelMaker(selector="single", task="binary", n_iter=20,
                         k_outer=3).fit(c.X, c.y)


# --------------------------------------------------------------------------- #
# BUG 9: the frequency map saturates, so top-k among ties was arbitrary
# --------------------------------------------------------------------------- #
def test_ranking_breaks_frequency_ties_on_strength():
    """pi_ takes at most B+1 values and saturates, so several patches routinely sit at
    pi=1.0 with none above them. Asking that map for a small top-k then returned an
    arbitrary subset chosen by index order, which cost 0.11 AUC at 5% coverage on the
    matched-coverage benchmark. ranking() must order ties by mean gate strength while
    preserving the frequency ordering wherever it actually discriminates."""
    c = make_synthetic_images(n=200, seed=0)
    sel = BootstrapMaskSelector(task="binary", n_bootstrap=8, n_iter=200,
                                seed=0).fit(c.X, c.y)
    order = sel.ranking()
    assert len(order) == len(sel.pi_)
    assert sorted(order.tolist()) == list(range(len(sel.pi_)))   # a permutation

    # frequency is non-increasing along the ranking: ties never reorder pi itself
    pi_sorted = sel.pi_[order]
    assert np.all(np.diff(pi_sorted) <= 1e-12)

    # within any tied block, strength is non-increasing
    for value in np.unique(sel.pi_):
        block = [i for i in order if sel.pi_[i] == value]
        s = sel.strength_[block]
        assert np.all(np.diff(s) <= 1e-9), f"strength not ordered inside the pi={value} tie"

    assert sel.top_patches(3).shape == (3,)
    assert set(sel.top_patches(3).tolist()) <= set(order[:3].tolist())
    assert sel.top_patches(0).size == 0
    assert sel.top_patches(10_000).size == len(sel.pi_)          # clamped, not an error


def test_strength_is_recorded_and_finite():
    c = make_synthetic_images(n=150, seed=0)
    sel = BootstrapMaskSelector(task="binary", n_bootstrap=6, n_iter=150,
                                seed=0).fit(c.X, c.y)
    assert sel.strength_.shape == sel.pi_.shape
    assert np.isfinite(sel.strength_).all()
    assert (sel.strength_ >= 0).all()


# --------------------------------------------------------------------------- #
# BUG 10: a linear frozen head cannot reward a gate with no linear main effect
# --------------------------------------------------------------------------- #
def test_mlp_head_recovers_a_region_the_linear_head_misses_entirely():
    """The linear head returns an EMPTY selection when the informative cells act only
    through an interaction: phase 2 prunes gates that do not help the frozen head, and
    a gate with no main effect cannot help a linear one. Measured on the benchmark's
    own control with a symmetric label, the linear head scores recovery F1 exactly
    0.000 on every seed while the MLP head reaches about 0.47 (FINDINGS sec.16). This
    pins the mechanism, not the exact number."""
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / "benchmarks"))
    from run_nonlinear_control import make_control

    X, y, info_mask, _ = make_control("saturating", n=400, seed=0)
    grid = PatchGrid.from_image_shape(X.shape, patch=4)
    truth = set(np.unique(grid.pixel_patch[info_mask]).tolist())

    def recovered(head):
        s = BootstrapMaskSelector(task="binary", patch=4, lam=0.03, n_iter=400,
                                  n_bootstrap=6, tau=0.6, head=head, seed=0).fit(X, y)
        return len(set(s.selected_patches_.tolist()) & truth)

    lin, mlp = recovered("linear"), recovered("mlp")
    assert lin == 0, f"the linear-head failure this test exists for is gone (recovered {lin})"
    assert mlp > lin, f"the MLP head must recover an interactive region (got {mlp})"


def test_mlp_head_is_wired_through_every_layer():
    c = make_synthetic_images(n=150, seed=0)
    s = SoftMaskSelector(task="binary", head="mlp", hidden=8, n_iter=200, seed=0).fit(c.X, c.y)
    assert s.head_params_ is not None and len(s.head_params_) == 4
    assert np.isfinite(s.predict_proba(c.X[:4])).all()
    b = BootstrapMaskSelector(task="binary", head="mlp", n_bootstrap=3,
                              n_iter=150, seed=0).fit(c.X, c.y)
    assert b._inner_kwargs()["head"] == "mlp"
    # the linear path must keep head_params_ None so predict stays on the linear branch
    lin = SoftMaskSelector(task="binary", n_iter=100, seed=0).fit(c.X, c.y)
    assert lin.head_params_ is None
    with pytest.raises(ValueError, match="head must be"):
        SoftMaskSelector(task="binary", head="transformer")


# --------------------------------------------------------------------------- #
# BUG 11: a more expressive frozen head made the mask stop pruning entirely
# --------------------------------------------------------------------------- #
def test_auto_lam_prevents_a_degenerate_flat_mask():
    """With the MLP head at the linear head's lam, every gate stayed at 1.000 on a
    multiclass task: the frozen head uses every input, so no gate looks dispensable.
    A flat mask carries no ranking information, so top_patches then returned patches
    by index order and the selection was not data-determined at all. lam_scale="auto"
    searches lam for a mask that actually discriminates (FINDINGS sec.17)."""
    c = make_synthetic_images(n=250, task="regression", seed=0)
    q = np.quantile(c.y, [1 / 3, 2 / 3])
    y = np.digitize(c.y, q)                       # 3 classes, the multiclass path

    flat = SoftMaskSelector(task="multiclass", patch=4, lam=0.001, n_iter=300,
                            head="mlp", lam_scale="fixed", seed=0).fit(c.X, y)
    auto = SoftMaskSelector(task="multiclass", patch=4, lam=0.001, n_iter=300,
                            head="mlp", lam_scale="auto", seed=0).fit(c.X, y)

    assert auto.mask_.min() < auto.mask_.max(), "auto must produce a discriminating mask"
    assert auto.lam_used_ >= flat.lam_used_, "auto should not weaken the penalty here"
    # auto reports which lam it actually used, so the operating point is never hidden
    assert auto.lam_used_ != auto.lam or auto.lam_scale == "fixed"


def test_lam_scale_default_is_unchanged_behaviour():
    """The default must reproduce fixed-lam behaviour exactly, or every published
    number in this repository would silently move."""
    c = make_synthetic_images(n=200, seed=0)
    a = SoftMaskSelector(task="binary", lam=0.03, n_iter=250, seed=0).fit(c.X, c.y)
    b = SoftMaskSelector(task="binary", lam=0.03, n_iter=250,
                         lam_scale="fixed", seed=0).fit(c.X, c.y)
    np.testing.assert_array_equal(a.mask_, b.mask_)
    assert a.lam_used_ == 0.03
    with pytest.raises(ValueError, match="lam_scale must be"):
        SoftMaskSelector(task="binary", lam_scale="adaptive")


def test_auto_lam_is_calibrated_once_not_per_resample(monkeypatch):
    """Every resample fit must share one lam, or pi_ stops measuring what it reports.

    Stability selection counts how often a patch survives AT A FIXED operating point.
    `lam_scale` used to be forwarded straight into each resample fit, so with "auto"
    every bootstrap replicate ran its own ladder and could land on a different penalty.
    The frequency map then mixed "which patches matter" with "which lam this resample
    picked". Measured cost on the matched-coverage protocol: 0.16 mean pairwise Jaccard
    (0.536 to 0.375) and 5.5x the runtime. FINDINGS sec.18.
    """
    import robustpixelmaker.selection as S

    seen = []
    real = S.SoftMaskSelector

    class Recording(real):
        def __init__(self, **kw):
            seen.append((kw.get("lam"), kw.get("lam_scale")))
            super().__init__(**kw)

    monkeypatch.setattr(S, "SoftMaskSelector", Recording)
    c = make_synthetic_images(n=160, seed=0)
    sel = S.BootstrapMaskSelector(task="binary", patch=8, lam=0.01, n_bootstrap=4,
                                  n_iter=120, head="mlp", lam_scale="auto",
                                  seed=0).fit(c.X, c.y)

    # exactly one fit is allowed to search; the calibration probe is the first
    searching = [x for x in seen if x[1] == "auto"]
    assert len(searching) == 1, f"lam searched {len(searching)} times, expected 1"

    # and every later fit is pinned to the single calibrated value
    after = [x for x in seen[1:] if x[1] != "auto"]
    assert after, "expected resample fits after calibration"
    assert {x[1] for x in after} == {"fixed"}
    assert {x[0] for x in after} == {sel.lam_used_}
    assert sel.lam_used_ == sel._lam_calibrated_


def test_fixed_lam_bootstrap_does_no_calibration_fit(monkeypatch):
    """The default path must not pay for, or be perturbed by, a calibration probe."""
    import robustpixelmaker.selection as S

    seen = []
    real = S.SoftMaskSelector

    class Recording(real):
        def __init__(self, **kw):
            seen.append((kw.get("lam"), kw.get("lam_scale")))
            super().__init__(**kw)

    monkeypatch.setattr(S, "SoftMaskSelector", Recording)
    c = make_synthetic_images(n=160, seed=0)
    sel = S.BootstrapMaskSelector(task="binary", patch=8, lam=0.02, n_bootstrap=3,
                                  n_iter=120, seed=0).fit(c.X, c.y)

    assert {x for x in seen} == {(0.02, "fixed")}
    assert sel._lam_calibrated_ is None
    assert sel.lam_used_ == 0.02


def test_refitting_recalibrates_lam_rather_than_reusing_a_stale_value():
    """A stale calibration would silently apply one dataset's operating point to
    another, which is the quiet-wrong-answer failure this class of state invites."""
    from robustpixelmaker.selection import BootstrapMaskSelector

    a = make_synthetic_images(n=160, seed=0)
    b = make_synthetic_images(n=160, noise=4.0, seed=1)
    sel = BootstrapMaskSelector(task="binary", patch=8, lam=0.01, n_bootstrap=3,
                                n_iter=120, head="mlp", lam_scale="auto", seed=0)
    sel.fit(a.X, a.y)
    first = sel.lam_used_
    sel.fit(b.X, b.y)
    second = sel.lam_used_
    # refitting the FIRST dataset again must return to the first answer exactly,
    # which it cannot do if the second fit left calibration state behind
    sel.fit(a.X, a.y)
    assert sel.lam_used_ == first
    assert first > 0 and second > 0


@pytest.mark.parametrize("selector", ["single", "bootstrap", "ensemble"])
def test_facade_can_actually_reach_the_mlp_head(selector):
    """The README documents head="mlp", so the documented entry point must set it.

    It could not. `RobustPixelMaker.__init__` took no `head`, `hidden`, `lam_scale`
    or `auto_target`, and neither `_fold_factory` nor `_final_selector` forwarded
    them, so the only way to use the MLP head was to bypass the facade and build a
    selector by hand. A documented option unreachable from the documented entry point
    is a documentation defect as much as a code one.
    """
    from robustpixelmaker.core import RobustPixelMaker

    c = make_synthetic_images(n=120, seed=0)
    r = RobustPixelMaker(selector=selector, head="mlp", hidden=8, lam_scale="auto",
                         auto_target=0.15, n_bootstrap=3, n_representations=2,
                         n_iter=100, k_outer=2, random_state=0).fit(c.X, c.y)
    inner = r.selector_
    assert inner.head == "mlp"
    assert inner.hidden == 8
    assert inner.lam_scale == "auto"
    assert inner.auto_target == 0.15
    assert 0.0 < r.coverage() <= 1.0


@pytest.mark.parametrize("cls_name", ["RobustPixelMaker", "BootstrapMaskSelector",
                                     "RepresentationEnsembleSelector",
                                     "SoftMaskSelector"])
@pytest.mark.parametrize("kw,match", [
    (dict(auto_target=1.5), "auto_target"),
    (dict(auto_target=0.0), "auto_target"),
    (dict(head="deep"), "head must be"),
    (dict(lam_scale="adaptive"), "lam_scale must be"),
])
def test_bad_mask_arguments_are_refused_at_construction(cls_name, kw, match):
    """A mistyped argument must name itself, not be reported as a data problem.

    `BootstrapMaskSelector._one_fit` swallows every exception a resample fit raises,
    deliberately, so a degenerate resample cannot abort the aggregation. That guard
    also swallowed argument errors: a bad `head` reached the inner constructor inside
    the try, every resample "failed", and the user was told "every one of the N
    resample fits failed; this indicates a systematic error in the inner mask fit",
    which points at the data. Validating at construction keeps the guard honest.
    """
    import robustpixelmaker
    cls = getattr(robustpixelmaker, cls_name)
    with pytest.raises(ValueError, match=match):
        cls(**kw)


def test_auto_target_is_a_coverage_prior_and_moves_the_selection():
    """auto_target must actually reach the mask, and it is not a neutral default.

    lam_scale="auto" searches lam for a mask landing near auto_target open, so
    auto_target is a PRIOR on coverage. Measured on the four label rules: at the 0.4
    default, against a true informative fraction of 0.062, the MLP head fell from
    16.0x chance to 4.7x on the linear rule purely by being pushed to keep more
    (FINDINGS sec.19). A user must therefore be able to set it, and it must bite.
    """
    from robustpixelmaker.selection import BootstrapMaskSelector

    c = make_synthetic_images(n=200, seed=0)
    kw = dict(task="binary", patch=4, lam=0.01, n_bootstrap=3, n_iter=150,
              head="mlp", lam_scale="auto", seed=0)
    lean = BootstrapMaskSelector(auto_target=0.1, **kw).fit(c.X, c.y)
    wide = BootstrapMaskSelector(auto_target=0.7, **kw).fit(c.X, c.y)
    assert lean.strength_.mean() < wide.strength_.mean(), (
        "a larger auto_target must leave more of the mask open")
