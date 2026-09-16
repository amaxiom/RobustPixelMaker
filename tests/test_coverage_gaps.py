"""Targeted tests for branches the main suites do not reach.

Two kinds of gap are closed here. First, genuine blind spots the coverage report
exposed, most notably that switching the default gate to Hard-Concrete silently
removed all coverage of the sigmoid phase-2 branch. Second, torch-dependent branches
in config and reproducibility, exercised against a minimal stub torch module so the
logic is tested even though torch is deliberately not installed.
"""
import sys
import types

import numpy as np
import pytest

from robustpixelmaker import (
    BaselineComparison,
    BootstrapMaskSelector,
    NestedCV,
    PatchGrid,
    RepresentationEnsembleSelector,
    RPMConfig,
    Seeds,
    SoftMaskSelector,
    adjusted_jaccard,
    detect_device,
    enable_determinism,
    expected_jaccard,
    make_synthetic_images,
    paired_comparison,
    score_predictions,
    set_global_seed,
)
from robustpixelmaker.metrics import rmse_from_score
from robustpixelmaker.synthetic import make_synthetic_images as make_imgs


# --------------------------------------------------------------------------- #
# a minimal torch stub, so torch-gated logic is testable without torch
# --------------------------------------------------------------------------- #
@pytest.fixture()
def stub_torch(monkeypatch):
    torch = types.ModuleType("torch")
    calls = {"manual_seed": [], "det": [], "cuda_seed": []}
    torch.manual_seed = lambda s: calls["manual_seed"].append(s)
    torch.use_deterministic_algorithms = (
        lambda flag, warn_only=True: calls["det"].append((flag, warn_only)))

    cuda = types.SimpleNamespace(is_available=lambda: False,
                                 manual_seed_all=lambda s: calls["cuda_seed"].append(s))
    torch.cuda = cuda
    cudnn = types.SimpleNamespace(deterministic=False, benchmark=True)
    mps = types.SimpleNamespace(is_available=lambda: False)
    torch.backends = types.SimpleNamespace(cudnn=cudnn, mps=mps)
    monkeypatch.setitem(sys.modules, "torch", torch)
    return torch, calls


def test_set_global_seed_reaches_torch(stub_torch):
    torch, calls = stub_torch
    set_global_seed(11)
    assert calls["manual_seed"] == [11]
    assert calls["cuda_seed"] == []          # cuda unavailable in the stub


def test_enable_determinism_with_and_without_torch(stub_torch):
    torch, calls = stub_torch
    report = enable_determinism(strict=True)
    assert report["torch"] and report["deterministic_algorithms"]
    assert report["cudnn_deterministic"]
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False


def test_enable_determinism_reports_absent_torch():
    report = enable_determinism()
    assert report == {"torch": False, "deterministic_algorithms": False,
                      "cudnn_deterministic": False}


def test_detect_device_with_stub_torch(stub_torch):
    assert detect_device("auto") == "cpu"     # stub has no cuda, no mps
    assert detect_device("cpu") == "cpu"
    with pytest.warns(UserWarning, match="unavailable"):
        assert detect_device("cuda") == "cpu"


def test_detect_device_warns_without_torch():
    with pytest.warns(UserWarning, match="torch not installed"):
        assert detect_device("cuda") == "cpu"


# --------------------------------------------------------------------------- #
# config and seeds
# --------------------------------------------------------------------------- #
def test_config_roundtrip_and_device():
    cfg = RPMConfig(registered=True, tau=0.6)
    d = cfg.to_dict()
    assert d["tau"] == 0.6 and d["registered"] is True
    assert cfg.resolved_device() in {"cuda", "mps", "cpu"}


def test_seeds_rng_and_representation_guard():
    s = Seeds(base=3, n_bootstrap=2, n_representations=2)
    r = s.rng(s.bootstrap(0))
    assert isinstance(r, np.random.Generator)
    with pytest.raises(ValueError, match="representations"):
        Seeds(base=0, n_representations=10_001)


# --------------------------------------------------------------------------- #
# metrics branches
# --------------------------------------------------------------------------- #
def test_score_predictions_all_tasks_and_degenerate():
    y = np.array([0, 1, 0, 1])
    proba = np.array([[0.9, 0.1], [0.2, 0.8], [0.8, 0.2], [0.1, 0.9]])
    assert score_predictions("binary", y, y_proba=proba) == 1.0
    y3 = np.array([0, 1, 2, 1])
    p3 = np.eye(3)[y3] * 0.94 + 0.02
    assert score_predictions("multiclass", y3, y_proba=p3) > 0.99
    assert score_predictions("regression", [1.0, 2.0], y_pred=[1.0, 2.0]) == 0.0
    # single-class fold: AUC undefined -> nan, never a crash
    assert np.isnan(score_predictions("binary", np.zeros(4), y_proba=proba))
    with pytest.raises(ValueError):
        score_predictions("ordinal", y, y_proba=proba)
    assert rmse_from_score(-2.5) == 2.5


def test_paired_comparison_all_outcomes():
    base = np.array([0.70, 0.72, 0.71, 0.73, 0.69, 0.71])
    better = paired_comparison(base + 0.05, base)
    worse = paired_comparison(base - 0.05, base)
    same = paired_comparison(base, base)
    assert better.outcome == "sig.better" and better.mean_delta > 0
    assert worse.outcome == "sig.worse" and worse.mean_delta < 0
    assert same.outcome == "preserved" and same.p_wilcoxon == 1.0
    assert set(better.to_dict()) == {"p_wilcoxon", "p_ttest", "mean_delta", "outcome"}
    assert isinstance(better, BaselineComparison)


def test_jaccard_chance_correction_edges():
    assert expected_jaccard(0, 0, 10) == 1.0
    assert expected_jaccard(5, 5, 0) == 1.0
    # both sets equal to the whole universe: expected overlap is total -> defined as 1
    assert adjusted_jaccard(range(10), range(10), 10) == 1.0


# --------------------------------------------------------------------------- #
# the sigmoid gate branch, orphaned when the default became Hard-Concrete
# --------------------------------------------------------------------------- #
def test_sigmoid_gate_branch_still_works():
    c = make_synthetic_images(n=250, task="binary", seed=0)
    sel = SoftMaskSelector(task="binary", patch=4, lam=0.03, gate="sigmoid",
                           n_iter=300, seed=0).fit(c.X, c.y)
    assert 0 < sel.coverage() < 0.5
    assert np.isfinite(sel.mask_).all()
    # sigmoid masks are soft in (0, 1), unlike the exact-0/1 Hard-Concrete ones
    assert ((sel.mask_ > 0) & (sel.mask_ < 1)).any()


def test_sigmoid_gate_with_tv_prior():
    c = make_synthetic_images(n=200, task="binary", seed=0)
    sel = SoftMaskSelector(task="binary", gate="sigmoid", tv=0.01,
                           n_iter=200, seed=0).fit(c.X, c.y)
    assert np.isfinite(sel.mask_).all()


# --------------------------------------------------------------------------- #
# representation validation and parallel paths
# --------------------------------------------------------------------------- #
def test_representation_shape_validation():
    class BadRep:
        def transform(self, X, grid):
            return np.zeros((X.shape[0], 3))          # wrong patch count

    c = make_synthetic_images(n=40, seed=0)
    with pytest.raises(ValueError, match="patches"):
        SoftMaskSelector(task="binary", representation=BadRep(), n_iter=20).fit(c.X, c.y)


def test_bootstrap_parallel_path_matches_serial():
    c = make_synthetic_images(n=150, seed=0)
    kw = dict(task="binary", n_bootstrap=4, n_iter=150, seed=0)
    serial = BootstrapMaskSelector(n_jobs=1, **kw).fit(c.X, c.y)
    parallel = BootstrapMaskSelector(n_jobs=2, **kw).fit(c.X, c.y)
    np.testing.assert_array_equal(serial.pi_, parallel.pi_)


def test_ensemble_parallel_path_matches_serial():
    c = make_synthetic_images(n=120, seed=0)
    kw = dict(task="binary", n_bootstrap=2, n_representations=2, n_iter=120, seed=0)
    serial = RepresentationEnsembleSelector(n_jobs=1, **kw).fit(c.X, c.y)
    parallel = RepresentationEnsembleSelector(n_jobs=2, **kw).fit(c.X, c.y)
    np.testing.assert_array_equal(serial.pi_, parallel.pi_)


def test_natural_inner_failure_is_counted_via_real_exception():
    """Fits fail NATURALLY here (binary task, 3-class y), covering the except path."""
    c = make_synthetic_images(n=90, seed=0)
    y3 = np.arange(90) % 3
    with pytest.raises(RuntimeError, match="systematic"):
        BootstrapMaskSelector(task="binary", n_bootstrap=3, n_iter=40, seed=0).fit(c.X, y3)


# --------------------------------------------------------------------------- #
# regimes and synthetic validation branches
# --------------------------------------------------------------------------- #
def test_grid_validation_errors():
    with pytest.raises(ValueError, match="patch"):
        PatchGrid(8, 8, patch=0)
    with pytest.raises(ValueError, match="H, W"):
        PatchGrid.from_image_shape((10,), patch=2)
    grid = PatchGrid(8, 8, patch=4)
    with pytest.raises(ValueError, match="does not match"):
        grid.pool(np.zeros((2, 10, 10)))


def test_synthetic_validation_and_no_signal_branch():
    with pytest.raises(ValueError, match="do not fit"):
        make_imgs(n=10, shape=(12, 12), region_side=4, cell=4)
    c = make_imgs(n=30, shape=(32, 32), region_side=0, distractor_side=1, seed=0)
    assert not c.informative.any()          # label is pure chance by construction
    with pytest.raises(ValueError, match="task"):
        make_imgs(n=10, task="ordinal")


# --------------------------------------------------------------------------- #
# results and engine odds and ends
# --------------------------------------------------------------------------- #
def test_regression_result_display_and_grouped_regression_splitter():
    from sklearn.model_selection import GroupKFold

    from robustpixelmaker import make_inner_splitter, make_outer_splitter

    assert isinstance(make_outer_splitter("regression", np.arange(12), 3, 0), GroupKFold)
    assert isinstance(make_inner_splitter("regression", True, 3, 0), GroupKFold)

    c = make_synthetic_images(n=90, task="regression", seed=0)
    res = NestedCV(k_outer=3, l_inner=3, n_iter=3, random_state=0).run(c.X, c.y)
    val, sd, name = res.display_score()
    assert name == "RMSE" and val > 0


def test_binary_result_proba_passthrough_and_choose_tau_fallback():
    c = make_synthetic_images(n=90, task="binary", seed=0)
    res = NestedCV(k_outer=3, l_inner=3, n_iter=3, random_state=0).run(c.X, c.y)
    assert res.predict_proba(c.X[:4]).shape == (4, 2)

    # target_coverage with an all-zero frequency map falls back to the fixed tau
    sel = BootstrapMaskSelector(task="binary", target_coverage=0.2, tau=0.7)
    assert sel._choose_tau(np.zeros(10)) == 0.7
