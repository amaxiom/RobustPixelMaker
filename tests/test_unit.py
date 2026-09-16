"""Unit tests for the Milestone-1 scaffold."""
import numpy as np
import pytest

from robustpixelmaker import (
    NestedCV,
    RPMResult,
    Seeds,
    detect_device,
    fit_count,
    infer_task,
    make_outer_splitter,
)
from robustpixelmaker.reproducibility import BOOTSTRAP_OFFSET, REPEAT_STRIDE, REPRESENTATION_OFFSET


# --- fixtures -------------------------------------------------------------- #
def make_binary(n=120, d=10, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    w = np.zeros(d); w[:3] = [2.0, -2.0, 1.5]
    y = ((X @ w) + rng.normal(scale=0.5, size=n) > 0).astype(int)
    return X, y


def make_multiclass(n=150, d=10, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    W = rng.normal(size=(d, 3)); W[3:] = 0
    y = (X @ W + rng.normal(scale=0.5, size=(n, 3))).argmax(1)
    return X, y


def make_regression(n=120, d=10, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    w = np.zeros(d); w[:3] = [3.0, -1.0, 2.0]
    y = X @ w + rng.normal(scale=0.3, size=n)
    return X, y


# --- task inference -------------------------------------------------------- #
def test_infer_task_binary():
    _, y = make_binary()
    assert infer_task(y) == "binary"


def test_infer_task_multiclass():
    _, y = make_multiclass()
    assert infer_task(y) == "multiclass"


def test_infer_task_regression():
    _, y = make_regression()
    assert infer_task(y) == "regression"


def test_infer_task_float_integer_labels_are_classification():
    y = np.array([0.0, 1.0, 2.0, 1.0, 0.0, 2.0] * 5)
    assert infer_task(y) == "multiclass"


# --- cost model & device --------------------------------------------------- #
def test_fit_count_formula():
    # per = 3*20 + 20*5 + 1 = 161; total = 5*1*161 + 161 = 966
    assert fit_count(5, 1, 3, 20, 20, 5) == 966


def test_detect_device_returns_valid():
    assert detect_device("auto") in {"cuda", "mps", "cpu"}


def test_detect_device_absent_torch_falls_back(monkeypatch):
    import robustpixelmaker.config as cfg
    monkeypatch.setattr(cfg, "torch_available", lambda: False)
    assert cfg.detect_device("cuda") == "cpu"  # graceful, no crash


# --- splitter selection ---------------------------------------------------- #
def test_grouped_classification_uses_stratified_group_kfold():
    from sklearn.model_selection import StratifiedGroupKFold
    sp = make_outer_splitter("binary", groups=np.arange(10), n_splits=5, seed=0)
    assert isinstance(sp, StratifiedGroupKFold)


def test_ungrouped_regression_uses_kfold():
    from sklearn.model_selection import KFold
    sp = make_outer_splitter("regression", groups=None, n_splits=5, seed=0)
    assert isinstance(sp, KFold)


# --- seed derivation ------------------------------------------------------- #
def test_seeds_are_deterministic_offsets():
    s = Seeds(base=7, n_bootstrap=50, n_representations=4)
    assert s.bootstrap(0) == 7 + BOOTSTRAP_OFFSET
    assert s.representation(0) == 7 + REPRESENTATION_OFFSET
    assert s.outer(3, repeat=0) == 10
    assert s.outer(0, repeat=1) == 7 + REPEAT_STRIDE


def test_seeds_collision_guard():
    with pytest.raises(ValueError):
        Seeds(base=0, n_bootstrap=REPRESENTATION_OFFSET)  # would alias representation seeds


# --- end to end ------------------------------------------------------------ #
@pytest.mark.parametrize("maker,task", [
    (make_binary, "binary"),
    (make_multiclass, "multiclass"),
    (make_regression, "regression"),
])
def test_end_to_end_runs(maker, task):
    X, y = maker()
    res = NestedCV(k_outer=5, l_inner=3, n_iter=5, random_state=0).run(X, y)
    assert isinstance(res, RPMResult)
    assert res.task == task
    assert len(res.per_fold_scores) == 5
    assert np.isfinite(res.mean_score)
    assert 0.0 <= res.selection_stability <= 1.0 or np.isnan(res.selection_stability)
    # final estimator predicts on new data
    pred = res.predict(X[:8])
    assert len(pred) == 8


def test_binary_scaffold_beats_chance():
    X, y = make_binary()
    res = NestedCV(k_outer=5, l_inner=3, n_iter=5, random_state=0).run(X, y)
    assert res.mean_score > 0.6  # informative features -> better than chance AUC
