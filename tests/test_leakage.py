"""Leakage-safety tests, the structural guarantee, verified with a canary.

The strong test is a canary fold estimator that hashes every training row it is
fitted on and raises if it is ever asked to predict a row it saw during fit. Run
through the engine, it proves the held-out partition is never handed to fold fitting.
"""
import numpy as np
import pytest

from robustpixelmaker import NestedCV
from robustpixelmaker.nested_cv import make_outer_splitter


class LeakageError(AssertionError):
    pass


class CanaryFoldEstimator:
    """Records fit-row hashes; raises on any predict-row that overlaps them."""

    def __init__(self):
        self._fit_rows = frozenset()
        self.selected_ = np.array([], dtype=int)
        self._task = None
        self._classes = None

    @staticmethod
    def _hashes(X):
        X = np.ascontiguousarray(np.asarray(X, dtype=float))
        return {X[i].tobytes() for i in range(X.shape[0])}

    def fit(self, X, y, *, task, seeds, fold_idx, repeat, inner_splitter, groups=None):
        self._task = task
        self._fit_rows = frozenset(self._hashes(X))
        self._classes = None if task == "regression" else np.unique(y)
        return self

    def _check(self, X):
        if self._hashes(X) & self._fit_rows:
            raise LeakageError("a held-out row was seen during fit -> leakage")

    def predict(self, X):
        self._check(X)
        return np.zeros(np.asarray(X).shape[0])

    def predict_proba(self, X):
        self._check(X)
        n = np.asarray(X).shape[0]
        k = len(self._classes)
        return np.full((n, k), 1.0 / k)


def _data(seed=0, n=120, d=8):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = (X[:, 0] + rng.normal(scale=0.5, size=n) > 0).astype(int)
    groups = np.repeat(np.arange(n // 3), 3)  # 3 samples per group
    return X, y, groups


def test_canary_no_leak_ungrouped():
    X, y, _ = _data()
    # Should complete without the canary firing.
    NestedCV(estimator_factory=CanaryFoldEstimator, k_outer=5, random_state=0).run(X, y)


def test_canary_no_leak_grouped():
    X, y, groups = _data()
    NestedCV(estimator_factory=CanaryFoldEstimator, k_outer=4, random_state=0).run(X, y, groups=groups)


def test_canary_detects_injected_leak():
    # Sanity-check the canary itself: overlapping rows must raise.
    X, y, _ = _data()
    est = CanaryFoldEstimator().fit(
        X[:60], y[:60], task="binary", seeds=None, fold_idx=0, repeat=0, inner_splitter=None
    )
    with pytest.raises(AssertionError):
        est.predict(X[:10])  # these rows were in the fit set


def test_no_group_spans_train_and_test():
    X, y, groups = _data()
    sp = make_outer_splitter("binary", groups=groups, n_splits=4, seed=0)
    for tr, te in sp.split(X, y, groups):
        assert set(groups[tr]).isdisjoint(set(groups[te]))
