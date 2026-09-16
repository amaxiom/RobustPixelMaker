"""Reproducibility suite (RMM-style): exact repro, seed diversity, serialisation."""
import numpy as np

from robustpixelmaker import NestedCV, RPMResult


def _data(seed=0, n=120, d=10):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    w = np.zeros(d); w[:3] = [2.0, -2.0, 1.5]
    y = ((X @ w) + rng.normal(scale=0.5, size=n) > 0).astype(int)
    return X, y


def test_exact_reproducibility():
    X, y = _data()
    r1 = NestedCV(k_outer=5, l_inner=3, n_iter=5, random_state=0).run(X, y)
    r2 = NestedCV(k_outer=5, l_inner=3, n_iter=5, random_state=0).run(X, y)
    # identical to machine precision: same data, params, seed
    np.testing.assert_array_equal(r1.per_fold_scores, r2.per_fold_scores)
    np.testing.assert_array_equal(r1.oof_predictions, r2.oof_predictions)
    assert [list(s) for s in r1.selected_per_fold] == [list(s) for s in r2.selected_per_fold]


def test_seed_diversity_changes_outputs():
    # Negative control: guards against silent caching / seed being ignored.
    X, y = _data()
    r0 = NestedCV(k_outer=5, l_inner=3, n_iter=5, random_state=0).run(X, y)
    r1 = NestedCV(k_outer=5, l_inner=3, n_iter=5, random_state=1).run(X, y)
    assert not np.array_equal(r0.per_fold_scores, r1.per_fold_scores)


def test_serialization_roundtrip(tmp_path):
    X, y = _data()
    res = NestedCV(k_outer=5, l_inner=3, n_iter=5, random_state=0).run(X, y)
    out = res.save(tmp_path)
    assert (out / "rpm_summary.json").exists()
    assert (out / "rpm_folds.csv").exists()
    loaded = RPMResult.load(out / "rpm_result.pkl")
    np.testing.assert_array_equal(loaded.predict(X[:10]), res.predict(X[:10]))
    np.testing.assert_array_equal(loaded.per_fold_scores, res.per_fold_scores)


def test_repeated_outer_cv_reproducible():
    X, y = _data()
    kw = dict(k_outer=5, l_inner=3, n_iter=5, repeated_outer_cv=2, random_state=0)
    r1 = NestedCV(**kw).run(X, y)
    r2 = NestedCV(**kw).run(X, y)
    assert len(r1.per_fold_scores) == 10
    np.testing.assert_array_equal(r1.per_fold_scores, r2.per_fold_scores)
