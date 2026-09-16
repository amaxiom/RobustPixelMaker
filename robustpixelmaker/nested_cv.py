"""Leakage-safe nested cross-validation engine.

The leakage-safety contract is *structural*, not conventional (README.md, How it works):
the engine slices train/test indices and only ever passes the TRAIN partition to a
fold estimator's ``fit``. A fold estimator performs all preprocessing, inner
hyperparameter search and (later) bootstrap mask selection inside ``fit`` on train
data alone; the held-out partition is touched only by ``predict`` at scoring time.
Because the test rows are never handed to ``fit``, no fold-level step can leak.

Milestone 1 ships ``SimpleFoldEstimator``, a scikit-learn placeholder that obeys
the contract end-to-end so the scaffold is runnable, reproducible and leakage-safe.
It is replaced in later milestones by the representation-marginalised bootstrap mask
selector; the engine and contract do not change.
"""
from __future__ import annotations

import time
from typing import Optional, Protocol, runtime_checkable

import numpy as np

from .metrics import mean_pairwise_jaccard, score_predictions
from .reproducibility import Seeds
from .results import RPMResult
from typing import Any
from numpy.typing import ArrayLike


# --------------------------------------------------------------------------- #
# Task inference & splitter selection
# --------------------------------------------------------------------------- #
def infer_task(y: ArrayLike) -> str:
    """Infer 'binary' | 'multiclass' | 'regression' from the target (RMM-style).

    Float targets with many distinct values are regression; integer/categorical
    targets are classification, binary when exactly two classes are present.
    """
    y = np.asarray(y)
    if y.dtype.kind == "f":
        # floats that are actually whole-numbered class labels -> classification
        if np.all(np.equal(np.mod(y, 1), 0)) and len(np.unique(y)) <= max(20, int(0.05 * len(y))):
            n_classes = len(np.unique(y))
            return "binary" if n_classes == 2 else "multiclass"
        return "regression"
    n_classes = len(np.unique(y))
    return "binary" if n_classes == 2 else "multiclass"


def make_outer_splitter(task: str, groups: ArrayLike | None, n_splits: int,
                        seed: int) -> Any:
    from sklearn.model_selection import (
        GroupKFold,
        KFold,
        StratifiedGroupKFold,
        StratifiedKFold,
    )

    if groups is not None:
        # Subject/specimen grouping: no group may span train and test (slice leakage).
        if task in ("binary", "multiclass"):
            return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return GroupKFold(n_splits=n_splits)  # deterministic by group order
    if task in ("binary", "multiclass"):
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return KFold(n_splits=n_splits, shuffle=True, random_state=seed)


def make_inner_splitter(task: str, grouped: bool, n_splits: int, seed: int) -> Any:
    from sklearn.model_selection import (
        GroupKFold,
        KFold,
        StratifiedGroupKFold,
        StratifiedKFold,
    )

    if grouped:
        if task in ("binary", "multiclass"):
            return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return GroupKFold(n_splits=n_splits)
    if task in ("binary", "multiclass"):
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return KFold(n_splits=n_splits, shuffle=True, random_state=seed)


def align_proba(proba: ArrayLike, fold_classes: ArrayLike | None,
                classes: ArrayLike) -> np.ndarray:
    """Map a fold's probability columns onto the GLOBAL class order.

    A fold estimator only knows the classes present in its training partition. When a
    rare class is missing from one training fold, which is routine in the small-sample
    regime this framework targets, the fold's ``predict_proba`` has fewer columns than
    the global class list and a naive accumulation crashes (or worse, silently
    misaligns if the widths happen to match by coincidence of ordering). Missing
    classes get probability zero, which is exactly what the fold's model believes.
    """
    proba = np.asarray(proba, dtype=float)
    classes = np.asarray(classes)
    if fold_classes is None:
        if proba.shape[1] != len(classes):
            raise ValueError(
                f"fold produced {proba.shape[1]} probability columns for "
                f"{len(classes)} classes and did not expose classes_ to align them")
        return proba
    fold_classes = np.asarray(fold_classes)
    if proba.shape[1] != len(fold_classes):
        raise ValueError(
            f"predict_proba returned {proba.shape[1]} columns but classes_ has "
            f"{len(fold_classes)} entries")
    out = np.zeros((proba.shape[0], len(classes)), dtype=float)
    col = {cls: k for k, cls in enumerate(classes.tolist())}
    for j, cls in enumerate(fold_classes.tolist()):
        out[:, col[cls]] = proba[:, j]
    return out


# --------------------------------------------------------------------------- #
# Fold-estimator contract
# --------------------------------------------------------------------------- #
@runtime_checkable
class FoldEstimator(Protocol):
    """Contract for the per-fold model. ``fit`` sees TRAIN data only."""

    def fit(self, X, y, *, task: str, seeds: Seeds, fold_idx: int, repeat: int,
            inner_splitter, groups=None) -> "FoldEstimator": ...

    def predict(self, X: ArrayLike) -> np.ndarray: ...

    def predict_proba(self, X: ArrayLike) -> np.ndarray: ...

    # populated by fit: 1-D array of selected feature/region indices
    selected_: np.ndarray


class SimpleFoldEstimator:
    """Placeholder fold estimator (Milestone-1 scaffold).

    Per-fold standardisation + a randomized inner hyperparameter search, all fitted
    on the training partition only. "Selection" is a stand-in: features whose
    |standardised coefficient| exceeds the median. Replaced later by the bootstrap
    mask selector; kept only to exercise the scaffold end-to-end.
    """

    def __init__(self, n_iter: int = 20):
        self.n_iter = n_iter
        self.pipeline_: Any = None      # set by fit; predict is only valid after
        self.task_ = None
        self.selected_ = np.array([], dtype=int)

    @staticmethod
    def _flatten(X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return X.reshape(X.shape[0], -1) if X.ndim > 2 else X

    def fit(self, X, y, *, task, seeds, fold_idx, repeat, inner_splitter, groups=None):
        from scipy.stats import loguniform
        from sklearn.linear_model import LogisticRegression, Ridge
        from sklearn.model_selection import RandomizedSearchCV
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        self.task_ = task
        Xf = self._flatten(X)
        y = np.asarray(y)
        seed = seeds.inner(fold_idx, repeat)

        if task == "regression":
            model = Ridge(random_state=seed)
            params = {"model__alpha": loguniform(1e-2, 1e2)}
            scoring = "neg_root_mean_squared_error"
        else:
            model = LogisticRegression(max_iter=1000, random_state=seed)
            params = {"model__C": loguniform(1e-2, 1e2)}
            scoring = "roc_auc" if task == "binary" else "roc_auc_ovr_weighted"

        pipe = Pipeline([("scaler", StandardScaler()), ("model", model)])
        search = RandomizedSearchCV(
            pipe, params, n_iter=self.n_iter, cv=inner_splitter,
            scoring=scoring, random_state=seed, n_jobs=1, refit=True,
        )
        search.fit(Xf, y, groups=groups)
        self.pipeline_ = search.best_estimator_
        # classes seen by THIS fold's training partition, for engine-side alignment
        self.classes_ = (None if task == "regression"
                         else self.pipeline_.named_steps["model"].classes_)

        coef = self.pipeline_.named_steps["model"].coef_
        importance = np.abs(coef).max(axis=0) if coef.ndim == 2 else np.abs(coef)
        thresh = np.median(importance)
        self.selected_ = np.where(importance > thresh)[0].astype(int)
        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        return self.pipeline_.predict(self._flatten(X))

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        return self.pipeline_.predict_proba(self._flatten(X))


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #
class NestedCV:
    """Leakage-safe nested CV over a :class:`FoldEstimator` factory."""

    def __init__(self, estimator_factory=None, *, k_outer=5, l_inner=5,
                 repeated_outer_cv=1, n_iter=20, random_state=0):
        self.estimator_factory = estimator_factory or (lambda: SimpleFoldEstimator(n_iter=n_iter))
        self.k_outer = k_outer
        self.l_inner = l_inner
        self.repeated_outer_cv = repeated_outer_cv
        self.n_iter = n_iter
        self.random_state = random_state

    def run(self, X: ArrayLike, y: ArrayLike,
            groups: ArrayLike | None = None) -> RPMResult:
        X = np.asarray(X)
        y = np.asarray(y)
        groups = None if groups is None else np.asarray(groups)
        n = X.shape[0]
        task = infer_task(y)
        seeds = Seeds(base=self.random_state)

        # out-of-fold accumulators (averaged across repeats). `classes` is derived
        # inside the branch that uses it: computing it above with a separate
        # `if task != "regression"` left the correlation between the two conditions
        # implicit, so a reader (and a type checker) could not see that `classes` is
        # never None where it is indexed.
        classes: np.ndarray | None
        if task == "regression":
            classes = None
            oof_sum = np.zeros(n, dtype=float)
        else:
            classes = np.unique(y)
            oof_sum = np.zeros((n, len(classes)), dtype=float)
        oof_count = np.zeros(n, dtype=int)

        fold_records: list[dict] = []
        per_fold_scores: list[float] = []
        selected_per_fold: list[np.ndarray] = []

        for repeat in range(self.repeated_outer_cv):
            outer = make_outer_splitter(task, groups, self.k_outer, seeds.outer(0, repeat))
            for fold_idx, (tr, te) in enumerate(outer.split(X, y, groups)):
                t0 = time.perf_counter()
                inner = make_inner_splitter(
                    task, groups is not None, self.l_inner, seeds.inner(fold_idx, repeat)
                )
                est = self.estimator_factory()
                est.fit(
                    X[tr], y[tr], task=task, seeds=seeds, fold_idx=fold_idx,
                    repeat=repeat, inner_splitter=inner,
                    groups=None if groups is None else groups[tr],
                )
                if task == "regression":
                    pred = est.predict(X[te])
                    oof_sum[te] += pred
                    proba = None
                    score = score_predictions(task, y[te], y_pred=pred)
                else:
                    assert classes is not None   # set on every non-regression path
                    proba = align_proba(est.predict_proba(X[te]),
                                        getattr(est, "classes_", None), classes)
                    oof_sum[te] += proba
                    score = score_predictions(task, y[te], y_proba=proba)
                oof_count[te] += 1

                per_fold_scores.append(score)
                selected_per_fold.append(np.asarray(est.selected_, dtype=int))
                fold_records.append({
                    "repeat": repeat, "fold": fold_idx,
                    "n_test": int(len(te)), "n_selected": int(len(est.selected_)),
                    "score": float(score),
                    "seed_outer": seeds.outer(fold_idx, repeat),
                    "seed_inner": seeds.inner(fold_idx, repeat),
                    "seconds": time.perf_counter() - t0,
                })

        safe_count = np.maximum(oof_count, 1)
        if task == "regression":
            oof = oof_sum / safe_count
        else:
            oof = oof_sum / safe_count[:, None]

        # Final refit on ALL training data for downstream predict().
        final = self.estimator_factory()
        final_inner = make_inner_splitter(task, groups is not None, self.l_inner, seeds.inner(0, 0))
        final.fit(X, y, task=task, seeds=seeds, fold_idx=0, repeat=0,
                  inner_splitter=final_inner, groups=groups)

        return RPMResult(
            task=task,
            random_state=self.random_state,
            per_fold_scores=np.asarray(per_fold_scores, dtype=float),
            oof_predictions=oof,
            oof_count=oof_count,
            selected_per_fold=selected_per_fold,
            selection_stability=mean_pairwise_jaccard(selected_per_fold),
            fold_records=fold_records,
            final_estimator=final,
            classes=None if classes is None else np.asarray(classes),
            config={
                "k_outer": self.k_outer, "l_inner": self.l_inner,
                "repeated_outer_cv": self.repeated_outer_cv, "n_iter": self.n_iter,
                "grouped": groups is not None,
            },
        )
