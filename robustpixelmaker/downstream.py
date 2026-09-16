"""The model fitted on the selected region, which need not be the mask's own head.

RobustModelMaker lets the caller choose the model that consumes the selected features.
RPM did not: both aggregators end by refitting ``SoftMaskSelector`` with a fixed mask,
so whatever you selected, the deployable predictor was always the mask's linear (or,
since sec.17, one-hidden-layer) head on patch means.

That gap had a measurable cost. Every benchmark in this suite scores a selection by
refitting a RANDOM FOREST on the selected patches, because that is the sensible
downstream model for this data, while the library handed users a linear head. The
library was therefore evaluated with one model class and shipped another, and part of
the score gap between RPM and the full-image baseline is that difference rather than
anything about the selection.

``resolve_model`` turns a specification into an estimator fitted on the selected patches
only. ``None`` keeps the previous behaviour exactly, so no existing result moves.

The selection is unchanged by this choice: the mask is still what picks the patches, and
the downstream model only decides what is fitted on them. That separation is the point,
because it lets the library answer a question it previously could not, namely whether a
selection helps a good model or only a weak one.
"""
from __future__ import annotations

import numpy as np
from typing import Any
from numpy.typing import ArrayLike

#: named models, so a caller need not import sklearn to ask for a common one
NAMED_MODELS = ("mask", "rf", "logreg", "ridge", "gb", "svm", "knn")


def _build(name: str, task: str, seed: int):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    regression = task == "regression"
    if name == "rf":
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        RF = RandomForestRegressor if regression else RandomForestClassifier
        return RF(n_estimators=200, random_state=seed)
    if name == "gb":
        from sklearn.ensemble import (GradientBoostingClassifier,
                                      GradientBoostingRegressor)
        GB = GradientBoostingRegressor if regression else GradientBoostingClassifier
        return GB(random_state=seed)
    if name in ("logreg", "ridge"):
        from sklearn.linear_model import LogisticRegression, Ridge
        est = Ridge(random_state=seed) if regression else LogisticRegression(
            max_iter=2000, random_state=seed)
        return make_pipeline(StandardScaler(), est)
    if name == "svm":
        from sklearn.svm import SVC, SVR
        est = SVR() if regression else SVC(probability=True, random_state=seed)
        return make_pipeline(StandardScaler(), est)
    if name == "knn":
        from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
        KN = KNeighborsRegressor if regression else KNeighborsClassifier
        return make_pipeline(StandardScaler(), KN(n_neighbors=5))
    raise ValueError(f"unknown model {name!r}; expected one of {NAMED_MODELS} "
                     "or a scikit-learn estimator instance")


def resolve_model(spec: Any, task: str, seed: int) -> Any:
    """None or "mask" keeps the mask head; a name or an estimator gives a new model."""
    if spec is None or spec == "mask":
        return None
    if isinstance(spec, str):
        return _build(spec, task, seed)
    if hasattr(spec, "fit") and hasattr(spec, "predict"):
        from sklearn.base import clone
        try:
            return clone(spec)          # never mutate the caller's object across folds
        except Exception:
            return spec
    raise ValueError(f"model must be None, one of {NAMED_MODELS}, or a scikit-learn "
                     f"estimator; got {type(spec).__name__}")


class DownstreamModel:
    """A model fitted on the selected patches, sharing the selector's patch grid.

    Holds the grid and the selected ids so ``predict`` reproduces exactly the feature
    matrix ``fit`` saw. An empty selection is refused rather than silently scored at
    chance, because a downstream model fitted on nothing is not a model.
    """

    def __init__(self, estimator: Any, grid: Any, selected: ArrayLike,
                 task: str) -> None:
        self.estimator = estimator
        self.grid = grid
        self.selected = np.asarray(selected, dtype=int)
        self.task = task

    def _features(self, X):
        F = self.grid.pool(np.asarray(X, dtype=float))
        return F[:, self.selected]

    def fit(self, X: ArrayLike, y: ArrayLike) -> DownstreamModel:
        if self.selected.size == 0:
            raise ValueError(
                "the selection is empty, so there is nothing for a downstream model to "
                "fit; loosen lam or raise target_coverage")
        self.estimator.fit(self._features(X), np.asarray(y))
        self.classes_ = getattr(self.estimator, "classes_", None)
        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        return self.estimator.predict(self._features(X))

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        if self.task == "regression":
            raise ValueError("predict_proba is undefined for task='regression'")
        if not hasattr(self.estimator, "predict_proba"):
            raise ValueError(
                f"{type(self.estimator).__name__} has no predict_proba; choose a "
                "probabilistic model or use predict")
        return self.estimator.predict_proba(self._features(X))


def attach(spec: Any, selector: Any, X: ArrayLike, y: ArrayLike, task: str,
           seed: int) -> DownstreamModel | None:
    """Fit a downstream model on ``selector``'s selection, or None to keep the head."""
    est = resolve_model(spec, task, seed)
    if est is None:
        return None
    return DownstreamModel(est, selector.grid, selector.selected_patches_, task).fit(X, y)
