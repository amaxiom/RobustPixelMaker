"""The RobustPixelMaker estimator: the RobustMaker-standard public surface.

This facade gives RPM the same shape as RobustModelMaker: one class with a
scikit-learn-style ``fit`` / ``predict`` / ``predict_proba`` interface, a functional
``run_pipeline`` equivalent, a rich result object, and file exports. Everything it does
is composed from the primitives in this package, which remain public for users who want
the pieces individually.

``fit`` does two things, in this order:

1. **Leakage-safe nested cross-validation** over the training data, using the chosen
   selector inside every outer fold, to produce the HONEST deliverables: per-fold
   scores, cross-fold selection stability, and out-of-fold predictions. Nothing from
   any held-out fold informs any fitted quantity.
2. **A final refit on all of the data**, which yields the deployable predictor, the
   selection-frequency map ``pi_``, the stable region ``selected_patches_``, and the
   go/no-go diagnostics (``ambiguous_fraction``, ``stability_report``).

The three deliverables the family treats as first class are therefore all present: a
reproducible selected region, an honest performance estimate, and the diagnostic that
says whether the region deserves to be trusted at all.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .masking import SoftMaskSelector
from .downstream import NAMED_MODELS, attach
from .nested_cv import NestedCV, infer_task
from .selection import (
    BootstrapMaskFoldEstimator,
    BootstrapMaskSelector,
    EnsembleMaskFoldEstimator,
    RepresentationEnsembleSelector,
)
from numpy.typing import ArrayLike
from .metrics import BaselineComparison

#: selector kinds, ordered from cheapest to the full method
SELECTOR_KINDS = ("single", "bootstrap", "ensemble")


class RobustPixelMaker:
    """Reproducible, leakage-safe pixel/patch selection for registered scientific images.

    Parameters mirror the underlying selectors; every knob has a working default except
    ``registered``, which the caller must own (see README.md, Quick start): a global mask
    over unregistered images is meaningless, and no default guess is safe.

    ``selector`` picks the machinery inside each fold and for the final refit:

      "single"     one mask fit (Milestone 2). Cheapest; no frequency map.
      "bootstrap"  stability selection over data resamples (Milestone 4).
      "ensemble"   the full method: the double bootstrap over data resamples AND
                   representations (Milestone 5). Default, and the configuration the
                   benchmark evidence in benchmarks/FINDINGS.md refers to.

    ``task`` is inferred from ``y`` when None, exactly as RobustModelMaker infers it.
    """

    def __init__(self, registered: bool = True, selector: str = "ensemble",
                 task: str | None = None, patch: int = 4, lam: float = 0.03,
                 tau: float = 0.7, target_coverage: float | None = None,
                 n_bootstrap: int = 20, n_representations: int = 4,
                 gate: str = "hardconcrete", tv: float = 0.0,
                 subsample: str = "bootstrap", k_outer: int = 5,
                 n_iter: int = 500, n_jobs: int = 1, random_state: int = 0,
                 model=None, head: str = "linear", hidden: int = 16,
                 lam_scale: str = "fixed", auto_target: float = 0.15):
        if selector not in SELECTOR_KINDS:
            raise ValueError(f"selector must be one of {SELECTOR_KINDS}")
        if head not in ("linear", "mlp"):
            raise ValueError("head must be one of {'linear','mlp'}")
        if lam_scale not in ("fixed", "auto"):
            raise ValueError("lam_scale must be one of {'fixed','auto'}")
        if not 0.0 < float(auto_target) < 1.0:
            raise ValueError("auto_target is the fraction of the mask to leave open "
                             "and must be in (0, 1)")
        self.registered = registered
        # The model fitted on the selected region. None keeps the mask's own head, which
        # is what every result before this option existed used. A name from NAMED_MODELS
        # or any scikit-learn estimator fits that instead, inside the same leakage-safe
        # folds. The selection is unaffected either way: the mask still chooses the
        # patches, and this only decides what is fitted on them.
        self.model = model
        # The mask's own head, and the penalty scaling it needs. These were reachable
        # only by constructing a selector directly, while the README documented
        # head="mlp" as a library feature: the facade could not set it, so the
        # documented option was unreachable from the documented entry point.
        #
        # head="mlp" matters when the informative pixels act through an interaction
        # rather than a main effect, since a frozen LINEAR head cannot reward a gate
        # with no main effect and phase 2 then prunes the whole informative region
        # (FINDINGS sec.16). It is opt-in: it costs more, and it is not uniformly
        # better, sec.19. With lam_scale="auto", auto_target is a coarse prior on
        # coverage rather than a neutral knob, sec.21.
        self.head = head
        self.hidden = hidden
        self.lam_scale = lam_scale
        self.auto_target = auto_target
        self.selector = selector
        self.task = task
        self.patch = patch
        self.lam = lam
        self.tau = tau
        self.target_coverage = target_coverage
        self.n_bootstrap = n_bootstrap
        self.n_representations = n_representations
        self.gate = gate
        self.tv = tv
        self.subsample = subsample
        self.k_outer = k_outer
        self.n_iter = n_iter
        self.n_jobs = n_jobs
        self.random_state = random_state

    # ------------------------------------------------------------------ #
    def _check_regime(self):
        if not self.registered:
            raise NotImplementedError(
                "registered=False (the conditional per-image mask for unregistered "
                "fields of view, e.g. wide-field TEM) is a documented extension point "
                "and is not implemented in this release. RELATED_WORK.md sec.10 records the "
                "design; the stable object there is a gating function rather than a "
                "coordinate region, and its stability metric is an open problem."
            )

    def _fold_factory(self):
        common = dict(patch=self.patch, lam=self.lam, tau=self.tau,
                      n_bootstrap=self.n_bootstrap, subsample=self.subsample,
                      target_coverage=self.target_coverage, gate=self.gate,
                      tv=self.tv, n_iter=self.n_iter, n_jobs=self.n_jobs,
                      model=self.model, head=self.head, hidden=self.hidden,
                      lam_scale=self.lam_scale, auto_target=self.auto_target)
        if self.selector == "ensemble":
            return lambda: EnsembleMaskFoldEstimator(
                n_representations=self.n_representations, **common)
        if self.selector == "bootstrap":
            return lambda: BootstrapMaskFoldEstimator(**common)
        from .masking import SoftMaskFoldEstimator
        return lambda: SoftMaskFoldEstimator(patch=self.patch, lam=self.lam,
                                             gate=self.gate, tv=self.tv,
                                             model=self.model, head=self.head,
                                             hidden=self.hidden,
                                             lam_scale=self.lam_scale,
                                             auto_target=self.auto_target,
                                             n_iter=self.n_iter)

    def _final_selector(self, task):
        common = dict(task=task, patch=self.patch, lam=self.lam, gate=self.gate,
                      tv=self.tv, n_iter=self.n_iter, seed=self.random_state,
                      head=self.head, hidden=self.hidden,
                      lam_scale=self.lam_scale, auto_target=self.auto_target)
        if self.selector == "ensemble":
            return RepresentationEnsembleSelector(
                tau=self.tau, target_coverage=self.target_coverage,
                n_bootstrap=self.n_bootstrap, subsample=self.subsample,
                n_representations=self.n_representations, n_jobs=self.n_jobs, **common)
        if self.selector == "bootstrap":
            return BootstrapMaskSelector(
                tau=self.tau, target_coverage=self.target_coverage,
                n_bootstrap=self.n_bootstrap, subsample=self.subsample,
                n_jobs=self.n_jobs, **common)
        return SoftMaskSelector(**common)

    # ------------------------------------------------------------------ #
    def fit(self, X: ArrayLike, y: ArrayLike,
            groups: ArrayLike | None = None) -> RobustPixelMaker:
        """Nested-CV evaluation, then a final refit on all data. Returns self."""
        self._check_regime()
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        inferred = infer_task(y)
        if self.task is not None and self.task != inferred:
            # The nested-CV engine infers the task from y for its fold estimators; if
            # an explicit facade task disagreed, the honest evaluation half and the
            # deployable final selector would silently be fitted for DIFFERENT tasks.
            raise ValueError(
                f"task={self.task!r} was requested but y looks like {inferred!r}; "
                "pass task=None to accept the inference, or fix y")
        self.task_ = inferred

        self.result_ = NestedCV(
            estimator_factory=self._fold_factory(),
            k_outer=self.k_outer,
            random_state=self.random_state,
        ).run(X, y, groups=groups)

        self.selector_ = self._final_selector(self.task_).fit(X, y)
        self.grid_ = self.selector_.grid
        self.selected_patches_ = self.selector_.selected_patches_
        self.mask_ = self.selector_.mask_
        self.pi_ = getattr(self.selector_, "pi_", None)
        self.tau_ = getattr(self.selector_, "tau_", None)
        # the deployable predictor: a downstream model on the selected region when one
        # was asked for, otherwise the mask's own head as before
        self.downstream_ = attach(self.model, self.selector_, X, y,
                                  self.task_, self.random_state)
        return self

    # ------------------------------------------------------------------ #
    def _predictor(self):
        return self.selector_ if self.downstream_ is None else self.downstream_

    def predict(self, X: ArrayLike) -> np.ndarray:
        return self._predictor().predict(np.asarray(X, dtype=float))

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        return self._predictor().predict_proba(np.asarray(X, dtype=float))

    # -- selection outputs --------------------------------------------- #
    def coverage(self) -> float:
        return len(self.selected_patches_) / self.grid_.n_patches

    def selected_pixels(self) -> np.ndarray:
        return self.grid_.patches_to_pixels(self.selected_patches_)

    def pi_map(self) -> np.ndarray:
        """Selection-frequency heatmap in image space (None for selector='single')."""
        if self.pi_ is None:
            return self.grid_.expand(self.mask_)
        return self.grid_.expand(self.pi_)

    def stability_report(self) -> dict:
        """The go/no-go diagnostics; minimal for selector='single'."""
        if hasattr(self.selector_, "stability_report"):
            return self.selector_.stability_report()
        return {"coverage": self.coverage(), "note": "single fit: no frequency map"}

    def summary(self) -> dict:
        """Honest nested-CV numbers plus the final selection diagnostics."""
        out = self.result_.summary()
        out.update({f"final_{k}": v for k, v in self.stability_report().items()})
        out["selector"] = self.selector
        out["model"] = (self.model if isinstance(self.model, str) or self.model is None
                        else type(self.model).__name__)
        out["coverage"] = self.coverage()
        return out

    def compare_to_baseline(self,
                            baseline_per_fold_scores: ArrayLike) -> BaselineComparison:
        return self.result_.compare_to_baseline(baseline_per_fold_scores)

    # -- persistence ---------------------------------------------------- #
    def save(self, directory: str | Path) -> Path:
        """RMM-style exports: nested-CV artefacts plus the selection itself."""
        directory = Path(directory)
        self.result_.save(directory)
        sel = {
            "selector": self.selector,
            "task": self.task_,
            "patch": self.patch,
            "selected_patches": [int(p) for p in self.selected_patches_],
            "coverage": self.coverage(),
            "stability_report": self.stability_report(),
        }
        if self.pi_ is not None:
            sel["pi"] = [float(v) for v in self.pi_]
        (directory / "rpm_selection.json").write_text(json.dumps(sel, indent=2))
        np.save(directory / "rpm_pi_map.npy", self.pi_map())
        return directory


def run_pipeline(X: ArrayLike, y: ArrayLike, groups: ArrayLike | None = None,
                 **kwargs) -> RobustPixelMaker:
    """Functional equivalent of the class (family convention): fit and return."""
    return RobustPixelMaker(**kwargs).fit(X, y, groups=groups)
