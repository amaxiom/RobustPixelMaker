"""Bootstrap stability selection over the patch mask (Milestone 4).

This is the piece that makes RobustPixelMaker a stability-selection method rather than
a single-run mask learner. A single mask fit is a function of the particular training
sample: on small datasets it moves substantially under resampling, which the benchmarks
showed directly (selection stability falling to 0.29 on an n=780 ultrasound set as
sparsity rose). Meinshausen and Buehlmann's construction replaces the variance of one
selection with the variance of a frequency estimate:

    pi_k = (1 / B) * sum_b  I[ patch k selected on resample b ]
    S_hat = { k : pi_k >= tau }

A patch that only survives because of the particular sample gets a low pi and drops
out; a patch carrying real signal recurs across resamples and survives. Noise does not
reproduce, so thresholding pi is the mechanism that separates signal from noise, which
is exactly the property the noisy-image regime needs.

Three resampling schemes are provided. ``bootstrap`` (draw n with replacement) matches
the parent tabular framework. ``half`` (draw n/2 without replacement) is the original
Meinshausen-Buehlmann subsampling. ``complementary`` implements the Shah and Samworth
complementary-pairs variant, in which each pair contributes two disjoint halves; it
gives tighter error control per fit, which matters here because every fit is expensive.

Classification resampling is stratified so that a small or imbalanced dataset cannot
lose a class in a resample, which would otherwise make individual fits meaningless.

The outputs are the stability heatmap ``pi_`` (per patch, and via ``pi_map()`` in image
space), the stable region ``selected_patches_``, and a predictor refitted on that
region. Everything is fitted on the data handed to ``fit`` only, so the leakage-safe
contract is unchanged when this runs inside the nested cross-validation engine.
"""
from __future__ import annotations

import numpy as np

from .masking import SoftMaskSelector
from .downstream import attach
from .regimes import PatchGrid, ensure_finite
from .reproducibility import Seeds
from numpy.typing import ArrayLike


def resample_indices(n: int, mode: str, rng: np.random.Generator,
                     y: ArrayLike | None = None) -> np.ndarray:
    """Indices for one resample. Stratified when ``y`` is supplied (classification)."""
    if mode == "bootstrap":
        if y is None:
            return rng.integers(0, n, size=n)
        out = []
        for cls in np.unique(y):
            idx = np.where(y == cls)[0]
            out.append(rng.choice(idx, size=len(idx), replace=True))
        return np.concatenate(out)
    if mode == "half":
        if y is None:
            return rng.permutation(n)[: max(1, n // 2)]
        out = []
        for cls in np.unique(y):
            idx = rng.permutation(np.where(y == cls)[0])
            out.append(idx[: max(1, len(idx) // 2)])
        return np.concatenate(out)
    raise ValueError(f"unknown resample mode {mode!r}")


def complementary_pair(n: int, rng: np.random.Generator,
                       y: ArrayLike | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Two disjoint halves covering the sample (Shah and Samworth complementary pairs)."""
    if y is None:
        perm = rng.permutation(n)
        cut = n // 2
        return perm[:cut], perm[cut:]
    a, b = [], []
    for cls in np.unique(y):
        idx = rng.permutation(np.where(y == cls)[0])
        cut = len(idx) // 2
        a.append(idx[:cut])
        b.append(idx[cut:])
    return np.concatenate(a), np.concatenate(b)


class BootstrapMaskSelector:
    """Stability selection over bootstrap resamples of a learned patch mask.

    Parameters mirror :class:`~robustpixelmaker.masking.SoftMaskSelector` for the inner
    mask fits, plus the aggregation controls ``n_bootstrap``, ``tau``, ``subsample`` and
    ``target_coverage``.

    ``target_coverage`` addresses a failure the benchmarks made unavoidable: a single
    fixed sparsity strength under-pruned on abdominal CT (88% of patches retained) and
    over-pruned on breast ultrasound at the very same value. Because the aggregation
    already produces a full frequency map, an operating point can be chosen on that map
    for free, by picking the threshold whose retained fraction is closest to the target,
    with no refitting. When it is None the fixed ``tau`` is used, matching the parent
    framework's per-dataset threshold.
    """

    def __init__(self, task: str = "binary", patch: int = 4, lam: float = 0.03,
                 tau: float = 0.7, n_bootstrap: int = 20, subsample: str = "bootstrap",
                 target_coverage: float | None = None, gate: str = "hardconcrete",
                 tv: float = 0.0, l2: float = 1e-3, lr: float = 0.08, n_iter: int = 500,
                 warmup_frac: float = 0.3, mask_threshold: float = 0.5,
                 head: str = "linear", hidden: int = 16,
                 lam_scale: str = "fixed", auto_target: float = 0.15,
                 n_jobs: int = 1, seed: int = 0):
        if subsample not in ("bootstrap", "half", "complementary"):
            raise ValueError("subsample must be one of {'bootstrap','half','complementary'}")
        if not 0.0 < tau <= 1.0:
            raise ValueError("tau must be in (0, 1]")
        # validated HERE as well as in SoftMaskSelector, because _one_fit deliberately
        # swallows resample-fit exceptions: without this, a mistyped head or an
        # out-of-range auto_target would be reported as "every one of the N resample
        # fits failed", which points at the data rather than at the argument.
        if head not in ("linear", "mlp"):
            raise ValueError("head must be one of {'linear','mlp'}")
        if lam_scale not in ("fixed", "auto"):
            raise ValueError("lam_scale must be one of {'fixed','auto'}")
        if not 0.0 < float(auto_target) < 1.0:
            raise ValueError("auto_target is the fraction of the mask to leave open "
                             "and must be in (0, 1)")
        self.task = task
        self.patch = patch
        self.lam = lam
        self.tau = tau
        self.n_bootstrap = n_bootstrap
        self.subsample = subsample
        self.target_coverage = target_coverage
        self.gate = gate
        self.tv = tv
        self.l2 = l2
        self.lr = lr
        self.n_iter = n_iter
        self.warmup_frac = warmup_frac
        self.mask_threshold = mask_threshold
        self.head = head
        self.hidden = hidden
        self.lam_scale = lam_scale
        # the fraction of the mask lam_scale="auto" aims to leave open. It is a PRIOR
        # on coverage, not a neutral default, and it is consequential: at the old 0.4
        # default, against a true informative fraction of 0.062, it cost recovery
        # precision 16.0x chance to 4.7x on the linear rule (sec.19) and halved
        # recovery F1 on the synthetic control, 1.000 to 0.441 (sec.21). The default
        # is now 0.15, which recovers the control exactly while still closing a mask
        # that a fixed lam leaves wide open. Note the dial is COARSE: the lam ladder
        # has six geometric rungs, so 0.25 and 0.40 land on the same rung and give
        # identical selections. It picks a rung, not a coverage.
        self.auto_target = auto_target
        self.n_jobs = n_jobs
        self.seed = seed
        self._lam_calibrated_ = None

    def _resolved_lam(self):
        """The lam and lam_scale handed to the inner fits, after any calibration."""
        if self.lam_scale == "auto" and self._lam_calibrated_ is not None:
            return self._lam_calibrated_, "fixed"
        return self.lam, self.lam_scale

    def _calibrate_lam(self, X, y):
        """Choose lam ONCE, on the training data, and hold it fixed across resamples.

        ``lam_scale="auto"`` searches a geometric ladder for the penalty whose mask
        lands near ``auto_target`` open, which is what makes the MLP head usable at
        all: at a fixed lam an expressive head keeps every gate open and the mask never
        closes (FINDINGS sec.17). Running that search INSIDE each resample, which is
        what forwarding ``lam_scale`` through ``_inner_kwargs`` used to do, breaks the
        aggregation it feeds. Stability selection counts how often a patch survives at
        a FIXED operating point; if lam is re-chosen per resample then the frequency
        map confounds "which patches matter" with "which lam this resample happened to
        land on", and pi_ no longer measures what it reports.

        The cost was measured, not guessed. On the matched-coverage protocol the
        per-resample search lost 0.16 mean pairwise Jaccard (0.536 to 0.375) and went
        from losing stability to random-forest importance in 16 cells of 20 to losing
        in 20 of 20, while scores did not improve. It was also 5.5x slower, since each
        of B resamples repeated the whole ladder. Calibrating once costs one extra fit
        and removes B ladders. FINDINGS sec.18.

        Calibration sees only the data passed to ``fit``, which under nested CV is the
        outer training split, so this adds no leakage: it is a training-set choice of
        hyperparameter, refitted inside every fold like the rest of the selection.
        """
        self._lam_calibrated_ = None
        if self.lam_scale == "auto":
            probe = SoftMaskSelector(seed=int(self.seed),
                                     **self._inner_kwargs()).fit(X, y)
            self._lam_calibrated_ = float(probe.lam_used_)
        self.lam_used_ = self._resolved_lam()[0]

    def _inner_kwargs(self):
        lam, lam_scale = self._resolved_lam()
        return dict(task=self.task, patch=self.patch, lam=lam, l2=self.l2,
                    lr=self.lr, n_iter=self.n_iter, warmup_frac=self.warmup_frac,
                    mask_threshold=self.mask_threshold, gate=self.gate, tv=self.tv,
                    head=self.head, hidden=self.hidden, lam_scale=lam_scale,
                    auto_target=self.auto_target)

    def _one_fit(self, X, y, idx, seed, n_patches):
        """Selection indicator vector for a single resample, or None on failure.

        A degenerate resample (for example a class that vanished) must not abort the
        whole aggregation, but failures must not be INVISIBLE either: an early version
        returned a zero vector here, which meant a systematic error in the inner fit
        would silently produce an all-zero frequency map that looked like a legitimate
        "nothing is stable" verdict. Failures are now counted, reported in
        ``stability_report``, and fatal when every fit fails.
        """
        try:
            sel = SoftMaskSelector(seed=int(seed), **self._inner_kwargs()).fit(X[idx], y[idx])
        except Exception:
            return None
        hit = np.zeros(n_patches)
        hit[sel.selected_patches_] = 1.0
        return np.stack([hit, np.asarray(sel.mask_, dtype=float)])

    def _resamples(self, n, y_strat, seeds):
        """(indices, seed) pairs, deterministic and independent of execution order.

        Complementary pairs come two at a time, so an odd ``n_bootstrap`` yields
        ``2 * (n_bootstrap // 2)`` fits (never fewer than 2); the realised count is
        always visible as ``n_fits_`` in ``stability_report`` rather than silent.
        """
        jobs = []
        if self.subsample == "complementary":
            for p in range(max(1, self.n_bootstrap // 2)):
                rng = np.random.default_rng(seeds.bootstrap(p))
                a, b = complementary_pair(n, rng, y_strat)
                jobs.append((a, seeds.bootstrap(2 * p)))
                jobs.append((b, seeds.bootstrap(2 * p + 1)))
        else:
            for b in range(self.n_bootstrap):
                rng = np.random.default_rng(seeds.bootstrap(b))
                jobs.append((resample_indices(n, self.subsample, rng, y_strat),
                             seeds.bootstrap(b)))
        return jobs

    def _record_coverage_target(self, n_patches):
        """Say whether ``target_coverage`` was reached, or only approached.

        ``pi_`` takes at most B+1 distinct values, so achievable coverages are
        quantised, and heavy ties make the quantisation coarse. On blood at B=12,
        24.5% of patches sit at pi_ exactly 1.0, so NO threshold yields 0.12 and the
        lowest reachable coverage is 0.245: a request for 0.12 silently returned
        double it. `examples/ex4` asked for matched coverage on two datasets, got
        0.245 against 0.122, and reported a shape comparison as matched when it was
        not (FINDINGS sec.22). A request that degrades into a confident-looking answer
        is the failure this reports rather than leaves to be inferred.

        ``coverage_target_met_`` is None when no target was asked for.
        """
        self.coverage_realised_ = (len(self.selected_patches_) / n_patches
                                   if n_patches else 0.0)
        if self.target_coverage is None:
            self.coverage_target_met_ = None
            return
        gap = abs(self.coverage_realised_ - self.target_coverage)
        # 25% relative separates "quantisation rounded it" from "not reachable at
        # all"; the blood case above misses by 104%
        self.coverage_target_met_ = bool(gap <= 0.25 * self.target_coverage)
        if not self.coverage_target_met_:
            import warnings
            warnings.warn(
                f"target_coverage={self.target_coverage:.3f} was not reachable: the "
                f"selection covers {self.coverage_realised_:.3f}. pi_ takes only "
                f"{len(np.unique(self.pi_))} distinct values at n_bootstrap="
                f"{self.n_bootstrap}, so coverage is quantised and ties make it "
                f"coarse. Raise n_bootstrap for a finer grid, or use top_patches(k) "
                f"to force exactly k patches.", UserWarning, stacklevel=3)

    def _choose_tau(self, pi) -> float:
        """Fixed tau, or the threshold whose retained fraction is nearest the target.

        When no patch was ever selected (pi identically zero) every positive threshold
        is equivalent, so the fixed ``tau`` is returned; an earlier version guarded a
        branch that could never trigger, because the 1/B floor candidate is always
        positive, and would have reported a misleading tiny threshold instead.
        """
        if self.target_coverage is None:
            return self.tau
        positive = pi[pi > 0]
        if positive.size == 0:
            return self.tau
        candidates = np.unique(np.concatenate([positive, [1.0 / max(1, self.n_bootstrap)]]))
        cov = np.array([(pi >= t).mean() for t in candidates])
        return float(candidates[int(np.argmin(np.abs(cov - self.target_coverage)))])

    def fit(self, X: ArrayLike, y: ArrayLike) -> BootstrapMaskSelector:
        X = ensure_finite(X, "X")
        y = np.asarray(y)
        if self.task == "regression":
            y = ensure_finite(y, "y")
        self.grid = PatchGrid.from_image_shape(X.shape, patch=self.patch)
        P = self.grid.n_patches
        n = X.shape[0]
        seeds = Seeds(base=self.seed, n_bootstrap=max(self.n_bootstrap, 1))
        y_strat = None if self.task == "regression" else y
        self._calibrate_lam(X, y)

        jobs = self._resamples(n, y_strat, seeds)
        if self.n_jobs == 1:
            raw = [self._one_fit(X, y, idx, s, P) for idx, s in jobs]
        else:
            from joblib import Parallel, delayed
            raw = Parallel(n_jobs=self.n_jobs)(
                delayed(self._one_fit)(X, y, idx, s, P) for idx, s in jobs)

        hits = [h for h in raw if h is not None]
        self.n_failed_ = len(raw) - len(hits)
        if raw and not hits:
            raise RuntimeError(
                f"every one of the {len(raw)} resample fits failed; this indicates a "
                "systematic error in the inner mask fit, not degenerate resamples")
        self.n_fits_ = len(hits)
        self.pi_ = np.mean([h[0] for h in hits], axis=0) if hits else np.zeros(P)
        self.strength_ = (np.mean([h[1] for h in hits], axis=0) if hits
                          else np.zeros(P))
        self.tau_ = self._choose_tau(self.pi_)
        self.selected_patches_ = np.where(self.pi_ >= self.tau_)[0].astype(int)
        self._record_coverage_target(P)

        # Refit the predictor against the stable region. If aggregation retained
        # nothing, fall back to the highest-frequency patch so the model is still
        # usable and the degenerate case is visible in `coverage()` rather than
        # crashing at predict time.
        if self.selected_patches_.size == 0 and P:
            self.selected_patches_ = np.array([int(np.argmax(self.pi_))])
            self.degenerate_ = True
        else:
            self.degenerate_ = False
        binary = np.zeros(P)
        binary[self.selected_patches_] = 1.0
        self.mask_ = binary
        self.model_ = SoftMaskSelector(seed=int(self.seed), fixed_mask=binary,
                                       **self._inner_kwargs()).fit(X, y)
        self.coef_ = self.model_.coef_
        self.intercept_ = self.model_.intercept_
        return self

    # -- prediction --------------------------------------------------------- #
    @property
    def classes_(self):
        """Classes seen at fit time (None for regression), for proba alignment."""
        return getattr(self.model_, "classes_", None)

    def predict(self, X: ArrayLike) -> np.ndarray:
        return self.model_.predict(X)

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        return self.model_.predict_proba(X)

    # -- selection outputs -------------------------------------------------- #
    def ranking(self) -> np.ndarray:
        """Patch indices best first, with frequency ties broken by gate strength.

        ``pi_`` is a count over B resamples, so it takes at most B+1 distinct values and
        saturates: on a 49-patch grid with B=10 it is common for seven patches to sit at
        pi=1.0 with none above them. Asking that map for a top-2 selection then picks two
        of the seven arbitrarily, by index order, which is exactly the collapse the
        matched-coverage benchmark exposed at 5% coverage.

        ``strength_`` records the mean gate VALUE over the same fits, a continuous
        quantity, so it separates patches the counter cannot. Using it only to order
        within a tie is a strict refinement: the frequency ranking is preserved wherever
        it actually discriminates, and arbitrary index order is replaced by evidence
        everywhere else.
        """
        strength = getattr(self, "strength_", None)
        if strength is None:
            strength = np.zeros_like(self.pi_)
        return np.lexsort((-strength, -self.pi_)).astype(int)

    def top_patches(self, k: int) -> np.ndarray:
        """The k best patches under :meth:`ranking`, sorted by index."""
        k = int(max(0, min(k, len(self.pi_))))
        return np.sort(self.ranking()[:k]).astype(int)

    def pi_map(self) -> np.ndarray:
        """Per-patch selection frequency expanded to image space (stability heatmap)."""
        return self.grid.expand(self.pi_)

    def selected_pixels(self) -> np.ndarray:
        return self.grid.patches_to_pixels(self.selected_patches_)

    def coverage(self) -> float:
        return len(self.selected_patches_) / self.grid.n_patches

    def ambiguous_fraction(self, lo: float = 0.2, hi: float = 0.8) -> float:
        """Fraction of patches whose selection frequency sits in the undecided band.

        This is the go/no-go diagnostic for whether a reproducible region exists at
        all. A bimodal ``pi_`` (most patches near 0 or near 1, small ambiguous
        fraction) is the structural signature stability selection is meant to expose:
        some patches recur across resamples, the rest do not, and a threshold cleanly
        separates them. A diffuse ``pi_`` concentrated in the middle band means no
        subset reproducibly carries the signal, so no threshold can yield a stable
        selection and aggregation cannot manufacture one.

        Observed in the benchmarks: blood cell microscopy at n=3000 gives 0.16
        (bimodal, selection is reproducible), whereas breast ultrasound at n=780 gives
        0.59 (diffuse, and measured cross-fold stability was correspondingly poor at
        about 0.3 regardless of aggregation). Read this before trusting a selected
        region as a scientific finding.
        """
        return float(np.mean((self.pi_ >= lo) & (self.pi_ < hi)))

    def stability_report(self) -> dict:
        """Compact summary of the frequency map, for logging beside any result."""
        return {
            "n_fits": self.n_fits_,
            "n_failed": getattr(self, "n_failed_", 0),
            "tau": self.tau_,
            "coverage": self.coverage(),
            "pi_mean": float(self.pi_.mean()),
            "pi_max": float(self.pi_.max()),
            "frac_confident_in": float(np.mean(self.pi_ >= 0.8)),
            "frac_confident_out": float(np.mean(self.pi_ < 0.2)),
            "ambiguous_fraction": self.ambiguous_fraction(),
            "degenerate": self.degenerate_,
            # None when no target was requested; False means the request was not
            # reachable and the coverage above is NOT what was asked for
            "coverage_target_met": self.coverage_target_met_,
        }


class RepresentationEnsembleSelector(BootstrapMaskSelector):
    """Double bootstrap: marginalise selection over data resamples AND representations.

    This is the method's central claim (RELATED_WORK.md sec.12). The selection frequency is
    aggregated over the product of resamples and representations,

        pi_k = (1 / (R * B)) * sum_r sum_b  I[ patch k selected by lens r on resample b ]

    so a patch survives only if it is chosen regardless of which lens is looking. The
    representation is thereby treated as a nuisance variable and integrated out, rather
    than fixed and defended.

    Beyond the aggregate, the per-representation frequency maps are kept, because the
    disagreement between them is itself the quantity the claim rests on.
    ``representation_agreement()`` measures how much the selected region depends on the
    lens: if it is already high, marginalising buys little and the honest report is
    that a single representation would have sufficed; if it is low, a single-lens
    result was lens-specific and the marginalised one is the defensible finding.
    """

    def __init__(self, representations=None, n_representations: int = 4, **kwargs):
        super().__init__(**kwargs)
        if representations is None:
            from .representations import default_ensemble
            representations = default_ensemble(n_representations, seed=self.seed)
        self.representations = list(representations)
        if not self.representations:
            raise ValueError("need at least one representation")

    def _one_fit_rep(self, X, y, idx, seed, n_patches, rep):
        """Indicator vector for one (resample, lens) fit, or None on failure."""
        try:
            sel = SoftMaskSelector(seed=int(seed), representation=rep,
                                   **self._inner_kwargs()).fit(X[idx], y[idx])
        except Exception:
            return None
        hit = np.zeros(n_patches)
        hit[sel.selected_patches_] = 1.0
        return hit

    def fit(self, X: ArrayLike, y: ArrayLike) -> RepresentationEnsembleSelector:
        X = ensure_finite(X, "X")
        y = np.asarray(y)
        if self.task == "regression":
            y = ensure_finite(y, "y")
        self.grid = PatchGrid.from_image_shape(X.shape, patch=self.patch)
        P = self.grid.n_patches
        n = X.shape[0]
        seeds = Seeds(base=self.seed, n_bootstrap=max(self.n_bootstrap, 1),
                      n_representations=max(len(self.representations), 1))
        y_strat = None if self.task == "regression" else y
        # one lam for the whole ensemble, so a lens difference stays a lens difference
        # rather than a lens-plus-operating-point difference
        self._calibrate_lam(X, y)
        jobs = self._resamples(n, y_strat, seeds)

        # every (resample, lens) fit gets a unique seed from the dedicated ensemble
        # stream; an earlier additive scheme collided for pairs with equal b + r
        tasks = [(r_idx, rep, idx, seeds.ensemble(r_idx * len(jobs) + j_idx))
                 for r_idx, rep in enumerate(self.representations)
                 for j_idx, (idx, s) in enumerate(jobs)]
        if self.n_jobs == 1:
            raw = [self._one_fit_rep(X, y, idx, s, P, rep) for _, rep, idx, s in tasks]
        else:
            from joblib import Parallel, delayed
            raw = Parallel(n_jobs=self.n_jobs)(
                delayed(self._one_fit_rep)(X, y, idx, s, P, rep) for _, rep, idx, s in tasks)

        hits = [h for h in raw if h is not None]
        self.n_failed_ = len(raw) - len(hits)
        if raw and not hits:
            raise RuntimeError(
                f"every one of the {len(raw)} (resample x representation) fits failed; "
                "this indicates a systematic error in the inner mask fit")

        self.n_fits_ = len(hits)
        self.pi_ = np.mean(hits, axis=0) if hits else np.zeros(P)
        self.tau_ = self._choose_tau(self.pi_)
        self.selected_patches_ = np.where(self.pi_ >= self.tau_)[0].astype(int)
        self._record_coverage_target(P)

        # Per-representation frequency maps, thresholded at the OPERATING tau_. An
        # earlier version thresholded them at the constructor tau, so with
        # target_coverage in play representation_agreement() was measured at a
        # different operating point from the selection it was reported beside.
        self.pi_by_representation_ = {}
        per_rep_sets = []
        for r_idx, rep in enumerate(self.representations):
            rows = [h for (ri, _, _, _), h in zip(tasks, raw) if ri == r_idx and h is not None]
            pi_r = np.mean(rows, axis=0) if rows else np.zeros(P)
            self.pi_by_representation_[getattr(rep, "name", f"rep{r_idx}")] = pi_r
            per_rep_sets.append(np.where(pi_r >= self.tau_)[0].astype(int))
        self._per_rep_sets = per_rep_sets

        if self.selected_patches_.size == 0 and P:
            self.selected_patches_ = np.array([int(np.argmax(self.pi_))])
            self.degenerate_ = True
        else:
            self.degenerate_ = False
        binary = np.zeros(P)
        binary[self.selected_patches_] = 1.0
        self.mask_ = binary
        # The final predictor uses the plain patch-mean lens: the ensemble decided
        # WHICH region to keep, and the deliverable should not inherit one member's
        # feature space.
        self.model_ = SoftMaskSelector(seed=int(self.seed), fixed_mask=binary,
                                       **self._inner_kwargs()).fit(X, y)
        self.coef_ = self.model_.coef_
        self.intercept_ = self.model_.intercept_
        return self

    def representation_agreement(self) -> float:
        """Mean pairwise Jaccard between the regions chosen by each single lens.

        Low agreement means a single-representation result would have been an artefact
        of that lens, which is precisely the case marginalisation is needed for.
        """
        from .metrics import mean_pairwise_jaccard

        return mean_pairwise_jaccard(self._per_rep_sets)

    def stability_report(self) -> dict:
        rep = super().stability_report()
        rep["n_representations"] = len(self.representations)
        rep["representation_agreement"] = self.representation_agreement()
        return rep


class BootstrapMaskFoldEstimator:
    """FoldEstimator adapter for :class:`BootstrapMaskSelector` (plugs into NestedCV).

    The whole aggregation, including the threshold choice and the final refit, happens
    inside ``fit`` on the training partition, so the held-out fold informs nothing.
    """

    def __init__(self, patch: int = 4, lam: float = 0.03, tau: float = 0.7,
                 n_bootstrap: int = 20, subsample: str = "bootstrap",
                 target_coverage: float | None = None, gate: str = "hardconcrete",
                 tv: float = 0.0, n_iter: int = 500, n_jobs: int = 1, model=None,
                 head: str = "linear", hidden: int = 16,
                 lam_scale: str = "fixed", auto_target: float = 0.15):
        self.kw = dict(patch=patch, lam=lam, tau=tau, n_bootstrap=n_bootstrap,
                       subsample=subsample, target_coverage=target_coverage,
                       gate=gate, tv=tv, n_iter=n_iter, n_jobs=n_jobs,
                       head=head, hidden=hidden, lam_scale=lam_scale,
                       auto_target=auto_target)
        self.model = model
        self.selected_ = np.array([], dtype=int)

    def fit(self, X, y, *, task, seeds, fold_idx, repeat, inner_splitter, groups=None):
        seed = 0 if seeds is None else seeds.outer(fold_idx, repeat)
        self.selector_ = BootstrapMaskSelector(task=task, seed=int(seed), **self.kw).fit(X, y)
        self.selected_ = self.selector_.selected_patches_
        self.pi_ = self.selector_.pi_
        # fitted on the TRAIN partition only, like everything else in this method
        self.downstream_ = attach(self.model, self.selector_, X, y, task, int(seed))
        self.classes_ = (self.downstream_.classes_ if self.downstream_ is not None
                         else self.selector_.classes_)
        return self

    def _predictor(self):
        """The downstream model when one was requested, else the mask's own head."""
        return self.selector_ if self.downstream_ is None else self.downstream_

    def predict(self, X: ArrayLike) -> np.ndarray:
        return self._predictor().predict(X)

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        return self._predictor().predict_proba(X)


class EnsembleMaskFoldEstimator(BootstrapMaskFoldEstimator):
    """FoldEstimator adapter for :class:`RepresentationEnsembleSelector`."""

    def __init__(self, n_representations: int = 4, **kwargs):
        super().__init__(**kwargs)
        self.n_representations = n_representations

    def fit(self, X, y, *, task, seeds, fold_idx, repeat, inner_splitter, groups=None):
        seed = 0 if seeds is None else seeds.outer(fold_idx, repeat)
        self.selector_ = RepresentationEnsembleSelector(
            task=task, seed=int(seed), n_representations=self.n_representations, **self.kw
        ).fit(X, y)
        self.selected_ = self.selector_.selected_patches_
        self.pi_ = self.selector_.pi_
        self.downstream_ = attach(self.model, self.selector_, X, y, task, int(seed))
        self.classes_ = (self.downstream_.classes_ if self.downstream_ is not None
                         else self.selector_.classes_)
        return self
