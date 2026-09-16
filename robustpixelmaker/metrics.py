"""Scoring, baseline comparison, and set-stability metrics.

Score convention (RMM sec.4.5): regression scores are stored as *negative* RMSE so
that "higher is better" holds uniformly; ``rmse_from_score`` converts back for display.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from typing import Any, Iterable, Sequence
from numpy.typing import ArrayLike


def score_predictions(task: str, y_true: ArrayLike, y_pred: ArrayLike | None = None,
                      y_proba: ArrayLike | None = None) -> float:
    """Task-appropriate score, higher-is-better.

    binary       -> ROC-AUC (needs positive-class probability)
    multiclass   -> weighted one-vs-rest ROC-AUC
    regression   -> negative RMSE

    Returns ``nan`` if a classification metric is undefined for this fold (e.g. a
    class missing from ``y_true``); the engine records it rather than crashing.
    """
    from sklearn.metrics import roc_auc_score, mean_squared_error

    y_true = np.asarray(y_true)
    if task == "regression":
        rmse = float(np.sqrt(mean_squared_error(y_true, np.asarray(y_pred))))
        return -rmse
    try:
        if task == "binary":
            proba = np.asarray(y_proba)
            pos = proba[:, 1] if proba.ndim == 2 else proba
            return float(roc_auc_score(y_true, pos))
        if task == "multiclass":
            return float(
                roc_auc_score(y_true, np.asarray(y_proba), multi_class="ovr", average="weighted")
            )
    except ValueError:
        return float("nan")
    raise ValueError(f"unknown task: {task!r}")


def rmse_from_score(score: float) -> float:
    """Convert a stored (negative-RMSE) regression score back to positive RMSE."""
    return -score


def jaccard(set_a: Iterable[int], set_b: Iterable[int]) -> float:
    """Jaccard similarity of two index/feature sets. J(empty, empty) := 1.0."""
    a, b = set(map(int, set_a)), set(map(int, set_b))
    if not a and not b:
        return 1.0
    union = len(a | b)
    return len(a & b) / union if union else 1.0


def expected_jaccard(size_a: int, size_b: int, n_total: int) -> float:
    """Jaccard expected by chance for two random subsets of the given sizes."""
    if n_total <= 0 or (size_a == 0 and size_b == 0):
        return 1.0
    inter = size_a * size_b / n_total
    union = size_a + size_b - inter
    return inter / union if union > 0 else 1.0


def adjusted_jaccard(set_a: Iterable[int], set_b: Iterable[int],
                     n_total: int) -> float:
    """Jaccard corrected for chance agreement, given the universe size ``n_total``.

    Raw Jaccard is strongly confounded with how much each method selects: two sets that
    each cover most of the universe overlap heavily no matter what, so a selector that
    barely selects anything scores as highly "stable". In the benchmarks this confound
    was nearly total on one dataset, where stability correlated with coverage at
    r = +0.99 and the apparently most stable configuration was simply the one retaining
    73% of the image.

    This rescales against the chance baseline in the manner of a corrected-for-chance
    index, so 0 means no better than random sets of the same sizes and 1 means
    identical. Negative values (worse than chance) are possible and are not clipped,
    because they are informative.
    """
    a, b = set(map(int, set_a)), set(map(int, set_b))
    obs = jaccard(a, b)
    exp = expected_jaccard(len(a), len(b), n_total)
    if exp >= 1.0:
        return 1.0
    return (obs - exp) / (1.0 - exp)


def mean_pairwise_jaccard(sets: Sequence[Iterable[int]], min_size: int = 1,
                          n_total: int | None = None) -> float:
    """Mean pairwise Jaccard across a list of selected sets (RMM Eq.4).

    This is the selection-stability metric; for RPM's global regime the "sets" are
    selected patch/voxel indices per outer fold. Undefined (nan) for fewer than two
    sets.

    ``min_size`` guards a reporting hazard observed in the benchmarks: the Jaccard of
    two empty sets is 1, so a selector that collapses and selects NOTHING scores a
    perfect stability of 1.00. A degenerate non-selection then looks like the most
    reproducible result in the table. Any fold selecting fewer than ``min_size``
    patches therefore makes the statistic undefined (nan) rather than perfect. Always
    report coverage alongside stability so the two cannot be confused.

    ``n_total`` (the number of candidate patches) switches to the chance-corrected
    index of :func:`adjusted_jaccard`, which removes the coverage confound: without it,
    raw Jaccard rewards selectors that simply keep more. Prefer it whenever comparing
    configurations that retain different amounts.
    """
    sets = list(sets)
    n = len(sets)
    if n < 2:
        return float("nan")
    if any(len(set(map(int, s))) < min_size for s in sets):
        return float("nan")
    if n_total is None:
        vals = [jaccard(sets[i], sets[j]) for i in range(n) for j in range(i + 1, n)]
    else:
        vals = [adjusted_jaccard(sets[i], sets[j], n_total)
                for i in range(n) for j in range(i + 1, n)]
    return float(np.mean(vals))


@dataclass
class BaselineComparison:
    """Paired comparison of per-fold scores vs a baseline (RMM sec.6.1, compact)."""

    p_wilcoxon: float
    p_ttest: float
    mean_delta: float
    outcome: str  # preserved | sig.better | sig.worse

    def to_dict(self) -> dict:
        return {
            "p_wilcoxon": self.p_wilcoxon,
            "p_ttest": self.p_ttest,
            "mean_delta": self.mean_delta,
            "outcome": self.outcome,
        }


def paired_comparison(scores: ArrayLike, baseline_scores: ArrayLike,
                      alpha: float = 0.05) -> BaselineComparison:
    """Wilcoxon signed-rank (primary) + paired t-test of per-fold scores vs baseline.

    Outcome label follows the rank test (RMM's primary), with delta = method - baseline
    (higher-is-better convention). Note the RMM caveat: at K=5 folds the two-sided
    Wilcoxon floor is 0.0625, so ``preserved`` can hide a real per-fold advantage --
    the t-test p and mean_delta are reported alongside so the caller can see it.
    """
    from scipy.stats import wilcoxon, ttest_rel

    s = np.asarray(scores, dtype=float)
    b = np.asarray(baseline_scores, dtype=float)
    delta = s - b
    mean_delta = float(np.mean(delta))

    if np.allclose(delta, 0.0):
        return BaselineComparison(1.0, 1.0, 0.0, "preserved")
    try:
        p_w = float(wilcoxon(s, b).pvalue)
    except ValueError:
        p_w = float("nan")
    try:
        p_t = float(ttest_rel(s, b).pvalue)
    except ValueError:
        p_t = float("nan")

    if not np.isnan(p_w) and p_w < alpha:
        outcome = "sig.better" if mean_delta > 0 else "sig.worse"
    else:
        outcome = "preserved"
    return BaselineComparison(p_w, p_t, mean_delta, outcome)
