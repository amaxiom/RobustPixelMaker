"""Result container and serialisation.

The result object exposes the analysis directly (RMM sec.4.3 ethos): per-fold
scores, out-of-fold predictions, per-fold selected regions, selection stability,
and the final refit estimator for prediction on new data. Milestone 1 populates
placeholder selections; the fields and API are stable as later milestones add the
stability heatmap, coverage frontier and baseline verdict.
"""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .metrics import BaselineComparison, paired_comparison, rmse_from_score
from numpy.typing import ArrayLike
from typing import Any


@dataclass
class RPMResult:
    task: str
    random_state: int
    per_fold_scores: np.ndarray
    oof_predictions: np.ndarray
    oof_count: np.ndarray
    selected_per_fold: list
    selection_stability: float
    fold_records: list
    final_estimator: Any        # any fitted predictor; it exists to be called
    classes: Optional[np.ndarray] = None
    config: dict = field(default_factory=dict)

    # ---- convenience ------------------------------------------------------ #
    @property
    def mean_score(self) -> float:
        return float(np.nanmean(self.per_fold_scores))

    @property
    def std_score(self) -> float:
        return float(np.nanstd(self.per_fold_scores))

    def display_score(self) -> tuple[float, float, str]:
        """(value, sd, name) with the regression sign flipped back to RMSE."""
        if self.task == "regression":
            per = np.array([rmse_from_score(s) for s in self.per_fold_scores])
            return float(np.nanmean(per)), float(np.nanstd(per)), "RMSE"
        name = "AUC" if self.task == "binary" else "AUC-OVR"
        return self.mean_score, self.std_score, name

    def predict(self, X: ArrayLike) -> np.ndarray:
        return self.final_estimator.predict(X)

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        return self.final_estimator.predict_proba(X)

    def compare_to_baseline(self,
                            baseline_per_fold_scores: ArrayLike) -> BaselineComparison:
        """Paired verdict vs a matched full-image baseline (preserved/sig.better/worse)."""
        return paired_comparison(self.per_fold_scores, baseline_per_fold_scores)

    def summary(self) -> dict:
        val, sd, name = self.display_score()
        return {
            "task": self.task,
            "score_name": name,
            "score_mean": val,
            "score_sd": sd,
            "n_folds": int(len(self.per_fold_scores)),
            "selection_stability_jaccard": self.selection_stability,
            "mean_n_selected": float(
                np.mean([len(s) for s in self.selected_per_fold]) if self.selected_per_fold else 0.0
            ),
            "random_state": self.random_state,
            **self.config,
        }

    # ---- serialisation ---------------------------------------------------- #
    def _json_safe(self) -> dict:
        d = self.summary()
        d["per_fold_scores"] = [float(s) for s in self.per_fold_scores]
        d["fold_records"] = self.fold_records
        return d

    def save(self, directory: str | Path) -> Path:
        """Write JSON metadata, per-fold CSV, and a full pickle (RMM export style)."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "rpm_summary.json").write_text(json.dumps(self._json_safe(), indent=2))
        # per-fold CSV without a pandas hard dependency at import time
        lines = ["repeat,fold,score,n_selected,n_test,seconds"]
        for r in self.fold_records:
            lines.append(
                f"{r['repeat']},{r['fold']},{r['score']},{r['n_selected']},{r['n_test']},{r['seconds']}"
            )
        (directory / "rpm_folds.csv").write_text("\n".join(lines))
        with open(directory / "rpm_result.pkl", "wb") as fh:
            pickle.dump(self, fh)
        return directory

    @staticmethod
    def load(path) -> "RPMResult":
        with open(path, "rb") as fh:
            return pickle.load(fh)
