"""RobustPixelMaker (RPM).

Pixel/voxel-direct, representation-marginalised, leakage-safe feature selection for
scientific images. One of the three RobustMaker libraries, with RobustModelMaker
(tabular columns) and RobustSignalMaker (signals and spectra).

The one-class entry point is :class:`RobustPixelMaker` (or the functional
:func:`run_pipeline`); every primitive it composes stays public below for users who
want the pieces individually. Design and evidence: README.md and
benchmarks/FINDINGS.md in the repository.
"""
from __future__ import annotations

__version__ = "0.2.0"

from .config import RPMConfig, detect_device, fit_count, torch_available
from .core import RobustPixelMaker, run_pipeline
from .maskfill import (
    fabricated_edge,
    fill_blur,
    fill_mean,
    fill_zero,
    masked_conv_features,
    renormalised_convolve,
)
from .masking import SoftMaskFoldEstimator, SoftMaskSelector
from .metrics import (
    BaselineComparison,
    adjusted_jaccard,
    expected_jaccard,
    jaccard,
    mean_pairwise_jaccard,
    paired_comparison,
    score_predictions,
)
from .downstream import NAMED_MODELS, DownstreamModel, resolve_model
from .regimes import PatchGrid, ensure_finite
from .representations import (
    GaussianScale,
    PatchMean,
    PatchStats,
    RandomConvFeatures,
    default_ensemble,
)
from .selection import (
    BootstrapMaskFoldEstimator,
    BootstrapMaskSelector,
    EnsembleMaskFoldEstimator,
    RepresentationEnsembleSelector,
    complementary_pair,
    resample_indices,
)
from .synthetic import SyntheticControl, make_synthetic_images
from .nested_cv import (
    FoldEstimator,
    NestedCV,
    SimpleFoldEstimator,
    infer_task,
    make_inner_splitter,
    make_outer_splitter,
)
from .reproducibility import Seeds, enable_determinism, set_global_seed
from .results import RPMResult

__all__ = [
    "__version__",
    "RobustPixelMaker",
    "run_pipeline",
    "RPMConfig",
    "detect_device",
    "fit_count",
    "torch_available",
    "BaselineComparison",
    "jaccard",
    "adjusted_jaccard",
    "expected_jaccard",
    "mean_pairwise_jaccard",
    "paired_comparison",
    "score_predictions",
    "PatchGrid",
    "NAMED_MODELS",
    "DownstreamModel",
    "resolve_model",
    "ensure_finite",
    "fabricated_edge",
    "fill_blur",
    "fill_mean",
    "fill_zero",
    "masked_conv_features",
    "renormalised_convolve",
    "SoftMaskSelector",
    "SoftMaskFoldEstimator",
    "BootstrapMaskSelector",
    "BootstrapMaskFoldEstimator",
    "RepresentationEnsembleSelector",
    "EnsembleMaskFoldEstimator",
    "PatchMean",
    "PatchStats",
    "GaussianScale",
    "RandomConvFeatures",
    "default_ensemble",
    "resample_indices",
    "complementary_pair",
    "SyntheticControl",
    "make_synthetic_images",
    "FoldEstimator",
    "NestedCV",
    "SimpleFoldEstimator",
    "infer_task",
    "make_inner_splitter",
    "make_outer_splitter",
    "Seeds",
    "enable_determinism",
    "set_global_seed",
    "RPMResult",
]
