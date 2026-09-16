"""Configuration: defaults, device auto-detection with graceful fallback, cost model.

Design constraints (README.md, Development):
  * GPU-enabled with graceful fallback: CUDA -> MPS -> CPU; everything runs on CPU
    with identical results up to the FP-under-parallelism caveat. torch is optional.
  * User-defined where possible with robust defaults: every knob here has a sane
    default; ``registered`` is the one field the user must assert (see sec.2).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field, asdict
from typing import Optional


def torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


def detect_device(prefer: str = "auto") -> str:
    """Resolve a compute device string, degrading gracefully.

    ``prefer="auto"`` picks the best available (cuda > mps > cpu). An explicit
    preference is honoured when available, otherwise we warn and fall back to CPU
    rather than crashing, following this library's "degrade, don't crash" rule.
    """
    if not torch_available():
        if prefer not in ("auto", "cpu"):
            warnings.warn(f"torch not installed; requested device '{prefer}' -> 'cpu'")
        return "cpu"
    import torch

    has_cuda = torch.cuda.is_available()
    has_mps = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
    if prefer == "auto":
        return "cuda" if has_cuda else ("mps" if has_mps else "cpu")
    if prefer == "cuda" and has_cuda:
        return "cuda"
    if prefer == "mps" and has_mps:
        return "mps"
    if prefer == "cpu":
        return "cpu"
    warnings.warn(f"requested device '{prefer}' unavailable -> 'cpu'")
    return "cpu"


@dataclass
class RPMConfig:
    """All user-facing knobs with robust defaults.

    ``registered`` has no safe default guess (a global mask over unregistered
    images is meaningless), so it is required-by-convention: the user should set
    it explicitly. We default to True (the v1 regime) but document that it must be
    owned. Fields beyond the Milestone-1 scaffold are declared here so the config
    surface is stable as later milestones land.
    """

    # --- regime (sec.2) ---
    registered: bool = True

    # --- cross-validation ---
    k_outer: int = 5
    l_inner: int = 5
    repeated_outer_cv: int = 1
    n_iter: int = 20  # randomized hyperparameter configurations per inner CV

    # --- stability selection (later milestones) ---
    n_bootstrap: int = 20
    n_representations: int = 3
    tau: float = 0.7  # stability threshold (RMM default)
    lam: float = 1e-3  # mask sparsity strength

    # --- masking / granularity (later milestones) ---
    granularity: str = "superpixel"  # superpixel | patch | roi | pixel
    mask_gate: str = "hardconcrete"  # hardconcrete | sigmoid
    masking_op: str = "renorm"  # renorm | baseline_fill

    # --- execution ---
    device: str = "auto"
    n_jobs: int = -1
    strict_determinism: bool = False
    random_state: int = 0

    def resolved_device(self) -> str:
        return detect_device(self.device)

    def to_dict(self) -> dict:
        return asdict(self)


def fit_count(
    k_outer: int,
    repeated_outer_cv: int,
    n_representations: int,
    n_bootstrap: int,
    n_iter: int,
    l_inner: int,
) -> int:
    """Total model-fit count (extends RMM Eq.3 with the representation axis).

    F = K * R_out * (R_rep * B + n_iter * L + 1)  +  (R_rep * B + n_iter * L + 1)

    The trailing term is the final selection + final inner search + final refit on
    all training data. Use this to size runs before launching.
    """
    per = n_representations * n_bootstrap + n_iter * l_inner + 1
    return k_outer * repeated_outer_cv * per + per
