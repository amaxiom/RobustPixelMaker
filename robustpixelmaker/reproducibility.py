"""Reproducibility: deterministic seeding and determinism switches.

RobustPixelMaker (RPM) mirrors RobustModelMaker's reproducibility contract: a
single integer ``random_state`` deterministically derives every downstream seed
via documented, transparent offsets. Two runs with the same data, parameters and
seed return identical selected regions, fold scores, stability frequencies and
predictions, up to the floating-point-under-parallelism caveat documented in
``config.detect_device`` (see README.md, Development).

Seed-derivation scheme (kept additive and transparent, per RMM):
    outer fold      : base + fold_idx                       (+ repeat * REPEAT_STRIDE)
    bootstrap       : base + BOOTSTRAP_OFFSET      + b_idx
    representation  : base + REPRESENTATION_OFFSET + r_idx
    inner search    : base + INNER_OFFSET          + fold_idx

The offsets are spaced so realistic loop sizes cannot collide; ``Seeds`` asserts
this at construction time rather than trusting it silently.
"""
from __future__ import annotations

import os
import random
import warnings
from dataclasses import dataclass

import numpy as np

# --- documented offsets (transparent, RMM-style) ---------------------------
BOOTSTRAP_OFFSET = 10_000
REPRESENTATION_OFFSET = 20_000
INNER_OFFSET = 30_000
ENSEMBLE_OFFSET = 40_000
REPEAT_STRIDE = 100_000


def set_global_seed(seed: int) -> None:
    """Seed every global RNG channel we can reach.

    Sets ``PYTHONHASHSEED``, the ``random`` module, NumPy's legacy global RNG,
    and, if torch is importable, torch CPU/CUDA seeds. Prefer the per-stream
    generators from :class:`Seeds` for actual work; this closes ambient channels.
    """
    os.environ["PYTHONHASHSEED"] = str(int(seed))
    random.seed(seed)
    np.random.seed(seed)
    try:  # torch is optional, graceful no-op when absent
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def enable_determinism(strict: bool = False) -> dict:
    """Turn on deterministic execution where the backend allows it.

    Returns a report of what was actually set so the caller can log it. With
    ``strict=True`` we request fully deterministic torch algorithms (may raise at
    runtime on ops without a deterministic implementation, that is intentional;
    the user asked for bit-exactness). torch absent -> report says so.
    """
    report: dict = {"torch": False, "deterministic_algorithms": False, "cudnn_deterministic": False}
    try:
        import torch
    except ImportError:
        return report
    report["torch"] = True
    try:
        torch.use_deterministic_algorithms(strict, warn_only=not strict)
        report["deterministic_algorithms"] = True
    except Exception as exc:  # pragma: no cover - backend dependent
        warnings.warn(f"could not set deterministic algorithms: {exc}")
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        report["cudnn_deterministic"] = True
    return report


@dataclass(frozen=True)
class Seeds:
    """Deterministic per-stream seed derivation from one base seed.

    All methods are pure functions of ``(base, index, repeat)`` so results never
    depend on call order or execution schedule, a requirement for resuming and
    for reproducibility under parallelism.
    """

    base: int
    n_bootstrap: int = 0
    n_representations: int = 0

    def __post_init__(self) -> None:
        # Guard the offset spacing so loops cannot alias onto one another.
        if not (0 <= self.n_bootstrap < REPRESENTATION_OFFSET - BOOTSTRAP_OFFSET):
            raise ValueError(
                f"n_bootstrap={self.n_bootstrap} would collide with representation seeds; "
                f"must be < {REPRESENTATION_OFFSET - BOOTSTRAP_OFFSET}"
            )
        if not (0 <= self.n_representations < INNER_OFFSET - REPRESENTATION_OFFSET):
            raise ValueError(
                f"n_representations={self.n_representations} would collide with inner seeds; "
                f"must be < {INNER_OFFSET - REPRESENTATION_OFFSET}"
            )

    def outer(self, fold_idx: int, repeat: int = 0) -> int:
        return self.base + repeat * REPEAT_STRIDE + fold_idx

    def inner(self, fold_idx: int, repeat: int = 0) -> int:
        return self.base + repeat * REPEAT_STRIDE + INNER_OFFSET + fold_idx

    def bootstrap(self, b_idx: int) -> int:
        return self.base + BOOTSTRAP_OFFSET + b_idx

    def representation(self, r_idx: int) -> int:
        return self.base + REPRESENTATION_OFFSET + r_idx

    def ensemble(self, task_idx: int) -> int:
        """Seed for the task_idx-th (resample x representation) fit of an ensemble.

        A dedicated stream because any ADDITIVE combination of the bootstrap and
        representation streams collides: seeds.bootstrap(b) + representation-offset(r)
        is equal for every pair with the same b + r, so, for example, resample 0 under
        lens 1 and resample 1 under lens 0 received identical noise. The task index
        enumerates the (resample, lens) grid uniquely.
        """
        limit = REPEAT_STRIDE - ENSEMBLE_OFFSET
        if not 0 <= task_idx < limit:
            raise ValueError(f"ensemble task index {task_idx} outside [0, {limit})")
        return self.base + ENSEMBLE_OFFSET + task_idx

    def rng(self, seed: int) -> np.random.Generator:
        """A fresh NumPy Generator for a derived seed (preferred over global RNG)."""
        return np.random.default_rng(seed)
