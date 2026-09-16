"""Guards on how the benchmark suite REPORTS, not on what it computes.

A selection benchmark has a standing hazard: every headline column flatters an arm
that declines to select. Keeping the whole image scores the full-image score, and it
scores 1.000 on stability because keeping everything is perfectly reproducible. An
arm that does nothing therefore appears at or near the top of both columns at once.

This bit the shipped results. The `rpm_mlp` arm returned coverage 1.000 on both
multiclass datasets, and read down the score column it looked tied for best on blood
(FINDINGS sec.17, sec.20). These tests pin the reporting rule that makes that
visible, because the rule is worth more than any single arm's configuration.
"""
import pandas as pd
import pytest

from benchmarks.competitors import (NO_SELECTION, NO_SELECTION_NOTE, SELECTORS,
                                    mark_non_selectors)


def test_an_arm_that_kept_everything_is_marked_as_not_selecting():
    frame = pd.DataFrame({"score": [0.954, 0.954, 0.955],
                          "coverage": [1.0, 0.384, 0.9995]},
                         index=["rpm_mlp", "rpm_boot", "rfe"])
    out = mark_non_selectors(frame)
    assert list(out["selects"]) == ["no", "yes", "no"]


def test_the_guard_does_not_mutate_the_frame_it_was_given():
    """The summary writers reuse these frames for figures, so a hidden column added
    in place would reach a plot legend without anyone asking for it."""
    frame = pd.DataFrame({"coverage": [1.0]}, index=["x"])
    mark_non_selectors(frame)
    assert "selects" not in frame.columns


def test_the_threshold_is_tolerant_of_float_noise_but_not_of_real_selection():
    """coverage is a mean over folds, so an arm that kept everything can land a
    hair under 1.0. An arm that dropped even a few percent is genuinely selecting."""
    assert NO_SELECTION < 1.0
    frame = pd.DataFrame({"coverage": [0.9999, 0.97]}, index=["all", "most"])
    assert list(mark_non_selectors(frame)["selects"]) == ["no", "yes"]


def test_both_summary_writers_carry_the_explanatory_note():
    """A marked column with no explanation invites the reader to ignore it."""
    import pathlib
    root = pathlib.Path(__file__).resolve().parent.parent
    for script in ("run_real_benchmark.py", "run_synthetic_benchmark.py"):
        src = (root / "benchmarks" / script).read_text(encoding="utf-8")
        assert "mark_non_selectors" in src, f"{script} does not mark non-selectors"
        assert "NO_SELECTION_NOTE" in src, f"{script} does not explain the marking"
    assert "full-image score" in NO_SELECTION_NOTE


def test_the_mlp_arm_searches_lam_because_it_cannot_close_otherwise():
    """Pinned because reverting it silently restores a non-selecting arm.

    At the linear head's fixed lam this arm kept every patch on multiclass data. The
    fix is a configuration change with no visible error if undone, which is exactly
    the kind that comes back.
    """
    import inspect
    src = inspect.getsource(SELECTORS["rpm_mlp"])
    assert 'lam_scale="auto"' in src, (
        "the rpm_mlp benchmark arm must search lam; at a fixed lam it returns "
        "coverage 1.000 on multiclass and selects nothing")


def test_the_benchmarks_readme_lists_every_registered_selector():
    """A method table that goes stale understates what the suite compares.

    It had gone stale: `rpm_mlp` and `rf_boot` were both registered and running in
    every component while the table listed neither, so a reader could not tell that
    the wrapper ablation (sec.15) or the MLP head (sec.17) were even being measured.
    This is the same defect the module-table guard catches for the package.
    """
    import pathlib
    root = pathlib.Path(__file__).resolve().parent.parent
    text = (root / "benchmarks" / "README.md").read_text(encoding="utf-8")
    missing = [name for name in SELECTORS if f"`{name}`" not in text]
    assert not missing, f"benchmarks/README.md method table is missing {missing}"
