"""The model fitted on the selected region, which need not be the mask's own head.

RobustModelMaker lets the caller choose the model that consumes the selected features;
RPM always refitted its own mask head, so the library was benchmarked with a random
forest and shipped a linear model. These tests pin the new `model=` argument, and above
all pin the invariant that makes it safe: choosing a model must not change WHICH patches
are selected.
"""
import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from robustpixelmaker import (
    NAMED_MODELS,
    BootstrapMaskSelector,
    DownstreamModel,
    PatchGrid,
    RobustPixelMaker,
    make_synthetic_images,
    resolve_model,
)


def _fit(model, **kw):
    c = make_synthetic_images(n=200, seed=0)
    r = RobustPixelMaker(selector="bootstrap", patch=4, n_iter=180, k_outer=3,
                         n_bootstrap=4, target_coverage=0.15, model=model,
                         random_state=0, **kw).fit(c.X, c.y)
    return c, r


def test_model_choice_never_changes_the_selection():
    """The separation the whole feature rests on: the mask picks the patches, the model
    only decides what is fitted on them. If this fails, `model=` is silently a selection
    knob and every comparison across models is confounded."""
    base = None
    for m in (None, "rf", "logreg", "knn", "svm"):
        c, r = _fit(m)
        if base is None:
            base = r.selected_patches_
        np.testing.assert_array_equal(r.selected_patches_, base)


def test_default_is_the_mask_head_unchanged():
    c, a = _fit(None)
    c2, b = _fit("mask")
    np.testing.assert_array_equal(a.selected_patches_, b.selected_patches_)
    np.testing.assert_allclose(a.predict_proba(c.X[:8]), b.predict_proba(c.X[:8]))
    assert a.downstream_ is None and b.downstream_ is None


def test_named_models_all_fit_and_predict():
    c = make_synthetic_images(n=160, seed=0)
    for name in NAMED_MODELS:
        _, r = _fit(name)
        assert r.predict(c.X[:5]).shape == (5,)
        if name != "mask":
            assert r.downstream_ is not None
            assert r.predict_proba(c.X[:5]).shape == (5, 2)


def test_a_sklearn_estimator_instance_is_accepted_and_cloned():
    tree = DecisionTreeClassifier(max_depth=3, random_state=0)
    c, r = _fit(tree)
    assert isinstance(r.downstream_.estimator, DecisionTreeClassifier)
    assert r.downstream_.estimator is not tree      # cloned, never mutated across folds
    assert r.summary()["model"] == "DecisionTreeClassifier"


def test_downstream_sees_only_the_selected_patches():
    c = make_synthetic_images(n=200, seed=0)
    sel = BootstrapMaskSelector(task="binary", patch=4, n_bootstrap=4,
                                n_iter=180, target_coverage=0.15, seed=0).fit(c.X, c.y)
    d = DownstreamModel(RandomForestClassifier(n_estimators=20, random_state=0),
                        sel.grid, sel.selected_patches_, "binary").fit(c.X, c.y)
    assert d._features(c.X).shape == (len(c.X), len(sel.selected_patches_))
    assert d._features(c.X).shape[1] < sel.grid.n_patches


def test_empty_selection_is_refused_not_scored_at_chance():
    c = make_synthetic_images(n=120, seed=0)
    grid = PatchGrid.from_image_shape(c.X.shape, patch=4)
    d = DownstreamModel(RandomForestClassifier(n_estimators=5), grid, [], "binary")
    with pytest.raises(ValueError, match="nothing for a downstream model"):
        d.fit(c.X, c.y)


def test_resolve_model_rejects_nonsense_clearly():
    with pytest.raises(ValueError, match="unknown model"):
        resolve_model("xgboost", "binary", 0)
    with pytest.raises(ValueError, match="must be None"):
        resolve_model(42, "binary", 0)
    assert resolve_model(None, "binary", 0) is None
    assert resolve_model("mask", "binary", 0) is None


def test_regression_downstream_refuses_predict_proba():
    c = make_synthetic_images(n=150, task="regression", seed=0)
    r = RobustPixelMaker(selector="bootstrap", patch=4, n_iter=150, k_outer=3,
                         n_bootstrap=3, target_coverage=0.2, model="rf",
                         random_state=0).fit(c.X, c.y)
    assert r.predict(c.X[:4]).shape == (4,)
    with pytest.raises(ValueError, match="undefined for task='regression'"):
        r.predict_proba(c.X[:4])


# --------------------------------------------------------------------------- #
# Documentation is a claim about behaviour, so it gets tested like one
# --------------------------------------------------------------------------- #
def test_every_named_model_the_docs_advertise_actually_works():
    """The README lists the named models. A name that appears in the docs
    and fails at runtime is a documentation defect, and this is what catches it."""
    c = make_synthetic_images(n=120, seed=0)
    import pathlib
    root = pathlib.Path(__file__).resolve().parent.parent
    docs = (root / "README.md").read_text(encoding="utf-8")
    for name in NAMED_MODELS:
        assert name in docs, f"README does not document the named model {name!r}"
        r = RobustPixelMaker(selector="bootstrap", patch=4, n_iter=100, k_outer=3,
                             n_bootstrap=3, model=name, random_state=0).fit(c.X, c.y)
        assert r.predict(c.X[:2]).shape == (2,)


def test_readme_module_table_lists_every_shipped_module():
    """A module table that silently goes stale is worse than none: it tells a reader
    the package is smaller than it is."""
    import pathlib
    root = pathlib.Path(__file__).resolve().parent.parent
    readme = (root / "README.md").read_text(encoding="utf-8")
    for mod in sorted((root / "robustpixelmaker").glob("*.py")):
        if mod.stem == "__init__":
            continue
        assert f"`{mod.name}`" in readme, f"README module table is missing {mod.name}"


def test_findings_cross_references_resolve():
    """Code comments and docs cite FINDINGS sections by number. A citation past the
    end of the document sends a reader looking for evidence that does not exist, which
    is how a caveat quietly becomes unsupported."""
    import pathlib, re
    root = pathlib.Path(__file__).resolve().parent.parent
    findings = (root / "benchmarks" / "FINDINGS.md").read_text(encoding="utf-8")
    n = len(re.findall(r"(?m)^## (\d+)\.", findings))
    assert n > 0
    sources = list((root / "robustpixelmaker").glob("*.py")) + [
        root / "README.md",
        root / "CHANGELOG.md"]
    for src in sources:
        for m in re.finditer(r"sec\.(\d+)", src.read_text(encoding="utf-8", errors="ignore")):
            assert int(m.group(1)) <= n, f"{src.name} cites sec.{m.group(1)}, only {n} exist"


def test_every_constructor_argument_the_readme_shows_is_actually_accepted():
    """The README's argument tables must name arguments the facade really takes.

    This exists because `head="mlp"` was documented in the README, described in the
    "How it works" section and listed in the module table, while `RobustPixelMaker`
    took no `head` argument at all: the documented option could only be used by
    bypassing the documented entry point. Prose and signature had drifted apart with
    nothing watching, and a reader following the README would have hit a TypeError.

    Only backtick-quoted names appearing in the README's argument tables are checked,
    which keeps the test specific: a name in a table with a default column is being
    advertised as a constructor argument.
    """
    import inspect
    import pathlib
    import re

    root = pathlib.Path(__file__).resolve().parent.parent
    readme = (root / "README.md").read_text(encoding="utf-8")
    params = set(inspect.signature(RobustPixelMaker.__init__).parameters) - {"self"}

    # rows of a markdown table whose first cell is `name` and which carry a default
    documented = set()
    for row in re.findall(r"^\|\s*`([a-z_]+)`\s*\|\s*`[^`]+`\s*\|", readme, re.M):
        documented.add(row)
    assert documented, "no argument table found; has the README structure changed?"

    missing = sorted(documented - params)
    assert not missing, (
        f"README documents constructor argument(s) {missing} that "
        f"RobustPixelMaker.__init__ does not accept")


def test_documented_arguments_reach_the_object_that_uses_them():
    """Accepting an argument is not the same as forwarding it.

    The facade could have been given `head` while `_fold_factory` and
    `_final_selector` still dropped it, which is the half-fix that would leave the
    README true and the behaviour unchanged. This pins the forwarding.
    """
    c = make_synthetic_images(n=100, seed=0)
    r = RobustPixelMaker(selector="bootstrap", head="mlp", hidden=8,
                         lam_scale="auto", auto_target=0.2, n_bootstrap=2,
                         n_iter=80, k_outer=2, random_state=0).fit(c.X, c.y)
    for name, expected in (("head", "mlp"), ("hidden", 8),
                           ("lam_scale", "auto"), ("auto_target", 0.2)):
        assert getattr(r.selector_, name) == expected, f"{name} was not forwarded"
    fold = r._fold_factory()()
    for name, expected in (("head", "mlp"), ("hidden", 8),
                           ("lam_scale", "auto"), ("auto_target", 0.2)):
        assert fold.kw[name] == expected, f"{name} not forwarded to the fold estimator"


def _published_test_count(root):
    """How many tests a visitor to the published repository can run.

    The badge and the prose live in the README, which is read in the PUBLISHED tree,
    so they must describe that tree. tests/test_staging.py covers the PyPI staging
    area and is deliberately excluded from the upload, so it is excluded here too.
    Works in both trees: locally it is ignored, and in the published tree it is
    already absent.
    """
    import re
    import subprocess
    import sys

    cmd = [sys.executable, "-m", "pytest", str(root / "tests"), "--collect-only", "-q"]
    staging = root / "tests" / "test_staging.py"
    if staging.exists():
        cmd += ["--ignore", str(staging)]
    out = subprocess.run(cmd, capture_output=True, text=True, cwd=str(root), timeout=600)
    m = re.search(r"(\d+) tests collected", out.stdout)
    assert m, "could not read the collected count: " + out.stdout[-2000:]
    return int(m.group(1))


def test_no_document_claims_a_stale_test_count():
    """Every doc that quotes a test count must quote the real one.

    Counts rot silently: they are written once, in prose, and nothing recomputes them.
    At the time this guard was added, README said 195, the design notes said 170 and
    the release checklist said 142, against an actual 200. A reader uses those to judge
    how well tested the library is, so a stale one is a false claim about evidence.

    `--collect-only` does not execute anything, so this cannot recurse.
    """
    import pathlib
    import re
    import subprocess
    import sys

    root = pathlib.Path(__file__).resolve().parent.parent
    actual = _published_test_count(root)

    # historical statements about a PAST state are legitimate and are skipped: a
    # changelog entry for a released version, or a bug sweep describing what was
    # green at the time, should not be rewritten as the suite grows
    HISTORICAL = {"CHANGELOG.md"}
    # only PUBLISHED docs are policed. An untracked local document (the private
    # design plan) must not be able to fail this suite, and would fail it for
    # anyone who does not have a copy.
    import subprocess
    listing = subprocess.run(["git", "ls-files", "*.md", "benchmarks/*.md"],
                             cwd=str(root), capture_output=True, text=True)
    if listing.returncode != 0:
        import pytest
        pytest.skip("not a git checkout")
    docs = [root / rel for rel in listing.stdout.split(chr(10)) if rel.strip()]
    stale = []
    for doc in sorted(docs):
        if doc.name in HISTORICAL or not doc.is_file():
            continue
        for claim in re.findall(r"(\d+) tests\b", doc.read_text(encoding="utf-8")):
            if int(claim) != actual:
                stale.append(f"{doc.name} says {claim} tests, actual {actual}")
    assert not stale, "stale test counts: " + "; ".join(stale)


def test_the_benchmark_docs_count_the_rpm_variants_they_list():
    """A count written beside a list must match the list.

    BENCHMARKS.md said "four RPM variants" and then named five. Small, but it is the
    kind of error that makes a reader distrust the numbers that matter.
    """
    import pathlib
    import re

    root = pathlib.Path(__file__).resolve().parent.parent
    text = (root / "benchmarks" / "BENCHMARKS.md").read_text(encoding="utf-8")
    m = re.search(r"(\w+) RPM variants \(([^)]*)\)", text)
    assert m, "the RPM-variants sentence has changed shape; update this guard"
    words = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7}
    claimed = words.get(m.group(1).lower())
    assert claimed is not None, f"unrecognised count word {m.group(1)!r}"
    listed = len(re.findall(r"`[^`]+`", m.group(2)))
    assert claimed == listed, (
        f"BENCHMARKS.md claims {m.group(1)} RPM variants but lists {listed}")


def test_changelog_has_one_unreleased_block_at_most():
    """Prepending a new Unreleased block instead of merging is silent and lossy.

    It happened: two `## Unreleased` headings accumulated, each with its own Added and
    Fixed subsections, and at release time they merged into one version's notes and
    had to be reconciled by hand. Nothing complained, because a second heading is
    valid markdown.
    """
    import pathlib
    import re

    root = pathlib.Path(__file__).resolve().parent.parent
    text = (root / "CHANGELOG.md").read_text(encoding="utf-8")
    n = len(re.findall(r"(?m)^## Unreleased\s*$", text))
    assert n <= 1, f"CHANGELOG has {n} Unreleased blocks; merge them into one"


def test_the_shipped_version_has_a_changelog_section():
    """A release whose notes are still filed under Unreleased ships undocumented.

    The version is read from `__version__`, which is what the build uses, so this also
    catches a bump applied to the changelog but not the package or the reverse.
    """
    import pathlib
    import re

    import robustpixelmaker

    root = pathlib.Path(__file__).resolve().parent.parent
    text = (root / "CHANGELOG.md").read_text(encoding="utf-8")
    version = robustpixelmaker.__version__
    versions = re.findall(r"(?m)^## (\d+\.\d+\.\d+)", text)
    assert versions, "no version sections found in CHANGELOG"
    assert version in versions, (
        f"__version__ is {version} but CHANGELOG documents {versions}")
    assert versions[0] == version, (
        f"__version__ is {version} but the newest CHANGELOG section is {versions[0]}")


def test_package_metadata_names_every_robustmaker_the_readme_does():
    """The three RobustMakers must agree between the README and the package metadata.

    The README documents RobustModelMaker, RobustPixelMaker and RobustSignalMaker as
    the three RobustMakers, while `pyproject.toml` listed only two siblings, so the
    PyPI page understated the family. The README table had itself omitted
    RobustSignalMaker until recently, which is why this is worth pinning in both
    directions rather than trusting either one.
    """
    import pathlib
    import re

    root = pathlib.Path(__file__).resolve().parent.parent
    readme = (root / "README.md").read_text(encoding="utf-8")
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    for sibling in ("RobustModelMaker", "RobustSignalMaker"):
        assert sibling in readme, f"README does not mention {sibling}"
        assert sibling in pyproject, (
            f"pyproject.toml does not link {sibling}, so the PyPI page understates "
            "the RobustMaker family")


def test_no_tracked_file_references_a_private_document():
    """Three documents are deliberately unpublished, so nothing shipped may cite them,
    by filename OR in prose.

    The design plan was cited twenty times across fifteen files, ten of them modules
    that go inside the wheel, so a PyPI user would have read a docstring pointing at a
    document in neither the repository nor the package. The bug-sweep record was cited
    eleven times and the release checklist three. Each was repointed at something
    published: RELATED_WORK for the novelty and the regimes, README for the design
    summary, CHANGELOG for the defects and milestones, and BUILD_INSTRUCTIONS for the
    release steps.

    Matching the filename alone was not enough. Five more references survived that
    pass by naming the document in prose, one of them in config.py, which ships inside
    the wheel. The prose patterns below close that hole.

    Checked against the git index rather than the working tree, because all three
    files are present locally and must stay that way. Needles are assembled at runtime,
    and this file excludes itself, since a guard has to be able to describe what it
    forbids.
    """
    import pathlib
    import subprocess

    import pytest

    names = ["RPM" + "_PLAN", "BUG" + "_SWEEPS", "RELEAS" + "ING"]
    prose = ["see " + "plan", "plan " + chr(167), "(" + "plan)",
             "the " + "plan's", "the " + "plan not", "in the " + "plan "]
    root = pathlib.Path(__file__).resolve().parent.parent
    listing = subprocess.run(["git", "ls-files"], cwd=str(root),
                             capture_output=True, text=True)
    if listing.returncode != 0:
        pytest.skip("not a git checkout")
    tracked = [f for f in listing.stdout.split(chr(10)) if f.strip()]
    for needle in names:
        assert needle + ".md" not in tracked, (
            needle + ".md is tracked and could be pushed")

    # Scope: files that will actually be PUBLISHED. Release machinery is tracked but
    # excluded from the upload, and it names these documents precisely in order to
    # keep them out, so flagging it would be backwards. In the published tree the
    # excluder is absent and everything present is published, which is also correct.
    excluded = ()
    prep = root / "prepare_upload.py"
    if prep.exists():
        import importlib.util
        spec = importlib.util.spec_from_file_location("_prep", prep)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        excluded = tuple(mod.PUBLISH_EXCLUDE) + ("prepare_upload.py",)

    myself = pathlib.Path(__file__).name
    offenders = {}
    for rel in tracked:
        path = root / rel
        if not path.is_file() or path.name == myself:
            continue
        if any(rel == e or rel.startswith(e) for e in excluded):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        hits = [n for n in names + prose if n in text]
        if hits:
            offenders[rel] = hits
    assert not offenders, "tracked files cite private documents: " + str(offenders)

def test_readme_badges_state_the_truth():
    """Badges are the most-read and least-maintained thing in a README.

    They sit above the fold, a reader trusts them at a glance, and nothing
    recomputes them. The test-count and version badges are therefore checked against
    the same sources as the prose: `pytest --collect-only` and `__version__`. The
    coverage badge is rounded, so it is checked for presence and plausibility rather
    than to the decimal.
    """
    import pathlib
    import re
    import subprocess
    import sys

    import robustpixelmaker

    root = pathlib.Path(__file__).resolve().parent.parent
    readme = (root / "README.md").read_text(encoding="utf-8")

    version = re.search(r"badge/version-([0-9.]+)-", readme)
    assert version, "the version badge is missing from the README"
    assert version.group(1) == robustpixelmaker.__version__, (
        f"version badge says {version.group(1)}, __version__ is "
        f"{robustpixelmaker.__version__}")

    actual = _published_test_count(root)
    badge = re.search(r"badge/tests-(\d+)-", readme)
    assert badge, "the tests badge is missing from the README"
    assert int(badge.group(1)) == actual, (
        f"tests badge says {badge.group(1)}, actual collected count is {actual}")

    cov = re.search(r"badge/coverage-(\d+)%25-", readme)
    assert cov, "the coverage badge is missing from the README"
    assert 90 <= int(cov.group(1)) <= 100, "coverage badge is not a plausible figure"


def test_readme_badges_use_the_house_colours():
    """Viridis, like every other figure in these libraries, and the same five hex
    values RobustSignalMaker uses, so the family's pages look like one set."""
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent
    readme = (root / "README.md").read_text(encoding="utf-8")
    for colour in ("440154", "414487", "31688e", "21918c", "22a884"):
        assert colour in readme, f"badge colour {colour} is missing; expected viridis"


def _tracked_docs(root):
    import subprocess
    out = subprocess.run(["git", "ls-files", "*.md", "*.cff"], cwd=str(root),
                         capture_output=True, text=True)
    if out.returncode != 0:
        return None
    return [root / r for r in out.stdout.split(chr(10)) if r.strip()]


def test_every_internal_link_in_every_document_resolves():
    """A link to a file that is not there is the cheapest possible broken promise.

    Checked across every tracked document, not just the README, because the ones that
    rot are the files nobody opens: a results provenance note, a benchmark readme.
    """
    import pathlib
    import re

    import pytest

    root = pathlib.Path(__file__).resolve().parent.parent
    docs = _tracked_docs(root)
    if docs is None:
        pytest.skip("not a git checkout")
    broken = []
    for d in docs:
        for target in re.findall(r"\]\(([^)]+)\)", d.read_text(encoding="utf-8")):
            if target.startswith(("http", "#", "mailto:")):
                continue
            clean = target.split("#")[0]
            if clean and not (d.parent / clean).exists():
                broken.append(f"{d.relative_to(root)} -> {target}")
    assert not broken, "dead internal links: " + str(broken)


def test_no_last_updated_stamp_predates_its_own_last_edit():
    """A "Last updated" line is a claim, and it is the one nobody remembers to change.

    RELATED_WORK said 2026-07-21 and FINDINGS said 2026-07-29, both for files edited
    almost two months later. Amanda spotted the first by eye, which is exactly the job
    a test should be doing.
    """
    import pathlib
    import re
    import subprocess

    import pytest

    root = pathlib.Path(__file__).resolve().parent.parent
    docs = _tracked_docs(root)
    if docs is None:
        pytest.skip("not a git checkout")

    # A freshly imported repository (the manual GitHub upload, or a clone of it) has
    # one commit, so EVERY file's last-change date is the import date and every stamp
    # looks stale. That is an artefact of the history, not of the documents, and this
    # check would otherwise fail for everyone who clones the published tree.
    n_commits = subprocess.run(["git", "rev-list", "--count", "HEAD"], cwd=str(root),
                               capture_output=True, text=True).stdout.strip()
    if n_commits.isdigit() and int(n_commits) < 5:
        pytest.skip("history was imported in one commit; per-file dates carry no "
                    "information here")

    stale = []
    for d in docs:
        m = re.search(r"Last updated: (\d{4}-\d{2}-\d{2})", d.read_text(encoding="utf-8"))
        if not m:
            continue
        rel = str(d.relative_to(root)).replace(chr(92), "/")
        last = subprocess.run(["git", "log", "-1", "--format=%ad", "--date=short",
                               "--", rel], cwd=str(root),
                              capture_output=True, text=True).stdout.strip()
        if last and last > m.group(1):
            stale.append(f"{d.relative_to(root)}: says {m.group(1)}, last changed {last}")
    assert not stale, "stale Last-updated stamps: " + str(stale)


def test_no_document_contains_an_em_or_en_dash():
    """House style, applied to every document rather than remembered per file."""
    import pathlib

    import pytest

    root = pathlib.Path(__file__).resolve().parent.parent
    docs = _tracked_docs(root)
    if docs is None:
        pytest.skip("not a git checkout")
    em, en = chr(8212), chr(8211)          # 8213 is a horizontal bar, not an en dash
    offenders = []
    for d in docs:
        t = d.read_text(encoding="utf-8")
        hits = [name for name, ch in (("em", em), ("en", en)) if ch in t]
        if hits:
            offenders.append(f"{d.relative_to(root)} ({', '.join(hits)})")
    assert not offenders, "documents containing a dash: " + str(offenders)


def test_every_tracked_result_file_has_a_provenance_entry():
    """`benchmarks/results/` holds runs from several months and several defaults, and
    the files do not say which is which.

    `run_coverage_frontier.py` alone was run six times on different arms; every run
    writes the same generic title and the same table shape, so the file name is the
    only thing distinguishing `COVERAGE_HEAD.md` from `COVERAGE_HEAD4.md`. Three of
    those six were superseded by a bug fix. A reader who opens the wrong one gets a
    number that was true of a version that no longer exists, with nothing to warn them.

    `PROVENANCE.md` is the index that resolves this, so a result file it does not name
    is a result file nobody can date. When this guard was added it caught twenty-seven,
    including the run that exposed FINDINGS sec.20 and the whole timing family.

    Names are matched as globs, because PROVENANCE legitimately writes `synthetic_*.png`
    for a group that arrives together from one script.
    """
    import fnmatch
    import pathlib
    import re
    import subprocess

    import pytest

    root = pathlib.Path(__file__).resolve().parent.parent
    listing = subprocess.run(["git", "ls-files", "benchmarks/results"], cwd=str(root),
                             capture_output=True, text=True)
    if listing.returncode != 0:
        pytest.skip("not a git checkout")
    tracked = [f for f in listing.stdout.split(chr(10)) if f.strip()]
    if not tracked:
        pytest.skip("no tracked result files")

    prov = root / "benchmarks" / "results" / "PROVENANCE.md"
    patterns = set(re.findall(r"`([^`]+)`", prov.read_text(encoding="utf-8")))
    missing = [pathlib.Path(f).name for f in tracked
               if pathlib.Path(f).name != "PROVENANCE.md"
               and not any(fnmatch.fnmatch(pathlib.Path(f).name, p) for p in patterns)]
    assert not missing, "result files with no provenance entry: " + str(sorted(missing))


def test_no_document_cites_a_result_file_that_is_not_published():
    """A source line naming a file the reader cannot open is not a source line.

    FINDINGS cited six run logs as the evidence for its claims. Console logs are
    gitignored, 879 KB of transcript whose substance is already in the CSVs, so all six
    resolved on this machine and nowhere else. Two were the only cited evidence for
    sec.20. The fix was both directions: repoint what a tracked CSV already carries, and
    track the three small logs that carry numbers nothing else does.

    Globs are skipped, since a citation like `results/synthetic_*.png` names a group.
    """
    import pathlib
    import re
    import subprocess

    import pytest

    root = pathlib.Path(__file__).resolve().parent.parent
    listing = subprocess.run(["git", "ls-files"], cwd=str(root),
                             capture_output=True, text=True)
    if listing.returncode != 0:
        pytest.skip("not a git checkout")
    tracked = {f for f in listing.stdout.split(chr(10)) if f.strip()}

    dangling = []
    for rel in sorted(f for f in tracked if f.endswith(".md")):
        text = (root / rel).read_text(encoding="utf-8")
        for token in re.findall(r"`(results/[^`]+|benchmarks/results/[^`]+)`", text):
            token = token.strip()
            if "*" in token or "?" in token:
                continue
            full = token if token.startswith("benchmarks/") else "benchmarks/" + token
            if full not in tracked:
                dangling.append(rel + " cites " + token)
    assert not dangling, "citations of unpublished files: " + str(dangling)
