"""Competitor feature selectors and the common scoring protocol for benchmarks.

Every selector operates on the SAME patch-pooled feature space and returns the set of
selected patch indices, so the comparison is about *which patches* each method keeps.
A common downstream Random Forest is then fit on the selected patches and scored on a
held-out fold (this mirrors RobustModelMaker's benchmark methodology: refit a fresh
model on the selected subset). RobustPixelMaker (`rpm`) is the only method that reads
the images directly; the rest read the pooled features.

Selectors: full (no selection, the baseline), anova (F-test filter), lasso (L1
embedded), rfe (recursive elimination), rf (random-forest Gini importance), rpm.
"""
from __future__ import annotations

import numpy as np

from robustpixelmaker.masking import SoftMaskSelector


# --------------------------------------------------------------------------- #
# Selectors: (X_img_train, F_train, y_train, grid, task, seed) -> patch indices
# --------------------------------------------------------------------------- #
def select_full(X_img, F, y, grid, task, seed):
    return np.arange(F.shape[1])


def select_anova(X_img, F, y, grid, task, seed):
    from sklearn.feature_selection import SelectFpr, f_classif, f_regression
    score_fn = f_regression if task == "regression" else f_classif
    try:
        support = SelectFpr(score_fn, alpha=0.05).fit(F, y).get_support()
    except Exception:
        return np.array([], dtype=int)
    return np.where(support)[0].astype(int)


def select_lasso(X_img, F, y, grid, task, seed):
    """L1 embedded selection.

    Iteration and grid sizes are capped: on real imagery an uncapped liblinear L1
    search failed to converge and dominated the whole benchmark (over 500s for a
    single dataset), which is a benchmarking artefact rather than a property of the
    method. saga with a modest Cs grid gives the same selection far more cheaply.
    """
    from sklearn.preprocessing import StandardScaler
    Fs = StandardScaler().fit_transform(F)
    if task == "regression":
        from sklearn.linear_model import LassoCV
        coef = LassoCV(cv=3, random_state=seed, n_alphas=20, max_iter=2000).fit(Fs, y).coef_
    else:
        from sklearn.linear_model import LogisticRegressionCV
        est = LogisticRegressionCV(
            cv=3, penalty="l1", solver="saga", Cs=4, random_state=seed,
            max_iter=300, tol=1e-3, n_jobs=1,
        ).fit(Fs, y)
        coef = np.abs(est.coef_).max(axis=0) if est.coef_.ndim == 2 else np.abs(est.coef_)
        return np.where(coef > 1e-8)[0].astype(int)
    return np.where(np.abs(coef) > 1e-8)[0].astype(int)


def select_rfe(X_img, F, y, grid, task, seed):
    from sklearn.feature_selection import RFECV
    P = F.shape[1]
    step = max(1, P // 10)
    if task == "regression":
        from sklearn.linear_model import Ridge
        est = Ridge(random_state=seed)
        scoring = "neg_root_mean_squared_error"
    else:
        from sklearn.linear_model import LogisticRegression
        est = LogisticRegression(max_iter=2000, random_state=seed)
        scoring = "roc_auc"
    try:
        sel = RFECV(est, step=step, cv=3, scoring=scoring, min_features_to_select=1).fit(F, y)
        return np.where(sel.support_)[0].astype(int)
    except Exception:
        return np.arange(P)


def select_rf(X_img, F, y, grid, task, seed):
    if task == "regression":
        from sklearn.ensemble import RandomForestRegressor as RF
    else:
        from sklearn.ensemble import RandomForestClassifier as RF
    est = RF(n_estimators=200, random_state=seed).fit(F, y)
    imp = est.feature_importances_
    return np.where(imp > imp.mean())[0].astype(int)


def select_rpm(X_img, F, y, grid, task, seed):
    """RPM with sigmoid gates and an L1 penalty.

    The gate is named explicitly rather than left to the library default. When the
    default changed to Hard-Concrete this selector silently became a duplicate of
    ``select_rpm_l0`` and both reported identical numbers, so the two variants must
    each pin their own gate.
    """
    sel = SoftMaskSelector(
        task=task, patch=grid.patch, lam=0.03, n_iter=500, gate="sigmoid", seed=seed
    ).fit(X_img, y)
    return np.asarray(sel.selected_patches_, dtype=int)


def select_rpm_l0(X_img, F, y, grid, task, seed):
    """RPM with Hard-Concrete L0 gates: penalises the gate COUNT, not magnitude."""
    sel = SoftMaskSelector(
        task=task, patch=grid.patch, lam=0.03, n_iter=500, gate="hardconcrete", seed=seed
    ).fit(X_img, y)
    return np.asarray(sel.selected_patches_, dtype=int)


def select_rpm_boot(X_img, F, y, grid, task, seed):
    """RPM with bootstrap stability selection over the mask (Milestone 4).

    Aggregates per-patch selection frequency across resamples and keeps the patches
    above the threshold, rather than trusting one mask fit. Costs ``n_bootstrap`` times
    a single fit, which is the price of the frequency estimate.
    """
    from robustpixelmaker.selection import BootstrapMaskSelector

    sel = BootstrapMaskSelector(
        task=task, patch=grid.patch, lam=0.03, n_iter=400, n_bootstrap=10,
        tau=0.6, gate="hardconcrete", seed=seed,
    ).fit(X_img, y)
    return np.asarray(sel.selected_patches_, dtype=int)


def select_rpm_ens(X_img, F, y, grid, task, seed):
    """RPM with the full method: double bootstrap over resamples AND representations."""
    from robustpixelmaker.selection import RepresentationEnsembleSelector

    sel = RepresentationEnsembleSelector(
        task=task, patch=grid.patch, lam=0.03, n_iter=400, n_bootstrap=5,
        n_representations=4, tau=0.6, gate="hardconcrete", seed=seed,
    ).fit(X_img, y)
    return np.asarray(sel.selected_patches_, dtype=int)


SELECTORS = {
    "full": select_full,
    "anova": select_anova,
    "lasso": select_lasso,
    "rfe": select_rfe,
    "rf": select_rf,
    "rpm": select_rpm,
    "rpm_l0": select_rpm_l0,
    "rpm_boot": select_rpm_boot,
    "rpm_ens": select_rpm_ens,
}

#: Display order for tables and plots. Derived from SELECTORS so a newly registered
#: selector can never be silently dropped from the reported results.
METHOD_ORDER = list(SELECTORS)


def ordered_methods(present) -> list:
    """Methods in display order, keeping any not listed in SELECTORS at the end."""
    present = list(present)
    known = [m for m in METHOD_ORDER if m in present]
    return known + [m for m in present if m not in known]


# --------------------------------------------------------------------------- #
# Common downstream scorer and recovery metrics
# --------------------------------------------------------------------------- #
def score_selection(F_tr, y_tr, F_te, y_te, sel, task, seed):
    """Fit a common Random Forest on the selected patches and score on the held-out fold.

    AUC (binary), weighted one-vs-rest AUC (multiclass), RMSE (regression). An empty
    selection scores at chance rather than raising, so an over-aggressive selector is
    penalised instead of crashing the sweep.
    """
    from sklearn.metrics import mean_squared_error, roc_auc_score

    if len(sel) == 0:
        if task == "regression":
            return float(np.sqrt(mean_squared_error(y_te, np.full_like(y_te, y_tr.mean(), dtype=float))))
        return 0.5  # chance AUC
    Xtr, Xte = F_tr[:, sel], F_te[:, sel]
    if task == "regression":
        from sklearn.ensemble import RandomForestRegressor
        pred = RandomForestRegressor(n_estimators=200, random_state=seed).fit(Xtr, y_tr).predict(Xte)
        return float(np.sqrt(mean_squared_error(y_te, pred)))

    from sklearn.ensemble import RandomForestClassifier
    est = RandomForestClassifier(n_estimators=200, random_state=seed).fit(Xtr, y_tr)
    proba = est.predict_proba(Xte)
    try:
        if task == "multiclass":
            return float(roc_auc_score(y_te, proba, multi_class="ovr", average="weighted"))
        return float(roc_auc_score(y_te, proba[:, 1]))
    except ValueError:
        return float("nan")


def recovery_metrics(sel, info_patches, dist_patches, n_patches):
    """Patch-level recovery of the ground-truth informative region."""
    sel = set(int(x) for x in sel)
    info = set(int(x) for x in info_patches)
    dist = set(int(x) for x in dist_patches)
    inter = len(sel & info)
    precision = inter / len(sel) if sel else 0.0
    recall = inter / len(info) if info else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    iou = inter / len(sel | info) if (sel | info) else 1.0
    distractor_fp = len(sel & dist) / len(dist) if dist else 0.0
    return dict(precision=precision, recall=recall, f1=f1, iou=iou, distractor_fp=distractor_fp)


# --------------------------------------------------------------------------- #
# Non-selectors must not read as results
# --------------------------------------------------------------------------- #
#: coverage at or above this counts as "kept everything", allowing for float noise
NO_SELECTION = 0.999


def mark_non_selectors(frame, coverage_col="coverage"):
    """Add a `selects` column so an arm that kept everything cannot read as a win.

    This exists because the summary tables compare a `score` column across arms
    without saying whether an arm selected anything. On multiclass data the
    `rpm_mlp` arm returned coverage 1.000, stability 1.000 and a score equal to the
    full-image baseline to three decimals, because an expressive head at a fixed lam
    finds a use for every input and no gate closes (FINDINGS sec.17). Read down the
    score column it looked tied for best on blood. It had selected nothing.

    Keeping everything is also perfectly reproducible, so it scores 1.000 on
    stability too, and every headline column flatters it at once. That is the whole
    hazard family: a metric that rewards not selecting will be won by a method that
    does not select. `rfe`, `anova` and `lasso` reach coverage 1.000 on some datasets
    legitimately, and they are marked the same way for the same reason.
    """
    out = frame.copy()
    out["selects"] = ["no" if c >= NO_SELECTION else "yes" for c in out[coverage_col]]
    return out


#: prose for the summary files, kept beside the rule it describes
NO_SELECTION_NOTE = (
    "`selects` is `no` where an arm kept the whole image (coverage at or above "
    f"{NO_SELECTION}). Such an arm is not a selection result: its score is the "
    "full-image score and its stability is 1.000 because keeping everything is "
    "perfectly reproducible. Compare it against `full`, not against the selectors. "
    "A `yes` is not a claim that the arm selected WELL: coverage 0.956 is marked "
    "`yes` and is barely a selection, so read the coverage column alongside the "
    "score rather than treating this as a pass mark."
)


# --------------------------------------------------------------------------- #
# Panel scoring: marginalise the verdict over the evaluator
# --------------------------------------------------------------------------- #
JUDGES = ("rf", "logreg", "svm", "knn")


def score_panel(F_tr, y_tr, F_te, y_te, sel, task, seed, judges=JUDGES) -> dict:
    """Score one selection with several downstream models, not just one.

    ``score_selection`` refits a Random Forest, and that quietly advantages a single
    competitor: random-forest importance ranks features by their usefulness *to a random
    forest*, so selector and evaluator share an inductive bias. Measured on the
    matched-coverage sweep, the effect is about the size of the between-method
    differences: judged by a random forest the `rf` selector wins 9 of 20 operating
    points and RPM 2, and judged by logistic regression RPM wins 10 and `rf` 4
    (FINDINGS sec.14).

    Swapping to one different judge does not fix this, it relocates it. A linear judge
    favours whatever ranks linearly informative features, which is what an F test does.
    The defensible construction is a PANEL spanning different inductive biases (trees,
    linear, kernel, instance-based) with the verdict marginalised over it, which is the
    same argument this library makes about representations, applied to its own protocol.

    Returns one score per judge. `rf` reproduces ``score_selection`` exactly, so panel
    results stay comparable with every result recorded before this existed.
    """
    import numpy as np

    out = {}
    for j in judges:
        if j == "rf":
            out[j] = score_selection(F_tr, y_tr, F_te, y_te, sel, task, seed)
            continue
        out[j] = _score_one(j, F_tr, y_tr, F_te, y_te, sel, task, seed)
    finite = [v for v in out.values() if v == v]
    out["panel_mean"] = float(np.mean(finite)) if finite else float("nan")
    return out


def _score_one(judge, F_tr, y_tr, F_te, y_te, sel, task, seed):
    import numpy as np
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import mean_squared_error, roc_auc_score
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC, SVR

    if len(sel) == 0:
        if task == "regression":
            return float(np.sqrt(mean_squared_error(
                y_te, np.full_like(y_te, y_tr.mean(), dtype=float))))
        return 0.5
    Xtr, Xte = F_tr[:, sel], F_te[:, sel]

    if task == "regression":
        model = {"logreg": Ridge(),
                 "svm": SVR(),
                 "knn": KNeighborsRegressor(n_neighbors=5)}[judge]
        pred = make_pipeline(StandardScaler(), model).fit(Xtr, y_tr).predict(Xte)
        return float(np.sqrt(mean_squared_error(y_te, pred)))

    model = {"logreg": LogisticRegression(max_iter=2000),
             "svm": SVC(probability=True, random_state=seed),
             "knn": KNeighborsClassifier(n_neighbors=5)}[judge]
    est = make_pipeline(StandardScaler(), model).fit(Xtr, y_tr)
    proba = est.predict_proba(Xte)
    try:
        if task == "multiclass":
            return float(roc_auc_score(y_te, proba, multi_class="ovr", average="weighted"))
        return float(roc_auc_score(y_te, proba[:, 1]))
    except ValueError:
        return float("nan")


def select_rf_boot(X_img, F, y, grid, task, seed):
    """Random-forest importance under RPM's OWN stability wrapper.

    The ablation the suite was missing. Every stability-selected arm here uses the
    learned mask (`rpm_boot`, `rpm_ens`) and every non-mask arm is a single fit
    (`anova`, `lasso`, `rfe`, `rf`), so the benchmark confounds two separate things:
    the learned mask against other rankers, and bootstrap aggregation against one fit.
    Comparing `rpm_boot` to `rf` cannot say which component does the work.

    This arm holds the wrapper fixed and swaps the base ranker. Same resampling scheme,
    same B, same stratification, same selection-frequency threshold as `select_rpm_boot`;
    only the per-resample ranking changes, from a learned mask to RF importance above
    its own mean. If this matches `rpm_boot`, the value is in the wrapper, which is
    Meinshausen-Buehlmann and applies to any ranker; if `rpm_boot` wins, the mask earns
    its cost.
    """
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    from robustpixelmaker.reproducibility import Seeds
    from robustpixelmaker.selection import resample_indices

    B, tau = 10, 0.6
    n, P = F.shape
    seeds = Seeds(base=int(seed), n_bootstrap=B)
    y_strat = None if task == "regression" else y
    RF = RandomForestRegressor if task == "regression" else RandomForestClassifier

    hits = np.zeros(P)
    for b in range(B):
        rng = np.random.default_rng(seeds.bootstrap(b))
        idx = resample_indices(n, "bootstrap", rng, y_strat)
        try:
            imp = RF(n_estimators=200,
                     random_state=int(seeds.bootstrap(b)) % (2**31)).fit(F[idx], y[idx]).feature_importances_
        except Exception:
            continue
        hits[imp > imp.mean()] += 1.0
    pi = hits / B
    out = np.where(pi >= tau)[0].astype(int)
    return out if len(out) else np.array([int(np.argmax(pi))], dtype=int)


SELECTORS["rf_boot"] = select_rf_boot


def select_rpm_mlp(X_img, F, y, grid, task, seed):
    """RPM stability selection with a nonlinear (one hidden layer) frozen head.

    Phase 1 fits the MLP on all patches and freezes it; phase 2 prunes gates against
    it, exactly as the linear head does. The difference is what a gate can be rewarded
    for: a linear head only sees main effects, so on label rules with none it prunes
    the entire informative region (FINDINGS sec.16, recovery F1 exactly 0.000 on two of
    four rules). Routing the phase-2 gradient through a hidden layer lets a patch earn
    its place through an interaction.
    """
    from robustpixelmaker.selection import BootstrapMaskSelector

    # lam_scale="auto" is not a tuning preference here, it is the difference between
    # selecting and not selecting. At the linear head's fixed lam=0.03 this arm
    # returned coverage 1.000 on both multiclass sets, scoring the full-image score
    # with stability 1.000 while choosing nothing (FINDINGS sec.17, sec.20). auto
    # searches lam for a mask that actually closes, and since sec.18 it is calibrated
    # once per fit rather than inside every resample, so it is affordable here.
    sel = BootstrapMaskSelector(
        task=task, patch=grid.patch, lam=0.03, n_iter=400, n_bootstrap=10,
        tau=0.6, gate="hardconcrete", head="mlp", lam_scale="auto", seed=seed,
    ).fit(X_img, y)
    return sel.selected_patches_


SELECTORS["rpm_mlp"] = select_rpm_mlp
