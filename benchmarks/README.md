# RobustPixelMaker benchmarks

Comprehensive comparison of RobustPixelMaker against standard feature selectors. This
directory starts with **synthetic** controls (where the informative region is known,
so recovery can be measured exactly); real-world datasets come next.

## What is compared

All methods select among the SAME patch-pooled features and return a set of patches,
which is then refit and scored on a held-out fold (RobustModelMaker's "refit on the
selected subset" protocol). Only `rpm` reads the images directly.

Scoring uses a **panel of four downstream judges** (random forest, logistic regression,
SVM, kNN) rather than the single random forest this file originally described. The
change is not cosmetic: a random-forest judge shares a model family with the `rf`
selector, and swapping the judge moves the RPM-minus-RF margin by more than the methods
differ (p < 0.0001 for a linear judge). FINDINGS sec.14.

| method | selector |
|---|---|
| `full` | no selection (full-feature baseline) |
| `anova` | univariate F-test filter (`SelectFpr`, alpha 0.05) |
| `lasso` | L1 embedded selection (LassoCV / L1 LogisticRegressionCV) |
| `rfe` | recursive feature elimination with CV (`RFECV`) |
| `rf` | random-forest Gini importance above mean |
| `rpm` | RobustPixelMaker soft-mask selector, sigmoid gates + L1 penalty |
| `rpm_l0` | RobustPixelMaker with Hard-Concrete L0 gates (penalises gate count, exact 0/1 at test) |
| `rpm_boot` | RPM with bootstrap stability selection over data resamples |
| `rpm_ens` | RPM full method: double bootstrap over data resamples AND representations |
| `rpm_mlp` | RPM with a one-hidden-layer head, so a patch can be kept for an interaction; runs `lam_scale="auto"` because at a fixed `lam` it keeps everything (FINDINGS sec.17, sec.20) |
| `rf_boot` | RF importance under RPM's OWN bootstrap wrapper, which isolates the learned mask from the aggregation around it (FINDINGS sec.15) |

## Metrics

- **Recovery** (synthetic only, vs the known region): `precision`, `recall`, `f1`, `iou`
  of selected patches against the informative region; `distractor_fp` = fraction of the
  equal-variance non-predictive distractor wrongly selected (lower is better).
- **coverage**: fraction of patches kept (leaner is better at equal score).
- **stability**: mean pairwise Jaccard of the selected sets across folds (higher is
  better; this is the reproducibility axis). Read it **next to coverage**: raw Jaccard
  is confounded with how much a method keeps (r = +0.99 on one dataset), so use
  `adjusted_jaccard` when comparing configurations at different coverage.
- **score**: AUC (binary, higher better) or RMSE (regression, lower better), from the
  common Random Forest on the selected subset.

## Sweeps

One factor at a time around a default (`n=400`, `32x32`, binary, `signal=2`, `noise=1`,
2x2 informative + 2x2 distractor):

- sample size `n`: 100, 200, 400, 800
- image shape: 16x16, 24x24, 32x32, 48x48 (patch size scales so patch count stays comparable)
- noise: 0.5, 1, 2, 4
- signal: 1, 2, 3
- task: binary, regression
- complexity: informative `region_side` 1, 2, 3 and `distractor_side` 0, 1, 2 (on a 48x48 frame)

Three seeds per scenario for variance.

## Run

```bash
py benchmarks/run_synthetic_benchmark.py
```

```bash
py benchmarks/run_synthetic_benchmark.py --quick
```

Restrict to particular sweep axes and write to a separate prefix (used for the gate
ablation, so it does not overwrite the full results):

```bash
py benchmarks/run_synthetic_benchmark.py --axes noise,region_side --out-prefix gate_ablation
```

Real scientific datasets:

```bash
py benchmarks/run_real_benchmark.py --datasets mnist-3v8,mnist-all
```

Sparsity-strength frontier (sweeps `lam` for both gate types so the operating point is
an explicit choice rather than a hidden default):

```bash
py benchmarks/run_lam_frontier.py --dataset mnist-3v8
```

## Outputs (in `results/`)

- `synthetic_results.csv`: one row per (scenario, method, seed) with all metrics.
- `SYNTHETIC_SUMMARY.md`: headline table, binary score/coverage/stability table,
  recovery-vs-factor tables, and embedded figures.
- `synthetic_recovery_stability.png`, `synthetic_recovery_sweeps.png`,
  `synthetic_score_coverage_frontier.png`. The prefix is `--out-prefix`, which
  defaults to `synthetic`, so a rerun under another prefix writes its own set.

## Real-world datasets

Scope decision: **scientific imaging only**. MNIST digits are included because the
images are centred (hence registered), because the VTF paper used the hard 3-vs-8
subset so it gives a direct comparison with our own prior work, and because it forces
multiclass support. Consumer and product image sets are out of scope.

Loaders live in `datasets.py` and return `(X, y, meta)` with `X` shaped (n, H, W).

| dataset | regime | task | status |
|---|---|---|---|
| MNIST digits 3-vs-8 | registered | binary | implemented |
| MNIST digits 10-class | registered | multiclass | implemented |
| MedMNIST pneumonia (chest X-ray) | registered | binary | benchmarked |
| MedMNIST blood (cell microscopy) | registered | multiclass, 8 | benchmarked |
| MedMNIST organa (abdominal CT) | registered | multiclass, 11 | benchmarked |
| MedMNIST breast (ultrasound, n=780) | registered | binary, small-sample | benchmarked |
| MedMNIST oct (retinal OCT) | registered | multiclass, 4 | benchmarked (showcase regime) |
| MedMNIST derma (dermatoscopy) | registered | multiclass, 7 | benchmarked (showcase regime) |
| MedMNIST tissue, path, retina | registered | multiclass | loaders ready |
| Galaxy10 DECaLS (morphology) | registered | multiclass | planned |
| PatchCamelyon (histopathology) | registered | binary | planned |
| OASIS / ABIDE / IXI brain MRI | registered (atlas) | binary or regression | planned, free registration |
| ADNI, UK Biobank, ABCD, HCP | registered (atlas) | various | planned, access-gated |
| NFFA-EUROPE SEM (microscopy) | unregistered | multiclass | planned |
| Cryo-EM single particles (EMPIAR) | registered after cropping | binary | planned |
| NIH ChestX-ray14 / retinal fundus | registered-ish | binary | planned, robustness showcase |

On real data there is no ground-truth informative region, so the recovery metrics are
not reported; score, coverage, and stability still are, plus (where a background is
identifiable) the robustness delta under corruption or shift.

## Extending

Add a selector by writing a `select_*(X_img, F, y, grid, task, seed) -> patch indices`
function in `competitors.py` and registering it in `SELECTORS`. Add a sweep axis in
`scenarios()` in the runner. Add a dataset by writing a loader in `datasets.py` and
registering it in `LOADERS`. Real-world runs reuse `competitors.py` and the same
metrics (minus recovery).
