# The RobustPixelMaker benchmark suite

The complete, reproducible evaluation of RPM against standard selectors. One command
reproduces everything:

```bash
py benchmarks/run_all.py --tier standard
```

Tiers: `smoke` (~10 min sanity pass), `standard` (the configurations behind
FINDINGS.md; the last two full runs took 248 and 268 minutes on this machine, which is
an observation rather than a cost, since the machine was not idle either time), `full`
(adds every registered MedMNIST set and a second frontier, an overnight run). Components run as independent subprocesses, so one
failure cannot poison the rest.

The consolidated conclusions, including every negative result, live in
[FINDINGS.md](FINDINGS.md). This file documents what each component measures and how.

## Protocol shared by every component

All selectors compete on the SAME patch-pooled feature space and return a set of
patches. A common Random Forest is then refit on the selected patches and scored on a
held-out fold, so the comparison isolates *which patches were chosen* from *which
classifier was used*. Every fitted quantity, selection included, is computed on the
training partition of each fold only.

Reported everywhere, as a triple that must be read together (no member of it means
anything alone; see FINDINGS sections 6, 7 and 9 for the three hazards this rule
guards against):

- **score**: AUC (binary), weighted one-vs-rest AUC (multiclass) or RMSE (regression).
  Reported per judge as `score_rf`, `score_logreg`, `score_svm`, `score_knn` and
  `score_panel_mean`; the bare `score` column remains the random-forest judge so tables
  written before the panel existed keep their meaning
- **coverage**: fraction of patches retained
- **stability**: mean pairwise Jaccard of the selected sets across folds; the
  chance-corrected variant (`adjusted_jaccard`) is preferred when comparing
  configurations at different coverage, because raw Jaccard rewards keeping more

On synthetic data, where the informative region is planted and known, recovery
(precision, recall, F1, IoU) and the fraction of an equal-variance decoy wrongly
selected are also reported. On centred real imagery, `border_coverage` (fraction of
the outer patch ring kept) is the background-rejection diagnostic, and the
`ambiguous_fraction` of the selection-frequency map is the go/no-go test for whether a
reproducible region exists at all.

## Components

| component | script | measures |
|---|---|---|
| Synthetic sweep | `run_synthetic_benchmark.py` | 23 scenarios x methods x 3 seeds over sample size, image shape, noise, signal, task, region size and decoy count, with ground-truth recovery |
| Gate ablation | same, `--axes noise,region_side` | sigmoid+L1 vs Hard-Concrete L0 gates in the two regimes where they differ |
| Sparsity frontier | `run_lam_frontier.py` | score vs coverage traced over the sparsity strength, both gates, competitors as reference points; the argument against single-operating-point comparisons |
| Real datasets | `run_real_benchmark.py` | digits plus MedMNIST chest X-ray, blood microscopy, abdominal CT, breast ultrasound (and, in `full`, OCT, dermatoscopy, tissue, pathology, retina) |
| Representation ablation | `run_representation_ablation.py` | the make-or-break novelty test: each single lens vs the double bootstrap over data and representations, with per-lens agreement |
| Cross-dataset table | `compare_real.py` | every real-data result merged into one side-by-side table |
| Selection maps | `compare_selection_maps.py` | WHICH patches each method keeps, drawn over the mean image, with a contiguity score |
| Nonlinear label rules | `run_nonlinear_control.py` | recovery F1 when the label is linear, conjunctive, interactive or symmetric in the planted cells; the axis that exposed the linear head returning an empty selection (FINDINGS sec.16) |
| Matched-coverage frontier | `run_coverage_frontier.py` | every arm forced to the same k patches, with a swappable downstream judge; the protocol that separates "better patches" from "patches that suit the evaluator" |
| Wrapper ablation | `competitors.select_rf_boot` | RF importance under RPM's OWN stability wrapper, isolating the learned mask from the bootstrap aggregation (FINDINGS sec.15) |
| Timing and scaling | `run_timing.py` | fit-time scaling across every axis, best-of-k, and a refusal to measure on a busy machine (see `results/TIMING_PROVISIONAL.md`) |

## Competitors

`full` (no selection), `anova` (univariate F-test), `lasso` (L1 embedded, capped: an
uncapped liblinear search is a benchmarking artefact, see FINDINGS sec.11), `rfe`
(RFECV), `rf` (random-forest importance), and five RPM variants (`rpm` sigmoid,
`rpm_l0`, `rpm_boot`, `rpm_ens`, `rpm_mlp`), plus `rf_boot`, which is RF importance under RPM's own
bootstrap wrapper and exists so the mask can be separated from the aggregation around it.
`rpm_mlp` runs with `lam_scale="auto"`, which for that arm is the difference between
selecting and not selecting rather than a tuning choice (FINDINGS sec.20).
Method display order derives from the selector
registry so a new selector can never silently vanish from the reports.

## Honesty rules the suite enforces

- Selector variants, output files and display lists all name themselves explicitly;
  three separate silent-collapse bugs came from implicit defaults (FINDINGS sec.11).
- An empty or near-empty selection reports stability as undefined, not perfect.
- **An arm that kept the whole image is marked `selects: no` in every summary table.**
  Not selecting wins on every headline column at once: the score is the full-image
  score by construction and the stability is 1.000 because keeping everything is
  perfectly reproducible. The suite shipped exactly this, an arm at coverage 1.000
  that read as tied for best, before the column existed (FINDINGS sec.20). The mark
  is applied by coverage alone, so baselines and RPM variants are treated alike.
- Predictions registered in advance stay in FINDINGS even when falsified (sec.10).
- A synthetic control whose label rule matches a method's model class flatters it. The
  shipped control is linear and RPM's head was linear, which hid a total failure on
  interactive labels; the nonlinear axis is now part of the smoke tier so that class of
  self-flattery cannot survive a single run.
- Wall-clock timings under background load are not quoted as method costs; only
  within-run relative comparisons are used. `run_timing.py` enforces this by refusing
  to measure on a busy machine, and a forced run is stamped `clean_run: false` in every
  output it writes.
