# Provenance of the files in this directory

Not every result here was produced by the same run, and a reader has no way to tell that
from the files themselves. This note records which is which, so no figure is quietly
assumed to be current.

## Current: the 2026-09-14 standard-tier run

Produced by `py benchmarks/run_all.py --tier standard`, all **ten** components
completing successfully in 268 minutes, on the library at v0.2.0. These are the numbers
`FINDINGS.md` now quotes.

- `synthetic_results.csv`, `SYNTHETIC_SUMMARY.md`, `synthetic_*.png`
- `nonlinear_control.csv` (the four label rules, a standing component since sec.16)
- `gate_ablation_results.csv`, `GATE_ABLATION_SUMMARY.md`, `gate_ablation_*.png`
- `lam_frontier_mnist-3v8.csv`, `lam_frontier_mnist-3v8.png`
- `real_results.csv`, `REAL_SUMMARY.md`, `real_score_coverage.png`
- `medmnist_results.csv`, `MEDMNIST_SUMMARY.md`, `medmnist_score_coverage.png`
- `representation_ablation.csv`, `REPRESENTATION_ABLATION.md`
- `CROSS_DATASET.md`
- `selection_maps_*.png`, `mask_*.png` (except `mask_medmnist-derma.png`)

**Only one arm moved between this run and the 2026-09-02 one it replaces.** Across 957
paired rows, score, coverage and stability are identical for every arm except
`rpm_mlp`, and identical on all 759 synthetic rows including that one. `rpm_mlp` moved
on all six real datasets because it had been returning coverage 1.000 on multiclass
data, scoring the full-image score while selecting nothing (FINDINGS sec.20). Its
corrected numbers are lower and honest. That an intentional one-arm change produced
exactly a one-arm difference is the check that the rerun was sound.

### Produced alongside it, from targeted runs rather than the tier

| file | what it is |
|---|---|
| `coverage_head4.csv`, `COVERAGE_HEAD4.md`, `coverage_head4.png` | the matched-coverage frontier at the shipped `auto_target`, FINDINGS sec.23 |
| `coverage_head2.csv`, `coverage_head3.csv` | the same protocol before the fix and at the old default, kept so sec.18 and sec.21 can be checked |
| `auto_target_sweep.csv` | the five-value sweep that set the `auto_target` default, sec.21 |
| `nonlinear_control_auto.csv` | the four label rules with the MLP arm searching `lam`, sec.19 |
| `ablation_wrapper_results.csv`, `ABLATION_WRAPPER_SUMMARY.md` | RF importance under RPM's own wrapper, the sec.15 ablation |
| `ablation_wrapper_score_coverage.png` | the figure for that ablation, 2026-09-10 |
| `ablation_head_results.csv`, `ABLATION_HEAD_SUMMARY.md`, `ablation_head_score_coverage.png` | 2026-09-11, the real-data run with the MLP arm added. This is the run that exposed sec.20: `rpm_mlp` reads coverage 1.000 on blood and 0.97 on breast, scoring the full image while selecting nothing. Kept as the evidence for that section |

### The matched-coverage frontier runs

`run_coverage_frontier.py` was run six times on different arms and defaults. Every one
writes the same generic title and the same table shape, so the **file name is the only
thing that distinguishes them**, which is why they are itemised here rather than left
to be told apart by eye.

| file group | produced | arms | what it is |
|---|---|---|---|
| `coverage_frontier.*`, `COVERAGE_FRONTIER.md` | 2026-09-07 | anova, mi, rf, rpm | the original frontier against the competitors, random-forest judge, 4 datasets. FINDINGS sec.14 |
| `coverage_frontier_logreg.*`, `COVERAGE_FRONTIER_LOGREG.md` | 2026-09-07 | anova, mi, rf, rpm | the same frontier under a linear judge, the pair that shows the evaluator moves the verdict. sec.14 |
| `coverage_tiebreak.*`, `COVERAGE_TIEBREAK.md` | 2026-09-07 | rf, rpm, rpm_pi | whether ranking by `pi_` rather than by the mask changes the frontier. 2 datasets. sec.14 |
| `coverage_head.*`, `COVERAGE_HEAD.md` | 2026-09-11 | rf, rpm, rpm_mlp | the first matched-coverage comparison of the MLP head. sec.17 |
| `coverage_head2.*`, `COVERAGE_HEAD2.md` | 2026-09-11 | rf, rpm, rpm_mlp | the same, **before** the sec.18 fix, with `lam` searched inside every resample |
| `coverage_head3.*`, `COVERAGE_HEAD3.md` | 2026-09-11 | rf, rpm, rpm_mlp | after that fix, still at the old `auto_target` default of 0.40. sec.18, sec.21 |
| `coverage_head4.*`, `COVERAGE_HEAD4.md` | 2026-09-14 | rf, rpm, rpm_mlp | after the fix and at the shipped default of 0.15. **This is the current one**, and the one sec.23 quotes |

## Earlier runs, retained deliberately

These come from configurations the standard tier does not include (the extended MedMNIST
set, the breast sparsity frontier, an earlier gate study). They remain valid records of
what was run, but they were produced **before** the second bug sweep, so any figure
involving the representation ensemble should be treated as superseded, and the rest as
historical rather than current. None of them contains an `rpm_mlp` arm, so none is
affected by the sec.20 correction.

| file group | produced | why kept |
|---|---|---|
| `medmnist2_*`, `MEDMNIST2_SUMMARY.md` | 2026-07-27 | extended MedMNIST sets, full tier only |
| `lam_frontier_medmnist-breast.*` | 2026-07-27 | the breast sparsity frontier, full tier only |
| `gate_region_*`, `GATE_REGION_SUMMARY.md` | 2026-07-26 | the region-size gate study behind the L0 default |
| `recovery_*.png`, `score_coverage_frontier.png` | 2026-07-26 | superseded by the prefixed `synthetic_*` figures |
| `boot_*`, `BOOT_SUMMARY.md` | 2026-08-03 | the bootstrap convergence study (B sweep) |
| `showcase_derma_*`, `mask_medmnist-derma.png` | 2026-08-03 | the registered-prediction showcase, FINDINGS sec.10 |

## What changed when the suite was re-run after the bug sweep

Verified by comparing against `results_pre_sweep/` (not tracked in git):

- **MedMNIST: 14 of 14 shared dataset and method results reproduced** within tolerance.
- **Synthetic: `rpm`, `anova`, `full`, `rf` and `rfe` were identical to three decimals.**
  The sweep rewrote pooling, standardisation and `PatchStats` variance, and changed none
  of the science.
- **`lasso` differed** because its solver had been capped from an unconverged liblinear
  search to saga before the sweep, not because of the sweep.
- **mnist differed** because the archived file was a partial earlier run (one dataset,
  seven rows) against the standard tier's two datasets at `--n-max 4000`.
- **The representation ablation genuinely changed**, and it was the ensemble seed
  collision fixed in the sweep that caused it. Marginalised stability moved from 0.960 to
  0.890 on synthetic while every single-lens figure stayed identical, which is what
  isolates the cause. `FINDINGS.md` section 9 has been corrected, and the previously
  published stronger claim withdrawn.

## Timings

Every `seconds` column in the files above was measured on a machine running other jobs,
and is usable only for ordering methods within one run, never as an absolute cost.
Dedicated timings come from `run_timing.py`, which refuses to measure on a busy machine
and records the machine conditions beside every number.

| file | what it is |
|---|---|
| `timing_results.csv`, `TIMING_SUMMARY.md`, `timing_methods.png`, `timing_scaling.png` | 2026-09-06, the fit-time scaling study and the head-to-head. FINDINGS sec.13 |
| `timing_machine.json` | the machine conditions recorded beside that run |
| `TIMING_PROVISIONAL.md` | why the above is stamped `clean_run: false` and is **not** quotable |
| `await_idle.out` | the raw 40-hour poll, 238 samples from 2026-09-03 to 2026-09-05, that never found a quiet machine. The evidence behind `TIMING_PROVISIONAL.md` |

## Raw run logs

Console logs are not tracked: they are 879 KB of transcript whose substance is already
in the CSVs. Three are exceptions, kept because they are the only record of numbers
`FINDINGS.md` quotes, and a citation a reader cannot open is not a citation.

| file | what it is |
|---|---|
| `tv_pneumonia.log` | the total-variation sweep behind sec.7, showing the degeneracy at tv >= 0.05 |
| `pi_diagnostic.log` | the `pi_` histograms behind sec.8, breast against blood |
| `b_sweep_breast.log` | the bootstrap-count sweep behind sec.8, B = 10 to 120 |
