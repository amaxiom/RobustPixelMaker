# Timing and scaling results

Measured 2026-09-06 12:25:33, finished after 20.2 minutes.

- Machine: 8 logical cores, CPU 16.5% busy before and 75.9% after, 1 competing python processes at the start.
- Clean run (idleness precondition met): **False**  <- provisional, not quotable
- python 3.9.10, numpy 2.0.2, scipy 1.13.1, scikit-learn 1.6.1.
- Each figure is the best of 3 repeats. Largest median-to-minimum spread seen: x1.78 (wide, evidence of interference during the run).

## Method comparison at n=400, 32x32, patch 4

| method | best (s) | median (s) | spread |
|---|---|---|---|
| anova | 0.001 | 0.001 | x1.11 |
| lasso | 0.009 | 0.012 | x1.31 |
| single | 0.093 | 0.094 | x1.01 |
| rf | 0.877 | 0.883 | x1.01 |
| bootstrap | 1.011 | 1.021 | x1.01 |
| ensemble | 3.701 | 3.747 | x1.01 |

## Scaling

Fit time against each axis, all methods, in `timing_scaling.png`. Raw numbers in `timing_results.csv`; machine conditions in `timing_machine.json`.

- **n_samples**: 100 to 1600, fit time 0.05s to 11.23s.
- **image_side**: 32 to 256, fit time 0.07s to 168.80s.
- **patch**: 2 to 16, fit time 0.07s to 24.82s.
- **n_bootstrap**: 5 to 40, fit time 0.55s to 3.85s.
- **n_representations**: 2 to 8, fit time 1.78s to 9.55s.
