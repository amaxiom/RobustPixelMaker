# Real-Data Benchmark Summary (scientific image sets)

Score is AUC (binary) or accuracy-equivalent OVR AUC (multiclass); higher is better. `coverage` is the fraction of patches kept, `stability` the mean pairwise Jaccard of selections across folds, `border_coverage` the fraction of the outer patch ring retained (a background-rejection diagnostic for centred imagery).

## mnist-3v8

| mnist-3v8 | score | coverage | stability | border_coverage | seconds |
| --- | --- | --- | --- | --- | --- |
| full | 0.994 | 1.000 | 1.000 | 1.000 | 62.140 |
| rf | 0.987 | 0.212 | 0.922 | 0.000 | 34.978 |
| rpm_boot | 0.979 | 0.124 | 0.762 | 0.011 | 23.917 |
| rf_boot | 0.986 | 0.199 | 0.953 | 0.000 | 118.880 |

## medmnist-blood

| medmnist-blood | score | coverage | stability | border_coverage | seconds |
| --- | --- | --- | --- | --- | --- |
| full | 0.954 | 1.000 | 1.000 | 1.000 | 45.718 |
| rf | 0.955 | 0.327 | 1.000 | 0.000 | 46.177 |
| rpm_boot | 0.955 | 0.384 | 0.799 | 0.006 | 42.973 |
| rf_boot | 0.955 | 0.328 | 0.992 | 0.000 | 224.512 |

## medmnist-pneumonia

| medmnist-pneumonia | score | coverage | stability | border_coverage | seconds |
| --- | --- | --- | --- | --- | --- |
| full | 0.977 | 1.000 | 1.000 | 1.000 | 16.305 |
| rf | 0.971 | 0.260 | 0.964 | 0.117 | 27.340 |
| rpm_boot | 0.973 | 0.359 | 0.805 | 0.247 | 17.887 |
| rf_boot | 0.971 | 0.264 | 0.990 | 0.122 | 135.192 |

## medmnist-breast

| medmnist-breast | score | coverage | stability | border_coverage | seconds |
| --- | --- | --- | --- | --- | --- |
| full | 0.845 | 1.000 | 1.000 | 1.000 | 4.172 |
| rf | 0.829 | 0.316 | 0.667 | 0.281 | 6.305 |
| rpm_boot | 0.814 | 0.197 | 0.319 | 0.236 | 9.107 |
| rf_boot | 0.818 | 0.259 | 0.638 | 0.211 | 51.111 |

## Figures

![RPM mask on mnist-3v8](mask_mnist-3v8.png)

![RPM mask on medmnist-blood](mask_medmnist-blood.png)

![RPM mask on medmnist-pneumonia](mask_medmnist-pneumonia.png)

![RPM mask on medmnist-breast](mask_medmnist-breast.png)

![Score vs coverage](ablation_wrapper_score_coverage.png)
