# Cross-dataset comparison (real scientific image sets)

One row per method per dataset. `score` is AUC (binary) or weighted one-vs-rest AUC (multiclass); `coverage` is the fraction of patches kept; `stability` is the mean pairwise Jaccard of selections across folds; `border_coverage` is the fraction of the outer patch ring retained, a background-rejection diagnostic for centred imagery (lower is better).

## medmnist-blood  (task: multiclass, seeds: 3)

| method | score | coverage | stability | border_coverage |
| --- | --- | --- | --- | --- |
| full | 0.954 | 1.000 | 1.000 | 1.000 |
| anova | 0.954 | 0.973 | 0.975 | 0.944 |
| lasso | 0.954 | 0.997 | 0.995 | 0.994 |
| rfe | 0.954 | 1.000 | 1.000 | 1.000 |
| rf | 0.955 | 0.327 | 1.000 | 0.000 |
| rpm | 0.952 | 0.263 | 0.845 | 0.000 |
| rpm_l0 | 0.955 | 0.378 | 0.837 | 0.006 |
| rpm_boot | 0.955 | 0.384 | 0.799 | 0.006 |
| rpm_ens | 0.955 | 0.559 | 0.804 | 0.136 |
| rf_boot | 0.955 | 0.328 | 0.992 | 0.000 |
| rpm_mlp | 0.933 | 0.552 | 0.758 | 0.500 |

## medmnist-breast  (task: binary, seeds: 3)

| method | score | coverage | stability | border_coverage |
| --- | --- | --- | --- | --- |
| full | 0.845 | 1.000 | 1.000 | 1.000 |
| anova | 0.839 | 0.668 | 0.842 | 0.567 |
| lasso | 0.844 | 0.997 | 0.995 | 0.997 |
| rfe | 0.842 | 0.826 | 0.737 | 0.833 |
| rf | 0.829 | 0.316 | 0.667 | 0.281 |
| rpm | 0.620 | 0.020 | 0.579 | 0.003 |
| rpm_l0 | 0.806 | 0.184 | 0.340 | 0.222 |
| rpm_boot | 0.814 | 0.197 | 0.319 | 0.236 |
| rpm_ens | 0.824 | 0.291 | 0.300 | 0.328 |
| rf_boot | 0.818 | 0.259 | 0.638 | 0.211 |
| rpm_mlp | 0.768 | 0.522 | 0.520 | 0.518 |

## medmnist-derma  (task: multiclass, seeds: 2)

| method | score | coverage | stability | border_coverage |
| --- | --- | --- | --- | --- |
| full | 0.767 | 1.000 | 1.000 | 1.000 |
| anova | 0.767 | 1.000 | 1.000 | 1.000 |
| lasso | 0.765 | 0.547 | 0.641 | 0.517 |
| rfe | 0.767 | 1.000 | 1.000 | 1.000 |
| rf | 0.752 | 0.271 | 0.740 | 0.008 |
| rpm | 0.598 | 0.020 | 1.000 | 0.000 |
| rpm_l0 | 0.754 | 0.412 | 0.473 | 0.421 |
| rpm_boot | 0.765 | 0.804 | 0.748 | 0.796 |
| rpm_ens | 0.768 | 0.990 | 0.982 | 0.988 |

## medmnist-organa  (task: multiclass, seeds: 3)

| method | score | coverage | stability | border_coverage |
| --- | --- | --- | --- | --- |
| full | 0.984 | 1.000 | 1.000 | 1.000 |
| anova | 0.984 | 1.000 | 1.000 | 1.000 |
| lasso | 0.984 | 1.000 | 1.000 | 1.000 |
| rfe | 0.984 | 1.000 | 1.000 | 1.000 |
| rf | 0.978 | 0.388 | 0.834 | 0.333 |
| rpm | 0.980 | 0.453 | 0.749 | 0.547 |
| rpm_l0 | 0.984 | 0.882 | 0.876 | 0.917 |
| rpm_boot | 0.984 | 0.956 | 0.967 | 0.964 |
| rpm_ens | 0.984 | 0.990 | 0.982 | 0.994 |
| rf_boot | 0.977 | 0.363 | 0.781 | 0.319 |
| rpm_mlp | 0.922 | 0.087 | 0.646 | 0.133 |

## medmnist-pneumonia  (task: binary, seeds: 3)

| method | score | coverage | stability | border_coverage |
| --- | --- | --- | --- | --- |
| full | 0.977 | 1.000 | 1.000 | 1.000 |
| anova | 0.977 | 0.903 | 0.982 | 0.844 |
| lasso | 0.977 | 1.000 | 1.000 | 1.000 |
| rfe | 0.978 | 0.706 | 0.869 | 0.589 |
| rf | 0.971 | 0.260 | 0.964 | 0.117 |
| rpm | 0.935 | 0.079 | 0.478 | 0.078 |
| rpm_l0 | 0.975 | 0.384 | 0.818 | 0.292 |
| rpm_boot | 0.973 | 0.359 | 0.805 | 0.247 |
| rpm_ens | 0.960 | 0.215 | 0.616 | 0.136 |
| rf_boot | 0.971 | 0.264 | 0.990 | 0.122 |
| rpm_mlp | 0.910 | 0.235 | 0.556 | 0.186 |

## mnist-3v8  (task: binary, seeds: 3)

| method | score | coverage | stability | border_coverage |
| --- | --- | --- | --- | --- |
| full | 0.995 | 1.000 | 1.000 | 1.000 |
| anova | 0.994 | 0.728 | 0.969 | 0.636 |
| lasso | 0.996 | 0.955 | 0.991 | 0.908 |
| rfe | 0.995 | 0.663 | 0.805 | 0.442 |
| rf | 0.988 | 0.207 | 0.921 | 0.000 |
| rpm | 0.972 | 0.080 | 0.967 | 0.000 |
| rpm_l0 | 0.983 | 0.137 | 0.789 | 0.019 |
| rpm_boot | 0.980 | 0.127 | 0.767 | 0.012 |
| rpm_ens | 0.979 | 0.114 | 0.773 | 0.000 |
| rf_boot | 0.987 | 0.196 | 0.948 | 0.000 |
| rpm_mlp | 0.974 | 0.182 | 0.606 | 0.029 |

## mnist-all  (task: multiclass, seeds: 3)

| method | score | coverage | stability | border_coverage |
| --- | --- | --- | --- | --- |
| full | 0.993 | 1.000 | 1.000 | 1.000 |
| anova | 0.993 | 0.950 | 0.979 | 0.897 |
| lasso | 0.993 | 0.999 | 0.997 | 0.997 |
| rfe | 0.993 | 1.000 | 1.000 | 1.000 |
| rf | 0.992 | 0.478 | 0.958 | 0.111 |
| rpm | 0.989 | 0.322 | 0.890 | 0.086 |
| rpm_l0 | 0.993 | 0.627 | 0.949 | 0.300 |
| rpm_boot | 0.993 | 0.659 | 0.944 | 0.347 |
| rpm_ens | 0.993 | 0.656 | 0.959 | 0.344 |
| rf_boot | 0.992 | 0.478 | 0.961 | 0.106 |
| rpm_mlp | 0.925 | 0.105 | 0.717 | 0.000 |

## Averaged over all real datasets

| method | score | coverage | stability | border_coverage |
| --- | --- | --- | --- | --- |
| full | 0.938 | 1.000 | 1.000 | 1.000 |
| anova | 0.932 | 0.872 | 0.953 | 0.822 |
| lasso | 0.933 | 0.959 | 0.969 | 0.951 |
| rfe | 0.933 | 0.887 | 0.902 | 0.850 |
| rf | 0.927 | 0.303 | 0.861 | 0.142 |
| rpm | 0.844 | 0.169 | 0.717 | 0.126 |
| rpm_l0 | 0.916 | 0.422 | 0.684 | 0.346 |
| rpm_boot | 0.926 | 0.354 | 0.696 | 0.230 |
| rpm_ens | 0.931 | 0.523 | 0.763 | 0.390 |
| rf_boot | 0.940 | 0.284 | 0.889 | 0.102 |
| rpm_mlp | 0.902 | 0.317 | 0.624 | 0.260 |
