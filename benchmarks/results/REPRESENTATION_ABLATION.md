# Representation-axis ablation (Experiment 1)

The make-or-break test for the novelty claim. `single:<lens>` fixes one representation and bootstraps over data only (the prior-art configuration); `marginalised` is the double bootstrap over data AND representations. The quantity the claim is about is **stability**, the mean pairwise Jaccard of the selected region across outer folds. `rep_agreement` is how much the individual lenses agreed with each other: a low value means a single-lens answer was lens-specific.

## medmnist-breast

| config | score | coverage | stability | recovery F1 |
| --- | --- | --- | --- | --- |
| marginalised | 0.8110 | 0.255 | 0.328 | n/a |
| single:blur1 | 0.8138 | 0.224 | 0.291 | n/a |
| single:mean | 0.8202 | 0.239 | 0.287 | n/a |
| single:randconv1000 | 0.8273 | 0.629 | 0.531 | n/a |
| single:randconv1001 | 0.8463 | 0.731 | 0.634 | n/a |
| single:stats | 0.8290 | 0.453 | 0.447 | n/a |

Best single lens stability 0.634, mean single lens 0.438, marginalised **0.328** (delta vs mean single -0.110, vs best single -0.306).
Lens agreement: 0.367

## synthetic

| config | score | coverage | stability | recovery F1 |
| --- | --- | --- | --- | --- |
| marginalised | 0.9514 | 0.067 | 0.890 | 0.967 |
| single:blur1 | 0.9525 | 0.072 | 0.800 | 0.938 |
| single:mean | 0.9515 | 0.066 | 0.923 | 0.978 |
| single:randconv1000 | 0.9279 | 0.237 | 0.534 | 0.409 |
| single:randconv1001 | 0.8790 | 0.200 | 0.431 | 0.378 |
| single:stats | 0.9466 | 0.150 | 0.401 | 0.603 |

Best single lens stability 0.923, mean single lens 0.618, marginalised **0.890** (delta vs mean single +0.272, vs best single -0.033).
Lens agreement: 0.406
