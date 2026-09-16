# Matched-coverage frontier: RPM against the competitors

Every arm is given the same dial and returns exactly the same number of patches, so a difference here is a difference in which patches were chosen. Downstream classifier, folds and features are identical.

## medmnist-blood

| coverage | anova | mi | rf | rpm | best |
|---|---|---|---|---|---|
| 0.05 | 0.7786 | 0.8338 | 0.8279 | 0.8013 | **mi** |
| 0.10 | 0.8543 | 0.8745 | 0.8740 | 0.8791 | **rpm** |
| 0.15 | 0.8762 | 0.8936 | 0.8909 | 0.9032 | **rpm** |
| 0.25 | 0.9111 | 0.9199 | 0.9199 | 0.9199 | **rf** |
| 0.40 | 0.9263 | 0.9263 | 0.9263 | 0.9258 | **rf** |

Wins by coverage level: rpm 2, rf 2, mi 1

## medmnist-breast

| coverage | anova | mi | rf | rpm | best |
|---|---|---|---|---|---|
| 0.05 | 0.7053 | 0.7006 | 0.7069 | 0.6570 | **rf** |
| 0.10 | 0.7070 | 0.7285 | 0.7211 | 0.7033 | **mi** |
| 0.15 | 0.7035 | 0.7392 | 0.7361 | 0.7387 | **mi** |
| 0.25 | 0.7322 | 0.7436 | 0.7305 | 0.7650 | **rpm** |
| 0.40 | 0.7349 | 0.7450 | 0.7376 | 0.7601 | **rpm** |

Wins by coverage level: mi 2, rpm 2, rf 1

## medmnist-pneumonia

| coverage | anova | mi | rf | rpm | best |
|---|---|---|---|---|---|
| 0.05 | 0.8752 | 0.8700 | 0.8716 | 0.6745 | **anova** |
| 0.10 | 0.8782 | 0.8750 | 0.8960 | 0.8869 | **rf** |
| 0.15 | 0.8802 | 0.8802 | 0.9321 | 0.9322 | **rpm** |
| 0.25 | 0.9304 | 0.9495 | 0.9576 | 0.9632 | **rpm** |
| 0.40 | 0.9635 | 0.9637 | 0.9684 | 0.9766 | **rpm** |

Wins by coverage level: rpm 3, anova 1, rf 1

## mnist-3v8

| coverage | anova | mi | rf | rpm | best |
|---|---|---|---|---|---|
| 0.05 | 0.9399 | 0.9399 | 0.9399 | 0.8290 | **rf** |
| 0.10 | 0.9584 | 0.9584 | 0.9598 | 0.9707 | **rpm** |
| 0.15 | 0.9677 | 0.9595 | 0.9714 | 0.9735 | **rpm** |
| 0.25 | 0.9739 | 0.9742 | 0.9767 | 0.9793 | **rpm** |
| 0.40 | 0.9788 | 0.9788 | 0.9811 | 0.9801 | **rf** |

Wins by coverage level: rpm 3, rf 2

## Overall

Operating points won: **rpm** 10, **rf** 6, **mi** 3, **anova** 1

