# Matched-coverage frontier: RPM against the competitors

Every arm is given the same dial and returns exactly the same number of patches, so a difference here is a difference in which patches were chosen. Downstream classifier, folds and features are identical.

## medmnist-blood

| coverage | rf | rpm | rpm_mlp | best |
|---|---|---|---|---|
| 0.05 | 0.8162 | 0.7453 | 0.5321 | **rf** |
| 0.10 | 0.9084 | 0.8824 | 0.5917 | **rf** |
| 0.15 | 0.9328 | 0.9229 | 0.6036 | **rf** |
| 0.25 | 0.9504 | 0.9504 | 0.7994 | **rf** |
| 0.40 | 0.9540 | 0.9531 | 0.8829 | **rf** |

Wins by coverage level: rf 5

## medmnist-breast

| coverage | rf | rpm | rpm_mlp | best |
|---|---|---|---|---|
| 0.05 | 0.6952 | 0.6232 | 0.6388 | **rf** |
| 0.10 | 0.7575 | 0.7594 | 0.7602 | **rpm_mlp** |
| 0.15 | 0.7966 | 0.7810 | 0.7909 | **rf** |
| 0.25 | 0.8163 | 0.8140 | 0.8185 | **rpm_mlp** |
| 0.40 | 0.8176 | 0.8102 | 0.8229 | **rpm_mlp** |

Wins by coverage level: rpm_mlp 3, rf 2

## medmnist-pneumonia

| coverage | rf | rpm | rpm_mlp | best |
|---|---|---|---|---|
| 0.05 | 0.8639 | 0.6649 | 0.7229 | **rf** |
| 0.10 | 0.9149 | 0.9020 | 0.9108 | **rf** |
| 0.15 | 0.9495 | 0.9394 | 0.9405 | **rf** |
| 0.25 | 0.9705 | 0.9679 | 0.9696 | **rf** |
| 0.40 | 0.9749 | 0.9750 | 0.9751 | **rpm_mlp** |

Wins by coverage level: rf 4, rpm_mlp 1

## mnist-3v8

| coverage | rf | rpm | rpm_mlp | best |
|---|---|---|---|---|
| 0.05 | 0.9218 | 0.8971 | 0.9042 | **rf** |
| 0.10 | 0.9664 | 0.9744 | 0.9759 | **rpm_mlp** |
| 0.15 | 0.9825 | 0.9819 | 0.9820 | **rf** |
| 0.25 | 0.9877 | 0.9874 | 0.9907 | **rpm_mlp** |
| 0.40 | 0.9925 | 0.9900 | 0.9929 | **rpm_mlp** |

Wins by coverage level: rpm_mlp 3, rf 2

## Overall

Operating points won: **rf** 13, **rpm_mlp** 7

