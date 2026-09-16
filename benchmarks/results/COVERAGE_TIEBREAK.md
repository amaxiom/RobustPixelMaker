# Matched-coverage frontier: RPM against the competitors

Every arm is given the same dial and returns exactly the same number of patches, so a difference here is a difference in which patches were chosen. Downstream classifier, folds and features are identical.

## medmnist-pneumonia

| coverage | rf | rpm | rpm_pi | best |
|---|---|---|---|---|
| 0.05 | 0.8639 | 0.6649 | 0.6620 | **rf** |
| 0.10 | 0.9149 | 0.9020 | 0.8922 | **rf** |
| 0.15 | 0.9495 | 0.9394 | 0.9356 | **rf** |
| 0.25 | 0.9705 | 0.9679 | 0.9665 | **rf** |
| 0.40 | 0.9749 | 0.9750 | 0.9752 | **rpm_pi** |

Wins by coverage level: rf 4, rpm_pi 1

## mnist-3v8

| coverage | rf | rpm | rpm_pi | best |
|---|---|---|---|---|
| 0.05 | 0.9218 | 0.8971 | 0.7871 | **rf** |
| 0.10 | 0.9664 | 0.9744 | 0.9744 | **rpm** |
| 0.15 | 0.9825 | 0.9819 | 0.9796 | **rf** |
| 0.25 | 0.9877 | 0.9874 | 0.9874 | **rf** |
| 0.40 | 0.9925 | 0.9900 | 0.9901 | **rf** |

Wins by coverage level: rf 4, rpm 1

## Overall

Operating points won: **rf** 8, **rpm_pi** 1, **rpm** 1

