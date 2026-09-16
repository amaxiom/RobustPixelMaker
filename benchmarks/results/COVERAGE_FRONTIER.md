# Matched-coverage frontier: RPM against the competitors

Every arm is given the same dial and returns exactly the same number of patches, so a difference here is a difference in which patches were chosen. Downstream classifier, folds and features are identical.

## medmnist-blood

| coverage | anova | mi | rf | rpm | best |
|---|---|---|---|---|---|
| 0.05 | 0.7565 | 0.8220 | 0.8162 | 0.7674 | **mi** |
| 0.10 | 0.8963 | 0.9105 | 0.9084 | 0.9016 | **mi** |
| 0.15 | 0.9272 | 0.9325 | 0.9328 | 0.9286 | **rf** |
| 0.25 | 0.9455 | 0.9512 | 0.9504 | 0.9481 | **mi** |
| 0.40 | 0.9540 | 0.9540 | 0.9540 | 0.9531 | **rf** |

Wins by coverage level: mi 3, rf 2

## medmnist-breast

| coverage | anova | mi | rf | rpm | best |
|---|---|---|---|---|---|
| 0.05 | 0.6931 | 0.6755 | 0.6952 | 0.6232 | **rf** |
| 0.10 | 0.7503 | 0.7746 | 0.7575 | 0.7574 | **mi** |
| 0.15 | 0.7805 | 0.7984 | 0.7966 | 0.7846 | **mi** |
| 0.25 | 0.7997 | 0.8060 | 0.8163 | 0.8112 | **rf** |
| 0.40 | 0.8101 | 0.8207 | 0.8176 | 0.8097 | **mi** |

Wins by coverage level: mi 3, rf 2

## medmnist-pneumonia

| coverage | anova | mi | rf | rpm | best |
|---|---|---|---|---|---|
| 0.05 | 0.8656 | 0.8642 | 0.8639 | 0.6620 | **anova** |
| 0.10 | 0.8997 | 0.9071 | 0.9149 | 0.8922 | **rf** |
| 0.15 | 0.9187 | 0.9187 | 0.9495 | 0.9356 | **rf** |
| 0.25 | 0.9515 | 0.9655 | 0.9705 | 0.9665 | **rf** |
| 0.40 | 0.9736 | 0.9733 | 0.9749 | 0.9752 | **rpm** |

Wins by coverage level: rf 3, anova 1, rpm 1

## mnist-3v8

| coverage | anova | mi | rf | rpm | best |
|---|---|---|---|---|---|
| 0.05 | 0.9218 | 0.9218 | 0.9218 | 0.7871 | **rf** |
| 0.10 | 0.9644 | 0.9644 | 0.9664 | 0.9744 | **rpm** |
| 0.15 | 0.9786 | 0.9707 | 0.9825 | 0.9796 | **rf** |
| 0.25 | 0.9836 | 0.9836 | 0.9877 | 0.9874 | **rf** |
| 0.40 | 0.9879 | 0.9880 | 0.9925 | 0.9901 | **rf** |

Wins by coverage level: rf 4, rpm 1

## Overall

Operating points won: **rf** 11, **mi** 6, **rpm** 2, **anova** 1

