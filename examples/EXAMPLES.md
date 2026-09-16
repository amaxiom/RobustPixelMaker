# RobustPixelMaker examples

Six runnable scripts, each a few minutes on a laptop, each built around one science
result that the method makes possible. Figures land in `examples/output/` (viridis,
no bold, per house style) and each script prints its takeaway. Real datasets are the
cached scientific sets from `benchmarks/datasets.py` (MNIST digits and MedMNIST
biomedical images download once, then re-run offline).

Run any example from the repository root:

```bash
py examples/ex1_ground_truth_recovery.py
```

| # | script | data | the science result |
|---|---|---|---|
| 1 | `ex1_ground_truth_recovery.py` | synthetic control, ground truth known | **Variance is not information.** Two planted regions fluctuate equally strongly; only one drives the label. RPM recovers the causal region exactly (recall 1.00 at 6% coverage) and selects not one patch of the equal-variance decoy. The frequency histogram is perfectly bimodal (ambiguous fraction 0.00), the signature that a reproducible region exists. This control licenses interpreting selected regions on real data. |
| 2 | `ex2_mnist_where_3_differs_from_8.py` | MNIST 3-vs-8 (the VTF paper pair, 13,966 images) | **A domain-correct region from predictive sufficiency alone.** The mask selects the waist band and left flank, exactly where an 8 closes into loops and a 3 stays open, and matches the true difference map it never saw. The blank border and the identical right side are discarded; held-out AUC 0.971 on 12% of the image. |
| 3 | `ex3_blood_cell_foreground.py` | blood-cell microscopy, 8 cell types | **Label-free foreground discovery.** With class labels only, no segmentation masks, the selection localises onto the cell body and discards the plasma at 33% coverage, with the downstream classifier's held-out score unchanged (0.957 selected vs 0.956 all patches). Univariate filters keep 97 to 100% of the image here, that is, they fail to select at all. |
| 4 | `ex4_diffuse_signal_diagnosis.py` | blood microscopy vs chest X-ray pneumonia | **Compact vs diffuse signal, read from the selection's shape.** Compared at matched lean coverage (exactly the top 12% of patches for both, the like-for-like comparison the benchmarks argue for), the compact blood cell yields a perfectly contiguous region (contiguity 1.00) while diffuse pneumonia opacity fragments (contiguity 0.50), matching the radiology: opacity is not an object. The match is now *enforced* with `top_patches`: this example previously requested it with `target_coverage=0.12` and did not get it, comparing blood at 0.245 against pneumonia at 0.122 while describing the comparison as matched. Contiguity is exactly what coverage inflates (pneumonia scores 0.50 at 0.122 and 0.83 at 0.245), so the claim was untested as stated. Enforcing the match leaves the result unchanged (FINDINGS sec.22). Fragmentation is the strong diagnostic; the ambiguity gap is mild here because both tasks are well powered (severe ambiguity is the underpowered signature, example 6's story). A fragmented selection warns against interpreting any single sub-region as "the" finding. This example was rewritten twice because its first two framings were contradicted by their own figures; the version here reports what is measured. |
| 5 | `ex5_representation_marginalisation.py` | noisy synthetic control | **The representation is a nuisance variable; integrate it out.** All four lenses FIND the planted region (recall 1.00 each), so recall does not separate them; what separates them is precision, which runs from 0.36 for the random-convolution lens to 1.00 for the plain mean, with pairwise agreement only 0.68. Marginalising over the lenses achieves precision 1.00 at 6% coverage, matching the best single lens without knowing in advance which one that would be. This is the method's central claim, shown against ground truth. (The example originally reported recall, which is identical across lenses and so demonstrated nothing; it now reports the axis that actually differs.) |
| 6 | `ex6_underpowered_study_honesty.py` | breast ultrasound, all 780 images | **An underpowered study should not yield a confident region, and does not.** At n=780 the frequency map is diffuse (ambiguous fraction 0.57, against 0.14 for the well-powered compact case, both at B=30), and more resampling does not change the verdict. The diffuse map is the correct scientific answer at this sample size: a test for whether a reproducible region *exists*, not merely a method for producing one. |

Reading order: 1 establishes trust on ground truth, 2 and 3 are the positive results
on real images, 4 and 6 are the diagnostic results (the method telling you what the
data can and cannot support, which is the reproducibility story), and 5 is the
methodological core.

The failed predictions and negative results behind examples 4 and 6 are documented,
deliberately, in `benchmarks/FINDINGS.md` sections 6 to 10.
