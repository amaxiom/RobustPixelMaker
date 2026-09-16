# Benchmark findings

Consolidated evidence from the synthetic sweeps, the gate ablation, the sparsity
frontiers, and seven real scientific datasets (MNIST 3-vs-8, and MedMNIST chest X-ray,
blood microscopy, abdominal CT, breast ultrasound, retinal OCT and dermatoscopy). Written to be reusable as the
empirical spine of the paper. Every number here comes from a script in this directory
and a CSV in `results/`.

Negative results are included deliberately and are not softened, and they outnumber
the positive ones. Sections 6 to 13 record where RPM is beaten, where a fixed sparsity
strength fails in both directions, where a predicted fix did not work, and a prediction
registered in advance that both of its datasets then falsified. Sections 14 to 23 are
harder: each is a defect in this suite's own protocol, found after it had already
published a number. They include a confound in the comparison itself (sec.14), the
ablation showing the learned mask adds nothing over random-forest importance (sec.15),
a hyperparameter search hidden inside the stability aggregation (sec.18), an arm
published at coverage 1.000 because it had selected nothing (sec.20), and a request for
a coverage that could not be met, reported as though it had been (sec.22).

One hazard family runs through them and has now been recorded five times: **a metric
that rewards not selecting will be won by a method that does not select.** Knowing
about it turned out not to be protection; the protection is a column in the table.

Last updated: 2026-09-16.

---

## 1. Synthetic sweep: RPM leads on recovery, leanness and stability

23 scenarios (sample size, image shape, noise, signal, task, region size, distractor
count) x 6 methods x 3 seeds = 414 evaluations. Means over all scenarios:

| method | recovery F1 | precision | distractor_fp | coverage | stability | AUC (binary) |
|---|---|---|---|---|---|---|
| full | 0.12 | 0.06 | 0.96 | 1.00 | 1.00 | 0.921 |
| lasso | 0.42 | 0.32 | 0.21 | 0.33 | 0.45 | 0.942 |
| rf | 0.74 | 0.61 | 0.03 | 0.12 | 0.59 | 0.955 |
| anova | 0.74 | 0.60 | 0.05 | 0.11 | 0.60 | 0.957 |
| rfe | 0.82 | 0.77 | 0.005 | 0.13 | 0.68 | 0.957 |
| **rpm** | **0.98** | **1.00** | **0.00** | **0.06** | **0.98** | 0.955 |

Reading: RPM recovers the planted region almost exactly, never selects a single patch
of the equal-variance distractor, keeps half as much of the image as the nearest
competitor, and is the only method whose selection is close to reproducible across
folds. Its predictive score sits inside the noise band of the best selector. This is
the RobustModelMaker story reproduced in image space: competitive on score, dominant
on the joint recovery, leanness and stability frontier.

Source: `run_synthetic_benchmark.py`, `results/synthetic_results.csv`.

## 2. Gate ablation: Hard-Concrete L0 fixes the large-region failure

The sweep above exposed one genuine weakness: with a large informative region (nine
independent cells) the L1 sigmoid gate over-prunes. Re-running that axis with both
gates (`run_synthetic_benchmark.py --axes region_side`):

| region_side=3 | recovery F1 | score | coverage | stability |
|---|---|---|---|---|
| anova | 0.88 | 0.918 | 0.18 | 0.76 |
| rfe | 0.73 | 0.910 | 0.29 | 0.57 |
| rpm (sigmoid, L1) | 0.80 | 0.836 | 0.10 | 0.73 |
| **rpm_l0 (Hard-Concrete)** | **0.99** | **0.922** | 0.14 | **0.97** |

At `region_side` 1 and 2 both gates are perfect (F1 1.00), so L0 costs nothing where
L1 already worked. The L1 gate is also strongly seed-dependent on the large region
(F1 0.61 to 0.98 across three seeds) whereas L0 is consistent (0.98 to 1.00).

Mechanism: L1 penalises gate MAGNITUDE, so it shrinks the gates it keeps and drops
interior cells of a genuinely large region. Hard-Concrete penalises the expected COUNT
of open gates, so a large region survives intact.

### The L0 gate also fixes the high-noise regime

The same ablation over the noise axis shows the second weakness closing too. Recovery
F1, then held-out score, then stability:

| noise | anova | rfe | rf | rpm (L1) | rpm_l0 |
|---|---|---|---|---|---|
| F1 at 0.5 | 0.76 | 0.72 | 0.76 | **1.00** | **1.00** |
| F1 at 1.0 | 0.76 | 0.79 | 0.74 | **1.00** | 0.99 |
| F1 at 2.0 | 0.76 | 0.78 | 0.74 | **1.00** | 0.93 |
| F1 at 4.0 | 0.76 | **0.91** | 0.61 | 0.84 | 0.89 |
| score at 4.0 | 0.898 | 0.901 | 0.900 | **0.827** | 0.900 |
| stability at 4.0 | 0.61 | 0.79 | 0.48 | 0.71 | **0.81** |

At low and moderate noise the L1 gate is perfect (F1 1.00) and slightly ahead of L0.
At extreme noise it **collapses**: its score falls to 0.827, the worst of any method
tested, while every competitor sits near 0.900. The L0 gate degrades gracefully
instead, holding score at 0.900 (matching all competitors) with the highest stability
of any method (0.81).

So Hard-Concrete closes **both** weaknesses the first sweep exposed, the large region
and the high noise, and its only cost is a small F1 concession at moderate noise where
L1 happens to be exactly perfect. Graceful degradation beats peak performance for a
default, which is why `gate="hardconcrete"` is now the default.

Across the whole noise axis RPM's margins over the competitors remain large: recovery
F1 around 1.00 versus 0.72 to 0.79, and stability 0.97 to 1.00 versus 0.52 to 0.75, at
noise levels up to 2.0.

Source: `results/gate_region_results.csv`, `results/gate_ablation_results.csv`.

## 3. Sparsity frontier: report the curve, never a single lam

A fixed `lam` conflates "this method loses score" with "this operating point was too
aggressive". Sweeping `lam` on MNIST 3-vs-8 (n=1500, 3 folds):

| coverage | rpm sigmoid | rpm L0 | reference |
|---|---|---|---|
| 0.03 to 0.06 | 0.956 | 0.947 | |
| 0.11 to 0.17 | 0.977 | 0.980 | |
| 0.22 to 0.31 | 0.982 | 0.984 | rf 0.984 at 0.22 |
| 0.46 to 0.52 | 0.989 | 0.989 | |
| 0.63 to 0.69 | | 0.989 | anova 0.988 at 0.63, rfe 0.990 at 0.65 |
| 1.00 | | | full 0.991 |

Two conclusions, one of which corrects an earlier reading:

1. **RPM's apparent score loss on real data was an operating-point artefact.** At
   `lam=0.001` RPM reaches AUC 0.989 using 46% of patches, beating ANOVA (0.988 at
   63%) with a third fewer features and within 0.001 of RFECV (0.990 at 65%) with 29%
   fewer. At the lean end it still holds 0.977 on 11% of the image.
2. **The two gates trace essentially the same frontier on MNIST.** An earlier
   single-`lam` comparison appeared to show L0 winning, but that was only because at
   the same `lam` the two gates sit at different coverages (0.06 versus 0.17). The
   real L0 advantage is the large-region regime in section 2, not MNIST.

Methodological lesson for the paper: always report the score against coverage
frontier. A single sparsity strength can make any selector look arbitrarily good or
bad.

Source: `run_lam_frontier.py`, `results/lam_frontier_mnist-3v8.csv`.

## 4. First real dataset: MNIST 3-vs-8 (13,966 images)

The 3-vs-8 subset is the hard pair used in the VTF paper, and the image count matches
it exactly, so the comparison is like for like.

| method | AUC | coverage | stability | border coverage |
|---|---|---|---|---|
| full | 0.991 | 1.00 | 1.00 | 1.00 |
| lasso | 0.991 | 0.65 | 0.75 | 0.51 |
| rfe | 0.990 | 0.54 | 0.71 | 0.29 |
| anova | 0.988 | 0.63 | 0.92 | 0.54 |
| rf | 0.981 | 0.20 | 0.76 | 0.00 |
| rpm_l0 | 0.984 | 0.18 | 0.74 | 0.05 |
| rpm (sigmoid) | 0.956 | 0.07 | 0.85 | 0.00 |

`border coverage` is the fraction of the outer patch ring retained, a background
rejection diagnostic for centred imagery. RPM keeps **none** of the blank MNIST
border; ANOVA and Lasso keep about half of it. Combined with the frontier in section
3, the real-data picture is: RPM matches the best selectors on score when placed at a
comparable operating point, is far leaner, and is uniquely clean at discarding
uninformative background.

**The selected region is domain-correct, which is the strongest qualitative evidence
so far** (`results/mask_mnist-3v8.png`). At 14% coverage the retained patches are the
middle horizontal band and the cells immediately to its left, and nothing else. That
is exactly where a 3 and an 8 differ: an 8 closes into a loop on the left of the waist
whereas a 3 stays open. The mask discards the entire border, the upper and lower
extremities, and the right-hand side, all of which are near-identical between the two
digits and therefore carry no discriminative information. Nothing in the method knows
about digits; the region was found by predictive sufficiency alone. This is the
image-space analogue of RobustModelMaker recovering clinically recognised biomarkers
on the PLCO panel, and it is the figure to lead with.

Source: `run_real_benchmark.py`, `results/real_results.csv`, `results/mask_mnist-3v8.png`.

## 5. Second and third real datasets: MedMNIST chest X-ray and blood microscopy

MedMNIST v2 supplies standardised 28x28 biomedical images, so the patch geometry and
every metric carry over from MNIST unchanged and the results are directly comparable.
Two sets were run at n=3000, 3 seeds: **pneumonia** (paediatric chest X-ray, binary)
and **blood** (blood cell microscopy, 8 classes).

Averaged over all three real datasets:

| method | score | coverage | stability | border kept |
|---|---|---|---|---|
| full | 0.969 | 1.00 | 1.00 | 1.00 |
| anova | 0.969 | 0.89 | 0.97 | 0.84 |
| lasso | 0.969 | 0.95 | 0.96 | 0.93 |
| rfe | 0.970 | 0.81 | 0.90 | 0.72 |
| rf | 0.965 | 0.28 | 0.95 | 0.05 |
| **rpm_l0** | **0.968** | 0.35 | 0.82 | 0.14 |
| rpm (sigmoid) | 0.945 | 0.16 | 0.69 | 0.03 |

**The headline is about the competitors, not about RPM.** On real scientific images the
classical selectors barely reduce at all: ANOVA keeps 89%, Lasso 95%, RFECV 81%, and on
blood microscopy they keep 97 to 100%, that is, effectively no selection. They also
retain most of the uninformative frame. Only the two importance-driven methods reduce,
and RPM matches the full-feature score (0.968 versus 0.969) on about a third of the
image.

**Random Forest is a genuinely strong competitor and this must be stated plainly.** It
is slightly leaner (28% versus 35%) and more stable (0.95 versus 0.82) than RPM on
average. RPM beats it on score on two of three datasets (MNIST 0.984 versus 0.981,
pneumonia 0.975 versus 0.971) and ties on blood. RPM's differentiators on real data are
the tunable frontier and the coherence of the selected region, not raw dominance.

### Where the coherent-region claim holds, and where it does not

`compare_selection_maps.py` plots which patches each method keeps and scores contiguity
(the fraction of selected patches with a selected neighbour). Contiguity is only
meaningful at comparable coverage, since keeping everything is trivially contiguous.

| blood microscopy | coverage | contiguity |
|---|---|---|
| anova | 0.98 | 1.00 (trivial, keeps everything) |
| rfe | 1.00 | 1.00 (trivial) |
| rf | 0.33 | 1.00 |
| rpm (sigmoid) | **0.24** | **1.00** |
| rpm_l0 | 0.41 | 0.95 |

| chest X-ray | coverage | contiguity |
|---|---|---|
| rf | 0.27 | 0.77 |
| rpm (sigmoid) | 0.08 | **0.00** |
| rpm_l0 | 0.39 | 0.74 |

On blood microscopy the mask localises tightly onto the cell body and discards all
plasma, at 24% coverage and perfect contiguity, which is the cleanest interpretable
result in the suite. On chest X-ray the selection fragments; the sigmoid gate keeps
four isolated patches with contiguity 0.00. This is scientifically plausible, since
pneumonia presents as diffuse opacity across the lung fields rather than a localised
object, but it bounds the claim: **the coherent-region advantage holds where the signal
is a region and degrades where it is diffuse.** Report it that way rather than as a
universal property.

Source: `run_real_benchmark.py`, `compare_real.py`, `compare_selection_maps.py`,
`results/medmnist_results.csv`, `results/CROSS_DATASET.md`,
`results/selection_maps_medmnist-*.png`.

## 6. Fourth and fifth real datasets, and the clearest negative result so far

Two datasets chosen to probe new regimes: **organa** (abdominal CT, 11 anatomical
classes, n=2992), the closest available proxy for the registered brain-MRI target, and
**breast** (breast ultrasound, n=780), the small-sample regime that motivates the whole
reproducibility argument.

| organa (11-class CT) | score | coverage | stability | border |
|---|---|---|---|---|
| full / anova / lasso / rfe | 0.984 | **1.00** | 1.00 | 1.00 |
| rf | 0.978 | 0.39 | 0.84 | 0.33 |
| rpm (sigmoid) | 0.980 | 0.45 | 0.75 | 0.55 |
| rpm_l0 | 0.984 | **0.88** | 0.88 | 0.92 |

| breast (n=780, small sample) | score | coverage | stability | border |
|---|---|---|---|---|
| full | 0.845 | 1.00 | 1.00 | 1.00 |
| anova | 0.839 | 0.67 | 0.84 | 0.56 |
| lasso | 0.844 | 1.00 | 0.99 | 1.00 |
| rfe | 0.842 | 0.83 | 0.74 | 0.84 |
| rf | 0.829 | 0.31 | 0.67 | 0.28 |
| rpm (sigmoid) | **0.620** | 0.02 | 0.55 | 0.00 |
| rpm_l0 | 0.806 | 0.19 | 0.34 | 0.22 |

Two failures, in **opposite directions, at the same lam**. On organa the L0 gate barely
selects at all (88% coverage); on breast it over-prunes badly, and the sigmoid gate
collapses to a single patch and near-chance score (0.620 against a 0.845 baseline).
On the four classical selectors organa is also a total non-reduction: ANOVA, Lasso and
RFECV all keep 100%.

### The frontier explains it, and mostly rehabilitates RPM

Sweeping lam on breast (n=780, 5 folds):

| lam | rpm_l0 score | coverage | stability | rpm sigmoid score | coverage |
|---|---|---|---|---|---|
| 0.001 | 0.839 | 0.88 | 0.82 | 0.828 | 0.43 |
| 0.003 | **0.841** | 0.77 | 0.71 | 0.798 | 0.19 |
| 0.01 | 0.823 | 0.55 | 0.54 | 0.730 | 0.06 |
| 0.03 | 0.796 | 0.20 | 0.29 | 0.608 | 0.02 |
| 0.1 | 0.574 | 0.01 | 0.40 | 0.500 | **0.00** |

references: full 0.838, anova 0.838 at 0.67, rfe 0.838 at 0.89, rf 0.822 at 0.32.

At `lam=0.003` the L0 gate reaches **0.841, the best score of any method tested,
including the full-feature baseline**. So the catastrophic fixed-lam numbers are
largely an operating-point artefact, as on MNIST.

**But two things must be conceded honestly.** First, at matched leanness RPM does not
beat Random Forest here: interpolating the L0 curve to RF's 32% coverage gives roughly
0.81 against RF's 0.822. Second, and more seriously, **RPM's stability degrades sharply
as sparsity increases on small samples** (0.82 down to 0.29). Selection instability
under small n is precisely the failure that bootstrap stability selection exists to
correct, so this is the strongest motivation yet for Milestone 4, and until that lands
the small-sample regime is a genuine weakness rather than a tuning detail.

### The design conclusion: lam must be tuned per dataset

A single global lam is simply wrong, and the evidence now points both ways: it
under-prunes on organa and over-prunes on breast. This mirrors the parent framework,
which tunes its stability threshold per dataset (tau = 0.60, 0.80 and 0.75 on its three
benchmarks) rather than fixing one value. A coverage-targeting or lam-selection
procedure therefore has to become a first-class library feature, not a benchmark
script. Every cross-dataset average in this document that uses a fixed lam understates
RPM for exactly this reason and should be read per dataset.

Source: `results/medmnist2_results.csv`, `results/lam_frontier_medmnist-breast.csv`,
`results/CROSS_DATASET.md`.

### A metric hazard this exposed

At `lam=0.1` the sigmoid gate on breast selects **nothing**, and the stability metric
reports **1.00**, because the Jaccard of two empty sets is defined as 1. An empty or
near-empty selection therefore looks perfectly reproducible. Stability must never be
read without coverage beside it, and a future version should report stability as
undefined below a minimum selection size. This is the same reporting hazard as the TV
degeneracy in the next section: a metric looking perfect because no selection happened.

## 7. The spatial smoothness prior works, but has a sharp degeneracy

Testing the TV prior on the fragmented chest X-ray case (Hard-Concrete gate, lam=0.03,
5 folds):

| tv | score | coverage | contiguity | stability |
|---|---|---|---|---|
| 0.0 | 0.9753 | 0.39 | 0.76 | 0.92 |
| 0.01 | 0.9750 | 0.52 | **0.96** | 0.84 |
| 0.05 | 0.9770 | **1.00** | 1.00 | 1.00 |
| 0.2 | 0.9770 | **1.00** | 1.00 | 1.00 |
| 1.0 | 0.9770 | **1.00** | 1.00 | 1.00 |

At `tv=0.01` coherence rises from 0.76 to 0.96 for no score cost, at the price of 13
percentage points of coverage: a genuine and usable interpretability gain.

At `tv>=0.05` the prior **defeats selection entirely**. The smoothest possible mask is a
constant one, and a constant-on mask fits the data better than constant-off, so every
gate opens and coverage goes to 1.00. The apparent perfect score and stability there
are an artefact of doing no selection at all, not a success.

Consequences: the default stays `tv=0`, any TV value must be tuned within a narrow
bounded range, and coverage must always be reported alongside contiguity so this
degeneracy cannot be mistaken for a good result. A future version should normalise the
TV term against the sparsity term so the useful range does not depend on patch count.

Source: `results/tv_pneumonia.log`.

## 8. Milestone 4: bootstrap aggregation did NOT fix the small-sample collapse

Section 6 identified small-sample stability as the top priority and predicted that
bootstrap stability selection would fix it. **It did not, and that prediction was
wrong.** On breast ultrasound (n=780, 3 seeds, B=10, tau=0.6):

| breast | score | coverage | stability |
|---|---|---|---|
| rf | 0.829 | 0.31 | **0.67** |
| rpm_l0 (single fit) | 0.803 | 0.19 | 0.30 |
| rpm_boot (B=10) | 0.814 | 0.20 | **0.32** |

Score improved slightly, stability essentially did not move. Random Forest remains
clearly more stable on this dataset. The same holds on the larger chest X-ray set,
where bootstrap is if anything marginally worse and costs roughly twice the time:

| pneumonia (n=3000) | score | coverage | stability | seconds |
|---|---|---|---|---|
| rf | 0.971 | 0.26 | **0.964** | 47 |
| rpm_l0 (single fit) | 0.975 | 0.38 | 0.818 | 23 |
| rpm_boot (B=10) | 0.973 | 0.36 | 0.805 | 50 |

So at B=10 the aggregation buys nothing on either real dataset while doubling the
cost. That is a clear negative result and is reported as such.

### Why: the selection-frequency distribution says no region exists

The diagnostic that explains it is the shape of `pi_`, run at B=40:

| dataset | pi histogram, bins 0 to 1 | ambiguous fraction (0.2 to 0.8) | shape |
|---|---|---|---|
| blood, n=3000 | [22, 4, 3, 1, 4, 15] | **0.16** | strongly bimodal |
| breast, n=780, lam=0.03 | [5, 13, 21, 6, 2, 2] | **0.59** | diffuse, no separation |
| breast, n=780, lam=0.003 | [0, 0, 0, 1, 25, 23] | 0.53 | all high, no separation |

On blood microscopy `pi_` is **bimodal**: 22 patches are essentially never selected and
15 essentially always, so a threshold separates them cleanly and the selection is
reproducible. On breast, `pi_` is **unimodal and centred in the undecided band**, with
only 4% of patches above 0.8 at the lean operating point, and no separation at the
permissive one either. **No patch subset reproducibly carries the signal, so no
threshold can produce a stable selection and aggregation cannot manufacture one.**

This is the honest and, on reflection, the more valuable reading. Bootstrap aggregation
is not a stability generator; it is a stability *estimator*. When the data cannot
support a reproducible region it reports that fact instead of inventing one, which is
precisely the behaviour the reproducibility literature asks for: an underpowered
imaging study should not yield a confident region. The parent framework makes the same
observation, describing bimodality as the structural signature stability selection
exists to expose.

### What shipped as a result

`ambiguous_fraction()` and `stability_report()` on `BootstrapMaskSelector` turn this
into a first-class go/no-go diagnostic. A low ambiguous fraction means the selected
region is trustworthy as a finding; a high one means the dataset does not support one
at this sample size, whatever the score says. This should be reported beside every
selected region in the paper, and it is arguably a contribution in its own right:
**a test for whether a reproducible region exists, not merely a method for finding
one.**

### The obvious rescue was tested and does not work

The natural objection is that B=10 is simply too few: at pi near 0.5 the frequency
estimate then has a standard error around 0.15, so the threshold crossing is noisy, and
the parent framework uses B=100 in production. Sweeping B on breast (5 folds):

| B | score | coverage | stability | ambiguous fraction | seconds |
|---|---|---|---|---|---|
| 10 | 0.809 | 0.17 | 0.29 | 0.64 | 20 |
| 30 | 0.785 | 0.12 | 0.34 | 0.60 | 39 |
| 60 | 0.775 | 0.12 | 0.34 | 0.59 | 74 |
| 120 | 0.766 | 0.12 | **0.36** | 0.62 | 141 |

**Stability plateaus by B=30.** Twelve times the compute buys +0.07 stability and
leaves it at half of Random Forest's 0.67, while score actually declines (0.809 to
0.766) as the tightening threshold trims coverage. Crucially the ambiguous fraction
never moves off roughly 0.6 at any B.

That settles the diagnosis: **the frequency estimate converges quickly, and converges
to a diffuse distribution. The instability is a property of the data, not of the
estimator.** No amount of resampling creates a reproducible region where none exists.

Two pieces of practical guidance follow. Do not crank B hoping for stability; check
`ambiguous_fraction()` instead, which reaches its verdict at small B and costs
proportionally nothing. And read a low measured stability on a small dataset as
information about the dataset rather than as a defect of the selector.

### Still open

The stability gap against Random Forest on small samples stands as a genuine
limitation, now understood rather than merely observed. Untested remedies are
coverage targeting placed where `pi_` is most bimodal, complementary pairs for tighter
per-fit error control, and the representation ensemble of Milestone 5, which
marginalises over a second axis of variation entirely and is the one mechanism in the
plan not yet brought to bear on this problem.

Source: `results/boot_results.csv`, `results/pi_diagnostic.log`,
`results/b_sweep_breast.log`.

## 9. Milestone 5: the representation ablation, and the claim it supports

Experiment 1, the make-or-break test for the novelty claim. `single:<lens>` fixes one
representation and bootstraps over data only (the prior-art configuration);
`marginalised` is the double bootstrap over data resamples AND representations.
Four lenses: patch mean, patch statistics, a coarser spatial scale, and random
convolutional features.

### The premise holds: the answer really does depend on the lens

Across both datasets the individual lenses agree with each other only at Jaccard
**0.37 to 0.41**. Different representations genuinely select different regions, so a
single-lens result is a lens-specific result. That is the premise the whole
marginalisation argument rests on, and it is not assumed here but measured.

### Synthetic (noise 2.0, known region): marginalisation buys insurance, not a win

| config | score | coverage | stability | recovery F1 |
|---|---|---|---|---|
| single:mean | 0.952 | 0.066 | 0.923 | 0.978 |
| single:blur1 | 0.952 | 0.072 | 0.800 | 0.938 |
| single:stats | 0.947 | 0.150 | 0.401 | 0.603 |
| single:randconv1000 | 0.928 | 0.238 | 0.534 | 0.409 |
| single:randconv1001 | 0.879 | 0.200 | 0.431 | 0.378 |
| **marginalised** | 0.951 | **0.067** | **0.890** | **0.967** |

Marginalising lands just **below** the best single lens on stability (0.890 against
0.923) and on recovery (0.967 against 0.978) at the same coverage, and far **above**
the average single lens (0.618) and the worst (0.401). Coverage and stability are
**negatively** correlated here (r = -0.84), so none of this is a coverage artefact.

The honest reading is insurance, not victory. You cannot know in advance which lens is
the good one; picking at random would have given about 0.62 stability and could have
given 0.40, whereas marginalising delivers within 0.03 of the best lens without having
to choose. That is a weaker claim than "wins", and it is the one the data supports.

**This number was corrected after a bug fix, and the correction matters.** An earlier
run reported marginalised stability 0.960, beating the best single lens. That figure
was inflated by a seed-collision defect in the ensemble path: the seed for each
(resample, lens) fit was the *sum* of the bootstrap and representation stream seeds, so
every pair with the same index sum shared randomness, correlating the gate sampling
across the grid and making the fits look more alike than independent fits would. The
deep sweep gave the ensemble its own seed stream (CHANGELOG.md, 0.1.0).
Re-running with independent seeds moved the synthetic figure to 0.890 and the breast
figure from 0.307 to 0.328, while every single-lens figure stayed identical to three
decimals, which is what isolates the cause to the ensemble path. The previously
published stronger claim should not be cited.

### But the stability metric itself is confounded with coverage

On breast the raw table appears to say the opposite, until the confound is visible:

| config | coverage | raw stability |
|---|---|---|
| single:randconv1001 | **0.731** | 0.634 |
| single:randconv1000 | **0.629** | 0.531 |
| single:stats | 0.453 | 0.447 |
| single:mean | 0.239 | 0.287 |
| single:blur1 | 0.224 | 0.291 |
| marginalised | 0.255 | 0.328 |

Coverage and stability correlate at **r = +0.99**. The apparently most stable
configuration is simply the one retaining three quarters of the image: two near
complete sets overlap heavily whatever they contain. Compared at **matched coverage**
(0.26 against 0.22 and 0.24), marginalisation is the best of the three, though
all are poor. Small-sample instability remains unsolved, consistent with section 8:
the frequency map there is diffuse and no axis of marginalisation manufactures a region
that the data does not support.

This is the fourth appearance of one hazard family: **metrics that reward not
selecting.** The empty set scores Jaccard 1, a large TV weight scores perfectly by
keeping everything, an empty selection reports perfect stability, and now raw Jaccard
rewards high coverage. `adjusted_jaccard` and the `n_total` argument to
`mean_pairwise_jaccard` correct stability against the chance agreement expected for
sets of the given sizes, and should be preferred whenever configurations with different
coverage are compared. Raw Jaccard remains available for continuity with the parent
framework, but every stability figure must be read next to its coverage.

### Verdict on the novelty claim

Partially supported, and the qualification is real. The premise is confirmed by
measurement: lenses agree only at 0.37 to 0.41, so a single-lens answer is a
lens-specific answer. Marginalisation dramatically beats an arbitrary lens (0.890
against an average of 0.618 and a worst case of 0.401) and lands marginally below the
best lens (0.923), at equal coverage. So the defensible claim is that marginalising
removes the risk of choosing badly, not that it outperforms a well-chosen lens. Since
the best lens cannot be identified in advance without the ground truth that real data
does not come with, removing that risk is the practically useful property, but it
should be stated as such.

It does not rescue the small-sample case, and that limitation is understood rather than
merely observed.

Source: `run_representation_ablation.py`, `results/representation_ablation.csv`,
`results/REPRESENTATION_ABLATION.md`.

## 10. Showcase datasets: a prediction registered before running

The five datasets so far were chosen to probe different regimes, and two of them
(breast, organa) fell outside the one RPM is designed for. To test the method where it
should work, two further sets were added, and the prediction is recorded **here, before
the numbers**, so that whatever happens cannot be reported as if it had been expected.

The regime RPM is designed for, as characterised by everything above: registered
imagery, enough samples for a frequency estimate to converge, and a **localised**
informative region surrounded by substantial uninformative area. Blood microscopy fit
that description and was the best case in the suite.

- **medmnist-oct**, retinal OCT, 4 diagnoses, n=109,309. The retina is a band; above it
  is vitreous and below it choroid, both largely uninformative. Strongly localised.
- **medmnist-derma**, dermatoscopy, 7 lesion types, n=10,015. A centred lesion against
  surrounding skin, structurally the same as the blood-cell case.

Predicted, in advance: a **low ambiguous fraction** (bimodal frequency map, so a
reproducible region exists), **low border coverage** (background rejected), and RPM
**competitive on score at substantially lower coverage** than the filter methods.
Predicted to fail: nothing specific, which is itself a risk worth noting, since a
prediction that cannot fail is not a prediction. The falsifier is a high ambiguous
fraction or coverage no lower than the filters'.

### Retinal OCT: the prediction failed

| medmnist-oct (4 classes, n=3000) | score | coverage | stability | border |
|---|---|---|---|---|
| full | 0.751 | 1.00 | 1.00 | 1.00 |
| lasso | 0.751 | **1.00** | 1.00 | 1.00 |
| rfe | 0.751 | **1.00** | 1.00 | 1.00 |
| anova | 0.745 | 0.83 | 0.87 | 0.80 |
| **rf** | 0.747 | **0.45** | 0.75 | **0.31** |
| rpm (sigmoid) | **0.500** | **0.00** | n/a | 0.00 |
| rpm_l0 | 0.750 | 0.58 | 0.72 | 0.40 |
| rpm_boot | 0.749 | 0.60 | 0.71 | 0.41 |
| rpm_ens | 0.746 | 0.80 | **0.90** | 0.63 |

The registered prediction was low coverage and low border retention. **Both were wrong.**
Random Forest is the leanest method here (0.45) and rejects the most background (0.31);
every RPM variant keeps more, and the full method keeps the most of all (0.80). The
falsifier stated in advance, "coverage no lower than the filters'", is met against RF.

The sigmoid variant did something worse: it **collapsed to selecting nothing**, scoring
exactly chance. The stability guard from section 6 caught it and reported the statistic
as undefined instead of a flattering number, which is the guard doing precisely the job
it was added for.

Why the prediction was wrong, stated as a hypothesis rather than a rescue: OCT is a hard
task here (the full-feature score is only 0.751, far below the 0.95 to 0.99 of the other
sets) and the discriminative evidence appears to be distributed across the retinal band
rather than concentrated in a compact object. That is the diffuse-signal regime of
sections 5 and 8, not the localised-object regime this dataset was chosen to represent.
Choosing it on anatomical intuition, that a retina is a band with empty space above and
below, was the error: the empty space is uninformative, but the informative part is
broad rather than compact, and it is compactness that RPM exploits.

One genuinely positive result: the representation ensemble gives the **highest stability
of any method** (0.90 against RF's 0.75), consistent with section 9. It buys that at
high coverage and at roughly 450 seconds per fold set, four times the next most
expensive method, which is a real cost to report.

### Dermatoscopy: the prediction failed again

| medmnist-derma (7 classes, n=2296) | score | coverage | stability | border |
|---|---|---|---|---|
| full | 0.767 | 1.00 | 1.00 | 1.00 |
| anova | 0.767 | **1.00** | 1.00 | 1.00 |
| rfe | 0.767 | **1.00** | 1.00 | 1.00 |
| lasso | 0.765 | 0.55 | 0.64 | 0.51 |
| **rf** | 0.752 | **0.27** | 0.74 | **0.01** |
| rpm (sigmoid) | **0.598** | 0.02 | 1.00 | 0.00 |
| rpm_l0 | 0.754 | 0.42 | 0.48 | 0.42 |
| rpm_boot | 0.765 | 0.81 | 0.75 | 0.80 |
| rpm_ens (1 seed) | 0.770 | **0.99** | 0.98 | 0.99 |

Random Forest is again the leanest (0.27) and rejects nearly all background (0.01),
while every RPM variant keeps more. **Both showcase datasets falsified the prediction,
on the criterion stated in advance.**

Note also the sigmoid variant at coverage 0.02 with stability **1.00**: it selects the
same single patch every fold, which is perfectly reproducible and useless (score 0.598
against a 0.767 baseline). Chance correction does not catch this, because identical
sets genuinely are identical. The lesson is not another metric but a reporting rule:
**the unit of reporting is the triple (score, coverage, stability)**, and no member of
it means anything alone.

### A confound in the dermatoscopy result that must be stated

Dermatoscopy is natively RGB, and colour is substantially diagnostic for lesion type.
The loader converts to luminance because the patch grid is single-channel, so **the most
informative channel was discarded before any method saw the data**. The comparison
between methods remains fair, since all of them see the same grayscale input, but the
dataset is not a fair test of the regime it was chosen for. This is a genuine scope
limitation of the current pipeline, not a property of the method, and multi-channel
input should be supported before any RGB dataset is used as evidence again.

### What the two failures actually teach

A pattern worth stating plainly, because it recurs across both: **the representation
ensemble is conservative**. On OCT it keeps 0.80 and on dermatoscopy 0.99. Marginalising
over lenses means a patch has to be rejected by most lenses to be dropped, so when
diverse lenses disagree the aggregate keeps almost everything. On the synthetic control,
where the lenses largely agreed about a compact region, that same mechanism delivered
both leanness and stability. On hard real data where they disagree, it delivers
stability by declining to select. **Marginalisation trades leanness for stability, and
the exchange rate depends on how much the lenses agree**, which is exactly what
`representation_agreement()` measures and should therefore be reported beside it.

The honest summary of the showcase attempt: choosing datasets on anatomical intuition
about where a region "should" be was the wrong selection criterion. The property RPM
exploits is **compactness of the discriminative evidence**, which is not the same as
the presence of visible empty space, and it is measurable in advance from the
frequency map rather than guessable from the modality.

## 11. Benchmarking artefacts worth recording

- **Uncapped L1 logistic regression is not a fair competitor.** With liblinear and an
  unbounded iteration budget, `lasso` took 532s on a single MNIST dataset and later
  wedged an entire ablation run. Capped to saga with a bounded `Cs` grid it produces
  the same selection far more cheaply. This is a harness artefact, not a property of
  L1 selection, and is documented in `competitors.py`.
- **Hardcoded method lists silently drop results.** `rpm_l0` ran correctly but was
  omitted from every summary table and plot because the display order was hardcoded.
  Method order is now derived from the selector registry, so a newly registered
  selector cannot vanish from the reported results.
- **Relying on a library default silently duplicated a variant.** `select_rpm` did not
  name its gate, so when the library default changed to Hard-Concrete it became an
  exact copy of `select_rpm_l0`, and a whole MedMNIST run reported identical numbers
  for the two variants. Both now pin their gate explicitly. This is the same failure
  mode as the two above: an implicit default or a shared output name quietly collapsing
  two things into one. Every variant, filename and display list in this suite now names
  itself explicitly.
- **Fixed output names overwrite earlier runs.** A single-axis ablation regenerated the
  full-sweep figures because plot filenames ignored the run prefix. All benchmark
  outputs are now prefixed per run.
- **Timings on this machine are contended.** Several unrelated long-running jobs share
  the CPU, so wall-clock numbers in the logs are not clean benchmarks of method cost.
  Relative comparisons within a single run are still valid because all methods in a
  run face the same contention.

## 12. What these results imply for the next milestones

Ordered by the strength of the evidence behind them.

1. **Milestone 5 (the representation ensemble) is now the highest priority.** This was
   always the paper's novelty spine, and section 8 has now promoted it from
   conceptual to necessary: bootstrap over data resamples is exhausted as a route to
   stability, converging by B=30 to a diffuse frequency map. Marginalising over
   representations is a second, independent axis of variation and the only mechanism in
   the method not yet tried against the small-sample problem. The representation-axis
   ablation was already the make-or-break experiment for novelty; it is now also the
   leading candidate fix for the one regime where RPM loses.
   (Milestone 4 is complete; its result is section 8 and it did not fix the collapse.)
2. **Per-dataset lam selection must become a library feature, not a benchmark script.**
   Section 6 shows a single global lam failing in both directions: under-pruning on
   abdominal CT (88% coverage) and over-pruning on breast ultrasound. The parent
   framework already tunes its threshold per dataset, and RPM needs the equivalent,
   most naturally as coverage targeting on the frontier from section 3.
3. **Two metric hazards need guarding in code**, both of which make a non-result look
   perfect: an empty selection reports stability 1.00 (section 6), and a large TV
   weight reports perfect score and stability by keeping everything (section 7).
   Stability should be undefined below a minimum selection size, and coverage should be
   reported beside every stability figure.
4. **Random Forest is the competitor to beat on real data**, not the filter methods.
   ANOVA, Lasso and RFECV frequently fail to reduce at all on scientific images, which
   is itself a reportable finding, but RF reduces well and is often more stable. RPM's
   case against RF rests on the tunable frontier and the coherence of the selected
   region, so both need to be first-class in the paper.
5. **Recovery metrics only exist on synthetic data.** On real data the border
   diagnostic and the contiguity measure are the available proxies, and a corruption or
   shift robustness delta (still planned) remains the strongest untested evidence for
   the robust-predictor claim.

## 13. Cost and scaling (provisional)

Every number in this section is **provisional**. The measurement was forced on a machine
running an unrelated job, after a 40-hour watch failed to find a quiet window (238 polls,
never three consecutive quiet ones; see `results/TIMING_PROVISIONAL.md`). CPU was 16.5%
busy at the start and 75.9% at the end, and the largest median-to-minimum spread was
x1.78. Absolute seconds are therefore not quotable. What survives contamination is the
*shape*: contention inflates measurements roughly uniformly, so scaling exponents and the
ranking between methods remain informative even when the constants do not.

Source: `run_timing.py`, `results/timing_results.csv`, `results/TIMING_SUMMARY.md`,
`results/timing_scaling.png`. Best of three repeats, since contention can only make a
measurement slower and the minimum is therefore the estimate closest to the true cost.

### How cost scales

Log-log slopes of fit time against each axis:

| axis | single | bootstrap | ensemble | reading |
|---|---|---|---|---|
| sample count | +0.52 | +0.40 | +0.76 | strongly **sub-linear** in n |
| image side | +1.75 | +1.76 | +1.87 | about **linear in pixel count** (side squared) |
| patch size | -0.79 | -0.61 | -0.86 | larger patches, fewer of them, cheaper |
| bootstrap count | | +0.93 | | **linear**, as the construction implies |
| representation count | | | +1.21 | slightly super-linear |

Two of these matter in practice. Cost is dominated by the **image**, not the sample
count: going from 100 to 1600 images costs about 8x, while going from a 32 pixel side to
256 costs about 2400x (0.07s to 168.8s for the ensemble). And `n_bootstrap` is honestly
linear, so the stability estimate can be bought in predictable units, which is what makes
the convergence study in section 8 actionable.

The patch axis is the cheap lever. Doubling the patch size quarters the patch count and
roughly halves the cost, at the price of a coarser answer.

### How the methods rank

At n=400, 32x32, patch 4:

| method | best (s) | relative |
|---|---|---|
| anova | 0.001 | fastest, and it selects nothing useful on real imagery (sec.6) |
| lasso | 0.009 | |
| **RPM single** | **0.093** | |
| random forest | 0.877 | |
| **RPM bootstrap** | **1.011** | comparable to a random forest |
| **RPM ensemble** | **3.701** | about 4x a random forest |

A single RPM mask fit is about **nine times cheaper than random-forest importance** on
the same features, which is worth stating because the method's cost is usually assumed to
be the objection to it. The full stability selection lands level with a random forest,
and the representation ensemble costs about four times one. That is the price of the
central claim, and section 9 is where its value is assessed.

The univariate filters are two to three orders of magnitude cheaper than anything else
and should be understood accordingly: on the real datasets they retained 97 to 100% of
the image, so their speed is the speed of not selecting.

## 14. Matched coverage, and a confound in this suite's own protocol

Section 6 reported that random-forest importance selected leaner regions than RPM on
real data at lower score cost. That comparison had two flaws, and correcting them
changes the conclusion.

**Flaw one: the operating points were not matched.** RF importance keeps every feature
above the mean importance, which lands near a third of the patches by arithmetic rather
than by choice, while RPM's coverage came from a fixed sparsity strength. Comparing 0.33
against 0.45 compares two different questions. `run_coverage_frontier.py` gives every
arm the same dial: at each target coverage c, each selector returns exactly its best
ceil(c*P) patches, with folds, features and downstream model identical.

**Flaw two: the judge shared a family with one of the contestants.** Every benchmark in
this suite refits a common Random Forest on the selected patches. RF importance ranks
features by their usefulness *to a random forest*, so that arm was being scored by a
model with its own inductive bias. Swapping the downstream model separates "these are
better patches" from "these patches suit the judge".

### The result

Operating points won, over 4 datasets x 5 coverage levels:

| downstream model | rpm | rf | mi | anova |
|---|---|---|---|---|
| Random Forest (the suite default) | 2 | **9** | 6 | 3 |
| Logistic regression (control) | **10** | 4 | 3 | 3 |

The ranking inverts. Judged by a random forest, RF importance wins nine points and RPM
two; judged by a linear model, RPM wins ten and RF four. Excluding the 5% coverage level,
where every method is unstable, RPM wins 10 of 16 points under the linear judge and beats
RF on 11 of 16, mean margin +0.005.

The honest reading is not "RPM is better after all". It is that **a selector-evaluator
affinity effect of roughly the size of the between-method differences has been present in
every result this suite has produced**, and that section 6's verdict was measuring it.
Both protocols are now reported; neither alone is decisive, and a selection method should
be judged by a model other than the one that generated it.

### A real defect this exposed

At 5% coverage RPM collapsed (AUC 0.66 against RF's 0.86 on pneumonia), and the cause is
mechanical rather than statistical. ``pi_`` is a count over B resamples, so it takes at
most B+1 distinct values and saturates: on a 49-patch grid with B=10, seven patches sat
at pi=1.0 with *none* above them, so a top-2 request returned two of the seven chosen by
index order. The frequency map simply cannot rank inside its own saturated top group.

``BootstrapMaskSelector`` now records ``strength_``, the mean gate VALUE over the same
fits, and ``ranking()`` / ``top_patches()`` break frequency ties on it. This is a strict
refinement: the frequency ordering is preserved wherever it discriminates, and arbitrary
index order is replaced by evidence everywhere else. Measured effect: +0.11 AUC at 5%
coverage on MNIST, improving 6 of 10 tested points, never worse than -0.0002.

It does not rescue diffuse-signal pneumonia at 5% (+0.003), which is consistent with
section 10: no ranking recovers a compact region from a signal that is not compact.

Source: `run_coverage_frontier.py`, `results/coverage_frontier.csv`,
`results/coverage_frontier_logreg.csv`, `results/coverage_tiebreak.csv`.

### The panel result, and a correction to the paragraph above

The real-data components were re-run with `score_panel`, scoring each selection under
four judges (random forest, logistic regression, SVM, k-nearest neighbours) rather than
one. 162 rows, 6 datasets, 3 seeds. This both resolves section 14's question and
corrects two things stated too strongly before it.

**Correction one: the affinity effect is real but modest at natural operating points.**
The matched-coverage sweep suggested a dramatic inversion (RF winning 9 of 20 points
under its own judge, RPM 10 under a linear one). Paired properly by dataset and seed at
the methods' natural operating points, `rpm_boot` against `rf`:

| judge | mean margin | wins | Wilcoxon p |
|---|---|---|---|
| random forest | -0.0020 | 10/18 | 0.865 |
| logistic regression | **+0.0047** | 13/18 | **0.024** |
| SVM | +0.0004 | 12/18 | 0.442 |
| k-nearest neighbours | -0.0018 | 6/18 | 0.347 |

The interaction is what carries the claim, and it is significant: the RPM-minus-RF
margin is larger under logistic regression than under the random forest by +0.0067
(p < 0.0001) and under SVM by +0.0025 (p = 0.043), while kNN shows nothing (+0.0002,
p = 0.58). So the judge demonstrably shifts the verdict, in the predicted direction, for
two of the three alternatives. It is a real effect of a few thousandths of AUC, not the
wholesale reversal the matched-coverage figures implied.

**Correction two: section 6's verdict was not that RF wins, it was that nothing was
measurable.** Paired under the suite's own random-forest judge, `rpm_boot` and `rf` are
statistically indistinguishable (p = 0.865, RPM ahead on 10 of 18 dataset-seed pairs).
Section 6 read a difference of 0.002 in unpaired means over six datasets as RF being
better. It was not significant then either, and the honest statement is that the two
methods score the same on real data while RF retains a genuine leanness advantage
(coverage 0.33 against 0.45).

**A caution about the naive home-advantage statistic.** The obvious measure, a method's
score under the RF judge minus its mean under the others, does NOT isolate the effect:
every method gains about 0.028 there, because the random forest is simply the strongest
classifier on these features. `full` gains 0.0290 and `rf` 0.0269, so read that way RF
appears to have no home advantage at all. Only the paired margin between two methods,
compared across judges, separates affinity from judge difficulty. The interaction test
above is the one that means something.

### Where this leaves the comparison

Under the panel mean, `rpm_boot` (0.9325) and `rf` (0.9322) are level, at 45% and 33%
coverage. No selector dominates on real-data score, and the differences that exist are
smaller than the effect of choosing the evaluator. RPM's defensible advantages are the
judge-independent ones: recovery of a known region (F1 0.984 against 0.82 for the best
competitor), background rejection (border coverage 0.00 to 0.02 where filters keep 0.57
to 1.00), and the `ambiguous_fraction` diagnostic. Every benchmark from here reports the
panel, and `score` continues to mean the random-forest judge so older tables still read
correctly.

## 15. The ablation the suite was missing: does the learned mask earn its place?

Every stability-selected arm in this suite uses the learned mask (`rpm_boot`,
`rpm_ens`) and every non-mask arm is a single fit (`anova`, `lasso`, `rfe`, `rf`). So
every comparison drawn so far confounds two independent things: the learned mask
against another ranker, and bootstrap aggregation against one fit. RPM's expensive,
novel component is the mask; the wrapper is Meinshausen-Buehlmann and applies to any
ranker. Nothing measured before this section separates them.

`select_rf_boot` holds the wrapper fixed and swaps the base ranker: same resampling,
same B=10, same stratification, same tau, only RF importance in place of the mask.
Four datasets, three seeds, scored by the four-judge panel.

| method | panel score | coverage | stability | border coverage | seconds |
|---|---|---|---|---|---|
| full | 0.9170 | 1.00 | 1.00 | 1.00 | 32 |
| rf | 0.9128 | 0.28 | 0.89 | 0.099 | 29 |
| rf_boot | 0.9101 | 0.26 | **0.89** | **0.083** | 132 |
| rpm_boot | 0.9104 | 0.27 | 0.67 | 0.125 | 23 |

**The mask adds nothing over RF importance under the same wrapper**: mean +0.0003,
winning 5 of 12 dataset-seed pairs, p = 1.000. Per dataset the differences scatter
either side of zero (-0.0042, -0.0014, 0.0000, +0.0068).

**The wrapper adds nothing to RF either**, and if anything costs a little
(rf_boot minus rf, -0.0027, p = 0.050) for 4.6x the runtime.

Two further results contradict claims made earlier in this document.

**RPM is markedly LESS stable than RF under the identical wrapper** (0.67 against 0.89).
Stability selection is supposed to be what buys reproducibility, so a learned mask that
is less reproducible than feature importances inside the same aggregation is a real
finding against the mask, not a presentational detail.

**RF rejects background slightly better than RPM here** (border coverage 0.083 against
0.125). Section 6's background-rejection claim compared RPM against the univariate
filters, which retain 0.57 to 1.00 of the border. Against RF specifically the advantage
reverses, and the earlier phrasing implied a general one.

### What survives

The mask's clear win is ground-truth recovery on the synthetic control (F1 0.984
against 0.737 for RF), which is the only setting with a known answer. One reading is
that the mask genuinely finds compact planted regions better; another is that real
images may not contain the compact recoverable region the synthetic control plants, a
possibility RPM's own `ambiguous_fraction` supports on breast. These are not
distinguished by anything measured here.

Note also that `ambiguous_fraction` is **not** mask-specific: any bootstrapped ranker
yields a selection-frequency map, so the diagnostic could be built on RF importance at
a fraction of the cost. It is a property of the wrapper, not of the mask.

### Limits of this ablation

Four datasets, three seeds, twelve paired comparisons. p = 1.000 at 5 of 12 wins is
"no detectable difference", not "proven identical", and a real effect below roughly
0.01 AUC would not be visible at this power. The result does not touch the leakage-safe
nested-CV engine, which is infrastructure shared by every arm.

Source: `competitors.select_rf_boot`, `results/ablation_wrapper_results.csv`.

## 16. The synthetic control is matched to RPM's model class, and that hid a hard limit

Section 15 left one result standing for the learned mask: recovery F1 0.984 against
0.737 for RF on the synthetic control. That control builds its label as
`agg = S.sum(axis=1)` then `y = agg > 0`, a LINEAR function of the informative cell
signals, and RPM's mask sits under a linear or logistic head. The benchmark where RPM
wins largest is therefore one where its inductive bias is correct by construction.

`run_nonlinear_control.py` holds the geometry, cells, noise and equal-variance decoy
fixed and varies only how cell signals combine into the label. Five seeds each,
recovery F1 against the planted region.

| label rule | max abs corr(patch, y) | RF F1 | RPM F1 | RPM advantage |
|---|---|---|---|---|
| linear, `sum(S) > 0` | 0.397 | 0.898 | **1.000** | +0.102 |
| threshold, all cells above median | 0.219 | 0.335 | **0.889** | +0.554 |
| interaction, product of signs | 0.018 | **0.112** | 0.000 | -0.112 |
| saturating, `abs(tanh sum)` above median | 0.018 | **0.223** | 0.000 | -0.223 |

**On two of four rules RPM recovers nothing whatsoever.** Not a degraded region: F1
exactly 0.000, on every seed, with the informative region fully present in the data and
a random forest still finding a tenth to a fifth of it.

The mechanism is in the third column. The two rules RPM fails are precisely the two
where no informative patch has a linear main effect on the label (max correlation
0.018, against 0.22 to 0.40 where RPM succeeds). Both are symmetric in the cell
signals, so `E[y | patch]` is flat even though the patch is fully informative jointly.
Phase 1 fits a linear head, phase 2 freezes it and prunes gates that do not help it,
and a gate with no linear main effect cannot help a linear head. The mask prunes the
entire informative region and reports a confident, stable, empty answer.

This is not an exotic corner. Any XOR-like or magnitude-driven relationship has this
shape, and the second rule is a plain symmetric nonlinearity rather than an adversarial
construction.

### What this changes

The synthetic headline needs restating. RPM's recovery advantage is real but conditional
on additive structure: strongest where the label is additive in the patch features
(+0.10 linear, +0.55 conjunctive-with-main-effect), and negative where it is not. The
control shipped with the library tests only the favourable case.

Combined with section 15, the position is that the learned mask beats RF importance on
additive synthetic structure, ties it on real data, and loses badly to it when the
signal is purely interactive. Nothing here recommends the mask over a bootstrapped RF
importance for general use.

Source: `run_nonlinear_control.py`, `results/nonlinear_control.csv`.

## 17. A nonlinear head fixes half the hole, and breaks the pruning it relies on

Section 16 traced RPM's empty selections to the frozen head: phase 2 prunes gates that
do not help it, and a gate with no linear main effect cannot help a linear head.
`head="mlp"` puts one hidden layer under the same freeze-base scheme, so phase 2's
gradient routes through the hidden layer and a patch can earn its place through an
interaction. Both gradients are hand-derived and verified against central differences
to 5e-9 (`test_deep_sweep.py`).

### What it fixed, read against chance rather than against zero

Recovery F1 alone is misleading here, because a method that stops pruning scores
nonzero F1 by keeping everything. With 4 informative patches of 49, selecting all of
them gives precision 0.082. Judging each result against that:

| label rule | MLP coverage | MLP precision | verdict |
|---|---|---|---|
| linear | 0.062 | 1.000 | genuine, ties the linear head at F1 1.000 |
| threshold | 0.062 | 1.000 | **genuine gain**, F1 0.889 to 1.000 |
| saturating | 0.212 | 0.310 | **genuine**, 3.8x chance, F1 0.000 to 0.469 |
| interaction | 0.466 | **0.082** | **artefact**: exactly chance, it kept half the image |

So the MLP head repairs **one** of section 16's two zeros. The interaction result was
first reported here as a fix and is not one: at precision exactly equal to chance, the
recall came from volume, not recovery. That correction is recorded rather than quietly
amended.

### The defect the MLP head introduced

On multiclass real data the MLP mask **never closed**: all 49 gates stayed at 1.000,
the frequency map was flat, and `top_patches` therefore returned patches in index
order, a selection not determined by the data at all. The tell was that the score was
identical to four decimal places regardless of iterations or warmup, which is the
signature of a degenerate output rather than of undertraining.

The mechanism is that phase 2 trades the data gradient against `lam`, and a more
expressive frozen head genuinely uses every input, so no gate looks dispensable at the
linear head's `lam`. Unmatched real-data runs made this look like an improvement, since
`rpm_mlp` scored highest at coverage 0.66 against 0.27; that was the coverage confound
of section 9 again, and it is why the matched-coverage protocol exists.

### A fix that was tried, measured, and discarded

The obvious repair is to rescale the phase-2 data gradient by a one-off constant so
`lam` means the same thing to both heads. Measured on blood, the `lam` the two heads
need for equal coverage moved from 3.0x apart to only 2.5x. It does not work, because
what differs between heads is how steeply the data loss grows **as gates close**, not
the gradient's starting magnitude. The code was removed rather than kept because it
looked principled.

`lam_scale="auto"` instead searches `lam` on a geometric ladder for the mask landing
nearest `auto_target` open, which makes the knob "how much do you want kept" and is
head-independent by construction. On blood it selects `lam=0.48` for the MLP where 0.03
left every gate open, and `lam_used_` reports the operating point rather than hiding it.
The default remains `lam_scale="fixed"`, pinned by a test asserting it reproduces
previous masks exactly, so no number published before this section moves.

### Status

`head="mlp"` is opt-in and stays so. The matched-coverage rerun called for here has
since been run twice, and section 18 reports it: the pruning fix exposed a second
defect in how `lam_scale="auto"` interacted with the bootstrap, and with that also
fixed the MLP head still loses to random-forest importance on stability in 19 of 20
cells and on score at p = 0.019. It repairs one documented failure and costs more than
the linear head. Sections 15 and 16 are unchanged by it: the mask still ties
bootstrapped RF importance on real data, and the synthetic control still only tests the
favourable case unless the nonlinear axis is run.

Source: `masking.SoftMaskSelector(head=..., lam_scale=...)`,
`run_nonlinear_control.py`, `results/nonlinear_control.csv`,
`results/coverage_head.csv`.

## 18. The fix for section 17 put a hyperparameter search inside the aggregation

`lam_scale="auto"` (section 17) makes the MLP head usable by searching `lam` on a
geometric ladder instead of trusting a value tuned for the linear head. It was wired
in the obvious way: `BootstrapMaskSelector._inner_kwargs` forwards every inner-fit
setting to each resample fit, and `lam_scale` went through with the rest.

That is wrong, and wrong in a way the matched-coverage rerun made visible before the
reasoning did. Stability selection counts how often a patch survives **at a fixed
operating point**; that is the entire content of `pi_`. Searching `lam` inside each
resample makes the operating point itself a random variable across resamples, so a
patch can drop out of the frequency map because a different penalty was chosen, not
because the data stopped supporting it. `pi_` then reports a mixture of "which patches
matter" and "which `lam` this replicate happened to land on", while still being
documented, thresholded and plotted as if it were only the first.

### Measured, on the same 20 matched-coverage cells

`rpm` and `rf` are bit-identical between the two runs, which confirms the change
touches only the `auto` path.

| `rpm_mlp` | per-resample search | calibrated once | paired change |
|---|---|---|---|
| mean pairwise Jaccard | 0.375 | 0.455 | +0.081, better in 13/20, p = 0.097 |
| mean held-out score | 0.8773 | 0.8750 | -0.0023, p = 0.70 |
| beats RF on stability | 0/20 cells | 1/20 cells | |
| mean seconds per cell | 120 | 35 | at least 3.4x faster |

The speedup is the one figure that is safe to quote under load, because it is a lower
bound: the calibrated run shared the machine with the standard tier and the
per-resample run did not, so a quiet machine would widen the gap. Its cause is
structural rather than empirical, which is the better argument anyway: `B` resamples
each repeating a 6-step ladder becomes one ladder plus `B` single fits.

### What this does and does not establish

It does **not** establish that calibrating once makes RPM more stable. +0.081 at
p = 0.097 across 20 cells is a direction, not a result, and it is recorded here as a
direction. The change is justified on the other two grounds, which do not need a
significance test: `pi_` should measure one thing rather than two, and one ladder is
cheaper than `B` ladders.

It also does not move section 15. With the fix in, `rpm_mlp` still loses to
random-forest importance on stability in 19 of 20 cells and on score (median -0.0034,
Wilcoxon p = 0.019). The learned mask remains the part of this library that has not
earned its place; sections 15, 16 and this one are three independent measurements
saying so, and the open question in section 15 (whether to reframe around a pluggable
ranker with RF as the default) is not answered by repairing the mask's plumbing.

### The general shape of this defect

A hyperparameter search placed inside a resampling loop that exists to measure
stability will always spend some of that stability on itself, silently, and the
resulting number keeps its old name. Calibration belongs outside the aggregation and
inside the outer fold: it sees only what `fit` was given, which under nested CV is the
outer training split, so this adds no leakage. Two tests pin it, one asserting exactly
one searching fit per `fit` call and the rest pinned to the calibrated value, one
asserting the default path constructs no probe at all.

Source: `selection.BootstrapMaskSelector._calibrate_lam`, `_resolved_lam`,
`test_deep_sweep.py::test_auto_lam_is_calibrated_once_not_per_resample`,
`results/coverage_head2.csv` (before), `results/coverage_head3.csv` (after).

## 19. `auto_target` is a prior on coverage, and the interaction rule is still unsolved

Section 18 made `lam_scale="auto"` cheap and made the frequency map mean one thing
again. That made it worth asking the question section 17 left open: with the pruning
actually working, does the MLP head recover the informative region on the interaction
rule, or was the earlier result an artefact of keeping half the image?

The four label rules were rerun with the MLP arm at `lam_scale="auto"` against the
same rules at `lam_scale="fixed"`, 5 seeds each. The linear and RF arms are
bit-identical between the two runs, so any difference is the penalty scaling alone.
Recovery precision is read against the chance rate, 4 informative patches of 64
(0.062), because a method that stops pruning scores well on recall by volume.

| label rule | fixed | auto | coverage, fixed to auto |
|---|---|---|---|
| linear | 1.000 (16.0x chance) | 0.292 (4.7x) | 0.062 to 0.219 |
| threshold | 1.000 (16.0x) | 0.575 (9.2x) | 0.062 to 0.113 |
| saturating | 0.310 (5.0x) | 0.310 (5.0x) | 0.212 to 0.212 |
| **interaction** | **0.082 (1.3x)** | **0.076 (1.2x)** | 0.466 to 0.366 |

### The interaction rule is not solved

At 1.2x chance the MLP head is doing essentially nothing on interactive labels, and
the harder pruning did not help: coverage fell from 0.466 to 0.366 while precision
fell slightly too, so the patches it gave up were not the uninformative ones. The
linear head scores exactly 0.000 there, and RF importance manages 1.05x chance, so
this is not a defect peculiar to RPM, but nothing in this library solves it either.

Section 17 reported the interaction result as an artefact of not pruning and left open
whether fixing the pruning would turn it into a recovery. It does not. Section 16's
second zero stands, and it should be read as a limit of the method rather than a
tuning problem: the freeze-base trick prunes a gate that does not help a head fitted
with every gate open, and on a pure interaction no single gate does.

### `auto` is not an improvement, it is a different prior

The more useful result is the top three rows. `auto` made recovery **worse** wherever
the mask was already closing correctly, and the mechanism is plain in the coverage
column: `auto_target` defaults to 0.4, the true informative fraction is 0.062, and the
lam search dutifully found a penalty that keeps roughly what it was asked to keep.
Nothing malfunctioned. The default simply encodes a coverage prior that was wrong for
this control by a factor of six, and it cost 16.0x chance down to 4.7x.

That knob was also unreachable. `auto_target` existed only on `SoftMaskSelector`,
while `BootstrapMaskSelector`, both fold estimators and the `RobustPixelMaker` facade
neither accepted nor forwarded it, along with `head`, `hidden` and `lam_scale`. The
README documented `head="mlp"` as a feature of the library while the documented entry
point could not set it. All four are now plumbed through every layer and validated at
construction, in the facade and the aggregator as well as the mask, because
`_one_fit` swallows resample exceptions and would otherwise report a mistyped `head`
as "every one of the N resample fits failed".

### Guidance, which is the point of recording this

`lam_scale="fixed"` stays the default and remains right for the linear head. Reach for
`auto` for the reason section 17 gives, that an expressive head at a fixed lam may not
close the mask at all, and when you do, set `auto_target` near the coverage you
actually expect rather than accepting 0.4. If you have no expectation, `target_coverage`
on the aggregator is the more direct control and does not go through a lam search.

Source: `results/nonlinear_control.csv` (fixed), `results/nonlinear_control_auto.csv`
(auto), `run_nonlinear_control.py --lam-scale`, `core.RobustPixelMaker`,
`test_deep_sweep.py::test_facade_can_actually_reach_the_mlp_head`.

## 20. The suite published an arm that selected nothing, and nothing said so

The first standard-tier run after the section 18 and 19 work reproduced every score,
coverage and stability figure from the previous run exactly, across all four
components. Only the `seconds` column moved. That is the guarantee `lam_scale="fixed"`
as the default was supposed to give, and it is now verified on a full suite rather
than argued from a unit test.

The same run also shipped a result that was not one.

| dataset | method | score | coverage | stability |
|---|---|---|---|---|
| medmnist-blood | full | 0.954 | 1.000 | 1.000 |
| medmnist-blood | **rpm_mlp** | **0.954** | **1.000** | **1.000** |
| medmnist-organa | full | 0.984 | 1.000 | 1.000 |
| medmnist-organa | **rpm_mlp** | **0.984** | **1.000** | **1.000** |

On both multiclass sets the MLP arm kept every patch. This is section 17's flat-mask
failure exactly: an expressive head at a fixed `lam` finds a use for every input, so
no gate ever closes. `select_rpm_mlp` was left at the default `lam_scale="fixed"`,
which is correct for the linear head and is the one setting under which this arm
cannot work.

### Why it was invisible

Read down the summary table's `score` column, `rpm_mlp` is tied for best on blood.
Read down `stability`, it is perfect. Both columns flatter it for the same reason:
its score is the full-image score because it kept the full image, and keeping
everything is perfectly reproducible so it cannot disagree with itself across folds.

This is the hazard family this repository has now recorded five times: **a metric that
rewards not selecting will be won by a method that does not select.** Every previous
instance was caught in reasoning, in a claim about a result. This one was in the
shipped results, in a table, for a full run. The difference matters, because the
earlier catches encouraged the belief that knowing about the trap is protection. It
is not. The protection is a column in the table.

### The fix is the column, not the arm

Two changes, and the ordering is deliberate. The arm now uses `lam_scale="auto"`,
which since section 18 is calibrated once per fit and therefore affordable, and which
is the difference between selecting and not selecting rather than a tuning preference.
But an arm's configuration is a thing that can regress silently, so the durable fix is
that every summary table now carries a `selects` column, `no` wherever coverage is at
or above 0.999, with a note saying such a row must be compared against `full` and not
against the selectors.

That marking is not specific to RPM, which is the point. `rfe` reaches coverage 1.000
on blood and mnist-all, and `anova` and `lasso` do on organa. Those are legitimate
outcomes for those baselines and they are marked identically. A reader can now tell a
selector from a non-selector without knowing the method.

### The corrected run, and what the old numbers were actually measuring

The standard tier was rerun in full (2026-09-14, 268 min, all ten components ok). The
expectation that only `rpm_mlp` would move was checked rather than assumed, and it
holds exactly: across 957 paired rows, score, coverage and stability are identical to
the previous run for every arm except `rpm_mlp`, which moved on all six real datasets
and on none of the 759 synthetic rows.

| dataset | coverage | score | stability |
|---|---|---|---|
| medmnist-blood | 1.000 to **0.105** | 0.954 to 0.912 | 1.000 to 0.516 |
| medmnist-organa | 1.000 to **0.087** | 0.984 to 0.922 | 1.000 to 0.646 |
| medmnist-breast | 0.970 to **0.075** | 0.846 to 0.691 | 0.943 to 0.096 |
| medmnist-pneumonia | 0.426 to **0.044** | 0.976 to 0.844 | 0.703 to 0.409 |
| mnist-3v8 | 0.271 to **0.132** | 0.994 to 0.959 | 0.867 to 0.530 |
| mnist-all | 0.721 to **0.105** | 0.993 to 0.925 | 0.922 to 0.717 |

The scores fell everywhere, and that is the correction rather than a regression. This
arm's previous real-data numbers were substantially the full-image baseline: on blood
and organa it kept every patch, and on breast and mnist-all it kept 97% and 72%. A
score of 0.984 for a method that selected nothing is not a result about selection.

That the synthetic rows did not move at all is the cleanest available evidence that
`auto_target=0.15` reproduces fixed-`lam` behaviour wherever the mask was already
closing (sec.21), now confirmed across 759 rows rather than the five-value sweep.

Source: `competitors.mark_non_selectors`, `competitors.select_rpm_mlp`,
`tests/test_benchmark_reporting.py`, `results/ablation_head_results.csv` (the run that
exposed it: `rpm_mlp` at coverage 1.000 on blood and 0.97 on breast),
`results/medmnist_results.csv` and `results/real_results.csv` (the correction).
The console logs of both runs are local only, so the CSVs are cited instead.

## 21. The default that caused section 19 was avoidable, and the dial is quantised

Section 20 changed the `rpm_mlp` benchmark arm to `lam_scale="auto"`, because at a
fixed `lam` it returned coverage 1.000 on multiclass and selected nothing. The partial
rerun that followed showed the cure doing real harm on the synthetic control: recovery
F1 0.981 to 0.482, coverage 0.067 to 0.225. That looked like section 19 repeating, an
unavoidable trade between closing the mask and keeping a sensible operating point.

It was not unavoidable. It was the default. Sweeping `auto_target` over the two cases
that pull in opposite directions, the synthetic control (true informative fraction
0.062) and blood (multiclass, where fixed `lam` fails to close):

| `auto_target` | synthetic coverage | synthetic F1 | blood coverage | blood score |
|---|---|---|---|---|
| fixed `lam` | 0.062 | 1.000 | **1.000** | 0.937 |
| 0.10 | 0.062 | 1.000 | 0.041 | 0.755 |
| **0.15** | **0.062** | **1.000** | **0.095** | **0.896** |
| 0.25 | 0.234 | 0.441 | 0.095 | 0.896 |
| 0.40 | 0.234 | 0.441 | 0.095 | 0.896 |

At 0.15 the synthetic control is recovered **exactly as well as under a fixed `lam`**,
F1 1.000 at coverage 0.062, while the multiclass mask still closes. There is no trade
here at all. The damage reported in section 19, and the regression seen in the partial
rerun, were both the 0.4 default and nothing else. The library default is now 0.15.

Blood's score falling from 0.937 to 0.896 is not a loss: 0.937 was the full-image score
attached to a selection of everything. 0.896 at coverage 0.095 is the first honest
number this arm has produced on that dataset.

### The dial is coarse, which is not obvious from its name

0.25 and 0.40 give **identical** selections on both datasets, and 0.15 and 0.25 are
identical on blood. `auto_target` does not set a coverage. It picks one of six rungs on
a geometric ladder of `lam` multipliers, and neighbouring targets routinely land on the
same rung. A caller who reads "the fraction of the mask to leave open" and sets 0.30
expecting something between 0.25 and 0.40 gets exactly 0.25's answer.

This is worth stating plainly because it changes how the argument should be used. It is
a coarse regime selector, not a coverage control. `target_coverage` on the aggregator is
the direct control, and it does not go through a `lam` search at all.

### The process point

Section 19 measured the 0.4 default doing damage and drew a general conclusion about
`auto` being "a different prior". The narrower and more useful conclusion, that the
default was simply too high for sparse selection and a lower one costs nothing, needed
one sweep of five values taking under two minutes. It was not run, because the finding
felt complete once it had a mechanism. A mechanism that explains a result is not the
same as having checked whether the result was necessary.

Source: `results/auto_target_sweep.csv`.

## 22. A request that could not be met, reported as though it had been

`target_coverage` asks the aggregator to threshold `pi_` at whatever coverage the
caller wants. `examples/ex4` uses it to put two datasets at the same operating point
before comparing the *shape* of their selections, because at generous coverage any
selection looks contiguous and only a lean one tests shape. It asked both for 0.12.

It got 0.245 on blood and 0.122 on pneumonia, and reported the comparison as matched.

### Why the request could not be met

`pi_` is a count over B resamples, so it takes at most B+1 distinct values, and the
achievable coverages are exactly the retained fractions at those values. Ties make the
grid coarse. On blood at B=12 the reachable coverages are

    0.245, 0.286, 0.306, 0.388, 0.408, 0.469, 0.510, 0.571

and 0.12 is not among them, nor near them: 24.5% of patches sit at `pi_` exactly 1.0,
selected in every single resample, so no threshold can split them. `_choose_tau` picked
the nearest reachable value, which is correct behaviour, and the caller was told
nothing. On pneumonia the same request landed at 0.122 because that dataset's map
happened to have a rung there.

### Why it mattered here specifically

Contiguity is exactly the quantity coverage inflates. Measured on pneumonia:

| coverage | contiguity |
|---|---|
| 0.122 | 0.50 |
| 0.245 | 0.83 |

So the shipped comparison, blood at 0.245 against pneumonia at 0.122, was reading a
shape difference off two different operating points, in the direction that favours the
claim. The claim survives: forced to exactly 6 patches each with `top_patches`, blood
scores 1.00 and pneumonia 0.50, the same gap. It was true, and it was untested as
stated, and those are different things.

### Both fixes

The library now records `coverage_realised_` and `coverage_target_met_`, reports the
latter in `stability_report()`, and warns when a target is missed by more than 25%
relative, naming the cause and pointing at `top_patches(k)`. `coverage_target_met_` is
None when nothing was requested, so absence of a request never reads as a failed one.
The example enforces the match with `top_patches` rather than requesting it, and its
third panel now draws the set that was measured rather than the tau threshold.

This is the same family as sections 17 and 20, and the third member found in a week: a
call that cannot do what was asked returns something plausible instead of saying so.
Sections 17 and 20 were masks that would not close; this is a threshold that could not
be reached. In all three the wrong answer looked like a normal one.

Source: `selection.BootstrapMaskSelector._record_coverage_target`,
`examples/ex4_diffuse_signal_diagnosis.py`,
`tests/test_selection.py::test_an_unreachable_target_coverage_is_reported_not_silently_missed`.

## 23. The headline table cannot compare the MLP arm, and says so by accident

After section 20, `rpm_mlp` reads as far the worst arm in the suite: mean real-data
score 0.876 against 0.950 for `rpm_boot` and 0.952 for `rf`. Taken at face value that
says the MLP head is bad. It does not say that, and the reason is in the next column.

| arm | mean score | mean coverage | coverage range |
|---|---|---|---|
| rf | 0.952 | 0.327 | 0.184 to 0.482 |
| rf_boot | 0.950 | 0.314 | |
| rpm_boot | 0.950 | 0.448 | |
| **rpm_mlp** | **0.876** | **0.091** | 0.041 to 0.180 |

The MLP arm is operating at roughly a quarter of everyone else's coverage, because it
needs `lam_scale="auto"` to close its mask at all on multiclass data (sec.20) and
`auto_target` then puts it near 0.15. Comparing 0.876 at coverage 0.09 against 0.952 at
coverage 0.33 is the comparison section 14 was written to warn about.

### At matched coverage it is a different result

Rerunning the matched-coverage protocol at the shipped default, 20 operating points,
with the `rpm` and `rf` arms bit-identical between runs:

| `rpm_mlp` against `rf` | at `auto_target` 0.40 | at 0.15, shipped |
|---|---|---|
| mean score margin | -0.0235 | **-0.0058** |
| operating points won | 6/20 | 8/20 |
| Wilcoxon p | **0.019** | **0.277** |
| mean stability | 0.455 against 0.749 | 0.469 against 0.749 |
| stability wins | 1/20 | 1/20 |

So at equal coverage the MLP head is **statistically indistinguishable from
random-forest importance on score**, where under the old default it was significantly
worse. What does not change is stability: it loses to RF in 19 of 20 cells either way,
by a wide margin, and that is the real finding against it.

The default change itself is directional and not significant (score +0.0177 at p = 0.14,
stability +0.0139 at p = 0.12, paired over the same 20 cells). It is justified by
sec.21's argument, not by these numbers, and is recorded that way.

### What this means for reading the suite

The headline tables report each method at whatever operating point it chooses. That is
the right thing for them to report, and it is **not** a method comparison whenever the
coverages differ, which is most of the time. `run_coverage_frontier.py` exists for the
comparison, and its answer differs from the headline table's by enough to flip a
significance verdict. The `selects` column added in sec.20 catches only the degenerate
end of this, an arm at coverage 1.000; `rpm_mlp` at 0.09 is the same hazard at the
other end, and no column catches it, which is why it is written down here instead.

Source: `results/medmnist_results.csv`, `results/real_results.csv`,
`results/coverage_head3.csv` (0.40), `results/coverage_head4.csv` (0.15, shipped).
