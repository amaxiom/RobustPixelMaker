"""Example 6: an underpowered study should not yield a confident region, and does not.

Breast ultrasound at n=780 sits squarely in the small-sample regime where the imaging
reproducibility literature (Vabalas et al. 2019; Marek et al., Nature 2022) shows
selection instability and optimistic bias are worst. RobustPixelMaker's frequency map
says so BEFORE anyone interprets a region: on blood microscopy (n=3000, compact
signal) pi is bimodal with an ambiguous fraction near 0.15, while on breast ultrasound
it is diffuse with an ambiguous fraction near 0.6, and pushing the bootstrap harder
does not change that verdict (it converges by about 30 resamples; see FINDINGS sec.8).

Science takeaway: this is a TEST for whether a reproducible informative region exists
at the available sample size, not merely a method for producing one. A confident
region map from an n=780 study would be the red flag; the honest diffuse map is the
correct answer, and reporting it is what separates a finding from an artefact.
"""
import numpy as np

from _common import OUT, style, takeaway
from benchmarks.datasets import load_medmnist
from robustpixelmaker import BootstrapMaskSelector


def fit_one(key, task, n_max, seed=0):
    X, y, meta = load_medmnist(key, n_max=n_max)
    sel = BootstrapMaskSelector(task=task, patch=4, lam=0.03, n_bootstrap=30,
                                tau=0.6, n_iter=350, seed=seed).fit(X, y)
    return meta, sel


def main():
    plt = style()
    meta_b, sel_b = fit_one("blood", "multiclass", n_max=3000)
    meta_u, sel_u = fit_one("breast", "binary", n_max=None)     # all 780 images

    rep_b, rep_u = sel_b.stability_report(), sel_u.stability_report()
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    bins = np.linspace(0, 1, 21)
    ax[0].hist(sel_b.pi_, bins=bins, color=plt.cm.viridis(0.35))
    ax[0].set_title(f"blood microscopy, n={meta_b['n']}\nambiguous fraction "
                    f"{rep_b['ambiguous_fraction']:.2f}: a region exists", fontsize=10)
    ax[1].hist(sel_u.pi_, bins=bins, color=plt.cm.viridis(0.7))
    ax[1].set_title(f"breast ultrasound, n={meta_u['n']}\nambiguous fraction "
                    f"{rep_u['ambiguous_fraction']:.2f}: no reproducible region",
                    fontsize=10)
    for a in ax:
        a.set_xlabel("selection frequency pi")
        a.set_ylabel("patches")
    fig.tight_layout()
    fig.savefig(OUT / "ex6_underpowered_study_honesty.png", dpi=120)

    takeaway("Example 6: honesty on an underpowered study", [
        f"blood  (n={meta_b['n']}): ambiguous fraction {rep_b['ambiguous_fraction']:.2f}, "
        f"coverage {sel_b.coverage():.2f}",
        f"breast (n={meta_u['n']}) : ambiguous fraction {rep_u['ambiguous_fraction']:.2f}, "
        f"coverage {sel_u.coverage():.2f}",
        "Same pipeline; only the data differ. The diffuse frequency map is the",
        "correct scientific answer at this sample size, and no amount of extra",
        "resampling manufactures a region the data cannot support.",
    ])


if __name__ == "__main__":
    main()
