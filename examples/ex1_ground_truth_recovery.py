"""Example 1: variance is not information, and RPM knows the difference.

A registered synthetic control plants two regions of EQUAL variance: one drives the
label, the other fluctuates just as strongly but is pure distraction. Any method that
ranks pixels by how much they vary keeps both. RobustPixelMaker recovers the causal
region exactly, rejects every patch of the distractor, and its selection frequency map
is bimodal, the structural signature that a reproducible region exists.

Science takeaway: on data where the ground truth is known, the selected region IS the
informative region, and the equal-variance decoy is untouched. This is the control
experiment that licenses interpreting selected regions on real data.
"""
import numpy as np

from _common import OUT, style, takeaway
from robustpixelmaker import BootstrapMaskSelector, PatchGrid, make_synthetic_images


def main():
    plt = style()
    c = make_synthetic_images(n=400, task="binary", signal=2.0, noise=1.0, seed=0)
    grid = PatchGrid.from_image_shape(c.X.shape, patch=4)

    sel = BootstrapMaskSelector(task="binary", patch=4, lam=0.03, n_bootstrap=20,
                                tau=0.7, n_iter=400, seed=0).fit(c.X, c.y)

    info_p = np.unique(grid.pixel_patch[c.informative])
    dist_p = np.unique(grid.pixel_patch[c.distractor])
    selected = set(sel.selected_patches_.tolist())
    recall = len(selected & set(info_p.tolist())) / len(info_p)
    dist_fp = len(selected & set(dist_p.tolist())) / len(dist_p)
    rep = sel.stability_report()

    fig, ax = plt.subplots(1, 4, figsize=(14, 3.4))
    truth = c.informative.astype(float) + 0.5 * c.distractor.astype(float)
    ax[0].imshow(truth, cmap="viridis")
    ax[0].set_title("ground truth\n(bright = informative, dim = decoy)", fontsize=9)
    ax[1].imshow(c.X[0], cmap="viridis")
    ax[1].set_title("one sample image\n(both regions look identical)", fontsize=9)
    im = ax[2].imshow(sel.pi_map(), cmap="viridis", vmin=0, vmax=1)
    ax[2].set_title("selection frequency pi\n(bootstrap stability map)", fontsize=9)
    fig.colorbar(im, ax=ax[2], fraction=0.046)
    ax[3].hist(sel.pi_, bins=20, color=plt.cm.viridis(0.5))
    ax[3].set_title(f"pi histogram: bimodal\nambiguous fraction "
                    f"{rep['ambiguous_fraction']:.2f}", fontsize=9)
    ax[3].set_xlabel("selection frequency")
    for a in ax[:3]:
        a.set_xticks([]); a.set_yticks([])
    fig.tight_layout()
    fig.savefig(OUT / "ex1_ground_truth_recovery.png", dpi=120)

    takeaway("Example 1: ground-truth recovery", [
        f"informative region recall     : {recall:.2f}",
        f"equal-variance decoy selected : {dist_fp:.2f}  (0.00 is perfect)",
        f"coverage                      : {sel.coverage():.2f}",
        f"ambiguous fraction            : {rep['ambiguous_fraction']:.2f}  (bimodal pi)",
        "Variance alone cannot separate the two regions; predictive sufficiency",
        "under resampling can, and does.",
    ])


if __name__ == "__main__":
    main()
