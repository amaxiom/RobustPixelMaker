"""Example 5: the answer depends on the lens, so integrate the lens out.

Four representations of the same noisy images (raw patch means, texture statistics, a
coarser blur scale, random convolutional features) are each given the same task. On
this control every lens FINDS the planted region (recall 1.00 for all four), so recall
is not what separates them. What separates them is how much else they drag in: the
random-convolution lens fires on many uninformative patches (precision 0.36) while the
plain mean lens is clean (1.00), and their pairwise agreement is only about 0.68.

Marginalising over the lenses reaches precision 1.00, MATCHING the best single lens
rather than beating it. That is the honest claim, and it is still the useful one: the
best lens cannot be identified in advance without the ground truth real data does not
come with, so integrating the lens out buys insurance against picking a bad one.

Science takeaway: the representation is a nuisance variable. Treating it as one is the
method's central claim, demonstrated here against known ground truth. Reported below as
precision alongside recall, because precision is the axis on which the lenses differ.
"""
import numpy as np

from _common import OUT, style, takeaway
from robustpixelmaker import (
    PatchGrid,
    RepresentationEnsembleSelector,
    make_synthetic_images,
)


def main():
    plt = style()
    c = make_synthetic_images(n=400, task="binary", noise=2.0, seed=1)
    grid = PatchGrid.from_image_shape(c.X.shape, patch=4)
    info_p = set(np.unique(grid.pixel_patch[c.informative]).tolist())

    sel = RepresentationEnsembleSelector(
        task="binary", patch=4, lam=0.03, n_bootstrap=6, n_representations=4,
        tau=0.6, n_iter=400, seed=1).fit(c.X, c.y)

    def scores(selected):
        """Recall and precision of a selected patch set against the planted truth."""
        s = set(int(p) for p in selected)
        rec = len(s & info_p) / len(info_p)
        prec = len(s & info_p) / len(s) if s else 0.0
        return rec, prec

    names = list(sel.pi_by_representation_)
    fig, ax = plt.subplots(1, len(names) + 1, figsize=(3.0 * (len(names) + 1), 3.6))
    per_lens = []
    for a, name in zip(ax[:-1], names):
        pi_r = sel.pi_by_representation_[name]
        a.imshow(grid.expand(pi_r), cmap="viridis", vmin=0, vmax=1)
        rec, prec = scores(np.where(pi_r >= sel.tau_)[0])
        per_lens.append((name, rec, prec))
        a.set_title(f"lens: {name}\nrecall {rec:.2f}, precision {prec:.2f}", fontsize=9)
        a.set_xticks([]); a.set_yticks([])
    ax[-1].imshow(grid.expand(sel.pi_), cmap="viridis", vmin=0, vmax=1)
    rec, prec = scores(sel.selected_patches_)
    ax[-1].set_title(f"marginalised over lenses\nrecall {rec:.2f}, precision {prec:.2f}",
                     fontsize=9)
    ax[-1].set_xticks([]); ax[-1].set_yticks([])
    fig.tight_layout()
    fig.savefig(OUT / "ex5_representation_marginalisation.png", dpi=120)

    rep = sel.stability_report()
    best = max(per_lens, key=lambda r: r[2])
    worst = min(per_lens, key=lambda r: r[2])
    recalls = [r for _, r, _ in per_lens]
    takeaway("Example 5: marginalising over representations", [
        f"pairwise lens agreement   : {rep['representation_agreement']:.2f} "
        "(1.00 would be full agreement)",
        f"every lens finds the region: recall {min(recalls):.2f} to {max(recalls):.2f}, "
        "so recall does not separate them",
        f"precision by lens         : {worst[2]:.2f} ({worst[0]}, worst) to "
        f"{best[2]:.2f} ({best[0]}, best)",
        f"marginalised              : recall {rec:.2f}, precision {prec:.2f}, "
        f"coverage {sel.coverage():.2f}",
        "The lens you pick sets how much noise you carry. You cannot know in",
        "advance which one would have been right, so integrate it out.",
    ])


if __name__ == "__main__":
    main()
