"""Example 4: a compact signal and a diffuse one leave different fingerprints.

Two biomedical tasks, same pipeline, opposite anatomy. Blood microscopy carries its
signal in a compact object (the cell); pneumonia on chest X-ray presents as diffuse
opacity spread across the lung fields. The selection RPM returns reflects that
difference in two measurable ways, and honestly, they are not the same size:

  * contiguity of the selected region (fraction of selected patches with a selected
    neighbour): near-perfect for the compact cell, clearly fragmented for the diffuse
    opacity. This is the strong signal.
  * ambiguous fraction of the frequency map: also ordered the right way, but the gap
    is modest here (both tasks are well powered at n=3000). The dramatic ambiguity
    gap belongs to the underpowered regime, which is example 6's story, not this one.

Science takeaway: read the selected region's SHAPE, not just its size. A fragmented
selection on a well-powered dataset says the evidence is spread out, matching the
radiology (opacity is not an object), and warns against interpreting any single
sub-region as "the" finding.
"""
import numpy as np

from _common import OUT, style, takeaway
from benchmarks.datasets import load_medmnist
from robustpixelmaker import BootstrapMaskSelector, PatchGrid


def contiguity(sel_patches, grid) -> float:
    """Fraction of selected patches with at least one selected 4-neighbour."""
    if len(sel_patches) < 2:
        return 0.0
    on = np.zeros((grid.n_py, grid.n_px), dtype=bool)
    for p in sel_patches:
        on[int(p) // grid.n_px, int(p) % grid.n_px] = True
    nb = np.zeros_like(on)
    nb[1:, :] |= on[:-1, :]
    nb[:-1, :] |= on[1:, :]
    nb[:, 1:] |= on[:, :-1]
    nb[:, :-1] |= on[:, 1:]
    return float((on & nb).sum() / on.sum())


#: both datasets are compared at exactly this many patches, not approximately
MATCHED_COVERAGE = 0.12


def fit_one(key, task, n_max=3000, seed=0):
    """Matched LEAN coverage for both datasets, so shape is compared like for like.

    At generous coverage (0.3 to 0.4) any selection is contiguous simply because it is
    large; the shape difference only becomes measurable when both are forced to commit
    to their most-informative 12%. Comparing shapes at matched coverage is the same
    lesson the benchmark suite drew for score and stability (FINDINGS sec.9).

    This originally asked for it with ``target_coverage=0.12`` and did not get it.
    ``pi_`` is quantised, and on blood 24.5% of patches sit at pi_ exactly 1.0, so no
    threshold yields 0.12 and the request silently returned 0.245. The comparison was
    then 0.245 against 0.122 while claiming to be matched, and contiguity is precisely
    the quantity coverage inflates: pneumonia scores 0.50 at 0.122 and 0.83 at 0.245.
    ``top_patches(k)`` returns exactly k, so the match is now enforced rather than
    requested. The conclusion is unchanged (FINDINGS sec.22).
    """
    X, y, meta = load_medmnist(key, n_max=n_max)
    grid = PatchGrid.from_image_shape(X.shape, patch=4)
    sel = BootstrapMaskSelector(task=task, patch=4, lam=0.03, n_bootstrap=12,
                                n_iter=400, seed=seed).fit(X, y)
    k = max(1, int(round(MATCHED_COVERAGE * grid.n_patches)))
    patches = sel.top_patches(k)
    return X, sel, patches, sel.stability_report(), contiguity(patches, grid)


def main():
    plt = style()
    Xb, sel_b, pat_b, rep_b, cont_b = fit_one("blood", "multiclass")
    Xp, sel_p, pat_p, rep_p, cont_p = fit_one("pneumonia", "binary")

    fig, ax = plt.subplots(2, 3, figsize=(11, 6.6))
    for row, (X, sel, patches, rep, cont, name) in enumerate([
            (Xb, sel_b, pat_b, rep_b, cont_b, "blood microscopy (compact object)"),
            (Xp, sel_p, pat_p, rep_p, cont_p,
             "chest X-ray pneumonia (diffuse opacity)")]):
        grid = PatchGrid.from_image_shape(X.shape, patch=4)
        # the region DRAWN must be the region MEASURED. Panel three used the tau
        # threshold while contiguity is computed on the matched top-k set, and a
        # figure that disagrees with its own caption is how ex5 went wrong once.
        region = np.isin(grid.pixel_patch, patches)
        ax[row, 0].imshow(X.mean(axis=0), cmap="viridis")
        ax[row, 0].set_title(f"{name}\nmean image", fontsize=9)
        im = ax[row, 1].imshow(sel.pi_map(), cmap="viridis", vmin=0, vmax=1)
        ax[row, 1].set_title("selection frequency", fontsize=9)
        fig.colorbar(im, ax=ax[row, 1], fraction=0.046)
        ax[row, 2].imshow(X.mean(axis=0) * region, cmap="viridis")
        ax[row, 2].set_title(
            f"selected region, matched coverage "
            f"{len(patches) / grid.n_patches:.2f}\ncontiguity {cont:.2f}",
            fontsize=9)
        for a in (ax[row, 0], ax[row, 1], ax[row, 2]):
            a.set_xticks([]); a.set_yticks([])
    fig.tight_layout()
    fig.savefig(OUT / "ex4_diffuse_signal_diagnosis.png", dpi=120)

    takeaway("Example 4: compact versus diffuse signal, read from the selection", [
        f"blood     : contiguity {cont_b:.2f}, ambiguous {rep_b['ambiguous_fraction']:.2f}",
        f"pneumonia : contiguity {cont_p:.2f}, ambiguous {rep_p['ambiguous_fraction']:.2f}",
        f"both compared at exactly {MATCHED_COVERAGE:.2f} coverage, enforced "
        f"with top_patches, not requested",
        "The fragmentation gap is the strong diagnostic here; both tasks are well",
        "powered, so ambiguity separates only mildly (severe ambiguity is the",
        "underpowered signature, shown in example 6).",
    ])


if __name__ == "__main__":
    main()
