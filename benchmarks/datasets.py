"""Real-world dataset loaders for the RobustPixelMaker benchmarks.

Scope: scientific imaging only. Every dataset here is either a registered scientific
image set or a canonical registered digit benchmark (MNIST), which is included because
it is centred (hence registered), because the VTF paper used the hard 3-vs-8 subset,
and because it forces multiclass support. Consumer/product image sets are out of scope
by design.

Loaders return ``(X, y, meta)`` with ``X`` shaped (n, H, W) as float, ``y`` the target,
and ``meta`` a dict describing the set. Downloads are cached under ``benchmarks/data``
so a benchmark re-run does not refetch.

Currently implemented: MNIST digits (10-class and the 3-vs-8 binary subset).
Planned next, in priority order (see BENCHMARK_PLAN section of benchmarks/README.md):
Galaxy10 DECaLS, PatchCamelyon, OASIS / ABIDE brain MRI, NFFA-EUROPE SEM, cryo-EM
single-particle stacks from EMPIAR.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent / "data"


# Canonical MNIST mirror used by Keras: a plain npz, no extra dependency needed.
MNIST_NPZ_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"


def _mnist_raw() -> tuple[np.ndarray, np.ndarray]:
    """Fetch MNIST once and cache it as a compressed npz (n, 28, 28) uint8 + labels.

    Primary source is the Keras npz mirror (fast, no parsing). If that is unreachable
    we fall back to OpenML, which has been observed to return 504s under load.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache = DATA_DIR / "mnist.npz"
    if cache.exists():
        z = np.load(cache)
        return z["X"], z["y"]

    X = y = None
    try:
        import urllib.request

        raw = DATA_DIR / "mnist_keras_raw.npz"
        if not raw.exists():
            urllib.request.urlretrieve(MNIST_NPZ_URL, raw)
        with np.load(raw, allow_pickle=True) as z:
            X = np.concatenate([z["x_train"], z["x_test"]]).astype(np.uint8)
            y = np.concatenate([z["y_train"], z["y_test"]]).astype(int)
    except Exception as exc:  # noqa: BLE001 - fall back to the other source
        print(f"keras mirror unavailable ({exc}); trying OpenML")
        from sklearn.datasets import fetch_openml

        bunch = fetch_openml("mnist_784", version=1, as_frame=False, parser="liac-arff")
        X = np.asarray(bunch.data, dtype=np.uint8).reshape(-1, 28, 28)
        y = np.asarray(bunch.target).astype(int)

    np.savez_compressed(cache, X=X, y=y)
    return X, y


def load_mnist(subset: str = "3v8", n_max: int | None = 4000, seed: int = 0):
    """MNIST digits, scaled to [0, 1].

    ``subset="3v8"`` is the hard binary pair used in the VTF paper; ``subset="all"``
    is the full 10-class problem. ``n_max`` subsamples (class-stratified) so benchmark
    runs stay tractable; pass None to use everything.
    """
    X, y = _mnist_raw()
    if subset == "3v8":
        keep = np.isin(y, [3, 8])
        X, y = X[keep], y[keep]
        task = "binary"
    elif subset == "all":
        task = "multiclass"
    else:
        raise ValueError("subset must be '3v8' or 'all'")

    if n_max is not None and len(y) > n_max:
        rng = np.random.default_rng(seed)
        classes = np.unique(y)
        per = max(1, n_max // len(classes))
        idx = np.concatenate([
            rng.choice(np.where(y == cls)[0], size=min(per, int((y == cls).sum())), replace=False)
            for cls in classes
        ])
        idx.sort()
        X, y = X[idx], y[idx]

    meta = dict(name=f"mnist-{subset}", task=task, shape=X.shape[1:],
                n=len(y), classes=sorted(np.unique(y).tolist()), registered=True,
                source="OpenML mnist_784")
    return X.astype(float) / 255.0, y, meta


# --------------------------------------------------------------------------- #
# MedMNIST v2 (Yang et al., Scientific Data 2023): standardised 28x28 biomedical
# images. Ideal here because they are genuinely scientific, centred (hence
# registered), openly hosted, and the same grid as MNIST, so the patch geometry and
# every metric carry over unchanged and the two results are directly comparable.
# --------------------------------------------------------------------------- #
MEDMNIST_URL = "https://zenodo.org/records/10519652/files/{name}.npz?download=1"

#: key -> (zenodo file stem, modality description, whether the source is RGB)
MEDMNIST_SETS = {
    "pneumonia": ("pneumoniamnist", "paediatric chest X-ray, normal vs pneumonia", False),
    "breast": ("breastmnist", "breast ultrasound, malignant vs normal/benign", False),
    "organa": ("organamnist", "abdominal CT, axial, 11 organs", False),
    "oct": ("octmnist", "retinal OCT, 4 diagnoses", False),
    "tissue": ("tissuemnist", "kidney cortex microscopy, 8 tissue types", False),
    "blood": ("bloodmnist", "blood cell microscopy, 8 cell types", True),
    "derma": ("dermamnist", "dermatoscopy, 7 lesion types", True),
    "path": ("pathmnist", "colon pathology, 9 tissue types", True),
    "retina": ("retinamnist", "retinal fundus, 5 grades", True),
}


def _medmnist_raw(key: str):
    stem, _, _ = MEDMNIST_SETS[key]
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache = DATA_DIR / f"{stem}.npz"
    if not cache.exists():
        import urllib.request

        urllib.request.urlretrieve(MEDMNIST_URL.format(name=stem), cache)
    return np.load(cache)


def load_medmnist(key: str = "pneumonia", n_max: int | None = 4000, seed: int = 0):
    """A MedMNIST v2 dataset, scaled to [0, 1] and shaped (n, 28, 28).

    The official train/val/test splits are concatenated: this benchmark supplies its
    own cross-validation, and mixing the splits would otherwise hide part of the data
    from every method equally but for no useful reason. RGB sources are converted to
    luminance because the mask is defined over a single-channel patch grid; the
    conversion is applied identically for every method, so comparisons are unaffected.
    """
    if key not in MEDMNIST_SETS:
        raise ValueError(f"unknown MedMNIST set {key!r}; options: {sorted(MEDMNIST_SETS)}")
    stem, description, is_rgb = MEDMNIST_SETS[key]
    z = _medmnist_raw(key)
    X = np.concatenate([z["train_images"], z["val_images"], z["test_images"]])
    y = np.concatenate([z["train_labels"], z["val_labels"], z["test_labels"]]).ravel().astype(int)

    if is_rgb or X.ndim == 4:
        X = (0.2126 * X[..., 0] + 0.7152 * X[..., 1] + 0.0722 * X[..., 2])

    classes = np.unique(y)
    task = "binary" if len(classes) == 2 else "multiclass"

    if n_max is not None and len(y) > n_max:
        rng = np.random.default_rng(seed)
        per = max(1, n_max // len(classes))
        idx = np.concatenate([
            rng.choice(np.where(y == c)[0], size=min(per, int((y == c).sum())), replace=False)
            for c in classes
        ])
        idx.sort()
        X, y = X[idx], y[idx]

    meta = dict(name=f"medmnist-{key}", task=task, shape=X.shape[1:], n=len(y),
                classes=sorted(np.unique(y).tolist()), registered=True,
                modality=description, source=f"MedMNIST v2 ({stem})")
    return X.astype(float) / 255.0, y, meta


LOADERS = {
    "mnist-3v8": lambda **kw: load_mnist(subset="3v8", **kw),
    "mnist-all": lambda **kw: load_mnist(subset="all", **kw),
}
LOADERS.update({
    f"medmnist-{k}": (lambda k=k: (lambda **kw: load_medmnist(key=k, **kw)))()
    for k in MEDMNIST_SETS
})
