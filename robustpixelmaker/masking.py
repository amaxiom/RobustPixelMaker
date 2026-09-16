"""Soft-mask selection (Milestone 2 backend, extended).

A single GLOBAL soft mask over a patch grid (registered=True): one mask vector shared
across all images. The mask gates standardised per-patch features and a linear,
logistic or softmax head predicts the target. Training is two-phase, VTF-style:

  Phase 1  fit the head on the FULL feature set (mask = 1), then FREEZE it.
  Phase 2  train only the mask against the frozen head, with a sparsity penalty.

Freezing matters. Trained jointly, the mask and head are scale-degenerate: the head
can grow its weights to cancel a shrinking mask, so the sparsity penalty drives the
mask toward zero without changing predictions (observed as total mask collapse on
regression). With the head fixed, shrinking a gate genuinely changes predictions, so
only patches whose removal does not hurt the frozen head are pruned.

Two gate parameterisations are available:

  ``gate="hardconcrete"``   DEFAULT. Hard-Concrete / L0 gates (Louizos, Welling &
                            Kingma, ICLR 2018): stochastic and differentiable during
                            training, deterministic and exactly 0/1 at test time. The
                            penalty is on the expected COUNT of open gates, not their
                            magnitude, so a large true region is kept whole without
                            shrinkage drag.
  ``gate="sigmoid"``        m = sigmoid(theta) with an L1 penalty on sum(m). Simpler,
                            but L1 shrinks the gates it keeps (magnitude bias) and
                            only indirectly controls how many are open.

The default is set by benchmark evidence, not preference. On the synthetic sweep the
two gates are indistinguishable for small and medium informative regions (both recover
them perfectly), but on a large nine-cell region the L1 gate over-prunes and becomes
seed-dependent (recovery F1 0.80, score 0.836, stability 0.73, and F1 ranging 0.61 to
0.98 across seeds) whereas the L0 gate is near-exact and consistent (F1 0.99, score
0.922, stability 0.97) while retaining fewer patches than any competitor.

An optional spatial-smoothness (total variation) prior couples neighbouring patches so
contiguous regions are kept or dropped together. It is a regulariser on the mask, not
a feature transform, so the selection stays pixel-indexed (README.md, How it works).

Masking here is baseline-fill in standardised space: features are standardised on the
training partition, so the neutral baseline is 0 and a masked patch contributes its
training mean (that is, nothing). Renormalised "absent" masking and the frozen
representation ensemble arrive in later milestones (see CHANGELOG.md); the
gradient math and the interface do not change.

Deliberately numpy-only with hand-derived analytic gradients, so the backend runs with
no torch dependency. torch becomes an additional backend when GPU masking and
convolutional representations land; the FoldEstimator contract stays the same.
"""
from __future__ import annotations

import numpy as np

from .regimes import PatchGrid, ensure_finite
from numpy.typing import ArrayLike

# Hard-Concrete stretch constants (Louizos et al. 2018, sec.4)
HC_BETA = 2.0 / 3.0
HC_GAMMA = -0.1
HC_ZETA = 1.1


def sigmoid(z: np.ndarray) -> np.ndarray:
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50))),
                    np.exp(np.clip(z, -50, 50)) / (1.0 + np.exp(np.clip(z, -50, 50))))


class SoftMaskSelector:
    """Global patch-level soft mask plus a linear/logistic/softmax head.

    Supports ``task`` in {"binary", "multiclass", "regression"}. The head weights are
    always shaped (n_patches, K) with K = 1 for binary and regression and K = n_classes
    for multiclass, so one set of gradient expressions covers every task.
    """

    def __init__(self, task: str = "binary", patch: int = 4, lam: float = 0.02,
                 l2: float = 1e-3, lr: float = 0.08, n_iter: int = 800,
                 warmup_frac: float = 0.3, mask_threshold: float = 0.5,
                 gate: str = "hardconcrete", tv: float = 0.0, seed: int = 0,
                 fixed_mask=None, representation=None,
                 head: str = "linear", hidden: int = 16,
                 lam_scale: str = "fixed", auto_target: float = 0.15):
        if task not in ("binary", "multiclass", "regression"):
            raise ValueError("task must be one of {'binary','multiclass','regression'}")
        if gate not in ("sigmoid", "hardconcrete"):
            raise ValueError("gate must be one of {'sigmoid','hardconcrete'}")
        if head not in ("linear", "mlp"):
            raise ValueError("head must be one of {'linear','mlp'}")
        if lam_scale not in ("fixed", "auto"):
            raise ValueError("lam_scale must be one of {'fixed','auto'}")
        if not 0.0 < float(auto_target) < 1.0:
            raise ValueError("auto_target is the fraction of the mask to leave open "
                             "and must be in (0, 1)")
        self.task = task
        self.patch = patch
        self.lam = lam
        self.l2 = l2
        self.lr = lr
        self.n_iter = n_iter
        self.warmup_frac = warmup_frac
        self.mask_threshold = mask_threshold
        self.gate = gate
        self.tv = tv
        self.seed = seed
        # When supplied, the mask is not learned: the head is fitted against this fixed
        # mask and phase 2 is skipped. Used by the bootstrap aggregator to fit its final
        # model against the stability-selected region.
        # A linear head can only reward a gate that has a linear MAIN EFFECT on the
        # target. Where the informative patches act purely through interactions, or
        # symmetrically, every main effect is zero, the frozen linear head gains nothing
        # from any gate, and the mask prunes the whole informative region and reports a
        # confident empty answer (FINDINGS sec.16: recovery F1 exactly 0.000 on two of
        # four label rules). `head="mlp"` puts one hidden layer under the same
        # freeze-base scheme so a gate can earn its place through an interaction.
        self.head = head
        self.hidden = int(hidden)
        # Phase 2 trades the data gradient against lam, and that gradient's magnitude
        # depends on how well the frozen head fits. So lam does not mean the same thing
        # across heads: an MLP uses every input, no gate looks dispensable, and at the
        # linear head's lam the mask never closes at all, leaving every gate at 1.000
        # and the frequency map flat (FINDINGS sec.17).
        #
        # Rescaling the gradient by a one-off constant was tried and measurably does not
        # fix this: it moved the lam the two heads need for equal coverage from 3.0x
        # apart to 2.5x, because what differs is how steeply the data loss grows as
        # gates close, not the starting magnitude. `lam_scale="auto"` therefore searches
        # lam directly, on a geometric ladder, for the value whose mask lands nearest
        # `auto_target` open. That makes the knob "how much do you want kept", which is
        # head-independent by construction, at the cost of a few extra phase-2 passes.
        #
        # The default stays "fixed" because changing it would silently move every number
        # this library has already published.
        self.lam_scale = lam_scale
        self.auto_target = float(auto_target)
        self.fixed_mask = None if fixed_mask is None else np.asarray(fixed_mask, dtype=float)
        # A representation turns the image into per-patch feature CHANNELS. It changes
        # which patches rank highly; it never changes the coordinate system of the
        # answer, because the mask stays indexed by patch and is shared across that
        # patch's channels. None means the plain per-patch mean (one channel).
        self.representation = representation

    # -- feature pipeline (fit on train only) ------------------------------- #
    def _features(self, X) -> np.ndarray:
        """(n, n_patches, n_channels) patch features under the chosen representation."""
        if self.representation is None:
            return self.grid.pool(X)[:, :, None]
        f = np.asarray(self.representation.transform(X, self.grid), dtype=float)
        if f.ndim == 2:
            f = f[:, :, None]
        if f.shape[1] != self.grid.n_patches:
            raise ValueError(
                f"representation returned {f.shape[1]} patches, expected {self.grid.n_patches}")
        return f

    def _standardise_fit(self, f):
        """Column standardisation that survives the full float64 range.

        Two scale traps live here, both from the sibling library's defect records.
        First, the constant-feature cutoff must be RELATIVE to the data's own scale:
        an earlier absolute cutoff (sd < 1e-8) silently declared every feature
        constant on legitimately small-scale data (for example measurements in SI
        units of order 1e-9), zeroing the whole matrix and fitting a confident
        intercept-only model. Second, a textbook np.std squares raw deviations, so at
        column scale beyond about 1e154 the variance overflows, sd becomes infinite,
        and the standardised features silently become zero: the same confident
        nothing. The deviation is therefore max-normalised per column before squaring
        and the result rescaled, which is exact and finite at any representable scale.
        """
        n, P, C = f.shape
        flat = f.reshape(n, P * C)
        self._mu = flat.mean(axis=0)
        d = flat - self._mu
        scale = np.max(np.abs(d), axis=0)
        safe = np.where(scale == 0.0, 1.0, scale)
        sd = scale * np.sqrt(np.mean((d / safe) ** 2, axis=0))
        cutoff = max(np.finfo(float).tiny, 1e-12 * float(sd.max(initial=0.0)))
        self._sd = np.where(sd <= cutoff, 1.0, sd)
        return ((flat - self._mu) / self._sd).reshape(n, P, C)

    def _standardise(self, f):
        n, P, C = f.shape
        return ((f.reshape(n, P * C) - self._mu) / self._sd).reshape(n, P, C)

    def _prepare_targets(self, y):
        """Return the (n, K) target matrix and record any encoding needed at predict."""
        y = np.asarray(y)
        if self.task == "multiclass":
            self.classes_ = np.unique(y)
            idx = np.searchsorted(self.classes_, y)
            Y = np.zeros((len(y), len(self.classes_)))
            Y[np.arange(len(y)), idx] = 1.0
            return Y
        if self.task == "binary":
            self.classes_ = np.unique(y)
            if len(self.classes_) > 2:
                raise ValueError(
                    f"task='binary' but y has {len(self.classes_)} classes; "
                    "use task='multiclass' (or task=None on the facade to infer it)")
            # map to {0, 1} so arbitrary label values (e.g. {3, 8}) work
            pos = self.classes_[-1]
            return (y == pos).astype(float).reshape(-1, 1)
        # regression: standardise so MSE is O(1) and one `lam` is comparable to BCE
        self._y_mu = float(np.mean(y))
        self._y_sd = float(np.std(y)) or 1.0
        return ((y - self._y_mu) / self._y_sd).reshape(-1, 1).astype(float)

    def _loss_and_dz(self, Z, Y):
        """Task loss and its gradient wrt the head output Z, both shaped (n, K)."""
        n = Z.shape[0]
        if self.task == "multiclass":
            Zs = Z - Z.max(axis=1, keepdims=True)
            expZ = np.exp(Zs)
            Pm = expZ / expZ.sum(axis=1, keepdims=True)
            loss = -np.mean(np.sum(Y * np.log(Pm + 1e-12), axis=1))
            return loss, (Pm - Y) / n
        if self.task == "binary":
            Pm = sigmoid(Z)
            loss = -np.mean(Y * np.log(Pm + 1e-9) + (1 - Y) * np.log(1 - Pm + 1e-9))
            return loss, (Pm - Y) / n
        loss = 0.5 * np.mean((Z - Y) ** 2)
        return loss, (Z - Y) / n

    # -- spatial smoothness prior ------------------------------------------- #
    def _tv_grad(self, m):
        """Gradient of 0.5 * sum of squared 4-neighbour differences (grid Laplacian)."""
        M = m.reshape(self.grid.n_py, self.grid.n_px)
        g = np.zeros_like(M)
        d = M[1:, :] - M[:-1, :]
        g[1:, :] += d
        g[:-1, :] -= d
        d = M[:, 1:] - M[:, :-1]
        g[:, 1:] += d
        g[:, :-1] -= d
        return g.ravel()

    # -- Hard-Concrete gate helpers ----------------------------------------- #
    @staticmethod
    def _hc_sample(log_alpha, rng):
        """Stochastic gate value and d(gate)/d(log_alpha) for the training pass."""
        u = rng.uniform(1e-6, 1 - 1e-6, size=log_alpha.shape)
        s = sigmoid((np.log(u) - np.log1p(-u) + log_alpha) / HC_BETA)
        s_bar = s * (HC_ZETA - HC_GAMMA) + HC_GAMMA
        z = np.clip(s_bar, 0.0, 1.0)
        # gradient flows only where the stretched variable is strictly inside (0, 1)
        active = (s_bar > 0.0) & (s_bar < 1.0)
        dz = np.where(active, (HC_ZETA - HC_GAMMA) * s * (1 - s) / HC_BETA, 0.0)
        return z, dz

    @staticmethod
    def _hc_deterministic(log_alpha):
        """Test-time gate: no noise, exactly 0 or 1 outside the stretch interval."""
        s = sigmoid(log_alpha)
        return np.clip(s * (HC_ZETA - HC_GAMMA) + HC_GAMMA, 0.0, 1.0)

    @staticmethod
    def _hc_open_prob(log_alpha):
        """P(gate > 0), the differentiable stand-in for the L0 count penalty."""
        return sigmoid(log_alpha - HC_BETA * np.log(-HC_GAMMA / HC_ZETA))

    # -- fitting ------------------------------------------------------------ #
    def fit(self, X: ArrayLike, y: ArrayLike) -> SoftMaskSelector:
        X = ensure_finite(X, "X")
        if self.task != "regression":
            # classification labels may be non-numeric, but a float NaN label would
            # silently become its own class, and every equality test against it is
            # False, so the label mapping would break without an error
            y = np.asarray(y)
            if y.dtype.kind == "f" and np.isnan(y).any():
                raise ValueError("classification labels contain NaN")
        else:
            y = ensure_finite(y, "y")
        self.grid = PatchGrid.from_image_shape(X.shape, patch=self.patch)
        f = self._standardise_fit(self._features(X))
        Y = self._prepare_targets(y)
        n, P, C = f.shape
        K = Y.shape[1]
        self._n_channels = C
        rng = np.random.default_rng(self.seed)
        fflat = f.reshape(n, P * C)

        b1, b2, aeps = 0.9, 0.999, 1e-8

        if self.fixed_mask is not None:
            if self.fixed_mask.shape != (P,):
                raise ValueError(
                    f"fixed_mask has shape {self.fixed_mask.shape}, expected ({P},)")
            self._fit_fixed_mask(f, Y, P, C, K)
            return self

        # -- Phase 1: base head on the full feature set, then frozen ---------- #
        # The head spans patches x channels; the mask below gates whole patches, so a
        # patch is kept or dropped for all of its channels at once.
        base_steps = max(1, int(self.warmup_frac * self.n_iter))
        if self.head == "mlp":
            pars = self._mlp_init(P * C, K, rng)
            ms = [np.zeros_like(a) for a in pars]
            vs = [np.zeros_like(a) for a in pars]
            for t in range(1, base_steps + 1):
                _, _, Z = self._mlp_forward(fflat, pars)
                _, dz = self._loss_and_dz(Z, Y)
                gs = self._mlp_grads(fflat, pars, dz)
                for i, gi in enumerate(gs):
                    ms[i] = b1 * ms[i] + (1 - b1) * gi
                    vs[i] = b2 * vs[i] + (1 - b2) * gi ** 2
                    pars[i] = pars[i] - self.lr * (ms[i] / (1 - b1 ** t)) / (
                        np.sqrt(vs[i] / (1 - b2 ** t)) + aeps)
            W, c = None, None
        else:
            pars = None
            W = np.zeros((P * C, K))
            c = np.zeros(K)
            mt = np.zeros(P * C * K + K)
            vt = np.zeros(P * C * K + K)
            for t in range(1, base_steps + 1):
                Z = fflat @ W + c
                _, dz = self._loss_and_dz(Z, Y)
                grad = np.concatenate([(fflat.T @ dz + self.l2 * W).ravel(), dz.sum(axis=0)])
                mt = b1 * mt + (1 - b1) * grad
                vt = b2 * vt + (1 - b2) * grad ** 2
                step = (mt / (1 - b1 ** t)) / (np.sqrt(vt / (1 - b2 ** t)) + aeps)
                W = W - self.lr * step[:P * C * K].reshape(P * C, K)
                c = c - self.lr * step[P * C * K:]

        # -- Phase 2: freeze the head; train the gates ------------------------ #
        if self.lam_scale == "auto":
            # geometric ladder around the requested lam; pick the mask landing nearest
            # auto_target open. Degenerate masks (all open, all shut) carry no ranking
            # information at all, which is the failure this exists to prevent.
            best = None
            for mult in (0.25, 1.0, 4.0, 16.0, 64.0, 256.0):
                pr, ls = self._train_gates(self.lam * mult, f, Y, W, c, pars,
                                           P, C, K, n, base_steps, rng, b1, b2, aeps)
                mk = sigmoid(pr) if self.gate == "sigmoid" else self._hc_deterministic(pr)
                open_frac = float((mk >= self.mask_threshold).mean())
                score = abs(open_frac - self.auto_target)
                if best is None or score < best[0]:
                    best = (score, pr, ls, self.lam * mult)
            assert best is not None      # the ladder above is never empty
            params, loss = best[1], best[2]
            self.lam_used_ = best[3]
        else:
            params, loss = self._train_gates(self.lam, f, Y, W, c, pars, P, C, K,
                                             n, base_steps, rng, b1, b2, aeps)
            self.lam_used_ = self.lam

        self.gate_params_ = params
        self.mask_ = sigmoid(params) if self.gate == "sigmoid" else self._hc_deterministic(params)
        self.coef_ = W
        self.intercept_ = c
        self.head_params_ = pars
        self.selected_patches_ = np.where(self.mask_ >= self.mask_threshold)[0].astype(int)
        self.final_loss_ = loss
        return self

    def _fit_fixed_mask(self, f, Y, P, C, K):
        """Fit only the head, against a mask supplied by the caller."""
        m = self.fixed_mask
        n = f.shape[0]
        g = (f * m[None, :, None]).reshape(n, P * C)
        W = np.zeros((P * C, K))
        c = np.zeros(K)
        b1, b2, aeps = 0.9, 0.999, 1e-8
        mt = np.zeros(P * C * K + K)
        vt = np.zeros(P * C * K + K)
        loss = None
        for t in range(1, self.n_iter + 1):
            Z = g @ W + c
            loss, dz = self._loss_and_dz(Z, Y)
            grad = np.concatenate([(g.T @ dz + self.l2 * W).ravel(), dz.sum(axis=0)])
            mt = b1 * mt + (1 - b1) * grad
            vt = b2 * vt + (1 - b2) * grad ** 2
            step = (mt / (1 - b1 ** t)) / (np.sqrt(vt / (1 - b2 ** t)) + aeps)
            W = W - self.lr * step[:P * C * K].reshape(P * C, K)
            c = c - self.lr * step[P * C * K:]
        self.gate_params_ = None
        self.mask_ = m
        self.coef_ = W
        self.intercept_ = c
        # the fixed-mask path always fits a linear head; None keeps predict on it
        self.head_params_ = None
        self.selected_patches_ = np.where(m >= self.mask_threshold)[0].astype(int)
        self.final_loss_ = loss

    def _train_gates(self, lam, f, Y, W, c, pars, P, C, K, n, base_steps, rng,
                     b1, b2, aeps):
        """Phase 2 for one value of lam. Returns the gate parameters and final loss."""
    # -- Phase 2: freeze (W, c); train the gates -------------------------- #
        # sigmoid gates start at m = 0.5 (max gradient sensitivity); Hard-Concrete
        # gates start mostly open so pruning is a decision to close, not to open.
        params = np.zeros(P) if self.gate == "sigmoid" else np.full(P, 2.0)
        mt = np.zeros(P)
        vt = np.zeros(P)
        mask_steps = max(1, self.n_iter - base_steps)
        loss = None
        for t in range(1, mask_steps + 1):
            if self.gate == "sigmoid":
                m = sigmoid(params)
                dgate = m * (1 - m)
                penalty_grad = lam * dgate           # d(lam * sum m)/d(theta)
                penalty_val = lam * m.sum()
            else:
                m, dgate = self._hc_sample(params, rng)
                q = self._hc_open_prob(params)
                penalty_grad = lam * q * (1 - q)     # d(lam * E[#open])/d(log_alpha)
                penalty_val = lam * q.sum()

            g = (f * m[None, :, None]).reshape(n, P * C)
            if self.head == "mlp":
                _, _, Z = self._mlp_forward(g, pars)
                data_loss, dz = self._loss_and_dz(Z, Y)
                # chain through the frozen hidden layer, then contract the gated
                # features back onto their patch: dm_p = sum_n sum_c dL/dg[n,p,c] f[n,p,c]
                dg = self._mlp_dg(g, pars, dz).reshape(n, P, C)
                dm = np.einsum("npc,npc->p", dg, f)
            else:
                Z = g @ W + c
                data_loss, dz = self._loss_and_dz(Z, Y)
                # dLoss/dm_p sums the head's contribution over that patch's channels:
                #   dm_p = sum_c sum_k W[p,c,k] * sum_n f[n,p,c] dz[n,k]
                FD = np.einsum("npc,nk->pck", f, dz)
                dm = (W.reshape(P, C, K) * FD).sum(axis=(1, 2))
            if self.tv:
                dm = dm + self.tv * self._tv_grad(m)
            grad = dm * dgate + penalty_grad

            mt = b1 * mt + (1 - b1) * grad
            vt = b2 * vt + (1 - b2) * grad ** 2
            params = params - self.lr * (mt / (1 - b1 ** t)) / (np.sqrt(vt / (1 - b2 ** t)) + aeps)
            loss = data_loss + penalty_val

        return params, loss

    # -- MLP head: forward, and the two gradients the two phases need ------- #
    def _mlp_init(self, D, K, rng):
        """He-scaled hidden layer, zero output layer, so phase 1 starts at the mean."""
        Hd = self.hidden
        W1 = rng.normal(scale=np.sqrt(2.0 / max(1, D)), size=(D, Hd))
        return [W1, np.zeros(Hd), np.zeros((Hd, K)), np.zeros(K)]

    @staticmethod
    def _mlp_forward(g, pars):
        W1, b1_, W2, b2_ = pars
        pre = g @ W1 + b1_
        h = np.maximum(pre, 0.0)
        return pre, h, h @ W2 + b2_

    def _mlp_grads(self, g, pars, dz):
        """Parameter gradients, for phase 1 while the head is still training."""
        W1, b1_, W2, b2_ = pars
        pre, h, _ = self._mlp_forward(g, pars)
        dh = dz @ W2.T
        dpre = dh * (pre > 0)
        return [g.T @ dpre + self.l2 * W1, dpre.sum(axis=0),
                h.T @ dz + self.l2 * W2, dz.sum(axis=0)]

    def _mlp_dg(self, g, pars, dz):
        """Gradient wrt the GATED features, for phase 2 while the head is frozen.

        This is the term a linear head cannot supply: it routes through the hidden
        layer, so a patch that matters only in combination with another still receives
        gradient even when its own main effect is zero.
        """
        W1, _, W2, _ = pars
        pre, _, _ = self._mlp_forward(g, pars)
        dpre = (dz @ W2.T) * (pre > 0)
        return dpre @ W1.T

    # -- prediction --------------------------------------------------------- #
    def _decision(self, X):
        f = self._standardise(self._features(np.asarray(X, dtype=float)))
        n, P, C = f.shape
        g = (f * self.mask_[None, :, None]).reshape(n, P * C)
        if getattr(self, "head_params_", None) is not None:
            return self._mlp_forward(g, self.head_params_)[2]
        return g @ self.coef_ + self.intercept_

    def predict(self, X: ArrayLike) -> np.ndarray:
        Z = self._decision(X)
        if self.task == "regression":
            return (Z[:, 0] * self._y_sd) + self._y_mu
        if self.task == "multiclass":
            return self.classes_[Z.argmax(axis=1)]
        return np.where(sigmoid(Z[:, 0]) >= 0.5, self.classes_[-1], self.classes_[0])

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        if self.task == "regression":
            raise ValueError(
                "predict_proba is undefined for task='regression'; an earlier version "
                "silently returned a sigmoid of the regression output, which is a "
                "fabricated probability. Use predict.")
        Z = self._decision(X)
        if self.task == "multiclass":
            Zs = Z - Z.max(axis=1, keepdims=True)
            expZ = np.exp(Zs)
            return expZ / expZ.sum(axis=1, keepdims=True)
        p = sigmoid(Z[:, 0])
        return np.column_stack([1 - p, p])

    # -- selection outputs -------------------------------------------------- #
    def selected_pixels(self) -> np.ndarray:
        return self.grid.patches_to_pixels(self.selected_patches_)

    def coverage(self) -> float:
        """Fraction of patches retained."""
        return len(self.selected_patches_) / self.grid.n_patches


class SoftMaskFoldEstimator:
    """FoldEstimator adapter for :class:`SoftMaskSelector` (plugs into NestedCV).

    Obeys the leakage-safe contract: everything (standardisation, head, mask) is fitted
    on the training partition handed in by the engine. ``selected_`` reports the
    selected patch ids so NestedCV's Jaccard stability is computed over patch sets.
    """

    def __init__(self, patch: int = 4, lam: float = 0.02, l2: float = 1e-3,
                 lr: float = 0.05, n_iter: int = 500, mask_threshold: float = 0.5,
                 gate: str = "hardconcrete", tv: float = 0.0, model=None,
                 head: str = "linear", hidden: int = 16,
                 lam_scale: str = "fixed", auto_target: float = 0.15):
        self.kw = dict(patch=patch, lam=lam, l2=l2, lr=lr, n_iter=n_iter,
                       mask_threshold=mask_threshold, gate=gate, tv=tv,
                       head=head, hidden=hidden, lam_scale=lam_scale,
                       auto_target=auto_target)
        self.model = model
        self.selected_ = np.array([], dtype=int)

    def fit(self, X, y, *, task, seeds, fold_idx, repeat, inner_splitter, groups=None):
        seed = seeds.representation(0) + seeds.outer(fold_idx, repeat) if seeds is not None else 0
        self.selector_ = SoftMaskSelector(task=task, seed=int(seed), **self.kw).fit(X, y)
        self.selected_ = self.selector_.selected_patches_
        from .downstream import attach
        self.downstream_ = attach(self.model, self.selector_, X, y, task, int(seed))
        self.classes_ = (self.downstream_.classes_ if self.downstream_ is not None
                         else getattr(self.selector_, "classes_", None))
        return self

    def _predictor(self):
        """The downstream model when one was requested, else the mask's own head."""
        return self.selector_ if self.downstream_ is None else self.downstream_

    def predict(self, X: ArrayLike) -> np.ndarray:
        return self._predictor().predict(X)

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        return self._predictor().predict_proba(X)
