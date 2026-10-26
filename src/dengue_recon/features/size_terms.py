"""Annual-size covariates for B4s (Step 4).

B4s adds ``sum_j [gamma_j^s * n_mt + gamma_j^p * n_m,t-1] * H_j(w)``: the centred
log annual total of the target year and of the previous year, each interacted with
two week harmonics. These are known for a held-out year (the annual total is the
input of the reconstruction), so they belong in the comparator, not in the climate
effect.
"""
from __future__ import annotations

import numpy as np


def harmonics(S, weekly=True):
    """(S, 4) matrix [sin, cos] x j = 1, 2 of the slot index (period 52.18 weeks or 12 months)."""
    W = 52.18 if weekly else 12.0
    w = np.arange(1, S + 1)
    return np.stack([f(2 * np.pi * j * w / W) for j in (1, 2) for f in (np.sin, np.cos)], -1)  # (S,4)


def size_covariates(P, train_mask, usable, weekly=True):
    """Size design ``Z`` (M, T, S, 12) = [n_mt * H(4), n_prev * H(4), ind_prev_missing * H(4)].

    n = log1p(N) minus the municipality's fold-training mean of log1p(N).
    The previous-year total is 'missing' for 2010 (no 2009 dengue data) or when
    the previous year is excluded; then n_prev = 0 and the indicator is 1.
    ``usable`` is unused (kept for signature compatibility).

    Returns ``(Z, centre)`` where ``centre`` (M,) are the fold-training centring constants.
    """
    S = 53 if weekly else 12
    H = harmonics(S, weekly)
    lN = np.log1p(P.N)
    w = train_mask.astype(float)
    centre = np.where(train_mask, lN, 0.0).sum(1) / np.maximum(w.sum(1), 1)
    centre = np.where(w.sum(1) > 0, centre, np.nanmean(centre[w.sum(1) > 0]))
    n_cur = lN - centre[:, None]
    n_prev = np.zeros_like(n_cur)
    ind = np.ones_like(n_cur)
    ok_prev = ~P.excluded[:, :-1]
    n_prev[:, 1:] = np.where(ok_prev, lN[:, :-1] - centre[:, None], 0.0)
    ind[:, 1:] = np.where(ok_prev, 0.0, 1.0)
    Z = np.concatenate([n_cur[..., None, None] * H, n_prev[..., None, None] * H,
                        ind[..., None, None] * H], -1)
    return Z, centre
