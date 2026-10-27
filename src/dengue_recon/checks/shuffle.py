"""Climate-shuffle sanity checks (Step 7, checks 4 and 5).

* **Train-and-test shuffle (M1-shufTT).** For each municipality, a random
  derangement of the years is applied to the rows of the feature array: whole-year
  blocks of lag features (computed from the real contiguous series) are moved to
  other years. The model is refitted in every fold, with 3 seeds. A real climate
  signal must disappear.
* **Test-only shuffle (M1-shufTest).** The trained M1 is given another year's
  features in the held-out year. It must get worse.
"""
from __future__ import annotations

import numpy as np

from .. import config as C


def derangement(n, rng):
    """Random permutation of ``range(n)`` with no fixed point."""
    while True:
        p = rng.permutation(n)
        if not (p == np.arange(n)).any():
            return p


def train_test_permutations(M, T, n_shuffle=3, seed=C.SEED):
    """One (M, T) year derangement per shuffle seed (seed ``C.SEED + s``)."""
    rng_shuf = [np.random.default_rng(seed + s) for s in range(n_shuffle)]
    return [np.stack([derangement(T, r) for _ in range(M)]) for r in rng_shuf]


def test_only_permutation(M, T, seed=C.SEED):
    """(M, T) derangements used by the test-only shuffle (one generator per municipality)."""
    return np.stack([derangement(T, np.random.default_rng(seed + 99 + m)) for m in range(M)])


def shuffle_years(X, perm):
    """Train-and-test shuffle: X[m, t] <- X[m, perm[m, t]] for all years."""
    M = X.shape[0]
    return X[np.arange(M)[:, None], perm]


def shuffle_test_year(X, t, test_perm):
    """Test-only shuffle: replace only the held-out year ``t`` with a deranged year's features."""
    M = X.shape[0]
    Xs = X.copy()
    Xs[:, t] = X[np.arange(M), test_perm[:, t]]
    return Xs


def fold_year_permutations(M, T, year_idx, seed=C.SEED):
    """Train-and-test shuffle restricted to the fold's years (official designs).

    ``year_idx`` are the panel positions of the fold's training years and test year.
    For each municipality the features of these years are deranged among themselves
    (generator ``seed``, one draw per municipality in order); other years are left in
    place, so no climate from outside the fold (e.g. later years) is used. With all
    years of the panel this equals :func:`train_test_permutations` seed 0.
    """
    year_idx = np.asarray(year_idx)
    rng = np.random.default_rng(seed)
    perm = np.tile(np.arange(T), (M, 1))
    for m in range(M):
        perm[m, year_idx] = year_idx[derangement(len(year_idx), rng)]
    return perm


def fold_test_permutation(M, T, year_idx, seed=C.SEED):
    """Test-only shuffle restricted to the fold's years: (M, T) derangements among
    ``year_idx`` (one generator per municipality, as :func:`test_only_permutation`)."""
    year_idx = np.asarray(year_idx)
    perm = np.tile(np.arange(T), (M, 1))
    for m in range(M):
        perm[m, year_idx] = year_idx[derangement(len(year_idx), np.random.default_rng(seed + 99 + m))]
    return perm
