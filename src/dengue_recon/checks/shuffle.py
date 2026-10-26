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
