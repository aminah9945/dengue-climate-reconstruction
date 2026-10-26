"""Baselines that need no fitting (Step 3).

======  =========================================================================
B0      uniform shares, 1 / W_t
B1      seasonal profile: exp(h_m), i.e. the municipality's mean training share
        (leave-one-year-out), renormalised over the real weeks of year t
B1p     the same profile from years *before* t only (no future information)
======  =========================================================================

B1 and B1p use the in-fold profile computed by
:func:`dengue_recon.features.seasonal.offsets` / ``past_only_profile``, so they
cannot see the held-out year.
"""
from __future__ import annotations

import numpy as np

from ..evaluation.metrics import normalise
from ..features.seasonal import adapt_length, past_only_profile


def uniform_shares(n: int) -> np.ndarray:
    """B0: equal share for each of the ``n`` slots."""
    return np.full(n, 1.0 / n)


def seasonal_shares(profile_m: np.ndarray, n: int, weekly: bool = True) -> np.ndarray:
    """B1: the fold's training profile of one municipality, cut to ``n`` slots and renormalised."""
    return adapt_length(profile_m, n) if weekly else normalise(profile_m)


def past_only_shares(P, m: int, t: int, train_ok: np.ndarray, n: int, weekly: bool = True):
    """B1p: profile from training rows of years before ``t`` only; ``None`` if there are none."""
    pp = past_only_profile(P, m, t, train_ok, weekly)
    if pp is None:
        return None
    return adapt_length(pp, n) if weekly else normalise(pp)
