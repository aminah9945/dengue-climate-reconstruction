"""Exact-sum checks (Step 7, check 1).

Every reconstruction (largest-remainder allocation) and every probabilistic draw
must add up to the known annual total. The assertions run inside scoring
(:func:`dengue_recon.evaluation.scoring.score`) and inside
:func:`dengue_recon.models.dirichlet_multinomial.draw_counts`; these helpers make
the check reusable.
"""
from __future__ import annotations

import numpy as np

from ..evaluation.allocation import allocate_integers


def allocation_sum_ok(shares, total: int) -> bool:
    """True if the largest-remainder allocation of ``shares`` sums to ``total``."""
    return int(allocate_integers(shares, total).sum()) == int(total)


def draws_sum_ok(draws: np.ndarray, total: int) -> bool:
    """True if every row of ``draws`` (n_draws, n_slots) sums to ``total``."""
    return bool((np.asarray(draws).sum(1) == int(total)).all())
