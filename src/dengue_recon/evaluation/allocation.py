"""Exact-total integer allocation (Step 6).

Predicted shares are multiplied by the known annual total and rounded with the
largest-remainder rule, so reconstructed weekly counts always add up exactly to
the total and no week differs from ``N * p`` by one case or more.
"""
from __future__ import annotations

import numpy as np

from .metrics import normalise


def allocate_integers(shares, total: int) -> np.ndarray:
    """Largest-remainder rounding: integer counts whose sum is exactly ``total``.

    Each slot receives ``floor(total * share)``; the leftover cases go one each to
    the slots with the largest fractional parts (stable order on ties).
    """
    shares = normalise(shares)
    raw = shares * total
    base = np.floor(raw).astype(int)
    rem = int(total - base.sum())
    if rem > 0:
        order = np.argsort(-(raw - base), kind="stable")
        base[order[:rem]] += 1
    return base


def assert_exact_sum(counts, total: int) -> None:
    """Raise ``AssertionError`` if ``counts`` do not sum to ``total`` (sanity check 1)."""
    s = int(np.asarray(counts).sum())
    assert s == int(total), f"allocation sums to {s}, expected {int(total)}"
