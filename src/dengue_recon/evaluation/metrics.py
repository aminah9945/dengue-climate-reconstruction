"""Metrics on within-year share vectors (Step 6).

All share functions take non-negative vectors of length ``n_weeks_year`` (52 or
53; 12 at monthly resolution) and renormalise them first.

* :func:`jsd` - Jensen-Shannon divergence, base 2 (primary metric)
* :func:`peak_week_error` - circular distance between peak weeks
* :func:`mae_shares` - mean absolute error on shares
* :func:`emd_weeks` - earth mover's (timing) distance on cumulative shares
* :func:`week50_error` - error in the week at which 50 % of cases is reached
* :func:`crps_samples`, :func:`interval_coverage` - probabilistic scores from draws
* :func:`jsd_floor` - sampling-noise floor of the JSD for a given annual total
"""
from __future__ import annotations

import numpy as np


def normalise(p, eps: float = 0.0) -> np.ndarray:
    """Clip at 0, add ``eps`` and rescale to sum 1 (uniform if the sum is 0)."""
    p = np.clip(np.asarray(p, float), 0, None) + eps
    s = p.sum()
    return np.full_like(p, 1.0 / len(p)) if s <= 0 else p / s


def jsd(p, q, base: float = 2.0) -> float:
    """Jensen-Shannon divergence (0 = identical, 1 = disjoint when base=2)."""
    p, q = normalise(p), normalise(q)
    m = 0.5 * (p + q)

    def kl(a, b):
        mask = a > 0
        return np.sum(a[mask] * np.log(a[mask] / b[mask]))

    return float(0.5 * kl(p, m) + 0.5 * kl(q, m)) / np.log(base)


def peak_week_error(p, q, circular: bool = True) -> int:
    """|argmax(p) - argmax(q)| in weeks; circular distance by default
    (a peak in week 52 vs week 1 is 1 week apart)."""
    d = abs(int(np.argmax(p)) - int(np.argmax(q)))
    return min(d, len(p) - d) if circular else d


def mae_shares(p, q) -> float:
    """Mean absolute difference between two share vectors."""
    return float(np.mean(np.abs(normalise(p) - normalise(q))))


def emd_weeks(p, q) -> float:
    """Earth mover's distance in weeks: sum_w |F_p(w) - F_q(w)| (cumulative shares)."""
    return float(np.sum(np.abs(np.cumsum(normalise(p)) - np.cumsum(normalise(q)))))


def week50(p) -> int:
    """First week (1-based) at which the cumulative share reaches 50 %."""
    return int(np.searchsorted(np.cumsum(normalise(p)), 0.5) + 1)


def week50_error(p, q) -> int:
    """Absolute difference of the week-of-50 % between two share vectors."""
    return abs(week50(p) - week50(q))


def crps_samples(y: np.ndarray, S: np.ndarray) -> np.ndarray:
    """Sample-based CRPS, same estimator as Lim et al. (``crps_mc_safe_py``).

    y: (N,) observations, S: (N, S) draws. Returns (N,).
    """
    y = np.asarray(y, float)
    S = np.sort(np.asarray(S, float), axis=1)
    n = S.shape[1]
    a = np.mean(np.abs(S - y[:, None]), axis=1)
    k = np.arange(1, n + 1, dtype=float)
    b = (2.0 / n ** 2) * np.sum((2 * k - n - 1) * S, axis=1)
    return a - 0.5 * b


def interval_coverage(y: np.ndarray, S: np.ndarray, level: float) -> np.ndarray:
    """Boolean per row: observation inside the central ``level`` interval of the draws."""
    lo = np.quantile(S, (1 - level) / 2, axis=1)
    hi = np.quantile(S, 1 - (1 - level) / 2, axis=1)
    return (y >= lo) & (y <= hi)


def jsd_floor(p_true, total: int, rng: np.random.Generator, n_rep: int = 20) -> float:
    """Mean JSD between the true shares and multinomial resamples of the same total.

    Even a perfect model of the expected curve cannot beat this on average.
    """
    p = normalise(p_true)
    return float(np.mean([jsd(p, rng.multinomial(int(total), p)) for _ in range(n_rep)]))
