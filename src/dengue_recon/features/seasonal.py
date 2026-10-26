"""Seasonal profiles and the in-fold log-share offset h_m (Step 4).

The offset h_m(w) = log(mean training share + eps) is the base paper's
"historical log-shares" moved to municipality-week resolution. It is the fixed
starting point of every Dirichlet-multinomial model and, renormalised, it *is*
the seasonal-profile baseline B1.

Weekly profiles live on a 53-slot grid: for 52-week years a pseudo week 53 is
inserted as the mean of weeks 52 and 1, and profiles are cut back to the length
of the target year with :func:`adapt_length`.
"""
from __future__ import annotations

import numpy as np

from .. import config as C
from ..evaluation.metrics import normalise


def adapt_length(profile53: np.ndarray, n: int) -> np.ndarray:
    """Map a 53-slot week profile onto a 52- or 53-week year and renormalise.

    For 52-week years slot 53 is dropped; profiles are always estimated on 53 slots
    where slot 53 is filled (when absent) by the mean of slots 52 and 1.
    """
    p = np.asarray(profile53, float)[:n]
    return normalise(p)


def seasonal_profile(share_by_year: dict[int, np.ndarray], exclude: set[int] | None = None,
                     only_before: int | None = None) -> np.ndarray | None:
    """Historical mean weekly share on a 53-slot grid.

    share_by_year: {year: share vector (52 or 53)} for COMPLETE years only.
    exclude: years to leave out (leave-one-year-out -> {test_year}).
    only_before: if given, use only years < only_before (no future information).
    Returns None if no training years remain.
    """
    rows = []
    for y, s in share_by_year.items():
        if exclude and y in exclude:
            continue
        if only_before is not None and y >= only_before:
            continue
        s = np.asarray(s, float)
        if len(s) == 52:
            s = np.append(s, 0.5 * (s[51] + s[0]))  # pseudo week 53
            s = s / s.sum()
        rows.append(s)
    if not rows:
        return None
    return np.mean(rows, axis=0)


def _shares(Y, N, t, nweeks_t, weekly):
    if weekly:
        return Y[t, :nweeks_t] / N[t]
    return Y[t] / N[t]


def offsets(P, train_mask, weekly=True, eps=C.EPS_H, Y=None):
    """In-fold offset h[m, s] = log(mean training share + eps).

    Weekly: 53-slot grid via :func:`seasonal_profile` (pseudo week 53). Monthly:
    plain mean of the 12 monthly shares. Municipalities with < 3 training years use
    their region's mean training share. Only rows with ``train_mask`` are read.

    Returns ``(h (M,S), used_region_fallback (M,) bool, profile (M,S))``.
    """
    Y = (P.Yw if weekly else P.Ym) if Y is None else Y
    M, T = train_mask.shape
    S = 53 if weekly else 12
    prof = np.full((M, S), np.nan)
    fallback = np.zeros(M, bool)
    per_row = {}
    for m in range(M):
        sb = {}
        for t in range(T):
            if train_mask[m, t]:
                sb[int(P.years[t])] = (_shares(Y[m], P.N[m], t, P.nweeks[t], weekly))
        per_row[m] = sb
        if len(sb) >= 3:
            prof[m] = seasonal_profile(sb) if weekly else np.mean(list(sb.values()), axis=0)
        else:
            fallback[m] = True
    for r in range(len(C.REGIONS)):
        rows = [s for m in np.where(P.region == r)[0] for s in per_row[m].values()]
        if not rows:
            continue
        if weekly:
            reg = seasonal_profile({i: s for i, s in enumerate(rows)})
        else:
            reg = np.mean(rows, axis=0)
        prof[(P.region == r) & fallback] = reg
    assert np.isfinite(prof).all()
    return np.log(prof + eps), fallback, prof


def past_only_profile(P, m, t, train_ok, weekly=True):
    """Profile for B1p: mean training share using years < t only (rows with ``train_ok``)."""
    Y = P.Yw if weekly else P.Ym
    sb = {int(P.years[k]): _shares(Y[m], P.N[m], k, P.nweeks[k], weekly)
          for k in range(t) if train_ok[m, k]}
    if not sb:
        return None
    return seasonal_profile(sb) if weekly else np.mean(list(sb.values()), axis=0)
