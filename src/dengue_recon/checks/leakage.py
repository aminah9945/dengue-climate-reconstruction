"""Leakage audit (Step 7, check 6).

Every quantity estimated from data must be computed inside the training fold
only. The audit recomputes each in-fold statistic with the held-out year's weekly
counts (and annual totals / climate) set to NaN and checks that nothing changes:

* ``h_m``           the log historical-share offset
* ``centring``      centring constants of the annual-size covariates
* ``climatology``   climatology coefficients and residual SDs
* ``train_arrays``  the held-out year never has training weight
"""
from __future__ import annotations

import numpy as np

from ..features.climate import climate_anomalies
from ..features.seasonal import offsets
from ..features.size_terms import size_covariates


def leakage_audit(P, Xc, epi_year, epi_week, train_mask, heldout_t, weekly=True):
    """Recompute every in-fold statistic with the held-out year's weekly counts
    (and climate) set to NaN; all must be identical. Returns a dict of booleans."""
    y = P.years[heldout_t]
    Y = (P.Yw if weekly else P.Ym).copy()
    Y_nan = Y.copy(); Y_nan[:, heldout_t] = np.nan
    h1, _, _ = offsets(P, train_mask, weekly, Y=Y)
    h2, _, _ = offsets(P, train_mask, weekly, Y=Y_nan)
    out = {"h_m": bool(np.array_equal(h1, h2))}
    # centring constants: also blank the held-out annual total
    N_bak = P.N.copy()
    Z1, c1 = size_covariates(P, train_mask, None, weekly)
    P.N = P.N.copy(); P.N[:, heldout_t] = np.nan
    _, c2 = size_covariates(P, train_mask, None, weekly)
    P.N = N_bak
    out["centring"] = bool(np.allclose(c1, c2, rtol=0, atol=0))
    # climatology / SD
    train_years = [yy for yy in range(2009, 2024) if yy != y]
    A1, b1, s1, _ = climate_anomalies(Xc, epi_year, epi_week, train_years)
    Xn = Xc.copy(); Xn[:, :, epi_year == y] = np.nan
    _, b2, s2, _ = climate_anomalies(Xn, epi_year, epi_week, train_years)
    out["climatology"] = bool(np.array_equal(b1, b2) and np.array_equal(s1, s2))
    # training arrays: held-out year never has training weight
    out["train_arrays"] = bool(not train_mask[:, heldout_t].any())
    return out
