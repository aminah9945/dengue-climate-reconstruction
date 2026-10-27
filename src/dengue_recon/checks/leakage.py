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

from .. import config as C
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


def official_leakage_audit(P, Xc, epi_year, epi_week, train_mask, fold, clim_years, oni=None, weekly=True):
    """Leakage audit for one official fold (designs A, B, C).

    Every year that is not a training year of the fold (the test year and, in
    design C, the other non-training years) has its weekly counts, annual totals,
    climate and ONI set to NaN (for the climatology: every week outside
    ``clim_years``); the offsets, size centring, climatology coefficients and SDs,
    and the ONI mean and SD must not change. Also checks that no training weight
    falls outside the training years and, for designs A and B, that no training
    year is >= the test year and no year after the test year is in the panel.
    """
    from ..features.climate import oni_standardise
    tr_years = set(fold["train_years"])
    other = ~np.isin(P.years, list(tr_years))
    Y = (P.Yw if weekly else P.Ym)
    Y_nan = Y.copy(); Y_nan[:, other] = np.nan
    h1, _, _ = offsets(P, train_mask, weekly, Y=Y)
    h2, _, _ = offsets(P, train_mask, weekly, Y=Y_nan)
    out = {"h_m": bool(np.array_equal(h1, h2))}
    N_bak = P.N
    _, c1 = size_covariates(P, train_mask, None, weekly)
    P.N = P.N.copy(); P.N[:, other] = np.nan
    try:
        _, c2 = size_covariates(P, train_mask, None, weekly)
    finally:
        P.N = N_bak
    out["centring"] = bool(np.array_equal(c1, c2))
    keep = epi_year <= int(P.years.max())
    ey, ew, X0 = epi_year[keep], epi_week[keep], Xc[:, :, keep]
    _, b1, s1, _ = climate_anomalies(X0, ey, ew, clim_years)
    blank = ~np.isin(ey, list(clim_years))
    Xn = X0.copy(); Xn[:, :, blank] = np.nan
    _, b2, s2, _ = climate_anomalies(Xn, ey, ew, clim_years)
    out["climatology"] = bool(np.array_equal(b1, b2) and np.array_equal(s1, s2))
    out["clim_years_ok"] = bool(set(clim_years) <= (tr_years | set(C.CLIM_EXTRA_YEARS)))
    if oni is not None:
        o = np.asarray(oni, float)[keep]
        _, m1, d1 = oni_standardise(o, ey, clim_years)
        on = o.copy(); on[blank] = np.nan
        _, m2, d2 = oni_standardise(on, ey, clim_years)
        out["oni_std"] = bool(m1 == m2 and d1 == d2)
    out["train_arrays"] = bool(not train_mask[:, other].any())
    if fold["design"] in ("A", "B"):
        out["no_future_train"] = bool(max(tr_years) < fold["test_year"])
        out["no_future_in_panel"] = bool(int(P.years.max()) == fold["test_year"])
    return out
