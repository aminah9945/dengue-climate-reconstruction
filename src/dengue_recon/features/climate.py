"""Climate anomalies and distributed-lag block features (Step 4).

* **Climatology** (per municipality and variable, inside each fold): OLS on
  intercept + 3 annual harmonics of the epi week, fitted on 2009 plus the training
  years only (never the held-out year).
* **Anomaly**: (value - climatology) / training residual SD. Rainfall enters as
  log(1 + mm).
* **Lag blocks**: mean anomaly over lags 0-3, 4-7, 8-11, 12-15, 16-20 weeks on the
  continuous weekly timeline (crossing year boundaries); 3 variables x 5 blocks =
  15 features per week.
* Monthly resolution: the feature of a month is the mean over its epi weeks.
"""
from __future__ import annotations

import numpy as np

from .. import config as C


def climatology_design(epi_week):
    """(K, 1 + 2*N_HARM_CLIM) design: intercept + sin/cos of k * 2 pi w / 52.18."""
    w = np.asarray(epi_week, float)
    cols = [np.ones_like(w)]
    for k in range(1, C.N_HARM_CLIM + 1):
        cols += [np.sin(2 * np.pi * k * w / 52.18), np.cos(2 * np.pi * k * w / 52.18)]
    return np.stack(cols, -1)


def climate_anomalies(Xc, epi_year, epi_week, train_years):
    """Standardised anomalies from an in-fold harmonic climatology.

    Xc (V, M, K) raw weekly climate on the continuous climate index. For each
    variable/municipality the OLS climatology is fitted on weeks of ``train_years``
    only (2009 may be included); the anomaly is divided by the training residual SD.

    Returns ``(A (V,M,K), beta, sd, n_nan)``.
    """
    D = climatology_design(epi_week)
    tr = np.isin(epi_year, list(train_years))
    A = np.empty_like(Xc)
    betas, sds = [], []
    for v in range(Xc.shape[0]):
        Yv = Xc[v][:, tr].T                                   # (Ktr, M)
        beta, *_ = np.linalg.lstsq(D[tr], Yv, rcond=None)     # (P, M)
        fit = D @ beta                                        # (K, M)
        res = Xc[v].T - fit
        sd = np.sqrt(np.mean(res[tr] ** 2, axis=0))
        A[v] = (res / sd).T
        betas.append(beta); sds.append(sd)
    n_nan = int(np.isnan(A).sum())
    return A, np.stack(betas), np.stack(sds), n_nan


def lag_block_features(A, blocks=C.LAG_BLOCKS):
    """F (V*B, M, K): mean of the anomaly over the lags in each block (continuous timeline).

    Missing anomalies are set to 0 before averaging; windows that start before the
    series are NaN.
    """
    A0 = np.nan_to_num(A, nan=0.0)
    V, M, K = A0.shape
    cs = np.concatenate([np.zeros((V, M, 1)), np.cumsum(A0, -1)], -1)   # cs[k] = sum_{<k}
    out = []
    for v in range(V):
        for (lo, hi) in blocks:
            k = np.arange(K)
            a, b = k - hi, k - lo + 1            # window [k-hi, k-lo]
            valid = a >= 0
            f = np.where(valid, (cs[v][:, np.clip(b, 0, K)] - cs[v][:, np.clip(a, 0, K)]) / (hi - lo + 1), np.nan)
            out.append(f)
    return np.stack(out)   # (F, M, K)


def to_slots(F, P, weekly=True):
    """Map continuous-index features (F, M, K) to (M, T, S, F) on the panel's slot grid."""
    Fn, M, K = F.shape
    T = len(P.years)
    Xw = np.zeros((M, T, 53, Fn))
    for t in range(T):
        idx = P.week_index[t]
        ok = idx >= 0
        Xw[:, t, ok, :] = np.moveaxis(F[:, :, idx[ok]], 0, -1)
    assert np.isfinite(Xw).all()
    if weekly:
        return Xw
    Xm = np.zeros((M, T, 12, Fn))
    for t in range(T):
        for mth in range(12):
            sel = P.month_of_week[t] == mth
            Xm[:, t, mth, :] = Xw[:, t, sel, :].mean(1)
    return Xm


def climate_features(P, Xc, epi_year, epi_week, heldout_year, weekly=True, short=False):
    """The 15 M1 climate features for one LOYO fold (climatology without ``heldout_year``).

    ``short=True`` keeps only block L1 (0-3 weeks) of each variable (ablation M1-short).
    Returns ``(X (M,T,S,15), info)``.
    """
    train_years = [y for y in range(2009, 2024) if y != heldout_year]
    A, beta, sd, n_nan = climate_anomalies(Xc, epi_year, epi_week, train_years)
    F = lag_block_features(A)
    X = to_slots(F, P, weekly)
    if short:   # keep only block L1 (0-3 weeks) of each variable
        keep = np.zeros(X.shape[-1], bool)
        keep[[v * len(C.LAG_BLOCKS) for v in range(len(C.CLIM_VARS))]] = True
        X = X * keep
    return X, dict(beta=beta, sd=sd, n_nan=n_nan)
