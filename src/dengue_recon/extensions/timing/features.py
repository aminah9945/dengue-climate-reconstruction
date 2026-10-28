"""Feature blocks of the timing-informed share model (EXPLORATORY).

All choices below were fixed before any model was fitted:

* **Smoothing**: 3-week centred moving average *within the epi year* (the two
  edge weeks average the available neighbours), so no week of another year
  enters a year's curve.
* **Share and log-share**: smoothed counts / their year sum; ``log(share + EPS)``
  with ``EPS = 1e-3`` (an average week holds about 0.019).
* **Missing signal**: a (municipality, year) whose raw signal total is below
  ``MIN_SIGNAL_TOTAL`` (20 cases or 20 index points; also the state-years
  flagged as excluded, e.g. ES 2021) has no curve: its anomaly is 0 in every week
  (after standardisation a constant across weeks, which cancels in the softmax).
* **Climatology**: for each municipality's signal, the mean log-share per week
  over the fold's *training years* with a curve (53-slot grid; 52-week years get a
  pseudo week 53 = mean of the log-shares of weeks 52 and 1).
* **Anomaly**: same-year log-share minus that climatology, i.e. how this year's
  regional/search curve departs from its usual season.
* **Lags** (fixed sets): state curve ``x_L[w] = a[w - L]`` for L in
  (-2, -1, 0, 1, 2) (2 weeks lead to 2 weeks lag); search ``x_L[w] = a[w + L]`` for
  L in (0, 1, 2, 3) (searches follow cases by about one week, so the search value
  L weeks *later* informs week w). Indices are clipped to the year (edge weeks
  repeat), never crossing into another year.
* **Standardisation**: each feature column is centred and scaled by its mean and
  SD over the fold's training rows (training-mask rows, valid weeks) only.
* **Timing shift (T1)**: an integer shift delta in -10..+10 weeks maximising the
  Pearson correlation between the same-year smoothed state share curve and the
  state's training-year mean share curve rolled by delta (ties: smallest |delta|);
  delta = 0 when the year has no curve. The municipality's own offset h_m is
  rolled by delta (circularly over the year's weeks).
"""
from __future__ import annotations

import numpy as np

EPS = 1e-3
MIN_SIGNAL_TOTAL = 20.0
STATE_LAGS = (-2, -1, 0, 1, 2)
TRENDS_LAGS = (0, 1, 2, 3)
MAX_SHIFT = 10


def smooth3(v: np.ndarray) -> np.ndarray:
    """Centred 3-week moving average within one year; edges average the available neighbours."""
    v = np.asarray(v, float)
    n = len(v)
    s = v.copy()
    if n >= 2:
        s[1:-1] = (v[:-2] + v[1:-1] + v[2:]) / 3.0
        s[0] = (v[0] + v[1]) / 2.0
        s[-1] = (v[-2] + v[-1]) / 2.0
    return s


def year_curves(V, nweeks, missing=None):
    """Smoothed shares and log-shares per (m, t) on a 53-slot grid.

    V (M, T, 53) raw weekly signal. Returns ``(share (M,T,53), logshare53 (M,T,53),
    ok (M,T))``; slot 53 of 52-week years holds the pseudo week (log scale) for the
    climatology, and rows without a curve are NaN.
    """
    M, T, _ = V.shape
    share = np.full((M, T, 53), np.nan)
    L = np.full((M, T, 53), np.nan)
    ok = np.zeros((M, T), bool)
    for t in range(T):
        n = int(nweeks[t])
        raw = V[:, t, :n]
        tot = raw.sum(1)
        good = (tot >= MIN_SIGNAL_TOTAL) & np.isfinite(tot)
        if missing is not None:
            good &= ~missing[:, t]
        for m in np.where(good)[0]:
            sm = smooth3(raw[m])
            sh = sm / sm.sum()
            share[m, t, :n] = sh
            lg = np.log(sh + EPS)
            L[m, t, :n] = lg
            if n == 52:
                L[m, t, 52] = 0.5 * (lg[51] + lg[0])
                share[m, t, 52] = 0.5 * (sh[51] + sh[0])
        ok[:, t] = good
    return share, L, ok


def climatology(L, ok, train_years_mask):
    """(M, 53) mean log-share over training years with a curve; NaN if none."""
    use = ok & train_years_mask[None, :]
    num = np.where(use[..., None], L, 0.0).sum(1)
    den = use.sum(1)[:, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / np.maximum(den, 1), np.nan)


def mean_share_template(share, ok, train_years_mask):
    """(M, 53) mean smoothed share over training years with a curve (T1 template)."""
    use = ok & train_years_mask[None, :]
    num = np.where(use[..., None], share, 0.0).sum(1)
    den = use.sum(1)[:, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / np.maximum(den, 1), np.nan)


def anomalies(V, nweeks, train_years_mask, missing=None):
    """(M, T, 53) same-year log-share anomalies (0 where no curve or no climatology),
    plus diagnostics ``dict(ok, clim)``."""
    _, L, ok = year_curves(V, nweeks, missing)
    clim = climatology(L, ok, train_years_mask)
    A = L - clim[:, None, :]
    A = np.where(np.isfinite(A), A, 0.0)
    for t in range(len(nweeks)):
        A[:, t, int(nweeks[t]):] = 0.0
    return A, dict(ok=ok, clim=clim)


def lagged(A, nweeks, lags, sign=-1):
    """(M, T, 53, len(lags)) with ``x_L[w] = A[clip(w + sign * L)]`` within each year."""
    M, T, _ = A.shape
    X = np.zeros((M, T, 53, len(lags)))
    for t in range(T):
        n = int(nweeks[t])
        w = np.arange(n)
        for k, L in enumerate(lags):
            X[:, t, :n, k] = A[:, t, np.clip(w + sign * L, 0, n - 1)]
    return X


def standardise(X, train_mask, valid):
    """Centre and scale each feature column with training rows / valid weeks only.

    Returns ``(Xs, mu, sd)``; invalid slots are set to 0.
    """
    sel = train_mask[:, :, None] & valid[None, :, :]          # (M, T, 53)
    vals = X[sel]                                             # (n, F)
    mu = vals.mean(0)
    sd = vals.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    Xs = (X - mu) / sd
    Xs = np.where(valid[None, :, :, None], Xs, 0.0)
    return Xs, mu, sd


def state_block(pool, P, train_years_mask, train_mask, missing=None):
    """Feature block (a): standardised state-curve anomalies at lags ``STATE_LAGS``."""
    A, info = anomalies(pool, P.nweeks, train_years_mask, missing)
    X = lagged(A, P.nweeks, STATE_LAGS, sign=-1)
    Xs, mu, sd = standardise(X, train_mask, P.valid_w)
    info.update(mu=mu, sd=sd, A=A)
    return Xs, info


def trends_block(trends, P, train_years_mask, train_mask):
    """Feature block (b): standardised search anomalies, search value L weeks later (L = 0..3)."""
    A, info = anomalies(trends, P.nweeks, train_years_mask, None)
    X = lagged(A, P.nweeks, TRENDS_LAGS, sign=+1)
    Xs, mu, sd = standardise(X, train_mask, P.valid_w)
    info.update(mu=mu, sd=sd, A=A)
    return Xs, info


def best_shift(curve, template, max_shift=MAX_SHIFT):
    """Integer delta in [-max_shift, max_shift] maximising corr(curve, roll(template, delta));
    ties go to the smallest |delta| (then the negative one)."""
    order = sorted(range(-max_shift, max_shift + 1), key=lambda d: (abs(d), d))
    best, best_r = 0, -np.inf
    c = curve - curve.mean()
    for d in order:
        tt = np.roll(template, d)
        tt = tt - tt.mean()
        den = np.sqrt((c ** 2).sum() * (tt ** 2).sum())
        r = (c * tt).sum() / den if den > 0 else -np.inf
        if r > best_r + 1e-12:
            best, best_r = d, r
    return best, best_r


def shifts(pool, P, train_years_mask, missing=None, max_shift=MAX_SHIFT):
    """(M, T) integer shifts delta[m, t] of the state's same-year curve against its
    training-year mean curve (0 where there is no curve or no template)."""
    share, _, ok = year_curves(pool, P.nweeks, missing)
    tmpl = mean_share_template(share, ok, train_years_mask)
    M, T = ok.shape
    D = np.zeros((M, T), int)
    R = np.full((M, T), np.nan)
    for m in range(M):
        if not np.isfinite(tmpl[m]).all():
            continue
        for t in range(T):
            if not ok[m, t]:
                continue
            n = int(P.nweeks[t])
            D[m, t], R[m, t] = best_shift(share[m, t, :n], tmpl[m, :n], max_shift)
    return D, dict(ok=ok, template=tmpl, corr=R)


def shifted_offsets(h, D, nweeks):
    """(M, T, 53) year-specific offsets: h_m rolled by D[m, t] over the year's weeks.

    With D = 0 every year's offset equals h exactly (slot 53 is kept as is)."""
    M, T = D.shape
    H = np.broadcast_to(h[:, None, :], (M, T, h.shape[1])).copy()
    for t in range(T):
        n = int(nweeks[t])
        for m in np.where(D[:, t] != 0)[0]:
            H[m, t, :n] = np.roll(h[m, :n], int(D[m, t]))
    return H
