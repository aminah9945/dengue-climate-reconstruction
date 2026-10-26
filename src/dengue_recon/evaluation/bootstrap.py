"""Cluster bootstraps and paired comparisons (Step 6).

Two cluster bootstraps are used for every paired difference between models:
(i) resample municipalities, (ii) resample LOYO years (year blocks). Climate,
ENSO and reporting shocks are shared by all municipalities in a year, so the
year-block interval is usually wider.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config as C


def cluster_bootstrap(diff: np.ndarray, cluster: np.ndarray, n_boot: int = 2000,
                      seed: int = 1, weights: np.ndarray | None = None):
    """Percentile CI of the (weighted) mean of ``diff``, resampling whole clusters.

    Returns ``(mean, lo, hi)`` (95 % percentile interval).
    """
    diff = np.asarray(diff, float)
    w = np.ones_like(diff) if weights is None else np.asarray(weights, float)
    labels, inv = np.unique(cluster, return_inverse=True)
    sums = np.bincount(inv, weights=diff * w)
    wts = np.bincount(inv, weights=w)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(labels), size=(n_boot, len(labels)))
    boots = sums[idx].sum(1) / wts[idx].sum(1)
    mean = float((diff * w).sum() / w.sum())
    return mean, float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def paired(df: pd.DataFrame, a: str, b: str, metric: str = "jsd", sub=None) -> dict | None:
    """Paired difference ``metric(a) - metric(b)`` per municipality-year.

    ``df`` is a long metrics table (one row per model x municipality-year) with
    columns geocode, year, region, N, jsd_floor, model and the metric. Returns the
    mean difference, relative change, municipality and year-block bootstrap CIs,
    the number of LOYO years in which ``a`` is better, the N-weighted difference
    and (for JSD) the headroom captured relative to the noise floor.
    """
    x = df[df.model == a].set_index(["geocode", "year"])
    y = df[df.model == b].set_index(["geocode", "year"])
    j = x[[metric, "region", "N", "jsd_floor"]].join(y[[metric]], rsuffix="_b", how="inner")
    if sub is not None:
        j = j[sub(j)]
    if len(j) == 0:
        return None
    j = j.reset_index()
    diff = (j[metric] - j[f"{metric}_b"]).values
    mm, ml, mh = cluster_bootstrap(diff, j.geocode.values, C.N_BOOT, C.BOOT_SEED)
    _, yl, yh = cluster_bootstrap(diff, j.year.values, C.N_BOOT, C.BOOT_SEED)
    by_year = pd.Series(diff).groupby(j.year.values).mean()
    return dict(model_a=a, model_b=b, metric=metric, n=len(j), mean_a=j[metric].mean(),
                mean_b=j[f"{metric}_b"].mean(), mean_diff=mm, rel_change=mm / j[f"{metric}_b"].mean(),
                ci_muni_lo=ml, ci_muni_hi=mh, ci_year_lo=yl, ci_year_hi=yh,
                n_years=len(by_year), n_years_a_better=int((by_year < 0).sum()),
                mean_diff_Nweighted=float(np.average(diff, weights=j.N)),
                headroom_captured=(j[f"{metric}_b"].mean() - j[metric].mean()) /
                (j[f"{metric}_b"].mean() - j.jsd_floor.mean()) if metric == "jsd" else np.nan)


def h3_contrast(df: pd.DataFrame, model: str = "M1", ref: str = "B4s",
                group_a=("South", "Southeast"), group_b=("North",)) -> tuple[float, float, float]:
    """H3 contrast: mean gain ``JSD(ref) - JSD(model)`` in ``group_a`` regions minus
    that in ``group_b``, with a municipality bootstrap (resampled within each group).

    Returns ``(estimate, lo, hi)``.
    """
    x = df[df.model == model].set_index(["geocode", "year"])[["jsd", "region"]]
    y = df[df.model == ref].set_index(["geocode", "year"])[["jsd"]]
    j = x.join(y, rsuffix="_b").reset_index()
    j["gain"] = j.jsd_b - j.jsd
    rng = np.random.default_rng(C.BOOT_SEED)
    g_s = j[j.region.isin(list(group_a))]
    g_n = j[j.region.isin(list(group_b))]
    est = g_s.gain.mean() - g_n.gain.mean()
    ms, mn = g_s.geocode.unique(), g_n.geocode.unique()
    gs_by = g_s.groupby("geocode").gain.agg(["sum", "count"])
    gn_by = g_n.groupby("geocode").gain.agg(["sum", "count"])
    boots = []
    for _ in range(C.N_BOOT):
        a_ = gs_by.loc[rng.choice(ms, len(ms))]
        b_ = gn_by.loc[rng.choice(mn, len(mn))]
        boots.append(a_["sum"].sum() / a_["count"].sum() - b_["sum"].sum() / b_["count"].sum())
    return est, float(np.quantile(boots, .025)), float(np.quantile(boots, .975))
