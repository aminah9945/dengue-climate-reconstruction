"""Result tables (Step 9): main table, paired differences, region table, gain by year,
absolute means with CIs and convergence summaries.

All tables are computed from the long metrics files ``metrics_detail_{tag}.csv``
written by :func:`dengue_recon.evaluation.scoring.score`.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .. import config as C
from ..evaluation.bootstrap import cluster_bootstrap, h3_contrast, paired

PRELIM = "preliminary (60 municipalities, LOYO 2010-2023, pilot hyperparameters)"


def prelim(tag):
    """Label printed on every pilot table/figure."""
    return PRELIM if "300" not in tag else "preliminary (300 municipalities, LOYO 2010-2023, frozen pilot settings)"


METRICS = ["jsd", "peak_err", "mae_share", "emd", "w50_err", "mae_counts", "cov80", "cov95", "crps"]
ORDER = ["B0", "B1p", "B1", "B4", "B4s", "B4s-eps1e-4", "M1", "M1-short", "M1-mult", "M1-shufTT",
         "M1-shufTest", "M2"]
PAIRS = [("M1", "B4s"), ("M1", "B1"), ("B4s", "B4"), ("B4", "B1"), ("M1", "M1-short"),
         ("M1-shufTT", "B4s"), ("M1-shufTest", "M1"), ("M1-mult", "M1"), ("B4s-eps1e-4", "B4s"),
         ("M2", "B4s"), ("B1", "B1p")]


def load_detail(tag):
    """Long metrics table for ``tag``; adds a pooled ``M1-shufTT`` row (mean over seeds)."""
    f = C.BRAZIL_DIR / f"metrics_detail_{tag}.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f)
    sh = df[df.model.str.startswith("M1-shufTT")]
    if len(sh):   # pooled shuffle row = mean over the 3 seeds per municipality-year
        agg = sh.groupby(["geocode", "year", "region", "N"], as_index=False)[
            [c for c in METRICS if c in sh] + ["jsd_floor"]].mean()
        agg["model"] = "M1-shufTT"
        df = pd.concat([df, agg], ignore_index=True)
    return df


def main_table(df, label):
    """Mean, median and N-weighted value per model and metric (all years and without 2020)."""
    rows = []
    for subset, d in [("all", df), ("without_2020", df[df.year != 2020])]:
        for m, g in d.groupby("model"):
            r = dict(model=m, subset=subset, n=len(g))
            for k in METRICS:
                if k in g and g[k].notna().any():
                    r[f"{k}_mean"] = g[k].mean()
                    r[f"{k}_median"] = g[k].median()
                    r[f"{k}_Nweighted"] = np.average(g[k], weights=g.N) if g[k].notna().all() else np.nan
            rows.append(r)
    t = pd.DataFrame(rows)
    t["order"] = t.model.map(lambda m: ORDER.index(m) if m in ORDER else 50)
    t = t.sort_values(["subset", "order", "model"]).drop(columns="order")
    t.insert(0, "note", label)
    return t


def paired_table(df, label):
    """Paired differences for the pre-specified contrasts (JSD, peak error, EMD)."""
    rows = []
    for subset, sub in [("all", None), ("without_2020", lambda j: j.index.get_level_values("year") != 2020)]:
        for a, b in PAIRS:
            for metric in ["jsd", "peak_err", "emd"]:
                r = paired(df, a, b, metric, sub)
                if r:
                    rows.append(dict(note=label, subset=subset, **r))
    return pd.DataFrame(rows)


def region_table(df, label):
    """M1 - B4s and M1 - B1 by region (municipality bootstrap), plus the H3 contrast."""
    rows = []
    for reg in C.REGIONS:
        for a, b in [("M1", "B4s"), ("M1", "B1")]:
            r = paired(df[df.region == reg], a, b, "jsd")
            if r:
                rows.append(dict(note=label, region=reg, **r))
    # H3 contrast: gain (b - a) in South+Southeast minus North, municipality bootstrap
    est, lo, hi = h3_contrast(df, "M1", "B4s")
    rows.append(dict(note=label, region="H3 contrast (South+Southeast) - North, gain B4s-M1",
                     model_a="M1", model_b="B4s", metric="jsd", mean_diff=est,
                     ci_muni_lo=lo, ci_muni_hi=hi))
    return pd.DataFrame(rows)


def gain_by_year(df):
    """Mean, median and count of the climate gain JSD(B4s) - JSD(M1) by held-out year."""
    j = df[df.model.isin(["M1", "B4s"])].pivot_table(index=["geocode", "year"], columns="model", values="jsd").reset_index()
    return j.assign(gain=j.B4s - j.M1).groupby("year").gain.agg(["mean", "median", "count"])


def abs_means_ci(dets):
    """Mean JSD and peak error per model with municipality-bootstrap 95 % CIs."""
    rows = []
    for tag, df in dets.items():
        for m in ["B0", "B1p", "B1", "B4", "B4s", "M1", "M2"]:
            g = df[df.model == m]
            if len(g) == 0:
                continue
            for k in ["jsd", "peak_err"]:
                mu, lo, hi = cluster_bootstrap(g[k].values, g.geocode.values, C.N_BOOT, C.BOOT_SEED)
                rows.append(dict(tag=tag, model=m, metric=k, mean=mu, ci_lo=lo, ci_hi=hi, n=len(g)))
    return pd.DataFrame(rows)


def convergence_table(tag):
    """Per model and fold: iterations, max |gradient|, success flag and fitted phi per region."""
    fp = json.loads((C.BRAZIL_DIR / f"fitted_params_{tag}.json").read_text())
    rows = [dict(model=m, year=r["year"], nit=r["nit"], grad_inf=r["grad_inf"], success=r["success"],
                 **{f"phi_{C.REGIONS[i]}": v for i, v in enumerate(r["phi"])}) for m, L in fp.items() for r in L]
    return pd.DataFrame(rows)
