"""Official experiment report (designs A/B/C, 2010-2023; 2024 pending).

Computes the pre-registered statistics from the per municipality-year files
``results/official/<tag>/metrics_detail_<tag>.csv`` and the fold checkpoints
(``folds/*.json``), and writes tables to ``results/official/tables/`` and figures
to ``results/official/figures/``. No model is refitted and nothing is tuned.

* Paired contrasts (``contrast``): mean difference a - b per municipality-year,
  relative change, 95 % percentile CIs from the municipality cluster bootstrap and
  the year-block bootstrap (``config.N_BOOT`` = 2,000 replicates, seed
  ``config.BOOT_SEED``), two-sided bootstrap p-values, folds in which ``a`` is
  better, headroom captured (JSD only). Metrics: JSD (primary), peak-week error,
  N-weighted JSD.
* Hypotheses H1-H3 on design A, NASA POWER (primary), with Holm correction.
* H4 on design B: JSD vs k per model and test year, average over test years for
  k = 2..9 (all three test years present), slope on log(k) with a municipality
  bootstrap CI, and the slope of the M1 - B4s gap. k >= 10 is reported separately
  because 2019 has no k >= 10.
* Robustness: per test year, region, epidemic vs quiet years, CHIRPS vs POWER,
  without 2020, design C vs the pilot, fitted climate coefficients.

2024 is never read: every loaded table is checked against
``config.max_loaded_year()``.

CLI: ``python scripts/run_pipeline.py --step official-report``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config as C

RUNS = {
    "designA_power_300": "Design A (primary), NASA POWER",
    "designA_chirps_300": "Design A, CHIRPS rain (robustness)",
    "designA_power_300_ex2020": "Design A without 2020 (robustness)",
    "designB_power_300": "Design B (learning curve), NASA POWER",
    "designC_power_300": "Design C (LOYO, pilot comparison), NASA POWER",
}
PRIMARY = "designA_power_300"
LC_RUN = "designB_power_300"
LC_K_MAIN = tuple(range(2, 10))          # k with all three test years (2019 has k <= 9 only)

# (label, a, b): difference a - b; negative = a better for JSD and peak error
CONTRASTS = [
    ("H1: M1 - B4s", "M1", "B4s"),
    ("H2: M1 - M1-short", "M1", "M1-short"),
    ("Local climate only: M1-noONI - B4s", "M1-noONI", "B4s"),
    ("ONI effect: M1 - M1-noONI", "M1", "M1-noONI"),
    ("Size terms: B4s - B4", "B4s", "B4"),
    ("M1 - B1 (seasonal profile)", "M1", "B1"),
    ("M2 - B4s", "M2", "B4s"),
    ("Shuffled climate (train+test): M1-shufTT - B4s", "M1-shufTT", "B4s"),
    ("Real vs shuffled climate: M1 - M1-shufTT", "M1", "M1-shufTT"),
    ("Wrong-year climate at test: M1-shufTest - M1", "M1-shufTest", "M1"),
]
METRICS = [("jsd", "jsd", False), ("peak_err", "peak_err", False), ("jsd_nweighted", "jsd", True)]

# fixed model colours (validated categorical palette; one colour per model in every figure)
MODEL_COL = {"B0": "#c3c2b7", "B1": "#52514e", "B4": "#eda100", "B4s": "#eb6834",
             "M1": "#2a78d6", "M1-noONI": "#1baf7a", "M1-short": "#e87ba4", "M2": "#6250d6",
             "M1-shufTT": "#8a8984", "M1-shufTest": "#a8a7a0"}
RUN_COL = {"designA_power_300": "#2a78d6", "designA_chirps_300": "#1baf7a",
           "designA_power_300_ex2020": "#6250d6", "designC_power_300": "#eb6834",
           "designB_power_300": "#eda100"}
FEATURE_VARS = ["temperature", "rain (log)", "humidity", "ONI"]


def feature_names():
    """Names of the 20 M1 features in model order (3 local variables + ONI) x 5 lag blocks."""
    return [f"{v} {lo}-{hi} wk" for v in FEATURE_VARS for (lo, hi) in C.LAG_BLOCKS]


def out_dirs(base: Path | None = None):
    """(tables dir, figures dir) under ``results/official`` (created)."""
    base = Path(base) if base is not None else C.OFFICIAL_DIR
    t, f = base / "tables", base / "figures"
    t.mkdir(parents=True, exist_ok=True)
    f.mkdir(parents=True, exist_ok=True)
    return t, f


# ------------------------------------------------------------------ loading
def check_no_locked_years(df: pd.DataFrame, col: str = "year"):
    """Raise if any row is after ``config.max_loaded_year()`` (2024 while locked)."""
    mx = C.max_loaded_year()
    if len(df) and int(df[col].max()) > mx:
        raise AssertionError(f"rows with {col} > {mx} found: 2024 must not be loaded")
    return df


def load_detail(tag: str, base: Path | None = None) -> pd.DataFrame | None:
    """Per municipality-year metrics of one official run (None if the file is absent)."""
    base = Path(base) if base is not None else C.OFFICIAL_DIR
    f = base / tag / f"metrics_detail_{tag}.csv"
    if not f.exists():
        return None
    return check_no_locked_years(pd.read_csv(f))


def load_run_info(tag: str, base: Path | None = None) -> dict:
    base = Path(base) if base is not None else C.OFFICIAL_DIR
    return json.loads((base / tag / "run_info.json").read_text())


# ------------------------------------------------------------------ bootstrap core
def boot_replicates(diff, cluster, n_boot: int = C.N_BOOT, seed: int = C.BOOT_SEED, weights=None):
    """Point estimate and bootstrap replicates of the (weighted) mean of ``diff``,
    resampling whole clusters. Same random draws as ``evaluation.bootstrap.cluster_bootstrap``.
    """
    diff = np.asarray(diff, float)
    w = np.ones_like(diff) if weights is None else np.asarray(weights, float)
    labels, inv = np.unique(cluster, return_inverse=True)
    sums = np.bincount(inv, weights=diff * w)
    wts = np.bincount(inv, weights=w)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(labels), size=(n_boot, len(labels)))
    boots = sums[idx].sum(1) / wts[idx].sum(1)
    return float((diff * w).sum() / w.sum()), boots


def ci(boots) -> tuple[float, float]:
    """95 % percentile interval."""
    return float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def boot_p(boots) -> float:
    """Two-sided bootstrap p-value for 'mean = 0': 2 x the smaller tail beyond 0,
    with the +1 correction so that it is never exactly 0."""
    b = np.asarray(boots)
    n = len(b)
    lo = ((b <= 0).sum() + 1) / (n + 1)
    hi = ((b >= 0).sum() + 1) / (n + 1)
    return float(min(1.0, 2 * min(lo, hi)))


def holm(pvals) -> np.ndarray:
    """Holm step-down adjusted p-values (same order as the input)."""
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    run = 0.0
    for i, j in enumerate(order):
        run = max(run, min(1.0, (m - i) * p[j]))
        adj[j] = run
    return adj


# ------------------------------------------------------------------ paired contrasts
def pair_frame(df: pd.DataFrame, a: str, b: str, metric: str = "jsd") -> pd.DataFrame:
    """Rows of models ``a`` and ``b`` joined on (fold_id, geocode); columns xa, xb, diff."""
    keys = ["fold_id", "geocode"]
    cols = keys + ["year", "region", "N", "jsd_floor", "k"]
    x = df.loc[df.model == a, cols + [metric]].rename(columns={metric: "xa"})
    y = df.loc[df.model == b, keys + [metric]].rename(columns={metric: "xb"})
    j = x.merge(y, on=keys, how="inner")
    j["diff"] = j.xa - j.xb
    return j


def contrast(df: pd.DataFrame, a: str, b: str, metric: str = "jsd", weighted: bool = False,
             n_boot: int = C.N_BOOT, seed: int = C.BOOT_SEED) -> dict | None:
    """Paired difference ``metric(a) - metric(b)`` with both cluster bootstraps.

    ``weighted=True`` weights municipality-years by their annual total N (N-weighted JSD).
    Year blocks are the test years. ``n_folds_a_better`` counts folds (test years in
    designs A and C; test year x k in design B) whose mean difference is < 0.
    """
    j = pair_frame(df, a, b, metric)
    j = j[np.isfinite(j["diff"])]
    if len(j) == 0:
        return None
    w = j.N.values.astype(float) if weighted else None
    est, bm = boot_replicates(j["diff"].values, j.geocode.values, n_boot, seed, w)
    _, by = boot_replicates(j["diff"].values, j.year.values, n_boot, seed, w)
    ww = np.ones(len(j)) if w is None else w
    mean_a = float(np.average(j.xa, weights=ww))
    mean_b = float(np.average(j.xb, weights=ww))
    floor = float(np.average(j.jsd_floor, weights=ww))
    per_fold = j.assign(_wd=j["diff"] * ww, _w=ww).groupby("fold_id")[["_wd", "_w"]].sum()
    per_fold = per_fold._wd / per_fold._w
    ml, mh = ci(bm)
    yl, yh = ci(by)
    return dict(model_a=a, model_b=b, metric=metric + ("_nweighted" if weighted else ""),
                n=len(j), n_municipalities=int(j.geocode.nunique()), n_test_years=int(j.year.nunique()),
                mean_a=mean_a, mean_b=mean_b, mean_diff=est, rel_change=est / mean_b,
                ci_muni_lo=ml, ci_muni_hi=mh, ci_year_lo=yl, ci_year_hi=yh,
                rel_ci_muni_lo=ml / mean_b, rel_ci_muni_hi=mh / mean_b,
                rel_ci_year_lo=yl / mean_b, rel_ci_year_hi=yh / mean_b,
                p_muni=boot_p(bm), p_year=boot_p(by),
                n_folds=int(len(per_fold)), n_folds_a_better=int((per_fold < 0).sum()),
                headroom_captured=(mean_b - mean_a) / (mean_b - floor) if metric == "jsd" else np.nan)


def contrast_table(df: pd.DataFrame, run: str, contrasts=CONTRASTS, metrics=METRICS,
                   n_boot: int = C.N_BOOT, seed: int = C.BOOT_SEED) -> pd.DataFrame:
    """All available contrasts x metrics for one run."""
    rows = []
    have = set(df.model.unique())
    for label, a, b in contrasts:
        if a not in have or b not in have:
            continue
        for _, metric, weighted in metrics:
            r = contrast(df, a, b, metric, weighted, n_boot, seed)
            if r:
                rows.append(dict(run=run, contrast=label, **r))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ H3 (two-group contrast)
def group_contrast(df: pd.DataFrame, model: str = "M1", ref: str = "B4s",
                   group_a=("South", "Southeast"), group_b=("North",),
                   n_boot: int = C.N_BOOT, seed: int = C.BOOT_SEED) -> dict:
    """H3: mean gain JSD(ref) - JSD(model) in ``group_a`` minus that in ``group_b``.

    Municipality bootstrap: municipalities resampled within each group (as
    ``evaluation.bootstrap.h3_contrast``). Year-block bootstrap: test years resampled,
    each carrying all its rows of both groups.
    """
    j = pair_frame(df, model, ref, "jsd")
    j["gain"] = -j["diff"]
    ga = j.region.isin(list(group_a)).values
    gb = j.region.isin(list(group_b)).values
    est = float(j.gain[ga].mean() - j.gain[gb].mean())
    rng = np.random.default_rng(seed)

    def grp_boot(mask):
        g = j[mask].groupby("geocode").gain.agg(["sum", "count"])
        idx = rng.integers(0, len(g), size=(n_boot, len(g)))
        return g["sum"].values[idx].sum(1) / g["count"].values[idx].sum(1)
    bm = grp_boot(ga) - grp_boot(gb)
    yrs = j.year.values
    labels, inv = np.unique(yrs, return_inverse=True)
    sa = np.bincount(inv, weights=j.gain.values * ga, minlength=len(labels))
    ca = np.bincount(inv, weights=ga.astype(float), minlength=len(labels))
    sb = np.bincount(inv, weights=j.gain.values * gb, minlength=len(labels))
    cb = np.bincount(inv, weights=gb.astype(float), minlength=len(labels))
    idx = rng.integers(0, len(labels), size=(n_boot, len(labels)))
    by = sa[idx].sum(1) / ca[idx].sum(1) - sb[idx].sum(1) / cb[idx].sum(1)
    ml, mh = ci(bm)
    yl, yh = ci(by)
    return dict(model_a=model, model_b=ref, metric="jsd gain (b - a)", estimate=est,
                gain_group_a=float(j.gain[ga].mean()), gain_group_b=float(j.gain[gb].mean()),
                n_a=int(ga.sum()), n_b=int(gb.sum()),
                ci_muni_lo=ml, ci_muni_hi=mh, ci_year_lo=yl, ci_year_hi=yh,
                p_muni=boot_p(bm), p_year=boot_p(by))


def hypothesis_table(df: pd.DataFrame, n_boot: int = C.N_BOOT, seed: int = C.BOOT_SEED,
                     threshold: float = -0.05, alpha: float = 0.05) -> pd.DataFrame:
    """Decisions for H1-H3 (plan section 5) on one run, with Holm correction.

    * H1: rel. JSD change M1 vs B4s <= ``threshold`` and both CIs exclude 0 (below 0);
      the 2024 sign condition is pending (reported as such).
    * H2: M1 - M1-short < 0 with both CIs below 0.
    * H3: (South + Southeast) - North gain of M1 vs B4s > 0 with both CIs above 0.

    The p-value of each hypothesis is the larger of the two bootstrap p-values
    (both intervals must agree). The Holm decision replaces 'both CIs exclude 0'
    by 'Holm-adjusted p < alpha' and keeps the direction and threshold conditions.
    """
    h1 = contrast(df, "M1", "B4s", "jsd", False, n_boot, seed)
    h2 = contrast(df, "M1", "M1-short", "jsd", False, n_boot, seed)
    h3 = group_contrast(df, "M1", "B4s", n_boot=n_boot, seed=seed)
    rows = []
    rows.append(dict(hypothesis="H1", statement="M1 beats B4s (JSD, >= 5 % lower)", estimate=h1["mean_diff"],
                     rel_change=h1["rel_change"], **{k: h1[k] for k in ["ci_muni_lo", "ci_muni_hi", "ci_year_lo", "ci_year_hi", "p_muni", "p_year"]},
                     direction_ok=h1["mean_diff"] < 0, threshold_ok=h1["rel_change"] <= threshold,
                     cis_ok=h1["ci_muni_hi"] < 0 and h1["ci_year_hi"] < 0))
    rows.append(dict(hypothesis="H2", statement="M1 beats M1-short (long lags add information)", estimate=h2["mean_diff"],
                     rel_change=h2["rel_change"], **{k: h2[k] for k in ["ci_muni_lo", "ci_muni_hi", "ci_year_lo", "ci_year_hi", "p_muni", "p_year"]},
                     direction_ok=h2["mean_diff"] < 0, threshold_ok=True,
                     cis_ok=h2["ci_muni_hi"] < 0 and h2["ci_year_hi"] < 0))
    rows.append(dict(hypothesis="H3", statement="Gain of M1 vs B4s larger in South+Southeast than North", estimate=h3["estimate"],
                     rel_change=np.nan, **{k: h3[k] for k in ["ci_muni_lo", "ci_muni_hi", "ci_year_lo", "ci_year_hi", "p_muni", "p_year"]},
                     direction_ok=h3["estimate"] > 0, threshold_ok=True,
                     cis_ok=h3["ci_muni_lo"] > 0 and h3["ci_year_lo"] > 0))
    t = pd.DataFrame(rows)
    t["p_both"] = t[["p_muni", "p_year"]].max(axis=1)
    t["p_holm"] = holm(t.p_both.values)
    t["supported_unadjusted"] = t.direction_ok & t.threshold_ok & t.cis_ok
    t["supported_holm"] = t.direction_ok & t.threshold_ok & (t.p_holm < alpha)

    def word(ok, h):
        if not ok:
            return "NOT SUPPORTED"
        return "SUPPORTED on 2010-2023 (2024 sign pending)" if h == "H1" else "SUPPORTED"
    t["decision"] = [word(o, h) for o, h in zip(t.supported_unadjusted, t.hypothesis)]
    t["decision_holm"] = [word(o, h) for o, h in zip(t.supported_holm, t.hypothesis)]
    return t


# ------------------------------------------------------------------ H4: learning curve
def _cube(rows: pd.DataFrame, value: str, geos, years, ks):
    """Per municipality x test year x k sums and counts of ``value``."""
    gi = pd.Index(geos).get_indexer(rows.geocode)
    yi = pd.Index(years).get_indexer(rows.year)
    ki = pd.Index(ks).get_indexer(rows.k)
    S = np.zeros((len(geos), len(years), len(ks)))
    Cn = np.zeros_like(S)
    np.add.at(S, (gi, yi, ki), rows[value].values)
    np.add.at(Cn, (gi, yi, ki), 1.0)
    return S, Cn


def _slope(y, logk):
    """OLS slope of each row of ``y`` (..., K) on ``logk`` (K,)."""
    x = logk - logk.mean()
    return ((y - y.mean(-1, keepdims=True)) * x).sum(-1) / (x ** 2).sum()


def lc_slope(rows: pd.DataFrame, value: str, ks, n_boot: int = C.N_BOOT, seed: int = C.BOOT_SEED) -> dict:
    """Slope of the k-curve on log(k) with a municipality bootstrap CI.

    The curve is the equal-weight average over test years of the per (test year, k)
    mean of ``value``. Every test year in ``rows`` must have every k in ``ks``
    (otherwise the average would change composition with k); this is asserted.
    """
    rows = rows[rows.k.isin(ks)]
    geos = np.sort(rows.geocode.unique())
    years = np.sort(rows.year.unique())
    present = rows.groupby(["year", "k"]).size().unstack()
    assert present.reindex(columns=list(ks)).notna().all().all(), "every test year needs every k"
    S, Cn = _cube(rows, value, geos, years, list(ks))
    logk = np.log(np.asarray(ks, float))
    curve = (S.sum(0) / Cn.sum(0)).mean(0)
    est = float(_slope(curve, logk))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(geos), size=(n_boot, len(geos)))
    W = np.zeros((n_boot, len(geos)))
    np.add.at(W, (np.repeat(np.arange(n_boot), len(geos)), idx.ravel()), 1.0)
    M = len(geos)
    bs = (W @ S.reshape(M, -1)) / (W @ Cn.reshape(M, -1))
    bcurve = bs.reshape(n_boot, len(years), len(ks)).mean(1)
    boots = _slope(bcurve, logk)
    lo, hi = ci(boots)
    return dict(slope_per_log_k=est, ci_lo=lo, ci_hi=hi, p=boot_p(boots),
                k_min=int(min(ks)), k_max=int(max(ks)), test_years=" ".join(map(str, years)),
                value_at_kmin=float(curve[0]), value_at_kmax=float(curve[-1]))


def learning_curve_tables(df: pd.DataFrame, models=("B1", "B4s", "M1", "M2"), k_main=LC_K_MAIN,
                          n_boot: int = C.N_BOOT, seed: int = C.BOOT_SEED):
    """H4 tables from design B.

    Returns ``(by_year, averaged, slopes, gap)``:
    * ``by_year``: mean JSD per model x test year x k (all k);
    * ``averaged``: equal-weight mean over test years per model x k, with the
      number of test years in the average and a flag ``main`` (k in ``k_main``);
    * ``slopes``: slope on log(k) per model (and for the M1 - B4s gap) over k_main,
      over k = 3..9 (sensitivity without the k = 2 regional profiles), and per test
      year over all of its k (no composition change within a test year);
    * ``gap``: M1 - B4s per k (pooled over the test years present) with both CIs.
    """
    d = df[df.model.isin(models)]
    by_year = d.groupby(["model", "year", "k"]).agg(jsd=("jsd", "mean"), peak_err=("peak_err", "mean"),
                                                    n=("jsd", "size")).reset_index()
    av = by_year.groupby(["model", "k"]).agg(jsd=("jsd", "mean"), peak_err=("peak_err", "mean"),
                                             n_test_years=("year", "nunique"),
                                             test_years=("year", lambda s: " ".join(map(str, sorted(s))))).reset_index()
    n_years_all = df.year.nunique()
    av["main"] = av.k.isin(k_main) & (av.n_test_years == n_years_all)
    av["note"] = np.where(av.main, "all test years", "k >= 10: fewer test years in the average (not comparable)")
    slopes = []
    pairs = [(m, None) for m in models] + [("M1", "B4s")]
    for a, b in pairs:
        if b is None:
            rows = df[df.model == a].assign(v=lambda x: x.jsd)
            name = a
        else:
            rows = pair_frame(df, a, b, "jsd").assign(v=lambda x: x["diff"])
            name = f"{a} - {b} (gap)"
        for scope, ks, yrs in [("main k=2..9, all test years", list(k_main), None),
                               ("sensitivity k=3..9 (no regional-profile k=2)", [k for k in k_main if k >= 3], None)] + \
                [(f"test year {y}, all its k", sorted(df[df.year == y].k.unique()), [y]) for y in sorted(df.year.unique())]:
            r = rows if yrs is None else rows[rows.year.isin(yrs)]
            s = lc_slope(r, "v", ks, n_boot, seed)
            slopes.append(dict(series=name, scope=scope, **s))
    gap = []
    for k in sorted(df.k.unique()):
        r = contrast(df[df.k == k], "M1", "B4s", "jsd", False, n_boot, seed)
        if r:
            gap.append(dict(k=int(k), n_test_years=r["n_test_years"], main=k in k_main and r["n_test_years"] == n_years_all,
                            **{x: r[x] for x in ["n", "mean_a", "mean_b", "mean_diff", "rel_change", "ci_muni_lo",
                                                 "ci_muni_hi", "ci_year_lo", "ci_year_hi"]}))
    return by_year, av, pd.DataFrame(slopes), pd.DataFrame(gap)


# ------------------------------------------------------------------ stratified tables
def per_fold_table(df: pd.DataFrame, run: str, models=None, n_boot: int = C.N_BOOT, seed: int = C.BOOT_SEED):
    """Per test year: mean JSD per model (wide) and M1 - B4s / M1-noONI - B4s with
    municipality-bootstrap CIs."""
    models = models or [m for m in MODEL_COL if m in df.model.unique()]
    wide = df[df.model.isin(models)].pivot_table(index=["year", "k"], columns="model", values="jsd").reset_index()
    wide = wide[["year", "k"] + [m for m in models if m in wide.columns]]
    rows = []
    for y, g in df.groupby("year"):
        r = dict(run=run, year=int(y), k=int(g.k.iloc[0]), n=int((g.model == "B4s").sum()))
        for a, b in [("M1", "B4s"), ("M1-noONI", "B4s")]:
            if a not in g.model.unique():
                continue
            j = pair_frame(g, a, b)
            est, bm = boot_replicates(j["diff"].values, j.geocode.values, n_boot, seed)
            lo, hi = ci(bm)
            tag = a.replace("-", "_")
            r.update({f"{tag}_minus_B4s": est, f"{tag}_rel": est / j.xb.mean(), f"{tag}_ci_lo": lo, f"{tag}_ci_hi": hi})
        rows.append(r)
    return wide.assign(run=run), pd.DataFrame(rows)


def region_table(df: pd.DataFrame, run: str, n_boot: int = C.N_BOOT, seed: int = C.BOOT_SEED):
    """M1 - B4s and M1-noONI - B4s by region (both bootstraps), plus the H3 contrast row."""
    rows = []
    for reg in C.REGIONS:
        for a, b in [("M1", "B4s"), ("M1-noONI", "B4s")]:
            if a not in df.model.unique():
                continue
            r = contrast(df[df.region == reg], a, b, "jsd", False, n_boot, seed)
            if r:
                rows.append(dict(run=run, region=reg, **r))
    for a in ["M1", "M1-noONI"]:
        if a in df.model.unique():
            h = group_contrast(df, a, "B4s", n_boot=n_boot, seed=seed)
            rows.append(dict(run=run, region="(South+Southeast) - North, gain", model_a=a, model_b="B4s",
                             metric="jsd gain (b - a)", mean_diff=h["estimate"],
                             **{k: h[k] for k in ["ci_muni_lo", "ci_muni_hi", "ci_year_lo", "ci_year_hi", "p_muni", "p_year"]}))
    return pd.DataFrame(rows)


def annual_totals(data_dir: Path | None = None) -> pd.DataFrame | None:
    """Annual totals per municipality-year (2010 .. max loaded year) from the processed
    dengue table, read with a year filter so 2024 never enters memory. None if absent."""
    f = Path(data_dir or C.DATA_DIR) / "dengue_weekly.parquet"
    if not f.exists():
        return None
    mx = C.max_loaded_year()
    d = pd.read_parquet(f, columns=["geocode", "epi_year", "year_total", "exclude_suggested"],
                        filters=[("epi_year", "<=", mx)])
    d = d[d.epi_year <= mx]
    t = d.groupby(["geocode", "epi_year"]).agg(N=("year_total", "first"),
                                               excluded=("exclude_suggested", "max")).reset_index()
    return check_no_locked_years(t.rename(columns={"epi_year": "year"}))


def epidemic_labels(df: pd.DataFrame, folds: list[dict], totals: pd.DataFrame) -> pd.DataFrame:
    """Label each scored (fold_id, geocode) as epidemic (top tercile of N / median training-year N
    of the municipality) or quiet (the other two terciles), plan section 6.

    The training median uses the fold's training years (excluded municipality-years
    dropped). The tercile cut is taken over all scored rows of the run.
    """
    tot = totals[~totals.excluded.astype(bool)].set_index(["geocode", "year"]).N
    keys = df[["fold_id", "geocode", "year", "N"]].drop_duplicates(["fold_id", "geocode"])
    med = {}
    for f in folds:
        tr = tot[tot.index.get_level_values("year").isin(f["train_years"])]
        med[f["fold_id"]] = tr.groupby(level="geocode").median()
    keys["train_median"] = [med[f].get(g, np.nan) for f, g in zip(keys.fold_id, keys.geocode)]
    keys["ratio"] = keys.N / keys.train_median
    cut = float(np.nanquantile(keys.ratio, 2 / 3))
    keys["stratum"] = np.where(keys.ratio >= cut, "epidemic", "quiet")
    keys["tercile_cut"] = cut
    return keys


def epidemic_table(df, run, labels, n_boot: int = C.N_BOOT, seed: int = C.BOOT_SEED):
    """M1 - B4s, M1-noONI - B4s, B4s - B4 by epidemic / quiet stratum."""
    d = df.merge(labels[["fold_id", "geocode", "stratum", "ratio", "tercile_cut"]], on=["fold_id", "geocode"])
    rows = []
    for s in ["epidemic", "quiet"]:
        for a, b in [("M1", "B4s"), ("M1-noONI", "B4s"), ("B4s", "B4")]:
            if a in d.model.unique() and b in d.model.unique():
                r = contrast(d[d.stratum == s], a, b, "jsd", False, n_boot, seed)
                rows.append(dict(run=run, stratum=s, tercile_cut=float(d.tercile_cut.iloc[0]), **r))
    return pd.DataFrame(rows)


def cross_run(df1: pd.DataFrame, df2: pd.DataFrame, model: str, label1: str, label2: str,
              n_boot: int = C.N_BOOT, seed: int = C.BOOT_SEED) -> dict:
    """Same model, two runs (e.g. CHIRPS vs POWER): JSD(run1) - JSD(run2) per
    municipality-year with both bootstraps."""
    x = df1[df1.model == model][["geocode", "year", "region", "jsd"]]
    y = df2[df2.model == model][["geocode", "year", "jsd"]]
    j = x.merge(y, on=["geocode", "year"], suffixes=("_1", "_2"))
    d = (j.jsd_1 - j.jsd_2).values
    est, bm = boot_replicates(d, j.geocode.values, n_boot, seed)
    _, by = boot_replicates(d, j.year.values, n_boot, seed)
    return dict(model=model, run_1=label1, run_2=label2, n=len(j), mean_1=j.jsd_1.mean(), mean_2=j.jsd_2.mean(),
                mean_diff=est, rel_change=est / j.jsd_2.mean(), ci_muni_lo=ci(bm)[0], ci_muni_hi=ci(bm)[1],
                ci_year_lo=ci(by)[0], ci_year_hi=ci(by)[1])


def model_means(df: pd.DataFrame, run: str, n_boot: int = C.N_BOOT, seed: int = C.BOOT_SEED) -> pd.DataFrame:
    """Mean JSD, N-weighted JSD and peak-week error per model with municipality-bootstrap CIs."""
    rows = []
    for m in [x for x in MODEL_COL if x in df.model.unique()]:
        g = df[df.model == m]
        mu, b = boot_replicates(g.jsd.values, g.geocode.values, n_boot, seed)
        rows.append(dict(run=run, model=m, n=len(g), jsd=mu, jsd_ci_lo=ci(b)[0], jsd_ci_hi=ci(b)[1],
                         jsd_nweighted=float(np.average(g.jsd, weights=g.N)), peak_err=g.peak_err.mean(),
                         jsd_floor=g.jsd_floor.mean()))
    return pd.DataFrame(rows)


def pilot_comparison(dfC: pd.DataFrame, pilot_dir: Path | None = None) -> pd.DataFrame | None:
    """Design C (official, NASA POWER, 300) vs the pilot's 300-municipality LOYO run.

    The pilot M1 is the official M1-noONI (15 local features)."""
    f = Path(pilot_dir or C.BRAZIL_DIR) / "table_main_weekly_300.csv"
    if not f.exists():
        return None
    p = pd.read_csv(f)
    p = p[p.subset == "all"] if "subset" in p.columns else p
    rows = []
    for off, pil in [("B0", "B0"), ("B1", "B1"), ("B4", "B4"), ("B4s", "B4s"), ("M1-noONI", "M1"), ("M1", None),
                     ("M1-shufTest", "M1-shufTest")]:
        g = dfC[dfC.model == off]
        pr = p[p.model == pil] if pil else p.iloc[0:0]
        rows.append(dict(official_model=off, pilot_model=pil or "-", official_n=len(g), official_jsd=g.jsd.mean() if len(g) else np.nan,
                         pilot_n=int(pr.n.iloc[0]) if len(pr) else np.nan,
                         pilot_jsd=float(pr.jsd_mean.iloc[0]) if len(pr) else np.nan))
    t = pd.DataFrame(rows)
    t["difference"] = t.official_jsd - t.pilot_jsd
    notes = {"M1-noONI": "official M1-noONI = pilot M1 (15 local features)",
             "M1": "new in the official run (20 features incl. ONI); no pilot counterpart",
             "M1-shufTest": "not comparable: official shuffles the 20-feature M1, pilot the 15-feature M1"}
    t["note"] = t.official_model.map(notes).fillna("same model")
    return t


# ------------------------------------------------------------------ coefficients and fold logs
def coefficient_table(tag: str, models=("M1", "M1-noONI"), base: Path | None = None) -> pd.DataFrame | None:
    """Fitted climate coefficients across folds from ``folds/*.json``: national theta
    (mean, SD, min, max, folds > 0) and region-specific theta + delta (mean over folds).
    Coefficients are on the logit scale per SD of the standardised lag-block anomaly."""
    base = Path(base) if base is not None else C.OFFICIAL_DIR
    files = sorted((base / tag / "folds").glob("*.json"))
    if not files:
        return None
    names = feature_names()
    rows = []
    for m in models:
        th, de = [], []
        for f in files:
            fit = json.loads(f.read_text()).get("fits", {}).get(m)
            if fit and "theta" in fit:
                th.append(fit["theta"]); de.append(fit["delta"])
        if not th:
            continue
        th, de = np.array(th), np.array(de)            # (folds, F), (folds, R, F)
        eff = th[:, None, :] + de
        for i in range(th.shape[1]):
            r = dict(run=tag, model=m, feature=names[i], variable=FEATURE_VARS[i // len(C.LAG_BLOCKS)],
                     lag_block=f"{C.LAG_BLOCKS[i % len(C.LAG_BLOCKS)][0]}-{C.LAG_BLOCKS[i % len(C.LAG_BLOCKS)][1]}",
                     n_folds=th.shape[0], theta_mean=th[:, i].mean(), theta_sd=th[:, i].std(ddof=1),
                     theta_min=th[:, i].min(), theta_max=th[:, i].max(), folds_positive=int((th[:, i] > 0).sum()))
            for ri, reg in enumerate(C.REGIONS):
                r[f"effective_{reg}"] = eff[:, ri, i].mean()
            rows.append(r)
    return pd.DataFrame(rows)


def fold_log_summary(tag: str, base: Path | None = None) -> pd.DataFrame | None:
    """Per model: folds, DM convergence, and M2 rounds (how often early stopping kept 1 round)."""
    base = Path(base) if base is not None else C.OFFICIAL_DIR
    f = base / tag / f"fold_log_{tag}.csv"
    if not f.exists():
        return None
    lg = pd.read_csv(f)
    out = lg.groupby("model").agg(n_folds=("fold_id", "size"), all_success=("success", lambda s: bool(s.dropna().all()) if s.notna().any() else np.nan),
                                  max_nit=("nit", "max"), m2_rounds_median=("m2_rounds", "median"),
                                  m2_rounds_min=("m2_rounds", "min"), m2_rounds_max=("m2_rounds", "max"),
                                  m2_folds_1_round=("m2_rounds", lambda s: int((s == 1).sum()))).reset_index()
    return out.assign(run=tag)


# ------------------------------------------------------------------ figures
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13, "legend.fontsize": 11,
                         "xtick.labelsize": 12, "ytick.labelsize": 12, "figure.dpi": 100, "savefig.dpi": 160,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.color": "#e4e3df", "grid.linewidth": 0.8, "axes.edgecolor": "#52514e",
                         "axes.titleweight": "bold", "axes.titlelocation": "left"})
    return plt


def fig_jsd_by_year(df, figdir, title, fname="figA_designA_jsd_by_test_year.png",
                    models=("B1", "B4", "B4s", "M1-noONI", "M1", "M2")):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(12, 6.5))
    t = df[df.model.isin(models)].groupby(["model", "year"]).jsd.mean().unstack(0)
    for m in models:
        if m not in t:
            continue
        lw = 2.8 if m in ("M1", "B4s") else 1.8
        ax.plot(t.index, t[m], marker="o", ms=7, lw=lw, color=MODEL_COL[m], label=m)
    fl = df[df.model == "B4s"].groupby("year").jsd_floor.mean()
    ax.fill_between(fl.index, 0, fl.values, color="#c3c2b7", alpha=0.35, label="noise floor")
    ks = df.groupby("year").k.first()
    ax.set_xticks(list(ks.index))
    ax.set_xticklabels([f"{y}\nk={k}" for y, k in ks.items()], fontsize=10)
    ax.set_xlim(ks.index.min() - 0.4, ks.index.max() + 0.4)
    ax.set_ylim(0, None)
    ax.set_ylabel("mean JSD per municipality-year (lower = better)")
    ax.set_xlabel("test year (k = training years)")
    ax.set_title(title)
    ax.legend(ncol=7, loc="upper center", bbox_to_anchor=(0.5, -0.16), frameon=False)
    fig.tight_layout(); fig.savefig(figdir / fname); plt.close(fig)


def fig_learning_curves(by_year, av, slopes, figdir, title, fname="figB_learning_curves.png",
                        models=("B1", "B4s", "M1", "M2")):
    plt = _plt()
    years = sorted(by_year.year.unique())
    fig, axs = plt.subplots(1, 1 + len(years), figsize=(18, 6.2), sharey=True,
                            gridspec_kw=dict(width_ratios=[1.3] + [1] * len(years)))
    ax = axs[0]
    for m in models:
        g = av[(av.model == m) & av.main]
        ax.plot(g.k, g.jsd, marker="o", ms=7, lw=2.6 if m in ("M1", "B4s") else 1.8, color=MODEL_COL[m], label=m)
    ax.set_xscale("log"); ax.set_xticks(list(LC_K_MAIN)); ax.set_xticklabels(list(LC_K_MAIN))
    ax.minorticks_off()
    ax.axvspan(1.9, 2.1, color="#c3c2b7", alpha=0.4)
    ax.set_title("Average of 2019, 2021, 2023\n(k = 2..9: all three test years)", fontsize=13)
    ax.set_xlabel("training years k (log scale)")
    ax.set_ylabel("mean JSD (lower = better)")
    ax.legend(frameon=False, loc="upper right")
    for a, y in zip(axs[1:], years):
        for m in models:
            g = by_year[(by_year.model == m) & (by_year.year == y)]
            main = g[g.k <= max(LC_K_MAIN)]
            ext = g[g.k >= max(LC_K_MAIN)]
            a.plot(main.k, main.jsd, marker="o", ms=6, lw=2.2 if m in ("M1", "B4s") else 1.5, color=MODEL_COL[m])
            if (g.k > max(LC_K_MAIN)).any():
                a.plot(ext.k, ext.jsd, marker="o", ms=6, lw=1.5, ls="--", color=MODEL_COL[m], mfc="white")
        if by_year[by_year.year == y].k.max() > max(LC_K_MAIN):
            a.axvspan(max(LC_K_MAIN) + 0.5, by_year[by_year.year == y].k.max() + 0.5, color="#c3c2b7", alpha=0.3)
            a.text(max(LC_K_MAIN) + 0.6, 0.995, "k >= 10\n(not in average)", transform=a.get_xaxis_transform(),
                   va="top", fontsize=10, color="#52514e")
        ks = sorted(by_year[by_year.year == y].k.unique())
        a.set_xscale("log"); a.set_xticks(ks); a.set_xticklabels(ks, fontsize=9); a.minorticks_off()
        a.set_title(f"test year {y}", fontsize=13)
        a.set_xlabel("k (log scale)")
    axs[0].annotate("k = 2: regional\nprofiles (G3)", xy=(2, axs[0].get_ylim()[1]), xytext=(2.1, axs[0].get_ylim()[1]),
                    va="top", fontsize=10, color="#52514e")
    fig.suptitle(title, x=0.01, ha="left", fontsize=15, fontweight="bold")
    fig.tight_layout(); fig.savefig(figdir / fname); plt.close(fig)


def _forest(ax, items, xlab):
    """items: list of (label, est, mlo, mhi, ylo, yhi, colour). Thick = municipality CI, thin = year-block CI."""
    for i, (lab, e, ml, mh, yl, yh, col) in enumerate(items):
        y = len(items) - 1 - i
        ax.plot([yl, yh], [y, y], color=col, lw=1.4, alpha=0.8)
        ax.plot([yl, yl], [y - 0.12, y + 0.12], color=col, lw=1.4); ax.plot([yh, yh], [y - 0.12, y + 0.12], color=col, lw=1.4)
        ax.plot([ml, mh], [y, y], color=col, lw=6, solid_capstyle="round")
        ax.plot(e, y, "o", ms=10, color="white", mec=col, mew=2.5)
        ax.annotate(f"{e:+.1f}%" if "%" in xlab else f"{e:+.1f}", (max(mh, yh), y), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=11)
    ax.set_yticks(range(len(items))); ax.set_yticklabels([it[0] for it in items][::-1])
    ax.axvline(0, color="#0b0b0b", lw=1)
    ax.set_xlabel(xlab)


def fig_gain_ci(ct, figdir, title, fname="figC_gain_with_CIs.png"):
    plt = _plt()
    order = ["designA_power_300", "designA_chirps_300", "designA_power_300_ex2020", "designC_power_300", "designB_power_300"]
    short = {"designA_power_300": "A, POWER (primary)", "designA_chirps_300": "A, CHIRPS",
             "designA_power_300_ex2020": "A, without 2020", "designC_power_300": "C, LOYO",
             "designB_power_300": "B, k=2..9 pooled"}
    fig, axs = plt.subplots(1, 2, figsize=(16, 6.2), sharex=True)
    for ax, (a, b) in zip(axs, [("M1", "B4s"), ("M1-noONI", "B4s")]):
        items = []
        for run in order:
            r = ct[(ct.run == run) & (ct.model_a == a) & (ct.model_b == b) & (ct.metric == "jsd")]
            if len(r):
                r = r.iloc[0]
                items.append((short[run], 100 * r.rel_change, 100 * r.rel_ci_muni_lo, 100 * r.rel_ci_muni_hi,
                              100 * r.rel_ci_year_lo, 100 * r.rel_ci_year_hi, RUN_COL[run]))
        _forest(ax, items, "relative JSD change vs B4s, % (negative = better)")
        ax.axvline(-5, color="#e34948", ls="--", lw=1.5)
        ax.text(-5, len(items) - 0.4, " H1 threshold -5 %", color="#e34948", fontsize=10, va="bottom")
        ax.set_title(f"{a} - {b}")
    fig.text(0.01, 0.01, "Thick bar: 95 % municipality-bootstrap CI.  Thin bar: 95 % year-block-bootstrap CI.  "
             "CIs scaled by the B4s mean of each run.", fontsize=10, color="#52514e")
    fig.suptitle(title, x=0.01, ha="left", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(figdir / fname); plt.close(fig)


def fig_region(rt_list, figdir, title, fname="figD_gain_by_region.png"):
    plt = _plt()
    fig, axs = plt.subplots(1, len(rt_list), figsize=(8 * len(rt_list), 6.2), sharex=True, squeeze=False)
    for ax, (lab, rt) in zip(axs[0], rt_list):
        items = []
        for reg in C.REGIONS:
            for a in ["M1", "M1-noONI"]:
                r = rt[(rt.region == reg) & (rt.model_a == a)]
                if len(r):
                    r = r.iloc[0]
                    items.append((f"{reg}, {a}", 1000 * r.mean_diff, 1000 * r.ci_muni_lo, 1000 * r.ci_muni_hi,
                                  1000 * r.ci_year_lo, 1000 * r.ci_year_hi, MODEL_COL[a]))
        _forest(ax, items, "JSD difference vs B4s x 1000 (negative = climate helps)")
        ax.set_title(lab)
    fig.text(0.01, 0.01, "Thick bar: 95 % municipality-bootstrap CI.  Thin bar: 95 % year-block-bootstrap CI.",
             fontsize=10, color="#52514e")
    fig.suptitle(title, x=0.01, ha="left", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(figdir / fname); plt.close(fig)


def fig_shuffle(ct, figdir, title, fname="figE_shuffle_test.png"):
    plt = _plt()
    want = [("M1", "B4s", "M1 - B4s (real climate)"), ("M1-shufTT", "B4s", "shuffled climate, refitted - B4s"),
            ("M1", "M1-shufTT", "M1 - M1 with shuffled climate"),
            ("M1-shufTest", "M1", "wrong-year climate at test - M1")]
    runs = [("designA_power_300", "Design A (primary)"), ("designC_power_300", "Design C (LOYO)")]
    fig, axs = plt.subplots(1, 2, figsize=(16, 5.6), sharex=True)
    for ax, (run, lab) in zip(axs, runs):
        items = []
        for a, b, txt in want:
            r = ct[(ct.run == run) & (ct.model_a == a) & (ct.model_b == b) & (ct.metric == "jsd")]
            if len(r):
                r = r.iloc[0]
                items.append((txt, 1000 * r.mean_diff, 1000 * r.ci_muni_lo, 1000 * r.ci_muni_hi,
                              1000 * r.ci_year_lo, 1000 * r.ci_year_hi, MODEL_COL.get(a, "#52514e")))
        _forest(ax, items, "JSD difference x 1000 (negative = first model better)")
        ax.set_title(lab)
    fig.text(0.01, 0.01, "Thick bar: 95 % municipality-bootstrap CI.  Thin bar: 95 % year-block-bootstrap CI.",
             fontsize=10, color="#52514e")
    fig.suptitle(title, x=0.01, ha="left", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(figdir / fname); plt.close(fig)


def fig_weather(ct, rts, figdir, title, fname="figF_weather_source.png"):
    """POWER vs CHIRPS: M1 - B4s and M1-noONI - B4s overall and by region."""
    plt = _plt()
    fig, axs = plt.subplots(1, 2, figsize=(16, 6.4), sharey=True)
    for ax, a in zip(axs, ["M1-noONI", "M1"]):
        cats = ["all"] + C.REGIONS
        w = 0.38
        for j, (run, lab) in enumerate([("designA_power_300", "NASA POWER"), ("designA_chirps_300", "CHIRPS rain")]):
            vals, lo, hi = [], [], []
            for c in cats:
                if c == "all":
                    r = ct[(ct.run == run) & (ct.model_a == a) & (ct.model_b == "B4s") & (ct.metric == "jsd")].iloc[0]
                else:
                    rt = rts[run]
                    r = rt[(rt.region == c) & (rt.model_a == a)].iloc[0]
                vals.append(1000 * r.mean_diff); lo.append(1000 * r.ci_muni_lo); hi.append(1000 * r.ci_muni_hi)
            x = np.arange(len(cats)) + (j - 0.5) * w
            vals, lo, hi = map(np.array, (vals, lo, hi))
            ax.bar(x, vals, width=w - 0.04, color=RUN_COL[run], label=lab)
            ax.errorbar(x, vals, yerr=[vals - lo, hi - vals], fmt="none", ecolor="#0b0b0b", capsize=4, lw=1.3)
        ax.axhline(0, color="#0b0b0b", lw=1)
        ax.set_xticks(range(len(cats))); ax.set_xticklabels(cats, rotation=15)
        ax.set_title(f"{a} - B4s")
        ax.set_ylabel("JSD difference x 1000 (negative = climate helps)")
        ax.legend(frameon=False)
    fig.text(0.01, 0.01, "Error bars: 95 % municipality-bootstrap CI.", fontsize=10, color="#52514e")
    fig.suptitle(title, x=0.01, ha="left", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(figdir / fname); plt.close(fig)


def fig_coefficients(coef, figdir, title, fname="figG_climate_coefficients.png"):
    plt = _plt()
    fig, axs = plt.subplots(1, 4, figsize=(18, 5.6), sharey=True)
    blocks = [f"{lo}-{hi}" for lo, hi in C.LAG_BLOCKS]
    for ax, v in zip(axs, FEATURE_VARS):
        for off, m in [(-0.12, "M1"), (0.12, "M1-noONI")]:
            g = coef[(coef.model == m) & (coef.variable == v)]
            if not len(g):
                continue
            x = np.arange(len(blocks)) + off
            ax.errorbar(x, g.theta_mean, yerr=g.theta_sd, fmt="o", ms=8, capsize=4, lw=1.6, color=MODEL_COL[m], label=m)
        ax.axhline(0, color="#0b0b0b", lw=1)
        ax.set_xticks(range(len(blocks))); ax.set_xticklabels(blocks)
        ax.set_xlabel("lag block (weeks before)")
        ax.set_title(v)
    axs[0].set_ylabel("national coefficient (logit per SD)\nmean +/- SD across folds")
    axs[0].legend(frameon=False)
    fig.suptitle(title, x=0.01, ha="left", fontsize=15, fontweight="bold")
    fig.tight_layout(); fig.savefig(figdir / fname); plt.close(fig)


# ------------------------------------------------------------------ orchestrator
def main(base: Path | None = None, data_dir: Path | None = None, n_boot: int = C.N_BOOT, seed: int = C.BOOT_SEED,
         figures: bool = True):
    """Compute every official table (CSV) and figure (PNG) from the saved run files."""
    base = Path(base) if base is not None else C.OFFICIAL_DIR
    tdir, fdir = out_dirs(base)
    dets = {t: load_detail(t, base) for t in RUNS}
    dets = {t: d for t, d in dets.items() if d is not None}
    if PRIMARY not in dets:
        raise FileNotFoundError(f"metrics_detail_{PRIMARY}.csv not found under {base}")
    print("[official-report] runs:", ", ".join(dets))

    # model means and contrasts
    mm = pd.concat([model_means(d if not t.startswith("designB") else d[d.k.isin(LC_K_MAIN)], t, n_boot, seed)
                    for t, d in dets.items()])
    mm.to_csv(tdir / "model_means.csv", index=False)
    cts = []
    for t, d in dets.items():
        dd = d[d.k.isin(LC_K_MAIN)] if t.startswith("designB") else d
        cts.append(contrast_table(dd, t, n_boot=n_boot, seed=seed))
    ct = pd.concat(cts, ignore_index=True)
    ct.to_csv(tdir / "paired_contrasts.csv", index=False)

    # hypotheses on the primary run (+ the same rule on the other design-A/C runs, descriptive)
    hyp = []
    for t in ["designA_power_300", "designA_chirps_300", "designA_power_300_ex2020", "designC_power_300"]:
        if t in dets:
            hyp.append(hypothesis_table(dets[t], n_boot, seed).assign(run=t, primary=t == PRIMARY))
    hyp = pd.concat(hyp, ignore_index=True)
    hyp.to_csv(tdir / "hypotheses.csv", index=False)

    # per test year, region, epidemic/quiet
    pf_w, pf_c, rts = [], [], {}
    for t in ["designA_power_300", "designA_chirps_300", "designA_power_300_ex2020", "designC_power_300"]:
        if t in dets:
            w, c = per_fold_table(dets[t], t, n_boot=n_boot, seed=seed)
            pf_w.append(w); pf_c.append(c)
            rts[t] = region_table(dets[t], t, n_boot, seed)
    pd.concat(pf_w).to_csv(tdir / "jsd_by_test_year.csv", index=False)
    pd.concat(pf_c).to_csv(tdir / "gain_by_test_year.csv", index=False)
    pd.concat(rts.values()).to_csv(tdir / "gain_by_region.csv", index=False)

    totals = annual_totals(data_dir)
    if totals is None:
        print("[official-report] dengue_weekly.parquet not found: epidemic/quiet table skipped")
    else:
        eps = []
        for t in ["designA_power_300", "designA_chirps_300", "designA_power_300_ex2020"]:
            if t in dets:
                lab = epidemic_labels(dets[t], load_run_info(t, base)["folds"], totals)
                eps.append(epidemic_table(dets[t], t, lab, n_boot, seed))
        pd.concat(eps).to_csv(tdir / "gain_epidemic_vs_quiet.csv", index=False)

    # CHIRPS vs POWER, ex2020 vs all (same model, cross-run)
    xr = []
    for t2, lab in [("designA_chirps_300", "CHIRPS"), ("designA_power_300_ex2020", "POWER without 2020")]:
        if t2 in dets:
            for m in ["B4s", "M1-noONI", "M1", "M2"]:
                xr.append(cross_run(dets[t2], dets[PRIMARY], m, lab, "POWER (primary)", n_boot, seed))
    pd.DataFrame(xr).to_csv(tdir / "cross_run_same_model.csv", index=False)

    if "designC_power_300" in dets:
        pc = pilot_comparison(dets["designC_power_300"])
        if pc is not None:
            pc.to_csv(tdir / "designC_vs_pilot.csv", index=False)

    # H4
    lc = None
    if LC_RUN in dets:
        lc = learning_curve_tables(dets[LC_RUN], n_boot=n_boot, seed=seed)
        for name, tab in zip(["lc_jsd_by_test_year_k", "lc_jsd_averaged", "lc_slopes", "lc_gap_M1_B4s_by_k"], lc):
            tab.to_csv(tdir / f"{name}.csv", index=False)

    coefs = [c for c in (coefficient_table(t, base=base) for t in ["designA_power_300", "designC_power_300"]) if c is not None]
    coef = pd.concat(coefs) if coefs else None
    if coef is not None:
        coef.to_csv(tdir / "climate_coefficients.csv", index=False)
    else:
        print("[official-report] no fold checkpoints: coefficient table skipped")
    logs = [x for x in (fold_log_summary(t, base) for t in dets) if x is not None]
    if logs:
        pd.concat(logs).to_csv(tdir / "fold_log_summary.csv", index=False)

    if figures:
        _figures(dets, ct, hyp, rts, lc, coef, fdir)
    print(f"[official-report] tables -> {tdir}\n[official-report] figures -> {fdir}")
    return dict(contrasts=ct, hypotheses=hyp)


def _figures(dets, ct, hyp, rts, lc, coef, fdir):
    h1 = hyp[(hyp.run == PRIMARY) & (hyp.hypothesis == "H1")].iloc[0]
    d = dets[PRIMARY]
    t = d.groupby(["year", "model"]).jsd.mean().unstack()
    nb = int((t["M1"] < t["B4s"]).sum())
    fig_jsd_by_year(d, fdir, f"Design A (primary): M1 beats B4s in {nb} of {len(t)} test years; "
                             f"overall {100 * h1.rel_change:+.1f}% (H1 needs <= -5%)")
    if lc is not None:
        by_year, av, slopes, gap = lc
        s = slopes[slopes.scope.str.startswith("main")].set_index("series")
        s3 = slopes[slopes.scope.str.startswith("sensitivity")].set_index("series")
        fig_learning_curves(by_year, av, slopes, fdir,
                            f"Design B: from k=3 to k=9 JSD falls for every model; M1 falls fastest "
                            f"(slope per log k: M1 {s3.loc['M1', 'slope_per_log_k']:+.3f}, B4s {s3.loc['B4s', 'slope_per_log_k']:+.3f}) "
                            f"and is below B4s for k=5..9")
    r = ct[(ct.run == PRIMARY) & (ct.model_a == "M1-noONI") & (ct.model_b == "B4s") & (ct.metric == "jsd")].iloc[0]
    fig_gain_ci(ct, fdir, f"Climate gain is far from the -5 % threshold: M1 {100 * h1.rel_change:+.1f}%, "
                          f"M1-noONI {100 * r.rel_change:+.1f}% (design A, primary)")
    reg_list = [(RUNS[t], rts[t]) for t in ["designA_power_300", "designA_chirps_300"] if t in rts]
    h3 = hyp[(hyp.run == PRIMARY) & (hyp.hypothesis == "H3")].iloc[0]
    fig_region(reg_list, fdir, f"Climate hurts in the North and Northeast; local climate (M1-noONI) helps in the South and Southeast "
                               f"(H3: {1000 * h3.estimate:+.1f} x 1e-3; year-block CI includes 0)")
    fig_shuffle(ct, fdir, "Wrong-year climate at test makes M1 worse, but in design A real climate is not clearly "
                          "better than shuffled climate")
    if "designA_chirps_300" in rts:
        fig_weather(ct, rts, fdir, "Weather source barely matters: CHIRPS rain gives about the same results as NASA POWER")
    if coef is not None:
        fig_coefficients(coef[coef.run == PRIMARY], fdir,
                         "Design A (12 folds): rain 4-11 weeks before is positive in every fold; ONI coefficients are large and change sign")
