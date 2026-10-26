"""Figures (Step 9). Every function reads saved results and writes a PNG to
``config.FIG_DIR``; functions that need the large prediction files
(``preds_*.npz``) are skipped by the report when those files are absent.
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .. import config as C  # noqa: E402
from ..evaluation.bootstrap import cluster_bootstrap, paired  # noqa: E402
from .tables import prelim  # noqa: E402

plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
                     "legend.fontsize": 9, "figure.dpi": 110, "savefig.dpi": 200,
                     "axes.spines.top": False, "axes.spines.right": False})


def fig_repro():
    """Bar chart: our MAP/NUTS reproduction vs the published DM (MAE, 80 % coverage, CRPS)."""
    f = C.REPRO_DIR / "table_repro.csv"
    if not f.exists():
        return
    t = pd.read_csv(f)
    t = t[t.fold.astype(str) == "all"]
    labels, mae, cov, crps = [], [], [], []
    for _, r in t.iterrows():
        if r.model == "DM" or r.model.startswith("DM (published"):
            labels.append("Published\n(3 reps)" if r.model.startswith("DM (pub") else f"Ours {r.fit}\n(rep01)")
            mae.append(r.MAE_median); cov.append(r.COV80); crps.append(r.CRPS)
    fig, ax = plt.subplots(1, 3, figsize=(12, 4.2))
    cols = ["#636363" if l.startswith("Pub") else "#3182bd" for l in labels]
    for a, vals, name in zip(ax, [mae, cov, crps], ["MAE (incidence /100k, median-based)", "80% coverage", "CRPS"]):
        a.bar(range(len(labels)), vals, color=cols)
        a.set_xticks(range(len(labels))); a.set_xticklabels(labels, fontsize=10)
        a.set_title(name)
        for i, v in enumerate(vals):
            if np.isfinite(v):
                a.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    ax[1].axhline(0.8, ls=":", c="k", lw=1)
    fig.suptitle("Reproduction of Lim et al. (2026) DM, interpolation mask (ours: rep01, 3 folds)")
    fig.tight_layout(); fig.savefig(C.FIG_DIR / "fig_repro_lim.png"); plt.close(fig)


def fig_jsd_by_model(df, tag):
    """JSD distribution per model (box plots) with the mean noise floor."""
    models = [m for m in ["B0", "B1p", "B1", "B4", "B4s", "M1", "M1-short", "M1-mult", "M1-shufTT", "M2"] if m in df.model.unique()]
    fig, ax = plt.subplots(figsize=(11, 4.8))
    data = [df[df.model == m].jsd.values for m in models]
    bp = ax.boxplot(data, tick_labels=models, patch_artist=True, showfliers=False, showmeans=True,
                    meanprops=dict(marker="D", markerfacecolor="k", markersize=5))
    for p_, m in zip(bp["boxes"], models):
        p_.set_facecolor(C.MODEL_COL.get(m, "#cccccc"))
    fl = df[df.model == "B1"].jsd_floor
    ax.axhspan(0, fl.mean(), color="k", alpha=0.08, label=f"mean noise floor {fl.mean():.3f}")
    ax.set_ylabel("JSD (base 2) per municipality-year")
    ax.set_title(f"Weekly share reconstruction error by model ({tag}); diamond = mean\n{prelim(tag)}")
    ax.legend(loc="upper right")
    fig.tight_layout(); fig.savefig(C.FIG_DIR / f"fig_jsd_by_model{'' if tag == 'weekly_60' else '_' + tag}.png"); plt.close(fig)


def fig_main_ci(df, tag):
    """Baseline vs climate: mean JSD and peak error with municipality-bootstrap 95% CIs."""
    models = [m for m in ["B0", "B1p", "B1", "B4", "B4s", "M1", "M1-short", "M2"] if m in df.model.unique()]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for a, metric, lab in zip(ax, ["jsd", "peak_err"], ["mean JSD", "mean peak-week error (weeks)"]):
        for i, m in enumerate(models):
            g = df[df.model == m]
            mu, lo, hi = cluster_bootstrap(g[metric].values, g.geocode.values, C.N_BOOT, C.BOOT_SEED)
            a.errorbar(i, mu, yerr=[[mu - lo], [hi - mu]], fmt="o", color=C.MODEL_COL.get(m, "k"), capsize=4, ms=7)
            a.text(i + 0.1, mu, f"{mu:.3f}" if metric == "jsd" else f"{mu:.2f}", fontsize=8, va="center")
        a.set_xticks(range(len(models))); a.set_xticklabels(models)
        a.set_ylabel(lab)
    ax[0].set_title("JSD (95% municipality-bootstrap CI)")
    ax[1].set_title("Peak-week error (95% municipality-bootstrap CI)")
    fig.suptitle(f"Baselines vs climate model, {tag}; {prelim(tag)}", fontsize=10)
    fig.tight_layout(); fig.savefig(C.FIG_DIR / f"fig_baseline_vs_climate_{tag}.png"); plt.close(fig)


def fig_gain_region_year(df, tag):
    """Heat map region x year of the climate gain, plus gain by region and by year."""
    x = df[df.model == "M1"].set_index(["geocode", "year"])
    y = df[df.model == "B4s"].set_index(["geocode", "year"])
    j = x[["jsd", "region"]].join(y[["jsd"]], rsuffix="_b").reset_index()
    j["gain"] = j.jsd_b - j.jsd
    piv = j.pivot_table(index="region", columns="year", values="gain", aggfunc="mean").reindex(C.REGIONS)
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1])
    a0 = fig.add_subplot(gs[0, :])
    v = np.nanmax(np.abs(piv.values))
    im = a0.imshow(piv.values, cmap="RdBu", vmin=-v, vmax=v, aspect="auto")
    a0.set_xticks(range(piv.shape[1])); a0.set_xticklabels(piv.columns)
    a0.set_yticks(range(piv.shape[0])); a0.set_yticklabels(piv.index)
    for (i, k), val in np.ndenumerate(piv.values):
        if np.isfinite(val):
            a0.text(k, i, f"{val * 1000:.1f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=a0, label="mean JSD gain B4s - M1 (blue = climate helps)")
    a0.set_title("Climate gain by region x held-out year (cell text = gain x 1000)")
    a1 = fig.add_subplot(gs[1, 0])
    for i, reg in enumerate(C.REGIONS):
        g = j[j.region == reg]
        mu, lo, hi = cluster_bootstrap(g.gain.values, g.geocode.values, C.N_BOOT, C.BOOT_SEED)
        a1.bar(i, mu, color=C.REG_COL[reg]); a1.errorbar(i, mu, yerr=[[mu - lo], [hi - mu]], color="k", capsize=4)
    a1.axhline(0, c="k", lw=0.8); a1.set_xticks(range(5)); a1.set_xticklabels(C.REGIONS, rotation=15)
    a1.set_ylabel("mean gain (JSD B4s - M1)"); a1.set_title("Gain by region (95% municipality bootstrap)")
    a2 = fig.add_subplot(gs[1, 1])
    yrs = sorted(j.year.unique())
    for i, yy in enumerate(yrs):
        g = j[j.year == yy]
        mu, lo, hi = cluster_bootstrap(g.gain.values, g.geocode.values, C.N_BOOT, C.BOOT_SEED)
        a2.bar(i, mu, color="#31a354" if mu > 0 else "#de2d26"); a2.errorbar(i, mu, yerr=[[mu - lo], [hi - mu]], color="k", capsize=3)
    a2.axhline(0, c="k", lw=0.8); a2.set_xticks(range(len(yrs))); a2.set_xticklabels(yrs, rotation=60)
    a2.set_title("Gain by held-out year (95% municipality bootstrap)")
    fig.suptitle(f"Where and when does climate help? ({tag}; {prelim(tag)})", fontsize=11)
    fig.tight_layout(); fig.savefig(C.FIG_DIR / f"fig_gain_by_region_year{'' if tag == 'weekly_60' else '_' + tag}.png"); plt.close(fig)


def fig_shuffle(df, tag):
    """Forest plot of the climate-shuffle checks against B4s."""
    items = [("M1", "B4s", "M1 - B4s (real climate)"), ("M1-short", "B4s", "M1-short - B4s"),
             ("M1-shufTT", "B4s", "M1-shufTT - B4s (3 seeds pooled)")] + \
            [(f"M1-shufTT-s{s}", "B4s", f"  seed {s}") for s in range(3)] + \
            [("M1-shufTest", "B4s", "M1 given wrong-year climate at test - B4s")]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    k = 0
    for a, b, lab in items:
        r = paired(df, a, b)
        if r is None:
            continue
        ax.errorbar(r["mean_diff"], k, xerr=[[r["mean_diff"] - r["ci_muni_lo"]], [r["ci_muni_hi"] - r["mean_diff"]]],
                    fmt="o", color=("#969696" if "shuf" in a else C.MODEL_COL.get(a, "#444444")), capsize=4)
        ax.errorbar(r["mean_diff"], k - 0.25, xerr=[[r["mean_diff"] - r["ci_year_lo"]], [r["ci_year_hi"] - r["mean_diff"]]],
                    fmt="none", color="k", alpha=0.5, capsize=2)
        ax.text(ax.get_xlim()[1] if False else r["ci_muni_hi"], k, f"  {r['mean_diff']:+.4f}", va="center", fontsize=8)
        ax.annotate(lab, (0, k), xytext=(-8, 0), textcoords="offset points", ha="right", va="center", fontsize=9,
                    xycoords=("axes fraction", "data"))
        k -= 1
    ax.axvline(0, c="k", lw=0.8); ax.set_yticks([])
    ax.set_xlabel("mean paired JSD difference (negative = better than B4s)\ncoloured: municipality bootstrap 95% CI; grey: year-block bootstrap")
    ax.set_title(f"Climate shuffle sanity checks ({tag})")
    fig.tight_layout(); fig.subplots_adjust(left=0.38); fig.savefig(C.FIG_DIR / f"fig_shuffle{'' if tag == 'weekly_60' else '_' + tag}.png"); plt.close(fig)


def fig_lag_coefficients(tag):
    """National lag-block theta by variable (real climate vs one shuffled fit)."""
    f = C.BRAZIL_DIR / f"fitted_params_{tag}.json"
    fp = json.loads(f.read_text())
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.3), sharey=True)
    blocks = [f"{lo}-{hi}" for lo, hi in C.LAG_BLOCKS]
    cols = {"temp_mean": "#e6550d", "precip": "#3182bd", "humid_mean": "#31a354"}
    for a, model in zip(ax, ["M1", "M1-shufTT-s0"]):
        if model not in fp:
            continue
        th = np.array([r["theta"] for r in fp[model]])     # (folds, 15)
        for v, var in enumerate(C.CLIM_VARS):
            x = np.arange(len(blocks)) + (v - 1) * 0.12
            vals = th[:, v * len(blocks):(v + 1) * len(blocks)]
            a.plot(x, vals.mean(0), "-o", color=cols[var], label=var.replace("precip", "log(1+precip)"))
            a.vlines(x, vals.min(0), vals.max(0), color=cols[var], alpha=0.6)
        a.axhline(0, c="k", lw=0.8)
        a.set_xticks(range(len(blocks))); a.set_xticklabels(blocks); a.set_xlabel("lag block (weeks)")
        a.set_title(f"{model}: national theta (mean, range over 14 folds)")
    ax[0].set_ylabel("coefficient on standardised anomaly (logit scale)")
    ax[0].legend()
    fig.suptitle(f"Lag-block climate coefficients ({tag}); {prelim(tag)}", fontsize=10)
    fig.tight_layout(); fig.savefig(C.FIG_DIR / f"fig_lag_coefficients{'' if tag == 'weekly_60' else '_' + tag}.png"); plt.close(fig)


def fig_examples(tag):
    """Belo Horizonte 2016/2019 and the largest M1 gain and loss (needs preds_{tag}.npz)."""
    from ..models.dirichlet_multinomial import draw_counts
    z = np.load(C.BRAZIL_DIR / f"preds_{tag}.npz")
    geos, years, region = list(z["geocodes"]), list(z["years"]), z["region"]
    muni = pd.read_csv(C.DATA_DIR / "municipalities.csv").set_index("geocode")
    picks = []
    if C.BELO_HORIZONTE in geos:
        picks += [(C.BELO_HORIZONTE, 2016), (C.BELO_HORIZONTE, 2019)]
    det = pd.read_csv(C.BRAZIL_DIR / f"metrics_detail_{tag}.csv")
    x = det[det.model == "M1"].set_index(["geocode", "year"]).jsd
    y = det[det.model == "B4s"].set_index(["geocode", "year"]).jsd
    gain = (y - x).dropna()
    big = det[det.model == "M1"].set_index(["geocode", "year"]).N
    cand = gain[big.loc[gain.index] >= 2000]
    picks += [cand.idxmax(), cand.idxmin()]   # largest climate gain and largest loss
    rng = np.random.default_rng(3)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for a, (g, yy) in zip(axes.flat, picks):
        m, t = geos.index(g), years.index(yy)
        n = int(z["nsl"][t]); N = int(z["N"][m, t])
        wk = np.arange(1, n + 1)
        a.bar(wk, z["Y"][m, t, :n], color="#bdbdbd", label="observed (hidden)", width=0.9)
        D = draw_counts(z["pred__M1"][m, t, :n], z["phi__M1"][m, t], N, 500, rng, n)
        a.fill_between(wk, np.quantile(D, .1, 0), np.quantile(D, .9, 0), color=C.MODEL_COL["M1"], alpha=0.2, label="M1 80% interval")
        for mod, ls in [("B1", ":"), ("B4s", "--"), ("M1", "-")]:
            a.plot(wk, N * z[f"pred__{mod}"][m, t, :n], ls, color=C.MODEL_COL[mod], lw=2, label=mod)
        jj = {mod: det[(det.model == mod) & (det.geocode == g) & (det.year == yy)].jsd.iloc[0] for mod in ["B1", "B4s", "M1"]}
        a.set_title(f"{muni.loc[g, 'name']} ({muni.loc[g, 'uf']}) {yy}, N={N:,}\n"
                    f"JSD B1 {jj['B1']:.3f} | B4s {jj['B4s']:.3f} | M1 {jj['M1']:.3f}", fontsize=10)
        a.set_xlabel("epi week"); a.set_ylabel("cases per week")
    axes.flat[0].legend(fontsize=8)
    fig.suptitle(f"Example reconstructions from the annual total (top: Belo Horizonte; bottom: largest M1 gain and loss with N>=2000)\n{prelim(tag)}", fontsize=10)
    fig.tight_layout(); fig.savefig(C.FIG_DIR / "fig_example_bh.png"); plt.close(fig)


