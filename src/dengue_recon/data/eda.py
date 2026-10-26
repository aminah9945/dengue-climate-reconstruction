"""Data-foundation EDA and smoke-test baselines (Step 1 checks; Step 3 preview).

Produces the coverage heat map, seasonal curves by region, peak-week
variability, the smoke test of the evaluation (uniform, seasonal LOYO, seasonal
past-only, with exact-sum assertions) and the JSD sampling-noise floor. Writes
figures to ``REPORTS_DIR/figures`` and tables to ``REPORTS_DIR``.
``baseline_smoke_test_detail.csv`` is also the reference for sanity check 7.

Usage:  python -m dengue_recon.data.eda
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .. import config as C  # noqa: E402
from ..evaluation.allocation import allocate_integers  # noqa: E402
from ..evaluation.metrics import jsd, mae_shares, peak_week_error  # noqa: E402
from ..features.seasonal import adapt_length, seasonal_profile  # noqa: E402

MIN_TOTAL = 100          # a year enters peak/evaluation analyses only if >= this many cases
EVAL_YEARS = range(2010, 2025)  # 2025 provisional -> excluded from evaluation
REGIONS = ["North", "Northeast", "Center-West", "Southeast", "South"]
REG_COL = {"North": "#1b9e77", "Northeast": "#d95f02", "Center-West": "#7570b3",
           "Southeast": "#e7298a", "South": "#66a61e"}


def circ_stats(weeks, n=52.18):
    """Circular mean and SD (in weeks) of peak weeks."""
    ang = 2 * np.pi * (np.asarray(weeks, float) - 1) / n
    C, S = np.cos(ang).mean(), np.sin(ang).mean()
    R = np.hypot(C, S)
    mean_w = (np.arctan2(S, C) % (2 * np.pi)) * n / (2 * np.pi) + 1
    sd_w = np.sqrt(-2 * np.log(max(R, 1e-12))) * n / (2 * np.pi)
    return mean_w, sd_w


def circ_dist(a, b, n=52):
    """Circular distance between weeks."""
    d = np.abs(np.asarray(a) - np.asarray(b))
    return np.minimum(d, n - d)


# centre-of-mass week (circular) is a more stable timing measure than argmax
def com_week(share):
    """Circular centre-of-mass week of a share vector."""
    n = len(share)
    ang = 2 * np.pi * np.arange(n) / n
    return (np.arctan2((share * np.sin(ang)).sum(), (share * np.cos(ang)).sum()) % (2 * np.pi)) * n / (2 * np.pi) + 1


def main():
    """Run the EDA and the smoke-test baselines; print the summaries."""
    DATA_DIR = C.DATA_DIR
    REPORT_DIR = C.REPORTS_DIR
    FIG = REPORT_DIR / "figures"
    FIG.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(DATA_DIR / "dengue_weekly.parquet")
    ys = pd.read_csv(DATA_DIR / "year_summary.csv")
    muni = pd.read_csv(DATA_DIR / "municipalities.csv")
    out = {}

    # ------------------------------------------------------------------ coverage
    cov = df.assign(obs=df["id_temp_mean"].notna()).groupby(["geocode", "epi_year"])["obs"].mean().unstack()
    cov = cov.loc[muni.sort_values(["region", "uf"])["geocode"]]
    logc = np.log10(1 + ys.pivot(index="geocode", columns="epi_year", values="year_total")).loc[cov.index]
    fig, ax = plt.subplots(1, 2, figsize=(12, 7), sharey=True)
    im0 = ax[0].imshow(cov.values, aspect="auto", cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
    ax[0].set_title("Fraction of weeks with InfoDengue station temperature\n(case counts and NASA POWER climate: 100% complete)")
    im1 = ax[1].imshow(logc.values, aspect="auto", cmap="magma_r", interpolation="nearest")
    ax[1].set_title("log10(1 + annual cases)")
    for a in ax:
        a.set_xticks(range(len(cov.columns)))
        a.set_xticklabels(cov.columns, rotation=90, fontsize=7)
        a.set_xlabel("epi year")
    reg_sorted = muni.set_index("geocode").loc[cov.index, "region"].values
    bounds = [0] + [i for i in range(1, len(reg_sorted)) if reg_sorted[i] != reg_sorted[i - 1]] + [len(reg_sorted)]
    ax[0].set_yticks([(bounds[i] + bounds[i + 1]) / 2 for i in range(len(bounds) - 1)])
    ax[0].set_yticklabels([reg_sorted[b] for b in bounds[:-1]])
    for b in bounds[1:-1]:
        for a in ax:
            a.axhline(b - 0.5, color="white", lw=1)
    fig.colorbar(im0, ax=ax[0], fraction=0.04)
    fig.colorbar(im1, ax=ax[1], fraction=0.04)
    fig.suptitle("Data coverage: municipalities (rows, grouped by region) x epi years")
    fig.tight_layout()
    fig.savefig(FIG / "coverage_heatmap.png", dpi=130)
    plt.close(fig)

    clim_miss = df[df.epi_year <= 2024][["temp_mean", "precip_mm", "humid_mean", "id_temp_mean", "id_humid_mean"]].isna().mean()
    out["climate_missing_fraction_2010_2024"] = clim_miss.round(4).to_dict()
    out["case_weeks_missing_fraction_2010_2024"] = float(df[df.epi_year <= 2024]["cases"].isna().mean())

    # ------------------------------------------------------- seasonal curves by region
    good = ys[(ys.year_complete) & (ys.year_total >= MIN_TOTAL) & (ys.epi_year <= 2024) & (~ys.exclude_suggested)]
    good = good.merge(muni[["geocode", "region"]], on="geocode")
    dg = df.merge(good[["geocode", "epi_year"]], on=["geocode", "epi_year"])
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True)
    for ax, reg in zip(axes.flat, REGIONS):
        sub = dg[dg.region == reg]
        if sub.empty:
            continue
        piv = sub.pivot_table(index=["geocode", "epi_year"], columns="epi_week", values="share")
        q = piv.quantile([0.25, 0.5, 0.75])
        ax.fill_between(q.columns, q.loc[0.25], q.loc[0.75], color=REG_COL[reg], alpha=0.25, lw=0)
        ax.plot(q.columns, q.loc[0.5], color=REG_COL[reg], lw=2)
        ax.plot(piv.columns, piv.mean(), color="k", lw=1, ls="--")
        ax.set_title(f"{reg} ({sub.geocode.nunique()} munis, {len(piv)} muni-years)")
        ax.set_xlabel("epi week")
        ax.set_ylabel("weekly share of annual cases")
    ax = axes.flat[5]
    for reg in REGIONS:
        sub = dg[dg.region == reg]
        if not sub.empty:
            ax.plot(sub.groupby("epi_week")["share"].mean(), color=REG_COL[reg], label=reg, lw=2)
    ax.legend(fontsize=8)
    ax.set_title("Mean weekly share, all regions")
    fig.suptitle(f"Seasonal curves (median + IQR across municipality-years; dashed = mean); years with >= {MIN_TOTAL} cases, 2010-2024")
    fig.tight_layout()
    fig.savefig(FIG / "seasonal_curves_by_region.png", dpi=130)
    plt.close(fig)

    # ------------------------------------------------------- peak-week variability
    com = dg.groupby(["geocode", "epi_year"]).apply(lambda s: com_week(s["share"].values), include_groups=False).rename("com_week")
    good = good.merge(com.reset_index(), on=["geocode", "epi_year"])
    rows = []
    for g, sub in good.groupby("geocode"):
        if len(sub) < 5:
            continue
        m, sd = circ_stats(sub["peak_week"])
        med = np.median(sub["peak_week"])  # fine for most munis (peaks mid-year)
        dev = circ_dist(sub["peak_week"], round(m))
        mc, sdc = circ_stats(sub["com_week"])
        rows.append({"geocode": g, "n_years": len(sub), "peak_mean": m, "peak_circ_sd": sd,
                     "peak_range": sub["peak_week"].max() - sub["peak_week"].min(),
                     "peak_mad_from_circmean": float(np.median(dev)),
                     "frac_years_peak_off_gt4wk": float((dev > 4).mean()),
                     "com_circ_sd": sdc})
    pv = pd.DataFrame(rows).merge(muni[["geocode", "name", "uf", "region"]], on="geocode")
    pv.to_csv(REPORT_DIR / "peak_week_variability.csv", index=False)
    out["peak_var_overall"] = pv[["n_years", "peak_circ_sd", "peak_mad_from_circmean",
                                  "frac_years_peak_off_gt4wk", "com_circ_sd"]].describe().round(2).to_dict()
    out["peak_var_by_region_median"] = pv.groupby("region")[["peak_circ_sd", "peak_mad_from_circmean",
                                                             "frac_years_peak_off_gt4wk", "com_circ_sd"]].median().round(2).to_dict()

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    data = [pv.loc[pv.region == r, "peak_circ_sd"].values for r in REGIONS]
    bp = ax[0].boxplot(data, tick_labels=REGIONS, patch_artist=True)
    for patch, r in zip(bp["boxes"], REGIONS):
        patch.set_facecolor(REG_COL[r])
        patch.set_alpha(0.5)
    ax[0].set_ylabel("circular SD of peak week across years (weeks)")
    ax[0].set_title("Year-to-year variation of peak week, per municipality")
    # year-by-year peak week of the 30 largest municipalities
    top = muni.sort_values("total_cases_2010_2025", ascending=False).head(30)["geocode"].tolist()
    pk = good[good.geocode.isin(top)].pivot(index="geocode", columns="epi_year", values="peak_week")
    pk = pk.loc[[g for g in top if g in pk.index]]
    im = ax[1].imshow(pk.values, aspect="auto", cmap="twilight", vmin=1, vmax=53, interpolation="nearest")
    ax[1].set_yticks(range(len(pk)))
    ax[1].set_yticklabels(muni.set_index("geocode").loc[pk.index, "name"].str[:18] + " (" +
                          muni.set_index("geocode").loc[pk.index, "uf"] + ")", fontsize=6)
    ax[1].set_xticks(range(len(pk.columns)))
    ax[1].set_xticklabels(pk.columns, rotation=90, fontsize=7)
    ax[1].set_title(f"Peak epi week, 30 highest-burden municipalities (blank: <{MIN_TOTAL} cases)")
    fig.colorbar(im, ax=ax[1], label="peak epi week")
    fig.tight_layout()
    fig.savefig(FIG / "peak_week_variation.png", dpi=130)
    plt.close(fig)

    # ---------------------------------------------------------- smoke-test baselines
    shares = {}
    for (g, y), sub in df[df.year_complete & (df.year_total > 0) & ~df.exclude_suggested].groupby(["geocode", "epi_year"]):
        shares.setdefault(g, {})[y] = sub.sort_values("epi_week")["share"].values
    counts = {(g, y): sub.sort_values("epi_week")["cases"].values.astype(int)
              for (g, y), sub in df[df.year_complete].groupby(["geocode", "epi_year"])}

    res = []
    for _, r in good.iterrows():
        g, y = r.geocode, r.epi_year
        if y not in EVAL_YEARS:
            continue
        true = shares[g][y]
        n, tot = len(true), int(r.year_total)
        # training pool excludes provisional 2025 as well as the test year
        sb = {k: v for k, v in shares[g].items() if k <= 2024}
        preds = {"uniform": np.full(n, 1.0 / n)}
        loyo = seasonal_profile(sb, exclude={y})
        if loyo is not None:
            preds["seasonal_LOYO"] = adapt_length(loyo, n)
        past = seasonal_profile(sb, only_before=y)
        if past is not None:
            preds["seasonal_past_only"] = adapt_length(past, n)
        for name, p in preds.items():
            alloc = allocate_integers(p, tot)
            assert alloc.sum() == tot
            res.append({"geocode": g, "epi_year": y, "region": r.region, "model": name,
                        "year_total": tot, "jsd": jsd(true, p), "peak_err": peak_week_error(true, p),
                        "mae_share": mae_shares(true, p),
                        "mae_counts": float(np.mean(np.abs(alloc - counts[(g, y)])))})
    res = pd.DataFrame(res)
    res.to_csv(REPORT_DIR / "baseline_smoke_test_detail.csv", index=False)
    common = res[res.model == "seasonal_past_only"][["geocode", "epi_year"]]
    summ = (res.merge(common, on=["geocode", "epi_year"])
            .groupby("model")[["jsd", "peak_err", "mae_share", "mae_counts"]].agg(["mean", "median"]).round(4))
    summ.to_csv(REPORT_DIR / "baseline_smoke_test_summary.csv")
    byreg = (res.merge(common, on=["geocode", "epi_year"])
             .groupby(["region", "model"])[["jsd", "peak_err"]].mean().round(3).unstack())
    byreg.to_csv(REPORT_DIR / "baseline_smoke_test_by_region.csv")
    byyear = res.groupby(["epi_year", "model"])["jsd"].mean().round(3).unstack()
    byyear.to_csv(REPORT_DIR / "baseline_smoke_test_by_year.csv")

    print("n_municipalities", df.geocode.nunique())
    print("good muni-years", len(good), "munis with >=5 good years", len(pv))
    for k, v in out.items():
        print(k, v)
    print(summ.to_string())
    print(byreg.to_string())
    print(byyear.to_string())
    print("n eval muni-years (common set):", len(common))

    # ------------------------------------------------ sampling-noise floor for JSD
    # JSD between the true weekly shares and a multinomial resample of the same total:
    # even a perfect model of the expected curve cannot beat this on average.
    rng = np.random.default_rng(0)
    floor = []
    for _, r in common.merge(good, on=["geocode", "epi_year"]).iterrows():
        p = shares[r.geocode][r.epi_year]
        sim = rng.multinomial(int(r.year_total), p)
        floor.append({"region": r.region, "year_total": r.year_total, "jsd_floor": jsd(p, sim)})
    floor = pd.DataFrame(floor)
    floor.to_csv(REPORT_DIR / "jsd_noise_floor.csv", index=False)
    print("JSD noise floor mean/median:", round(floor.jsd_floor.mean(), 4), round(floor.jsd_floor.median(), 4))
    print(floor.groupby(pd.cut(floor.year_total, [99, 300, 1000, 5000, 1e7]), observed=True).jsd_floor.mean().round(4))


if __name__ == "__main__":
    main()
