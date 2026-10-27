"""Compare NASA POWER weather with an alternative daily source (and with Cox et al. ERA5-Land).

For every municipality and variable the script computes the Pearson correlation
between NASA POWER and the alternative source (``--source``: chirps, era5land, era5)

* weekly (epi weeks, from the climate_weekly tables) and monthly (calendar months,
  from the raw daily series: precipitation summed, temperature/humidity averaged);
* on raw values and on anomalies (deviation from the municipality's week-of-year,
  respectively month-of-year, climatology over the compared period; epi week 53 uses
  the week-52 climatology);

and summarises the per-municipality correlations (median, IQR) by macro-region and
overall.  If the optional Cox et al. subset is present (raw/cox_zenodo, ERA5-Land,
2013-2020), monthly values of both sources are also compared with it.  Cox weeks are
ISO-style and binned by their MONTH label, so edge weeks blur month boundaries a little.

Only variables that the alternative source really provides are compared (CHIRPS is
rainfall only).  Outputs go to ``RESULTS_DIR/weather_validation/``:
``summary.csv``, ``per_municipality.csv``, ``summary.md`` and small PNG figures.

Usage:  python -m dengue_recon.data.validate_weather --source chirps
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .. import config as C
from .build_dataset import read_daily

VARS = {"precip": ("precip_mm", "precip", "sum"), "temp_mean": ("temp_mean", "t2m", "mean"),
        "humid_mean": ("humid_mean", "rh", "mean")}
SOURCE_VARS = {"chirps": ["precip"], "era5land": list(VARS), "era5": list(VARS)}
COX_COLS = {"precip": "Rf_cum", "temp_mean": "Tmean", "humid_mean": "RH_mean"}
LABEL = {"power": "NASA POWER", "chirps": "CHIRPS", "era5land": "ERA5-Land (Open-Meteo)",
         "era5": "ERA5 (Open-Meteo)", "cox": "Cox ERA5-Land"}
VAR_LABEL = {"precip": "precipitation", "temp_mean": "mean temperature", "humid_mean": "relative humidity"}


def _r(a: pd.Series, b: pd.Series) -> float:
    ok = a.notna() & b.notna()
    if ok.sum() < 10 or a[ok].std() == 0 or b[ok].std() == 0:
        return np.nan
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def _anom(df: pd.DataFrame, col: str, season: str) -> pd.Series:
    return df[col] - df.groupby(["geocode", season])[col].transform("mean")


def per_muni_corr(df: pd.DataFrame, a: str, b: str, season: str) -> pd.DataFrame:
    """Per-municipality r of columns a and b, raw and as seasonal anomalies."""
    df = df.copy()
    df["_a"], df["_b"] = _anom(df, a, season), _anom(df, b, season)
    out = df.groupby("geocode").apply(
        lambda x: pd.Series({"r_raw": _r(x[a], x[b]), "r_anom": _r(x["_a"], x["_b"]),
                             "mean_a": x[a].mean(), "mean_b": x[b].mean(),
                             "n": int((x[a].notna() & x[b].notna()).sum())}),
        include_groups=False)
    return out.reset_index()


def monthly_from_daily(geocodes, source: str, y0: int, y1: int, var_keys) -> pd.DataFrame:
    rows = []
    for g in geocodes:
        d = read_daily(int(g), source)
        d = d[(d.date.dt.year >= y0) & (d.date.dt.year <= y1)]
        grp = d.groupby([d.date.dt.year.rename("year"), d.date.dt.month.rename("month")])
        m = {}
        for v in var_keys:
            _, dcol, how = VARS[v]
            m[v] = grp[dcol].sum(min_count=1) if how == "sum" else grp[dcol].mean()
            if how == "sum":  # require complete months
                m[v] = m[v].where(grp[dcol].count() == grp[dcol].size())
        rows.append(pd.DataFrame(m).reset_index().assign(geocode=int(g)))
    return pd.concat(rows, ignore_index=True)


def triple_collocation(df: pd.DataFrame, cols: dict, season: str = "month") -> pd.DataFrame:
    """Per-municipality triple collocation on seasonal anomalies (covariance notation;
    Stoffelen 1998, McColl et al. 2014): estimated correlation of each of three sources
    with the unknown truth, R_x^2 = C_xy C_xz / (C_xx C_yz).  Assumes errors that are
    independent between sources and of the signal, which reanalyses sharing
    assimilated observations only approximately satisfy, so the values are indicative."""
    df = df.copy()
    names = list(cols)
    for k, c in cols.items():
        df[k] = _anom(df, c, season)
    rows = []
    for g, x in df.groupby("geocode"):
        x = x[names].dropna()
        if len(x) < 24:
            continue
        cv = x.cov().values
        out = {"geocode": g, "n": len(x)}
        for i, k in enumerate(names):
            j, l = [m for m in range(3) if m != i]
            r2 = cv[i, j] * cv[i, l] / (cv[i, i] * cv[j, l]) if cv[j, l] > 0 else np.nan
            out[f"R_{k}"] = float(np.sqrt(r2)) if r2 > 0 else np.nan
            out[f"R_{k}"] = min(out[f"R_{k}"], 1.0) if np.isfinite(out[f"R_{k}"]) else np.nan
        rows.append(out)
    return pd.DataFrame(rows)


def summarise(pm: pd.DataFrame, muni: pd.DataFrame) -> pd.DataFrame:
    pm = pm.merge(muni[["geocode", "region"]], on="geocode")
    keys = ["comparison", "variable", "resolution"]
    parts = []
    for reg, sub in [("All", pm)] + list(pm.groupby("region")):
        s = sub.groupby(keys).agg(
            n_muni=("geocode", "nunique"),
            r_raw_median=("r_raw", "median"),
            r_anom_median=("r_anom", "median"),
            r_anom_q25=("r_anom", lambda x: x.quantile(0.25)),
            r_anom_q75=("r_anom", lambda x: x.quantile(0.75)),
            mean_a=("mean_a", "mean"), mean_b=("mean_b", "mean")).reset_index()
        parts.append(s.assign(region=reg))
    out = pd.concat(parts, ignore_index=True)
    out["ratio_b_over_a"] = out["mean_b"] / out["mean_a"]
    order = {r: i for i, r in enumerate(["All"] + C.REGIONS)}
    return out.sort_values(keys + ["region"], key=lambda s: s.map(order) if s.name == "region" else s)


def figures(pm: pd.DataFrame, summ: pd.DataFrame, muni: pd.DataFrame, monthly: pd.DataFrame,
            source: str, out_dir) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": "#e5e5e5", "grid.linewidth": 0.6,
                         "axes.axisbelow": True})
    files = []
    alt = LABEL[source]

    # 1. median per-municipality anomaly r by region for each comparison of precipitation
    s = summ[(summ.variable == "precip") & (summ.region != "All")]
    pref = [(f"POWER vs {alt}", "weekly"), (f"POWER vs {alt}", "monthly"),
            ("POWER vs Cox ERA5-Land", "monthly 2013-20"), (f"{alt} vs Cox ERA5-Land", "monthly 2013-20")]
    have = set(s.groupby(["comparison", "resolution"]).groups)
    combos = [c for c in pref if c in have]
    colors = ["#9ecae1", "#2171b5", "#969696", "#e6550d"]
    fig, ax = plt.subplots(figsize=(6.4, 2.9), dpi=110)
    width = 0.8 / max(len(combos), 1)
    for k, (comp, res) in enumerate(combos):
        sub = s[(s.comparison == comp) & (s.resolution == res)].set_index("region").reindex(C.REGIONS)
        x = np.arange(len(C.REGIONS)) + (k - (len(combos) - 1) / 2) * width
        ax.bar(x, sub.r_anom_median, width * 0.92, color=colors[k], label=f"{comp}, {res}")
    ax.set_xticks(range(len(C.REGIONS)), C.REGIONS)
    ax.set_ylim(0, 1)
    ax.set_ylabel("median anomaly r")
    ax.set_title("Precipitation anomaly correlation by region (per-municipality median)", fontsize=9)
    ax.legend(fontsize=6.5, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    f = out_dir / "precip_anomaly_r_by_region.png"
    fig.savefig(f, bbox_inches="tight")
    plt.close(fig)
    files.append(f.name)

    # 2. map of per-municipality monthly precip anomaly r (POWER vs alternative)
    p = pm[(pm.comparison == f"POWER vs {alt}") & (pm.variable == "precip") & (pm.resolution == "monthly")]
    p = p.merge(muni[["geocode", "lat", "lon", "region"]], on="geocode")
    fig, ax = plt.subplots(figsize=(4.2, 4.2), dpi=110)
    sc = ax.scatter(p.lon, p.lat, c=p.r_anom, cmap="viridis", vmin=0.2, vmax=1.0, s=16,
                    edgecolors="white", linewidths=0.4)
    fig.colorbar(sc, ax=ax, shrink=0.75, label="monthly anomaly r")
    ax.set_aspect("equal")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title(f"Rain anomaly r, NASA POWER vs {alt}\nmonthly, per municipality", fontsize=9)
    fig.tight_layout()
    f = out_dir / "precip_anomaly_r_map.png"
    fig.savefig(f, bbox_inches="tight")
    plt.close(fig)
    files.append(f.name)

    # 3. North: example monthly series (Manaus if present)
    g = 1302603 if 1302603 in set(monthly.geocode) else int(
        muni[muni.region == "North"].geocode.iloc[0])
    name = muni.set_index("geocode").loc[g, "name"]
    m = monthly[(monthly.geocode == g) & monthly.year.between(2013, 2020)].sort_values(["year", "month"])
    t = pd.to_datetime(dict(year=m.year, month=m.month, day=15))
    fig, ax = plt.subplots(figsize=(6.4, 2.4), dpi=110)
    ax.plot(t, m.precip_power, lw=1.2, color="#bdbdbd", label="NASA POWER")
    ax.plot(t, m[f"precip_{source}"], lw=1.2, color="#08519c", label=alt)
    if "precip_cox" in m and m.precip_cox.notna().any():
        ax.plot(t, m.precip_cox, lw=1.2, color="#e6550d", ls="--", label="Cox ERA5-Land")
    ax.set_ylabel("mm / month")
    ax.set_title(f"Monthly rainfall, {name} (North), 2013-2020", fontsize=9)
    ax.legend(fontsize=7, frameon=False, ncol=3, loc="upper left")
    fig.tight_layout()
    f = out_dir / "precip_monthly_north_example.png"
    fig.savefig(f, bbox_inches="tight")
    plt.close(fig)
    files.append(f.name)

    return files


def to_markdown(summ: pd.DataFrame) -> str:
    cols = ["comparison", "variable", "resolution", "region", "n_muni", "r_raw_median",
            "r_anom_median", "r_anom_q25", "r_anom_q75", "ratio_b_over_a"]
    t = summ[cols].copy()
    for c in cols[5:]:
        t[c] = t[c].map(lambda v: f"{v:.2f}")
    t["region"] = t["region"].map(lambda r: f"**{r}**" if r == "North" else r)
    head = ("| comparison | variable | res. | region | n | r raw | r anomaly | anom. IQR | mean ratio (B/A) |\n"
            "|---|---|---|---|---:|---:|---:|---|---:|\n")
    body = "".join(f"| {r.comparison} | {r.variable} | {r.resolution} | {r.region} | {r.n_muni} | "
                   f"{r.r_raw_median} | {r.r_anom_median} | {r.r_anom_q25}-{r.r_anom_q75} | {r.ratio_b_over_a} |\n"
                   for r in t.itertuples())
    return head + body


def main(source: str = "chirps", y0: int = 2009, y1: int = 2024):
    out_dir = C.RESULTS_DIR / "weather_validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    muni = pd.read_csv(C.DATA_DIR / "municipalities.csv")
    var_keys = SOURCE_VARS[source]
    alt = LABEL[source]
    pms = []

    # weekly: climate_weekly (POWER) vs climate_weekly_<source>
    pw = pd.read_parquet(C.DATA_DIR / "climate_weekly.parquet")
    aw = pd.read_parquet(C.DATA_DIR / f"climate_weekly_{source}.parquet")
    w = pw.merge(aw, on=["geocode", "epi_year", "epi_week"], suffixes=("_power", "_alt"))
    w = w[w.epi_year.between(y0, y1)]
    w["woy"] = w.epi_week.clip(upper=52)
    for v in var_keys:
        col = VARS[v][0]
        pm = per_muni_corr(w, f"{col}_power", f"{col}_alt", "woy")
        pms.append(pm.assign(comparison=f"POWER vs {alt}", variable=v, resolution="weekly"))

    # monthly from daily
    geos = sorted(set(w.geocode))
    mp = monthly_from_daily(geos, "power", y0, y1, var_keys)
    ma = monthly_from_daily(geos, source, y0, y1, var_keys)
    monthly = mp.merge(ma, on=["geocode", "year", "month"], suffixes=("_power", f"_{source}"))
    for v in var_keys:
        pm = per_muni_corr(monthly, f"{v}_power", f"{v}_{source}", "month")
        pms.append(pm.assign(comparison=f"POWER vs {alt}", variable=v, resolution="monthly"))

    # Cox et al. ERA5-Land (2013-2020), monthly
    cox_f = C.RAW_DIR / "cox_zenodo" / "cox_denv_report_subset.parquet"
    if cox_f.exists():
        cox = pd.read_parquet(cox_f, columns=["MUNICIP", "YEAR", "MONTH"] + list(COX_COLS.values()))
        cox = cox.dropna(subset=["YEAR", "MUNICIP"])
        cox["geocode"] = cox.MUNICIP.astype(int)
        cox = cox[cox.geocode.isin(geos)]
        g = cox.groupby(["geocode", "YEAR", "MONTH"])
        cm = pd.DataFrame({"precip": g["Rf_cum"].sum() * 1000.0,  # m -> mm
                           "temp_mean": g["Tmean"].mean(), "humid_mean": g["RH_mean"].mean()})
        cm = cm.add_suffix("_cox").reset_index().rename(columns={"YEAR": "year", "MONTH": "month"})
        cm[["year", "month"]] = cm[["year", "month"]].astype(int)
        # POWER daily for all three variables over 2013-2020
        mp3 = monthly_from_daily(geos, "power", 2013, 2020, list(VARS)).add_suffix("_power3")
        mp3 = mp3.rename(columns={"geocode_power3": "geocode", "year_power3": "year", "month_power3": "month"})
        monthly = monthly.merge(cm, on=["geocode", "year", "month"], how="left")
        mc = mp3.merge(cm, on=["geocode", "year", "month"]).merge(
            ma, on=["geocode", "year", "month"], how="left")
        for v in VARS:
            pm = per_muni_corr(mc, f"{v}_power3", f"{v}_cox", "month")
            pms.append(pm.assign(comparison="POWER vs Cox ERA5-Land", variable=v, resolution="monthly 2013-20"))
        for v in var_keys:
            pm = per_muni_corr(mc, v, f"{v}_cox", "month")
            pms.append(pm.assign(comparison=f"{alt} vs Cox ERA5-Land", variable=v, resolution="monthly 2013-20"))

    tc_md = ""
    if cox_f.exists() and "precip" in var_keys:
        tc = triple_collocation(mc, {"power": "precip_power3", source: "precip", "cox": "precip_cox"})
        tc = tc.merge(muni[["geocode", "region"]], on="geocode")
        rc = [c for c in tc.columns if c.startswith("R_")]
        tcs = pd.concat([tc[rc].median().to_frame("All").T] +
                        [tc[tc.region == r][rc].median().to_frame(r).T for r in C.REGIONS])
        tcs.insert(0, "n_muni", [len(tc)] + [int((tc.region == r).sum()) for r in C.REGIONS])
        tcs.index.name = "region"
        tcs.to_csv(out_dir / "triple_collocation_precip.csv", float_format="%.4f")
        tc.to_csv(out_dir / "triple_collocation_precip_per_municipality.csv", index=False, float_format="%.4f")
        tc_md = ("\n## Triple collocation, monthly rainfall anomalies 2013-2020\n\n"
                 "Median per-municipality estimated correlation with the (unknown) true anomaly.\n\n"
                 f"| region | n | NASA POWER | {alt} | Cox ERA5-Land |\n|---|---:|---:|---:|---:|\n" +
                 "".join(f"| {'**North**' if i == 'North' else i} | {int(r.n_muni)} | {r.R_power:.2f} | "
                         f"{r['R_' + source]:.2f} | {r.R_cox:.2f} |\n" for i, r in tcs.iterrows()))
        print(tc_md)

    pm = pd.concat(pms, ignore_index=True)
    summ = summarise(pm, muni)
    pm.merge(muni[["geocode", "name", "uf", "region"]], on="geocode").to_csv(
        out_dir / "per_municipality.csv", index=False, float_format="%.4f")
    summ.to_csv(out_dir / "summary.csv", index=False, float_format="%.4f")
    md = to_markdown(summ)
    (out_dir / "summary.md").write_text(
        f"# Weather source validation ({alt} vs NASA POWER, {y0}-{y1})\n\n"
        "Median of per-municipality Pearson r; anomaly = deviation from the municipality's "
        "week-of-year (weekly) or month-of-year (monthly) climatology. "
        "Mean ratio = mean of second source / mean of first source.\n\n" + md + tc_md)
    files = figures(pm, summ, muni, monthly, source, out_dir)
    print(md)
    print("figures:", files)
    return summ


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="chirps", choices=list(SOURCE_VARS))
    ap.add_argument("--y0", type=int, default=2009)
    ap.add_argument("--y1", type=int, default=2024)
    a = ap.parse_args()
    main(a.source, a.y0, a.y1)
