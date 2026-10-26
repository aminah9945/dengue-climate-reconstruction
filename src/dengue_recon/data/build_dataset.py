"""Build the tidy weekly dengue + climate dataset from raw downloads.

Output (in DATA_DIR):
  dengue_weekly.parquet / .csv.gz   one row per (geocode, epi_year, epi_week), 2010-2025
  climate_weekly.parquet            NASA POWER weekly climate 2009-2025 (2009 kept for lags)
  municipalities.csv                selected municipalities with metadata
  year_summary.csv                  one row per (geocode, epi_year): totals, completeness

Usage:  python -m dengue_recon.data.build_dataset      (DATA_DIR env var; default <repo>/data)
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd

from .. import config as C
from .epiweeks import calendar, date_to_epiweek

DATA_DIR = C.DATA_DIR
RAW = C.RAW_DIR
Y0, Y1 = 2010, 2025
CLIM_Y0 = 2009
# Last complete epi year.  2025 is downloaded but flagged provisional: notifications
# for recent weeks are still being back-filled when the data are pulled.
PROVISIONAL_YEARS = {2025}
# Known reporting artefacts (state, year) -> flag.  Espirito Santo 2021: ~100 cases
# state-wide vs 10^4-10^5 in other years (notifications recorded outside SINAN).
BAD_STATE_YEARS = {("ES", 2021): "ES_2021_reporting_gap"}
SOFT_FLAG_YEARS = {2020: "covid_2020_reporting_disruption", 2025: "provisional_2025"}

ID_COLS = {"casos": "cases", "casos_est": "cases_est", "tempmin": "id_temp_min",
           "tempmed": "id_temp_mean", "tempmax": "id_temp_max", "umidmin": "id_humid_min",
           "umidmed": "id_humid_mean", "umidmax": "id_humid_max", "pop": "pop"}


def load_infodengue(geocode: int) -> pd.DataFrame:
    df = pd.read_csv(RAW / "infodengue" / f"{geocode}.csv")
    if df.empty:
        return df
    # keep the most recent model version if a week appears more than once
    df = df.sort_values("versao_modelo").drop_duplicates("SE", keep="last")
    df["epi_year"], df["epi_week"] = df["SE"] // 100, df["SE"] % 100
    df["data_iniSE"] = pd.to_datetime(df["data_iniSE"])
    out = df[["epi_year", "epi_week", "data_iniSE"] + list(ID_COLS)].rename(columns=ID_COLS)
    out["geocode"] = geocode
    return out


def load_power(geocode: int) -> pd.DataFrame:
    txt = (RAW / "power" / f"{geocode}.csv").read_text()
    body = txt.split("-END HEADER-")[-1].strip()
    d = pd.read_csv(io.StringIO(body)).replace(-999, np.nan)
    d["date"] = pd.to_datetime(d["YEAR"].astype(str), format="%Y") + pd.to_timedelta(d["DOY"] - 1, unit="D")
    ew = date_to_epiweek(d["date"], CLIM_Y0, Y1)
    d = pd.concat([d.reset_index(drop=True), ew], axis=1)
    g = d.groupby(["epi_year", "epi_week"])
    w = pd.DataFrame({
        "precip_mm": g["PRECTOTCORR"].sum(min_count=7),  # weekly total, needs 7 valid days
        "precip_days": g["PRECTOTCORR"].apply(lambda s: int((s >= 1.0).sum())),  # rainy days >=1mm
        "temp_mean": g["T2M"].mean(),
        "temp_min": g["T2M_MIN"].mean(),   # mean of daily minima
        "temp_max": g["T2M_MAX"].mean(),   # mean of daily maxima
        "humid_mean": g["RH2M"].mean(),
        "n_days": g["T2M"].size(),
    }).reset_index()
    w = w[w["n_days"] == 7].drop(columns="n_days")  # drop partial edge weeks
    w["geocode"] = geocode
    return w


def main():
    sel = pd.read_csv(RAW / "meta" / "selected_municipalities.csv")
    meta = pd.read_csv(RAW / "meta" / "municipalities.csv")
    sel = sel[["geocode", "total_cases"]].rename(columns={"total_cases": "total_cases_2010_2025"}).merge(meta, on="geocode", how="left")
    cal = calendar(Y0, Y1)
    clim_cal = calendar(CLIM_Y0, Y1)

    frames, clim_frames, problems = [], [], []
    for g in sel["geocode"]:
        g = int(g)
        idd = load_infodengue(g)
        if idd.empty:
            problems.append((g, "empty infodengue"))
            continue
        full = cal.assign(geocode=g).merge(idd, on=["geocode", "epi_year", "epi_week"], how="left")
        bad = full["data_iniSE"].notna() & (full["data_iniSE"] != full["week_start"])
        if bad.any():
            problems.append((g, f"{bad.sum()} week-start mismatches"))
        frames.append(full.drop(columns="data_iniSE"))
        pf = RAW / "power" / f"{g}.csv"
        if pf.exists():
            clim_frames.append(load_power(g))
        else:
            problems.append((g, "no POWER file"))

    df = pd.concat(frames, ignore_index=True)
    clim = pd.concat(clim_frames, ignore_index=True)
    clim = clim_cal.merge(clim, on=["epi_year", "epi_week"], how="right")
    clim = clim.sort_values(["geocode", "week_index"]).reset_index(drop=True)

    df = df.merge(clim.drop(columns=["week_start", "n_weeks_year", "week_index"]),
                  on=["geocode", "epi_year", "epi_week"], how="left")
    df = df.merge(sel[["geocode", "name", "uf", "region", "lat", "lon", "capital", "pop2021"]],
                  on="geocode", how="left")
    df["provisional"] = df["epi_year"].isin(PROVISIONAL_YEARS)

    # year-level summaries and within-year shares (shares only for complete years)
    df["week_observed"] = df["cases"].notna()
    ys = df.groupby(["geocode", "epi_year"]).agg(
        n_weeks_year=("n_weeks_year", "first"), weeks_observed=("week_observed", "sum"),
        year_total=("cases", "sum"), peak_week=("cases", lambda s: s.idxmax() if s.notna().any() else np.nan),
        precip_missing=("precip_mm", lambda s: int(s.isna().sum())),
        id_temp_missing=("id_temp_mean", lambda s: int(s.isna().sum()))).reset_index()
    ys["peak_week"] = ys["peak_week"].map(lambda i: df.at[int(i), "epi_week"] if pd.notna(i) else np.nan)
    ys["year_complete"] = ys["weeks_observed"] == ys["n_weeks_year"]
    ys["provisional"] = ys["epi_year"].isin(PROVISIONAL_YEARS)
    ys = ys.merge(sel[["geocode", "uf"]], on="geocode", how="left")
    ys["data_flag"] = [BAD_STATE_YEARS.get((u, y), SOFT_FLAG_YEARS.get(y, ""))
                       for u, y in zip(ys["uf"], ys["epi_year"])]
    # suggested exclusion from training/evaluation: hard artefacts + provisional year
    ys["exclude_suggested"] = ys["data_flag"].isin(list(BAD_STATE_YEARS.values())) | ys["provisional"]
    ys = ys.drop(columns="uf")
    df = df.merge(ys[["geocode", "epi_year", "year_total", "year_complete", "data_flag", "exclude_suggested"]],
                  on=["geocode", "epi_year"], how="left")
    df["share"] = np.where(df["year_complete"] & (df["year_total"] > 0),
                           df["cases"] / df["year_total"].where(df["year_total"] > 0), np.nan)
    df = df.drop(columns="week_observed")

    order = ["geocode", "name", "uf", "region", "epi_year", "epi_week", "week_start",
             "n_weeks_year", "week_index", "cases", "cases_est", "year_total", "share",
             "year_complete", "provisional", "data_flag", "exclude_suggested", "temp_mean", "temp_min", "temp_max",
             "humid_mean", "precip_mm", "precip_days", "id_temp_mean", "id_temp_min",
             "id_temp_max", "id_humid_mean", "id_humid_min", "id_humid_max", "pop",
             "pop2021", "lat", "lon", "capital"]
    df = df[order].sort_values(["geocode", "week_index"]).reset_index(drop=True)

    df.to_parquet(DATA_DIR / "dengue_weekly.parquet", index=False)
    df.to_csv(DATA_DIR / "dengue_weekly.csv.gz", index=False)
    clim.to_parquet(DATA_DIR / "climate_weekly.parquet", index=False)
    ys.to_csv(DATA_DIR / "year_summary.csv", index=False)
    sel.to_csv(DATA_DIR / "municipalities.csv", index=False)
    print(f"rows={len(df)} municipalities={df.geocode.nunique()} "
          f"years={df.epi_year.min()}-{df.epi_year.max()}")
    print("problems:", problems[:20], "... total", len(problems))


if __name__ == "__main__":
    main()
