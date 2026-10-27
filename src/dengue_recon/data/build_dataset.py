"""Build the tidy weekly dengue + climate dataset from raw downloads.

Output (in DATA_DIR):
  dengue_weekly.parquet / .csv.gz   one row per (geocode, epi_year, epi_week), 2010-2025
  climate_weekly.parquet            NASA POWER weekly climate 2009-2025 (2009 kept for lags)
  municipalities.csv                selected municipalities with metadata
  year_summary.csv                  one row per (geocode, epi_year): totals, completeness

Optional alternative weather sources (same weekly schema, new file names only):
  climate_weekly_<source>.parquet   source in {era5land, era5, chirps}
  oni_weekly.parquet, oni_monthly.csv   NOAA CPC ONI (see oni.py)

Usage:  python -m dengue_recon.data.build_dataset      (DATA_DIR env var; default <repo>/data)
        python -m dengue_recon.data.build_dataset --source chirps   (climate table only)
"""
from __future__ import annotations

import argparse
import io
import json

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

SOURCES = ("power", "era5land", "era5", "chirps")
DAILY_COLS = ("date", "precip", "t2m", "tmax", "tmin", "rh")
CLIM_ORDER = ["epi_year", "epi_week", "week_start", "n_weeks_year", "week_index", "precip_mm",
              "precip_days", "temp_mean", "temp_min", "temp_max", "humid_mean", "geocode"]

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


def aggregate_daily_to_epiweeks(d: pd.DataFrame, geocode: int) -> pd.DataFrame:
    """Daily -> complete epi weeks (the NASA POWER rule, shared by every source).

    `d` needs columns date, precip, t2m, tmax, tmin, rh (daily values; NaN = missing).
    Precipitation is summed (all 7 days required), temperatures and humidity are
    weekly means of the daily values; weeks with fewer than 7 daily rows are dropped.
    """
    d = d.reset_index(drop=True)
    ew = date_to_epiweek(d["date"], CLIM_Y0, Y1)
    d = pd.concat([d, ew], axis=1)
    g = d.groupby(["epi_year", "epi_week"])
    w = pd.DataFrame({
        "precip_mm": g["precip"].sum(min_count=7),  # weekly total, needs 7 valid days
        "precip_days": g["precip"].apply(lambda s: int((s >= 1.0).sum())),  # rainy days >=1mm
        "temp_mean": g["t2m"].mean(),
        "temp_min": g["tmin"].mean(),   # mean of daily minima
        "temp_max": g["tmax"].mean(),   # mean of daily maxima
        "humid_mean": g["rh"].mean(),
        "n_days": g["date"].size(),
    }).reset_index()
    w = w[w["n_days"] == 7].drop(columns="n_days")  # drop partial edge weeks
    w["geocode"] = geocode
    return w


def read_power_daily(geocode: int) -> pd.DataFrame:
    """NASA POWER raw CSV -> daily frame (date, precip, t2m, tmax, tmin, rh)."""
    txt = (RAW / "power" / f"{geocode}.csv").read_text()
    body = txt.split("-END HEADER-")[-1].strip()
    d = pd.read_csv(io.StringIO(body)).replace(-999, np.nan)
    d["date"] = pd.to_datetime(d["YEAR"].astype(str), format="%Y") + pd.to_timedelta(d["DOY"] - 1, unit="D")
    return d.rename(columns={"PRECTOTCORR": "precip", "T2M": "t2m", "T2M_MAX": "tmax",
                             "T2M_MIN": "tmin", "RH2M": "rh"})[list(DAILY_COLS)]


def read_openmeteo_daily(geocode: int, source: str = "era5land") -> pd.DataFrame:
    """Open-Meteo archive JSON (raw/<source>/<geocode>.json) -> daily frame."""
    js = json.loads((RAW / source / f"{geocode}.json").read_text())
    dd = js["daily"]
    d = pd.DataFrame({"date": pd.to_datetime(dd["time"]),
                      "precip": dd["precipitation_sum"], "t2m": dd["temperature_2m_mean"],
                      "tmax": dd["temperature_2m_max"], "tmin": dd["temperature_2m_min"],
                      "rh": dd["relative_humidity_2m_mean"]})
    return d.astype({c: float for c in DAILY_COLS if c != "date"})


def read_chirps_daily(geocode: int) -> pd.DataFrame:
    """ClimateSERV CHIRPS JSON (raw/chirps/<geocode>.json) -> daily precipitation.
    Temperature and humidity are NaN (CHIRPS is rainfall only)."""
    js = json.loads((RAW / "chirps" / f"{geocode}.json").read_text())
    rows = js["data"]
    v = np.array([r["value"]["avg"] for r in rows], float)
    v[v < -900] = np.nan  # ClimateSERV no-data value (-9999)
    d = pd.DataFrame({"date": pd.to_datetime([r["date"] for r in rows], format="%m/%d/%Y"),
                      "precip": v})
    for c in ("t2m", "tmax", "tmin", "rh"):
        d[c] = np.nan
    return d.sort_values("date").drop_duplicates("date")[list(DAILY_COLS)]


def read_daily(geocode: int, source: str = "power") -> pd.DataFrame:
    """Daily weather of one municipality from the raw cache of `source`."""
    if source == "power":
        return read_power_daily(geocode)
    if source in ("era5land", "era5"):
        return read_openmeteo_daily(geocode, source)
    if source == "chirps":
        return read_chirps_daily(geocode)
    raise ValueError(f"unknown weather source {source!r}; choose from {SOURCES}")


def load_power(geocode: int) -> pd.DataFrame:
    return aggregate_daily_to_epiweeks(read_power_daily(geocode), geocode)


def load_weather(geocode: int, source: str = "power") -> pd.DataFrame:
    """Weekly climate of one municipality from `source` (same columns as load_power)."""
    return aggregate_daily_to_epiweeks(read_daily(geocode, source), geocode)


def build_climate(source: str = "power", geocodes=None, write: bool = True) -> pd.DataFrame:
    """Weekly climate table for `source` with exactly the schema of climate_weekly.parquet.

    source="power" reproduces climate_weekly.parquet.  Other sources are written to
    climate_weekly_<source>.parquet and never overwrite the NASA POWER file.
    For "chirps" (rainfall only) precip_mm and precip_days come from CHIRPS and the
    temperature/humidity columns are taken from NASA POWER, so the table stays a
    drop-in replacement; this is documented in data/README.md.
    """
    if geocodes is None:
        geocodes = pd.read_csv(RAW / "meta" / "selected_municipalities.csv")["geocode"]
    clim_cal = calendar(CLIM_Y0, Y1)
    frames, missing = [], []
    for g in geocodes:
        g = int(g)
        try:
            frames.append(load_weather(g, source))
        except FileNotFoundError:
            missing.append(g)
    if missing:
        print(f"[build_climate:{source}] no raw file for {len(missing)} municipalities: {missing[:10]}")
    clim = pd.concat(frames, ignore_index=True)
    if source == "chirps":
        pw = pd.concat([load_power(int(g)) for g in clim["geocode"].unique()], ignore_index=True)
        tcols = ["temp_mean", "temp_min", "temp_max", "humid_mean"]
        clim = clim.drop(columns=tcols).merge(pw[["geocode", "epi_year", "epi_week"] + tcols],
                                              on=["geocode", "epi_year", "epi_week"], how="left")
    clim = clim_cal.merge(clim, on=["epi_year", "epi_week"], how="right")
    clim = clim.sort_values(["geocode", "week_index"]).reset_index(drop=True)
    clim = clim[CLIM_ORDER]
    if write and source != "power":
        out = DATA_DIR / f"climate_weekly_{source}.parquet"
        clim.to_parquet(out, index=False)
        print(f"[build_climate:{source}] wrote {out} rows={len(clim)}")
    return clim


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
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="power", choices=SOURCES,
                    help="power (default): full build; other sources: only climate_weekly_<source>.parquet")
    a = ap.parse_args()
    if a.source == "power":
        main()
    else:
        build_climate(a.source)
