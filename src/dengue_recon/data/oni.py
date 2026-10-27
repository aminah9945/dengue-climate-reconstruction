"""Oceanic Nino Index (ONI, NOAA CPC) as a monthly table and on the epi-week calendar.

Source: https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt (public domain).
Columns of the raw file: SEAS (3-month season, e.g. DJF), YR, TOTAL (Nino-3.4 SST,
deg C, ERSSTv5), ANOM (the ONI: anomaly vs. a centred 30-year base period).

* Monthly table: each 3-month season is assigned to its **centre month**
  (DJF -> January, JFM -> February, ..., NDJ -> December).  The YR of the file is the
  year of the centre month (DJF 2010 = Dec 2009 - Feb 2010, centred on Jan 2010).
* Weekly table: every epi week gets the ONI of the calendar month that contains its
  midpoint (the Wednesday, week_start + 3 days).

Note: the ONI of month m averages months m-1, m, m+1, so it is known only about one
month after m ends; use lags of at least ~6 weeks if the index must be available in
real time.

Usage:  python -m dengue_recon.data.oni     (downloads if needed, writes
        DATA_DIR/oni_monthly.csv and DATA_DIR/oni_weekly.parquet)
"""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd

from .. import config as C
from .epiweeks import calendar

SEASON_CENTER_MONTH = {s: i + 1 for i, s in enumerate(
    ["DJF", "JFM", "FMA", "MAM", "AMJ", "MJJ", "JJA", "JAS", "ASO", "SON", "OND", "NDJ"])}


def parse_oni(text: str) -> pd.DataFrame:
    """Raw CPC text -> monthly table: year, month, season, oni, nino34_total."""
    d = pd.read_csv(io.StringIO(text), sep=r"\s+")
    d.columns = [c.strip().upper() for c in d.columns]
    out = pd.DataFrame({
        "year": d["YR"].astype(int),
        "month": d["SEAS"].map(SEASON_CENTER_MONTH).astype(int),
        "season": d["SEAS"],
        "oni": pd.to_numeric(d["ANOM"], errors="coerce"),
        "nino34_total": pd.to_numeric(d["TOTAL"], errors="coerce"),
    })
    out = out[out["oni"] > -90]  # guard against fill values
    return out.sort_values(["year", "month"]).reset_index(drop=True)


def oni_weekly(monthly: pd.DataFrame, y0: int = 2009, y1: int = 2025) -> pd.DataFrame:
    """Epi-week ONI: the value of the month containing each week's midpoint (Wednesday)."""
    cal = calendar(y0, y1)
    mid = cal["week_start"] + pd.Timedelta(days=3)
    cal["year"], cal["month"] = mid.dt.year, mid.dt.month
    w = cal.merge(monthly[["year", "month", "oni"]], on=["year", "month"], how="left")
    w = w.rename(columns={"year": "mid_year", "month": "mid_month"})
    return w[["epi_year", "epi_week", "week_start", "n_weeks_year", "week_index",
              "mid_year", "mid_month", "oni"]]


def main(data_dir: Path | None = None) -> pd.DataFrame:
    from .download import download_oni

    data_dir = Path(data_dir or C.DATA_DIR)
    download_oni()
    monthly = parse_oni((C.RAW_DIR / "oni" / "oni.ascii.txt").read_text())
    monthly.to_csv(data_dir / "oni_monthly.csv", index=False)
    w = oni_weekly(monthly)
    w.to_parquet(data_dir / "oni_weekly.parquet", index=False)
    print(f"[oni] monthly {monthly.year.min()}-{monthly.year.max()} ({len(monthly)} rows); "
          f"weekly rows={len(w)}, missing={int(w.oni.isna().sum())}")
    return w


if __name__ == "__main__":
    main()
