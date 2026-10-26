"""Cross-check our sources against Cox et al. (Zenodo 10.5281/zenodo.13286671), 2013-2020.

* annual dengue totals: InfoDengue `casos` vs Cox SINAN-derived CASES
* monthly precipitation: NASA POWER (MERRA-2, municipal seat point) vs Cox ERA5-Land
  population-weighted Rf_cum (metres -> mm), aggregated by calendar month.
Cox weeks are labelled differently from SINAN epi weeks (ISO-style labels), so we
compare at annual / monthly resolution only.

Requires the optional Cox subset (``python -m dengue_recon.data.download cox``).
Writes ``source_validation_precip.csv`` and ``source_validation_annual_cases.csv``
to ``config.REPORTS_DIR``.

Usage:  python -m dengue_recon.data.validate_sources
"""
import io

import numpy as np
import pandas as pd

from .. import config as C

DATA_DIR = C.DATA_DIR
RAW = C.RAW_DIR


def main():
    """Run the annual-total and monthly-rainfall cross-checks and print the summaries."""
    REPORT_DIR = C.REPORTS_DIR
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    cox = pd.read_parquet(RAW / "cox_zenodo" / "cox_denv_report_subset.parquet")
    cox = cox.dropna(subset=["YEAR"])
    cox["geocode"] = cox["MUNICIP"].astype(int)
    muni = pd.read_csv(DATA_DIR / "municipalities.csv")
    cox = cox[cox.geocode.isin(muni.geocode)]
    ys = pd.read_csv(DATA_DIR / "year_summary.csv")

    a = cox.groupby(["geocode", "YEAR"])["CASES"].sum().rename("cox_cases").reset_index()
    a = a.merge(ys.rename(columns={"epi_year": "YEAR"})[["geocode", "YEAR", "year_total"]], on=["geocode", "YEAR"])
    a["rel_diff"] = (a.year_total - a.cox_cases) / a.cox_cases.where(a.cox_cases > 0)
    print("annual totals: n=", len(a), "corr(log)=", np.corrcoef(np.log1p(a.cox_cases), np.log1p(a.year_total))[0, 1].round(4),
          "median |rel diff|=", a.rel_diff.abs().median().round(4), "90th pct=", a.rel_diff.abs().quantile(0.9).round(4))

    # monthly precip: Cox weekly Rf_cum assigned to MONTH column; POWER daily summed by month
    cm = cox.groupby(["geocode", "YEAR", "MONTH"])["Rf_cum"].sum().mul(1000).rename("cox_mm").reset_index()
    rows = []
    for g in muni.geocode:
        txt = (RAW / "power" / f"{g}.csv").read_text().split("-END HEADER-")[-1].strip()
        d = pd.read_csv(io.StringIO(txt)).replace(-999, np.nan)
        d["date"] = pd.to_datetime(d.YEAR.astype(str), format="%Y") + pd.to_timedelta(d.DOY - 1, unit="D")
        d = d[(d.date.dt.year >= 2013) & (d.date.dt.year <= 2020)]
        m = d.groupby([d.date.dt.year, d.date.dt.month])["PRECTOTCORR"].sum()
        m.index.names = ["YEAR", "MONTH"]
        rows.append(m.rename("power_mm").reset_index().assign(geocode=g))
    pm = pd.concat(rows)
    mm = cm.merge(pm, on=["geocode", "YEAR", "MONTH"])
    r_all = np.corrcoef(mm.cox_mm, mm.power_mm)[0, 1]
    r_per = mm.groupby("geocode").apply(lambda x: np.corrcoef(x.cox_mm, x.power_mm)[0, 1], include_groups=False)
    # anomaly correlation (remove each municipality's monthly climatology) = information on year-to-year timing
    for c in ["cox_mm", "power_mm"]:
        mm[c + "_anom"] = mm[c] - mm.groupby(["geocode", "MONTH"])[c].transform("mean")
    r_anom = mm.groupby("geocode").apply(lambda x: np.corrcoef(x.cox_mm_anom, x.power_mm_anom)[0, 1], include_groups=False)
    print("monthly precip (Cox weeks binned by their MONTH label, so ~edge mismatch): pooled r=", round(r_all, 3),
          "| per-muni r median=", round(r_per.median(), 3), "| per-muni anomaly r median=", round(r_anom.median(), 3),
          "IQR", r_anom.quantile([.25, .75]).round(3).tolist())
    print("mean monthly precip mm: cox", mm.cox_mm.mean().round(1), "power", mm.power_mm.mean().round(1))
    out = pd.DataFrame({"precip_r_monthly": r_per, "precip_r_anomaly": r_anom}).reset_index()
    out.merge(muni[["geocode", "region"]], on="geocode").to_csv(REPORT_DIR / "source_validation_precip.csv", index=False)
    a.to_csv(REPORT_DIR / "source_validation_annual_cases.csv", index=False)
    print(out.merge(muni[["geocode", "region"]], on="geocode").groupby("region")[["precip_r_monthly", "precip_r_anomaly"]].median().round(3))


if __name__ == "__main__":
    main()
