"""Build the Mexico data folder (EXPLORATORY extension), same layout as the Brazil DATA_DIR.

Run with ``DENGUE_COUNTRY=MX`` and ``DATA_DIR`` pointing at the (new) Mexico folder::

    DENGUE_COUNTRY=MX DATA_DIR=C:/dr_data_mx python scripts/build_mexico_data.py \
        --opendengue C:/dr_data/raw/opendengue/opendengue_mexico_V1_3_1.parquet \
        --dge C:/dr_data/raw/mx_dge/x --oni-from C:/dr_data

Units: the 32 states (geocode = INEGI state code 1..32, ``uf`` = Google Trends /
ISO 3166-2 suffix). Cases: all notified dengue cases (every ``ESTATUS_CASO``), by
assigned state and epi week of symptom onset, 2017-2023:

* OpenDengue V1.3.1 Admin1 weekly (source WHO/PAHO, "Total") for 2017-2022; a state
  absent from a reported week has 0 cases (OpenDengue omits zero rows).
* Whole weeks OpenDengue does not report (2020 W53, 2021 W50-52, 2022 W33-51) and all
  of 2023 come from the DGE open data (year-end snapshots). On 2020-2022 both
  sources agree (totals within 3 %, weekly r >= 0.98, 83-99 % of state-weeks identical).

Climate: NASA POWER daily at the state capital, 2009-2023 (same request and weekly
aggregation as Brazil's municipal seats). ONI: copied from the Brazil data folder.
"""
from __future__ import annotations

import argparse
import glob
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from dengue_recon import config as C

assert C.COUNTRY == "MX", "run with DENGUE_COUNTRY=MX"

# INEGI code: (name as in OpenDengue, Trends/ISO suffix, region, capital, lat, lon)
STATES = {
    1: ("AGUASCALIENTES", "AGU", "West", "Aguascalientes", 21.8853, -102.2916),
    2: ("BAJA CALIFORNIA", "BCN", "Northwest", "Mexicali", 32.6245, -115.4523),
    3: ("BAJA CALIFORNIA SUR", "BCS", "Northwest", "La Paz", 24.1426, -110.3128),
    4: ("CAMPECHE", "CAM", "South-Southeast", "Campeche", 19.8301, -90.5349),
    5: ("COAHUILA", "COA", "Northeast", "Saltillo", 25.4232, -101.0053),
    6: ("COLIMA", "COL", "West", "Colima", 19.2433, -103.7250),
    7: ("CHIAPAS", "CHP", "South-Southeast", "Tuxtla Gutierrez", 16.7528, -93.1152),
    8: ("CHIHUAHUA", "CHH", "Northwest", "Chihuahua", 28.6320, -106.0691),
    9: ("DISTRITO FEDERAL", "DIF", "Center", "Ciudad de Mexico", 19.4326, -99.1332),
    10: ("DURANGO", "DUR", "Northwest", "Durango", 24.0277, -104.6532),
    11: ("GUANAJUATO", "GUA", "West", "Guanajuato", 21.0190, -101.2574),
    12: ("GUERRERO", "GRO", "South-Southeast", "Chilpancingo", 17.5506, -99.5058),
    13: ("HIDALGO", "HID", "Center", "Pachuca", 20.1011, -98.7591),
    14: ("JALISCO", "JAL", "West", "Guadalajara", 20.6597, -103.3496),
    15: ("MEXICO", "MEX", "Center", "Toluca", 19.2826, -99.6557),
    16: ("MICHOACAN", "MIC", "West", "Morelia", 19.7060, -101.1950),
    17: ("MORELOS", "MOR", "Center", "Cuernavaca", 18.9242, -99.2216),
    18: ("NAYARIT", "NAY", "West", "Tepic", 21.5042, -104.8946),
    19: ("NUEVO LEON", "NLE", "Northeast", "Monterrey", 25.6866, -100.3161),
    20: ("OAXACA", "OAX", "South-Southeast", "Oaxaca", 17.0732, -96.7266),
    21: ("PUEBLA", "PUE", "Center", "Puebla", 19.0414, -98.2063),
    22: ("QUERETARO", "QUE", "Center", "Queretaro", 20.5888, -100.3899),
    23: ("QUINTANA ROO", "ROO", "South-Southeast", "Chetumal", 18.5001, -88.2961),
    24: ("SAN LUIS POTOSI", "SLP", "Northeast", "San Luis Potosi", 22.1565, -100.9855),
    25: ("SINALOA", "SIN", "Northwest", "Culiacan", 24.8091, -107.3940),
    26: ("SONORA", "SON", "Northwest", "Hermosillo", 29.0729, -110.9559),
    27: ("TABASCO", "TAB", "South-Southeast", "Villahermosa", 17.9892, -92.9475),
    28: ("TAMAULIPAS", "TAM", "Northeast", "Ciudad Victoria", 23.7369, -99.1411),
    29: ("TLAXCALA", "TLA", "Center", "Tlaxcala", 19.3139, -98.2404),
    30: ("VERACRUZ", "VER", "South-Southeast", "Xalapa", 19.5438, -96.9102),
    31: ("YUCATAN", "YUC", "South-Southeast", "Merida", 20.9674, -89.5926),
    32: ("ZACATECAS", "ZAC", "West", "Zacatecas", 22.7709, -102.5832),
}
OD_YEARS = list(range(2017, 2023))  # OpenDengue weeks used (2023 has none)
YEARS = list(range(2017, 2024))     # extended to 2024 with --last-year 2024 (final test only)


def meta() -> pd.DataFrame:
    m = pd.DataFrame([dict(geocode=k, name=v[0].title(), uf=v[1], region=v[2], capital_city=v[3], lat=v[4], lon=v[5],
                           capital=1, pop2021=np.nan, uf_code=k) for k, v in STATES.items()])
    assert set(m.region) == set(C.REGIONS)
    return m


def opendengue_cases(path, cal) -> tuple[pd.DataFrame, set]:
    od = pd.read_parquet(path)
    od = od[(od.S_res == "Admin1") & (od.T_res == "Week")].copy()
    od["week_start"] = pd.to_datetime(od.calendar_start_date)
    od = od.merge(cal[["week_start", "epi_year", "epi_week"]], on="week_start", how="inner")
    od = od[od.epi_year.isin(OD_YEARS)]
    assert (od.case_definition_standardised == "Total").all()
    code = {v[0]: k for k, v in STATES.items()}
    assert set(od.adm_1_name) <= set(code), set(od.adm_1_name) - set(code)
    od["geocode"] = od.adm_1_name.map(code)
    reported = set(zip(od.epi_year, od.epi_week))
    c = od.groupby(["geocode", "epi_year", "epi_week"], as_index=False).dengue_total.sum()
    return c.rename(columns={"dengue_total": "cases"}), reported


def dge_cases(folder, cal) -> pd.DataFrame:
    rows = []
    for f in sorted(glob.glob(str(Path(folder) / "*" / "*.csv"))):
        d = pd.read_csv(f, dtype=str, encoding="latin-1", usecols=["ENTIDAD_ASIG", "FECHA_SIGN_SINTOMAS"])
        # 2020-2023 snapshots write ISO dates, the 2024 snapshot writes dd/mm/yyyy: parse both strictly
        iso = pd.to_datetime(d.FECHA_SIGN_SINTOMAS, format="%Y-%m-%d", errors="coerce")
        d["onset"] = iso.fillna(pd.to_datetime(d.FECHA_SIGN_SINTOMAS, format="%d/%m/%Y", errors="coerce"))
        assert d.onset.notna().mean() > 0.999, f"{f}: unparsed onset dates"
        d["week_start"] = d.onset - pd.to_timedelta((d.onset.dt.day_of_week + 1) % 7, unit="D")  # Sunday weeks
        d["geocode"] = pd.to_numeric(d.ENTIDAD_ASIG, errors="coerce")
        d = d[d.geocode.between(1, 32)]
        rows.append(d.groupby(["geocode", "week_start"]).size().rename("cases").reset_index().assign(src=Path(f).parent.name))
    x = pd.concat(rows)
    x = x.merge(cal[["week_start", "epi_year", "epi_week"]], on="week_start", how="inner")
    # one snapshot per onset year: the latest one that contains it
    x["snap"] = pd.to_datetime(x.src.str[-6:], format="%d%m%y")
    best = x.groupby("epi_year").snap.max().rename("best")
    x = x.merge(best, on="epi_year")
    x = x[x.snap == x.best]
    return x.groupby(["geocode", "epi_year", "epi_week"], as_index=False).cases.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--opendengue", required=True)
    ap.add_argument("--dge", required=True, help="folder with one unzipped DGE snapshot per subfolder")
    ap.add_argument("--oni-from", required=True, help="Brazil DATA_DIR holding oni_weekly.parquet / oni_monthly.csv")
    ap.add_argument("--skip-power", action="store_true")
    ap.add_argument("--last-year", type=int, default=2023,
                    help="2024 only for the final test (docs/FINAL_TEST_2024_MX.md); needs DENGUE_RUN_FINAL_2024=1")
    a = ap.parse_args()
    global YEARS
    assert a.last_year in (2023, 2024)
    assert a.last_year <= C.max_loaded_year(), "2024 is locked: set DENGUE_RUN_FINAL_2024=1"
    YEARS = list(range(2017, a.last_year + 1))
    from dengue_recon.data import build_dataset as BDS, download as DL
    from dengue_recon.data.epiweeks import calendar

    out = C.DATA_DIR
    (out / "raw" / "power").mkdir(parents=True, exist_ok=True)
    m = meta()
    cal = calendar(min(YEARS), max(YEARS)).assign(week_start=lambda c: pd.to_datetime(c.week_start))
    clim_cal = calendar(2009, max(YEARS))

    # ---- cases
    od, reported = opendengue_cases(a.opendengue, cal)
    dge = dge_cases(a.dge, cal)
    grid = cal.merge(m[["geocode"]], how="cross")
    key = list(zip(grid.epi_year, grid.epi_week))
    grid["source"] = ["opendengue" if k in reported else "dge" for k in key]
    g = grid.merge(od, on=["geocode", "epi_year", "epi_week"], how="left").rename(columns={"cases": "od"})
    g = g.merge(dge, on=["geocode", "epi_year", "epi_week"], how="left").rename(columns={"cases": "dge"})
    dge_years = set(dge.epi_year)
    g["cases"] = np.where(g.source == "opendengue", g.od.fillna(0), g.dge.fillna(0))
    bad = g[(g.source == "dge") & ~g.epi_year.isin(dge_years)]
    assert bad.empty, f"weeks with neither source: {sorted(set(zip(bad.epi_year, bad.epi_week)))[:10]}"
    g["cases"] = g.cases.astype(int)
    print("weeks filled from DGE:", g[g.source == "dge"].drop_duplicates(["epi_year", "epi_week"])
          .groupby("epi_year").size().to_dict())

    ys = g.groupby(["geocode", "epi_year"]).cases.sum().rename("year_total").reset_index()
    g = g.merge(ys, on=["geocode", "epi_year"])
    g = g.merge(m[["geocode", "name", "uf", "region", "lat", "lon", "capital", "pop2021"]], on="geocode")
    g["exclude_suggested"] = False
    g["year_complete"] = True
    g["provisional"] = False
    g["data_flag"] = np.where(g.epi_year == 2020, "covid_2020_reporting_disruption", "")
    g["share"] = np.where(g.year_total > 0, g.cases / g.year_total.where(g.year_total > 0), np.nan)
    g["week_start"] = g.week_start.astype("datetime64[ms]")

    # ---- climate (NASA POWER at the state capital)
    if not a.skip_power:
        DL.download_power(m[["geocode", "lat", "lon"]])
    clim = pd.concat([BDS.load_power(int(c)) for c in m.geocode], ignore_index=True)
    clim = clim_cal.merge(clim, on=["epi_year", "epi_week"], how="right")
    clim = clim[clim.epi_year <= max(YEARS)].sort_values(["geocode", "week_index"]).reset_index(drop=True)
    clim = clim.astype({"n_weeks_year": int, "week_index": int})
    assert clim.groupby("geocode").size().nunique() == 1, "uneven climate coverage"
    assert not clim[["precip_mm", "temp_mean", "humid_mean"]].isna().any().any()
    clim = clim[BDS.CLIM_ORDER]

    g = g.merge(clim.drop(columns=["week_start", "n_weeks_year", "week_index"]), on=["geocode", "epi_year", "epi_week"],
                how="left")
    cols = ["geocode", "name", "uf", "region", "epi_year", "epi_week", "week_start", "n_weeks_year", "week_index",
            "cases", "source", "year_total", "share", "year_complete", "provisional", "data_flag", "exclude_suggested",
            "temp_mean", "temp_min", "temp_max", "humid_mean", "precip_mm", "precip_days", "pop2021", "lat", "lon",
            "capital"]
    g = g[cols].sort_values(["geocode", "epi_year", "epi_week"]).reset_index(drop=True)

    tot = ys[ys.epi_year <= 2023].groupby("geocode").year_total.sum().rename("total_cases_2017_2023")  # never 2024
    m = m.merge(tot, on="geocode")
    g.to_parquet(out / "dengue_weekly.parquet", index=False)
    clim.to_parquet(out / "climate_weekly.parquet", index=False)
    ys.to_csv(out / "year_summary.csv", index=False)
    m.to_csv(out / "municipalities.csv", index=False)
    for f in ("oni_weekly.parquet", "oni_monthly.csv"):
        shutil.copy2(Path(a.oni_from) / f, out / f)
    print(f"rows={len(g)} states={g.geocode.nunique()} years={g.epi_year.min()}-{g.epi_year.max()}")
    print(ys.pivot(index="geocode", columns="epi_year", values="year_total").assign(
        name=m.set_index("geocode").name).to_string())


if __name__ == "__main__":
    main()
