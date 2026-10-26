"""Download raw data for the weekly dengue reconstruction study.

Sources (all public, no API key):
  * InfoDengue  - weekly dengue notifications per municipality (+ station-based
                  temperature / humidity).  https://info.dengue.mat.br/api/alertcity
  * NASA POWER  - daily gridded climate (precipitation, temperature, humidity)
                  at the municipal seat.  https://power.larc.nasa.gov/api/temporal/daily/point
  * IBGE        - municipality list, 2021 population estimates (SIDRA table 6579).
  * kelvins/municipios-brasileiros (GitHub, MIT) - lat/lon of municipal seats.

Usage (local or Colab; DATA_DIR defaults to <repo>/data):
    python -m dengue_recon.data.download meta
    python -m dengue_recon.data.download infodengue --top-pop 700
    python -m dengue_recon.data.download select --n 300
    python -m dengue_recon.data.download power
    python -m dengue_recon.data.download all          # everything above
    python -m dengue_recon.data.download cox          # optional Cox et al. cross-check
or ``python scripts/run_pipeline.py --step data``.

Each step caches files in DATA_DIR/raw/ and skips anything already downloaded,
so the script can be interrupted and resumed.  To scale up, raise --top-pop and
--n (InfoDengue: ~2 s/municipality; POWER: ~2 s/municipality).
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time

import pandas as pd
import requests

from .. import config as C

DATA_DIR = C.DATA_DIR
RAW = C.RAW_DIR

EY_START, EY_END = 2010, 2025          # epidemiological years requested from InfoDengue
CLIM_START, CLIM_END = "20090101", "20251231"  # climate starts 1 yr early for lags

INFODENGUE_URL = ("https://info.dengue.mat.br/api/alertcity?geocode={geo}&disease=dengue"
                  "&format=csv&ew_start=1&ew_end=53&ey_start={y0}&ey_end={y1}")
POWER_URL = ("https://power.larc.nasa.gov/api/temporal/daily/point?parameters="
             "PRECTOTCORR,T2M,T2M_MAX,T2M_MIN,RH2M&community=AG&longitude={lon}"
             "&latitude={lat}&start={s}&end={e}&format=CSV")
COORDS_URL = "https://raw.githubusercontent.com/kelvins/municipios-brasileiros/main/csv/municipios.csv"
IBGE_MUNI_URL = "https://servicodados.ibge.gov.br/api/v1/localidades/municipios"
SIDRA_POP_URL = "https://apisidra.ibge.gov.br/values/t/6579/n6/all/v/all/p/2021"

UF_REGION = {  # IBGE state code (first two digits of geocode) -> (UF, macro-region)
    11: ("RO", "North"), 12: ("AC", "North"), 13: ("AM", "North"), 14: ("RR", "North"),
    15: ("PA", "North"), 16: ("AP", "North"), 17: ("TO", "North"),
    21: ("MA", "Northeast"), 22: ("PI", "Northeast"), 23: ("CE", "Northeast"),
    24: ("RN", "Northeast"), 25: ("PB", "Northeast"), 26: ("PE", "Northeast"),
    27: ("AL", "Northeast"), 28: ("SE", "Northeast"), 29: ("BA", "Northeast"),
    31: ("MG", "Southeast"), 32: ("ES", "Southeast"), 33: ("RJ", "Southeast"),
    35: ("SP", "Southeast"),
    41: ("PR", "South"), 42: ("SC", "South"), 43: ("RS", "South"),
    50: ("MS", "Center-West"), 51: ("MT", "Center-West"), 52: ("GO", "Center-West"),
    53: ("DF", "Center-West"),
}

SESSION = requests.Session()
SESSION.headers["User-Agent"] = "dengue-weekly-reconstruction-research/0.1 (academic)"


def get(url: str, tries: int = 6, timeout: int = 120, pause: float = 1.0) -> requests.Response:
    """GET with polite pause and exponential backoff (handles 429/5xx)."""
    wait = 5.0
    for k in range(tries):
        try:
            r = SESSION.get(url, timeout=timeout)
            if r.status_code == 200:
                time.sleep(pause)
                return r
            if r.status_code not in (429, 500, 502, 503, 504):
                r.raise_for_status()
            print(f"  HTTP {r.status_code}; retry in {wait:.0f}s", file=sys.stderr)
        except requests.RequestException as exc:  # network hiccup
            print(f"  {exc!r}; retry in {wait:.0f}s", file=sys.stderr)
        time.sleep(wait)
        wait = min(wait * 2, 300)
    raise RuntimeError(f"failed after {tries} tries: {url}")


# ----------------------------------------------------------------------------- meta
def download_meta() -> pd.DataFrame:
    """Municipality table: geocode, name, uf, region, lat, lon, capital, pop2021."""
    (RAW / "meta").mkdir(parents=True, exist_ok=True)
    out = RAW / "meta" / "municipalities.csv"
    if out.exists():
        return pd.read_csv(out)
    coords = pd.read_csv(io.StringIO(get(COORDS_URL).text))
    coords.to_csv(RAW / "meta" / "kelvins_municipios.csv", index=False)
    pop_json = get(SIDRA_POP_URL, timeout=300).json()
    (RAW / "meta" / "sidra_6579_2021.json").write_text(json.dumps(pop_json))
    pop = pd.DataFrame(pop_json[1:])[["D1C", "V"]]
    pop.columns = ["geocode", "pop2021"]
    pop["geocode"] = pop["geocode"].astype(int)
    pop["pop2021"] = pd.to_numeric(pop["pop2021"], errors="coerce")
    m = coords.rename(columns={"codigo_ibge": "geocode", "nome": "name",
                               "latitude": "lat", "longitude": "lon"})
    m = m[["geocode", "name", "lat", "lon", "capital"]].merge(pop, on="geocode", how="left")
    m["uf_code"] = m["geocode"] // 100000
    m["uf"] = m["uf_code"].map(lambda c: UF_REGION[c][0])
    m["region"] = m["uf_code"].map(lambda c: UF_REGION[c][1])
    m.to_csv(out, index=False)
    return m


# ----------------------------------------------------------------------- infodengue
def download_infodengue(geocodes, y0=EY_START, y1=EY_END) -> None:
    d = RAW / "infodengue"
    d.mkdir(parents=True, exist_ok=True)
    for i, g in enumerate(geocodes):
        f = d / f"{int(g)}.csv"
        if f.exists() and f.stat().st_size > 0:
            continue
        r = get(INFODENGUE_URL.format(geo=int(g), y0=y0, y1=y1))
        f.write_bytes(r.content)
        if i % 25 == 0:
            print(f"infodengue {i+1}/{len(geocodes)} {g}", flush=True)


def select_municipalities(meta: pd.DataFrame, n: int = 300, min_per_region: int = 45) -> pd.DataFrame:
    """Rank downloaded candidates by total notified cases; guarantee regional spread.

    Each macro-region gets at least `min_per_region` municipalities (or all its
    candidates if fewer); the remaining slots go to the highest-case municipalities.
    """
    rows = []
    for f in sorted((RAW / "infodengue").glob("*.csv")):
        try:
            df = pd.read_csv(f, usecols=["SE", "casos"])
        except Exception:
            continue
        rows.append({"geocode": int(f.stem), "total_cases": df["casos"].sum(),
                     "n_weeks": len(df)})
    tot = pd.DataFrame(rows).merge(meta, on="geocode", how="left")
    tot = tot[tot["n_weeks"] > 0].sort_values("total_cases", ascending=False)
    chosen = []
    for reg, sub in tot.groupby("region"):
        chosen += sub.head(min_per_region)["geocode"].tolist()
    for g in tot["geocode"]:
        if len(set(chosen)) >= n:
            break
        if g not in chosen:
            chosen.append(g)
    sel = tot[tot["geocode"].isin(chosen)].copy()
    sel.to_csv(RAW / "meta" / "selected_municipalities.csv", index=False)
    tot.to_csv(RAW / "meta" / "candidate_case_totals.csv", index=False)
    return sel


# ---------------------------------------------------------------------- NASA POWER
def download_power(sel: pd.DataFrame, workers: int = 3) -> None:
    """One request per municipality (daily 2009-2025).  A few parallel workers
    keep it polite but ~3x faster than sequential."""
    from concurrent.futures import ThreadPoolExecutor

    d = RAW / "power"
    d.mkdir(parents=True, exist_ok=True)

    def one(row):
        f = d / f"{int(row.geocode)}.csv"
        if f.exists() and f.stat().st_size > 1000:
            return
        r = get(POWER_URL.format(lat=round(row.lat, 4), lon=round(row.lon, 4),
                                 s=CLIM_START, e=CLIM_END), timeout=300)
        tmp = f.with_suffix(".part")
        tmp.write_bytes(r.content)
        tmp.replace(f)

    rows = list(sel.itertuples())
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, _ in enumerate(ex.map(one, rows)):
            if i % 25 == 0:
                print(f"power {i+1}/{len(rows)}", flush=True)


# ------------------------------------------------- Cox et al. (optional cross-check)
COX_URL = "https://zenodo.org/api/records/13286671/files/denv_report.RDS/content"
COX_KEEP = ["MUNICIP", "GEOM", "STATE", "WEEK", "WEEK_NO", "YEAR", "MONTH", "CASES", "pop",
            "Rf_cum", "Rf_mean", "Tmin", "Tmean", "Tmax", "RH_min", "RH_mean", "RH_max",
            "raindays", "spei1", "spei3"]


def download_cox(keep_rds: bool = False) -> None:
    """Cox et al. 2024 dengue file (1.1 GB .RDS, all 5,570 municipalities, 2013-2020,
    ERA5-Land climate).  Streams to disk, extracts a compact parquet subset
    (~140 MB) and deletes the .RDS unless keep_rds.  Needs `pyreadr` and ~8 GB RAM."""
    import pyreadr

    d = RAW / "cox_zenodo"
    d.mkdir(parents=True, exist_ok=True)
    out = d / "cox_denv_report_subset.parquet"
    if out.exists():
        return
    rds = d / "denv_report.RDS"
    if not rds.exists():
        with SESSION.get(COX_URL, stream=True, timeout=600) as r:
            r.raise_for_status()
            with open(rds, "wb") as fh:
                for chunk in r.iter_content(1 << 20):
                    fh.write(chunk)
    v = pyreadr.read_r(str(rds))[None][COX_KEEP].copy()
    for c in ("CASES", "pop"):
        v[c] = pd.to_numeric(v[c], errors="coerce")
    v.to_parquet(out, index=False)
    if not keep_rds:
        rds.unlink()


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("step", choices=["meta", "infodengue", "select", "power", "cox", "all"])
    p.add_argument("--top-pop", type=int, default=700,
                   help="candidate pool: N most populous municipalities")
    p.add_argument("--n", type=int, default=300, help="final number of municipalities")
    p.add_argument("--min-per-region", type=int, default=45)
    a = p.parse_args(argv)
    meta = download_meta()
    if a.step in ("infodengue", "all"):
        cand = meta.sort_values("pop2021", ascending=False).head(a.top_pop)
        # also make sure every state capital is a candidate
        cand = pd.concat([cand, meta[meta["capital"] == 1]]).drop_duplicates("geocode")
        download_infodengue(cand["geocode"].tolist())
    if a.step in ("select", "all"):
        sel = select_municipalities(meta, a.n, a.min_per_region)
        print(sel.groupby("region").size())
    if a.step in ("power", "all"):
        sel = pd.read_csv(RAW / "meta" / "selected_municipalities.csv")
        download_power(sel)
    if a.step == "cox":  # optional, not part of "all"
        download_cox()


if __name__ == "__main__":
    main()
