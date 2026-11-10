"""Download weekly Google Trends interest for "dengue" (EXPLORATORY, timing model).

Writes ``<TIMING_TRENDS_DIR>/dengue_<geo>_<start>_<end>.csv`` with columns
``date, dengue`` (Sunday weeks = epi weeks), the format read by
:func:`dengue_recon.extensions.timing.signals.load_trends`.

* Geos: ``BR`` plus ``BR-<UF>`` for all 27 states. Google returns nothing for
  states with too few searches; those are recorded in ``empty_geos.txt`` and the
  model then uses the national series for them (flagged as a fallback).
* Windows: 4-year, overlapping by one year, so Google returns weekly values
  (longer ranges come back monthly). ``load_trends`` uses one window per
  (geo, year), so the 0-100 scalings of different windows are never mixed.
* Polite and resumable: files that already exist are skipped, requests are
  spaced out and a rate-limit answer (HTTP 429) waits and retries.
* The last window ends in January 2024 only so that epi week 52 of 2023 is
  complete; ``load_trends`` drops every row after the last allowed year before
  parsing, so 2024 values never enter the analysis.

Google Trends values are a sample: a later download can differ slightly from an
earlier one. Keep the files you used (for example on Google Drive).

CLI: ``python -m dengue_recon.extensions.timing.trends_download [--sleep 20] [--country MX --out DIR]``.
"""
from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import pandas as pd

from ... import config as C
from . import signals as SG

KEYWORD = "dengue"
#: (file start year, file end year, timeframe)
WINDOWS = [
    (2009, 2013, "2009-12-01 2013-12-31"),
    (2013, 2016, "2013-01-01 2016-12-31"),
    (2016, 2019, "2016-01-01 2019-12-31"),
    (2019, 2022, "2019-01-01 2022-12-31"),
    (2022, 2024, "2022-01-01 2024-01-15"),
]


#: first-level regions of the other study countries, as Google Trends codes them (ISO 3166-2 suffixes,
#: except Mexico City, which Google keeps as the former "DIF")
REGIONS = {
    "MX": ["AGU", "BCN", "BCS", "CAM", "CHH", "CHP", "DIF", "COA", "COL", "DUR", "GRO", "GUA", "HID", "JAL", "MEX",
           "MIC", "MOR", "NAY", "NLE", "OAX", "PUE", "QUE", "ROO", "SIN", "SLP", "SON", "TAB", "TAM", "TLA", "VER",
           "YUC", "ZAC"],
    "CO": ["AMA", "ANT", "ARA", "ATL", "BOL", "BOY", "CAL", "CAQ", "CAS", "CAU", "CES", "CHO", "COR", "CUN", "DC",
           "GUA", "GUV", "HUI", "LAG", "MAG", "MET", "NAR", "NSA", "PUT", "QUI", "RIS", "SAN", "SAP", "SUC", "TOL",
           "VAC", "VAU", "VID"],
}


def study_geos(country: str = "BR") -> list[str]:
    """Country code plus its first-level regions (Brazil: the 27 states of the study municipalities)."""
    if country == "BR":
        return ["BR"] + [f"BR-{u}" for u in sorted(SG.UF_CODE.values())]
    return [country] + [f"{country}-{r}" for r in REGIONS[country]]


def fetch_one(pt, geo, timeframe, max_tries=6, wait=60):
    """Weekly series for one geo and window (DataFrame date, dengue), or None
    when Google has no data for it (too few searches)."""
    for k in range(max_tries):
        try:
            pt.build_payload([KEYWORD], timeframe=timeframe, geo=geo)
            df = pt.interest_over_time()
            if df.empty or KEYWORD not in df:
                return None
            df = df.reset_index()[["date", KEYWORD]]
            step = pd.to_datetime(df["date"]).diff().dt.days.dropna()
            if not (step == 7).all():
                raise RuntimeError(f"{geo} {timeframe}: not weekly data (steps {sorted(step.unique())[:3]})")
            df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
            return df
        except Exception as e:  # noqa: BLE001 - rate limits come as several exception types
            if "not weekly" in str(e):
                raise
            if "code 400" in str(e):     # Google rejects the geo code itself: retrying cannot help
                print(f"  {geo} {timeframe}: rejected by Google (400), recorded as empty", flush=True)
                return None
            w = wait * (2 ** k) + random.uniform(0, 10)
            print(f"  {geo} {timeframe}: {type(e).__name__}: {str(e)[:80]} -> wait {w:.0f}s", flush=True)
            time.sleep(w)
    raise RuntimeError(f"gave up on {geo} {timeframe} after {max_tries} tries")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sleep", type=float, default=20, help="seconds between requests")
    ap.add_argument("--out", default=None, help="folder (default TIMING_TRENDS_DIR or DATA_DIR/raw/google_trends)")
    ap.add_argument("--country", default="BR", choices=["BR", *REGIONS], help="country and its regions")
    a = ap.parse_args(argv)
    from pytrends.request import TrendReq

    out = Path(a.out) if a.out else SG.trends_dir()
    out.mkdir(parents=True, exist_ok=True)
    geos = study_geos(a.country)
    empty_f = out / "empty_geos.txt"
    empty = set(empty_f.read_text().split()) if empty_f.exists() else set()
    name = lambda g, w: f"dengue_{g}_{w[0]}_{w[1]}"  # noqa: E731
    todo = [(g, w) for g in geos for w in WINDOWS
            if not (out / f"{name(g, w)}.csv").exists() and name(g, w) not in empty]
    print(f"{len(geos)} geos x {len(WINDOWS)} windows; {len(todo)} files to download into {out}", flush=True)
    pt = TrendReq(hl="pt-BR", tz=180)
    for i, (g, (y0, y1, tf)) in enumerate(todo, 1):
        df = fetch_one(pt, g, tf)
        if df is None:
            empty.add(name(g, (y0, y1)))
            empty_f.write_text("\n".join(sorted(empty)) + "\n")
            print(f"[{i}/{len(todo)}] {name(g, (y0, y1))}: no data from Google (recorded)", flush=True)
            time.sleep(a.sleep)
            continue
        f = out / f"{name(g, (y0, y1))}.csv"
        df.to_csv(f.with_suffix(".tmp"), index=False)
        f.with_suffix(".tmp").rename(f)
        print(f"[{i}/{len(todo)}] {f.name}: {len(df)} weeks", flush=True)
        time.sleep(a.sleep + random.uniform(0, 5))
    n = len(list(out.glob("dengue_*.csv")))
    print(f"done: {n} files with data, {len(empty)} empty, of {len(geos) * len(WINDOWS)}", flush=True)
    return n + len(empty) == len(geos) * len(WINDOWS)


if __name__ == "__main__":
    main()
