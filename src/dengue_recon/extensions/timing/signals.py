"""Same-year auxiliary weekly signals for the timing-informed share model (EXPLORATORY).

Two signals are loaded onto the panel grid (municipality m x epi year t x slot s):

* **Regional (state) curve.** Weekly notified cases of the *other* municipalities of
  the target's state, summed. The source is the InfoDengue download of about 700
  municipalities (``DATA_DIR/raw/infodengue/<geocode>.csv``), which contains the 300
  study municipalities. For every study municipality the pool is built
  explicitly as the list of state members *without the target*; the target's own
  counts are never added (no subtract-after-adding). The other study
  municipalities of the state do contribute: in the mixed-resolution scenario their
  weekly data are assumed observed while the target has only its annual total.
  Variants: ``"all"`` (every other InfoDengue municipality of the state),
  ``"nonstudy"`` (only municipalities outside the 300-municipality study set, a
  stricter sensitivity) and ``"p25"`` / ``"p50"`` (a fixed random fraction of the
  other municipalities, see :func:`fraction_pools`).
* **Search signal.** Weekly Google Trends interest for the topic word ``dengue``
  in the target's state (18 states are available), else the national (BR) series;
  the fallback is flagged.

The 2024 lock is respected at the line level: rows after
``config.max_loaded_year()`` are dropped from the raw text *before* parsing (the
InfoDengue CSVs are sorted newest first and also contain 2024-2025), so locked
years never enter a DataFrame.
"""
from __future__ import annotations

import glob
import io
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from ... import config as C
from ...data.epiweeks import calendar

#: IBGE state code (first two digits of the municipality geocode) -> state abbreviation
UF_CODE = {11: "RO", 12: "AC", 13: "AM", 14: "RR", 15: "PA", 16: "AP", 17: "TO", 21: "MA", 22: "PI",
           23: "CE", 24: "RN", 25: "PB", 26: "PE", 27: "AL", 28: "SE", 29: "BA", 31: "MG", 32: "ES",
           33: "RJ", 35: "SP", 41: "PR", 42: "SC", 43: "RS", 50: "MS", 51: "MT", 52: "GO", 53: "DF"}

#: seed of the random pool fractions (mixed-resolution scenario), fixed a priori
POOL_SEED = 20261028
POOL_FRACTIONS = {"p25": 0.25, "p50": 0.50}


def infodengue_dir() -> Path:
    return Path(os.environ.get("TIMING_INFODENGUE_DIR", C.RAW_DIR / "infodengue"))


def trends_dir() -> Path:
    return Path(os.environ.get("TIMING_TRENDS_DIR", C.RAW_DIR / "google_trends"))


def _filtered_lines(path: Path, keep) -> str:
    """Header plus the lines for which ``keep(line)`` is True (before any parsing)."""
    with open(path, encoding="utf-8") as fh:
        header = fh.readline()
        body = [ln for ln in fh if ln.strip() and keep(ln)]
    return header + "".join(body)


def read_infodengue_csv(path: Path, max_year: int | None = None) -> pd.DataFrame:
    """One raw InfoDengue municipality file, epi weeks <= ``max_year`` only.

    Lines are filtered on the ``SE`` field (YYYYWW, second column) before parsing.
    Duplicated weeks (several model versions) keep the latest ``versao_modelo``.
    """
    max_year = C.max_loaded_year() if max_year is None else max_year
    lim = max_year * 100 + 53

    def keep(ln):
        se = ln.split(",", 2)[1]
        return se.isdigit() and int(se) <= lim
    txt = _filtered_lines(path, keep)
    d = pd.read_csv(io.StringIO(txt), usecols=["SE", "casos", "versao_modelo"])
    d = d.sort_values("versao_modelo").drop_duplicates("SE", keep="last")
    d["epi_year"], d["epi_week"] = d.SE // 100, d.SE % 100
    assert len(d) == 0 or d.epi_year.max() <= max_year
    return d


def load_infodengue_counts(years=None, cache: Path | None = None) -> pd.DataFrame:
    """Weekly cases of all downloaded InfoDengue municipalities, long format
    ``geocode, uf, epi_year, epi_week, cases`` for ``years`` (default 2010 .. max loaded year).

    A parquet cache (outside the repository, ``DATA_DIR/extensions/``) is written
    on first use and re-read with a year filter.
    """
    mx = C.max_loaded_year()
    years = list(range(2010, mx + 1)) if years is None else [int(y) for y in years]
    C.assert_year_allowed(years)
    cache = Path(cache) if cache else C.DATA_DIR / "extensions" / f"timing_infodengue_counts_2010_{mx}.parquet"
    if cache.exists():
        d = pd.read_parquet(cache, filters=[("epi_year", "<=", mx)])
    else:
        rows = []
        for f in sorted(glob.glob(str(infodengue_dir() / "*.csv"))):
            g = int(Path(f).stem)
            x = read_infodengue_csv(Path(f), mx)
            rows.append(pd.DataFrame(dict(geocode=g, epi_year=x.epi_year.values, epi_week=x.epi_week.values,
                                          cases=x.casos.values)))
        d = pd.concat(rows, ignore_index=True)
        d["uf"] = (d.geocode // 100000).map(UF_CODE)
        cache.parent.mkdir(parents=True, exist_ok=True)
        d.to_parquet(cache, index=False)
    d = d[d.epi_year.isin(years)]
    assert d.epi_year.max() <= mx
    return d.reset_index(drop=True)


def count_cube(d: pd.DataFrame, years) -> tuple[np.ndarray, np.ndarray, list]:
    """(G, T, 53) weekly counts of every InfoDengue municipality on the epi-week grid
    of ``years``; returns (cube, geocodes, uf per geocode)."""
    geos = np.array(sorted(d.geocode.unique()))
    gi = {g: i for i, g in enumerate(geos)}
    ti = {int(y): i for i, y in enumerate(years)}
    d = d[d.epi_year.isin(list(ti))]
    A = np.zeros((len(geos), len(years), 53))
    A[d.geocode.map(gi).values, d.epi_year.map(ti).values, d.epi_week.values - 1] = d.cases.values
    ufs = [UF_CODE[int(g) // 100000] for g in geos]
    return A, geos, ufs


def pools(study_geos, all_geos, all_ufs, variant="all", study_set=None):
    """{study geocode: array of pool geocodes} for one pool variant.

    ``"all"``: every other municipality of the state; ``"nonstudy"``: only
    municipalities outside ``study_set``; ``"p25"``/``"p50"``: see :func:`fraction_pools`.
    The target is never a member of its own pool.
    """
    if variant in POOL_FRACTIONS:
        return fraction_pools(study_geos, all_geos, all_ufs, POOL_FRACTIONS[variant])
    uf_of = dict(zip(all_geos, all_ufs))
    by_uf = {}
    for g, u in zip(all_geos, all_ufs):
        by_uf.setdefault(u, []).append(g)
    study_set = set(study_set if study_set is not None else study_geos)
    out = {}
    for g in study_geos:
        mem = [x for x in by_uf[uf_of[g]] if x != g]
        if variant == "nonstudy":
            mem = [x for x in mem if x not in study_set]
        elif variant != "all":
            raise ValueError(variant)
        out[int(g)] = np.array(mem, dtype=int)
    return out


def fraction_pools(study_geos, all_geos, all_ufs, p, seed=POOL_SEED):
    """Mixed-resolution scenario: only a random fraction ``p`` of the state's other
    municipalities report weekly.

    For each state one random permutation of its members is drawn (generator
    ``seed`` + state code). A target's pool is the first ``k = max(1, round(p * (n - 1)))``
    members of that permutation after removing the target (n = state members), so
    every target of a state sees (almost) the same reporting set.
    """
    uf_of = dict(zip(all_geos, all_ufs))
    by_uf = {}
    for g, u in zip(all_geos, all_ufs):
        by_uf.setdefault(u, []).append(int(g))
    perm = {}
    for u, mem in by_uf.items():
        code = [k for k, v in UF_CODE.items() if v == u][0]
        rng = np.random.default_rng(seed + code)
        perm[u] = [mem[i] for i in rng.permutation(len(mem))]
    out = {}
    for g in study_geos:
        u = uf_of[int(g)]
        others = [x for x in perm[u] if x != int(g)]
        k = max(1, int(round(p * len(others)))) if others else 0
        out[int(g)] = np.array(others[:k], dtype=int)
    return out


def pool_counts(A, all_geos, pool_map, study_geos):
    """(M, T, 53) pooled weekly counts per study municipality and the pool sizes (M,)."""
    gi = {int(g): i for i, g in enumerate(all_geos)}
    out = np.zeros((len(study_geos),) + A.shape[1:])
    size = np.zeros(len(study_geos), int)
    for m, g in enumerate(study_geos):
        idx = [gi[int(x)] for x in pool_map[int(g)]]
        assert gi[int(g)] not in idx, "target municipality inside its own pool"
        size[m] = len(idx)
        if idx:
            out[m] = A[idx].sum(0)
    return out, size


# ------------------------------------------------------------------ Google Trends
def load_trends(years=None) -> pd.DataFrame:
    """Weekly Google Trends 'dengue' per geo (``BR`` and ``BR-<UF>``), long format
    ``geo, win, epi_year, epi_week, value``, epi years <= max loaded year only.

    Raw files ``dengue_<geo>_<start>_<end>.csv`` (Sunday weeks = epi weeks) are
    filtered on the date string before parsing. Each 5-year window has its own
    0-100 scaling, so for every (geo, epi year) a single window is used: the one
    with the most weeks of that year (ties: the earlier window). Within-year
    shares are therefore never mixed across windows.
    """
    mx = C.max_loaded_year()
    years = list(range(2010, mx + 1)) if years is None else [int(y) for y in years]
    cal = calendar(2009, mx)
    last = str(cal.week_start.max())[:10]
    cal = cal[cal.epi_year.isin(years)]
    cal = cal.assign(week_start=pd.to_datetime(cal.week_start).dt.strftime("%Y-%m-%d"))
    rows = []
    for f in sorted(glob.glob(str(trends_dir() / "dengue_*.csv"))):
        mt = re.search(r"dengue_(BR(?:-[A-Z]{2})?)_(\d{4})_(\d{4})\.csv$", f)
        if not mt:
            continue
        txt = _filtered_lines(Path(f), lambda ln: ln[:10] <= last)
        x = pd.read_csv(io.StringIO(txt))
        x = x.rename(columns={"date": "week_start", "dengue": "value"})
        x["geo"], x["win"] = mt.group(1), int(mt.group(2))
        rows.append(x[["geo", "win", "week_start", "value"]])
    tr = pd.concat(rows, ignore_index=True).merge(cal[["week_start", "epi_year", "epi_week"]], on="week_start")
    nwk = tr.groupby(["geo", "epi_year", "win"]).size().rename("nwk").reset_index()
    nwk = nwk.sort_values(["geo", "epi_year", "nwk", "win"], ascending=[True, True, False, True])
    best = nwk.drop_duplicates(["geo", "epi_year"])[["geo", "epi_year", "win"]]
    tr = tr.merge(best, on=["geo", "epi_year", "win"])
    assert tr.epi_year.max() <= mx
    return tr.sort_values(["geo", "epi_year", "epi_week"]).reset_index(drop=True)


def trends_cube(tr: pd.DataFrame, study_ufs, years):
    """(M, T, 53) Trends values per study municipality (its state's series, else BR)
    and a (M,) bool flag 'national fallback used'."""
    ti = {int(y): i for i, y in enumerate(years)}
    geos = sorted(tr.geo.unique())
    cube = {}
    for g in geos:
        z = tr[(tr.geo == g) & tr.epi_year.isin(list(ti))]
        a = np.zeros((len(years), 53))
        a[z.epi_year.map(ti).values, z.epi_week.values - 1] = z.value.values
        cube[g] = a
    out = np.zeros((len(study_ufs), len(years), 53))
    fb = np.zeros(len(study_ufs), bool)
    for m, u in enumerate(study_ufs):
        g = f"BR-{u}"
        if g not in cube:
            g, fb[m] = "BR", True
        out[m] = cube[g]
    return out, fb


# ------------------------------------------------------------------ everything for one panel
def build_signals(Pfull, study_ufs, pool_variants=("all", "nonstudy", "p25", "p50"), with_trends=True,
                  excluded_state_years=None):
    """All raw same-year signals aligned to ``Pfull`` (municipality order and years).

    Returns a dict with ``pool_<variant>`` (M, T, 53) counts, ``pool_size_<variant>``
    (M,), ``trends`` (M, T, 53), ``trends_fallback`` (M,), ``signal_missing`` (M, T)
    (state-years flagged as excluded in the study data, e.g. ES 2021: the state
    signal is treated as missing there), and the pool membership maps.
    """
    years = [int(y) for y in Pfull.years]
    C.assert_year_allowed(years)
    d = load_infodengue_counts(years)
    A, geos, ufs = count_cube(d, years)
    study = [int(g) for g in Pfull.geocodes]
    missing = set(study) - set(int(g) for g in geos)
    assert not missing, f"study municipalities missing from the InfoDengue pool data: {sorted(missing)[:5]}"
    out = dict(years=np.array(years), study_ufs=list(study_ufs), pool_maps={})
    for v in pool_variants:
        pm = pools(study, geos, ufs, v, study_set=study)
        cnt, size = pool_counts(A, geos, pm, study)
        out[f"pool_{v}"], out[f"pool_size_{v}"], out["pool_maps"][v] = cnt, size, pm
    if with_trends:
        out["trends"], out["trends_fallback"] = trends_cube(load_trends(years), study_ufs, years)
    miss = np.zeros((len(study), len(years)), bool)
    if excluded_state_years:
        for m, u in enumerate(study_ufs):
            for t, y in enumerate(years):
                if (u, y) in excluded_state_years:
                    miss[m, t] = True
    out["signal_missing"] = miss
    return out


def excluded_state_years_from(d: pd.DataFrame, muni: pd.DataFrame) -> set:
    """{(uf, year)} where any study municipality-year carries ``exclude_suggested``."""
    x = d[d.exclude_suggested].merge(muni[["geocode", "uf"]], on="geocode")
    return set(zip(x.uf, x.epi_year.astype(int)))


def subset_signals(sig: dict, years) -> dict:
    """Restrict every (M, T, 53) / (M, T) array of :func:`build_signals` to ``years``."""
    idx = np.where(np.isin(sig["years"], list(years)))[0]
    out = {}
    for k, v in sig.items():
        if isinstance(v, np.ndarray) and v.ndim >= 2 and v.shape[1] == len(sig["years"]):
            out[k] = v[:, idx]
        else:
            out[k] = v
    out["years"] = sig["years"][idx]
    return out
