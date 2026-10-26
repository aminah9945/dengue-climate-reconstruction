"""Model-ready panel arrays from the processed tables (Step 1, preparation).

Builds dense arrays over (municipality m, epi year t, slot s):

* weekly : 53 slots; slot 53 is masked (invalid) in 52-week years
* monthly: 12 slots; epi week -> calendar month of its Wednesday (week_start + 3 d)

2024 is NOT loaded unless ``config.RUN_FINAL_2024`` is True (the pilot keeps it
untouched), and 2025 (provisional) is never loaded.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .. import config as C


@dataclass
class Panel:
    """Dense municipality x year x slot arrays used by every model."""
    geocodes: np.ndarray            # (M,)
    names: list
    region: np.ndarray              # (M,) region index into C.REGIONS
    years: np.ndarray               # (T,)
    nweeks: np.ndarray              # (T,) 52/53
    Yw: np.ndarray                  # (M, T, 53) weekly counts
    Ym: np.ndarray                  # (M, T, 12) monthly counts
    N: np.ndarray                   # (M, T) annual totals
    excluded: np.ndarray            # (M, T) bool (exclude_suggested)
    valid_w: np.ndarray             # (T, 53) bool
    week_index: np.ndarray          # (T, 53) continuous index (climate table), -1 if invalid
    month_of_week: np.ndarray       # (T, 53) 0..11, -1 if invalid
    extra: dict = field(default_factory=dict)


def load_tables():
    """Read the processed dengue, climate and municipality tables (2024+ dropped unless enabled)."""
    d = pd.read_parquet(C.DATA_DIR / "dengue_weekly.parquet",
                        columns=["geocode", "name", "region", "epi_year", "epi_week", "week_start",
                                 "n_weeks_year", "cases", "year_total", "exclude_suggested"])
    clim = pd.read_parquet(C.DATA_DIR / "climate_weekly.parquet")
    muni = pd.read_csv(C.DATA_DIR / "municipalities.csv")
    max_year = C.FINAL_TEST_YEAR if C.RUN_FINAL_2024 else C.FINAL_TEST_YEAR - 1
    d = d[d.epi_year <= max_year]
    clim = clim[clim.epi_year <= max_year]
    return d, clim, muni


def select_subset60(d: pd.DataFrame, muni: pd.DataFrame, per_region: int = 12) -> pd.DataFrame:
    """Pilot subset: per region the 12 municipalities with most evaluable years (N >= 100,
    2010-2023), ties broken by total cases; Belo Horizonte is forced into the Southeast."""
    ys = d.groupby(["geocode", "epi_year"]).agg(year_total=("year_total", "first"),
                                                excl=("exclude_suggested", "first")).reset_index()
    ys = ys[(ys.epi_year.between(2010, 2023)) & (~ys.excl)]
    s = ys.groupby("geocode").agg(n_good=("year_total", lambda x: int((x >= C.MIN_TOTAL_EVAL).sum())),
                                  total=("year_total", "sum")).reset_index()
    s = s.merge(muni[["geocode", "name", "uf", "region"]], on="geocode")
    out = []
    for r in C.REGIONS:
        sub = s[s.region == r].sort_values(["n_good", "total"], ascending=False).head(per_region)
        if r == "Southeast" and C.BELO_HORIZONTE not in sub.geocode.values:
            sub = pd.concat([sub.head(per_region - 1), s[s.geocode == C.BELO_HORIZONTE]])
        out.append(sub)
    return pd.concat(out, ignore_index=True)


def build_panel(d: pd.DataFrame, clim: pd.DataFrame, geocodes) -> Panel:
    """Assemble the :class:`Panel` for ``geocodes`` (ordered by region, then geocode)."""
    geocodes = np.asarray(sorted(geocodes, key=lambda g: (C.REGIONS.index(
        d.loc[d.geocode == g, "region"].iloc[0]), g)))
    d = d[d.geocode.isin(geocodes)].copy()
    years = np.array(sorted(d.epi_year.unique()))
    M, T = len(geocodes), len(years)
    gi = {g: i for i, g in enumerate(geocodes)}
    ti = {y: i for i, y in enumerate(years)}
    Yw = np.zeros((M, T, 53))
    mi, tix, wi = d.geocode.map(gi).values, d.epi_year.map(ti).values, d.epi_week.values - 1
    Yw[mi, tix, wi] = d.cases.values
    nweeks = d.groupby("epi_year").n_weeks_year.first().reindex(years).values
    valid_w = np.arange(53)[None, :] < nweeks[:, None]
    # calendar from the climate table (same epi-week definition, continuous index from 2009 W1)
    cal = clim[["epi_year", "epi_week", "week_start", "week_index"]].drop_duplicates(["epi_year", "epi_week"])
    week_index = -np.ones((T, 53), int)
    month_of_week = -np.ones((T, 53), int)
    for r in cal.itertuples():
        if r.epi_year in ti:
            week_index[ti[r.epi_year], r.epi_week - 1] = r.week_index
            month_of_week[ti[r.epi_year], r.epi_week - 1] = (pd.Timestamp(r.week_start) + pd.Timedelta(days=3)).month - 1
    assert (week_index[valid_w] >= 0).all() and (week_index[~valid_w] == -1).all()
    Ym = np.zeros((M, T, 12))
    for t in range(T):
        for s in range(nweeks[t]):
            Ym[:, t, month_of_week[t, s]] += Yw[:, t, s]
    yr = d.groupby(["geocode", "epi_year"]).agg(N=("year_total", "first"), ex=("exclude_suggested", "first")).reset_index()
    N = np.zeros((M, T)); excluded = np.zeros((M, T), bool)
    N[yr.geocode.map(gi), yr.epi_year.map(ti)] = yr.N
    excluded[yr.geocode.map(gi), yr.epi_year.map(ti)] = yr.ex
    assert np.allclose(Yw.sum(-1), N) and np.allclose(Ym.sum(-1), N)
    meta = d.groupby("geocode").agg(name=("name", "first"), region=("region", "first")).loc[geocodes]
    return Panel(geocodes=geocodes, names=meta.name.tolist(),
                 region=np.array([C.REGIONS.index(r) for r in meta.region]), years=years, nweeks=nweeks,
                 Yw=Yw, Ym=Ym, N=N, excluded=excluded, valid_w=valid_w, week_index=week_index,
                 month_of_week=month_of_week)


def climate_matrix(clim: pd.DataFrame, geocodes) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Continuous weekly climate (V=3, M, K) on the climate table's week_index, plus
    epi_year and epi_week per index. Precipitation enters as log(1 + mm)."""
    c = clim[clim.geocode.isin(geocodes)].copy()
    c["precip"] = np.log1p(c["precip_mm"])
    K = int(c.week_index.max()) + 1
    gi = {g: i for i, g in enumerate(geocodes)}
    X = np.full((len(C.CLIM_VARS), len(geocodes), K), np.nan)
    for v, var in enumerate(C.CLIM_VARS):
        X[v, c.geocode.map(gi).values, c.week_index.values] = c[var].values
    cal = c.drop_duplicates("week_index").set_index("week_index").sort_index()
    return X, cal.epi_year.values, cal.epi_week.values
