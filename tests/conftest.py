"""Small synthetic panel and climate arrays so the tests run without downloaded data."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")

from dengue_recon import config as C  # noqa: E402
from dengue_recon.data.epiweeks import calendar  # noqa: E402
from dengue_recon.data.panel import Panel  # noqa: E402


def make_panel(years=(2010, 2011, 2012, 2013, 2014), n_muni=5, seed=0):
    """Synthetic weekly panel (one municipality per region) on the real epi-week calendar,
    with climate from 2009 so that lags exist for the first year."""
    rng = np.random.default_rng(seed)
    years = np.asarray(years)
    cal = calendar(2009, int(years.max()))
    T, M = len(years), n_muni
    nweeks = cal.groupby("epi_year").n_weeks_year.first().reindex(years).values
    valid_w = np.arange(53)[None, :] < nweeks[:, None]
    week_index = -np.ones((T, 53), int)
    month_of_week = -np.ones((T, 53), int)
    for r in cal.itertuples():
        if r.epi_year in years:
            t = int(np.where(years == r.epi_year)[0][0])
            week_index[t, r.epi_week - 1] = r.week_index
            month_of_week[t, r.epi_week - 1] = (pd.Timestamp(r.week_start) + pd.Timedelta(days=3)).month - 1
    Yw = np.zeros((M, T, 53))
    N = rng.integers(300, 3000, (M, T)).astype(float)
    for m in range(M):
        for t in range(T):
            n = nweeks[t]
            w = np.arange(n)
            shift = rng.normal(0, 3)
            p = np.exp(-0.5 * ((w - 14 - shift - 2 * m) / 6.0) ** 2) + 0.02
            Yw[m, t, :n] = rng.multinomial(int(N[m, t]), p / p.sum())
    Ym = np.zeros((M, T, 12))
    for t in range(T):
        for s in range(nweeks[t]):
            Ym[:, t, month_of_week[t, s]] += Yw[:, t, s]
    P = Panel(geocodes=np.arange(1000, 1000 + M), names=[f"m{i}" for i in range(M)],
              region=np.arange(M) % len(C.REGIONS), years=years, nweeks=nweeks, Yw=Yw, Ym=Ym, N=N,
              excluded=np.zeros((M, T), bool), valid_w=valid_w, week_index=week_index,
              month_of_week=month_of_week)
    K = len(cal)
    wk = cal.epi_week.values
    Xc = np.stack([
        25 + 3 * np.sin(2 * np.pi * wk / 52.18)[None] + rng.normal(0, 1, (M, K)),
        np.log1p(np.clip(30 + 25 * np.cos(2 * np.pi * wk / 52.18)[None] + rng.normal(0, 8, (M, K)), 0, None)),
        70 + 8 * np.sin(2 * np.pi * wk / 52.18 + 1)[None] + rng.normal(0, 3, (M, K)),
    ])
    return P, Xc, cal.epi_year.values, cal.epi_week.values


@pytest.fixture(scope="session")
def synth():
    return make_panel()
