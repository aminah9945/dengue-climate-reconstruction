"""Epidemiological-week (SE / MMWR) calendar used by Brazil's SINAN and InfoDengue.

Week 1 of epi-year Y is the Sunday-to-Saturday week containing 4 January of Y
(equivalently: the first week with >= 4 days in Y). Years have 52 or 53 weeks
(53 in 2014, 2020 and 2025 within 2009-2025).
"""
from __future__ import annotations

import datetime as dt

import pandas as pd


def epiyear_start(year: int) -> dt.date:
    """Sunday that starts epi week 1 of ``year``."""
    jan4 = dt.date(year, 1, 4)
    # Python weekday(): Mon=0 ... Sun=6  -> days back to the preceding Sunday
    return jan4 - dt.timedelta(days=(jan4.weekday() + 1) % 7)


def weeks_in_epiyear(year: int) -> int:
    """Number of epi weeks (52 or 53) in ``year``."""
    return (epiyear_start(year + 1) - epiyear_start(year)).days // 7


def calendar(y0: int, y1: int) -> pd.DataFrame:
    """One row per epi week: epi_year, epi_week, week_start (Sunday), n_weeks_year, week_index."""
    rows = []
    for y in range(y0, y1 + 1):
        s, n = epiyear_start(y), weeks_in_epiyear(y)
        for w in range(1, n + 1):
            rows.append((y, w, pd.Timestamp(s + dt.timedelta(weeks=w - 1)), n))
    cal = pd.DataFrame(rows, columns=["epi_year", "epi_week", "week_start", "n_weeks_year"])
    cal["week_index"] = range(len(cal))  # consecutive index, convenient for lags
    return cal


def date_to_epiweek(dates: pd.Series, y0: int, y1: int) -> pd.DataFrame:
    """Map daily dates to (epi_year, epi_week) via the calendar."""
    cal = calendar(y0 - 1, y1 + 1)
    idx = pd.Series(range(len(cal)), index=cal["week_start"])
    d = pd.to_datetime(dates)
    start = d - pd.to_timedelta((d.dt.weekday + 1) % 7, unit="D")
    pos = idx.reindex(start.values).values
    return cal.iloc[pos][["epi_year", "epi_week"]].reset_index(drop=True)


if __name__ == "__main__":
    for y in range(2009, 2027):
        print(y, epiyear_start(y), weeks_in_epiyear(y))
