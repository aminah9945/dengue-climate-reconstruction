import datetime as dt

from dengue_recon.data.epiweeks import calendar, date_to_epiweek, epiyear_start, weeks_in_epiyear


def test_53_week_years():
    long_years = [y for y in range(2009, 2027) if weeks_in_epiyear(y) == 53]
    assert long_years == [2014, 2020, 2025]
    assert all(weeks_in_epiyear(y) == 52 for y in (2010, 2015, 2019, 2021, 2024))


def test_week1_contains_jan4_and_starts_sunday():
    for y in range(2009, 2027):
        s = epiyear_start(y)
        assert s.weekday() == 6                       # Sunday
        assert s <= dt.date(y, 1, 4) < s + dt.timedelta(days=7)


def test_calendar_is_contiguous_and_complete():
    cal = calendar(2013, 2015)
    assert len(cal) == 52 + 53 + 52
    gaps = cal.week_start.diff().dropna().dt.days
    assert (gaps == 7).all()
    assert list(cal.week_index) == list(range(len(cal)))
    w53 = cal[(cal.epi_year == 2014) & (cal.epi_week == 53)]
    assert len(w53) == 1


def test_date_to_epiweek_roundtrip():
    import pandas as pd
    cal = calendar(2019, 2021)
    days = pd.Series(pd.date_range("2019-01-01", "2021-12-31", freq="D"))
    ew = date_to_epiweek(days, 2019, 2021)
    # the last days of 2020-12-31 .. 2021-01-02 belong to 2020 W53
    i = days.index[days == pd.Timestamp("2021-01-01")][0]
    assert (ew.epi_year[i], ew.epi_week[i]) == (2020, 53)
    assert ew.groupby(["epi_year", "epi_week"]).size().max() == 7
    assert set(zip(ew.epi_year, ew.epi_week)) <= set(zip(cal.epi_year, cal.epi_week))
