import pandas as pd

from dengue_recon.data.oni import oni_weekly, parse_oni

RAW = """ SEAS  YR   TOTAL   ANOM
  NDJ 2013  26.30  -0.40
  DJF 2014  26.06  -0.46
  JFM 2014  26.18  -0.38
  NDJ 2014  27.49   0.62
  DJF 2015  27.18   0.53
"""


def test_parse_assigns_centre_month():
    m = parse_oni(RAW)
    assert list(zip(m.year, m.month)) == [(2013, 12), (2014, 1), (2014, 2), (2014, 12), (2015, 1)]
    assert m.loc[(m.year == 2014) & (m.month == 1), "oni"].item() == -0.46


def test_weekly_uses_month_of_week_midpoint():
    m = parse_oni(RAW)
    w = oni_weekly(m, 2014, 2014).set_index(["epi_year", "epi_week"])
    # 2014 W1 starts Sun 2013-12-29; its Wednesday is 2014-01-01 -> January 2014
    assert w.loc[(2014, 1), "week_start"] == pd.Timestamp("2013-12-29")
    assert w.loc[(2014, 1), "oni"] == -0.46
    # 2014 W53 starts Sun 2014-12-28; midpoint 2014-12-31 -> December 2014 (NDJ)
    assert w.loc[(2014, 53), "oni"] == 0.62
    # a week whose Wednesday falls in February takes the JFM value
    feb = w[(w.mid_month == 2) & (w.mid_year == 2014)]
    assert len(feb) == 4 and (feb.oni == -0.38).all()
    # months absent from the table stay missing
    assert w[w.mid_month == 6].oni.isna().all()
