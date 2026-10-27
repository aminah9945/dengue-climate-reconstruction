import numpy as np
import pandas as pd

from dengue_recon.data.build_dataset import aggregate_daily_to_epiweeks


def test_daily_to_epiweek_rules():
    days = pd.date_range("2019-12-29", "2020-01-14", freq="D")  # 2020 W1, W2 and 3 days of W3
    d = pd.DataFrame({"date": days, "precip": 2.0, "t2m": np.arange(len(days), dtype=float),
                      "tmax": 30.0, "tmin": 20.0, "rh": 80.0})
    d.loc[d.date == "2020-01-06", "precip"] = np.nan        # one missing rain day in W2
    w = aggregate_daily_to_epiweeks(d, geocode=1).set_index(["epi_year", "epi_week"])
    assert list(w.index) == [(2020, 1), (2020, 2)]            # partial W3 dropped
    assert w.loc[(2020, 1), "precip_mm"] == 14.0              # summed over 7 days
    assert np.isnan(w.loc[(2020, 2), "precip_mm"])            # needs all 7 days
    assert w.loc[(2020, 1), "temp_mean"] == 3.0               # mean of 0..6
    assert w.loc[(2020, 1), "precip_days"] == 7
    assert (w.geocode == 1).all()
