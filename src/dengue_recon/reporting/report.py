"""Step 9 orchestrator: regenerate tables, sanity checks and figures from saved results.

Reads ``results/brazil/metrics_detail_*.csv``, ``fitted_params_*.json``,
``checks_*.json``, ``leakage_audit_*.csv`` (and ``preds_*.npz`` for the example
figure) plus ``results/repro_lim/table_repro.csv``. Anything whose inputs are
missing is skipped with a message (the large prediction files are not committed
to the repository; re-create them with the ``brazil`` step).

CLI: ``python scripts/run_pipeline.py --step report``.
"""
from __future__ import annotations

import pandas as pd

from .. import config as C
from ..checks.sanity import sanity_table
from . import figures as F
from . import tables as T

TAGS = ["weekly_60", "monthly_60", "weekly_300"]


def main():
    """Regenerate every table and figure that the available result files allow."""
    C.ensure_dirs()
    F.fig_repro()
    dets = {}
    for tag in TAGS:
        df = T.load_detail(tag)
        if df is None:
            print(f"[report] no metrics_detail_{tag}.csv - skipped")
            continue
        dets[tag] = df
        res = tag.split("_")[0]
        label = T.PRELIM if tag.endswith("60") else T.PRELIM.replace("60 municipalities", "300 municipalities, frozen pilot settings")
        sfx = "" if tag.endswith("60") else "_300"
        T.main_table(df, label).to_csv(C.BRAZIL_DIR / f"table_main_{res}{sfx}.csv", index=False)
        T.paired_table(df, label).to_csv(C.BRAZIL_DIR / f"table_paired_{res}{sfx}.csv", index=False)
        if res == "weekly":
            T.region_table(df, label).to_csv(C.BRAZIL_DIR / f"table_by_region_weekly{sfx}.csv", index=False)
            F.fig_jsd_by_model(df, tag); F.fig_gain_region_year(df, tag); F.fig_shuffle(df, tag)
            F.fig_lag_coefficients(tag); F.fig_main_ci(df, tag)
            T.gain_by_year(df).to_csv(C.BRAZIL_DIR / f"table_gain_by_year_weekly{sfx}.csv")
        else:
            F.fig_main_ci(df, tag)
    if "weekly_60" in dets and (C.BRAZIL_DIR / "preds_weekly_60.npz").exists():
        F.fig_examples("weekly_60")
    else:
        print("[report] preds_weekly_60.npz missing - example figure skipped")
    T.abs_means_ci(dets).to_csv(C.BRAZIL_DIR / "table_abs_means_ci.csv", index=False)
    sanity_table({k: v for k, v in dets.items() if (C.BRAZIL_DIR / f"checks_{k}.json").exists()}).to_csv(
        C.BRAZIL_DIR / "sanity_checks.csv", index=False)
    for tag in dets:
        if (C.BRAZIL_DIR / f"fitted_params_{tag}.json").exists():
            T.convergence_table(tag).to_csv(C.BRAZIL_DIR / f"convergence_{tag}.csv", index=False)
    print(pd.read_csv(C.BRAZIL_DIR / "sanity_checks.csv").to_string())


if __name__ == "__main__":
    main()
