#!/usr/bin/env python
"""Regenerate the Colab/local notebooks in ``notebooks/`` (outputs empty).

Each notebook starts with the same setup cell: in Google Colab it clones this
repository and installs it with ``pip install -e .``; locally it uses the
checkout it lives in. Paths come from environment variables (DATA_DIR,
RESULTS_DIR, ...), see ``src/dengue_recon/config.py``.

Usage:  python scripts/build_notebooks.py
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

NB_DIR = Path(__file__).resolve().parent.parent / "notebooks"

SETUP = r'''# ---- Setup: works in Google Colab and locally (VS Code / Jupyter) ----
import os, sys, subprocess
from pathlib import Path

REPO_URL = "https://github.com/aminah9945/dengue-climate-reconstruction"
IN_COLAB = "google.colab" in sys.modules

if IN_COLAB:
    REPO = Path("/content/dengue-climate-reconstruction")
    if not REPO.exists():
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(REPO)], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", f"{REPO}[all]"], check=True)
    # Optional: keep the downloaded data on Google Drive so it survives the session.
    USE_DRIVE = False
    if USE_DRIVE:
        from google.colab import drive
        drive.mount("/content/drive")
        os.environ.setdefault("DATA_DIR", "/content/drive/MyDrive/dengue_recon_data")
else:
    # Local: this notebook lives in <repo>/notebooks/
    REPO = Path(os.environ.get("DENGUE_ROOT", Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd())).resolve()

os.environ.setdefault("DENGUE_ROOT", str(REPO))
os.environ.setdefault("DATA_DIR", str(REPO / "data"))
os.environ.setdefault("RESULTS_DIR", str(REPO / "results"))
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))      # makes the package importable without a kernel restart

from dengue_recon import config as C
C.ensure_dirs()
print("repo       :", C.ROOT)
print("DATA_DIR   :", C.DATA_DIR, "| processed data present:", (C.DATA_DIR / "dengue_weekly.parquet").exists())
print("RESULTS_DIR:", C.RESULTS_DIR)
'''

HELPERS = r'''import json
import numpy as np, pandas as pd
from IPython.display import Image, Markdown, display
pd.set_option("display.width", 200, "display.max_columns", 30)


def show(path, width=900):
    p = Path(path)
    p = p if p.is_absolute() else C.FIG_DIR / p
    display(Image(str(p), width=width)) if p.exists() else print("missing figure:", p)


def have_data():
    ok = (C.DATA_DIR / "dengue_weekly.parquet").exists()
    if not ok:
        print("Processed data not found in DATA_DIR - run notebook 00 first (or set DATA_DIR).")
    return ok
'''


def md(s):
    return nbf.v4.new_markdown_cell(s.strip())


def code(s):
    return nbf.v4.new_code_cell(s.strip())


def save(name, cells):
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                      "language_info": {"name": "python"}, "colab": {"provenance": []}}
    NB_DIR.mkdir(exist_ok=True)
    nbf.write(nb, NB_DIR / name)


def build():
    save("00_setup_and_data.ipynb", [
        md("""
# 00 Setup and data (Step 1)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aminah9945/dengue-climate-reconstruction/blob/main/notebooks/00_setup_and_data.ipynb)

This notebook prepares the inputs for all other notebooks:

1. clones and installs this repository (in Colab);
2. downloads the **data** into `DATA_DIR` if the processed files are missing: weekly dengue notifications from the InfoDengue API and daily NASA POWER climate for 300 Brazilian municipalities, then builds the tidy tables (`dengue_recon.data`). The download is resumable; expect 30-45 minutes from scratch;
3. fetches the base paper's cross-validation mask (Lim et al. 2026) at a pinned commit. Only the data file is read; none of that repository's code is executed;
4. shows the data-foundation EDA.

Nothing large is stored in the repository: data are always re-downloaded from the original sources (see `data/README.md`).
"""),
        code(SETUP),
        code(HELPERS),
        md("""
## 1. Data (InfoDengue + NASA POWER + IBGE)

Set `DOWNLOAD = True` to fetch and build the data if it is missing. The steps are those of `python scripts/run_pipeline.py --step data`. 2024 is downloaded but never used by the pilot (`RUN_FINAL_2024 = False`).
"""),
        code(r'''
DOWNLOAD = True
NEEDED = ["dengue_weekly.parquet", "climate_weekly.parquet", "municipalities.csv", "year_summary.csv"]
missing = [f for f in NEEDED if not (C.DATA_DIR / f).exists()]
print("missing processed files:", missing or "none")
if missing and DOWNLOAD:
    from dengue_recon.data import build_dataset, download, eda
    download.main(["infodengue", "--top-pop", "700"])
    download.main(["select", "--n", "300"])
    download.main(["power"])
    build_dataset.main()
    eda.main()          # EDA figures and the smoke-test baselines (needed for sanity check 7)
if have_data():
    d = pd.read_parquet(C.DATA_DIR / "dengue_weekly.parquet", columns=["geocode", "epi_year", "cases"])
    print(d.geocode.nunique(), "municipalities;", d.epi_year.min(), "-", d.epi_year.max(), ";", len(d), "weekly rows")
'''),
        md("## 2. Base-paper data (Lim et al. 2026 CV mask, read as data only)"),
        code(r'''
from dengue_recon.base_paper.fetch import fetch_lim_mask
mask = fetch_lim_mask()
print("mask:", mask, mask.exists())
pd.read_csv(C.LIM_TARGET_TABLE).head(3)
'''),
        md("## 3. Data-foundation EDA (precomputed in `results/data_foundation/`)"),
        code(r'''
for f in ["seasonal_curves_by_region.png", "peak_week_variation.png", "coverage_heatmap.png"]:
    show(C.REPORTS_DIR / "figures" / f)
pd.read_csv(C.REPORTS_DIR / "baseline_smoke_test_summary.csv", header=[0, 1], index_col=0)
'''),
        md("Setup done. Continue with `01_reproduce_base_paper.ipynb`."),
    ])

    save("01_reproduce_base_paper.ipynb", [
        md("""
# 01 Reproduce the base paper (Step 2)

**Base model (Lim et al. 2026, DM_v3_pooled).** For each country the annual total is known; monthly shares are `softmax(log historical shares + pooled month deviations + year effect)` and monthly counts follow a Dirichlet-multinomial.

**Cancellation.** The year effect is the same for all 12 months, so it cancels in the softmax: every held-out year of a country gets the same predicted shape. We check this numerically.

**What runs here** (our re-implementation in `dengue_recon.base_paper.lim_dm`):
- *MAP*: the same log posterior maximised with L-BFGS (about 5 s per fold after compilation) - run live;
- *NUTS*: NumPyro, 2 chains x (500 warm-up + 500 draws) - precomputed in `results/repro_lim/folds/`; set `RUN_NUTS = True` to refit (about 20 min per fold on 4 CPUs).

Reduced design vs the paper: 1 repetition (rep01) x interpolation mask x 3 folds (paper: 3 repetitions, 4 chains x 1,000 draws).
"""),
        code(SETUP),
        code(HELPERS),
        code(r'''
RUN_MAP = True
RUN_NUTS = False
from dengue_recon.base_paper import reproduce
from dengue_recon.base_paper.fetch import fetch_lim_mask
if RUN_MAP or RUN_NUTS:
    fetch_lim_mask()
    for f in (1, 2, 3):
        if RUN_MAP:
            reproduce.run("MAP", f, force=True)
        if RUN_NUTS:
            reproduce.run("NUTS", f)
tab = reproduce.table()
'''),
        md("""
**Reading the table.** `MAE_median`/`RMSE_median` use the posterior-predictive median (the definition of the published table); `MAE`/`RMSE` use the mean. `COV80` is 80 % interval coverage, `CRPS` the probabilistic score. `JSD_mean` and `peak_month_err` are our shape metrics. `uniform` and `historical_mean_shares` are reference rows on the same test months.
"""),
        code(r'''show("fig_repro_lim.png")'''),
        md("### Cancellation check (max |share difference| between held-out years of the same country)"),
        code(r'''pd.read_csv(C.REPRO_DIR / "cancellation_check.csv").groupby("fit").max_abs_diff.describe()'''),
    ])

    save("02_brazil_baselines.ipynb", [
        md("""
# 02 Brazil: baselines without climate (Steps 3-6)

**Task.** For each municipality-year we hide the weekly counts and keep only the annual total. A model predicts the 52/53 weekly *shares*; shares x total, rounded with the largest-remainder rule, gives integer weekly counts that add up exactly to the total.

**Validation.** Leave-one-year-out (LOYO) over 2010-2023 (14 folds, each holds out one year for all municipalities). Everything estimated from data is recomputed inside each fold from training years only. **2024 is never loaded.**

| ID | Model |
|---|---|
| B0 | uniform shares |
| B1 | municipality's historical mean weekly shares (LOYO) |
| B1p | same, years before the target year only |
| B4 | Dirichlet-multinomial with the base paper's structure (offset + national/regional/municipal week deviations, smoothness penalty, one concentration per region) |
| B4s | B4 + annual-size terms - **primary comparator for climate** |

Results are precomputed; set `RERUN = True` to refit (weekly, 60 municipalities, main models: about 5 min on 4 CPUs).
"""),
        code(SETUP),
        code(HELPERS),
        code(r'''
RERUN = False
if RERUN and have_data():
    from dengue_recon.experiments import brazil_loyo
    brazil_loyo.main("weekly", 60, models=["B4", "B4s"], tag="weekly_60_noclimate")
sub = pd.read_csv(C.RESULTS_DIR / "subset60.csv")
print(sub.groupby("region").size())
'''),
        md("### Live demo: B0 and B1 for one fold (no fitting needed; needs the data)"),
        code(r'''
if have_data():
    from dengue_recon.baselines import seasonal_shares, uniform_shares
    from dengue_recon.data import panel as BD
    from dengue_recon.evaluation.cv import loyo_folds, row_masks
    from dengue_recon.evaluation.metrics import jsd
    from dengue_recon.features.seasonal import offsets
    d, clim, muni = BD.load_tables()
    P = BD.build_panel(d, clim, sub.geocode.values)
    train_ok, eval_ok = row_masks(P)
    t, y, tm = next(loyo_folds(P, train_ok, years=[2019]))
    h, fb, prof = offsets(P, tm, weekly=True)
    rows = np.where(eval_ok[:, t])[0]; n = P.nweeks[t]
    j0 = [jsd(P.Yw[m, t, :n] / P.N[m, t], uniform_shares(n)) for m in rows]
    j1 = [jsd(P.Yw[m, t, :n] / P.N[m, t], seasonal_shares(prof[m], n)) for m in rows]
    print(f"{y}, {len(rows)} municipality-years: JSD uniform {np.mean(j0):.3f}, seasonal profile {np.mean(j1):.3f}")
'''),
        md("### Main table (all LOYO folds)"),
        code(r'''
cols = ["model", "subset", "n", "jsd_mean", "jsd_median", "peak_err_mean", "emd_mean", "mae_share_mean", "cov80_mean", "cov95_mean", "crps_mean", "jsd_Nweighted"]
for res in ["monthly", "weekly"]:
    t_ = pd.read_csv(C.BRAZIL_DIR / f"table_main_{res}.csv")
    display(Markdown(f"**{res}** - {t_.note.iloc[0]}"))
    display(t_[(t_.subset == "all") & t_.model.isin(["B0", "B1p", "B1", "B4", "B4s", "B4s-eps1e-4"])][[c for c in cols if c in t_]].round(4))
'''),
        md("### Paired differences between baselines (negative = first model better)"),
        code(r'''
p = pd.read_csv(C.BRAZIL_DIR / "table_paired_weekly.csv")
p[(p.subset == "all") & (p.metric == "jsd") & p.model_a.isin(["B4s", "B4", "B1", "B4s-eps1e-4"])][
    ["model_a", "model_b", "n", "mean_diff", "rel_change", "ci_muni_lo", "ci_muni_hi", "ci_year_lo", "ci_year_hi", "n_years_a_better"]].round(5)
'''),
    ])

    save("03_climate_extension.ipynb", [
        md("""
# 03 Climate extension M1 and sanity checks (Steps 4, 5, 7)

**M1 = B4s + climate.** Each week's logit gets sum(theta * X), where X are standardised climate **anomalies** (temperature, log(1 + rain), humidity; anomaly = value minus the municipality's harmonic climatology fitted on training years, divided by the residual SD), averaged over 5 lag blocks (0-3, 4-7, 8-11, 12-15, 16-20 weeks) on a continuous weekly timeline. Coefficients are national with shrunken regional deviations.

**Ablations and checks:** M1-short (0-3 weeks only), M1-mult (multinomial likelihood), M1-shufTT (train-and-test climate shuffle, 3 seeds), M1-shufTest (wrong-year climate at test time), zero-climate equivalence, exact sums, uniform-worst, leakage audit, pipeline consistency.

Results are precomputed. `RERUN = True` refits the full weekly pilot (about 22 min on 4 CPUs); `QUICK = True` refits only B4, B4s and M1 (about 5 min) and prints their mean JSD.
"""),
        code(SETUP),
        code(HELPERS),
        code(r'''
RERUN, QUICK = False, False
if (RERUN or QUICK) and have_data():
    from dengue_recon.experiments import brazil_loyo
    if QUICK:
        df = brazil_loyo.main("weekly", 60, main_only=True, tag="weekly_60_quick")
    else:
        brazil_loyo.main("weekly", 60)
tm = pd.read_csv(C.BRAZIL_DIR / "table_main_weekly.csv")
display(Markdown(f"**weekly, {tm.note.iloc[0]}**"))
tm[tm.subset == "all"][["model", "n", "jsd_mean", "jsd_median", "peak_err_mean", "emd_mean", "cov80_mean", "cov95_mean", "crps_mean", "jsd_Nweighted"]].round(4)
'''),
        md("### Primary contrast (pilot, not a test of H1)\n`mean_diff` < 0 means the first model has lower JSD; intervals from municipality and year-block bootstraps."),
        code(r'''
p = pd.read_csv(C.BRAZIL_DIR / "table_paired_weekly.csv")
keep = [("M1", "B4s"), ("M1", "B1"), ("M1", "M1-short"), ("M1-shufTT", "B4s"), ("M1-mult", "M1"), ("M2", "B4s")]
sel = p[p.apply(lambda r: (r.model_a, r.model_b) in keep, axis=1) & (p.metric == "jsd")]
sel[["subset", "model_a", "model_b", "n", "mean_diff", "rel_change", "ci_muni_lo", "ci_muni_hi", "ci_year_lo", "ci_year_hi", "n_years_a_better", "headroom_captured"]].round(5)
'''),
        md("### Sanity checks"),
        code(r'''pd.read_csv(C.BRAZIL_DIR / "sanity_checks.csv")'''),
        code(r'''show("fig_shuffle.png", 800)'''),
        md("### Where does climate help? (H3: South + Southeast vs North)"),
        code(r'''pd.read_csv(C.BRAZIL_DIR / "table_by_region_weekly.csv")[["region", "model_a", "model_b", "n", "mean_diff", "ci_muni_lo", "ci_muni_hi"]].round(5)'''),
        code(r'''show("fig_gain_by_region_year.png", 1000)'''),
        md("### Lag-block coefficients (left: real climate; right: one shuffled fit)"),
        code(r'''show("fig_lag_coefficients.png", 1000)'''),
    ])

    save("04_results_figures.ipynb", [
        md("""
# 04 Results tables and figures (Step 9)

Regenerates the tables and figures from the saved results (`dengue_recon.reporting`). The large prediction files (`preds_*.npz`) and the 300-municipality metrics file are not in the repository; figures that need them are kept from the pilot run and are re-created after running the `brazil` step (see `results/README.md`).
"""),
        code(SETUP),
        code(HELPERS),
        code(r'''
REGENERATE = False   # True rewrites results/ tables from the saved metrics (about 15 s)
if REGENERATE:
    from dengue_recon.reporting import rain_report, report
    report.main(); rain_report.main()
pd.read_csv(C.BRAZIL_DIR / "table_abs_means_ci.csv").round(4)
'''),
        md("## Reproduction of the base paper"), code(r'''show("fig_repro_lim.png")'''),
        md("## Baselines vs climate model: JSD and peak-week error with 95 % CIs"),
        code(r'''show("fig_baseline_vs_climate_weekly_60.png"); show("fig_baseline_vs_climate_weekly_300.png")'''),
        md("## JSD distribution per model (weekly)"), code(r'''show("fig_jsd_by_model.png")'''),
        md("## Gain by region and by year (300 municipalities)"), code(r'''show("fig_gain_by_region_year_weekly_300.png", 1000)'''),
        md("## Example reconstructions (Belo Horizonte 2016 and 2019; largest M1 gain and loss)"),
        code(r'''show("fig_example_bh.png", 1000)'''),
        md("## 300-municipality paired differences"),
        code(r'''
p = pd.read_csv(C.BRAZIL_DIR / "table_paired_weekly_300.csv")
p[(p.metric == "jsd")][["subset", "model_a", "model_b", "n", "mean_diff", "rel_change", "ci_muni_lo", "ci_muni_hi", "ci_year_lo", "ci_year_hi", "n_years_a_better"]].round(5)
'''),
    ])

    save("05_rain_lag_extension.ipynb", [
        md("""
# 05 Rain-lag extension (Step 8, pre-registered)

Does a more realistic rainfall effect raise the climate gain over B4s?

- **M1-R**: lags extended to 26 weeks; rain block mean split into wet = max(A, 0) and dry = min(A, 0) (24 features).
- **M1-RTH**: M1-R plus the same split for temperature and humidity (36 features).

The design was written in `results/rain_extension/PREREGISTRATION.md` before any new model was fitted. Results are precomputed; `RERUN = True` refits at 60 municipalities (about 6 min on 4 CPUs).
"""),
        code(SETUP),
        code(HELPERS),
        code(r'''
RERUN = False
if RERUN and have_data():
    from dengue_recon.extensions import rain_lags
    rain_lags.main(60, ("M1", "M1-R", "M1-RTH"), "weekly_60")
    rain_lags.main(60, ("M1-R-shufTT-s0",), "weekly_60_shuf")
display(Markdown((C.RAIN_DIR / "PREREGISTRATION.md").read_text()[:1500] + " ..."))
'''),
        code(r'''
for s in (60, 300):
    t = pd.read_csv(C.RAIN_DIR / f"table_paired_weekly_{s}.csv")
    display(Markdown(f"**{s} municipalities**"))
    display(t[(t.metric == "jsd") & (t.subset == "all")][["model_a", "model_b", "n", "mean_diff", "rel_change", "ci_muni_lo", "ci_muni_hi", "ci_year_lo", "ci_year_hi", "n_years_a_better"]].round(5))
'''),
        code(r'''show(C.RAIN_DIR / "fig_rain_gain_vs_M1_B4s.png"); show(C.RAIN_DIR / "fig_rain_coefficients_300.png", 1100)'''),
    ])
    print("notebooks written to", NB_DIR)


if __name__ == "__main__":
    build()
