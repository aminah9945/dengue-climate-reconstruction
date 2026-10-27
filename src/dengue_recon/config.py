"""Central configuration: paths, study design switches, seeds and penalty scales.

Every path can be overridden with an environment variable, so the same code runs
locally, in VS Code and in Google Colab:

==================  ===========================================================
``DENGUE_ROOT``     repository root (default: detected from the package location,
                    falling back to the current working directory).
                    ``DENGUE_EXP_DIR`` is accepted as a legacy alias.
``DATA_DIR``        processed data folder (default ``<root>/data``)
``RESULTS_DIR``     output folder (default ``<root>/results``)
``REPORTS_DIR``     data-foundation reports (EDA, smoke test, source validation;
                    default ``<RESULTS_DIR>/data_foundation``)
``LIM_MASK_PATH``   path to the Lim et al. (2026) interpolation CV mask ``.rds``
==================  ===========================================================

The numerical settings (seeds, penalty scales, lag blocks, exclusions) are the
frozen pilot settings that produced the reported results. Do not change them
unless you intend to run a different experiment.
"""
from __future__ import annotations

import math
import os
from pathlib import Path


def _default_root() -> Path:
    here = Path(__file__).resolve()
    # editable install / source checkout: <root>/src/dengue_recon/config.py
    cand = here.parents[2]
    if (cand / "pyproject.toml").exists():
        return cand
    return Path.cwd()


ROOT = Path(os.environ.get("DENGUE_ROOT", os.environ.get("DENGUE_EXP_DIR", _default_root())))
DATA_DIR = Path(os.environ.get("DATA_DIR", ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", ROOT / "results"))
REPORTS_DIR = Path(os.environ.get("REPORTS_DIR", RESULTS_DIR / "data_foundation"))
FIG_DIR = RESULTS_DIR / "figures"
BRAZIL_DIR = RESULTS_DIR / "brazil"
REPRO_DIR = RESULTS_DIR / "repro_lim"
RAIN_DIR = RESULTS_DIR / "rain_extension"
OFFICIAL_DIR = Path(os.environ.get("OFFICIAL_DIR", RESULTS_DIR / "official"))


def ensure_dirs() -> None:
    """Create the output folders (called by the pipeline entry points, not on import)."""
    for d in (RESULTS_DIR, FIG_DIR, BRAZIL_DIR, REPRO_DIR, RAIN_DIR, REPORTS_DIR, REPORTS_DIR / "figures"):
        d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------- base paper (Step 2)
LIM_COMMIT = "00ff7aeadcaa83f96be44fbc5197bedd88dc00cd"
LIM_REPO_URL = "https://github.com/ahyoung-lim/OD_gap_filling_public"
LIM_MASK_REL = "runs/CV/20260126/masks/downscaling_rep01/inla_down_mask_interp_gzip_v2.rds"
LIM_TABLE_REL = "output/tables/CV_tab1_cv_performance_disaggregation.csv"
LIM_DIR = RAW_DIR / "lim_repo"
LIM_MASK_PATH = Path(os.environ.get("LIM_MASK_PATH", LIM_DIR / "masks" / "inla_down_mask_interp_gzip_v2.rds"))
LIM_TARGET_TABLE = LIM_DIR / "CV_tab1_cv_performance_disaggregation.csv"

# ---------------------------------------------------------------- study design (Step 6)
RUN_FINAL_2024 = False          # 2024 stays untouched in the pilot (never set True here)
LOYO_YEARS = list(range(2010, 2024))   # 14 leave-one-year-out folds
FINAL_TEST_YEAR = 2024
MIN_TOTAL_TRAIN = 20            # training rows: annual total >= 20
MIN_TOTAL_EVAL = 100            # evaluation rows: annual total >= 100
EPS_H = 1e-6                    # base-paper epsilon for log historical shares
EPS_H_GUARD = 1e-4              # guard variant (ablation B4s-eps1e-4)
REGIONS = ["North", "Northeast", "Center-West", "Southeast", "South"]
REG_COL = {"North": "#1b9e77", "Northeast": "#d95f02", "Center-West": "#7570b3",
           "Southeast": "#e7298a", "South": "#66a61e"}
MODEL_COL = {"B0": "#bdbdbd", "B1": "#6baed6", "B1p": "#9ecae1", "B4": "#fd8d3c",
             "B4s": "#e6550d", "M1": "#31a354", "M1-short": "#a1d99b",
             "M1-mult": "#756bb1", "M1-shufTT": "#969696", "M2": "#de2d26",
             "B4s-eps1e-4": "#fdae6b"}
BELO_HORIZONTE = 3106200


def max_loaded_year() -> int:
    """Last epi year any loader may read: 2023 while the 2024 lock is on (2025 never)."""
    return FINAL_TEST_YEAR if RUN_FINAL_2024 else FINAL_TEST_YEAR - 1


def assert_year_allowed(year) -> None:
    """Fail loudly if ``year`` (or any of an iterable of years) is the locked final-test year."""
    ys = [int(y) for y in (year if hasattr(year, "__iter__") else [year])]
    assert max(ys, default=0) <= max_loaded_year(), \
        f"year {max(ys)} requested but RUN_FINAL_2024 is False: 2024 must not be loaded"


# ---------------------------------------------------------------- official experiment (designs A/B/C)
# Fixed before any official model run (plan sections 3-4; Amendment 1).
CLIMATE_FILES = {"power": "climate_weekly.parquet",          # NASA POWER, primary (Amendment 1)
                 "chirps": "climate_weekly_chirps.parquet"}  # CHIRPS rain + POWER temp/humid, robustness
OFFICIAL_FIRST_TRAIN = 2010
OFFICIAL_TEST_YEARS_A = list(range(2012, 2024))      # design A: expanding window, 12 folds
LC_TEST_YEARS = (2019, 2021, 2023)                   # design B: learning curve
LC_K_MIN = 2
CLIM_EXTRA_YEARS = (2009,)                           # climate-only year (no dengue target) for lags
OFFICIAL_MODELS = {
    "A": ["B0", "B1", "B4", "B4s", "M1", "M1-noONI", "M1-short", "M1-shufTT", "M1-shufTest", "M2"],
    "B": ["B0", "B1", "B4s", "M1", "M2"],
    "C": ["B0", "B1", "B4", "B4s", "M1", "M1-noONI", "M1-short", "M1-shufTT", "M1-shufTest", "M2"],
}
# M2: fixed grid, chosen by early stopping on the inner validation year (last training year)
M2_GRID = [dict(max_depth=d, eta=e) for d in (3, 5) for e in (0.05, 0.1)]
M2_MAX_ROUNDS = 2000
M2_EARLY_STOP = 50
M2_NTHREAD = 4

# ---------------------------------------------------------------- climate features (Step 4)
CLIM_VARS = ["temp_mean", "precip", "humid_mean"]
LAG_BLOCKS = [(0, 3), (4, 7), (8, 11), (12, 15), (16, 20)]
N_HARM_CLIM = 3

# ---------------------------------------------------------------- penalties (Step 5)
# Pilot hyperparameters, fixed in advance (experiment specification). Gaussian
# penalty scales mirror the base paper's prior scales.
PEN = dict(g=0.3, r=0.4, c=0.15, rw=0.1, gamma_nat=1.0, gamma_reg=0.5,
           theta=0.5, delta=0.25, logphi_mu=float(math.log(50.0)), logphi_sd=1.0)

# ---------------------------------------------------------------- seeds and sizes
SEED = 20250811
N_DRAWS = 500
N_BOOT = 2000
BOOT_SEED = 1
