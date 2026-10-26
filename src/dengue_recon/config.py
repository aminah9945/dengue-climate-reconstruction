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
