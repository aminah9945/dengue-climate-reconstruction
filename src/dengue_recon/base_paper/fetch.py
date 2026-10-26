"""Fetch the base paper's CV mask and published table (data only).

Clones https://github.com/ahyoung-lim/OD_gap_filling_public at the pinned commit
``00ff7ae`` (blob-less clone) and copies two files into ``DATA_DIR/raw/lim_repo/``:

* ``runs/CV/20260126/masks/downscaling_rep01/inla_down_mask_interp_gzip_v2.rds``
  (interpolation CV mask, repetition 1)
* ``output/tables/CV_tab1_cv_performance_disaggregation.csv`` (published numbers)

None of the repository's code is executed: the clone is treated as untrusted data.
"""
from __future__ import annotations

import shutil
import subprocess

from .. import config as C


def fetch_lim_mask(force: bool = False):
    """Ensure the mask and the published table exist under ``config.LIM_DIR``; return the mask path."""
    mask = C.LIM_MASK_PATH
    if mask.exists() and C.LIM_TARGET_TABLE.exists() and not force:
        return mask
    clone = C.RAW_DIR / "lim_repo_clone"
    if not clone.exists():
        subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout", C.LIM_REPO_URL, str(clone)], check=True)
    subprocess.run(["git", "-C", str(clone), "checkout", C.LIM_COMMIT], check=True)
    mask.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(clone / C.LIM_MASK_REL, mask)
    shutil.copy(clone / C.LIM_TABLE_REL, C.LIM_TARGET_TABLE)
    (C.LIM_DIR / "SOURCE.txt").write_text(f"commit {C.LIM_COMMIT}\n")
    return mask
