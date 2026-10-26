"""Validation design (Step 6): leave-one-year-out folds, exclusions and the 2024 lock.

* **LOYO over 2010-2023**: 14 folds; fold t holds out year t for *all*
  municipalities at once and trains on the other 13 years.
* **Training rows**: not excluded (``exclude_suggested``: Espirito Santo 2021,
  provisional 2025) and annual total >= 20.
* **Evaluation rows**: not excluded and annual total >= 100.
* **2024** is the untouched final-test year. ``config.RUN_FINAL_2024 = False``;
  :func:`dengue_recon.data.panel.load_tables` drops it, and
  :func:`final_test_split` (train on years < 2024, predict 2024) is only
  unit-tested on a dummy year.
"""
from __future__ import annotations

import numpy as np

from .. import config as C


def row_masks(P):
    """(train_ok, eval_ok) boolean (M, T) masks from exclusions and total thresholds."""
    train_ok = (~P.excluded) & (P.N >= C.MIN_TOTAL_TRAIN)
    eval_ok = (~P.excluded) & (P.N >= C.MIN_TOTAL_EVAL)
    return train_ok, eval_ok


def loyo_folds(P, train_ok, years=None):
    """Yield ``(t, year, train_mask)`` for each LOYO year present in the panel.

    ``train_mask`` is ``train_ok`` with the held-out year switched off.
    ``years`` optionally restricts the folds (smoke tests); default ``config.LOYO_YEARS``.
    """
    years = C.LOYO_YEARS if years is None else list(years)
    for t, y in enumerate(P.years):
        if y not in years:
            continue
        tm = train_ok.copy()
        tm[:, t] = False
        yield t, int(y), tm


def final_test_split(years, test_year):
    """Final-test path: train on all years < test_year, predict test_year.

    Used for 2024 only if ``config.RUN_FINAL_2024`` (disabled in the pilot).
    """
    years = np.asarray(years)
    return years < test_year, years == test_year


def assert_final_year_locked(P):
    """Fail loudly if the final-test year slipped into the panel while the lock is on."""
    assert P.years.max() <= C.FINAL_TEST_YEAR - 1 or C.RUN_FINAL_2024, \
        "2024 must not be loaded while RUN_FINAL_2024 is False"
