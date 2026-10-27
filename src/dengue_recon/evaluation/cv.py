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
* **Official designs** (below): A expanding window (primary), B learning curve,
  C LOYO, as fold specifications used by :mod:`dengue_recon.experiments.official`.
  Requesting 2024 while the lock is on raises AssertionError.
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


# ---------------------------------------------------------------- official designs A, B, C
#
# A fold is a plain dict so that it can be written to the fold logs:
#   design      "A" (expanding window), "B" (learning curve) or "C" (LOYO)
#   test_year   the held-out year T
#   train_years tuple of dengue training years (rows still need row_masks' rules)
#   k           number of training years
#   fold_id     unique, file-name safe identifier
# Everything learned inside a fold (offsets, size centring, climatology, ONI
# standardisation, M2 settings) uses only ``train_years`` (+ 2009 climate, which
# has no dengue target), see ``climate_years``.

def _fold(design, test_year, train_years):
    C.assert_year_allowed([test_year, *train_years])
    train_years = tuple(int(y) for y in train_years)
    return dict(design=design, test_year=int(test_year), train_years=train_years, k=len(train_years),
                fold_id=f"{design}_T{int(test_year)}_k{len(train_years)}")


def official_test_years_A():
    """Design A test years: 2012-2023, plus 2024 only when ``RUN_FINAL_2024`` is True."""
    return list(C.OFFICIAL_TEST_YEARS_A) + ([C.FINAL_TEST_YEAR] if C.RUN_FINAL_2024 else [])


def expanding_window_folds(first_train=C.OFFICIAL_FIRST_TRAIN, test_years=None, exclude_years=()):
    """Design A (primary): for each test year T, train on ``first_train`` ... T-1.

    Default test years 2012-2023 (+2024 only if ``RUN_FINAL_2024``). Asking for 2024
    while the lock is on raises AssertionError. ``exclude_years`` (e.g. 2020) are
    dropped both as test years and from every training window.
    """
    test_years = official_test_years_A() if test_years is None else list(test_years)
    C.assert_year_allowed(test_years)
    out = []
    for T in test_years:
        if T in exclude_years:
            continue
        tr = [y for y in range(first_train, T) if y not in exclude_years]
        assert tr, f"no training years before {T}"
        out.append(_fold("A", T, tr))
    return out


def learning_curve_folds(test_years=C.LC_TEST_YEARS, k_min=C.LC_K_MIN, first_year=C.OFFICIAL_FIRST_TRAIN,
                         k_max=None, exclude_years=()):
    """Design B (learning curve): for each test year T and k = k_min ... T - first_year,
    train on the k years immediately before T (T-k ... T-1).

    ``exclude_years`` are removed from a window (the window keeps its calendar span,
    so it then holds k - 1 usable years; ``k`` in the fold is the window length).
    """
    C.assert_year_allowed(test_years)
    out = []
    for T in test_years:
        if T in exclude_years:
            continue
        kmax = T - first_year if k_max is None else min(k_max, T - first_year)
        for k in range(k_min, kmax + 1):
            tr = [y for y in range(T - k, T) if y not in exclude_years]
            f = _fold("B", T, tr)
            f["k"], f["fold_id"] = k, f"B_T{T}_k{k}"
            out.append(f)
    return out


def loyo_fold_specs(years=None, exclude_years=()):
    """Design C (secondary): leave-one-year-out over 2010-2023, as in the pilot."""
    years = [y for y in (C.LOYO_YEARS if years is None else list(years)) if y not in exclude_years]
    C.assert_year_allowed(years)
    return [_fold("C", T, [y for y in years if y != T]) for T in years]


def design_folds(design, exclude_years=(), test_years=None):
    """Fold list of design ``"A"``, ``"B"`` or ``"C"`` in their canonical (warm-start) order."""
    if design == "A":
        return expanding_window_folds(test_years=test_years, exclude_years=exclude_years)
    if design == "B":
        return learning_curve_folds(test_years=test_years or C.LC_TEST_YEARS, exclude_years=exclude_years)
    if design == "C":
        return loyo_fold_specs(years=None, exclude_years=exclude_years) if test_years is None else \
            [f for f in loyo_fold_specs(exclude_years=exclude_years) if f["test_year"] in test_years]
    raise ValueError(f"unknown design {design!r}")


def climate_years(fold):
    """Years whose climate may be used to fit climatologies / ONI standardisation: 2009 + training years."""
    return tuple(sorted(set(C.CLIM_EXTRA_YEARS) | set(fold["train_years"])))


def panel_years(fold, all_years):
    """Panel years a fold needs. A and B: from the year before the first training year
    (its annual total is the previous-year size covariate; it is never a training row)
    up to the test year, so no later year is even present. C: every LOYO year."""
    all_years = [int(y) for y in all_years]
    if fold["design"] == "C":
        return all_years
    lo = min(fold["train_years"]) - 1
    return [y for y in all_years if lo <= y <= fold["test_year"]]


def fold_train_mask(P, train_ok, fold):
    """``train_ok`` restricted to the fold's training years (all other years off)."""
    keep = np.isin(P.years, list(fold["train_years"]))
    assert fold["test_year"] not in fold["train_years"]
    return train_ok & keep[None, :]
