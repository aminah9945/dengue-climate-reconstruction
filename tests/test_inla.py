"""EXPLORATORY INLA extension: data export, batches, leakage, scaling, draws (synthetic data).

The R-INLA fit itself is tested only when ``Rscript`` with the INLA package is available.
"""
import json
import shutil
import subprocess

import numpy as np
import pytest

from conftest import make_panel
from dengue_recon import config as C
from dengue_recon.data import panel as BD
from dengue_recon.evaluation.cv import climate_years, expanding_window_folds, fold_train_mask, panel_years, row_masks
from dengue_recon.extensions.inla import driver as D
from dengue_recon.features.climate import fold_climate_features

YEARS = tuple(range(2010, 2016))


@pytest.fixture(scope="module")
def fold_setup():
    P0, Xc, ey, ew = make_panel(years=YEARS, n_muni=10, seed=3)
    oni = np.sin(np.arange(len(ey)) / 20.0)
    fold = [f for f in expanding_window_folds(test_years=[2014])][0]
    P = BD.subset_panel(P0, panel_years(fold, P0.years))
    t = int(np.where(P.years == 2014)[0][0])
    tr, ev = row_masks(P)
    tm = fold_train_mask(P, tr, fold)
    X, _ = fold_climate_features(P, Xc, ey, ew, climate_years(fold), oni=oni)
    return P, t, tm, np.where(ev[:, t])[0], X, fold


def test_fold_panel_has_no_later_year(fold_setup):
    P, t, tm, rows, X, fold = fold_setup
    assert P.years.max() == fold["test_year"] == 2014
    assert not tm[:, t].any()


def test_long_table_test_rows_are_na_and_training_rows_observed(fold_setup):
    P, t, tm, rows, X, fold = fold_setup
    df, (m, tt, s) = D.fold_long_table(P, t, tm, rows, X)
    assert df.loc[df.test == 1, "y"].isna().all()
    assert df.loc[df.test == 0, "y"].notna().all()
    assert set(tt[df.test.values == 0]) <= set(np.where(np.isin(P.years, fold["train_years"]))[0])
    assert (tt[df.test.values == 1] == t).all()
    assert (df.year <= t + 1).all()
    # one row per valid week; continuous week index starts at 1 and steps by 1 within a year
    assert df.tseq.min() == 1
    assert df.slot.max() <= 53 and df.slot.min() == 1
    n_test = int(P.nweeks[t]) * len(rows)
    assert int(df.test.sum()) == n_test
    assert df.shape[1] == 8 + X.shape[-1]


def test_long_table_does_not_depend_on_test_year_counts(fold_setup):
    P, t, tm, rows, X, fold = fold_setup
    a, _ = D.fold_long_table(P, t, tm, rows, X)
    Yb = P.Yw.copy()
    P.Yw = P.Yw.copy(); P.Yw[:, t] = np.random.default_rng(0).integers(0, 999, P.Yw[:, t].shape)
    try:
        b, _ = D.fold_long_table(P, t, tm, rows, X)
    finally:
        P.Yw = Yb
    assert a.equals(b)


def test_batches_cover_all_municipalities_and_are_region_stratified():
    region = np.repeat(np.arange(5), 60)
    bs = D.municipality_batches(region, 60)
    assert len(bs) == 5
    allm = np.sort(np.concatenate(bs))
    assert np.array_equal(allm, np.arange(300))
    for b in bs:
        assert len(b) == 60 and np.array_equal(np.bincount(region[b]), np.full(5, 12))
    assert len(D.municipality_batches(np.repeat(np.arange(5), 12), 60)) == 1


def test_batch_table_renumbers_municipalities(fold_setup):
    P, t, tm, rows, X, fold = fold_setup
    munis = np.array([1, 4, 7])
    df, (m, _, _) = D.fold_long_table(P, t, tm, rows, X, munis=munis)
    assert set(m) <= set(munis)
    assert sorted(df.muni.unique()) == list(range(1, len(np.unique(m)) + 1))


def test_rescaled_draws_sum_exactly_to_total():
    rng = np.random.default_rng(1)
    eta = rng.normal(2, 1, (52, 200))
    size = np.full(200, 30.0)
    Y, mu = D.nb_draws(eta, size, rng)
    for N in (100, 1234, 7):
        S = D.rescale_draws(Y, mu, N)
        assert (S.sum(0) == N).all() and (S >= 0).all()


def test_nb_draws_mean_and_variance():
    rng = np.random.default_rng(2)
    eta = np.full((1, 200000), np.log(20.0))
    Y, _ = D.nb_draws(eta, np.full(200000, 5.0), rng)
    assert abs(Y.mean() - 20) < 0.2
    assert abs(Y.var() - (20 + 20 ** 2 / 5)) < 3


def test_prob_scores_match_definitions():
    rng = np.random.default_rng(3)
    S = rng.poisson(10, (52, 500)).astype(float)
    y = np.full(52, 10.0)
    r = D.prob_scores(y, S)
    assert 0.9 < r["cov80"] <= 1 and r["cov95"] >= r["cov80"] and r["crps"] > 0


def test_design_a_never_includes_2024():
    assert not C.RUN_FINAL_2024
    assert all(f["test_year"] <= 2023 for f in expanding_window_folds())
    with pytest.raises(AssertionError):
        expanding_window_folds(test_years=[2024])


def _have_inla():
    if shutil.which("Rscript") is None:
        return False
    r = subprocess.run(["Rscript", "-e", "quit(status = !requireNamespace('INLA', quietly = TRUE))"],
                       capture_output=True)
    return r.returncode == 0


@pytest.mark.skipif(not _have_inla(), reason="R-INLA not installed")
def test_r_fit_runs_and_writes_outputs(fold_setup, tmp_path):
    P, t, tm, rows, X, fold = fold_setup
    df, _ = D.fold_long_table(P, t, tm, rows, X)
    df.to_csv(tmp_path / "data.csv", index=False, na_rep="NA", float_format="%.6g")
    info = D.run_r(tmp_path, "base", 20, 1, 1, tmp_path / "R.log")
    assert info["ok"] and info["n_test"] == int(df.test.sum())
    eta, size = D.read_draws(tmp_path, "base", info["n_test"])
    assert eta.shape == (info["n_test"], 20) and (size > 0).all()
    assert (tmp_path / "base_theta.txt").exists()
    # warm start from the saved mode
    info2 = D.run_r(tmp_path, "clim20", 0, 1, 1, tmp_path / "R.log", tmp_path / "base_theta.txt")
    assert info2["warm_start"] is True
    json.loads((tmp_path / "clim20_done.json").read_text())
