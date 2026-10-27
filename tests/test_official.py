"""Official designs A/B/C: fold construction, 2024 lock, in-fold learning, ONI features, M1 nests B4s."""
import numpy as np
import pandas as pd
import pytest

from conftest import make_panel
from dengue_recon import config as C
from dengue_recon.checks import shuffle as SH
from dengue_recon.checks.leakage import official_leakage_audit
from dengue_recon.checks.zero_climate import zero_climate_maxdiff_embedded
from dengue_recon.data import panel as BD
from dengue_recon.evaluation.cv import (climate_years, expanding_window_folds, fold_train_mask,
                                        learning_curve_folds, loyo_fold_specs, panel_years, row_masks)
from dengue_recon.features.climate import (fold_climate_features, lag_block_features, oni_block_features,
                                           oni_standardise)
from dengue_recon.features.seasonal import offsets
from dengue_recon.features.size_terms import size_covariates
from dengue_recon.models.dirichlet_multinomial import DMModel

YEARS = tuple(range(2010, 2017))


@pytest.fixture(scope="module")
def synth7():
    P, Xc, ey, ew = make_panel(years=YEARS, n_muni=5, seed=1)
    oni = np.sin(np.arange(len(ey)) / 20.0) + np.random.default_rng(2).normal(0, 0.1, len(ey))
    return P, Xc, ey, ew, oni


def _fold_panel(P, fold):
    return P if fold["design"] == "C" else BD.subset_panel(P, panel_years(fold, P.years))


# ------------------------------------------------------------------ fold construction

def test_expanding_window_never_trains_on_test_or_later_years():
    folds = expanding_window_folds()
    assert [f["test_year"] for f in folds] == list(range(2012, 2024))
    for f in folds:
        assert max(f["train_years"]) < f["test_year"]
        assert f["train_years"] == tuple(range(2010, f["test_year"]))
        assert f["k"] == f["test_year"] - 2010
    assert all(f["test_year"] != 2024 for f in folds)


def test_learning_curve_windows_are_the_k_preceding_years():
    folds = learning_curve_folds()
    for T in (2019, 2021, 2023):
        ks = [f["k"] for f in folds if f["test_year"] == T]
        assert ks == list(range(2, T - 2010 + 1))
    for f in folds:
        T, k = f["test_year"], f["k"]
        assert f["train_years"] == tuple(range(T - k, T))
        assert max(f["train_years"]) < T and min(f["train_years"]) >= 2010
    assert len(folds) == 8 + 10 + 12


def test_exclude_year_and_loyo_specs():
    a = expanding_window_folds(exclude_years=(2020,))
    assert 2020 not in [f["test_year"] for f in a]
    assert all(2020 not in f["train_years"] for f in a)
    c = loyo_fold_specs()
    assert len(c) == 14 and all(f["test_year"] not in f["train_years"] for f in c)
    assert all(len(f["train_years"]) == 13 for f in c)
    # climatology years: 2009 + training years (the pilot's set for LOYO)
    f = c[3]
    assert climate_years(f) == tuple(y for y in range(2009, 2024) if y != f["test_year"])


def test_panel_years_have_no_later_year(synth7):
    P = synth7[0]
    for f in expanding_window_folds(test_years=[2012, 2014, 2016]) + \
            learning_curve_folds(test_years=(2015, 2016)):
        yrs = panel_years(f, P.years)
        assert max(yrs) == f["test_year"]
        Pf = _fold_panel(P, f)
        assert Pf.years.max() == f["test_year"]
        tm = fold_train_mask(Pf, row_masks(Pf)[0], f)
        trained = set(Pf.years[tm.any(0)].tolist())
        assert trained == set(f["train_years"])


# ------------------------------------------------------------------ 2024 lock

def test_2024_is_blocked_while_locked(tmp_path, monkeypatch):
    assert C.RUN_FINAL_2024 is False and C.max_loaded_year() == 2023
    with pytest.raises(AssertionError):
        expanding_window_folds(test_years=[2024])
    with pytest.raises(AssertionError):
        learning_curve_folds(test_years=(2024,))
    with pytest.raises(AssertionError):
        C.assert_year_allowed(2024)
    # the loaders never return 2024 rows (filtered while reading)
    yrs = [2022, 2023, 2024, 2025]
    d = pd.DataFrame(dict(geocode=1, name="x", region="South", epi_year=yrs, epi_week=1,
                          week_start=pd.Timestamp("2022-01-02"), n_weeks_year=52, cases=1, year_total=1,
                          exclude_suggested=False))
    c = pd.DataFrame(dict(epi_year=yrs, epi_week=1, week_index=range(4), precip_mm=1.0, temp_mean=1.0,
                          humid_mean=1.0, geocode=1))
    o = pd.DataFrame(dict(epi_year=yrs, epi_week=1, week_index=range(4), oni=0.1))
    d.to_parquet(tmp_path / "dengue_weekly.parquet")
    c.to_parquet(tmp_path / "climate_weekly.parquet")
    c.to_parquet(tmp_path / "climate_weekly_chirps.parquet")
    o.to_parquet(tmp_path / "oni_weekly.parquet")
    pd.DataFrame(dict(geocode=[1])).to_csv(tmp_path / "municipalities.csv", index=False)
    monkeypatch.setattr(C, "DATA_DIR", tmp_path)
    for src in ("power", "chirps"):
        dd, cc, _ = BD.load_tables(src)
        assert dd.epi_year.max() == 2023 and cc.epi_year.max() == 2023
    assert BD.load_oni_weekly().epi_year.max() == 2023


# ------------------------------------------------------------------ in-fold learning (leakage)

def test_climatology_and_oni_standardisation_use_training_years_only(synth7):
    P, Xc, ey, ew, oni = synth7
    f = expanding_window_folds(test_years=[2014])[0]
    Pf = _fold_panel(P, f)
    cy = climate_years(f)
    assert cy == (2009, 2010, 2011, 2012, 2013)
    X1, i1 = fold_climate_features(Pf, Xc, ey, ew, cy, oni=oni)
    later = ey >= 2014                                   # test year and every later year
    X2c, o2 = Xc.copy(), oni.copy()
    X2c[:, :, later] += 50.0
    o2[later] += 7.0
    X2, i2 = fold_climate_features(Pf, X2c, ey, ew, cy, oni=o2)
    assert np.array_equal(i1["beta"], i2["beta"]) and np.array_equal(i1["sd"], i2["sd"])
    assert i1["oni_mean"] == i2["oni_mean"] and i1["oni_sd"] == i2["oni_sd"]
    t_test = list(Pf.years).index(2014)
    assert np.array_equal(X1[:, :t_test], X2[:, :t_test])        # training-year features unchanged
    # control: perturbing a training year does change the climatology and the ONI scaling
    X3c, o3 = Xc.copy(), oni.copy()
    X3c[:, :, ey == 2012] += 50.0
    o3[ey == 2012] += 7.0
    _, i3 = fold_climate_features(Pf, X3c, ey, ew, cy, oni=o3)
    assert not np.allclose(i1["beta"], i3["beta"]) and i1["oni_mean"] != i3["oni_mean"]


def test_offsets_and_centring_ignore_non_training_years(synth7):
    P, Xc, ey, ew, oni = synth7
    for f in learning_curve_folds(test_years=(2016,), k_max=4):
        Pf = _fold_panel(P, f)
        tm = fold_train_mask(Pf, row_masks(Pf)[0], f)
        other = ~np.isin(Pf.years, f["train_years"])
        h1, _, _ = offsets(Pf, tm, True)
        Y = Pf.Yw.copy(); Y[:, other] = np.nan
        h2, _, _ = offsets(Pf, tm, True, Y=Y)
        assert np.array_equal(h1, h2)
        _, c1 = size_covariates(Pf, tm, None, True)
        Pn = BD.subset_panel(Pf, Pf.years); Pn.N = Pf.N.copy(); Pn.N[:, other] = np.nan
        _, c2 = size_covariates(Pn, tm, None, True)
        assert np.array_equal(c1, c2)


def test_official_leakage_audit_passes_in_every_design(synth7):
    P, Xc, ey, ew, oni = synth7
    folds = (expanding_window_folds(test_years=list(range(2012, 2017)))
             + learning_curve_folds(test_years=(2015, 2016))
             + loyo_fold_specs(years=YEARS))
    for f in folds:
        Pf = _fold_panel(P, f)
        tm = fold_train_mask(Pf, row_masks(Pf)[0], f)
        aud = official_leakage_audit(Pf, Xc, ey, ew, tm, f, climate_years(f), oni=oni)
        assert all(aud.values()), (f["fold_id"], aud)
        if f["design"] in ("A", "B"):
            assert "no_future_train" in aud and "no_future_in_panel" in aud


# ------------------------------------------------------------------ ONI features

def test_oni_epiweek_block_features():
    rng = np.random.default_rng(0)
    K = 120
    ey = np.repeat([2009, 2010, 2011], 40)
    oni = rng.normal(0, 1, K)
    o, mu, sd = oni_standardise(oni, ey, (2009, 2010))
    assert np.isclose(mu, oni[:80].mean()) and np.isclose(sd, oni[:80].std())
    o2 = oni.copy(); o2[ey == 2011] = 99.0                      # non-training weeks: no effect
    assert oni_standardise(o2, ey, (2009, 2010))[1:] == (mu, sd)
    F = oni_block_features(o, M=3)
    assert F.shape == (len(C.LAG_BLOCKS), 3, K)
    assert np.array_equal(F[:, 0], F[:, 2], equal_nan=True)                    # national: same for every municipality
    for b, (lo, hi) in enumerate(C.LAG_BLOCKS):
        for k in (hi, 37, 80, K - 1):
            assert np.isclose(F[b, 0, k], o[k - hi:k - lo + 1].mean())   # mean of lags lo..hi weeks
        assert np.isnan(F[b, 0, hi - 1])                         # window starts before the series
    # matches the generic lag-block routine used for the local climate
    assert np.allclose(F[:, :1], lag_block_features(o[None, None]), equal_nan=True)


def test_oni_vector_aligns_by_epi_year_and_week():
    oni = pd.DataFrame(dict(epi_year=[2010, 2009, 2010, 2009], epi_week=[2, 1, 1, 2], oni=[4., 1., 3., 2.]))
    v = BD.oni_vector(oni.sample(frac=1, random_state=0), np.array([2009, 2009, 2010, 2010]),
                      np.array([1, 2, 1, 2]))
    assert v.tolist() == [1., 2., 3., 4.]
    with pytest.raises(AssertionError):
        BD.oni_vector(oni, np.array([2011]), np.array([1]))


def test_m1_has_20_features_with_oni_last(synth7):
    P, Xc, ey, ew, oni = synth7
    f = loyo_fold_specs(years=YEARS)[2]
    X20, _ = fold_climate_features(P, Xc, ey, ew, climate_years(f), oni=oni)
    X15, _ = fold_climate_features(P, Xc, ey, ew, climate_years(f), oni=None)
    assert X20.shape[-1] == 20 and X15.shape[-1] == 15
    assert np.array_equal(X20[..., :15], X15)
    assert np.array_equal(X20[0, ..., 15:], X20[3, ..., 15:])   # ONI identical across municipalities


# ------------------------------------------------------------------ models and shuffles

def test_m1_with_zero_climate_coefficients_equals_b4s(synth7):
    P, Xc, ey, ew, oni = synth7
    f = expanding_window_folds(test_years=[2014])[0]
    Pf = _fold_panel(P, f)
    tm = fold_train_mask(Pf, row_masks(Pf)[0], f)
    h, _, _ = offsets(Pf, tm, True)
    Z, _ = size_covariates(Pf, tm, None, True)
    X20, _ = fold_climate_features(Pf, Xc, ey, ew, climate_years(f), oni=oni)
    M, T = len(Pf.geocodes), len(Pf.years)
    dm15 = DMModel(M, T, 53, 12, 15, Pf.region, Pf.valid_w)
    dm20 = DMModel(M, T, 53, 12, 20, Pf.region, Pf.valid_w)
    flat, _ = dm15.fit(h, Z, np.zeros(X20.shape[:-1] + (15,)), Pf.Yw, Pf.N, tm.astype(float), maxiter=100)
    assert zero_climate_maxdiff_embedded(dm15, flat, dm20, h, Z, X20, Pf.valid_w) == 0.0


def test_fold_shuffles_stay_within_fold_years():
    perms = SH.fold_year_permutations(4, 14, np.arange(14))
    assert np.array_equal(perms, SH.train_test_permutations(4, 14, 1)[0])      # = pilot seed 0
    assert np.array_equal(SH.fold_test_permutation(4, 14, np.arange(14)), SH.test_only_permutation(4, 14))
    idx = np.array([0, 1, 2, 3])                                              # fold years of a 6-year panel
    for p in (SH.fold_year_permutations(5, 6, idx), SH.fold_test_permutation(5, 6, idx)):
        assert (p[:, 4:] == [4, 5]).all()                                    # later years untouched
        for m in range(5):
            assert sorted(p[m, :4]) == [0, 1, 2, 3] and not (p[m, :4] == idx).any()


def test_m2_early_stopping_metric_and_prediction(synth7):
    pytest.importorskip("xgboost")
    import xgboost as xgb
    from dengue_recon.models.xgb_softmax import design, fit_predict_official, group_cross_entropy
    P, Xc, ey, ew, oni = synth7
    f = expanding_window_folds(test_years=[2015])[0]
    Pf = _fold_panel(P, f)
    tok = row_masks(Pf)[0]
    tm = fold_train_mask(Pf, tok, f)
    inner = dict(f, train_years=f["train_years"][:-1])
    tm_in = fold_train_mask(Pf, tok, inner)
    val = tok & (Pf.years == f["train_years"][-1])[None]
    h, _, _ = offsets(Pf, tm, True)
    h_in, _, _ = offsets(Pf, tm_in, True)
    X, _ = fold_climate_features(Pf, Xc, ey, ew, climate_years(f), oni=oni)
    X_in, _ = fold_climate_features(Pf, Xc, ey, ew, climate_years(inner), oni=oni)
    t = list(Pf.years).index(2015)
    rows = [(m, t) for m in range(len(Pf.geocodes))]
    grid = [dict(max_depth=2, eta=0.3)]
    pred, info = fit_predict_official(Pf, True, dict(h=h_in, X=X_in, train_mask=tm_in, val_mask=val),
                                      dict(h=h, X=X, train_mask=tm), rows, grid=grid, max_rounds=40,
                                      early_stop=5, nthread=1)
    n = Pf.nweeks[t]
    assert np.allclose(pred[:, :n].sum(1), 1.0) and np.isnan(pred[:, n:]).all()
    assert info["n_rounds"] == info["trials"][0]["best_iteration"] + 1
    # the early-stopping metric is computed on the full margin (base margin = offset h included)
    va_rows = list(zip(*np.where(val)))
    Fv, bv, pv, gv = design(Pf, h_in, X_in, True, va_rows, tm_in)
    tr_rows = list(zip(*np.where(tm_in)))
    Ft, bt, pt, gt = design(Pf, h_in, X_in, True, tr_rows, tm_in)
    from dengue_recon.models.xgb_softmax import _params, _softmax_obj
    bst = xgb.train(_params(grid[0], C.SEED, 1), xgb.DMatrix(Ft, base_margin=bt), num_boost_round=info["n_rounds"],
                    obj=_softmax_obj(gt, len(tr_rows), pt))
    mv = bst.predict(xgb.DMatrix(Fv, base_margin=bv), output_margin=True)
    assert np.isclose(group_cross_entropy(mv, gv, len(va_rows), pv), info["trials"][0]["best_score"], atol=1e-5)
