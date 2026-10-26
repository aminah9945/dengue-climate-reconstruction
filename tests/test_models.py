"""Dirichlet-multinomial ladder: softmax cancellation, zero-climate equivalence, 2024 lock."""
import numpy as np
import pytest

from dengue_recon import config as C
from dengue_recon.base_paper.lim_dm import cancellation, jax_softmax
from dengue_recon.checks import shuffle as SH
from dengue_recon.checks.sanity import final_path_unit_test
from dengue_recon.checks.zero_climate import zero_climate_maxdiff
from dengue_recon.evaluation.cv import final_test_split, loyo_folds, row_masks
from dengue_recon.features.climate import climate_features
from dengue_recon.features.seasonal import offsets
from dengue_recon.features.size_terms import size_covariates
from dengue_recon.models.dirichlet_multinomial import DMModel


def test_year_effect_cancels_in_softmax():
    """A year-constant logit shift (the base paper's country-year effect) cannot change shares."""
    import jax
    jax.config.update("jax_enable_x64", True)
    rng = np.random.default_rng(0)
    n_cty, n_years = 6, 5
    seasonal = rng.normal(0, 1, (n_cty, 12))
    year_effect = rng.normal(0, 2, (n_cty, n_years))
    pi = np.asarray(jax_softmax(seasonal[:, None, :] + year_effect[:, :, None]))
    shares = pi.reshape(-1, 12)
    cidx = np.repeat(np.arange(n_cty), n_years)
    rows = cancellation(shares, cidx)
    assert len(rows) == n_cty
    assert max(r["max_abs_diff"] for r in rows) < 1e-12
    # a month-varying term (like climate) does change the shape between years
    x = rng.normal(0, 1, (n_cty, n_years, 12))
    pi2 = np.asarray(jax_softmax(seasonal[:, None, :] + year_effect[:, :, None] + 0.3 * x)).reshape(-1, 12)
    assert max(r["max_abs_diff"] for r in cancellation(pi2, cidx)) > 1e-3


@pytest.fixture(scope="module")
def fold(synth):
    P, Xc, ey, ew = synth
    train_ok, _ = row_masks(P)
    t, y, tm = next(loyo_folds(P, train_ok, years=[2012]))
    h, _, _ = offsets(P, tm, True)
    Z, _ = size_covariates(P, tm, None, True)
    X, _ = climate_features(P, Xc, ey, ew, y, True)
    M, T = len(P.geocodes), len(P.years)
    dm = DMModel(M, T, 53, 12, 15, P.region, P.valid_w, weekly=True)
    return P, t, tm, h, Z, X, dm


def test_zero_climate_m1_equals_b4s(fold):
    P, t, tm, h, Z, X, dm = fold
    zeroX = np.zeros_like(X)
    flat, info = dm.fit(h, Z, zeroX, P.Yw, P.N, tm.astype(float), maxiter=200)
    assert np.isfinite(info["fun"])
    assert zero_climate_maxdiff(dm, flat, h, Z, X, zeroX, P.valid_w) == 0.0
    pi = dm.predict(flat, h, Z, zeroX)
    assert np.allclose(pi.sum(-1), 1.0)
    assert np.all(pi[:, ~P.valid_w] < 1e-300)           # masked slot 53 in 52-week years
    # a non-zero theta changes the prediction
    moved = dm.predict(dm.with_params(flat, theta=0.2), h, Z, X)
    assert np.max(np.abs(moved - pi)) > 1e-4


def test_shuffles_are_derangements():
    rng = np.random.default_rng(3)
    for n in (2, 5, 14):
        p = SH.derangement(n, rng)
        assert sorted(p) == list(range(n)) and not (p == np.arange(n)).any()
    perms = SH.train_test_permutations(4, 14, 3)
    assert len(perms) == 3 and all(p.shape == (4, 14) for p in perms)
    X = np.arange(4 * 14).reshape(4, 14, 1, 1).astype(float)
    Xs = SH.shuffle_years(X, perms[0])
    for m in range(4):                                    # same years, different order
        assert sorted(Xs[m].ravel()) == sorted(X[m].ravel())
        assert not np.array_equal(Xs[m], X[m])
    tp = SH.test_only_permutation(4, 14)
    Xt = SH.shuffle_test_year(X, 5, tp)
    assert np.array_equal(np.delete(Xt, 5, 1), np.delete(X, 5, 1))


def test_final_year_lock_and_path():
    assert C.RUN_FINAL_2024 is False
    tr, te = final_test_split(np.arange(2010, 2025), 2024)
    assert tr.sum() == 14 and te.sum() == 1 and te[-1]
    ok, dev = final_path_unit_test()                     # dummy-year unit test of the 2024 path
    assert ok, dev
