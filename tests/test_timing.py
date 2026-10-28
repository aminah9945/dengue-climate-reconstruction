"""Timing-informed share model (exploratory extension): signals, features, model, checks.

Synthetic data only (no downloads needed)."""
import numpy as np
import pandas as pd
import pytest

from dengue_recon import config as C
from dengue_recon.checks import shuffle as SH
from dengue_recon.evaluation.allocation import allocate_integers
from dengue_recon.evaluation.cv import row_masks
from dengue_recon.extensions.timing import features as TF
from dengue_recon.extensions.timing import signals as SG
from dengue_recon.extensions.timing.model import YearOffsetDM, transfer_params, zero_coef_maxdiff
from dengue_recon.extensions.timing.run import onset_week
from dengue_recon.features.seasonal import offsets
from dengue_recon.features.size_terms import size_covariates
from dengue_recon.models.dirichlet_multinomial import DMModel


# ------------------------------------------------------------------ signals
def test_raw_csv_line_filter_drops_locked_years(tmp_path):
    """Rows after the max loaded year never reach the parser (files are newest first)."""
    f = tmp_path / "1234567.csv"
    f.write_text("data_iniSE,SE,casos,versao_modelo\n"
                 "2025-01-05,202501,999,2025-01-10\n2024-01-07,202401,888,2024-01-10\n"
                 "2023-12-24,202352,5,2024-01-02\n2023-12-24,202352,7,2024-02-02\n2010-01-03,201001,3,2010-01-08\n")
    d = SG.read_infodengue_csv(f, max_year=2023)
    assert d.epi_year.max() == 2023 and 888 not in d.casos.values and 999 not in d.casos.values
    assert d.set_index("SE").casos[202352] == 7          # latest model version kept


def _toy_geos():
    geos = [3100001, 3100002, 3100003, 3100004, 3500001, 3500002]
    ufs = [SG.UF_CODE[g // 100000] for g in geos]
    return geos, ufs


def test_pools_never_contain_target():
    geos, ufs = _toy_geos()
    study = [3100001, 3100002, 3500001]
    for v in ("all", "nonstudy", "p25", "p50"):
        pm = SG.pools(study, geos, ufs, v, study_set=study)
        for g in study:
            assert g not in pm[g]
            assert all(SG.UF_CODE[x // 100000] == SG.UF_CODE[g // 100000] for x in pm[g])
    pm = SG.pools(study, geos, ufs, "nonstudy", study_set=study)
    assert set(pm[3100001]) == {3100003, 3100004} and set(pm[3500001]) == {3500002}
    pa = SG.pools(study, geos, ufs, "all")
    assert set(pa[3100001]) == {3100002, 3100003, 3100004}


def test_pool_counts_independent_of_target():
    geos, ufs = _toy_geos()
    rng = np.random.default_rng(0)
    A = rng.integers(0, 50, (len(geos), 2, 53)).astype(float)
    study = [3100001, 3500001]
    pm = SG.pools(study, geos, ufs, "all")
    c1, size = SG.pool_counts(A, geos, pm, study)
    A2 = A.copy(); A2[0] = 1e9; A2[4] = 1e9            # the two targets' own counts
    c2, _ = SG.pool_counts(A2, geos, pm, study)
    assert np.array_equal(c1, c2) and list(size) == [3, 1]
    assert np.array_equal(c1[0], A[1:4].sum(0))


def test_fraction_pools_fixed_and_nested_size():
    geos = [3100000 + i for i in range(1, 21)]
    ufs = ["MG"] * 20
    p25 = SG.fraction_pools(geos[:3], geos, ufs, 0.25)
    p25b = SG.fraction_pools(geos[:3], geos, ufs, 0.25)
    p50 = SG.fraction_pools(geos[:3], geos, ufs, 0.50)
    for g in geos[:3]:
        assert np.array_equal(p25[g], p25b[g])              # fixed seed
        assert len(p25[g]) == 5 and len(p50[g]) == 10 and g not in p50[g]


# ------------------------------------------------------------------ features
def test_smooth_and_lags_stay_within_year():
    v = np.array([3.0, 0.0, 3.0, 6.0])
    assert np.allclose(TF.smooth3(v), [1.5, 2.0, 3.0, 4.5])
    A = np.zeros((1, 2, 53))
    A[0, 0, :52] = np.arange(52)
    A[0, 1, :53] = 100 + np.arange(53)
    X = TF.lagged(A, np.array([52, 53]), (-2, 0, 2), sign=-1)
    assert X[0, 0, 0, 2] == 0 and X[0, 0, 5, 2] == 3       # lag 2: w - 2, clipped at week 1
    assert X[0, 0, 51, 0] == 51                            # lead 2 clipped at the last week of the year
    assert X[0, 1, 0, 2] == 100                            # never reaches back into the previous year
    assert (X[0, 0, 52] == 0).all()                        # slot 53 of a 52-week year stays 0


def test_anomaly_climatology_uses_training_years_only(synth):
    P = synth[0]
    rng = np.random.default_rng(1)
    V = rng.gamma(2.0, 10.0, P.Yw.shape)
    tym = np.isin(P.years, [2010, 2011, 2012])
    A1, i1 = TF.anomalies(V, P.nweeks, tym)
    V2 = V.copy(); V2[:, ~tym] = rng.gamma(2.0, 999.0, V2[:, ~tym].shape)
    A2, i2 = TF.anomalies(V2, P.nweeks, tym)
    assert np.array_equal(i1["clim"], i2["clim"])
    assert np.array_equal(A1[:, tym], A2[:, tym])
    assert not np.array_equal(A1[:, ~tym], A2[:, ~tym])   # same-year signal does enter the test year


def test_missing_signal_gives_constant_feature(synth):
    P = synth[0]
    V = np.ones(P.Yw.shape)                              # year totals 52-53 >= MIN_SIGNAL_TOTAL
    V[0, 1] = 0.0                                        # municipality 0, year 1: no curve
    tym = np.ones(len(P.years), bool)
    A, info = TF.anomalies(V, P.nweeks, tym)
    assert not info["ok"][0, 1] and (A[0, 1] == 0).all()


def test_best_shift_recovers_known_shift():
    w = np.arange(52)
    tmpl = np.exp(-0.5 * ((w - 15) / 4.0) ** 2) + 0.01
    for d in (-7, 0, 4, 10):
        cur = np.roll(tmpl, d)
        assert TF.best_shift(cur, tmpl)[0] == d
    flat = np.ones(52)
    assert TF.best_shift(flat, tmpl)[0] == 0             # undefined correlation -> no shift


def test_shifted_offsets_zero_is_identity():
    rng = np.random.default_rng(0)
    h = rng.normal(size=(3, 53))
    D = np.zeros((3, 4), int)
    H = TF.shifted_offsets(h, D, np.array([52, 53, 52, 52]))
    assert np.array_equal(H, np.broadcast_to(h[:, None], H.shape))
    D[1, 2] = 3
    H = TF.shifted_offsets(h, D, np.array([52, 53, 52, 52]))
    assert np.array_equal(H[1, 2, :52], np.roll(h[1, :52], 3))


def test_standardise_uses_training_rows():
    rng = np.random.default_rng(0)
    X = rng.normal(2.0, 3.0, (4, 3, 53, 2))
    tm = np.zeros((4, 3), bool); tm[:, :2] = True
    valid = np.ones((3, 53), bool); valid[0, 52] = False
    Xs, mu, sd = TF.standardise(X, tm, valid)
    sel = tm[:, :, None] & valid[None]
    assert np.allclose(Xs[sel].mean(0), 0) and np.allclose(Xs[sel].std(0), 1)
    assert (Xs[:, 0, 52] == 0).all()


def test_onset_week():
    p = np.zeros(52); p[[4, 5, 6]] = [0.05, 0.05, 0.9]
    assert onset_week(p) == 6                            # cumulative 10 % reached in week 6


# ------------------------------------------------------------------ model
@pytest.fixture(scope="module")
def fitted(synth):
    P = synth[0]
    train_ok, _ = row_masks(P)
    tm = train_ok.copy(); tm[:, -1] = False
    h, _, _ = offsets(P, tm, True)
    Z, _ = size_covariates(P, tm, None, True)
    M, T = len(P.geocodes), len(P.years)
    zero15 = np.zeros((M, T, 53, 15))
    dm = DMModel(M, T, 53, 12, 15, P.region, P.valid_w, True, "dm")
    flat, _ = dm.fit(h, Z, zero15, P.Yw, P.N, tm.astype(float), maxiter=300)
    return P, tm, h, Z, dm, flat


def test_zero_coefficient_equivalence_S_and_T1(fitted):
    """S model with new coefficients 0 = B4s; T1 with all shifts 0 = B4s."""
    P, tm, h, Z, dm, flat = fitted
    M, T = len(P.geocodes), len(P.years)
    rng = np.random.default_rng(3)
    X5 = rng.normal(size=(M, T, 53, 5))
    dmS = DMModel(M, T, 53, 12, 5, P.region, P.valid_w, True, "dm")
    assert zero_coef_maxdiff(dm, flat, dmS, h, Z, X5, P.valid_w) < 1e-14
    dmT = YearOffsetDM(M, T, 53, 12, 1, P.region, P.valid_w, True, "dm")
    H0 = TF.shifted_offsets(h, np.zeros((M, T), int), P.nweeks)
    assert zero_coef_maxdiff(dm, flat, dmT, h, Z, np.zeros((M, T, 53, 1)), P.valid_w, h_new=H0) < 1e-14


def test_year_offset_dm_objective_matches_dm(fitted):
    """YearOffsetDM with h[m,t,s] = h[m,s] has the same objective and gradient as DMModel."""
    import jax.numpy as jnp
    P, tm, h, Z, dm, flat = fitted
    M, T = len(P.geocodes), len(P.years)
    X = np.zeros((M, T, 53, 15))
    dmT = YearOffsetDM(M, T, 53, 12, 15, P.region, P.valid_w, True, "dm")
    H = np.broadcast_to(h[:, None], (M, T, 53)).copy()
    args = [jnp.asarray(a, dtype=jnp.float64) for a in (Z, X, P.Yw, P.N, tm.astype(float))]
    v1, g1 = dm._vg(jnp.asarray(flat), jnp.asarray(h), *args)
    v2, g2 = dmT._vg(jnp.asarray(flat), jnp.asarray(H), *args)
    assert abs(float(v1) - float(v2)) < 1e-8 * abs(float(v1)) and np.allclose(g1, g2, atol=1e-8)


def test_transfer_params_keeps_blocks(fitted):
    P, tm, h, Z, dm, flat = fitted
    dm9 = DMModel(len(P.geocodes), len(P.years), 53, 12, 9, P.region, P.valid_w, True, "dm")
    z = dm9.params(transfer_params(dm, flat, dm9))
    p = dm.params(flat)
    assert np.array_equal(z["c"], p["c"]) and z["theta"].shape == (9,) and not z["theta"].any()


def test_exact_sums_after_allocation(fitted):
    P, tm, h, Z, dm, flat = fitted
    pi = dm.predict(flat, h, Z, np.zeros((len(P.geocodes), len(P.years), 53, 15)))
    t = len(P.years) - 1
    n = int(P.nweeks[t])
    for m in range(len(P.geocodes)):
        assert allocate_integers(pi[m, t, :n], int(P.N[m, t])).sum() == int(P.N[m, t])


def test_wrong_year_shuffle_moves_whole_years():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(4, 5, 53, 3))
    perm = SH.fold_year_permutations(4, 5, np.arange(5), seed=1)
    Xs = SH.shuffle_years(X, perm)
    for m in range(4):
        for t in range(5):
            assert perm[m, t] != t and np.array_equal(Xs[m, t], X[m, perm[m, t]])


def test_2024_lock_is_on():
    assert C.RUN_FINAL_2024 is False and C.max_loaded_year() == 2023
