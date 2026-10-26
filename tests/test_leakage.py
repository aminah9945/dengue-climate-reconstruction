"""Leakage: every in-fold statistic must ignore the held-out year."""
import numpy as np

from dengue_recon.checks.leakage import leakage_audit
from dengue_recon.evaluation.cv import loyo_folds, row_masks
from dengue_recon.features.climate import climate_anomalies, climate_features
from dengue_recon.features.seasonal import offsets
from dengue_recon.features.size_terms import size_covariates


def test_climatology_excludes_heldout_year(synth):
    P, Xc, ey, ew = synth
    held = 2012
    train_years = [y for y in range(2009, 2024) if y != held]
    _, b1, s1, _ = climate_anomalies(Xc, ey, ew, train_years)
    X2 = Xc.copy()
    X2[:, :, ey == held] = np.nan                       # blank the held-out year's climate
    _, b2, s2, _ = climate_anomalies(X2, ey, ew, train_years)
    assert np.array_equal(b1, b2) and np.array_equal(s1, s2)
    X3 = Xc.copy()
    X3[:, :, ey == held] += 100.0                       # or make it absurd
    _, b3, s3, _ = climate_anomalies(X3, ey, ew, train_years)
    assert np.array_equal(b1, b3) and np.array_equal(s1, s3)


def test_climatology_would_change_if_year_were_used(synth):
    """Control: including the perturbed year does change the climatology."""
    P, Xc, ey, ew = synth
    X3 = Xc.copy()
    X3[:, :, ey == 2012] += 100.0
    years = list(range(2009, 2024))
    _, b1, _, _ = climate_anomalies(Xc, ey, ew, years)
    _, b3, _, _ = climate_anomalies(X3, ey, ew, years)
    assert not np.allclose(b1, b3)


def test_offsets_and_centring_ignore_heldout_counts(synth):
    P, Xc, ey, ew = synth
    train_ok, _ = row_masks(P)
    for t, y, tm in loyo_folds(P, train_ok, years=[2011, 2013]):
        h1, _, _ = offsets(P, tm, True)
        Y = P.Yw.copy(); Y[:, t] = np.nan
        h2, _, _ = offsets(P, tm, True, Y=Y)
        assert np.array_equal(h1, h2)
        _, c1 = size_covariates(P, tm, None, True)
        N_bak = P.N.copy(); P.N = P.N.copy(); P.N[:, t] = np.nan
        try:
            _, c2 = size_covariates(P, tm, None, True)
        finally:
            P.N = N_bak
        assert np.array_equal(c1, c2)
        assert not tm[:, t].any()


def test_leakage_audit_passes_every_fold(synth):
    P, Xc, ey, ew = synth
    train_ok, _ = row_masks(P)
    for t, y, tm in loyo_folds(P, train_ok):
        aud = leakage_audit(P, Xc, ey, ew, tm, t, weekly=True)
        assert all(aud.values()), (y, aud)


def test_features_are_finite_and_respect_53_week_years(synth):
    P, Xc, ey, ew = synth
    X, info = climate_features(P, Xc, ey, ew, 2014, weekly=True)
    assert X.shape == (len(P.geocodes), len(P.years), 53, 15)
    assert info["n_nan"] == 0
    t14 = list(P.years).index(2014)
    t13 = list(P.years).index(2013)
    assert np.abs(X[:, t14, 52]).sum() > 0                # 2014 has a real week 53
    assert np.abs(X[:, t13, 52]).sum() == 0               # 2013 does not
