import numpy as np
import pytest

from dengue_recon.evaluation.bootstrap import cluster_bootstrap
from dengue_recon.evaluation.metrics import (crps_samples, emd_weeks, jsd, jsd_floor, mae_shares,
                                             peak_week_error, week50)


def test_jsd_properties():
    rng = np.random.default_rng(0)
    for _ in range(200):
        p, q = rng.dirichlet(np.ones(52)), rng.dirichlet(np.ones(52))
        d = jsd(p, q)
        assert 0.0 <= d <= 1.0
        assert d == pytest.approx(jsd(q, p), abs=1e-15)           # symmetric
        assert jsd(p, p) == pytest.approx(0.0, abs=1e-15)          # identity
    a = np.r_[np.ones(26), np.zeros(26)]
    b = np.r_[np.zeros(26), np.ones(26)]
    assert jsd(a, b) == pytest.approx(1.0)                         # disjoint -> 1 (base 2)
    assert np.isfinite(jsd(a, np.ones(52)))                        # finite with zero weeks
    assert jsd(10 * a, a) == pytest.approx(0.0, abs=1e-15)         # scale invariant (shares)


def test_jsd_floor_shrinks_with_total():
    rng = np.random.default_rng(0)
    p = rng.dirichlet(np.ones(52) * 2)
    assert jsd_floor(p, 200, rng) > jsd_floor(p, 20_000, rng)


def test_peak_emd_week50_mae():
    p = np.zeros(52); p[0] = 1
    q = np.zeros(52); q[51] = 1
    assert peak_week_error(p, q) == 1                              # circular
    assert peak_week_error(p, q, circular=False) == 51
    assert emd_weeks(p, p) == 0
    assert emd_weeks(p, q) == pytest.approx(51.0)
    r = np.zeros(52); r[9] = 1
    assert week50(r) == 10
    assert mae_shares(p, p) == 0


def test_crps_and_bootstrap():
    rng = np.random.default_rng(0)
    y = rng.poisson(20, 30).astype(float)
    S = rng.poisson(20, (30, 400)).astype(float)
    c = crps_samples(y, S)
    assert (c >= -1e-12).all()
    assert crps_samples(y, S + 50).mean() > c.mean()               # biased forecast scores worse
    diff = rng.normal(-0.01, 0.01, 300)
    mean, lo, hi = cluster_bootstrap(diff, np.repeat(np.arange(30), 10), 500, 1)
    assert lo <= mean <= hi
