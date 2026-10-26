import numpy as np

from dengue_recon.checks.sums import allocation_sum_ok, draws_sum_ok
from dengue_recon.evaluation.allocation import allocate_integers
from dengue_recon.models.dirichlet_multinomial import draw_counts


def test_largest_remainder_sums_exactly():
    rng = np.random.default_rng(0)
    for _ in range(500):
        n = rng.choice([12, 52, 53])
        p = rng.dirichlet(np.full(n, rng.uniform(0.1, 5)))
        total = int(rng.integers(0, 200_000))
        a = allocate_integers(p, total)
        assert a.sum() == total
        assert (a >= 0).all()
        assert np.all(np.abs(a - total * p / p.sum()) < 1.0)   # never off by a full case
        assert allocation_sum_ok(p, total)


def test_allocation_edge_cases():
    assert allocate_integers(np.zeros(52), 10).sum() == 10       # all-zero shares -> uniform
    assert allocate_integers(np.ones(53), 0).sum() == 0
    a = allocate_integers(np.array([0.5, 0.5]), 3)                # tie broken stably
    assert a.tolist() == [2, 1]


def test_probabilistic_draws_sum_exactly():
    rng = np.random.default_rng(1)
    p = rng.dirichlet(np.ones(53))
    D = draw_counts(p, 150.0, 1234, 200, rng, 52)
    assert D.shape == (200, 52)
    assert draws_sum_ok(D, 1234)
