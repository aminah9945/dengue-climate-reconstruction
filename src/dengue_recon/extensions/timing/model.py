"""Model pieces of the timing-informed share model (EXPLORATORY).

* The S variants (S1, S2, S3, S3C) need no new model code: their feature blocks
  are passed as the ``X`` design of the existing penalised Dirichlet-multinomial
  (:class:`dengue_recon.models.dirichlet_multinomial.DMModel`), so they get a
  national coefficient ``theta`` plus a shrunken regional deviation ``delta`` per
  feature with the climate penalties (``config.PEN['theta']``, ``['delta']``).
* T1 needs a *year-specific* offset h[m, t, s] (the seasonal template rolled by
  the estimated shift). :class:`YearOffsetDM` is the same penalised objective with
  that one change; with h[m, t, s] = h[m, s] it is identical to ``DMModel``
  (checked in ``tests/test_timing.py``).
"""
from __future__ import annotations

import numpy as np

from ... import config as C
from ...models.dirichlet_multinomial import DMModel, _jax, param_template


class YearOffsetDM(DMModel):
    """``DMModel`` whose offset ``h`` has shape (M, T, S) instead of (M, S)."""

    def __init__(self, M, T, S, K, F, region, valid, weekly=True, likelihood="dm", pen=None):
        jax = _jax()
        import jax.numpy as jnp
        from jax.flatten_util import ravel_pytree
        from jax.scipy.special import gammaln
        self.pen = dict(C.PEN, **(pen or {}))
        self.M, self.T, self.S, self.K, self.F, self.R = M, T, S, K, F, len(C.REGIONS)
        self.weekly, self.likelihood = weekly, likelihood
        self.region = np.asarray(region)
        reg = jnp.asarray(self.region)
        valid = jnp.asarray(valid, dtype=bool)
        pen = self.pen
        size_reg_mask = jnp.asarray((np.arange(K) < 8).astype(float))
        tmpl = param_template(M, S, self.R, K, F)
        flat0, self.unravel = ravel_pytree({k: jnp.asarray(v) for k, v in tmpl.items()})
        self.n_params = flat0.size

        def logits_fn(p, h, Z, X):
            base = p["g"][None] + p["r"][reg] + p["c"]                         # (M, S)
            gam = p["gam"][None] + p["gam_reg"][reg] * size_reg_mask[None]
            th = p["theta"][None] + p["delta"][reg]
            lg = h + base[:, None, :] + jnp.einsum("mtsk,mk->mts", Z, gam) + jnp.einsum("mtsf,mf->mts", X, th)
            return jnp.where(valid[None], lg, -1e30)

        def probs_fn(p, h, Z, X):
            return jax.nn.softmax(logits_fn(p, h, Z, X), axis=-1)

        def rw(x):
            return jnp.sum((x - jnp.roll(x, 1, axis=-1)) ** 2)

        def neg_obj(flat, h, Z, X, Y, N, w):
            p = self.unravel(flat)
            pi = probs_fn(p, h, Z, X)
            vm = valid[None].astype(float) * jnp.ones_like(pi)
            if likelihood == "dm":
                phi = jnp.exp(p["logphi"])[reg][:, None]
                a = jnp.where(vm > 0, phi[..., None] * pi, 1.0)
                ll_row = (gammaln(phi) - gammaln(N + phi)
                          + jnp.sum(vm * (gammaln(Y + a) - gammaln(a)), -1))
            else:
                ll_row = jnp.sum(vm * Y * jnp.log(jnp.where(vm > 0, pi, 1.0) + 1e-300), -1)
            ll = jnp.sum(w * ll_row)
            lp = (-0.5 * jnp.sum((p["g"] / pen["g"]) ** 2) - 0.5 * jnp.sum((p["r"] / pen["r"]) ** 2)
                  - 0.5 * jnp.sum((p["c"] / pen["c"]) ** 2)
                  - 0.5 * jnp.sum((p["gam"] / pen["gamma_nat"]) ** 2)
                  - 0.5 * jnp.sum((p["gam_reg"] / pen["gamma_reg"]) ** 2)
                  - 0.5 * jnp.sum((p["theta"] / pen["theta"]) ** 2)
                  - 0.5 * jnp.sum((p["delta"] / pen["delta"]) ** 2)
                  - 0.5 * jnp.sum(((p["logphi"] - pen["logphi_mu"]) / pen["logphi_sd"]) ** 2))
            if weekly:
                lam = 1.0 / pen["rw"] ** 2
                lp = lp - 0.5 * lam * (rw(p["g"]) + rw(p["r"]) + rw(p["c"]))
            return -(ll + lp)

        self._vg = jax.jit(jax.value_and_grad(neg_obj))
        self._probs = jax.jit(lambda flat, h, Z, X: probs_fn(self.unravel(flat), h, Z, X))
        self.flat0 = np.asarray(flat0)


def transfer_params(dm_src, flat, dm_dst, zero_new=True):
    """Copy a fitted parameter vector into a model with another number of features
    ``F``: all blocks are kept, ``theta``/``delta`` become zeros of the new shape.

    Used for the zero-coefficient equivalence check (S model with its new
    coefficients at 0 must reproduce B4s) and for warm starts.
    """
    from jax.flatten_util import ravel_pytree
    import jax.numpy as jnp
    p = dm_src.params(flat)
    th = np.zeros(dm_dst.F)
    de = np.zeros((p["delta"].shape[0], dm_dst.F))
    if not zero_new:
        k = min(dm_src.F, dm_dst.F)
        th[:k], de[:, :k] = p["theta"][:k], p["delta"][:, :k]
    p = dict(p, theta=th, delta=de)
    return np.asarray(ravel_pytree({k: jnp.asarray(v, dtype=jnp.float64) for k, v in p.items()})[0])


def zero_coef_maxdiff(dm_b4s, b4s_flat, dm_new, h, Z, X_new, valid, h_new=None):
    """Max |pi_new(B4s parameters, new coefficients = 0, real new features) - pi_B4s|
    over valid slots. ``h_new`` (M, T, S) is the year-specific offset of a
    :class:`YearOffsetDM` (for T1 with all shifts 0 it equals h)."""
    z = transfer_params(dm_b4s, b4s_flat, dm_new)
    p_new = dm_new.predict(z, h if h_new is None else h_new, Z, X_new)
    zeroX = np.zeros(X_new.shape[:-1] + (dm_b4s.F,))
    p_b4s = dm_b4s.predict(b4s_flat, h, Z, zeroX)
    return float(np.max(np.abs(p_new - p_b4s)[np.broadcast_to(valid[None], p_new.shape)]))
