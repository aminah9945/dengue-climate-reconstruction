"""Zero-climate equivalence (Step 7, check 3).

M1's prediction function, run with the fitted B4s parameters and theta = delta = 0
(but the real climate features), must reproduce B4s's predictions exactly. This
shows that M1 nests B4s and that any difference comes from the climate terms.
"""
from __future__ import annotations

import numpy as np


def zero_climate_maxdiff(dm, b4s_flat, h, Z, X, zeroX, valid) -> float:
    """Max |pi_M1(B4s params, theta=delta=0, X) - pi_B4s| over valid slots."""
    z = dm.with_params(b4s_flat, theta=0.0, delta=0.0)
    p_eq = dm.predict(z, h, Z, X)
    p_b4s = dm.predict(b4s_flat, h, Z, zeroX)
    return float(np.max(np.abs(p_eq - p_b4s)[:, :, :][np.broadcast_to(valid[None], p_eq.shape)]))


def embed_params(dm_src, flat, dm_dst):
    """Copy a fitted parameter vector of ``dm_src`` into ``dm_dst`` (more climate features):
    every block is kept, extra theta/delta entries are 0. Used to check that M1 (20
    features) with zero climate coefficients reproduces B4s fitted in the 15-feature model."""
    from jax.flatten_util import ravel_pytree
    import jax.numpy as jnp
    p = dm_src.params(flat)
    F = dm_dst.F
    th = np.zeros(F); th[:dm_src.F] = p["theta"]
    de = np.zeros((p["delta"].shape[0], F)); de[:, :dm_src.F] = p["delta"]
    p = dict(p, theta=th, delta=de)
    return np.asarray(ravel_pytree({k: jnp.asarray(v, dtype=jnp.float64) for k, v in p.items()})[0])


def zero_climate_maxdiff_embedded(dm_b4s, b4s_flat, dm_m1, h, Z, X, valid) -> float:
    """Max |pi_M1(B4s params embedded, theta=delta=0, real X) - pi_B4s(zero X)| over valid slots."""
    z = dm_m1.with_params(embed_params(dm_b4s, b4s_flat, dm_m1), theta=0.0, delta=0.0)
    p_eq = dm_m1.predict(z, h, Z, X)
    zeroX = np.zeros(X.shape[:-1] + (dm_b4s.F,))
    p_b4s = dm_b4s.predict(b4s_flat, h, Z, zeroX)
    return float(np.max(np.abs(p_eq - p_b4s)[np.broadcast_to(valid[None], p_eq.shape)]))
