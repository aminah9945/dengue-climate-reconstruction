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
