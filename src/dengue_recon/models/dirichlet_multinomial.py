"""Penalised-likelihood Dirichlet-multinomial share model (Step 5): B4, B4s, M1 and ablations.

The model ladder (all share one compiled objective)::

    logit[m,t,s] = h[m,s] + g[s] + r[R(m),s] + c[m,s]                        B4
                 + sum_k (gam[k] + gam_reg[R(m),k]) * Z[m,t,s,k]             + size terms -> B4s
                 + sum_f (theta[f] + delta[R(m),f]) * X[m,t,s,f]             + climate    -> M1
    pi = softmax over the valid slots of year t
    y[m,t,:] ~ DirichletMultinomial(N_mt, phi_R(m) * pi)

* ``h`` is the in-fold log historical share (offset, :mod:`dengue_recon.features.seasonal`).
* ``g, r, c`` are national, regional and municipal week deviations (the base
  paper's global -> region -> country hierarchy) with a circular first-difference
  (random-walk) smoothness penalty across weeks at weekly resolution.
* The base paper's year-constant effect is omitted: it cancels in the softmax
  (shown numerically in the reproduction, Step 2).
* B4 passes Z = 0 and X = 0; B4s passes X = 0. Unused coefficients are pulled to
  0 by their penalties and have no effect on predictions.
* Gaussian penalties with fixed scales (``config.PEN``) play the role of the base
  paper's priors. The fit is the penalised maximum (MAP) found with L-BFGS-B
  (SciPy) using JAX gradients in float64 on CPU.

Ablations: ``M1-short`` (lag block 0-3 only), ``M1-mult`` (multinomial
likelihood, phi -> infinity), ``B4s-eps1e-4`` (offset epsilon guard), climate
shuffles (``M1-shufTT-s*``; see :mod:`dengue_recon.checks.shuffle`).
"""
from __future__ import annotations

import time

import numpy as np

from .. import config as C

#: Fitted models that also get probabilistic metrics (500 DM draws).
PROB_MODELS = ["B4", "B4s", "M1"]
#: Main-model set (``--quick`` / 300-municipality run).
MAIN_MODELS = ["B4", "B4s", "M1"]
#: Full pilot set (``n_shuffle`` shuffle seeds are appended by the driver).
PILOT_MODELS = ["B4", "B4s", "M1", "M1-short", "M1-mult", "B4s-eps1e-4"]


def _jax():
    import jax
    jax.config.update("jax_enable_x64", True)
    return jax


def param_template(M, S, R, K, F):
    """Zero-initialised parameter blocks (log phi starts at the penalty mean log 50)."""
    return dict(g=np.zeros(S), r=np.zeros((R, S)), c=np.zeros((M, S)),
                gam=np.zeros(K), gam_reg=np.zeros((R, K)),
                theta=np.zeros(F), delta=np.zeros((R, F)),
                logphi=np.full(R, C.PEN["logphi_mu"]))


class DMModel:
    """Compiled penalised objective for given array shapes and likelihood.

    Parameters
    ----------
    M, T, S, K, F : municipalities, years, slots (53 or 12), size columns (12), climate features
    region : (M,) region index
    valid : (T, S) bool mask of real slots (slot 53 is invalid in 52-week years)
    weekly : add the random-walk smoothness penalty across slots
    likelihood : ``"dm"`` (Dirichlet-multinomial) or ``"mult"`` (multinomial ablation)
    pen : optional overrides of ``config.PEN``
    """

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
        valid = jnp.asarray(valid, dtype=bool)          # (T, S)
        pen = self.pen
        n_size_reg = 8  # the first 8 size columns are region-specific; indicator cols national only
        size_reg_mask = jnp.asarray((np.arange(K) < n_size_reg).astype(float))

        tmpl = param_template(M, S, self.R, K, F)
        flat0, self.unravel = ravel_pytree({k: jnp.asarray(v) for k, v in tmpl.items()})
        self.n_params = flat0.size

        def logits_fn(p, h, Z, X):
            base = h + p["g"][None] + p["r"][reg] + p["c"]                    # (M, S)
            gam = p["gam"][None] + p["gam_reg"][reg] * size_reg_mask[None]   # (M, K)
            th = p["theta"][None] + p["delta"][reg]                           # (M, F)
            lg = base[:, None, :] + jnp.einsum("mtsk,mk->mts", Z, gam) + jnp.einsum("mtsf,mf->mts", X, th)
            return jnp.where(valid[None], lg, -1e30)

        def probs_fn(p, h, Z, X):
            return jax.nn.softmax(logits_fn(p, h, Z, X), axis=-1)

        def rw(x):  # circular first-difference penalty along the slot axis
            return jnp.sum((x - jnp.roll(x, 1, axis=-1)) ** 2)

        def neg_obj(flat, h, Z, X, Y, N, w):
            p = self.unravel(flat)
            pi = probs_fn(p, h, Z, X)
            vm = valid[None].astype(float) * jnp.ones_like(pi)
            if likelihood == "dm":
                phi = jnp.exp(p["logphi"])[reg][:, None]                           # (M, 1)
                a = jnp.where(vm > 0, phi[..., None] * pi, 1.0)
                ll_row = (gammaln(phi) - gammaln(N + phi)
                          + jnp.sum(vm * (gammaln(Y + a) - gammaln(a)), -1))
            else:   # multinomial (phi -> infinity)
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

    def fit(self, h, Z, X, Y, N, w, x0=None, maxiter=2000, gtol=1e-6):
        """Minimise the penalised negative log-likelihood with L-BFGS-B.

        ``w`` (M, T) are row weights (1 = training row, 0 = not used). ``x0`` is a
        warm start (the previous fold's solution of the same model).
        Returns ``(flat_params, info)``.
        """
        from scipy.optimize import minimize
        import jax.numpy as jnp
        args = tuple(jnp.asarray(a, dtype=jnp.float64) for a in (h, Z, X, Y, N, w))

        def fun(x):
            v, g = self._vg(jnp.asarray(x), *args)
            return float(v), np.asarray(g, dtype=float)

        x0 = self.flat0 if x0 is None else x0
        t0 = time.time()
        res = minimize(fun, x0, jac=True, method="L-BFGS-B",
                       options=dict(maxiter=maxiter, maxfun=2 * maxiter, gtol=gtol, ftol=1e-14, maxcor=20))
        g = fun(res.x)[1]
        info = dict(nit=int(res.nit), fun=float(res.fun), grad_norm=float(np.linalg.norm(g)),
                    grad_inf=float(np.max(np.abs(g))), success=bool(res.success),
                    message=str(res.message), fit_s=round(time.time() - t0, 2))
        return res.x, info

    def predict(self, flat, h, Z, X):
        """Predicted shares pi (M, T, S); invalid slots get 0."""
        import jax.numpy as jnp
        return np.asarray(self._probs(jnp.asarray(flat), jnp.asarray(h), jnp.asarray(Z), jnp.asarray(X)))

    def params(self, flat):
        """Unravel a flat parameter vector into named numpy blocks."""
        return {k: np.asarray(v) for k, v in self.unravel(flat).items()}

    def with_params(self, flat, **overrides):
        """Return a flat vector with selected parameter blocks replaced (e.g. theta=0)."""
        from jax.flatten_util import ravel_pytree
        import jax.numpy as jnp
        p = {k: jnp.asarray(v) for k, v in self.params(flat).items()}
        for k, v in overrides.items():
            p[k] = jnp.asarray(v, dtype=jnp.float64) * jnp.ones_like(p[k])
        return np.asarray(ravel_pytree(p)[0])


def variant_inputs(name, h, Z, X, zeroZ, zeroX, dm, mult, fold_ctx):
    """Inputs ``(h, Z, X, model)`` for one rung of the ladder or ablation.

    ``fold_ctx`` provides lazily computed fold-specific alternatives:
    ``h_guard()`` (offset with eps 1e-4), ``X_short()`` (lag block 0-3 only) and
    ``X_shuffled(seed)`` (train-and-test climate shuffle).
    """
    hh, ZZ, XX, mod = h, Z, X, dm
    if name == "B4":
        ZZ, XX = zeroZ, zeroX
    elif name in ("B4s",):
        XX = zeroX
    elif name == "B4s-eps1e-4":
        hh = fold_ctx["h_guard"](); XX = zeroX
    elif name == "M1-short":
        XX = fold_ctx["X_short"]()
    elif name == "M1-mult":
        mod = mult
    elif name.startswith("M1-shufTT"):
        XX = fold_ctx["X_shuffled"](int(name[-1]))
    return hh, ZZ, XX, mod


def draw_counts(pi, phi, N, n_draws, rng, nslots):
    """(n_draws, nslots) counts from Dirichlet(phi*pi) -> Multinomial(N). Every draw sums to N."""
    a = np.clip(phi * pi[:nslots], 1e-12, None)
    g = rng.gamma(a, size=(n_draws, nslots))
    p = g / g.sum(1, keepdims=True)
    bad = ~np.isfinite(p).all(1)
    if bad.any():
        p[bad] = pi[:nslots] / pi[:nslots].sum()
    out = rng.multinomial(int(N), p)
    assert (out.sum(1) == int(N)).all()
    return out
