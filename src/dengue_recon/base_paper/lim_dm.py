"""Step 2: re-implementation of the Lim et al. (2026) Dirichlet-multinomial
annual -> monthly disaggregation model (DM_v3_pooled) and its CV scoring.

This file is OUR re-implementation, written after reading
script/CV/03_down_DM_full_repeatedCV_v2.py of
https://github.com/ahyoung-lim/OD_gap_filling_public (commit 00ff7ae). No code
from that repository is executed or redistributed; only its CV mask file (data)
is read.

Two fits of the same log posterior:
* NUTS (NumPyro), faithful Bayesian fit (``fit_nuts``);
* MAP (L-BFGS on the non-centred posterior with JAX gradients; ``fit_map``).

Model (country c, year y, month j):
  logit[c,y,j] = props_log[c,j] + c_month[c,j] + year_effect[c,y]
  c_month      = r_month[region(c), j] + tau_country_month * z_c[c, j]
  r_month      = g_month[j] + sigma_region_month[r, j] * z_r[r, j]
  year_effect  = (year_effect_raw @ chol(AR1(0.7)).T) * tau_region_year[region(c)]
  pi           = softmax_j(logit)
  alpha        = max(pi, 1e-6) * max(exp(log_conc[c, y]), 1)
  y[c,y,:] ~ DirichletMultinomial(annual_total, alpha)    (training rows only)
"""
from __future__ import annotations

import lzma
import os
import tempfile
import time

import numpy as np
import pandas as pd

PER = 1e5
POP_FLOOR = 1.0
MIN_DRAWS, MIN_PROP = 20, 0.1
PRIORS = dict(g_month=0.3, sigma_region_month=0.4, tau_country_month=0.15,
              tau_region_year=0.15, log_conc_mu=1.6, log_conc_sd=0.35)
RHO = 0.7


# ------------------------------------------------------------------ data
def read_mask(path) -> pd.DataFrame:
    """Read the base paper's CV mask (.rds) with pyreadr (handles XZ-compressed files)."""
    import pyreadr
    try:
        return next(iter(pyreadr.read_r(str(path)).values()))
    except Exception:
        with open(path, "rb") as f:
            head = f.read(6)
        if head != b"\xfd7zXZ\x00":
            raise
        with lzma.open(path, "rb") as f:
            data = f.read()
        with tempfile.NamedTemporaryFile(suffix=".rds", delete=False) as tmp:
            tmp.write(data)
        try:
            return next(iter(pyreadr.read_r(tmp.name).values()))
        finally:
            os.unlink(tmp.name)


def build_arrays(df_long: pd.DataFrame, fold: int, min_total_train: int = 1,
                 min_total_test: int = 1) -> dict:
    """Train/test definition copied from build_dm_arrays_from_mask."""
    df = df_long.copy()
    df["month"] = pd.to_numeric(df["month"], errors="coerce")
    df["Year"] = pd.to_numeric(df["Year"].astype(str), errors="coerce")
    df = df.dropna(subset=["adm_0_name", "Year", "month"]).copy()
    df["month"] = df["month"].astype(int)
    df["Year"] = df["Year"].astype(int)
    df["region"] = df["region"].astype(str)
    fold_num = pd.to_numeric(df["fold"], errors="coerce")
    df["is_test"] = fold_num == fold
    yr = df.groupby(["adm_0_name", "Year"], as_index=False).agg(
        n_months=("month", "nunique"), any_test=("is_test", "any"),
        annual_total=("annual_total", "first"))
    yr = yr[yr.n_months == 12]
    train_pairs = yr[(~yr.any_test) & (yr.annual_total >= min_total_train)]
    test_pairs = yr[(yr.any_test) & (yr.annual_total >= min_total_test)]
    pivot = (df.pivot_table(index=["adm_0_name", "Year"], columns="month", values="dengue_total",
                            aggfunc="sum").sort_index().reindex(columns=range(1, 13)).fillna(0))
    pop = (df.pivot_table(index=["adm_0_name", "Year"], columns="month", values="pop_est",
                          aggfunc="first").sort_index().reindex(columns=range(1, 13)))
    keep = pd.MultiIndex.from_frame(pd.concat([train_pairs[["adm_0_name", "Year"]],
                                               test_pairs[["adm_0_name", "Year"]]]))
    pivot, pop = pivot.loc[keep], pop.loc[keep]
    countries = sorted(pivot.index.get_level_values(0).unique())
    years = sorted(pivot.index.get_level_values(1).unique())
    cty_region = df[df.adm_0_name.isin(countries)].groupby("adm_0_name")["region"].agg(
        lambda s: s.mode().iat[0])
    regions = sorted(cty_region.unique())
    ci = {c: i for i, c in enumerate(countries)}
    yi = {y: i for i, y in enumerate(years)}
    ri = {r: i for i, r in enumerate(regions)}
    tr = list(zip(train_pairs.adm_0_name, train_pairs.Year))
    te = list(zip(test_pairs.adm_0_name, test_pairs.Year))
    fit_counts = np.stack([pivot.loc[k].values for k in tr]).astype(np.int64)
    test_counts = np.stack([pivot.loc[k].values for k in te]).astype(np.int64)
    props = pd.DataFrame(fit_counts, index=pd.MultiIndex.from_tuples(tr))
    props = props.div(props.sum(1).replace(0, 1), axis=0)
    props = props.groupby(level=0).mean().reindex(countries).fillna(1 / 12).values
    return dict(
        countries=countries, years=years, regions=regions,
        region_idx=np.array([ri[cty_region[c]] for c in countries]),
        props_log=np.log(props + 1e-6),
        fit_counts=fit_counts, fit_totals=fit_counts.sum(1),
        fit_cidx=np.array([ci[c] for c, _ in tr]), fit_yidx=np.array([yi[y] for _, y in tr]),
        test_counts=test_counts, test_totals=test_counts.sum(1),
        test_cidx=np.array([ci[c] for c, _ in te]), test_yidx=np.array([yi[y] for _, y in te]),
        test_pop=np.stack([pop.loc[k].values for k in te]).astype(float), test_idx=te)


# ------------------------------------------------------------------ model (NumPyro)
def numpyro_model(A, fit=True):
    """NumPyro version of DM_v3_pooled with the repository's priors (see module docstring)."""
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist

    C, Y, R = len(A["countries"]), len(A["years"]), len(A["regions"])
    ridx = jnp.asarray(A["region_idx"])
    sig_r = numpyro.sample("sigma_region_month", dist.HalfNormal(PRIORS["sigma_region_month"]).expand([R, 12]).to_event(2))
    g = numpyro.sample("g_month", dist.Normal(0, PRIORS["g_month"]).expand([12]).to_event(1))
    z_r = numpyro.sample("z_r_month", dist.Normal(0, 1).expand([R, 12]).to_event(2))
    r_month = g + z_r * sig_r
    tau_c = numpyro.sample("tau_country_month", dist.HalfNormal(PRIORS["tau_country_month"]))
    z_c = numpyro.sample("z_c_month", dist.Normal(0, 1).expand([C, 12]).to_event(2))
    c_month = r_month[ridx] + z_c * tau_c
    seasonal = jnp.asarray(A["props_log"]) + c_month
    tau_y = numpyro.sample("tau_region_year", dist.HalfNormal(PRIORS["tau_region_year"]).expand([R]).to_event(1))
    ye_raw = numpyro.sample("year_effect_raw", dist.Normal(0, 1).expand([C, Y]).to_event(2))
    corr = np.array([[RHO ** abs(i - j) for j in range(Y)] for i in range(Y)])
    L = jnp.asarray(np.linalg.cholesky(corr))
    ye = (ye_raw @ L.T) * tau_y[ridx][:, None]
    logits = seasonal[:, None, :] + ye[:, :, None]
    pi = jax_softmax(logits)
    log_conc = numpyro.sample("log_concentration", dist.Normal(PRIORS["log_conc_mu"], PRIORS["log_conc_sd"]).expand([C, Y]).to_event(2))
    a = jnp.maximum(pi, 1e-6) * jnp.maximum(jnp.exp(log_conc), 1.0)[:, :, None]
    a_fit = a[A["fit_cidx"], A["fit_yidx"]]
    numpyro.sample("obs", dist.DirichletMultinomial(a_fit, total_count=jnp.asarray(A["fit_totals"])).to_event(1),
                   obs=jnp.asarray(A["fit_counts"]))
    numpyro.deterministic("a_missing", a[A["test_cidx"], A["test_yidx"]])


def jax_softmax(x):
    """Softmax over the last (month) axis."""
    import jax
    return jax.nn.softmax(x, axis=-1)


def fit_nuts(A, seed, chains=2, warmup=500, draws=500, target_accept=0.95):
    """NumPyro NUTS fit (sequential chains). Returns (mcmc, samples by chain, extra fields)."""
    import jax
    from numpyro.infer import MCMC, NUTS
    kernel = NUTS(numpyro_model, target_accept_prob=target_accept)
    mcmc = MCMC(kernel, num_warmup=warmup, num_samples=draws, num_chains=chains,
                chain_method="sequential", progress_bar=False)
    mcmc.run(jax.random.PRNGKey(seed), A)
    s = mcmc.get_samples(group_by_chain=True)
    extra = mcmc.get_extra_fields(group_by_chain=True)
    return mcmc, s, extra


def nuts_diagnostics(samples, extra):
    """Divergences, max split R-hat and min ESS of the hierarchical scales and concentrations."""
    from numpyro.diagnostics import effective_sample_size, split_gelman_rubin
    out = {"divergences": int(np.asarray(extra["diverging"]).sum())}
    for v in ["tau_country_month", "sigma_region_month", "tau_region_year", "log_concentration"]:
        x = np.asarray(samples[v])
        x = x.reshape(x.shape[0], x.shape[1], -1)
        out[f"rhat_max_{v}"] = float(np.nanmax(split_gelman_rubin(x)))
        out[f"ess_min_{v}"] = float(np.nanmin(effective_sample_size(x)))
    return out


# ------------------------------------------------------------------ MAP
def fit_map(A, seed, restarts=3, maxiter=5000):
    """MAP of the same log posterior, non-centred, half-normal scales on log scale
    with Jacobian, optimised with scipy L-BFGS-B using JAX gradients."""
    import jax
    import jax.numpy as jnp
    from jax.flatten_util import ravel_pytree
    from jax.scipy.special import gammaln
    from scipy.optimize import minimize
    jax.config.update("jax_enable_x64", True)

    C, Y, R = len(A["countries"]), len(A["years"]), len(A["regions"])
    ridx = jnp.asarray(A["region_idx"])
    corr = np.array([[RHO ** abs(i - j) for j in range(Y)] for i in range(Y)])
    L = jnp.asarray(np.linalg.cholesky(corr))
    props_log = jnp.asarray(A["props_log"])
    fc, fy = jnp.asarray(A["fit_cidx"]), jnp.asarray(A["fit_yidx"])
    yobs = jnp.asarray(A["fit_counts"], dtype=jnp.float64)
    n = jnp.asarray(A["fit_totals"], dtype=jnp.float64)

    def hn_logpdf_logscale(u, s):  # log HalfNormal(e^u; s) + u (Jacobian)
        x = jnp.exp(u)
        return jnp.sum(-0.5 * (x / s) ** 2 + u)

    def norm(x, s, mu=0.0):
        return jnp.sum(-0.5 * ((x - mu) / s) ** 2)

    def alpha_all(p):
        r_month = p["g"] + p["z_r"] * jnp.exp(p["u_sig_r"])
        c_month = r_month[ridx] + p["z_c"] * jnp.exp(p["u_tau_c"])
        ye = (p["ye_raw"] @ L.T) * jnp.exp(p["u_tau_y"])[ridx][:, None]
        logits = (props_log + c_month)[:, None, :] + ye[:, :, None]
        pi = jax.nn.softmax(logits, axis=-1)
        return jnp.maximum(pi, 1e-6) * jnp.maximum(jnp.exp(p["log_conc"]), 1.0)[:, :, None]

    def neg_logpost(p):
        a = alpha_all(p)[fc, fy]
        a0 = a.sum(-1)
        ll = jnp.sum(gammaln(a0) - gammaln(n + a0) + jnp.sum(gammaln(yobs + a) - gammaln(a), -1))
        lp = (norm(p["g"], PRIORS["g_month"]) + hn_logpdf_logscale(p["u_sig_r"], PRIORS["sigma_region_month"])
              + norm(p["z_r"], 1) + hn_logpdf_logscale(p["u_tau_c"], PRIORS["tau_country_month"])
              + norm(p["z_c"], 1) + hn_logpdf_logscale(p["u_tau_y"], PRIORS["tau_region_year"])
              + norm(p["ye_raw"], 1) + norm(p["log_conc"], PRIORS["log_conc_sd"], PRIORS["log_conc_mu"]))
        return -(ll + lp)

    rng = np.random.default_rng(seed)
    template = dict(g=np.zeros(12), u_sig_r=np.zeros((R, 12)), z_r=np.zeros((R, 12)),
                    u_tau_c=np.zeros(()), z_c=np.zeros((C, 12)), u_tau_y=np.zeros(R),
                    ye_raw=np.zeros((C, Y)), log_conc=np.zeros((C, Y)))
    flat0, unravel = ravel_pytree(jax.tree_util.tree_map(jnp.asarray, template))
    vg = jax.jit(jax.value_and_grad(lambda f: neg_logpost(unravel(f))))

    def fun(x):
        v, g = vg(jnp.asarray(x))
        return float(v), np.asarray(g, dtype=float)

    best = None
    for k in range(restarts):
        init = dict(g=rng.normal(0, 0.1, 12), u_sig_r=np.log(0.3) + rng.normal(0, 0.3, (R, 12)),
                    z_r=rng.normal(0, 0.5, (R, 12)), u_tau_c=np.array(np.log(0.1) + rng.normal(0, 0.3)),
                    z_c=rng.normal(0, 0.5, (C, 12)), u_tau_y=np.log(0.1) + rng.normal(0, 0.3, R),
                    ye_raw=rng.normal(0, 0.5, (C, Y)), log_conc=1.6 + rng.normal(0, 0.2, (C, Y)))
        x0, _ = ravel_pytree(jax.tree_util.tree_map(jnp.asarray, init))
        res = minimize(fun, np.asarray(x0), jac=True, method="L-BFGS-B",
                       options=dict(maxiter=maxiter, maxfun=maxiter * 2, gtol=1e-6, ftol=1e-15, maxcor=30))
        if best is None or res.fun < best.fun:
            best = res
    p = unravel(jnp.asarray(best.x))
    a = np.asarray(alpha_all(p))
    a_missing = a[A["test_cidx"], A["test_yidx"]]
    info = dict(fun=float(best.fun), nit=int(best.nit), success=bool(best.success),
                grad_inf=float(np.max(np.abs(fun(best.x)[1]))),
                tau_country_month=float(np.exp(p["u_tau_c"])),
                sigma_region_month_mean=float(np.exp(p["u_sig_r"]).mean()),
                tau_region_year_mean=float(np.exp(p["u_tau_y"]).mean()),
                total_concentration_mean_train=float(np.exp(np.asarray(p["log_conc"])[A["fit_cidx"], A["fit_yidx"]]).mean()))
    return a_missing, info


# ------------------------------------------------------------------ PPC + scoring
def dm_ppc_from_alpha(alpha, n_vec, seed):
    """alpha (S, N, 12) -> counts (S, N, 12); Dirichlet via gamma, then multinomial."""
    rng = np.random.default_rng(seed)
    S, N, M = alpha.shape
    out = np.empty((S, N, M), dtype=np.int64)
    for s in range(S):
        g = rng.gamma(np.clip(alpha[s], 1e-12, None))
        p = g / g.sum(1, keepdims=True)
        p = np.where(np.isfinite(p), p, 1.0 / M)
        out[s] = rng.multinomial(np.asarray(n_vec, np.int64), p)
    assert (out.sum(-1) == np.asarray(n_vec)[None, :]).all()
    return out


def score_incidence(pred_counts, truth, pop):
    """Same as score_fold_incidence (mean- and median-based MAE/RMSE, COV80, CRPS)."""
    from ..evaluation.metrics import crps_samples
    ok = np.isfinite(truth) & np.isfinite(pop) & (pop >= POP_FLOOR)
    y = (PER * truth / pop)[ok]
    out = dict(n_rows=int(ok.sum()))
    if pred_counts.ndim == 2:   # deterministic point prediction
        pm = (PER * pred_counts / pop)[ok]
        out.update(MAE=np.mean(np.abs(pm - y)), RMSE=np.sqrt(np.mean((pm - y) ** 2)),
                   MAE_median=np.nan, RMSE_median=np.nan, COV80=np.nan, CRPS=np.nan)
        return out
    S = (PER * pred_counts / pop[None])[:, ok].T
    mean, med = S.mean(1), np.median(S, 1)
    q10, q90 = np.quantile(S, 0.1, 1), np.quantile(S, 0.9, 1)
    out.update(MAE=np.mean(np.abs(mean - y)), RMSE=np.sqrt(np.mean((mean - y) ** 2)),
               MAE_median=np.mean(np.abs(med - y)), RMSE_median=np.sqrt(np.mean((med - y) ** 2)),
               COV80=np.mean((y >= q10) & (y <= q90)), CRPS=float(np.mean(crps_samples(y, S))))
    return out


def shape_scores(pred_shares, truth_counts):
    """Mean JSD and peak-month error of predicted monthly shares against the truth."""
    from ..evaluation.metrics import jsd, peak_week_error
    j, pk = [], []
    for p, t in zip(pred_shares, truth_counts):
        if t.sum() <= 0:
            continue
        j.append(jsd(t / t.sum(), p))
        pk.append(peak_week_error(t, p))
    return float(np.mean(j)), float(np.mean(pk))


def cancellation(pred_mean_shares, cidx):
    """Max |share difference| between held-out years of the same country.

    The country-year effect is constant across months, so it cancels in the
    softmax; the predicted shapes of different test years of a country must be
    identical (up to Monte Carlo error for NUTS)."""
    rows = []
    for c in np.unique(cidx):
        sel = np.where(cidx == c)[0]
        if len(sel) < 2:
            continue
        P = pred_mean_shares[sel]
        rows.append(dict(country_idx=int(c), n_test_years=len(sel),
                         max_abs_diff=float(np.max(P.max(0) - P.min(0)))))
    return rows


def run_fold(df_long, fold, method, rep=1, chains=2, warmup=500, draws=500, target_accept=0.95,
             n_ppc=500):
    """Fit one fold with method in {'NUTS','MAP'} and return (metric rows, cancellation rows, diag)."""
    import jax
    jax.config.update("jax_enable_x64", True)
    A = build_arrays(df_long, fold)
    seed = 20250811 + 10000 * rep + 100 * fold
    t0 = time.time()
    if method == "NUTS":
        _, s, extra = fit_nuts(A, seed, chains, warmup, draws, target_accept)
        a = np.asarray(s["a_missing"])            # (chains, draws, N, 12)
        a = a.reshape(-1, *a.shape[2:])
        idx = np.linspace(0, a.shape[0] - 1, min(n_ppc, a.shape[0])).round().astype(int)
        alpha = a[idx]
        diag = nuts_diagnostics(s, extra)
        diag["log_conc_mean_train"] = float(np.asarray(s["log_concentration"]).reshape(-1, len(A["countries"]), len(A["years"]))[:, A["fit_cidx"], A["fit_yidx"]].mean())
        mean_shares = (alpha / alpha.sum(-1, keepdims=True)).mean(0)
    else:
        a_missing, diag = fit_map(A, seed)
        alpha = np.repeat(a_missing[None], n_ppc, axis=0)
        mean_shares = a_missing / a_missing.sum(-1, keepdims=True)
    fit_s = time.time() - t0
    ppc = dm_ppc_from_alpha(alpha, A["test_totals"], seed)
    sc = score_incidence(ppc, A["test_counts"].astype(float), A["test_pop"])
    jm, pk = shape_scores(mean_shares, A["test_counts"])
    rows = [dict(model="DM", fit=method, fold=fold, **sc, JSD_mean=jm, peak_month_err=pk,
                 runtime_s=round(time.time() - t0, 1), fit_s=round(fit_s, 1))]
    # reference rows on the same test rows
    hist = np.exp(A["props_log"]); hist = hist / hist.sum(1, keepdims=True)
    for name, sh in [("uniform", np.full((len(A["test_totals"]), 12), 1 / 12)),
                     ("historical_mean_shares", hist[A["test_cidx"]])]:
        sc = score_incidence(A["test_totals"][:, None] * sh, A["test_counts"].astype(float), A["test_pop"])
        jm, pk = shape_scores(sh, A["test_counts"])
        rows.append(dict(model=name, fit="-", fold=fold, **sc, JSD_mean=jm, peak_month_err=pk,
                         runtime_s=0.0, fit_s=0.0))
    canc = [dict(fit=method, fold=fold, **r) for r in cancellation(mean_shares, A["test_cidx"])]
    diag.update(fit=method, fold=fold, n_train=len(A["fit_totals"]), n_test_years=len(A["test_totals"]),
                n_countries=len(A["countries"]), n_years=len(A["years"]))
    return rows, canc, diag
