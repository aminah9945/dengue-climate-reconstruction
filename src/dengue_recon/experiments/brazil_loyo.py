"""LOYO driver for the Brazil experiment (Steps 3-7 for every fold), no 2024.

For each held-out year 2010-2023 it builds the in-fold features (Step 4), the
simple baselines (Step 3), fits the Dirichlet-multinomial ladder and ablations
with warm starts from the previous fold (Step 5), runs the per-fold checks
(zero-climate equivalence, test-only shuffle, leakage audit; Step 7), saves the
predictions and finally scores them (Step 6).

Outputs (tagged by resolution and subset) go to ``config.BRAZIL_DIR``:
``preds_{tag}.npz``, ``metrics_detail_{tag}.csv``, ``fitted_params_{tag}.json``,
``leakage_audit_{tag}.csv``, ``checks_{tag}.json``; runtime rows are appended to
``config.RESULTS_DIR/runtime_log.csv``.

CLI: ``python scripts/run_pipeline.py --step brazil --res weekly --subset 60``.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from .. import config as C
from ..baselines.simple import past_only_shares, seasonal_shares, uniform_shares
from ..checks import shuffle as SH
from ..checks.leakage import leakage_audit
from ..checks.zero_climate import zero_climate_maxdiff
from ..data import panel as BD
from ..evaluation.cv import assert_final_year_locked, loyo_folds, row_masks
from ..evaluation.scoring import score
from ..features import climate as FC
from ..features import seasonal as FS
from ..features.size_terms import size_covariates
from ..models.dirichlet_multinomial import MAIN_MODELS, PILOT_MODELS, DMModel, variant_inputs


def log_runtime(row):
    """Append one row (tag, model, fold, fit_s, nit, grad_inf) to ``runtime_log.csv``."""
    f = C.RESULTS_DIR / "runtime_log.csv"
    pd.DataFrame([row]).to_csv(f, mode="a", header=not f.exists(), index=False)


def load_subset(subset, d, muni, write_subset=True):
    """Geocodes for the 60-municipality pilot subset or all 300 municipalities."""
    if subset == 60:
        sel = BD.select_subset60(d, muni)
        if write_subset:
            sel.to_csv(C.RESULTS_DIR / "subset60.csv", index=False)
        return sel.geocode.values
    return muni.geocode.values


def main(res="weekly", subset=60, main_only=False, n_shuffle=3, tag=None, models=None, years=None):
    """Run the LOYO experiment.

    Parameters
    ----------
    res : ``"weekly"`` or ``"monthly"``
    subset : 60 (pilot) or 300 (all municipalities)
    main_only : fit only B4, B4s, M1 (plus B0, B1, B1p and the test-only shuffle)
    n_shuffle : number of train-and-test shuffle seeds in the full pilot set
    tag : output tag (default ``f"{res}_{subset}"``)
    models : explicit list of fitted models (overrides ``main_only``)
    years : restrict LOYO folds (smoke tests); default all 14
    """
    C.ensure_dirs()
    t_start = time.time()
    weekly = res == "weekly"
    S = 53 if weekly else 12
    d, clim, muni = BD.load_tables()
    geos = load_subset(subset, d, muni)
    P = BD.build_panel(d, clim, geos)
    Xc, c_year, c_week = BD.climate_matrix(clim, P.geocodes)
    tag = tag or f"{res}_{subset}"
    M, T = len(P.geocodes), len(P.years)
    assert_final_year_locked(P)
    Y = P.Yw if weekly else P.Ym
    valid = P.valid_w if weekly else np.ones((T, 12), bool)
    nsl = (P.nweeks if weekly else np.full(T, 12))
    train_ok, eval_ok = row_masks(P)
    print(f"[{tag}] M={M} T={T} train rows={train_ok.sum()} eval rows={eval_ok.sum()}", flush=True)

    if models is None:
        models = list(MAIN_MODELS) if main_only else list(PILOT_MODELS) + [f"M1-shufTT-s{s}" for s in range(n_shuffle)]
    models = list(models)
    K, F = 12, len(C.CLIM_VARS) * len(C.LAG_BLOCKS)
    dm = DMModel(M, T, S, K, F, P.region, valid, weekly, "dm")
    mult = DMModel(M, T, S, K, F, P.region, valid, weekly, "mult") if "M1-mult" in models else None
    zeroZ = np.zeros((M, T, S, K)); zeroX = np.zeros((M, T, S, F))
    perms = SH.train_test_permutations(M, T, n_shuffle)
    test_perm = SH.test_only_permutation(M, T)

    extra = ["B0", "B1", "B1p"] + (["M1-shufTest"] if "M1" in models else [])
    warm = {m: None for m in models}
    preds = {m: np.full((M, T, S), np.nan) for m in models + extra}
    phi_used = {m: np.full((M, T), np.nan) for m in models}
    fitted, audits, checks = {}, [], dict(zero_climate_maxdiff=[], clim_nan=[])

    for t, y, tm in loyo_folds(P, train_ok, years):
        tf = time.time()
        w = tm.astype(float)
        h, fb, prof = FS.offsets(P, tm, weekly, C.EPS_H)
        Z, _ = size_covariates(P, tm, None, weekly)
        Z = Z[:, :, :S]
        X, cinfo = FC.climate_features(P, Xc, c_year, c_week, y, weekly)
        checks["clim_nan"].append(cinfo["n_nan"])
        aud = leakage_audit(P, Xc, c_year, c_week, tm, t, weekly)
        audits.append(dict(year=int(y), **aud))
        rows = np.where(eval_ok[:, t])[0]
        # --- Step 3: simple baselines
        for m in rows:
            n = nsl[t]
            preds["B0"][m, t, :n] = uniform_shares(n)
            preds["B1"][m, t, :n] = seasonal_shares(prof[m], n, weekly)
            pp = past_only_shares(P, m, t, train_ok, n, weekly)
            if pp is not None:
                preds["B1p"][m, t, :n] = pp
        # --- Step 5: fitted models
        fold_ctx = dict(
            h_guard=lambda: FS.offsets(P, tm, weekly, C.EPS_H_GUARD)[0],
            X_short=lambda: FC.climate_features(P, Xc, c_year, c_week, y, weekly, short=True)[0],
            X_shuffled=lambda s: SH.shuffle_years(X, perms[s]),
        )
        b4s_flat = None
        for name in models:
            t0 = time.time()
            hh, ZZ, XX, mod = variant_inputs(name, h, Z, X, zeroZ, zeroX, dm, mult, fold_ctx)
            flat, info = mod.fit(hh, ZZ, XX, Y, P.N, w, x0=warm[name])
            warm[name] = flat
            pi = mod.predict(flat, hh, ZZ, XX)
            preds[name][rows, t] = pi[rows, t]
            par = mod.params(flat)
            phi_used[name][rows, t] = np.exp(par["logphi"])[P.region[rows]]
            fitted.setdefault(name, []).append(dict(
                year=int(y), **info, phi=np.exp(par["logphi"]).round(3).tolist(),
                gamma_nat=par["gam"].round(4).tolist(), theta=par["theta"].round(4).tolist(),
                delta=par["delta"].round(4).tolist()))
            log_runtime(dict(tag=tag, model=name, fold=int(y), fit_s=round(time.time() - t0, 2),
                             nit=info["nit"], grad_inf=info["grad_inf"]))
            if name == "B4s":
                b4s_flat = flat
            if name == "M1":
                # Step 7, check 3: zero-climate equivalence (B4s params, theta = delta = 0, real X)
                if b4s_flat is not None:
                    checks["zero_climate_maxdiff"].append(zero_climate_maxdiff(dm, b4s_flat, h, Z, X, zeroX, valid))
                # Step 7, check 5: test-only shuffle (trained M1, deranged year's features in test year)
                Xs = SH.shuffle_test_year(X, t, test_perm)
                preds["M1-shufTest"][rows, t] = dm.predict(flat, h, Z, Xs)[rows, t]
        summary = " ".join(f"{m} nit={fitted[m][-1]['nit']}" for m in ("B4s", "M1") if m in fitted)
        print(f"[{tag}] fold {y}: {len(rows)} eval rows, {time.time() - tf:.1f}s ({summary})", flush=True)

    np.savez_compressed(C.BRAZIL_DIR / f"preds_{tag}.npz", geocodes=P.geocodes, years=P.years,
                        region=P.region, N=P.N, Y=Y, eval_ok=eval_ok, nsl=nsl,
                        **{f"pred__{k}": v for k, v in preds.items()},
                        **{f"phi__{k}": v for k, v in phi_used.items()})
    (C.BRAZIL_DIR / f"fitted_params_{tag}.json").write_text(json.dumps(fitted, indent=1))
    pd.DataFrame(audits).to_csv(C.BRAZIL_DIR / f"leakage_audit_{tag}.csv", index=False)
    checks["fit_wall_s"] = round(time.time() - t_start, 1)
    (C.BRAZIL_DIR / f"checks_{tag}.json").write_text(json.dumps(checks, indent=1))
    df = score(tag)
    print(f"[{tag}] done in {time.time() - t_start:.0f}s", flush=True)
    return df
