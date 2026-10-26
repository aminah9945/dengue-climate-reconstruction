"""Step 8: rain / humidity / temperature lag extension (pre-registered follow-up).

See ``results/rain_extension/PREREGISTRATION.md``. Existing models are not
changed: this module only adds new feature builders and a LOYO driver.

Variants (all = B4s + national theta + regional delta on climate features):

* ``M1``      existing reference: block means of temp, log1p(rain), humid anomalies,
  blocks 0-3, 4-7, 8-11, 12-15, 16-20 weeks (15 features)
* ``M1-R``    "dual rain": blocks extended with 21-26; rain block mean split into
  wet = max(A, 0) and dry = min(A, 0); temp, humid unsplit (24 features)
* ``M1-RTH``  M1-R plus the same pos/neg split for temp and humid (36 features)
* ``<variant>-shufTT-s0``  train+test climate shuffle (whole-year derangement per
  municipality, seed 0 of the existing M1-shufTT permutation)

CLI (2024 is never loaded)::

    python scripts/run_pipeline.py --step extension --subset 60
    python scripts/run_pipeline.py --step extension --subset 300

Outputs go to ``results/rain_extension/`` (preds_*, metrics_detail_*,
fitted_params_*, leakage_audit_*, runtime_*).
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from .. import config as C
from ..checks import shuffle as SH
from ..checks.leakage import leakage_audit
from ..data import panel as BD
from ..evaluation.cv import row_masks
from ..evaluation.scoring import score as _score
from ..features import climate as FC
from ..features import seasonal as FS
from ..features.size_terms import size_covariates
from ..models.dirichlet_multinomial import DMModel

RAIN_DIR = C.RAIN_DIR

LAG_BLOCKS_EXT = list(C.LAG_BLOCKS) + [(21, 26)]
VARIANTS = ("M1", "M1-R", "M1-RTH")
VAR_SHORT = {"temp_mean": "temp", "precip": "rain", "humid_mean": "humid"}


def block_label(b):
    """'lo-hi' label of a lag block."""
    return f"{b[0]}-{b[1]}"


def feature_spec(variant):
    """List of (var, part, block) in feature order.  part in {'all','pos','neg'}
    ('pos' = max(A,0) = wet for rain; 'neg' = min(A,0) = dry for rain)."""
    if variant == "M1":
        return [(v, "all", b) for v in C.CLIM_VARS for b in C.LAG_BLOCKS]
    split = {"M1-R": {"precip"}, "M1-RTH": set(C.CLIM_VARS)}[variant]
    spec = []
    for v in C.CLIM_VARS:
        parts = ("pos", "neg") if v in split else ("all",)
        for part in parts:
            spec += [(v, part, b) for b in LAG_BLOCKS_EXT]
    return spec


def feature_names(variant):
    """Readable feature names, e.g. ``rain_wet[0-3]``."""
    out = []
    for v, part, b in feature_spec(variant):
        name = VAR_SHORT[v]
        if v == "precip" and part != "all":
            name += "_wet" if part == "pos" else "_dry"
        elif part != "all":
            name += "_pos" if part == "pos" else "_neg"
        out.append(f"{name}[{block_label(b)}]")
    return out


def split_features_continuous(A, variant):
    """A (V, M, K) anomalies -> F (Fn, M, K) on the continuous timeline."""
    blocks = C.LAG_BLOCKS if variant == "M1" else LAG_BLOCKS_EXT
    Fb = FC.lag_block_features(A, blocks)                    # (V*B, M, K), var-major
    B = len(blocks)
    out = []
    for v, part, b in feature_spec(variant):
        f = Fb[C.CLIM_VARS.index(v) * B + blocks.index(b)]
        if part == "pos":
            f = np.maximum(f, 0.0)
        elif part == "neg":
            f = np.minimum(f, 0.0)
        out.append(f)                                       # NaN (before series start) stays NaN
    return np.stack(out)


def ext_climate_features(P, Xc, epi_year, epi_week, heldout_year, variant, weekly=True):
    """In-fold features for a variant: climatology on 2009 + training years only."""
    train_years = [y for y in range(2009, 2024) if y != heldout_year]
    A, beta, sd, n_nan = FC.climate_anomalies(Xc, epi_year, epi_week, train_years)
    F = split_features_continuous(A, variant)
    return FC.to_slots(F, P, weekly), dict(beta=beta, sd=sd, n_nan=n_nan, A=A)


def ext_leakage_audit(P, Xc, epi_year, epi_week, heldout_t, variant, X_ref, weekly=True):
    """New-feature audit (see PREREGISTRATION.md):
    a) climatology/SD unchanged when the held-out year's climate is NaN;
    b) features unchanged when held-out weekly counts and totals are NaN;
    c) pos + neg == M1 block mean for the original 5 blocks;
    d) no NaN in any block (incl. 21-26) for any valid week of an evaluation year."""
    import copy
    y = int(P.years[heldout_t])
    train_years = [yy for yy in range(2009, 2024) if yy != y]
    _, b1, s1, _ = FC.climate_anomalies(Xc, epi_year, epi_week, train_years)
    Xn = Xc.copy(); Xn[:, :, epi_year == y] = np.nan
    _, b2, s2, _ = FC.climate_anomalies(Xn, epi_year, epi_week, train_years)
    out = {"ext_climatology": bool(np.array_equal(b1, b2) and np.array_equal(s1, s2))}
    P2 = copy.copy(P)
    P2.Yw = P.Yw.copy(); P2.Yw[:, heldout_t] = np.nan
    P2.Ym = P.Ym.copy(); P2.Ym[:, heldout_t] = np.nan
    P2.N = P.N.copy(); P2.N[:, heldout_t] = np.nan
    X2, _ = ext_climate_features(P2, Xc, epi_year, epi_week, y, variant, weekly)
    out["ext_features_no_dengue"] = bool(np.array_equal(X_ref, X2))
    A, *_ = FC.climate_anomalies(Xc, epi_year, epi_week, train_years)
    Fb = FC.lag_block_features(A, C.LAG_BLOCKS)
    Fv = split_features_continuous(A, variant)
    spec = feature_spec(variant)
    ok = True
    for v in C.CLIM_VARS:
        for b in C.LAG_BLOCKS:
            parts = [i for i, s in enumerate(spec) if s[0] == v and s[2] == b]
            tot = np.nansum(Fv[parts], 0) if len(parts) > 1 else Fv[parts[0]]
            ref = Fb[C.CLIM_VARS.index(v) * len(C.LAG_BLOCKS) + C.LAG_BLOCKS.index(b)]
            fin = np.isfinite(ref)
            ok &= bool(np.allclose(tot[fin], ref[fin], atol=1e-12))
    out["ext_split_sums_to_M1"] = ok
    idx = P.week_index[P.valid_w]
    out["ext_no_nan_eval_weeks"] = bool(np.isfinite(Fv[:, :, idx]).all())
    return out


def main(subset=60, models=("M1", "M1-R", "M1-RTH"), tag=None, weekly=True):
    assert not C.RUN_FINAL_2024
    C.ensure_dirs()
    t_start = time.time()
    S = 53 if weekly else 12
    tag = tag or f"weekly_{subset}"
    d, clim, muni = BD.load_tables()
    if subset == 60:
        f60 = C.RESULTS_DIR / "subset60.csv"
        geos = pd.read_csv(f60).geocode.values if f60.exists() else BD.select_subset60(d, muni).geocode.values
    else:
        geos = muni.geocode.values
    P = BD.build_panel(d, clim, geos)
    assert P.years.max() <= 2023
    Xc, c_year, c_week = BD.climate_matrix(clim, P.geocodes)
    M, T = len(P.geocodes), len(P.years)
    Y = P.Yw if weekly else P.Ym
    valid = P.valid_w if weekly else np.ones((T, 12), bool)
    nsl = P.nweeks if weekly else np.full(T, 12)
    train_ok, eval_ok = row_masks(P)
    print(f"[rain {tag}] M={M} T={T} train rows={train_ok.sum()} eval rows={eval_ok.sum()} models={models}", flush=True)

    K = 12
    base_of = {m: m.split("-shufTT")[0] for m in models}
    nfeat = {m: len(feature_spec(base_of[m])) for m in models}
    dms = {F: DMModel(M, T, S, K, F, P.region, valid, weekly, "dm") for F in set(nfeat.values())}
    # identical permutation to run_brazil's M1-shufTT-s0
    perm0 = SH.train_test_permutations(M, T, 1)[0]

    warm = {m: None for m in models}
    preds = {m: np.full((M, T, S), np.nan) for m in models}
    phi_used = {m: np.full((M, T), np.nan) for m in models}
    fitted, audits, runtime = {}, [], []
    for t, y in enumerate(P.years):
        if y not in C.LOYO_YEARS:
            continue
        tf = time.time()
        tm = train_ok.copy(); tm[:, t] = False
        w = tm.astype(float)
        h, _, _ = FS.offsets(P, tm, weekly, C.EPS_H)
        Z, _ = size_covariates(P, tm, None, weekly)
        Z = Z[:, :, :S]
        rows = np.where(eval_ok[:, t])[0]
        Xcache = {}
        aud = dict(year=int(y), **leakage_audit(P, Xc, c_year, c_week, tm, t, weekly))
        for name in models:
            base = base_of[name]
            if base not in Xcache:
                Xcache[base], _ = ext_climate_features(P, Xc, c_year, c_week, y, base, weekly)
                if base != "M1":
                    for k, v in ext_leakage_audit(P, Xc, c_year, c_week, t, base, Xcache[base], weekly).items():
                        aud[f"{base}:{k}"] = v
            X = Xcache[base]
            if "-shufTT" in name:
                X = SH.shuffle_years(X, perm0)
            mod = dms[nfeat[name]]
            t0 = time.time()
            flat, info = mod.fit(h, Z, X, Y, P.N, w, x0=warm[name])
            warm[name] = flat
            pi = mod.predict(flat, h, Z, X)
            preds[name][rows, t] = pi[rows, t]
            par = mod.params(flat)
            phi_used[name][rows, t] = np.exp(par["logphi"])[P.region[rows]]
            fitted.setdefault(name, []).append(dict(
                year=int(y), **info, phi=np.exp(par["logphi"]).round(4).tolist(),
                features=feature_names(base), theta=par["theta"].round(5).tolist(),
                delta=par["delta"].round(5).tolist()))
            runtime.append(dict(tag=f"rain_{tag}", model=name, fold=int(y), fit_s=round(time.time() - t0, 2),
                                nit=info["nit"], grad_inf=info["grad_inf"]))
        audits.append(aud)
        print(f"[rain {tag}] fold {y}: {len(rows)} eval rows, {time.time() - tf:.1f}s "
              + " ".join(f"{m}:nit={fitted[m][-1]['nit']}" for m in models), flush=True)

    np.savez_compressed(RAIN_DIR / f"preds_{tag}.npz", geocodes=P.geocodes, years=P.years,
                        region=P.region, N=P.N, Y=Y, eval_ok=eval_ok, nsl=nsl,
                        **{f"pred__{k}": v for k, v in preds.items()},
                        **{f"phi__{k}": v for k, v in phi_used.items()})
    (RAIN_DIR / f"fitted_params_{tag}.json").write_text(json.dumps(fitted, indent=1))
    pd.DataFrame(audits).to_csv(RAIN_DIR / f"leakage_audit_{tag}.csv", index=False)
    rt = pd.DataFrame(runtime)
    rt.to_csv(RAIN_DIR / f"runtime_{tag}.csv", index=False)
    fit_wall = time.time() - t_start
    score(tag, models)
    wall = time.time() - t_start
    (RAIN_DIR / f"run_info_{tag}.json").write_text(json.dumps(dict(
        tag=tag, subset=subset, models=list(models), fit_wall_s=round(fit_wall, 1),
        total_wall_s=round(wall, 1)), indent=1))
    print(f"[rain {tag}] done in {wall:.0f}s (fits {fit_wall:.0f}s)", flush=True)


def score(tag, models):
    """Score with the unchanged scoring function (probabilistic metrics for every
    fitted variant), reading and writing in ``results/rain_extension/``."""
    return _score(tag, out_dir=RAIN_DIR, prob_models=list(models))
