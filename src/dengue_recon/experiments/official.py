"""Official experiment driver: validation designs A, B and C (weekly resolution).

=========  =================================================================
Design A   expanding window (primary): test T = 2012 ... 2023, train 2010 ... T-1
Design B   learning curve: test T in (2019, 2021, 2023), train on the k years
           immediately before T, k = 2 ... T - 2010
Design C   leave-one-year-out 2010-2023 (as in the pilot; secondary)
=========  =================================================================

Models (``config.OFFICIAL_MODELS``; the pilot's model maths is unchanged):

* B0 uniform, B1 seasonal profile of the fold's training years
* B4, B4s, M1-noONI: the pilot's 15-feature Dirichlet-multinomial (M1-noONI = pilot M1)
* M1 = B4s + 15 local lag-block climate anomalies + 5 ONI lag-block features (20)
* M1-short: block 0-3 weeks of each of the 4 variables (incl. ONI)
* M1-shufTT: climate + ONI years deranged within municipality among the fold's
  years (refitted, seed 0); M1-shufTest: trained M1 given another fold year's
  climate in the test year
* M2: XGBoost softmax over weeks, settings chosen on the last training year
  (``models.xgb_softmax.fit_predict_official``)

Leakage rules. For designs A and B the panel of a fold holds only the years from
the one before the first training year (its annual total is the previous-year
size covariate) up to the test year, and climate after the test year is dropped,
so later data are not even present. In every design the offsets, size centring,
climatology, ONI standardisation and M2 settings use only the fold's training
years (+ 2009 climate, which has no dengue target). An audit checks this per fold.

Checkpoints: each fold writes ``folds/<fold_id>.npz`` (predictions, phi, fitted
parameter vectors used as warm starts), ``folds/<fold_id>_metrics.csv`` and, last,
``folds/<fold_id>.json`` (fold log). A rerun skips folds whose log exists and
warm-starts from their saved parameters, so a resumed run gives the same numbers.
At the end ``metrics_detail_<tag>.csv``, ``fold_log_<tag>.csv``,
``leakage_audit_<tag>.csv`` and ``checks_<tag>.json`` are assembled.

CLI: ``python scripts/run_pipeline.py --step official --design A --climate power --subset 300``.
2024 is never loaded while ``config.RUN_FINAL_2024`` is False.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config as C
from ..baselines.simple import seasonal_shares, uniform_shares
from ..checks import shuffle as SH
from ..checks.leakage import official_leakage_audit
from ..checks.zero_climate import zero_climate_maxdiff, zero_climate_maxdiff_embedded
from ..data import panel as BD
from ..evaluation.cv import (assert_final_year_locked, climate_years, design_folds, fold_train_mask,
                             panel_years, row_masks)
from ..evaluation.scoring import score_rows
from ..features import climate as FC
from ..features import seasonal as FS
from ..features.size_terms import size_covariates
from ..models.dirichlet_multinomial import DMModel

#: DM models in fitting order, with (number of climate features, feature variant)
DM_SPECS = {"B4": (15, "none"), "B4s": (15, "zero"), "M1-noONI": (15, "local"), "M1": (20, "full"),
            "M1-short": (20, "short"), "M1-shufTT": (20, "shufTT")}
#: models that also get 500 Dirichlet-multinomial draws (coverage, CRPS)
DM_PROB = ["B4", "B4s", "M1", "M1-noONI", "M1-short"]
N_LOCAL = len(C.CLIM_VARS) * len(C.LAG_BLOCKS)          # 15
N_FULL = N_LOCAL + len(C.LAG_BLOCKS)                     # 20


def run_tag(design, climate, subset, exclude_years=()):
    """Output tag, e.g. ``designA_power_300`` or ``designA_power_300_ex2020``."""
    ex = "".join(f"_ex{y}" for y in sorted(exclude_years))
    return f"design{design}_{climate}_{subset}{ex}"


def _ordered(models):
    order = ["B0", "B1"] + list(DM_SPECS) + ["M1-shufTest", "M2"]
    unknown = set(models) - set(order)
    assert not unknown, f"unknown models {unknown}"
    need = set(models)
    if "M1-shufTest" in need:
        need.add("M1")
    return [m for m in order if m in need]


class _Models:
    """DMModel cache keyed by (panel years, number of features): compiled once per shape."""

    def __init__(self, Pfull):
        self.cache, self.P = {}, Pfull

    def get(self, P, F):
        key = (tuple(int(y) for y in P.years), F)
        if key not in self.cache:
            self.cache = {k: v for k, v in self.cache.items() if k[0] == key[0]}   # free other shapes
            self.cache[key] =DMModel(len(P.geocodes), len(P.years), 53, 12, F, P.region, P.valid_w, True, "dm")
        return self.cache[key]


def run_fold(fold, Pfull, Xc, ey, ew, oni, models, warm, cache, seed=C.SEED):
    """Fit and predict every model of one fold. Returns (preds, phi, flats, log)."""
    tf = time.time()
    P = Pfull if fold["design"] == "C" else BD.subset_panel(Pfull, panel_years(fold, Pfull.years))
    assert_final_year_locked(P)
    T = len(P.years)
    t = int(np.where(P.years == fold["test_year"])[0][0])
    train_ok, eval_ok = row_masks(P)
    tm = fold_train_mask(P, train_ok, fold)
    w = tm.astype(float)
    cy = climate_years(fold)
    h, fb, prof = FS.offsets(P, tm, True, C.EPS_H)
    Z, _ = size_covariates(P, tm, None, True)
    X20, cinfo = FC.fold_climate_features(P, Xc, ey, ew, cy, oni=oni, weekly=True)
    X15 = X20[..., :N_LOCAL]
    aud = official_leakage_audit(P, Xc, ey, ew, tm, fold, cy, oni=oni, weekly=True)
    rows = np.where(eval_ok[:, t])[0]
    nsl = P.nweeks
    n = int(nsl[t])
    M = len(P.geocodes)
    fold_years = sorted(set(fold["train_years"]) | {fold["test_year"]})
    yidx = np.where(np.isin(P.years, fold_years))[0]
    preds, phi, flats = {}, {}, {}
    log = dict(fold, train_years=list(fold["train_years"]), clim_years=list(cy), panel_years=P.years.tolist(),
               n_train_rows=int(tm.sum()), n_eval_rows=int(len(rows)),
               n_region_fallback=int(fb.sum()), clim_nan=int(cinfo["n_nan"]),
               oni_mean=cinfo.get("oni_mean"), oni_sd=cinfo.get("oni_sd"), leakage=aud, fits={})
    # --- simple baselines
    if "B0" in models:
        preds["B0"] = np.tile(np.pad(uniform_shares(n), (0, 53 - n), constant_values=np.nan), (len(rows), 1))
    if "B1" in models:
        preds["B1"] = np.full((len(rows), 53), np.nan)
        for i, m in enumerate(rows):
            preds["B1"][i, :n] = seasonal_shares(prof[m], n, True)
    # --- Dirichlet-multinomial ladder
    zero15 = np.zeros_like(X15)
    Xin = {"none": zero15, "zero": zero15, "local": X15, "full": X20}
    if "M1-short" in models:
        Xin["short"] = X20 * FC.short_mask(len(C.CLIM_VARS) + 1)
    if "M1-shufTT" in models:
        Xin["shufTT"] = SH.shuffle_years(X20, SH.fold_year_permutations(M, T, yidx, seed=seed))
    dms = {}
    for name in models:
        if name not in DM_SPECS:
            continue
        F, var = DM_SPECS[name]
        dm = cache.get(P, F)
        ZZ = np.zeros_like(Z) if name == "B4" else Z
        t0 = time.time()
        flat, info = dm.fit(h, ZZ, Xin[var], P.Yw, P.N, w, x0=warm.get(name))
        pi = dm.predict(flat, h, ZZ, Xin[var])
        par = dm.params(flat)
        preds[name] = pi[rows, t]
        phi[name] = np.exp(par["logphi"])[P.region[rows]]
        flats[name] = flat
        dms[name] = dm
        log["fits"][name] = dict(**info, wall_s=round(time.time() - t0, 2),
                                 phi=np.exp(par["logphi"]).round(3).tolist(),
                                 gamma_nat=par["gam"].round(4).tolist(), theta=par["theta"].round(4).tolist(),
                                 delta=par["delta"].round(4).tolist())
    # --- per-fold checks: zero-climate equivalence, test-only shuffle
    zc = {}
    if "B4s" in flats and "M1-noONI" in flats:
        zc["M1-noONI"] = zero_climate_maxdiff(dms["B4s"], flats["B4s"], h, Z, X15, zero15, P.valid_w)
    if "B4s" in flats and "M1" in flats:
        zc["M1"] = zero_climate_maxdiff_embedded(dms["B4s"], flats["B4s"], dms["M1"], h, Z, X20, P.valid_w)
    log["zero_climate_maxdiff"] = zc
    if "M1-shufTest" in models:
        Xs = SH.shuffle_test_year(X20, t, SH.fold_test_permutation(M, T, yidx, seed=seed))
        preds["M1-shufTest"] = dms["M1"].predict(flats["M1"], h, Z, Xs)[rows, t]
    # --- M2: settings chosen on the last training year, then refitted on all training years
    if "M2" in models:
        from ..models.xgb_softmax import fit_predict_official
        t0 = time.time()
        last = max(fold["train_years"])
        inner_years = tuple(y for y in fold["train_years"] if y != last)
        if inner_years:
            ifold = dict(fold, train_years=inner_years)
            tm_in = fold_train_mask(P, train_ok, ifold)
            val = train_ok & (P.years == last)[None, :]
            h_in, _, _ = FS.offsets(P, tm_in, True, C.EPS_H)
            X_in, _ = FC.fold_climate_features(P, Xc, ey, ew, climate_years(ifold), oni=oni, weekly=True)
            pr, info = fit_predict_official(
                P, True, dict(h=h_in, X=X_in, train_mask=tm_in, val_mask=val),
                dict(h=h, X=X20, train_mask=tm), [(int(m), t) for m in rows], seed=seed)
            preds["M2"] = pr
            log["fits"]["M2"] = dict(**info, inner_years=list(inner_years), val_year=int(last),
                                     wall_s=round(time.time() - t0, 2))
        else:
            log["fits"]["M2"] = dict(skipped="fewer than 2 training years: no inner validation year")
    order = [m for m in _ordered(models) if m in preds]
    preds = {m: preds[m] for m in order}
    log["wall_s"] = round(time.time() - tf, 1)
    return P, t, rows, preds, phi, flats, log


def main(design="A", climate="power", subset=60, exclude_years=(), models=None, out_dir=None,
         test_years=None, seed=C.SEED):
    """Run one official design; returns the assembled per-row metrics DataFrame."""
    t_start = time.time()
    exclude_years = tuple(int(y) for y in exclude_years)
    models = _ordered(models or C.OFFICIAL_MODELS[design])
    tag = run_tag(design, climate, subset, exclude_years)
    out = Path(out_dir or C.OFFICIAL_DIR) / tag
    fdir = out / "folds"
    fdir.mkdir(parents=True, exist_ok=True)
    d, clim, muni = BD.load_tables(climate)
    from .brazil_loyo import load_subset
    geos = load_subset(subset, d, muni, write_subset=False)
    Pfull = BD.build_panel(d, clim, geos)
    assert_final_year_locked(Pfull)
    for y in exclude_years:                      # sensitivity: treat the year as excluded everywhere
        Pfull.excluded[:, Pfull.years == y] = True
    Xc, ey, ew = BD.climate_matrix(clim, Pfull.geocodes)
    oni = BD.oni_vector(BD.load_oni_weekly(), ey, ew)
    folds = design_folds(design, exclude_years, test_years)
    info = dict(design=design, climate=climate, subset=subset, exclude_years=list(exclude_years),
                models=models, n_municipalities=len(Pfull.geocodes), panel_years=Pfull.years.tolist(),
                folds=[dict(f, train_years=list(f["train_years"])) for f in folds],
                run_final_2024=C.RUN_FINAL_2024, seed=seed, pen=C.PEN, m2_grid=C.M2_GRID,
                m2_max_rounds=C.M2_MAX_ROUNDS, m2_early_stop=C.M2_EARLY_STOP)
    (out / "run_info.json").write_text(json.dumps(info, indent=1, default=str))
    print(f"[{tag}] M={len(Pfull.geocodes)} folds={len(folds)} models={models}", flush=True)
    cache = _Models(Pfull)
    warm = {}
    for fold in folds:
        fid = fold["fold_id"]
        f_log, f_npz, f_met = fdir / f"{fid}.json", fdir / f"{fid}.npz", fdir / f"{fid}_metrics.csv"
        if f_log.exists() and f_npz.exists() and f_met.exists():
            z = np.load(f_npz)
            warm.update({k[6:]: z[k] for k in z.files if k.startswith("flat__")})
            print(f"[{tag}] {fid}: checkpoint found, skipped", flush=True)
            continue
        P, t, rows, preds, phi, flats, log = run_fold(fold, Pfull, Xc, ey, ew, oni, models, warm, cache, seed)
        warm.update(flats)
        np.savez_compressed(f_npz, geocodes=P.geocodes[rows], rows=rows, test_year=fold["test_year"],
                            **{f"pred__{k}": v for k, v in preds.items()},
                            **{f"phi__{k}": v for k, v in phi.items()},
                            **{f"flat__{k}": v for k, v in flats.items()})
        ts = time.time()
        df = score_rows(preds, rows, P, t, P.Yw, P.nweeks, phi, [m for m in DM_PROB if m in phi],
                        seed=[seed, fold["test_year"], fold["k"]], floor_seed=[0, fold["test_year"]])
        df.insert(0, "fold_id", fid); df.insert(0, "design", design)
        df["k"], df["climate"] = fold["k"], climate
        df.to_csv(f_met, index=False)
        log["score_s"] = round(time.time() - ts, 1)
        f_log.write_text(json.dumps(log, indent=1, default=float))
        aud_ok = all(log["leakage"].values())
        zc = log["zero_climate_maxdiff"]
        print(f"[{tag}] {fid}: train years {fold['train_years'][0]}-{fold['train_years'][-1]} (k={fold['k']}), "
              f"{len(rows)} eval rows, fit {log['wall_s']}s + score {log['score_s']}s, "
              f"leakage audit {'pass' if aud_ok else 'FAIL'}, zero-climate {zc}", flush=True)
    return assemble(out, tag, folds, t_start)


def assemble(out, tag, folds, t_start=None):
    """Concatenate the fold outputs into the run-level CSV/JSON files."""
    fdir = out / "folds"
    mets, logs, auds, fits = [], [], [], []
    for f in folds:
        fid = f["fold_id"]
        mets.append(pd.read_csv(fdir / f"{fid}_metrics.csv"))
        lg = json.loads((fdir / f"{fid}.json").read_text())
        logs.append(lg)
        auds.append(dict(fold_id=fid, test_year=f["test_year"], k=f["k"], **lg["leakage"]))
        for m, fi in lg["fits"].items():
            fits.append(dict(fold_id=fid, test_year=f["test_year"], k=f["k"], model=m,
                             wall_s=fi.get("wall_s"), nit=fi.get("nit"), grad_inf=fi.get("grad_inf"),
                             success=fi.get("success"), m2_choice=json.dumps(fi.get("chosen")) if "chosen" in fi else None,
                             m2_rounds=fi.get("n_rounds")))
    df = pd.concat(mets, ignore_index=True)
    df.to_csv(out / f"metrics_detail_{tag}.csv", index=False)
    pd.DataFrame(fits).to_csv(out / f"fold_log_{tag}.csv", index=False)
    aud = pd.DataFrame(auds)
    aud.to_csv(out / f"leakage_audit_{tag}.csv", index=False)
    checks = dict(
        n_folds=len(folds),
        leakage_audit_all_pass=bool(aud.drop(columns=["fold_id", "test_year", "k"]).all().all()),
        zero_climate_maxdiff={lg["fold_id"]: lg["zero_climate_maxdiff"] for lg in logs},
        clim_nan=int(sum(lg["clim_nan"] for lg in logs)),
        years_scored=sorted(int(y) for y in df.year.unique()),
        no_2024=bool((df.year < C.FINAL_TEST_YEAR).all()) or C.RUN_FINAL_2024,
        fold_wall_s={lg["fold_id"]: lg["wall_s"] + lg.get("score_s", 0) for lg in logs},
    )
    if t_start is not None:
        checks["run_wall_s"] = round(time.time() - t_start, 1)
    (out / f"checks_{tag}.json").write_text(json.dumps(checks, indent=1))
    print(df.groupby("model")[["jsd", "peak_err"]].mean().round(6).to_string(), flush=True)
    return df
