"""Driver of the timing-informed share model experiment (EXPLORATORY).

For every fold of an official design (A: expanding window 2012-2023, C: LOYO):

1. the reference models B1, B4s, M1-noONI and M1 are fitted by the existing
   official driver (:func:`dengue_recon.experiments.official.run_fold`) on the same
   fold, with the same settings;
2. the timing models are fitted on the same panel, offset ``h`` and size terms ``Z``:

   ==========  ===============================================================
   S1          B4s + state-curve anomaly block (5 lags)
   S2          B4s + search (Google Trends) anomaly block (4 lags)
   S3          B4s + both blocks
   S3C         S3 + the 20 M1 climate features (incl. ONI)
   T1          B4s with the offset h rolled by the estimated state-curve shift
   S*-wy       wrong-year check: the new blocks deranged among the fold's years
               within each municipality (train and test), refitted
   S3-wyTest   fitted S3 given another year's signal in the test year only
   S1-ns       S1 with a pool of non-study municipalities only
   S1/S3/T1-p25, -p50   mixed-resolution scenario: state curve from a random
               25 % / 50 % of the other municipalities (fixed seed)
   ==========  ===============================================================

3. checks per fold: leakage audit of the new features, zero-coefficient
   equivalence with B4s, exact sums (inside the scoring);
4. scoring with the official metrics plus the onset-week error (onset = first
   week whose cumulative share reaches 10 %, fixed a priori).

Checkpoints: ``<out>/<tag>/folds/<fold_id>.{npz,json}`` and ``_metrics.csv``; a
rerun skips finished folds (warm starts restored), so it can be resumed after an
interruption. 2024 is never loaded (``config.RUN_FINAL_2024`` stays False).

CLI: ``python -m dengue_recon.extensions.timing.run --design A --subset 300``.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ... import config as C
from ...checks import shuffle as SH
from ...data import panel as BD
from ...evaluation.cv import climate_years, design_folds, fold_train_mask, panel_years, row_masks
from ...evaluation.metrics import normalise
from ...evaluation.scoring import score_rows
from ...experiments import official as OFF
from ...features import climate as FC
from ...features import seasonal as FS
from ...features.size_terms import size_covariates
from ...models.dirichlet_multinomial import DMModel
from . import features as TF
from . import signals as SG
from .model import YearOffsetDM, transfer_params, zero_coef_maxdiff

REF_MODELS = ["B1", "B4s", "M1-noONI", "M1"]
MAIN = ["S1", "S2", "S3", "S3C", "T1"]
CHECKS = ["S1-wy", "S2-wy", "S3-wy", "S3-wyTest", "S1-ns"]
MIXED = ["S1-p25", "S1-p50", "S3-p25", "S3-p50", "T1-p25", "T1-p50"]
MODELS = {"A": REF_MODELS + MAIN + CHECKS + MIXED, "C": REF_MODELS + MAIN}
PROB = ["B4s", "M1-noONI", "M1"] + MAIN
ONSET_SHARE = 0.10

#: name -> (blocks, pool variant, climate, wrong-year)
SPECS = {
    "S1": (("state",), "all", False, False), "S2": (("trends",), None, False, False),
    "S3": (("state", "trends"), "all", False, False), "S3C": (("state", "trends"), "all", True, False),
    "S1-wy": (("state",), "all", False, True), "S2-wy": (("trends",), None, False, True),
    "S3-wy": (("state", "trends"), "all", False, True), "S1-ns": (("state",), "nonstudy", False, False),
    "S1-p25": (("state",), "p25", False, False), "S1-p50": (("state",), "p50", False, False),
    "S3-p25": (("state", "trends"), "p25", False, False), "S3-p50": (("state", "trends"), "p50", False, False),
}
T_SPECS = {"T1": "all", "T1-p25": "p25", "T1-p50": "p50"}


def work_dir() -> Path:
    return Path(os.environ.get("TIMING_WORK_DIR", C.RESULTS_DIR / "extensions" / "timing" / "runs"))


def onset_week(p, q=ONSET_SHARE) -> int:
    """First week (1-based) whose cumulative share reaches ``q`` (10 %)."""
    return int(np.searchsorted(np.cumsum(normalise(p)), q - 1e-12) + 1)


def onset_errors(preds, rows, P, t):
    """DataFrame (geocode, model, onset_err, onset_true, onset_pred)."""
    n = int(P.nweeks[t])
    out = []
    for i, m in enumerate(rows):
        tr = onset_week(P.Yw[m, t, :n] / P.N[m, t])
        for name, pr in preds.items():
            p = pr[i, :n]
            if not np.isfinite(p).all():
                continue
            o = onset_week(p)
            out.append(dict(geocode=int(P.geocodes[m]), model=name, onset_err=abs(o - tr), onset_true=tr, onset_pred=o))
    return pd.DataFrame(out)


class _Cache:
    def __init__(self):
        self.c = {}

    def get(self, P, F, cls=DMModel):
        key = (tuple(int(y) for y in P.years), F, cls.__name__)
        if key not in self.c:
            self.c = {k: v for k, v in self.c.items() if k[0] == key[0]}
            self.c[key] = cls(len(P.geocodes), len(P.years), 53, 12, F, P.region, P.valid_w, True, "dm")
        return self.c[key]


def fold_features(P, sig, tm, fold, need):
    """New feature blocks for one fold. Returns dict of blocks + diagnostics."""
    tym = np.isin(P.years, list(fold["train_years"]))
    out, info = {}, {}
    for v in sorted({SPECS[n][1] for n in need if n in SPECS and "state" in SPECS[n][0]}
                    | {T_SPECS[n] for n in need if n in T_SPECS}):
        Xs, inf = TF.state_block(sig[f"pool_{v}"], P, tym, tm, sig["signal_missing"])
        out[f"state_{v}"] = Xs
        info[f"state_{v}"] = dict(n_rows_with_curve=int(inf["ok"].sum()), mu=inf["mu"].round(5).tolist(),
                                  sd=inf["sd"].round(5).tolist())
    if any(n in SPECS and "trends" in SPECS[n][0] for n in need):
        Xs, inf = TF.trends_block(sig["trends"], P, tym, tm)
        out["trends"] = Xs
        info["trends"] = dict(n_rows_with_curve=int(inf["ok"].sum()), mu=inf["mu"].round(5).tolist(),
                              sd=inf["sd"].round(5).tolist())
    return out, info


def feature_audit(P, sig, tm, fold, rng_seed=0):
    """Leakage audit of the new features for one fold (all must be True).

    * ``target_not_in_pool``: no study municipality is a member of its own pool (every variant).
    * ``state_clim_train_only`` / ``trends_clim_train_only``: the state / search
      climatologies and the standardisation constants are unchanged when every
      non-training year of the signal is replaced by random numbers.
    * ``shift_template_train_only``: the T1 template and the training-year shifts are unchanged likewise.
    * ``standardise_train_rows``: the standardisation uses training-mask rows only
      (recomputed with the non-training rows' features randomised).
    """
    tym = np.isin(P.years, list(fold["train_years"]))
    rng = np.random.default_rng(rng_seed)
    res = {}
    res["target_not_in_pool"] = bool(all(int(g) not in set(pm[int(g)].tolist())
                                         for pm in sig["pool_maps"].values() for g in P.geocodes))

    def perturbed(V):
        W = V.copy()
        W[:, ~tym] = rng.gamma(1.0, 50.0, size=W[:, ~tym].shape)
        return W
    V = sig["pool_all"]
    _, i1 = TF.state_block(V, P, tym, tm, sig["signal_missing"])
    _, i2 = TF.state_block(perturbed(V), P, tym, tm, sig["signal_missing"])
    res["state_clim_train_only"] = bool(np.array_equal(i1["clim"], i2["clim"], equal_nan=True)
                                        and np.allclose(i1["mu"], i2["mu"], rtol=0, atol=0)
                                        and np.allclose(i1["sd"], i2["sd"], rtol=0, atol=0))
    if "trends" in sig:
        V = sig["trends"]
        _, i1 = TF.trends_block(V, P, tym, tm)
        _, i2 = TF.trends_block(perturbed(V), P, tym, tm)
        res["trends_clim_train_only"] = bool(np.array_equal(i1["clim"], i2["clim"], equal_nan=True)
                                             and np.array_equal(i1["mu"], i2["mu"]) and np.array_equal(i1["sd"], i2["sd"]))
    V = sig["pool_all"]
    D1, j1 = TF.shifts(V, P, tym, sig["signal_missing"])
    D2, j2 = TF.shifts(perturbed(V), P, tym, sig["signal_missing"])
    res["shift_template_train_only"] = bool(np.array_equal(j1["template"], j2["template"], equal_nan=True)
                                            and np.array_equal(D1[:, tym], D2[:, tym]))
    X = rng.normal(size=(len(P.geocodes), len(P.years), 53, 2))
    _, mu1, sd1 = TF.standardise(X, tm, P.valid_w)
    X2 = X.copy()
    X2[~tm] = rng.normal(size=X2[~tm].shape) * 100
    _, mu2, sd2 = TF.standardise(X2, tm, P.valid_w)
    res["standardise_train_rows"] = bool(np.array_equal(mu1, mu2) and np.array_equal(sd1, sd2))
    return res


def run_fold(fold, Pfull, sig_full, Xc, ey, ew, oni, models, warm, ref_cache, cache, seed=C.SEED):
    tf = time.time()
    # 1. reference models with the existing official driver (same fold, same settings)
    refs = [m for m in models if m in REF_MODELS]
    P, t, rows, preds, phi, flats, log = OFF.run_fold(fold, Pfull, Xc, ey, ew, oni, refs, warm, ref_cache, seed)
    log["ref_wall_s"] = log.pop("wall_s")
    # 2. same panel quantities for the new models (deterministic recomputation)
    T = len(P.years)
    train_ok, eval_ok = row_masks(P)
    tm = fold_train_mask(P, train_ok, fold)
    w = tm.astype(float)
    h, _, _ = FS.offsets(P, tm, True, C.EPS_H)
    Z, _ = size_covariates(P, tm, None, True)
    X20, _ = FC.fold_climate_features(P, Xc, ey, ew, climate_years(fold), oni=oni, weekly=True)
    sig = SG.subset_signals(sig_full, P.years)
    new = [m for m in models if m not in REF_MODELS]
    blocks, finfo = fold_features(P, sig, tm, fold, new)
    log["features"] = finfo
    log["feature_audit"] = feature_audit(P, sig, tm, fold)
    M = len(P.geocodes)
    fold_years = sorted(set(fold["train_years"]) | {fold["test_year"]})
    yidx = np.where(np.isin(P.years, fold_years))[0]
    perm_tt = SH.fold_year_permutations(M, T, yidx, seed=seed)
    dm_b4s = ref_cache.get(P, OFF.N_LOCAL)
    zc, Xs_of, dms, shifts_log = {}, {}, {}, {}
    for name in new:
        t0 = time.time()
        if name == "S3-wyTest":
            continue
        if name in SPECS:
            blk, pool, clim, wy = SPECS[name]
            parts = [blocks[f"state_{pool}"] if b == "state" else blocks["trends"] for b in blk]
            X = np.concatenate(parts, -1)
            if wy:
                X = SH.shuffle_years(X, perm_tt)
            if clim:
                X = np.concatenate([X, X20], -1)
            dm = cache.get(P, X.shape[-1])
            hh = h
        else:   # T1 family: year-specific shifted offset, no new coefficient
            pool = T_SPECS[name]
            tym = np.isin(P.years, list(fold["train_years"]))
            D, dinf = TF.shifts(sig[f"pool_{pool}"], P, tym, sig["signal_missing"])
            hh = TF.shifted_offsets(h, D, P.nweeks)
            X = np.zeros((M, T, 53, 1))
            dm = cache.get(P, 1, YearOffsetDM)
            Dt = D[rows, t]
            shifts_log[name] = dict(mean_abs_shift_test=float(np.abs(Dt).mean()),
                                    share_nonzero_test=float((Dt != 0).mean()),
                                    hist_test={int(k): int(v) for k, v in zip(*np.unique(Dt, return_counts=True))},
                                    shifts_test=Dt.tolist())
        x0 = warm.get(name)
        if x0 is None or len(x0) != dm.n_params:
            x0 = transfer_params(dm_b4s, flats["B4s"], dm)
        flat, info = dm.fit(hh, Z, X, P.Yw, P.N, w, x0=x0)
        pi = dm.predict(flat, hh, Z, X)
        par = dm.params(flat)
        preds[name] = pi[rows, t]
        phi[name] = np.exp(par["logphi"])[P.region[rows]]
        flats[name] = flat
        dms[name], Xs_of[name] = (dm, hh), X
        log["fits"][name] = dict(**info, wall_s=round(time.time() - t0, 2), phi=np.exp(par["logphi"]).round(3).tolist(),
                                 theta=par["theta"].round(4).tolist(), delta=par["delta"].round(4).tolist())
        # zero-coefficient equivalence with B4s (S: new coefs = 0; T1: shifts = 0)
        if name in ("S1", "S2", "S3", "S3C"):
            zc[name] = zero_coef_maxdiff(dm_b4s, flats["B4s"], dm, h, Z, X, P.valid_w)
        elif name == "T1":
            zc[name] = zero_coef_maxdiff(dm_b4s, flats["B4s"], dm, h, Z, X, P.valid_w,
                                         h_new=TF.shifted_offsets(h, np.zeros((M, T), int), P.nweeks))
    if "S3-wyTest" in new and "S3" in flats:
        dm, hh = dms["S3"]
        Xs = SH.shuffle_test_year(Xs_of["S3"], t, SH.fold_test_permutation(M, T, yidx, seed=seed))
        preds["S3-wyTest"] = dm.predict(flats["S3"], hh, Z, Xs)[rows, t]
    log["zero_coef_maxdiff"] = zc
    log["shifts"] = shifts_log
    order = [m for m in models if m in preds]
    preds = {m: preds[m] for m in order}
    log["wall_s"] = round(time.time() - tf, 1)
    return P, t, rows, preds, phi, flats, log


def main(design="A", subset=300, climate="power", models=None, out_dir=None, test_years=None, seed=C.SEED):
    t_start = time.time()
    models = models or MODELS[design]
    tag = f"timing_design{design}_{climate}_{subset}"
    out = Path(out_dir or work_dir()) / tag
    fdir = out / "folds"
    fdir.mkdir(parents=True, exist_ok=True)
    d, clim, muni = BD.load_tables(climate)
    from ...experiments.brazil_loyo import load_subset
    geos = load_subset(subset, d, muni, write_subset=False)
    Pfull = BD.build_panel(d, clim, geos)
    assert Pfull.years.max() <= C.max_loaded_year()
    uf_of = dict(zip(muni.geocode, muni.uf))
    study_ufs = [uf_of[int(g)] for g in Pfull.geocodes]
    exy = SG.excluded_state_years_from(d, muni)
    sig = SG.build_signals(Pfull, study_ufs, excluded_state_years=exy)
    # pool invariance: the target's own counts never reach its pool
    pool_check = _pool_invariance(Pfull, sig)
    Xc, ey, ew = BD.climate_matrix(clim, Pfull.geocodes)
    oni = BD.oni_vector(BD.load_oni_weekly(), ey, ew)
    folds = design_folds(design, (), test_years)
    info = dict(design=design, subset=subset, climate=climate, models=models, n_municipalities=len(Pfull.geocodes),
                panel_years=Pfull.years.tolist(), folds=[dict(f, train_years=list(f["train_years"])) for f in folds],
                run_final_2024=C.RUN_FINAL_2024, seed=seed, pen=C.PEN, excluded_state_years=sorted(map(list, exy)),
                pool_sizes={v: dict(median=float(np.median(sig[f"pool_size_{v}"])), min=int(sig[f"pool_size_{v}"].min()),
                                    n_empty=int((sig[f"pool_size_{v}"] == 0).sum()))
                            for v in ("all", "nonstudy", "p25", "p50")},
                trends_national_fallback=int(sig["trends_fallback"].sum()),
                pool_invariance_check=pool_check, eps=TF.EPS, min_signal_total=TF.MIN_SIGNAL_TOTAL,
                state_lags=TF.STATE_LAGS, trends_lags=TF.TRENDS_LAGS, max_shift=TF.MAX_SHIFT,
                pool_seed=SG.POOL_SEED, onset_share=ONSET_SHARE, status="EXPLORATORY")
    (out / "run_info.json").write_text(json.dumps(info, indent=1, default=str))
    pd.DataFrame(dict(geocode=Pfull.geocodes, uf=study_ufs, trends_fallback=sig["trends_fallback"],
                      **{f"pool_size_{v}": sig[f"pool_size_{v}"] for v in ("all", "nonstudy", "p25", "p50")})
                 ).to_csv(out / "pool_sizes.csv", index=False)
    print(f"[{tag}] M={len(Pfull.geocodes)} folds={len(folds)} models={models}", flush=True)
    ref_cache, cache, warm = OFF._Models(Pfull), _Cache(), {}
    for fold in folds:
        fid = fold["fold_id"]
        f_log, f_npz, f_met = fdir / f"{fid}.json", fdir / f"{fid}.npz", fdir / f"{fid}_metrics.csv"
        if f_log.exists() and f_npz.exists() and f_met.exists():
            z = np.load(f_npz)
            warm.update({k[6:]: z[k] for k in z.files if k.startswith("flat__")})
            print(f"[{tag}] {fid}: checkpoint found, skipped", flush=True)
            continue
        P, t, rows, preds, phi, flats, log = run_fold(fold, Pfull, sig, Xc, ey, ew, oni, models, warm,
                                                      ref_cache, cache, seed)
        warm.update(flats)
        np.savez_compressed(f_npz, geocodes=P.geocodes[rows], rows=rows, test_year=fold["test_year"],
                            **{f"pred__{k}": v for k, v in preds.items()},
                            **{f"phi__{k}": v for k, v in phi.items()},
                            **{f"flat__{k}": v for k, v in flats.items()})
        ts = time.time()
        df = score_rows(preds, rows, P, t, P.Yw, P.nweeks, phi, [m for m in PROB if m in phi],
                        seed=[seed, fold["test_year"], fold["k"]], floor_seed=[0, fold["test_year"]])
        on = onset_errors(preds, rows, P, t)
        df = df.merge(on, on=["geocode", "model"], how="left")
        df.insert(0, "fold_id", fid); df.insert(0, "design", design)
        df["k"], df["climate"] = fold["k"], climate
        df.to_csv(f_met, index=False)
        log["score_s"] = round(time.time() - ts, 1)
        f_log.write_text(json.dumps(log, indent=1, default=float))
        print(f"[{tag}] {fid}: k={fold['k']}, {len(rows)} eval rows, fit {log['wall_s']}s + score {log['score_s']}s, "
              f"feature audit {'pass' if all(log['feature_audit'].values()) else 'FAIL'}, "
              f"official audit {'pass' if all(log['leakage'].values()) else 'FAIL'}, zero-coef {log['zero_coef_maxdiff']}",
              flush=True)
        print(df.groupby("model", sort=False)[["jsd", "peak_err", "onset_err"]].mean().round(4).T.to_string(), flush=True)
    return assemble(out, tag, folds, t_start)


def _pool_invariance(Pfull, sig):
    """Rebuild every study municipality's 'all' pool with its own InfoDengue counts
    replaced by 1e9: the pool must not change (the target never contributes)."""
    years = [int(y) for y in Pfull.years]
    d = SG.load_infodengue_counts(years)
    A, geos, ufs = SG.count_cube(d, years)
    gi = {int(g): i for i, g in enumerate(geos)}
    pm = sig["pool_maps"]["all"]
    ok = True
    A2 = A.copy()
    for m, g in enumerate(Pfull.geocodes):
        A2[gi[int(g)]] = 1e9
        idx = [gi[int(x)] for x in pm[int(g)]]
        v = A2[idx].sum(0) if idx else np.zeros(A.shape[1:])
        ok &= bool(np.array_equal(v, sig["pool_all"][m]))
        A2[gi[int(g)]] = A[gi[int(g)]]
    return bool(ok)


def assemble(out, tag, folds, t_start=None):
    fdir = out / "folds"
    mets, logs = [], []
    for f in folds:
        fid = f["fold_id"]
        mets.append(pd.read_csv(fdir / f"{fid}_metrics.csv"))
        logs.append(json.loads((fdir / f"{fid}.json").read_text()))
    df = pd.concat(mets, ignore_index=True)
    assert (df.year <= C.max_loaded_year()).all()
    df.to_csv(out / f"metrics_detail_{tag}.csv", index=False)
    checks = dict(
        n_folds=len(folds), status="EXPLORATORY",
        official_leakage_audit_all_pass=bool(all(all(lg["leakage"].values()) for lg in logs)),
        feature_audit_all_pass=bool(all(all(lg["feature_audit"].values()) for lg in logs)),
        feature_audit={lg["fold_id"]: lg["feature_audit"] for lg in logs},
        zero_coef_maxdiff={lg["fold_id"]: lg["zero_coef_maxdiff"] for lg in logs},
        zero_climate_maxdiff={lg["fold_id"]: lg["zero_climate_maxdiff"] for lg in logs},
        exact_sums="asserted for every allocation and every draw inside score_rows",
        years_scored=sorted(int(y) for y in df.year.unique()),
        fold_wall_s={lg["fold_id"]: lg["wall_s"] + lg.get("score_s", 0) for lg in logs},
        fit_convergence={lg["fold_id"]: {m: dict(success=fi.get("success"), grad_inf=fi.get("grad_inf"), nit=fi.get("nit"))
                                         for m, fi in lg["fits"].items()} for lg in logs},
    )
    if t_start is not None:
        checks["run_wall_s"] = round(time.time() - t_start, 1)
    (out / f"checks_{tag}.json").write_text(json.dumps(checks, indent=1, default=float))
    print(df.groupby("model", sort=False)[["jsd", "peak_err", "onset_err", "emd"]].mean().round(4).to_string(), flush=True)
    return df


def cli():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--design", default="A", choices=["A", "C"])
    ap.add_argument("--subset", type=int, default=300)
    ap.add_argument("--climate", default="power")
    ap.add_argument("--models", default=None)
    ap.add_argument("--years", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    models = a.models.split(",") if a.models else None
    years = [int(y) for y in a.years.split(",")] if a.years else None
    main(a.design, a.subset, a.climate, models, a.out, years)


if __name__ == "__main__":
    os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
    cli()
