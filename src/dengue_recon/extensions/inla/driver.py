"""INLA disaggregation models on validation design A (EXPLORATORY, not pre-registered).

The base paper (Lim et al. 2026) selected **scaled INLA** for annual-to-monthly
disaggregation: a negative-binomial INLA model of the monthly counts (country
intercepts, AR1 over time per country, three-level cyclic RW1 seasonality,
AR1 year effects per country) fitted with the held-out country-years set to
NA; the posterior-mean predictions of a held-out year are then rescaled so that
they add up to its known annual total. This module runs the same model at
municipality-week resolution (``R/fit_inla.R``), in the official design A folds
(train 2010 ... T-1, test T = 2012 ... 2023), with the same data, exclusions,
metrics and per-fold seeds as ``experiments.official``.

Models written to ``metrics_detail``:

==========================  ==================================================
INLA-unscaled               posterior mean of the held-out weeks (counts do not
                            add up to N); shape metrics use the normalised curve,
                            ``mae_counts`` the raw counts, coverage/CRPS raw
                            negative-binomial posterior-predictive draws
INLA-scaled                 the same curve rescaled to N (largest remainder);
                            draws rescaled per draw to N, then integer-allocated
INLA-scaled-clim20          + 20 climate features of M1 (15 local + 5 ONI)
INLA-scaled-clim15          + 15 local climate features (as M1-noONI)
INLA-seasonal-only          diagnostic: exp(seasonal terms) of the base fit only,
                            i.e. without the AR1(time) carry-over (not a model of
                            the base paper; used for the shape-variation check)
==========================  ==================================================

Note: rescaling by a constant does not change the shape, so INLA-unscaled and
INLA-scaled have identical JSD, peak-week error, EMD and MAE on shares; they
differ only in the count metrics (``mae_counts``, coverage, CRPS).

Leakage rules are those of the official driver: the fold panel holds only the
years up to T; the test year's counts are NA in the INLA data; climatology and
ONI scaling use 2009 + training years only; the official leakage audit is run in
every fold. 2024 is never loaded while ``config.RUN_FINAL_2024`` is False.

Checkpoints: per fold and variant, ``work/<fold_id>/<variant>_done.json`` (R fit),
and per fold ``folds/<fold_id>.json`` (written last), ``.npz`` and ``_metrics.csv``.
A rerun skips finished folds and finished R fits.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ... import config as C
from ...checks.leakage import official_leakage_audit
from ...data import panel as BD
from ...evaluation.allocation import allocate_integers
from ...evaluation.cv import (assert_final_year_locked, climate_years, design_folds, fold_train_mask,
                              panel_years, row_masks)
from ...evaluation.metrics import crps_samples, jsd, normalise
from ...evaluation.scoring import score_rows
from ...features import climate as FC

R_SCRIPT = Path(__file__).resolve().parent / "R" / "fit_inla.R"
INLA_DIR = Path(os.environ.get("INLA_DIR", C.RESULTS_DIR / "extensions" / "inla"))
#: R variant -> number of climate features
VARIANTS = {"base": 0, "clim20": 20, "clim15": 15}
#: R variant -> scaled model name
SCALED_NAME = {"base": "INLA-scaled", "clim20": "INLA-scaled-clim20", "clim15": "INLA-scaled-clim15"}
MODELS = ["INLA-unscaled", "INLA-scaled", "INLA-scaled-clim20", "INLA-scaled-clim15", "INLA-seasonal-only"]
PROB_MODELS = ["INLA-unscaled", "INLA-scaled", "INLA-scaled-clim20", "INLA-scaled-clim15"]
N_SAMPLES = 500           # posterior draws for coverage / CRPS (same number as the DM draws)
BATCH_SIZE = 60           # municipalities per INLA fit (memory / time; see docs/INLA.md)


def run_tag(climate, subset, design="A"):
    return f"design{design}_{climate}_{subset}"


# ------------------------------------------------------------------ data export
def municipality_batches(region, batch_size=BATCH_SIZE):
    """Split municipalities into ceil(M / batch_size) batches, stratified by region:
    within each region (panel order) municipality i goes to batch i mod n_batches.
    Returns a list of index arrays. One batch when M <= batch_size."""
    M = len(region)
    nb = max(1, int(np.ceil(M / batch_size)))
    lab = np.empty(M, int)
    for r in np.unique(region):
        idx = np.where(region == r)[0]
        lab[idx] = np.arange(len(idx)) % nb
    return [np.where(lab == b)[0] for b in range(nb)]


def fold_long_table(P, t, train_mask, eval_rows, X, munis=None):
    """Long INLA table of one fold: training rows (observed y) + test rows (y = NA).

    ``P`` fold panel, ``t`` index of the test year, ``train_mask`` (M, T), ``eval_rows``
    municipalities scored in the test year, ``X`` (M, T, 53, F) climate features,
    ``munis`` optional subset (batch) of municipalities, renumbered 1..len(munis).
    Rows: every valid week of every training municipality-year and of every
    evaluated test municipality-year. Returns (DataFrame, (m, tt, s) index arrays).
    """
    M, T = train_mask.shape
    assert not train_mask[:, t].any(), "a test-year row is marked as training"
    use = train_mask.copy()
    test_rows = np.zeros((M, T), bool)
    test_rows[eval_rows, t] = True
    use |= test_rows
    if munis is not None:
        keep = np.zeros(M, bool); keep[munis] = True
        use &= keep[:, None]
    sel = use[:, :, None] & P.valid_w[None, :, :]
    m, tt, s = np.nonzero(sel)
    k0 = int(P.week_index[P.valid_w].min())
    is_test = test_rows[m, tt]
    y = P.Yw[m, tt, s].astype(float)
    y[is_test] = np.nan
    mnum = np.searchsorted(np.unique(m), m) + 1 if munis is not None else m + 1
    df = pd.DataFrame(dict(row=np.arange(1, len(m) + 1), muni=mnum, region=P.region[m] + 1, slot=s + 1,
                           tseq=P.week_index[tt, s] - k0 + 1, year=tt + 1, y=y, test=is_test.astype(int)))
    for f in range(X.shape[-1]):
        df[f"x{f + 1:02d}"] = np.round(X[m, tt, s, f], 6)
    assert df.loc[df.test == 1, "y"].isna().all() and df.loc[df.test == 0, "y"].notna().all()
    assert (df.year <= t + 1).all()
    return df, (m, tt, s)


def run_r(wdir: Path, variant: str, n_samples: int, seed: int, n_threads: int, log: Path, theta: Path | None = None):
    """Fit one variant with R-INLA unless its checkpoint exists. Returns the run info dict.
    ``theta``: optional hyperparameter file of an earlier fit (optimiser warm start)."""
    done = wdir / f"{variant}_done.json"
    if not done.exists():
        cmd = ["Rscript", "--vanilla", str(R_SCRIPT), str(wdir), variant, str(n_samples), str(seed), str(n_threads),
               str(theta) if theta is not None and Path(theta).exists() else "none"]
        with open(log, "a") as fh:
            fh.write(f"$ {' '.join(cmd)}\n"); fh.flush()
            r = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT)
        if r.returncode != 0 or not done.exists():
            raise RuntimeError(f"R-INLA failed for {wdir.name}/{variant}; see {log}")
    return json.loads(done.read_text())


def read_draws(wdir: Path, variant: str, n_test: int):
    """(eta (n_test, S), size (S,)) posterior draws written by ``fit_inla.R``."""
    eta = np.fromfile(wdir / f"{variant}_eta.bin", dtype="<f8")
    size = np.fromfile(wdir / f"{variant}_size.bin", dtype="<f8")
    S = len(size)
    assert eta.size == n_test * S
    return eta.reshape(S, n_test).T, size


# ------------------------------------------------------------------ predictive draws
def nb_draws(eta, size, rng):
    """Negative-binomial posterior-predictive counts (n, S): mu = exp(eta), var = mu + mu^2/size."""
    mu = np.exp(np.minimum(eta, 30.0))
    p = size[None, :] / (size[None, :] + mu)
    return rng.negative_binomial(np.broadcast_to(size[None, :], mu.shape), p).astype(float), mu


def rescale_draws(Y, mu, total):
    """Scaled INLA per draw: Y_s * N / sum(Y_s) (base paper), then largest-remainder integers
    so that every draw adds up exactly to N. A draw with sum 0 uses its mean curve mu_s."""
    out = np.empty(Y.shape, dtype=int)
    for s in range(Y.shape[1]):
        v = Y[:, s] if Y[:, s].sum() > 0 else mu[:, s]
        out[:, s] = allocate_integers(v / v.sum(), int(total))
        assert out[:, s].sum() == int(total)
    return out


def prob_scores(truth_c, Sm):
    """cov80, cov95, CRPS of draws ``Sm`` (n, S) against counts ``truth_c`` (same as ``score_rows``)."""
    return dict(cov80=float(np.mean((truth_c >= np.quantile(Sm, .1, 1)) & (truth_c <= np.quantile(Sm, .9, 1)))),
                cov95=float(np.mean((truth_c >= np.quantile(Sm, .025, 1)) & (truth_c <= np.quantile(Sm, .975, 1)))),
                crps=float(np.mean(crps_samples(truth_c.astype(float), Sm))))


# ------------------------------------------------------------------ one fold
def run_fold(fold, Pfull, Xc, ey, ew, oni, variants, wroot, n_samples, n_threads, seed=C.SEED, log=None,
             prev_fid=None):
    """Export, fit (R), predict and score one design-A fold. Returns (metrics df, shape df, log, arrays)."""
    tf = time.time()
    fid = fold["fold_id"]
    P = BD.subset_panel(Pfull, panel_years(fold, Pfull.years))
    assert_final_year_locked(P)
    t = int(np.where(P.years == fold["test_year"])[0][0])
    train_ok, eval_ok = row_masks(P)
    tm = fold_train_mask(P, train_ok, fold)
    cy = climate_years(fold)
    X20, cinfo = FC.fold_climate_features(P, Xc, ey, ew, cy, oni=oni, weekly=True)
    aud = official_leakage_audit(P, Xc, ey, ew, tm, fold, cy, oni=oni, weekly=True)
    rows = np.where(eval_ok[:, t])[0]
    n = int(P.nweeks[t])
    wdir = Path(wroot) / fid
    batches = municipality_batches(P.region)
    rpos = {int(m): i for i, m in enumerate(rows)}
    grids = {v: {c: np.full((len(rows), 53), np.nan) for c in ("fit_mean", "eta_mean", "ar1_time", "seasonal",
                                                                 "climate")} for v in variants}
    eta_g = {v: np.full((len(rows), 53, max(n_samples, 1)), np.nan) for v in variants}
    size_g = {v: np.full((len(rows), max(n_samples, 1)), np.nan) for v in variants}
    flog = dict(fold, train_years=list(fold["train_years"]), clim_years=list(cy), panel_years=P.years.tolist(),
                n_train_rows=int(tm.sum()), n_eval_rows=int(len(rows)), n_batches=len(batches),
                n_obs_weeks=0, n_test_weeks=0, clim_nan=int(cinfo["n_nan"]), leakage=aud, test_y_all_na=True,
                variants=list(variants), fits={})
    assert variants[0] == "base", "the base variant is fitted first (seasonal-only diagnostic, warm starts)"
    for b, munis in enumerate(batches):
        bdir = wdir / f"b{b}"
        bdir.mkdir(parents=True, exist_ok=True)
        df_long, (mi, ti, si) = fold_long_table(P, t, tm, rows, X20, munis=munis)
        if not (bdir / "data.csv").exists():
            df_long.to_csv(bdir / "data.csv", index=False, na_rep="NA", float_format="%.6g")
        te = np.where(df_long.test.values == 1)[0]
        gi = np.array([rpos[int(m)] for m in mi[te]], int)    # position in the (len(rows), 53) grid
        gs = si[te]
        flog["n_obs_weeks"] += int((df_long.test == 0).sum())
        flog["n_test_weeks"] += int(len(te))
        flog["test_y_all_na"] &= bool(df_long.loc[df_long.test == 1, "y"].isna().all())
        for v in variants:
            # warm start: same batch and variant in the previous fold, else this fold's base fit
            theta = None
            if prev_fid is not None:
                theta = Path(wroot) / prev_fid / f"b{b}" / f"{v}_theta.txt"
            if (theta is None or not theta.exists()) and v != "base":
                theta = bdir / "base_theta.txt"
            info = run_r(bdir, v, n_samples, seed % (2 ** 31 - 1), n_threads, log or (bdir / "R.log"), theta)
            hy = pd.read_csv(bdir / f"{v}_hyper.csv")
            info["hyper"] = {r.parameter: [round(r.q50, 5), round(r.q025, 5), round(r.q975, 5)]
                             for r in hy.itertuples()}
            flog["fits"][f"{v}/b{b}"] = info
            if len(te) == 0:
                continue
            pr = pd.read_csv(bdir / f"{v}_pred.csv").set_index("row").loc[df_long.row.values[te]]
            for col in grids[v]:
                grids[v][col][gi, gs] = pr[col].values
            if n_samples > 0:
                eta, size = read_draws(bdir, v, len(te))
                eta_g[v][gi, gs, :] = eta
                size_g[v][np.unique(gi), :] = size[None, :]
    preds, comp, draws = {}, {}, {}
    for v in variants:
        comp[v] = grids[v]
        preds[SCALED_NAME[v]] = grids[v]["fit_mean"]          # shares = normalised posterior mean
        if v == "base":
            preds["INLA-unscaled"] = grids[v]["fit_mean"]
            sea = grids[v]["seasonal"]
            preds["INLA-seasonal-only"] = np.exp(sea - np.nanmax(sea, 1, keepdims=True))
        if n_samples > 0:
            draws[v] = (eta_g[v], size_g[v])
    order = [m for m in MODELS if m in preds]
    preds = {m: preds[m] for m in order}
    ts = time.time()
    df = score_rows(preds, rows, P, t, P.Yw, P.nweeks, None, (),
                    seed=[seed, fold["test_year"], fold["k"]], floor_seed=[0, fold["test_year"]])
    # unscaled counts: raw posterior means (no rescaling to N)
    if "INLA-unscaled" in preds:
        raw = preds["INLA-unscaled"]
        for i, m in enumerate(rows):
            sel = (df.model == "INLA-unscaled") & (df.geocode == int(P.geocodes[m]))
            truth_c = P.Yw[m, t, :n]
            df.loc[sel, "mae_counts"] = float(np.mean(np.abs(raw[i, :n] - truth_c)))
            df.loc[sel, "pred_total_ratio"] = float(raw[i, :n].sum() / P.N[m, t])
    # posterior-predictive coverage / CRPS (seeded per fold and variant)
    if draws:
        for vi, (v, (g, size)) in enumerate(draws.items()):
            rng = np.random.default_rng([seed, fold["test_year"], fold["k"], 7, vi])
            for i, m in enumerate(rows):
                truth_c = P.Yw[m, t, :n]
                Y, mu = nb_draws(g[i, :n, :], size[i], rng)
                gcode = int(P.geocodes[m])
                if v == "base":
                    sel = (df.model == "INLA-unscaled") & (df.geocode == gcode)
                    for k_, val in prob_scores(truth_c, Y).items():
                        df.loc[sel, k_] = val
                Ys = rescale_draws(Y, mu, P.N[m, t]).astype(float)
                sel = (df.model == SCALED_NAME[v]) & (df.geocode == gcode)
                for k_, val in prob_scores(truth_c, Ys).items():
                    df.loc[sel, k_] = val
    df.insert(0, "fold_id", fold["fold_id"]); df.insert(0, "design", fold["design"])
    df["k"] = fold["k"]
    flog["score_s"] = round(time.time() - ts, 1)
    shape = shape_diagnostics(P, t, rows, comp)
    flog["wall_s"] = round(time.time() - tf, 1)
    return df, shape, flog, dict(geocodes=P.geocodes[rows], rows=rows, test_year=fold["test_year"],
                                 **{f"pred__{k}": v for k, v in preds.items()},
                                 **{f"{c}__{v}": g for v, gg in comp.items() for c, g in gg.items()
                                    if c in ("ar1_time", "seasonal", "climate")})


def shape_diagnostics(P, t, rows, comp):
    """Per evaluated municipality: how much the held-out year's predicted shape differs
    from the seasonal-only shape of the same fit, i.e. what the AR1(time) term (and,
    for the climate variants, the climate term) adds within the test year."""
    n = int(P.nweeks[t])
    out = []
    for i, m in enumerate(rows):
        r = dict(geocode=int(P.geocodes[m]), year=int(P.years[t]))
        seas = comp["base"]["seasonal"][i, :n]
        p_seas = normalise(np.exp(seas - seas.max()))
        truth = P.Yw[m, t, :n] / P.N[m, t]
        for v, g in comp.items():
            full = normalise(g["fit_mean"][i, :n])
            ar = g["ar1_time"][i, :n]
            r[f"{v}_jsd_vs_seasonal"] = jsd(full, p_seas)
            r[f"{v}_peak_shift"] = int(min(abs(int(np.argmax(full)) - int(np.argmax(p_seas))),
                                           n - abs(int(np.argmax(full)) - int(np.argmax(p_seas)))))
            r[f"{v}_ar1_sd"] = float(np.std(ar))
            r[f"{v}_ar1_first_minus_last"] = float(ar[0] - ar[-1])
            r[f"{v}_clim_sd"] = float(np.std(g["climate"][i, :n]))
            r[f"{v}_jsd_truth"] = jsd(truth, full)
        r["seasonal_jsd_truth"] = jsd(truth, p_seas)
        out.append(r)
    return pd.DataFrame(out)


# ------------------------------------------------------------------ run
def main(subset=60, climate="power", variants=("base", "clim20", "clim15"), n_samples=N_SAMPLES,
         n_threads=4, test_years=None, out_dir=None, work_dir=None, seed=C.SEED, max_folds=None):
    """Run INLA on design A; resumable per fold. Returns the assembled metrics DataFrame."""
    t_start = time.time()
    tag = run_tag(climate, subset)
    out = Path(out_dir or INLA_DIR) / tag
    fdir = out / "folds"
    fdir.mkdir(parents=True, exist_ok=True)
    wroot = Path(work_dir or (out / "work"))
    wroot.mkdir(parents=True, exist_ok=True)
    rlog = out / "R_fits.log"
    d, clim, muni = BD.load_tables(climate)
    from ...experiments.brazil_loyo import load_subset
    geos = load_subset(subset, d, muni, write_subset=False)
    Pfull = BD.build_panel(d, clim, geos)
    assert_final_year_locked(Pfull)
    Xc, ey, ew = BD.climate_matrix(clim, Pfull.geocodes)
    oni = BD.oni_vector(BD.load_oni_weekly(), ey, ew)
    folds = design_folds("A", (), test_years)
    info = dict(label="EXPLORATORY (not pre-registered)", design="A", climate=climate, subset=subset,
                variants=list(variants), models=MODELS, n_samples=n_samples, n_threads=n_threads,
                n_municipalities=len(Pfull.geocodes), panel_years=Pfull.years.tolist(),
                folds=[dict(f, train_years=list(f["train_years"])) for f in folds],
                run_final_2024=C.RUN_FINAL_2024, seed=seed, r_script=str(R_SCRIPT.name))
    (out / "run_info.json").write_text(json.dumps(info, indent=1, default=str))
    print(f"[inla {tag}] M={len(Pfull.geocodes)} folds={len(folds)} variants={list(variants)}", flush=True)
    done_now, prev_fid = 0, None
    for fold in folds:
        fid = fold["fold_id"]
        pf, prev_fid = prev_fid, fid
        f_log, f_npz, f_met, f_shp = (fdir / f"{fid}.json", fdir / f"{fid}.npz", fdir / f"{fid}_metrics.csv",
                                      fdir / f"{fid}_shape.csv")
        if f_log.exists() and f_npz.exists() and f_met.exists() and \
                set(variants) <= set(json.loads(f_log.read_text()).get("variants", [])):
            print(f"[inla {tag}] {fid}: checkpoint found, skipped", flush=True)
            continue
        if max_folds is not None and done_now >= max_folds:
            break
        df, shape, flog, arrs = run_fold(fold, Pfull, Xc, ey, ew, oni, variants, wroot, n_samples, n_threads,
                                         seed, rlog, pf)
        df["climate"] = climate
        np.savez_compressed(f_npz, **arrs)
        df.to_csv(f_met, index=False)
        shape.to_csv(f_shp, index=False)
        f_log.write_text(json.dumps(flog, indent=1, default=float))
        done_now += 1
        fits = ", ".join(f"{v} {sum(fi['fit_s'] for k_, fi in flog['fits'].items() if k_.startswith(v + '/')):.0f}s"
                         for v in variants)
        print(f"[inla {tag}] {fid}: {flog['n_eval_rows']} eval rows, {flog['n_obs_weeks']} obs weeks; {fits}; "
              f"total {flog['wall_s']}s; leakage {'pass' if all(flog['leakage'].values()) else 'FAIL'}; "
              f"JSD " + " ".join(f"{m}={v:.4f}" for m, v in df.groupby('model').jsd.mean().items()), flush=True)
    complete = [f for f in folds if (fdir / f"{f['fold_id']}.json").exists() and
                set(variants) <= set(json.loads((fdir / f"{f['fold_id']}.json").read_text()).get("variants", []))]
    if len(complete) < len(folds):
        print(f"[inla {tag}] {len(complete)}/{len(folds)} folds complete; rerun to resume", flush=True)
        return None
    return assemble(out, tag, folds, t_start)


def assemble(out, tag, folds, t_start=None):
    """Concatenate fold outputs: metrics_detail, shape diagnostics, fold log, checks."""
    fdir = out / "folds"
    mets = pd.concat([pd.read_csv(fdir / f"{f['fold_id']}_metrics.csv") for f in folds], ignore_index=True)
    shp = pd.concat([pd.read_csv(fdir / f"{f['fold_id']}_shape.csv").assign(fold_id=f["fold_id"]) for f in folds],
                    ignore_index=True)
    mets.to_csv(out / f"metrics_detail_inla_{tag}.csv", index=False)
    shp.to_csv(out / f"shape_diag_inla_{tag}.csv", index=False)
    rows, logs = [], []
    for f in folds:
        lg = json.loads((fdir / f"{f['fold_id']}.json").read_text())
        logs.append(lg)
        for v, fi in lg["fits"].items():
            h = fi.get("hyper", {})
            rows.append(dict(fold_id=f["fold_id"], test_year=f["test_year"], k=f["k"], variant=v.split("/")[0],
                             batch=v.split("/")[1],
                             n_obs=fi["n_obs"], n_test=fi["n_test"], fit_s=fi["fit_s"], sample_s=fi["sample_s"],
                             ok=fi["ok"], mlik=fi.get("mlik"), inla_version=fi.get("inla_version"),
                             warm_start=fi.get("warm_start"), fn_calls=fi.get("fn_calls"),
                             **{k: val[0] for k, val in h.items()}))
    pd.DataFrame(rows).to_csv(out / f"fold_log_inla_{tag}.csv", index=False)
    checks = dict(
        label="EXPLORATORY (not pre-registered)",
        n_folds=len(folds),
        leakage_audit_all_pass=bool(all(all(lg["leakage"].values()) for lg in logs)),
        test_y_all_na=bool(all(lg["test_y_all_na"] for lg in logs)),
        all_fits_ok=bool(all(fi["ok"] for lg in logs for fi in lg["fits"].values())),
        years_scored=sorted(int(y) for y in mets.year.unique()),
        no_2024=bool((mets.year < C.FINAL_TEST_YEAR).all()) or C.RUN_FINAL_2024,
        fold_wall_s={lg["fold_id"]: lg["wall_s"] for lg in logs},
        total_fold_wall_s=round(sum(lg["wall_s"] for lg in logs), 1),
    )
    if t_start is not None:
        checks["run_wall_s"] = round(time.time() - t_start, 1)
    (out / f"checks_inla_{tag}.json").write_text(json.dumps(checks, indent=1))
    print(mets.groupby("model")[["jsd", "peak_err", "emd", "mae_counts", "cov80", "crps"]].mean().round(4)
          .to_string(), flush=True)
    return mets
