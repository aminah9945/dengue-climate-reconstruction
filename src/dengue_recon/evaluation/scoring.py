"""Scoring of saved predictions (Step 6).

For every model x evaluation municipality-year: JSD, peak-week error, MAE on
shares, EMD, week-of-50 % error, MAE on allocated counts and the JSD noise floor.
For probabilistic models, 500 draws Dirichlet(phi*pi) -> Multinomial(N) give 80 %
and 95 % interval coverage of the weekly counts and the sample CRPS. Exact sums
are asserted for every allocation and every draw.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .. import config as C
from ..models.dirichlet_multinomial import PROB_MODELS, draw_counts
from .allocation import allocate_integers
from .metrics import crps_samples, emd_weeks, jsd, jsd_floor, mae_shares, normalise, peak_week_error, week50_error


def score(tag, n_draws=C.N_DRAWS, out_dir: Path | None = None, prob_models=None):
    """Score ``preds_{tag}.npz`` in ``out_dir`` (default ``config.BRAZIL_DIR``) and write
    ``metrics_detail_{tag}.csv`` there. Returns the long metrics DataFrame."""
    out_dir = Path(out_dir or C.BRAZIL_DIR)
    prob_models = PROB_MODELS if prob_models is None else list(prob_models)
    z = np.load(out_dir / f"preds_{tag}.npz")
    geos, years, region, N, Y, eval_ok, nsl = (z[k] for k in ["geocodes", "years", "region", "N", "Y", "eval_ok", "nsl"])
    models = [k[6:] for k in z.files if k.startswith("pred__")]
    rng = np.random.default_rng(C.SEED)
    rng_floor = np.random.default_rng(0)
    out = []
    sum_fail = 0
    for m, t in zip(*np.where(eval_ok)):
        n, tot = int(nsl[t]), int(N[m, t])
        truth_c = Y[m, t, :n]
        truth = truth_c / tot
        floor = jsd_floor(truth, tot, rng_floor)
        for name in models:
            p = z[f"pred__{name}"][m, t, :n]
            if not np.isfinite(p).all():
                continue
            p = normalise(p)
            alloc = allocate_integers(p, tot)
            sum_fail += int(alloc.sum() != tot)
            r = dict(geocode=int(geos[m]), year=int(years[t]), region=C.REGIONS[region[m]], N=tot,
                     model=name, jsd=jsd(truth, p), peak_err=peak_week_error(truth, p),
                     mae_share=mae_shares(truth, p), emd=emd_weeks(truth, p),
                     w50_err=week50_error(truth, p), mae_counts=float(np.mean(np.abs(alloc - truth_c))),
                     jsd_floor=floor)
            if name in prob_models:
                phi = z[f"phi__{name}"][m, t]
                D = draw_counts(p, phi, tot, n_draws, rng, n)        # (draws, n); asserts sums
                Sm = D.T.astype(float)
                r["cov80"] = float(np.mean((truth_c >= np.quantile(Sm, .1, 1)) & (truth_c <= np.quantile(Sm, .9, 1))))
                r["cov95"] = float(np.mean((truth_c >= np.quantile(Sm, .025, 1)) & (truth_c <= np.quantile(Sm, .975, 1))))
                r["crps"] = float(np.mean(crps_samples(truth_c.astype(float), Sm)))
            out.append(r)
    assert sum_fail == 0, "allocate_integers did not preserve totals"
    df = pd.DataFrame(out)
    df.to_csv(out_dir / f"metrics_detail_{tag}.csv", index=False)
    print(df.groupby("model")[["jsd", "peak_err", "emd"]].mean().round(4).to_string(), flush=True)
    return df


def score_rows(preds, rows, P, t, Y, nsl, phi=None, prob_models=(), seed=C.SEED, floor_seed=0):
    """Score one fold's predictions (official designs); same columns as ``metrics_detail``.

    ``preds`` {model: (len(rows), S)} shares for municipalities ``rows`` in panel year
    ``t``; ``phi`` {model: (len(rows),)} for ``prob_models`` (500 DM draws -> coverage,
    CRPS). Random generators are seeded per fold (``seed``, ``floor_seed``), so a
    fold scores identically whether it runs alone or after a resume.
    """
    rng = np.random.default_rng(seed)
    rng_floor = np.random.default_rng(floor_seed)
    out, sum_fail = [], 0
    n = int(nsl[t])
    for i, m in enumerate(rows):
        tot = int(P.N[m, t])
        truth_c = Y[m, t, :n]
        truth = truth_c / tot
        floor = jsd_floor(truth, tot, rng_floor)
        for name, pr in preds.items():
            p = pr[i, :n]
            if not np.isfinite(p).all():
                continue
            p = normalise(p)
            alloc = allocate_integers(p, tot)
            sum_fail += int(alloc.sum() != tot)
            r = dict(geocode=int(P.geocodes[m]), year=int(P.years[t]), region=C.REGIONS[P.region[m]], N=tot,
                     model=name, jsd=jsd(truth, p), peak_err=peak_week_error(truth, p),
                     mae_share=mae_shares(truth, p), emd=emd_weeks(truth, p),
                     w50_err=week50_error(truth, p), mae_counts=float(np.mean(np.abs(alloc - truth_c))),
                     jsd_floor=floor)
            if name in prob_models:
                D = draw_counts(p, phi[name][i], tot, C.N_DRAWS, rng, n)
                Sm = D.T.astype(float)
                r["cov80"] = float(np.mean((truth_c >= np.quantile(Sm, .1, 1)) & (truth_c <= np.quantile(Sm, .9, 1))))
                r["cov95"] = float(np.mean((truth_c >= np.quantile(Sm, .025, 1)) & (truth_c <= np.quantile(Sm, .975, 1))))
                r["crps"] = float(np.mean(crps_samples(truth_c.astype(float), Sm)))
            out.append(r)
    assert sum_fail == 0, "allocate_integers did not preserve totals"
    return pd.DataFrame(out)
