"""Optional M2 (Step 5): gradient-boosted trees with a softmax over the weeks of each
municipality-year (XGBoost custom objective) and base margin h_m(w).

Fixed hyperparameters (no tuning): depth 4, learning rate 0.05, 300 rounds.
Deviation from the specification: no early stopping (the specified "early
stopping on the next fold" would use a held-out year inside LOYO; 300 rounds are
kept fixed instead).

Objective: cross-entropy between observed weekly shares and softmax(margin), one
group per municipality-year (equal weight per training municipality-year).
Adds ``pred__M2`` to ``results/brazil/preds_{tag}.npz`` (which must already exist
from the LOYO driver), then re-scores.

CLI: ``python scripts/run_pipeline.py --step m2 --res weekly --subset 60``.
"""
from __future__ import annotations

import time

import numpy as np

from .. import config as C
from ..data import panel as BD
from ..evaluation.cv import row_masks
from ..evaluation.scoring import score
from ..features import climate as FC
from ..features import seasonal as FS


def group_softmax(margin, gid, ngroups):
    """Softmax of ``margin`` within groups ``gid`` (numerically stabilised by the group max)."""
    mx = np.full(ngroups, -np.inf)
    np.maximum.at(mx, gid, margin)
    e = np.exp(margin - mx[gid])
    s = np.bincount(gid, weights=e, minlength=ngroups)
    return e / s[gid]


def design(P, h, X, weekly, rows_mt, train_mask_for_centre):
    """Rows = (m, t, slot) for the given municipality-years over valid slots.

    Features: slot, region, centred log total, centred log previous-year total,
    15 climate features. Returns (features, base margin, labels, group id).
    """
    lN = np.log1p(P.N)
    w = train_mask_for_centre
    centre = np.where(w, lN, 0).sum(1) / np.maximum(w.sum(1), 1)
    feats, base, lab, gid = [], [], [], []
    for g, (m, t) in enumerate(rows_mt):
        n = P.nweeks[t] if weekly else 12
        Y = (P.Yw if weekly else P.Ym)[m, t, :n]
        prev = lN[m, t - 1] - centre[m] if t > 0 and not P.excluded[m, t - 1] else np.nan
        sl = np.arange(n)
        f = np.column_stack([sl + 1, np.full(n, P.region[m]), np.full(n, lN[m, t] - centre[m]),
                             np.full(n, prev), X[m, t, :n, :]])
        feats.append(f); base.append(h[m, :n]); lab.append(Y / max(Y.sum(), 1)); gid.append(np.full(n, g))
    return (np.vstack(feats), np.concatenate(base), np.concatenate(lab), np.concatenate(gid))


def main(res="weekly", subset=60, rounds=300):
    """Fit M2 in every LOYO fold and add its predictions to the saved run."""
    import xgboost as xgb
    from ..experiments.brazil_loyo import log_runtime
    weekly = res == "weekly"
    tag = f"{res}_{subset}"
    d, clim, muni = BD.load_tables()
    geos = BD.select_subset60(d, muni).geocode.values if subset == 60 else muni.geocode.values
    P = BD.build_panel(d, clim, geos)
    Xc, cy, cw = BD.climate_matrix(clim, P.geocodes)
    f = C.BRAZIL_DIR / f"preds_{tag}.npz"
    z = dict(np.load(f))
    assert np.array_equal(z["geocodes"], P.geocodes)
    M, T = len(P.geocodes), len(P.years)
    S = 53 if weekly else 12
    pred = np.full((M, T, S), np.nan)
    train_ok, eval_ok = row_masks(P)
    for t, y in enumerate(P.years):
        if y not in C.LOYO_YEARS:
            continue
        t0 = time.time()
        tm = train_ok.copy(); tm[:, t] = False
        h, _, _ = FS.offsets(P, tm, weekly)
        X, _ = FC.climate_features(P, Xc, cy, cw, y, weekly)
        tr_rows = list(zip(*np.where(tm)))
        Ftr, btr, ptr, gtr = design(P, h, X, weekly, tr_rows, tm)
        ng = len(tr_rows)

        def obj(predt, dtrain):
            pi = group_softmax(predt, gtr, ng)
            return pi - ptr, np.maximum(pi * (1 - pi), 1e-6)

        dtr = xgb.DMatrix(Ftr, base_margin=btr)
        bst = xgb.train(dict(max_depth=4, eta=0.05, tree_method="hist", seed=C.SEED, nthread=1,
                             base_score=0.0), dtr, num_boost_round=rounds, obj=obj)
        te_rows = [(m, t) for m in np.where(eval_ok[:, t])[0]]
        Fte, bte, _, gte = design(P, h, X, weekly, te_rows, tm)
        mte = bst.predict(xgb.DMatrix(Fte, base_margin=bte), output_margin=True)
        pi = group_softmax(mte, gte, len(te_rows))
        k = 0
        for (m, tt) in te_rows:
            n = P.nweeks[tt] if weekly else 12
            pred[m, tt, :n] = pi[k:k + n]; k += n
        log_runtime(dict(tag=tag, model="M2", fold=int(y), fit_s=round(time.time() - t0, 2), nit=rounds, grad_inf=np.nan))
        print(f"M2 {tag} fold {y} {time.time() - t0:.1f}s", flush=True)
    z["pred__M2"] = pred
    np.savez_compressed(f, **z)
    return score(tag)
