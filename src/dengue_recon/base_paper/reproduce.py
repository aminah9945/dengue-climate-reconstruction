"""Step 2 driver: reduced reproduction of Lim et al. (2026) DM (interpolation mask, rep01, 3 folds).

* ``run("MAP", fold)``  - MAP fit of one fold (seconds)
* ``run("NUTS", fold)`` - NUTS fit of one fold (about 20 min on 4 CPUs; run folds in parallel)
* ``table()``           - assemble ``table_repro.csv``, ``cancellation_check.csv`` and
  ``fit_diagnostics.csv`` from the cached per-fold JSON files

Per-fold outputs are cached as JSON in ``results/repro_lim/folds/`` (resumable).
CLI: ``python scripts/run_pipeline.py --step reproduce [--nuts]``.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config as C
from . import lim_dm as L


def fold_dir(results_dir: Path | None = None) -> Path:
    d = Path(results_dir or C.REPRO_DIR) / "folds"
    d.mkdir(parents=True, exist_ok=True)
    return d


def run(method, fold, chains=2, warmup=500, draws=500, force=False):
    """Fit one fold with ``method`` in {"MAP", "NUTS"} and cache the result as JSON."""
    out = fold_dir() / f"{method}_fold{fold}.json"
    if out.exists() and not force:
        print("cached", out)
        return json.loads(out.read_text())
    df = L.read_mask(C.LIM_MASK_PATH)
    t0 = time.time()
    rows, canc, diag = L.run_fold(df, fold, method, chains=chains, warmup=warmup, draws=draws)
    diag.update(chains=chains, warmup=warmup, draws=draws, wall_s=round(time.time() - t0, 1))
    res = dict(rows=rows, canc=canc, diag=diag)
    out.write_text(json.dumps(res, default=float, indent=1))
    print(method, fold, json.dumps(rows[0], default=float), flush=True)
    return json.loads(out.read_text())


def wmean(x, w):
    """Row-weighted mean ignoring non-finite values."""
    x, w = np.asarray(x, float), np.asarray(w, float)
    ok = np.isfinite(x)
    return float(np.sum(x[ok] * w[ok]) / np.sum(w[ok])) if ok.any() else np.nan


def table():
    """Pool the cached folds (weighted by test rows) and add the published DM row."""
    rows, canc, diags = [], [], []
    for f in sorted(fold_dir().glob("*.json")):
        d = json.loads(f.read_text())
        rows += d["rows"]; canc += d["canc"]; diags.append(d["diag"])
    df = pd.DataFrame(rows)
    # reference rows appear once per method file: keep one copy per fold
    df = df.drop_duplicates(subset=["model", "fit", "fold"])
    agg = []
    for (m, fit), g in df.groupby(["model", "fit"], sort=False):
        r = dict(model=m, fit=fit, fold="all", n_rows=int(g.n_rows.sum()))
        for k in ["MAE", "RMSE", "MAE_median", "RMSE_median", "COV80", "CRPS", "JSD_mean", "peak_month_err"]:
            r[k] = wmean(g[k], g.n_rows)
        r["runtime_s"] = float(g.runtime_s.sum())
        agg.append(r)
    tab = pd.concat([df, pd.DataFrame(agg)], ignore_index=True)
    cols = ["model", "fit", "fold", "n_rows", "MAE", "RMSE", "COV80", "CRPS", "JSD_mean",
            "peak_month_err", "runtime_s", "MAE_median", "RMSE_median"]
    tab = tab[cols]
    if C.LIM_TARGET_TABLE.exists():
        pub = pd.read_csv(C.LIM_TARGET_TABLE)
        pub = pub[(pub.Task == "Interpolation") & (pub.Model == "DM")].iloc[0]
        tab = pd.concat([tab, pd.DataFrame([dict(model="DM (published, 3 reps, 4x1000 draws)", fit="NUTS",
                                                 fold="all", n_rows=int(pub.N), MAE=np.nan, RMSE=np.nan,
                                                 MAE_median=pub.MAE, RMSE_median=pub.RMSE,
                                                 COV80=pub["80% coverage"], CRPS=pub.CRPS)])], ignore_index=True)
    tab.to_csv(C.REPRO_DIR / "table_repro.csv", index=False)
    pd.DataFrame(canc).to_csv(C.REPRO_DIR / "cancellation_check.csv", index=False)
    pd.DataFrame(diags).to_csv(C.REPRO_DIR / "fit_diagnostics.csv", index=False)
    print(tab.round(3).to_string())
    return tab


def main(nuts=False, folds=(1, 2, 3), chains=2, warmup=500, draws=500, force=False):
    """Run MAP (and optionally NUTS) for the given folds, then build the table."""
    C.ensure_dirs()
    for f in folds:
        run("MAP", f, force=force)
    if nuts:
        for f in folds:
            run("NUTS", f, chains, warmup, draws, force=force)
    return table()
