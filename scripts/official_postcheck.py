"""Post-run checks and compact summary tables for one official run (step 4).

Usage: python scripts/official_postcheck.py results/official/<tag>

Re-checks a finished run from its saved files: all folds present, leakage audit,
zero-climate max difference, exact sums (recomputed from the saved predictions with
the largest-remainder rule), no 2024 anywhere, convergence. Writes
``postcheck_<tag>.json``, ``fold_summary_<tag>.csv`` (metric means per fold and
model) and ``model_summary_<tag>.csv`` (means per model over all scored rows).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from dengue_recon.evaluation.allocation import allocate_integers  # noqa: E402

out = Path(sys.argv[1])
tag = out.name
info = json.loads((out / "run_info.json").read_text())
folds = [f["fold_id"] for f in info["folds"]]
fdir = out / "folds"
res = {"tag": tag, "n_folds_planned": len(folds)}
present = [f for f in folds if (fdir / f"{f}.json").exists() and (fdir / f"{f}.npz").exists()
           and (fdir / f"{f}_metrics.csv").exists()]
res["n_folds_present"] = len(present)
res["missing_folds"] = sorted(set(folds) - set(present))

checks = json.loads((out / f"checks_{tag}.json").read_text())
aud = pd.read_csv(out / f"leakage_audit_{tag}.csv")
res["leakage_all_pass"] = bool(aud.drop(columns=["fold_id", "test_year", "k"]).all().all())
res["leakage_folds_pass"] = int(aud.drop(columns=["fold_id", "test_year", "k"]).all(axis=1).sum())
zc = [v for d in checks["zero_climate_maxdiff"].values() for v in d.values()]
res["zero_climate_maxdiff_max"] = float(max(zc)) if zc else None
res["clim_nan"] = checks["clim_nan"]

det = pd.read_csv(out / f"metrics_detail_{tag}.csv")
res["n_rows"] = len(det)
# exact-sum recomputation from the saved predictions; 2024 check in predictions
Ncheck = det.drop_duplicates(["geocode", "year"]).set_index(["geocode", "year"])["N"]
sum_fail, n_alloc, years_npz = 0, 0, set()
for f in present:
    z = np.load(fdir / f"{f}.npz")
    ty = int(z["test_year"]); years_npz.add(ty)
    geos = z["geocodes"]
    for k in z.files:
        if not k.startswith("pred__"):
            continue
        P = z[k]
        for g, p in zip(geos, P):
            p = p[np.isfinite(p)]
            if p.size == 0:
                continue
            tot = int(Ncheck.loc[(int(g), ty)])
            a = allocate_integers(p / p.sum(), tot)
            n_alloc += 1
            sum_fail += int(a.sum() != tot)
res["exact_sum_allocations"] = n_alloc
res["exact_sum_failures"] = sum_fail
res["years_scored"] = sorted(int(y) for y in det.year.unique())
res["no_2024"] = bool(det.year.max() < 2024 and max(years_npz) < 2024
                      and all(max(lg_y) < 2024 for lg_y in
                              (json.loads((fdir / f"{f}.json").read_text())["panel_years"] for f in present)))

fl = pd.read_csv(out / f"fold_log_{tag}.csv")
dm = fl[fl.model != "M2"]
res["dm_fits"] = int(len(dm))
res["dm_fits_success"] = int((dm.success == True).sum())  # noqa: E712
res["dm_max_nit"] = int(dm.nit.max()) if len(dm) else None
res["dm_max_grad_inf"] = float(dm.grad_inf.max()) if len(dm) else None
res["dm_not_converged"] = dm[dm.success != True][["fold_id", "model", "nit", "grad_inf"]].to_dict("records")  # noqa: E712
m2 = fl[fl.model == "M2"]
res["m2_fits"] = int(m2.m2_rounds.notna().sum())
res["m2_rounds_range"] = [int(m2.m2_rounds.min()), int(m2.m2_rounds.max())] if m2.m2_rounds.notna().any() else None
res["run_wall_s"] = checks.get("run_wall_s")
res["fold_wall_s_sum"] = round(sum(checks["fold_wall_s"].values()), 1)

# compact tables
mcols = [c for c in ["jsd", "peak_err", "mae_share", "emd", "w50_err", "mae_counts", "jsd_floor",
                     "cov80", "cov95", "crps"] if c in det.columns]
g = det.groupby(["design", "fold_id", "year", "k", "climate", "model"], sort=False)
fs = g[mcols].mean()
fs.insert(0, "n_rows", g.size())
fs["jsd_nweighted"] = g.apply(lambda d: np.average(d.jsd, weights=d.N))
fs = fs.reset_index().rename(columns={"year": "test_year"})
fs.round(6).to_csv(out / f"fold_summary_{tag}.csv", index=False)
go = det.groupby("model", sort=False)
ms = go[mcols].mean()
ms.insert(0, "n_folds", go.fold_id.nunique())
ms.insert(0, "n_rows", go.size())
ms["jsd_nweighted"] = go.apply(lambda d: np.average(d.jsd, weights=d.N))
ms = ms.reset_index()
ms.round(6).to_csv(out / f"model_summary_{tag}.csv", index=False)
(out / f"postcheck_{tag}.json").write_text(json.dumps(res, indent=1, default=str))
print(json.dumps(res, indent=1, default=str))
print(ms.round(5).to_string(index=False))
