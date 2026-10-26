"""Sanity-check table and the pipeline-level checks (Step 7).

Checks (all must pass before results are reported):

1. exact sums - asserted during scoring for every allocation and draw
2. uniform (B0) is worst
3. zero-climate equivalence (``checks_{tag}.json``)
4. train-and-test climate shuffle removes the gain
5. test-only climate shuffle makes M1 worse
6. leakage audit passes in every fold
7. pipeline consistency with the data-foundation smoke test (B1p, <= 2023)

plus the final-test (2024) path unit test on a dummy year and the 2024 lock.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .. import config as C
from ..evaluation.bootstrap import paired


def pipeline_consistency():
    """Check 7: recompute B1p (past-only) for the smoke-test rows with epi_year <= 2023."""
    from ..data import panel as BD
    from ..evaluation.metrics import jsd
    from ..features.seasonal import adapt_length, seasonal_profile
    det = pd.read_csv(C.REPORTS_DIR / "baseline_smoke_test_detail.csv")
    det = det[(det.model == "seasonal_past_only") & (det.epi_year <= 2023)]
    d, _, _ = BD.load_tables()
    d = d[(d.year_total > 0) & (~d.exclude_suggested)]
    shares = {}
    for (g, y), s in d.groupby(["geocode", "epi_year"]):
        shares.setdefault(g, {})[y] = s.sort_values("epi_week")["share"].values if "share" in s else \
            (s.sort_values("epi_week")["cases"].values / s.year_total.iloc[0])
    diffs = []
    for r in det.itertuples():
        truth = shares[r.geocode][r.epi_year]
        p = adapt_length(seasonal_profile(shares[r.geocode], only_before=r.epi_year), len(truth))
        diffs.append(abs(jsd(truth, p) - r.jsd))
    return len(diffs), float(np.max(diffs))


def final_path_unit_test():
    """Final-test (2024) code path on a DUMMY year: synthetic data, train on years < test."""
    from ..evaluation.cv import final_test_split
    from ..models.dirichlet_multinomial import DMModel
    rng = np.random.default_rng(0)
    years = np.arange(2000, 2006)
    tr, te = final_test_split(years, 2005)
    M, T, S = 6, len(years), 12
    base = np.exp(np.sin(2 * np.pi * np.arange(S) / 12)); base /= base.sum()
    N = rng.integers(200, 2000, (M, T)).astype(float)
    Y = np.stack([[rng.multinomial(int(N[m, t]), base) for t in range(T)] for m in range(M)]).astype(float)
    region = np.arange(M) % 5
    mod = DMModel(M, T, S, 12, 15, region, np.ones((T, S), bool), weekly=False)
    h = np.log(Y[:, tr].sum(1) / Y[:, tr].sum((1, 2))[:, None] + 1e-6)
    w = np.broadcast_to(tr, (M, T)).astype(float)
    flat, info = mod.fit(h, np.zeros((M, T, S, 12)), np.zeros((M, T, S, 15)), Y, N, w)
    pi = mod.predict(flat, h, np.zeros((M, T, S, 12)), np.zeros((M, T, S, 15)))[:, te][:, 0]
    ok = bool(tr.sum() == 5 and te.sum() == 1 and np.allclose(pi.sum(1), 1) and np.abs(pi - base).max() < 0.05)
    return ok, float(np.abs(pi - base).max())


def sanity_table(dets):
    """Table of all sanity checks for the given runs ({tag: metrics DataFrame})."""
    rows = []
    rows.append(dict(check="1 exact sums (allocate_integers and every draw)", result="pass",
                     statistic="asserted in evaluation.scoring.score / draw_counts for every model and row"))
    for tag, df in dets.items():
        mj = df.groupby("model").jsd.mean()
        others = mj.drop([m for m in mj.index if m == "B0"])
        rows.append(dict(check=f"2 uniform worst [{tag}]", result="pass" if (mj["B0"] > others).all() else "FAIL",
                         statistic=f"B0 {mj['B0']:.4f}; next worst {others.max():.4f} ({others.idxmax()})"))
        ch = json.loads((C.BRAZIL_DIR / f"checks_{tag}.json").read_text())
        z = max(ch["zero_climate_maxdiff"])
        rows.append(dict(check=f"3 zero-climate equivalence [{tag}]", result="pass" if z < 1e-10 else "FAIL",
                         statistic=f"max abs diff {z:.2e}"))
        if "M1-shufTT" in mj.index:
            r = paired(df, "M1-shufTT", "B4s")
            ok = (r["ci_muni_lo"] <= 0 <= r["ci_muni_hi"]) or r["mean_diff"] >= 0
            rows.append(dict(check=f"4 climate shuffle train+test (3 seeds) [{tag}]", result="pass" if ok else "FAIL",
                             statistic=f"shufTT-B4s {r['mean_diff']:+.5f} [{r['ci_muni_lo']:+.5f}, {r['ci_muni_hi']:+.5f}]"))
            for s in range(3):
                rs = paired(df, f"M1-shufTT-s{s}", "B4s")
                rows.append(dict(check=f"4 (seed {s}) [{tag}]", result="info",
                                 statistic=f"{rs['mean_diff']:+.5f} [{rs['ci_muni_lo']:+.5f}, {rs['ci_muni_hi']:+.5f}]"))
        if "M1-shufTest" in mj.index:
            rows.append(dict(check=f"5 climate shuffle test-only [{tag}]",
                             result="pass" if mj["M1-shufTest"] > mj["M1"] else "FAIL",
                             statistic=f"JSD shufTest {mj['M1-shufTest']:.5f} vs M1 {mj['M1']:.5f}"))
        la = pd.read_csv(C.BRAZIL_DIR / f"leakage_audit_{tag}.csv")
        okl = bool(la.drop(columns="year").all().all())
        rows.append(dict(check=f"6 leakage audit (h_m, climatology/SD, centring, train arrays) [{tag}]",
                         result="pass" if okl else "FAIL", statistic=f"{len(la)} folds audited"))
        rows.append(dict(check=f"climate NaN anomalies set to 0 [{tag}]", result="info",
                         statistic=f"{sum(ch['clim_nan'])} NaN anomaly values over all folds"))
    try:
        n, mx = pipeline_consistency()
        rows.append(dict(check="7 pipeline consistency with smoke test (B1p, <=2023)",
                         result="pass" if mx < 1e-9 else "FAIL", statistic=f"{n} rows, max |dJSD| {mx:.2e}"))
    except FileNotFoundError as exc:   # processed data or smoke-test file not available
        rows.append(dict(check="7 pipeline consistency with smoke test (B1p, <=2023)",
                         result="not run", statistic=f"input missing: {exc.filename}"))
    ok, dev = final_path_unit_test()
    rows.append(dict(check="final-test path unit test on dummy year (RUN_FINAL_2024=False)",
                     result="pass" if ok else "FAIL", statistic=f"max |pi - true| {dev:.4f}"))
    rows.append(dict(check="2024 untouched", result="pass" if not C.RUN_FINAL_2024 else "FAIL",
                     statistic="RUN_FINAL_2024=False; data.panel.load_tables drops epi_year >= 2024"))
    return pd.DataFrame(rows)
