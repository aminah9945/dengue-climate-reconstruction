"""Tables and figures of the timing-informed share model (EXPLORATORY).

Reads the per municipality-year metrics written by :mod:`.run`
(``<work>/timing_design{A,C}_power_{SUBSET}/metrics_detail_*.csv``) and writes small
CSV tables and PNG figures to ``results/extensions/timing/``. Paired contrasts,
bootstraps (municipality clusters and test-year blocks, 2,000 replicates) and
the epidemic/quiet labels reuse :mod:`dengue_recon.reporting.official_report`.

CLI: ``python -m dengue_recon.extensions.timing.report``.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from ... import config as C
from ...reporting import official_report as OR
from .run import work_dir

SUBSET = int(os.environ.get("TIMING_SUBSET", 300))
STATUS = "EXPLORATORY (developed after the pre-registered analysis)"
MAIN = ["S1", "S2", "S3", "S3C", "T1"]
REFS = ["B1", "B4s", "M1-noONI", "M1"]
METRICS = ["jsd", "peak_err", "onset_err", "emd"]
LABEL = {"S1": "S1 state curve", "S2": "S2 search (Trends)", "S3": "S3 state + search",
         "S3C": "S3C state + search + climate", "T1": "T1 timing shift", "B4s": "B4s", "M1": "M1",
         "M1-noONI": "M1-noONI", "B1": "B1 seasonal"}
COL = {"B1": "#52514e", "B4s": "#eb6834", "M1": "#2a78d6", "M1-noONI": "#1baf7a", "S1": "#eda100",
       "S2": "#e87ba4", "S3": "#008300", "S3C": "#6250d6", "T1": "#e34948", "check": "#8a8984"}

#: (label, a, b): difference a - b (negative = a better)
CONTRASTS = (
    [(f"{m} - B4s", m, "B4s") for m in MAIN] + [(f"{m} - M1", m, "M1") for m in MAIN]
    + [(f"{m} - M1-noONI", m, "M1-noONI") for m in MAIN]
    + [("S3 - S1 (search on top of state)", "S3", "S1"), ("S3 - S2 (state on top of search)", "S3", "S2"),
       ("S3C - S3 (climate on top of signals)", "S3C", "S3"), ("S1 - T1", "S1", "T1"),
       ("M1 - B4s", "M1", "B4s"), ("M1-noONI - B4s", "M1-noONI", "B4s"), ("B4s - B1", "B4s", "B1")]
    + [(f"wrong year: {m}-wy - B4s", f"{m}-wy", "B4s") for m in ("S1", "S2", "S3")]
    + [(f"wrong year: {m} - {m}-wy", m, f"{m}-wy") for m in ("S1", "S2", "S3")]
    + [("wrong year at test only: S3-wyTest - S3", "S3-wyTest", "S3"),
       ("wrong year at test only: S3-wyTest - B4s", "S3-wyTest", "B4s")]
    + [("non-study pool: S1-ns - B4s", "S1-ns", "B4s"), ("non-study pool: S1-ns - S1", "S1-ns", "S1")]
    + [(f"mixed resolution: {m}-{p} - B4s", f"{m}-{p}", "B4s") for m in ("S1", "S3", "T1") for p in ("p25", "p50")]
    + [(f"mixed resolution: {m}-{p} - {m}", f"{m}-{p}", m) for m in ("S1", "S3", "T1") for p in ("p25", "p50")]
)


def out_dirs():
    base = Path(os.environ.get("TIMING_RESULTS_DIR", C.RESULTS_DIR / "extensions" / "timing"))
    f = base / "figures"
    f.mkdir(parents=True, exist_ok=True)
    return base, f


def load(design):
    tag = f"timing_design{design}_power_{SUBSET}"
    f = work_dir() / tag / f"metrics_detail_{tag}.csv"
    if not f.exists():
        return None, None
    df = OR.check_no_locked_years(pd.read_csv(f))
    info = json.loads((work_dir() / tag / "run_info.json").read_text())
    return df, info


def model_means(df, design, n_boot=C.N_BOOT):
    rows = []
    for m in df.model.unique():
        g = df[df.model == m]
        r = dict(design=design, model=m, n=len(g), n_municipalities=g.geocode.nunique())
        for k in METRICS:
            mu, b = OR.boot_replicates(g[k].values, g.geocode.values, n_boot, C.BOOT_SEED)
            r[k], r[f"{k}_ci_lo"], r[f"{k}_ci_hi"] = mu, *OR.ci(b)
        r["jsd_median"] = g.jsd.median()
        r["jsd_nweighted"] = float(np.average(g.jsd, weights=g.N))
        for k in ("mae_counts", "w50_err", "cov80", "cov95", "crps", "jsd_floor"):
            r[k] = g[k].mean() if k in g and g[k].notna().any() else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def contrasts(df, design, metrics=METRICS, n_boot=C.N_BOOT):
    have = set(df.model.unique())
    rows = []
    for lab, a, b in CONTRASTS:
        if a in have and b in have:
            for k in metrics:
                r = OR.contrast(df, a, b, k, False, n_boot, C.BOOT_SEED)
                if r:
                    rows.append(dict(design=design, contrast=lab, status=STATUS, **r))
    return pd.DataFrame(rows)


def by_group(df, design, col, groups, models=MAIN, refs=("B4s", "M1"), metrics=("jsd", "peak_err"), n_boot=C.N_BOOT):
    rows = []
    for gval in groups:
        d = df[df[col] == gval]
        for m in models:
            for ref in refs:
                for k in metrics:
                    r = OR.contrast(d, m, ref, k, False, n_boot, C.BOOT_SEED)
                    if r:
                        rows.append(dict(design=design, group_col=col, group=gval, **r))
    return pd.DataFrame(rows)


def per_year(df, design):
    t = df.groupby(["year", "model"])[["jsd", "peak_err", "onset_err"]].mean().reset_index()
    t.insert(0, "design", design)
    return t


def trends_coverage_table(df, info, design, n_boot=C.N_BOOT):
    """S2 - B4s split by whether the state has its own Trends series or uses the national one."""
    ps = pd.read_csv(work_dir() / f"timing_design{design}_power_{SUBSET}" / "pool_sizes.csv")
    d = df.merge(ps[["geocode", "trends_fallback"]], on="geocode")
    d["trends_source"] = np.where(d.trends_fallback, "national fallback", "own state series")
    return by_group(d, design, "trends_source", ["own state series", "national fallback"], models=["S2", "S3"],
                    refs=("B4s",), metrics=("jsd",), n_boot=n_boot)


def shift_summary(design):
    """Distribution of the estimated T1 shifts in the test years (from the fold logs)."""
    fdir = work_dir() / f"timing_design{design}_power_{SUBSET}" / "folds"
    rows = []
    for f in sorted(fdir.glob("*.json")):
        lg = json.loads(f.read_text())
        for name, s in lg.get("shifts", {}).items():
            rows.append(dict(design=design, fold_id=lg["fold_id"], test_year=lg["test_year"], model=name,
                             mean_abs_shift=s["mean_abs_shift_test"], share_nonzero=s["share_nonzero_test"]))
    return pd.DataFrame(rows)


def coefficient_table(design):
    """National coefficients theta of the new feature blocks per fold."""
    fdir = work_dir() / f"timing_design{design}_power_{SUBSET}" / "folds"
    names = {"S1": [f"state lag {L}" for L in (-2, -1, 0, 1, 2)], "S2": [f"search +{L}" for L in range(4)]}
    names["S3"] = names["S1"] + names["S2"]
    names["S3C"] = names["S3"] + OR.feature_names()
    rows = []
    for f in sorted(fdir.glob("*.json")):
        lg = json.loads(f.read_text())
        for m in ("S1", "S2", "S3", "S3C"):
            fi = lg["fits"].get(m)
            if fi:
                for nm, v in zip(names[m], fi["theta"]):
                    rows.append(dict(design=design, fold_id=lg["fold_id"], test_year=lg["test_year"], model=m,
                                     feature=nm, theta=v))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ figures
def fig_model_means(mm, figdir, fname="fig1_jsd_peak_onset_by_model.png"):
    plt = OR._plt()
    order = ["B1", "B4s", "M1-noONI", "M1"] + MAIN
    mm = mm.set_index("model").loc[[m for m in order if m in set(mm.model)]]
    fig, axs = plt.subplots(1, 3, figsize=(18, 6.5), sharey=True)
    for ax, (k, xl) in zip(axs, [("jsd", "mean JSD"), ("peak_err", "mean peak-week error (weeks)"),
                                 ("onset_err", "mean onset-week error (weeks)")]):
        y = np.arange(len(mm))[::-1]
        for yi, (m, r) in zip(y, mm.iterrows()):
            ax.plot([r[f"{k}_ci_lo"], r[f"{k}_ci_hi"]], [yi, yi], color=COL[m], lw=5, solid_capstyle="round")
            ax.plot(r[k], yi, "o", ms=10, color="white", mec=COL[m], mew=2.5)
            ax.annotate(f"{r[k]:.3f}" if k == "jsd" else f"{r[k]:.2f}", (r[f"{k}_ci_hi"], yi), xytext=(6, 0),
                        textcoords="offset points", va="center", fontsize=11)
        ax.axvline(mm.loc["B4s", k], color=COL["B4s"], ls=":", lw=1.5)
        ax.set_yticks(y); ax.set_yticklabels([LABEL.get(m, m) for m in mm.index])
        ax.set_xlabel(xl + " (lower = better)")
    fig.suptitle("Design A (2012-2023, 300 municipalities): accuracy by model, 95 % municipality-bootstrap CI"
                 "  [EXPLORATORY]", x=0.01, ha="left", fontsize=15, fontweight="bold")
    fig.tight_layout(); fig.savefig(figdir / fname); plt.close(fig)


def _items(ct, pairs, metric, scale=100.0, rel=True):
    items = []
    for lab, a, b, col in pairs:
        r = ct[(ct.model_a == a) & (ct.model_b == b) & (ct.metric == metric)]
        if not len(r):
            continue
        r = r.iloc[0]
        if rel:
            items.append((lab, scale * r.rel_change, scale * r.rel_ci_muni_lo, scale * r.rel_ci_muni_hi,
                          scale * r.rel_ci_year_lo, scale * r.rel_ci_year_hi, col))
        else:
            items.append((lab, r.mean_diff, r.ci_muni_lo, r.ci_muni_hi, r.ci_year_lo, r.ci_year_hi, col))
    return items


def fig_gain(ct, figdir, design="A", fname="fig2_gain_vs_B4s_with_CIs.png"):
    plt = OR._plt()
    fig, axs = plt.subplots(2, 2, figsize=(18, 11))
    for row, ref in enumerate(["B4s", "M1"]):
        pairs = [(LABEL[m], m, ref, COL[m]) for m in MAIN]
        _items_j = _items(ct, pairs, "jsd")
        OR._forest(axs[row, 0], _items_j, f"relative JSD change vs {ref}, % (negative = better)")
        axs[row, 0].set_title(f"JSD: model - {ref}")
        OR._forest(axs[row, 1], _items(ct, pairs, "peak_err", rel=False),
                   f"peak-week error difference vs {ref}, weeks")
        axs[row, 1].set_title(f"Peak-week error: model - {ref}")
    fig.text(0.01, 0.005, "Thick bar: 95 % municipality-bootstrap CI.  Thin bar: 95 % year-block (test-year) "
             "bootstrap CI.  Paired by municipality-year.", fontsize=11, color="#52514e")
    fig.suptitle(f"Design {design}: gain of the timing-informed models  [EXPLORATORY]", x=0.01, ha="left",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(figdir / fname); plt.close(fig)


def fig_region(rt, figdir, fname="fig3_gain_by_region.png"):
    plt = OR._plt()
    fig, axs = plt.subplots(1, 2, figsize=(18, 9), sharey=True)
    for ax, k, xl in [(axs[0], "jsd", "JSD difference vs B4s x 1000"), (axs[1], "peak_err", "peak-week error difference vs B4s, weeks")]:
        items = []
        for reg in C.REGIONS:
            for m in ("S1", "S3", "T1"):
                r = rt[(rt.group == reg) & (rt.model_a == m) & (rt.model_b == "B4s") & (rt.metric == k)]
                if len(r):
                    r = r.iloc[0]
                    s = 1000 if k == "jsd" else 1
                    items.append((f"{reg}, {m}", s * r.mean_diff, s * r.ci_muni_lo, s * r.ci_muni_hi,
                                  s * r.ci_year_lo, s * r.ci_year_hi, COL[m]))
        OR._forest(ax, items, xl + " (negative = better)")
    fig.text(0.01, 0.005, "Thick bar: municipality-bootstrap 95 % CI.  Thin bar: year-block 95 % CI.",
             fontsize=11, color="#52514e")
    fig.suptitle("Design A: gain vs B4s by region  [EXPLORATORY]", x=0.01, ha="left", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(figdir / fname); plt.close(fig)


def fig_wrong_year(ct, figdir, fname="fig4_wrong_year_check.png"):
    plt = OR._plt()
    pairs = []
    for m in ("S1", "S2", "S3"):
        pairs += [(f"{m} (right year) - B4s", m, "B4s", COL[m]),
                  (f"{m}-wy (wrong year, refitted) - B4s", f"{m}-wy", "B4s", COL["check"])]
    pairs += [("S3-wyTest (wrong year at test) - B4s", "S3-wyTest", "B4s", COL["check"])]
    fig, ax = plt.subplots(figsize=(13, 7.5))
    OR._forest(ax, _items(ct, pairs, "jsd"), "relative JSD change vs B4s, % (negative = better)")
    ax.set_title("Wrong-year check: signals from another year of the same municipality lose the gain  [EXPLORATORY]",
                 fontsize=13)
    fig.tight_layout(); fig.savefig(figdir / fname); plt.close(fig)


def fig_mixed(ct, figdir, fname="fig5_mixed_resolution.png"):
    plt = OR._plt()
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    xs = [0.25, 0.5, 1.0]
    for ax, k, yl in [(axs[0], "jsd", "relative JSD change vs B4s, %"), (axs[1], "peak_err", "peak-week error change vs B4s, weeks")]:
        for m in ("S1", "S3", "T1"):
            names = [f"{m}-p25", f"{m}-p50", m]
            est, lo, hi = [], [], []
            for nm in names:
                r = ct[(ct.model_a == nm) & (ct.model_b == "B4s") & (ct.metric == k)]
                if not len(r):
                    est.append(np.nan); lo.append(np.nan); hi.append(np.nan); continue
                r = r.iloc[0]
                f = 100 if k == "jsd" else 1
                if k == "jsd":
                    est.append(f * r.rel_change); lo.append(f * r.rel_ci_muni_lo); hi.append(f * r.rel_ci_muni_hi)
                else:
                    est.append(r.mean_diff); lo.append(r.ci_muni_lo); hi.append(r.ci_muni_hi)
            ax.plot(xs, est, marker="o", ms=9, lw=2, color=COL[m], label=LABEL[m])
            ax.fill_between(xs, lo, hi, color=COL[m], alpha=0.15, lw=0)
        ax.axhline(0, color="#0b0b0b", lw=1)
        ax.set_xticks(xs); ax.set_xticklabels(["25 %", "50 %", "100 %"])
        ax.set_xlabel("share of the state's other municipalities that report weekly")
        ax.set_ylabel(yl + " (negative = better)")
        ax.legend(frameon=False)
    fig.suptitle("Mixed-resolution scenario (design A): fewer reporting units, band = municipality-bootstrap 95 % CI"
                 "  [EXPLORATORY]", x=0.01, ha="left", fontsize=14, fontweight="bold")
    fig.tight_layout(); fig.savefig(figdir / fname); plt.close(fig)


def fig_examples(figdir, cases=((C.BELO_HORIZONTE, 2016), (C.BELO_HORIZONTE, 2019)), design="A",
                 models=("B4s", "M1", "S3", "T1"), fname="fig6_examples_belo_horizonte.png"):
    """Truth vs reconstructions (allocated counts) for selected municipality-years."""
    from ...evaluation.allocation import allocate_integers
    plt = OR._plt()
    fdir = work_dir() / f"timing_design{design}_power_{SUBSET}" / "folds"
    mx = C.max_loaded_year()
    d = pd.read_parquet(C.DATA_DIR / "dengue_weekly.parquet", columns=["geocode", "name", "epi_year", "epi_week", "cases"],
                        filters=[("epi_year", "<=", mx), ("geocode", "in", [int(g) for g, _ in cases])])
    assert d.epi_year.max() <= mx
    fig, axs = plt.subplots(1, len(cases), figsize=(9 * len(cases), 6.5), squeeze=False)
    rows_out = []
    for ax, (g, y) in zip(axs[0], cases):
        assert y <= mx
        f = [p for p in fdir.glob(f"{design}_T{y}_k*.npz")]
        if not f:
            continue
        z = np.load(f[0])
        i = int(np.where(z["geocodes"] == g)[0][0])
        tr = d[(d.geocode == g) & (d.epi_year == y)].sort_values("epi_week")
        truth = tr.cases.values.astype(int)
        n, N = len(truth), int(truth.sum())
        wk = np.arange(1, n + 1)
        ax.bar(wk, truth, color="#c3c2b7", width=0.85, label="observed (hidden from the models)")
        for m in models:
            p = z[f"pred__{m}"][i, :n]
            a = allocate_integers(p, N)
            assert a.sum() == N
            ax.plot(wk, a, lw=2.6 if m in ("S3", "T1") else 1.8, color=COL[m], label=LABEL.get(m, m))
            from ...evaluation.metrics import jsd, peak_week_error
            rows_out.append(dict(geocode=g, name=tr.name.iloc[0], year=y, model=m, jsd=jsd(truth / N, p),
                                 peak_err=peak_week_error(truth / N, p)))
        ax.set_title(f"{tr.name.iloc[0]} {y} (annual total {N:,})")
        ax.set_xlabel("epi week"); ax.set_ylabel("weekly cases")
        ax.legend(frameon=False, fontsize=11)
    fig.suptitle("Example reconstructions, design A (counts sum exactly to the annual total)  [EXPLORATORY]",
                 x=0.01, ha="left", fontsize=14, fontweight="bold")
    fig.tight_layout(); fig.savefig(figdir / fname); plt.close(fig)
    return pd.DataFrame(rows_out)


def main(n_boot=C.N_BOOT):
    base, figdir = out_dirs()
    dfA, infoA = load("A")
    dfC, infoC = load("C")
    assert dfA is not None, "design A metrics not found; run extensions.timing.run first"
    mm = [model_means(dfA, "A", n_boot)]
    ct = [contrasts(dfA, "A", n_boot=n_boot)]
    if dfC is not None:
        mm.append(model_means(dfC, "C", n_boot))
        ct.append(contrasts(dfC, "C", metrics=("jsd", "peak_err", "onset_err"), n_boot=n_boot))
    mm, ct = pd.concat(mm, ignore_index=True), pd.concat(ct, ignore_index=True)
    mm.insert(1, "status", STATUS)
    mm.to_csv(base / "model_means.csv", index=False)
    ct.to_csv(base / "paired_contrasts.csv", index=False)
    rt = by_group(dfA, "A", "region", C.REGIONS, n_boot=n_boot)
    rt.to_csv(base / "gain_by_region.csv", index=False)
    folds = infoA["folds"]
    lab = OR.epidemic_labels(dfA, folds, OR.annual_totals())
    dE = dfA.merge(lab[["fold_id", "geocode", "stratum"]], on=["fold_id", "geocode"])
    et = by_group(dE, "A", "stratum", ["epidemic", "quiet"], n_boot=n_boot)
    et.to_csv(base / "gain_epidemic_vs_quiet.csv", index=False)
    py = [per_year(dfA, "A")] + ([per_year(dfC, "C")] if dfC is not None else [])
    pd.concat(py).to_csv(base / "metrics_by_test_year.csv", index=False)
    trends_coverage_table(dfA, infoA, "A", n_boot).to_csv(base / "search_signal_by_trends_source.csv", index=False)
    shift_summary("A").to_csv(base / "t1_shift_summary.csv", index=False)
    coefficient_table("A").to_csv(base / "coefficients_designA.csv", index=False)
    checks = {}
    for dsg in ("A", "C"):
        f = work_dir() / f"timing_design{dsg}_power_{SUBSET}" / f"checks_timing_design{dsg}_power_{SUBSET}.json"
        if f.exists():
            c = json.loads(f.read_text())
            zc = [v for d_ in c["zero_coef_maxdiff"].values() for v in d_.values()]
            checks[dsg] = dict(n_folds=c["n_folds"], official_leakage_audit_all_pass=c["official_leakage_audit_all_pass"],
                               feature_audit_all_pass=c["feature_audit_all_pass"], zero_coef_maxdiff_max=max(zc),
                               all_fits_converged=all(v["success"] for d_ in c["fit_convergence"].values()
                                                      for v in d_.values() if v["success"] is not None),
                               max_grad_inf=max(v["grad_inf"] for d_ in c["fit_convergence"].values()
                                                for v in d_.values() if v["grad_inf"] is not None),
                               exact_sums=c["exact_sums"], run_wall_s=c.get("run_wall_s"),
                               fold_wall_s_total=sum(c["fold_wall_s"].values()), years_scored=c["years_scored"])
    run_info = {d_: dict(pool_sizes=i["pool_sizes"], trends_national_fallback=i["trends_national_fallback"],
                         pool_invariance_check=i["pool_invariance_check"], excluded_state_years=i["excluded_state_years"])
                for d_, i in (("A", infoA), ("C", infoC)) if i}
    (base / "checks_summary.json").write_text(json.dumps(dict(status=STATUS, checks=checks, run_info=run_info),
                                                         indent=1, default=float))
    fig_model_means(mm[mm.design == "A"], figdir)
    fig_gain(ct[ct.design == "A"], figdir)
    fig_region(rt, figdir)
    fig_wrong_year(ct[ct.design == "A"], figdir)
    fig_mixed(ct[ct.design == "A"], figdir)
    fig_examples(figdir).to_csv(base / "examples_belo_horizonte.csv", index=False)
    if dfC is not None:
        fig_gain(ct[ct.design == "C"], figdir, "C", "fig7_designC_gain_vs_B4s_with_CIs.png")
    return mm, ct


if __name__ == "__main__":
    main()
