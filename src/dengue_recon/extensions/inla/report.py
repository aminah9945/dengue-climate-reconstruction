"""Report of the EXPLORATORY INLA extension (not pre-registered).

Pairs the INLA rows (``<INLA_DIR>/<tag>/metrics_detail_inla_<tag>.csv``) with the
official design-A rows of the Dirichlet-multinomial models
(``<OFFICIAL_DIR>/designA_<climate>_300/metrics_detail_designA_<climate>_300.csv``)
on (fold_id, geocode) and reuses the official statistics
(:func:`dengue_recon.reporting.official_report.contrast`: municipality and
year-block bootstraps, 2,000 replicates). For a 60-municipality INLA run the
official 300-municipality rows are restricted to the same 60 municipalities
(the DM models were still fitted on 300).

Writes small CSVs and two figures to ``<INLA_DIR>/tables`` and ``<INLA_DIR>/figures``
with the suffix ``_<subset>``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ... import config as C
from ...reporting import official_report as R
from .driver import INLA_DIR, run_tag

OFFICIAL_MODELS = ["B0", "B1", "B4", "B4s", "M1-noONI", "M1"]
INLA_MODELS = ["INLA-unscaled", "INLA-scaled", "INLA-scaled-clim20", "INLA-scaled-clim15", "INLA-seasonal-only"]
CONTRASTS = [
    ("INLA-scaled - B4s", "INLA-scaled", "B4s"),
    ("INLA-scaled - M1", "INLA-scaled", "M1"),
    ("INLA-scaled - M1-noONI", "INLA-scaled", "M1-noONI"),
    ("INLA-scaled - B4", "INLA-scaled", "B4"),
    ("INLA-scaled - B1", "INLA-scaled", "B1"),
    ("INLA climate (20): INLA-scaled-clim20 - INLA-scaled", "INLA-scaled-clim20", "INLA-scaled"),
    ("INLA climate (15 local): INLA-scaled-clim15 - INLA-scaled", "INLA-scaled-clim15", "INLA-scaled"),
    ("INLA-scaled-clim20 - M1", "INLA-scaled-clim20", "M1"),
    ("INLA-scaled-clim15 - M1-noONI", "INLA-scaled-clim15", "M1-noONI"),
    ("INLA-scaled-clim20 - B4s", "INLA-scaled-clim20", "B4s"),
    ("AR1 carry-over: INLA-scaled - INLA-seasonal-only", "INLA-scaled", "INLA-seasonal-only"),
]
FIG_COL = {"B4s": "#eb6834", "M1": "#2a78d6", "INLA-scaled": "#008300", "INLA-scaled-clim20": "#6250d6"}
LABEL = "EXPLORATORY (not pre-registered)"


def combined(subset=300, climate="power", inla_dir=None, official_dir=None):
    """INLA rows + official DM rows on the same (fold_id, geocode) pairs."""
    tag = run_tag(climate, subset)
    inla = pd.read_csv(Path(inla_dir or INLA_DIR) / tag / f"metrics_detail_inla_{tag}.csv")
    otag = f"designA_{climate}_300"
    off = R.load_detail(otag, Path(official_dir) if official_dir else None)
    assert off is not None, f"official metrics_detail of {otag} not found"
    off = off[off.model.isin(OFFICIAL_MODELS)]
    keys = inla[["fold_id", "geocode"]].drop_duplicates()
    off = off.merge(keys, on=["fold_id", "geocode"])
    # same municipality-years and totals
    chk = inla[inla.model == "INLA-scaled"][["fold_id", "geocode", "N", "jsd_floor"]].merge(
        off[off.model == "B4s"][["fold_id", "geocode", "N", "jsd_floor"]], on=["fold_id", "geocode"])
    assert len(chk) == len(keys), "INLA and official rows do not match"
    assert (chk.N_x == chk.N_y).all()
    if subset == 300:       # same rows in the same order -> the noise-floor draws match too
        assert np.allclose(chk.jsd_floor_x, chk.jsd_floor_y)
    df = pd.concat([off, inla[off.columns.intersection(inla.columns).tolist() +
                              [c for c in inla.columns if c not in off.columns]]], ignore_index=True)
    return R.check_no_locked_years(df)


def model_table(df, run):
    rows = []
    for m in OFFICIAL_MODELS + INLA_MODELS:
        g = df[df.model == m]
        if not len(g):
            continue
        mu, b = R.boot_replicates(g.jsd.values, g.geocode.values)
        rows.append(dict(run=run, model=m, n=len(g), jsd=mu, jsd_ci_lo=R.ci(b)[0], jsd_ci_hi=R.ci(b)[1],
                         jsd_nweighted=float(np.average(g.jsd, weights=g.N)), peak_err=g.peak_err.mean(),
                         emd=g.emd.mean(), mae_share=g.mae_share.mean(), mae_counts=g.mae_counts.mean(),
                         cov80=g.cov80.mean() if "cov80" in g and g.cov80.notna().any() else np.nan,
                         cov95=g.cov95.mean() if "cov95" in g and g.cov95.notna().any() else np.nan,
                         crps=g.crps.mean() if "crps" in g and g.crps.notna().any() else np.nan,
                         jsd_floor=g.jsd_floor.mean()))
    return pd.DataFrame(rows)


def shape_table(shape):
    """Does INLA's predicted shape vary with the held-out year? Per test year: mean JSD
    between the full prediction and the seasonal-only shape of the same fit (what the
    AR1 carry-over adds), its peak shift, the within-year SD of the AR1 term, and the
    share of municipality-years where the carry-over moves the shape towards the truth."""
    s = shape.copy()
    s["ar1_helps"] = s["base_jsd_truth"] < s["seasonal_jsd_truth"]
    agg = dict(n=("geocode", "size"), jsd_full_vs_seasonal=("base_jsd_vs_seasonal", "mean"),
               peak_shift_weeks=("base_peak_shift", "mean"), share_peak_moved=("base_peak_shift", lambda x: (x > 0).mean()),
               ar1_within_year_sd=("base_ar1_sd", "mean"), ar1_first_minus_last=("base_ar1_first_minus_last", "mean"),
               share_carryover_helps=("ar1_helps", "mean"))
    if "clim20_clim_sd" in s:
        agg["clim20_term_sd"] = ("clim20_clim_sd", "mean")
        agg["clim20_jsd_vs_seasonal"] = ("clim20_jsd_vs_seasonal", "mean")
    by = s.groupby("year").agg(**agg).reset_index()
    all_row = {"year": "all"}
    for k, (col, f) in agg.items():
        all_row[k] = len(s) if f == "size" else (s[col].mean() if f == "mean" else f(s[col]))
    return pd.concat([by, pd.DataFrame([all_row])], ignore_index=True)


def year_to_year(inla_npz_dir, official_npz_dir, models=("B4", "B4s", "M1")):
    """Mean JSD between a municipality's predicted shares in consecutive test years
    (52-slot common support), per model, plus the same for the observed shares."""
    from ...evaluation.metrics import jsd, normalise
    inla_npz_dir, official_npz_dir = Path(inla_npz_dir), Path(official_npz_dir)
    files = sorted(inla_npz_dir.glob("A_T*_k*.npz"))
    pred = {}
    for f in files:
        z = np.load(f)
        o = np.load(official_npz_dir / f.name)
        T = int(z["test_year"])
        og = {int(g): i for i, g in enumerate(o["geocodes"])}
        for i, g in enumerate(z["geocodes"]):
            g = int(g)
            pred[("INLA-scaled", g, T)] = z["pred__INLA-scaled"][i, :52]
            pred[("INLA-seasonal-only", g, T)] = z["pred__INLA-seasonal-only"][i, :52]
            if "pred__INLA-scaled-clim20" in z.files:
                pred[("INLA-scaled-clim20", g, T)] = z["pred__INLA-scaled-clim20"][i, :52]
            if g in og:
                for m in models:
                    pred[(m, g, T)] = o[f"pred__{m}"][og[g], :52]
    rows = []
    for (m, g, T), p in pred.items():
        q = pred.get((m, g, T + 1))
        if q is not None:
            rows.append(dict(model=m, geocode=g, year=T, jsd_next_year=jsd(normalise(p), normalise(q))))
    return pd.DataFrame(rows)


def fig_jsd_by_year(df, figdir, title, fname):
    plt = R._plt()
    fig, ax = plt.subplots(figsize=(12, 6.5))
    t = df[df.model.isin(FIG_COL)].groupby(["model", "year"]).jsd.mean().unstack(0)
    for m, col in FIG_COL.items():
        if m not in t:
            continue
        ls = "--" if m.startswith("INLA") else "-"
        ax.plot(t.index, t[m], marker="o", ms=7, lw=2.2, ls=ls, color=col, label=m)
    fl = df[df.model == "B4s"].groupby("year").jsd_floor.mean()
    ax.fill_between(fl.index, 0, fl.values, color="#c3c2b7", alpha=0.35, label="noise floor")
    ks = df.groupby("year").k.first()
    ax.set_xticks(list(ks.index))
    ax.set_xticklabels([f"{y}\nk={k}" for y, k in ks.items()], fontsize=10)
    ax.set_ylim(0, None)
    ax.set_ylabel("mean JSD per municipality-year (lower = better)")
    ax.set_xlabel("test year (k = training years)")
    ax.set_title(title)
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.16), frameon=False)
    fig.tight_layout(); fig.savefig(figdir / fname); plt.close(fig)


def fig_shape(npz_dir, shape, figdir, title, fname):
    """Left: mean AR1(time) contribution (log scale, posterior mean) by week of the held-out
    year, per test year - the carry-over from late in the previous year decays within the
    year. Right: JSD between full and seasonal-only shape, per municipality-year."""
    plt = R._plt()
    fig, axs = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw=dict(width_ratios=[1.4, 1]))
    files = sorted(Path(npz_dir).glob("A_T*_k*.npz"))
    for f in files:
        z = np.load(f)
        a = np.abs(z["ar1_time__base"][:, :52])
        axs[0].plot(np.arange(1, 53), np.nanmean(a, 0), lw=1.4, color="#2a78d6",
                    alpha=0.35 + 0.05 * (int(z["test_year"]) - 2012))
    axs[0].set_xlabel("epi week of the held-out year")
    axs[0].set_ylabel("mean |AR1(time) term| (log scale)")
    axs[0].set_title("AR1 carry-over decays within the held-out year (one line per test year)")
    v = shape.groupby("year").base_jsd_vs_seasonal
    yrs = sorted(shape.year.unique())
    axs[1].boxplot([v.get_group(y).values for y in yrs], positions=yrs, widths=0.6, showfliers=False,
                   medianprops=dict(color="#008300", lw=2))
    axs[1].set_xticks(yrs); axs[1].set_xticklabels(yrs, rotation=45, fontsize=10)
    axs[1].set_ylabel("JSD(full prediction, seasonal-only shape)")
    axs[1].set_title("Year-specific shape change from the AR1 term")
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(); fig.savefig(figdir / fname); plt.close(fig)


def main(subset=300, climate="power", out_dir=None, official_dir=None):
    base = Path(out_dir or INLA_DIR)
    tag = run_tag(climate, subset)
    tdir, fdir = base / "tables", base / "figures"
    tdir.mkdir(parents=True, exist_ok=True); fdir.mkdir(parents=True, exist_ok=True)
    odir = Path(official_dir) if official_dir else C.OFFICIAL_DIR
    df = combined(subset, climate, base, odir)
    run = f"INLA {tag} ({LABEL})"
    mt = model_table(df, run)
    mt.to_csv(tdir / f"model_means_{subset}.csv", index=False)
    ct = R.contrast_table(df, run, CONTRASTS)
    ct.to_csv(tdir / f"paired_contrasts_{subset}.csv", index=False)
    by = df.groupby(["year", "model"]).jsd.mean().unstack(1)
    by = by[[m for m in OFFICIAL_MODELS + INLA_MODELS if m in by]]
    by.round(5).to_csv(tdir / f"jsd_by_test_year_{subset}.csv")
    shape = pd.read_csv(base / tag / f"shape_diag_inla_{tag}.csv")
    st = shape_table(shape)
    st.to_csv(tdir / f"shape_variation_{subset}.csv", index=False)
    yy = year_to_year(base / tag / "folds", odir / f"designA_{climate}_300" / "folds")
    yt = yy.groupby("model").jsd_next_year.agg(["mean", "median", "size"]).reset_index()
    obs = observed_year_to_year(df, subset, climate)
    if obs is not None:
        yt = pd.concat([yt, obs], ignore_index=True)
    yt.to_csv(tdir / f"shape_year_to_year_{subset}.csv", index=False)
    hy = pd.read_csv(base / tag / f"fold_log_inla_{tag}.csv")
    hy.to_csv(tdir / f"fold_log_{subset}.csv", index=False)
    fig_jsd_by_year(df, fdir, f"Design A, {subset} municipalities: INLA vs DM (exploratory, not pre-registered)", f"fig_inla_jsd_by_year_{subset}.png")
    fig_shape(base / tag / "folds", shape, fdir, f"Does the INLA shape vary by held-out year? (exploratory, not pre-registered)",
              f"fig_inla_shape_variation_{subset}.png")
    summary = dict(label=LABEL, run=tag, n_rows=int((df.model == "INLA-scaled").sum()),
                   models=mt.set_index("model")[["jsd", "peak_err", "emd", "cov80", "crps"]].round(5).to_dict("index"))
    (tdir / f"summary_{subset}.json").write_text(json.dumps(summary, indent=1, default=float))
    print(mt.round(5).to_string(), flush=True)
    print(ct[ct.metric == "jsd"][["contrast", "mean_diff", "rel_change", "ci_muni_lo", "ci_muni_hi",
                                  "ci_year_lo", "ci_year_hi", "n_folds_a_better"]].round(5).to_string(), flush=True)
    print(st.round(4).to_string(), flush=True)
    print(yt.round(4).to_string(), flush=True)
    return mt, ct, st


def observed_year_to_year(df, subset, climate):
    """Observed shares: mean JSD between a municipality's observed curves in consecutive
    scored test years (52 common weeks), for reference."""
    from ...data import panel as BD
    from ...evaluation.metrics import jsd, normalise
    try:
        d, clim, muni = BD.load_tables(climate)
    except Exception:          # data not available (e.g. tests)
        return None
    keys = df[df.model == "INLA-scaled"][["geocode", "year"]]
    d = d[d.geocode.isin(keys.geocode.unique())]
    shares = {(g, y): normalise(grp.sort_values("epi_week").cases.values[:52].astype(float))
              for (g, y), grp in d.groupby(["geocode", "epi_year"]) if grp.cases.sum() > 0}
    kset = set(map(tuple, keys.values))
    vals = [jsd(shares[(g, y)], shares[(g, y + 1)]) for g, y in kset
            if (g, y + 1) in kset and (g, y) in shares and (g, y + 1) in shares]
    return pd.DataFrame([dict(model="observed", mean=float(np.mean(vals)), median=float(np.median(vals)),
                              size=len(vals))])
