"""Step 9 (extension): tables and figures for the rain / humidity / temperature lag extension.

Reads ``results/rain_extension/{metrics_detail,preds,fitted_params,leakage_audit}_*``
and the saved reference results in ``results/brazil/`` (B4s, M1, B1 from the pilot
runs). Writes CSV tables and PNG figures into ``results/rain_extension/``.

CLI: ``python scripts/run_pipeline.py --step report`` (also runs this when the
rain-extension metrics exist).
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .. import config as C  # noqa: E402
from ..evaluation.bootstrap import paired  # noqa: E402
from ..extensions.rain_lags import LAG_BLOCKS_EXT, RAIN_DIR, block_label  # noqa: E402

VARIANTS = ["M1-R", "M1-RTH"]
REF_MODELS = ["B1", "B4s", "M1"]
METRICS = ["jsd", "peak_err", "mae_share", "emd", "w50_err", "cov80", "cov95", "crps"]
PAIR_METRICS = ["jsd", "peak_err", "mae_share", "emd"]
COL = {"M1": "#31a354", "M1-R": "#2a78d6", "M1-RTH": "#eb6834", "B4s": "#e6550d", "B1": "#6baed6"}
LABEL = {60: "60 municipalities (pilot subset)", 300: "300 municipalities"}


def note(subset):
    return f"rain extension, weekly, {LABEL[subset]}, LOYO 2010-2023 (2024 never loaded)"


def load(subset):
    """Combined per-row metrics: reference models from the saved pilot run plus the
    new variants (and the shuffle check, if present)."""
    ref = pd.read_csv(C.BRAZIL_DIR / f"metrics_detail_weekly_{subset}.csv")
    ref = ref[ref.model.isin(REF_MODELS)]
    parts = [ref]
    for tag in (f"weekly_{subset}", f"weekly_{subset}_shuf"):
        f = RAIN_DIR / f"metrics_detail_{tag}.csv"
        if f.exists():
            d = pd.read_csv(f)
            d = d[d.model != "M1"]          # the M1 refit is used only for the reproduction check
            parts.append(d)
    return pd.concat(parts, ignore_index=True)


def reproduction_check():
    """Max |share difference| between the M1 refit (rain_ext, same code path) and the
    saved pilot M1 predictions, over evaluation rows."""
    f = RAIN_DIR / "preds_weekly_60.npz"
    if not f.exists() or not (C.BRAZIL_DIR / "preds_weekly_60.npz").exists():
        return None
    a = np.load(f); b = np.load(C.BRAZIL_DIR / "preds_weekly_60.npz")
    if "pred__M1" not in a.files:
        return None
    assert np.array_equal(a["geocodes"], b["geocodes"]) and np.array_equal(a["years"], b["years"])
    pa, pb = a["pred__M1"], b["pred__M1"]
    ok = np.isfinite(pb)
    return float(np.max(np.abs(pa[ok] - pb[ok])))


def main_table(df, subset):
    rows = []
    for sub, d in [("all", df), ("without_2020", df[df.year != 2020])]:
        for m, g in d.groupby("model"):
            r = dict(note=note(subset), subset=sub, model=m, n=len(g))
            for k in METRICS:
                if k in g and g[k].notna().any():
                    r[f"{k}_mean"] = g[k].mean()
            r["jsd_Nweighted"] = np.average(g.jsd, weights=g.N)
            rows.append(r)
    return pd.DataFrame(rows)


def pairs_for(df):
    models = set(df.model)
    pairs = [(v, "M1") for v in VARIANTS] + [(v, "B4s") for v in VARIANTS] + [("M1", "B4s"), ("M1-RTH", "M1-R")]
    for v in VARIANTS:
        s = f"{v}-shufTT-s0"
        if s in models:
            pairs += [(s, "B4s"), (v, s), (s, "M1")]
    return [p for p in pairs if p[0] in models and p[1] in models]


def paired_table(df, subset):
    rows = []
    for sub_name, sub in [("all", None), ("without_2020", lambda j: j.index.get_level_values("year") != 2020)]:
        for a, b in pairs_for(df):
            for metric in PAIR_METRICS:
                r = paired(df, a, b, metric, sub)
                if r:
                    rows.append(dict(note=note(subset), subset=sub_name, **r))
    t = pd.DataFrame(rows)
    t["improves_vs_M1_prereg"] = np.where(
        (t.model_b == "M1") & (t.metric == "jsd") & t.model_a.isin(VARIANTS),
        (t.mean_diff < 0) & (t.ci_muni_hi < 0) & (t.ci_year_hi < 0), np.nan).astype(object)
    t["meets_5pct_vs_B4s"] = np.where(
        (t.model_b == "B4s") & (t.metric == "jsd"),
        (t.rel_change <= -0.05) & (t.ci_muni_hi < 0) & (t.ci_year_hi < 0), np.nan).astype(object)
    return t


def region_table(df, subset):
    rows = []
    for reg in C.REGIONS:
        d = df[df.region == reg]
        for a, b in [(v, "M1") for v in VARIANTS] + [(v, "B4s") for v in VARIANTS] + [("M1", "B4s")]:
            r = paired(d, a, b, "jsd")
            if r:
                rows.append(dict(note=note(subset), region=reg, **r))
    return pd.DataFrame(rows)


def year_table(df, subset):
    rows = []
    for a, b in [(v, "M1") for v in VARIANTS] + [(v, "B4s") for v in VARIANTS] + [("M1", "B4s")]:
        x = df[df.model == a].set_index(["geocode", "year"]).jsd
        y = df[df.model == b].set_index(["geocode", "year"]).jsd
        j = (x - y).dropna().groupby(level="year").mean()
        for yr, v in j.items():
            rows.append(dict(note=note(subset), model_a=a, model_b=b, year=yr, mean_diff_jsd=v))
    return pd.DataFrame(rows)


def coef_table(subset):
    rows = []
    for tag in (f"weekly_{subset}", f"weekly_{subset}_shuf"):
        f = RAIN_DIR / f"fitted_params_{tag}.json"
        if not f.exists():
            continue
        fp = json.loads(f.read_text())
        for model, folds in fp.items():
            names = folds[0]["features"]
            th = np.array([r["theta"] for r in folds])
            de = np.array([r["delta"] for r in folds])           # (folds, R, F)
            for i, nm in enumerate(names):
                var, blk = nm.split("[")
                r = dict(subset=subset, model=model, feature=nm, variable=var, lag_block=blk.rstrip("]"),
                         theta_mean=th[:, i].mean(), theta_sd=th[:, i].std(ddof=1),
                         theta_min=th[:, i].min(), theta_max=th[:, i].max(),
                         n_folds_positive=int((th[:, i] > 0).sum()), n_folds=len(th))
                for k, reg in enumerate(C.REGIONS):
                    r[f"theta_plus_delta_{reg}"] = (th[:, i] + de[:, k, i]).mean()
                rows.append(r)
    return pd.DataFrame(rows)


def leakage_summary():
    rows = []
    for f in sorted(RAIN_DIR.glob("leakage_audit_*.csv")):
        la = pd.read_csv(f)
        checks = la.drop(columns="year")
        rows.append(dict(run=f.stem.replace("leakage_audit_", ""), folds=len(la),
                         all_pass=bool(checks.all().all()), checks=";".join(checks.columns)))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ figures
def fig_gains(pt, subsets):
    """Forest plot: mean JSD difference with municipality and year-block CIs."""
    contrasts = [("M1-R", "M1"), ("M1-RTH", "M1"), ("M1", "B4s"), ("M1-R", "B4s"), ("M1-RTH", "B4s")]
    fig, axes = plt.subplots(1, len(subsets), figsize=(6.2 * len(subsets), 4.2), sharey=True, squeeze=False)
    for ax, s in zip(axes[0], subsets):
        t = pt[s]
        t = t[(t.metric == "jsd") & (t.subset == "all")]
        for i, (a, b) in enumerate(contrasts):
            r = t[(t.model_a == a) & (t.model_b == b)]
            if not len(r):
                continue
            r = r.iloc[0]
            y = len(contrasts) - 1 - i
            ax.hlines(y + 0.12, r.ci_muni_lo, r.ci_muni_hi, color=COL[a], lw=2)
            ax.hlines(y - 0.12, r.ci_year_lo, r.ci_year_hi, color=COL[a], lw=2, ls=(0, (2, 1.5)))
            ax.plot(r.mean_diff, y, "o", color=COL[a], ms=8, mec="white", mew=1.5)
            ax.text(r.mean_diff, y + 0.3, f"{r.rel_change * 100:+.1f} %", fontsize=8.5,
                    color="#52514e", va="bottom", ha="center")
        if "M1" in [c[1] for c in contrasts]:
            b4s = t[(t.model_a == "M1") & (t.model_b == "B4s")]
            if len(b4s):
                ax.axvline(-0.05 * b4s.iloc[0].mean_b, color="#888", lw=1, ls=":")
                ax.text(-0.05 * b4s.iloc[0].mean_b, len(contrasts) - 0.55, "−5 % of B4s JSD", fontsize=8, color="#555", ha="left")
        ax.axvline(0, color="k", lw=0.8)
        ax.set_yticks(range(len(contrasts)))
        ax.set_yticklabels([f"{a} − {b}" for a, b in contrasts][::-1])
        ax.set_xlabel("mean JSD difference (negative = first model better)")
        ax.set_title(LABEL[s])
        ax.set_ylim(-1, len(contrasts))
    axes[0][0].plot([], [], "k-", lw=2, label="95 % CI, municipality bootstrap")
    axes[0][0].plot([], [], "k", lw=2, ls=(0, (2, 1.5)), label="95 % CI, year-block bootstrap")
    fig.legend(*axes[0][0].get_legend_handles_labels(), loc="lower center", ncol=2, fontsize=8.5, frameon=False)
    fig.suptitle("Rain lag extension: JSD change vs M1 and vs B4s (weekly, LOYO 2010-2023)", fontsize=11)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(RAIN_DIR / "fig_rain_gain_vs_M1_B4s.png"); plt.close(fig)


def fig_region(rt, subsets):
    fig, axes = plt.subplots(1, len(subsets), figsize=(6.2 * len(subsets), 4.2), sharey=False, squeeze=False)
    models = [("M1", "B4s"), ("M1-R", "B4s"), ("M1-RTH", "B4s")]
    for ax, s in zip(axes[0], subsets):
        t = rt[s]
        x = np.arange(len(C.REGIONS))
        for k, (a, b) in enumerate(models):
            r = t[(t.model_a == a) & (t.model_b == b)].set_index("region").reindex(C.REGIONS)
            gain = -r.mean_diff
            xx = x + (k - 1) * 0.26
            ax.bar(xx, gain, width=0.24, color=COL[a], label=f"{a} vs B4s", edgecolor="white", linewidth=1)
            ax.vlines(xx, -r.ci_muni_hi, -r.ci_muni_lo, color="#333", lw=1)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(x); ax.set_xticklabels(C.REGIONS, fontsize=9)
        ax.set_ylabel("JSD gain over B4s (positive = climate helps)")
        ax.set_title(LABEL[s])
    axes[0][0].legend(fontsize=8, frameon=False)
    fig.suptitle("Gain over B4s by region (bars: mean; lines: municipality-bootstrap 95 % CI)", fontsize=11)
    fig.tight_layout()
    fig.savefig(RAIN_DIR / "fig_rain_gain_by_region.png"); plt.close(fig)


def fig_coefficients(ct, subset, suffix):
    """Panels: rain wet/dry (M1-R and M1-RTH), temperature pos/neg (M1-RTH),
    humidity pos/neg (M1-RTH); M1's unsplit coefficient as grey reference."""
    t = ct[ct.subset == subset]
    blocks = [block_label(b) for b in LAG_BLOCKS_EXT]
    x = np.arange(len(blocks))

    def series(model, var):
        d = t[(t.model == model) & (t.variable == var)].set_index("lag_block").reindex(blocks)
        return d

    panels = [("Rainfall, M1-R", "M1-R", [("rain_wet", "wet part max(Ā,0)", "#2a78d6"), ("rain_dry", "dry part min(Ā,0)", "#eda100")], "rain"),
              ("Rainfall, M1-RTH", "M1-RTH", [("rain_wet", "wet part", "#2a78d6"), ("rain_dry", "dry part", "#eda100")], "rain"),
              ("Temperature, M1-RTH", "M1-RTH", [("temp_pos", "warm part max(Ā,0)", "#eb6834"), ("temp_neg", "cool part min(Ā,0)", "#6250d6")], "temp"),
              ("Humidity, M1-RTH", "M1-RTH", [("humid_pos", "humid part max(Ā,0)", "#1baf7a"), ("humid_neg", "dry-air part min(Ā,0)", "#e87ba4")], "humid")]
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.4), sharey=True)
    for ax, (title, model, ser, ref) in zip(axes, panels):
        for k, (var, lab, col) in enumerate(ser):
            d = series(model, var)
            if d.theta_mean.isna().all():
                continue
            xx = x + (k - 0.5) * 0.18
            ax.plot(xx, d.theta_mean, "-o", color=col, lw=2, ms=6, label=lab)
            ax.vlines(xx, d.theta_min, d.theta_max, color=col, alpha=0.7, lw=1.5)
        m1 = t[(t.model == "M1") & (t.variable == ref)].set_index("lag_block").reindex(blocks)
        if m1.theta_mean.notna().any():
            ax.plot(x, m1.theta_mean, "--", color="#888", lw=1.2, label="M1 (unsplit) reference")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(x); ax.set_xticklabels(blocks, fontsize=9)
        ax.set_xlabel("lag block (weeks before)")
        ax.set_title(title)
        ax.legend(fontsize=7.5, frameon=False, loc="best")
    axes[0].set_ylabel("national θ per SD of block-mean anomaly (logit share)")
    fig.suptitle(f"Fitted national coefficients: mean over 14 LOYO folds, line = min-max range ({LABEL[subset]}).\n"
                 "Positive wet θ: wetter → more cases that week; negative dry θ: drier (Ā<0) → more cases that week",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(RAIN_DIR / f"fig_rain_coefficients{suffix}.png"); plt.close(fig)


def main():
    """Build all rain-extension tables, checks and figures from the saved runs."""
    C.ensure_dirs()
    subsets = [s for s in (60, 300) if (RAIN_DIR / f"metrics_detail_weekly_{s}.csv").exists()
               and (C.BRAZIL_DIR / f"metrics_detail_weekly_{s}.csv").exists()]
    if not subsets:
        raise FileNotFoundError(f"No saved rain-extension results in {RAIN_DIR}. "
                                "Run the extension step first (scripts/run_pipeline.py --step extension).")
    pt, rt, cts = {}, {}, []
    rep = reproduction_check()
    for s in subsets:
        df = load(s)
        main_table(df, s).to_csv(RAIN_DIR / f"table_main_weekly_{s}.csv", index=False)
        pt[s] = paired_table(df, s); pt[s].to_csv(RAIN_DIR / f"table_paired_weekly_{s}.csv", index=False)
        rt[s] = region_table(df, s); rt[s].to_csv(RAIN_DIR / f"table_by_region_weekly_{s}.csv", index=False)
        year_table(df, s).to_csv(RAIN_DIR / f"table_by_year_weekly_{s}.csv", index=False)
        ct = coef_table(s)
        # M1 reference coefficients from the saved pilot fit
        fp = json.loads((C.BRAZIL_DIR / f"fitted_params_weekly_{s}.json").read_text())
        if "M1" in fp and not (ct.model == "M1").any():
            th = np.array([r["theta"] for r in fp["M1"]])
            for v, var in enumerate(["temp", "rain", "humid"]):
                for b, (lo, hi) in enumerate(C.LAG_BLOCKS):
                    col = th[:, v * len(C.LAG_BLOCKS) + b]
                    ct = pd.concat([ct, pd.DataFrame([dict(subset=s, model="M1", feature=f"{var}[{lo}-{hi}]", variable=var,
                                                           lag_block=f"{lo}-{hi}", theta_mean=col.mean(), theta_sd=col.std(ddof=1),
                                                           theta_min=col.min(), theta_max=col.max(),
                                                           n_folds_positive=int((col > 0).sum()), n_folds=len(col))])])
        cts.append(ct)
    ct = pd.concat(cts, ignore_index=True)
    ct.to_csv(RAIN_DIR / "table_coefficients.csv", index=False)
    ls = leakage_summary(); ls.to_csv(RAIN_DIR / "table_leakage_audit_summary.csv", index=False)
    checks = dict(m1_refit_max_abs_share_diff_vs_saved=rep, leakage_all_pass=bool(ls.all_pass.all()),
                  run_final_2024=C.RUN_FINAL_2024)
    if 60 in pt:   # pre-registered: shuffle check on the variant with the lower mean JSD at 60
        t = pt[60][(pt[60].metric == "jsd") & (pt[60].subset == "all") & (pt[60].model_b == "M1")
                   & pt[60].model_a.isin(VARIANTS)].set_index("model_a")
        best = t.mean_a.idxmin()
        checks["shuffle_variant"] = best
        s = pt[60][(pt[60].metric == "jsd") & (pt[60].subset == "all")]
        a = s[(s.model_a == f"{best}-shufTT-s0") & (s.model_b == "B4s")]
        b = s[(s.model_a == best) & (s.model_b == f"{best}-shufTT-s0")]
        if len(a) and len(b):
            a, b = a.iloc[0], b.iloc[0]
            checks["shuffle_minus_B4s"] = [a.mean_diff, a.ci_muni_lo, a.ci_muni_hi, a.ci_year_lo, a.ci_year_hi]
            checks["real_minus_shuffle"] = [b.mean_diff, b.ci_muni_lo, b.ci_muni_hi, b.ci_year_lo, b.ci_year_hi]
            checks["shuffle_check_pass"] = bool((a.mean_diff >= 0 or a.ci_muni_hi >= 0) and b.mean_diff < 0)
    (RAIN_DIR / "checks_rain.json").write_text(json.dumps(checks, indent=1))
    fig_gains(pt, subsets)
    fig_region(rt, subsets)
    for s in subsets:
        fig_coefficients(ct, s, "" if s == 60 else "_300")
    print(json.dumps(checks))
    for s in subsets:
        t = pt[s]
        print(t[(t.metric == "jsd") & (t.subset == "all")][["model_a", "model_b", "mean_a", "mean_b", "mean_diff", "rel_change",
                                                            "ci_muni_lo", "ci_muni_hi", "ci_year_lo", "ci_year_hi",
                                                            "n_years_a_better"]].round(5).to_string())
