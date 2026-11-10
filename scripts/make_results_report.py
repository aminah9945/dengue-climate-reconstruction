"""Build every table, figure and in-text number of the results report from the committed result files.

    python scripts/make_results_report.py            # writes report/figures/*.pdf, report/tables/*.tex, report/numbers.tex

Inputs (all committed in this repository; nothing is typed by hand):

* results/official/tables/*.csv                     official designs A/B/C (climate study, H1-H4)
* results/extensions/inla/tables/*_300.csv          scaled INLA, design A, 300 municipalities
* results/extensions/timing/runs/.../metrics_detail_*.csv      timing model, designs A and C (per municipality-year)
* results/extensions/timing/runs_s2c/.../metrics_detail_*.csv  search + weather variants (S2C, S2C-noONI)
* results/extensions/timing/examples_designA.csv    weekly shares for the qualitative figure; rebuilt from the
                                                    fold arrays (.npz) and the processed data when those exist locally

Paired bootstraps: 2,000 replicates, seed 1, resampling municipalities ("muni") or test years ("year"),
the same scheme as the official report. 2024 is never read.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OFF = ROOT / "results" / "official" / "tables"
INLA = ROOT / "results" / "extensions" / "inla" / "tables"
TIM = ROOT / "results" / "extensions" / "timing"
OUT = ROOT / "report"
FIG, TAB = OUT / "figures", OUT / "tables"
B, SEED = 2000, 1

UF = {11: "RO", 12: "AC", 13: "AM", 14: "RR", 15: "PA", 16: "AP", 17: "TO", 21: "MA", 22: "PI", 23: "CE", 24: "RN",
      25: "PB", 26: "PE", 27: "AL", 28: "SE", 29: "BA", 31: "MG", 32: "ES", 33: "RJ", 35: "SP", 41: "PR", 42: "SC",
      43: "RS", 50: "MS", 51: "MT", 52: "GO", 53: "DF"}
LOW_SEARCH = ["RR", "AP", "AC", "SE"]          # mostly zeros before 2013 (data_snapshots/README.md)

DESC = {
    "B0": "Uniform weeks", "B1": "Seasonal profile", "B4": "Base paper's DM", "B4s": "DM + annual size (baseline)",
    "M1": "B4s + weather + ONI", "M1-noONI": "B4s + local weather", "M1-short": "B4s + weather, lags 0--3 wk",
    "M2": "Boosted softmax (weather)", "INLA-scaled": "Scaled INLA (base paper's final)",
    "INLA-scaled-clim15": "Scaled INLA + local weather", "INLA-scaled-clim20": "Scaled INLA + weather + ONI",
    "T1": "B4s, seasonal shape shifted", "S2": "B4s + Google Trends", "S2C-noONI": "S2 + local weather",
    "S2C": "S2 + weather + ONI", "S1": "B4s + state curve (upper bound)", "S3": "S1 + Google Trends (upper bound)",
    "S3C": "S3 + weather + ONI (upper bound)",
}
COL = {"base": "#8a8f98", "climate": "#2f6fb0", "inla": "#6250d6", "search": "#e07a2e", "upper": "#b9bec7"}
GROUP = {"B0": "base", "B1": "base", "B4": "base", "B4s": "base", "M1": "climate", "M1-noONI": "climate",
         "M1-short": "climate", "M2": "climate", "INLA-scaled": "inla", "INLA-scaled-clim15": "inla",
         "INLA-scaled-clim20": "inla", "T1": "search", "S2": "search", "S2C-noONI": "search", "S2C": "search",
         "S1": "upper", "S3": "upper", "S3C": "upper"}

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.bbox": "tight", "pdf.fonttype": 42})


# ---------------------------------------------------------------- data
def timing_rows(design: str) -> pd.DataFrame:
    """Per municipality-year metrics of every timing-model variant (one run + the S2C run)."""
    main = pd.read_csv(TIM / "runs" / f"timing_design{design}_power_300" /
                       f"metrics_detail_timing_design{design}_power_300.csv")
    s2c = pd.read_csv(TIM / "runs_s2c" / f"timing_design{design}_power_300" /
                      f"metrics_detail_timing_design{design}_power_300.csv")
    key = ["geocode", "year", "model"]
    # the S2C run refits B4s, M1-noONI and S2 with the same seeds: they must agree with the main run
    both = main.merge(s2c, on=key, suffixes=("", "_r"))
    assert np.allclose(both.jsd, both.jsd_r, atol=1e-9), "S2C run does not reproduce the main run"
    m = pd.concat([main, s2c[s2c.model.isin(["S2C", "S2C-noONI"])]], ignore_index=True)
    m["uf"] = (m.geocode // 100000).map(UF)
    assert m.year.max() <= 2023
    return m


def wide(m: pd.DataFrame, metric="jsd") -> pd.DataFrame:
    return m.pivot_table(index=["geocode", "year", "region", "uf", "N"], columns="model", values=metric).reset_index()


def paired(w: pd.DataFrame, a: str, b: str, seed=SEED):
    """Mean of a - b, relative change, 95 % CIs from municipality and year bootstraps, years a better."""
    d = (w[a] - w[b]).to_numpy()
    bv = w[b].to_numpy()
    rng = np.random.default_rng(seed)
    out = {"mean_a": w[a].mean(), "mean_b": bv.mean(), "diff": d.mean(), "rel": d.mean() / bv.mean()}
    for unit, col in (("muni", "geocode"), ("year", "year")):
        g = pd.DataFrame({"u": w[col].to_numpy(), "d": d, "b": bv}).groupby("u").sum()
        k = len(g)
        wts = rng.multinomial(k, np.full(k, 1 / k), size=B)
        rel = (wts @ g.d.to_numpy()) / (wts @ g.b.to_numpy())
        out[f"rel_lo_{unit}"], out[f"rel_hi_{unit}"] = np.percentile(rel, [2.5, 97.5])
    py = w.assign(_d=d).groupby("year")._d.mean()
    out["years_better"], out["n_years"] = int((py < 0).sum()), int(len(py))
    return out


# ---------------------------------------------------------------- helpers
def pct(x, signed=True):
    return f"{x * 100:+.1f}" if signed else f"{x * 100:.1f}"


def ci(lo, hi):
    return f"[{lo * 100:+.1f}, {hi * 100:+.1f}]"


def write_table(name: str, header: list[str], rows: list[list[str]], align: str, caption: str, label: str,
                note: str = ""):
    lines = [r"\begin{table}[t]", r"\centering", r"\small", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             r"\resizebox{\linewidth}{!}{%", f"\\begin{{tabular}}{{{align}}}", r"\toprule", " & ".join(header) + r" \\", r"\midrule"]
    for r in rows:
        lines.append(r"\midrule" if r == ["--"] else " & ".join(r) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}"]
    if note:
        lines.append(f"\\par\\vspace{{2pt}}\\footnotesize {note}")
    lines.append(r"\end{table}")
    (TAB / f"{name}.tex").write_text("\n".join(lines) + "\n")


def sep_groups(rows: list, groups: list) -> list:
    """Insert a rule ("--") between consecutive rows of different groups."""
    out = []
    for i, (r, g) in enumerate(zip(rows, groups)):
        if i and g != groups[i - 1]:
            out.append(["--"])
        out.append(r)
    return out


def esc(s: str) -> str:
    return s.replace("_", r"\_").replace("%", r"\%")


NUM: dict[str, str] = {}


def num(key: str, value: str):
    assert key.isalpha(), key
    NUM[key] = value


# ---------------------------------------------------------------- main
def main():
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    mA, mC = timing_rows("A"), timing_rows("C")
    wA, wC = wide(mA), wide(mC)
    pA, pC = wide(mA, "peak_err"), wide(mC, "peak_err")
    oA = wide(mA, "onset_err")
    off = pd.read_csv(OFF / "model_means.csv")
    offA = off[off.run == "designA_power_300"].set_index("model")
    offC = off[off.run == "designC_power_300"].set_index("model")
    inla = pd.read_csv(INLA / "model_means_300.csv").set_index("model")
    inla_y = pd.read_csv(INLA / "jsd_by_test_year_300.csv").set_index("year")

    # consistency: the timing run's baselines equal the official run's
    for mdl in ("B1", "B4s", "M1", "M1-noONI"):
        assert abs(wA[mdl].mean() - offA.loc[mdl, "jsd"]) < 1e-6, mdl
        assert abs(wA[mdl].mean() - inla.loc[mdl, "jsd"]) < 1e-6, mdl
    b4s_A, b4s_C = wA.B4s.mean(), wC.B4s.mean()

    # ---- Table 1: main comparison
    order = ["B0", "B1", "B4", "B4s", "M1-noONI", "M1", "M1-short", "M2", "INLA-scaled", "INLA-scaled-clim15",
             "INLA-scaled-clim20", "T1", "S2", "S2C-noONI", "S2C", "S1", "S3", "S3C"]
    rows, fig1 = [], []
    for mdl in order:
        if mdl in wA:
            jA, peak, onset = wA[mdl].mean(), pA[mdl].mean(), oA[mdl].mean()
            rng = np.random.default_rng(SEED)
            sums = wA.groupby("geocode")[mdl].sum().to_numpy()
            cnts = wA.groupby("geocode")[mdl].count().to_numpy()
            idx = rng.integers(0, len(sums), size=(B, len(sums)))
            bt = sums[idx].sum(1) / cnts[idx].sum(1)
            lo, hi = np.percentile(bt, [2.5, 97.5])
            src = "timing"
        elif mdl in inla.index:
            jA, peak, onset = inla.loc[mdl, "jsd"], inla.loc[mdl, "peak_err"], np.nan
            lo, hi, src = inla.loc[mdl, "jsd_ci_lo"], inla.loc[mdl, "jsd_ci_hi"], "inla"
        else:
            jA, peak, onset = offA.loc[mdl, "jsd"], offA.loc[mdl, "peak_err"], np.nan
            lo, hi, src = offA.loc[mdl, "jsd_ci_lo"], offA.loc[mdl, "jsd_ci_hi"], "official"
        if mdl in wC:
            jC = wC[mdl].mean()
        elif mdl in offC.index:
            jC = offC.loc[mdl, "jsd"]
        else:
            jC = np.nan
        fig1.append((mdl, jA, lo, hi))
        rows.append([mdl, DESC[mdl], f"{jA:.4f}", f"[{lo:.4f}, {hi:.4f}]", pct(jA / b4s_A - 1) if mdl != "B4s" else "--",
                     f"{peak:.2f}", "--" if np.isnan(onset) else f"{onset:.2f}",
                     "--" if np.isnan(jC) else f"{jC:.4f}",
                     "--" if np.isnan(jC) or mdl == "B4s" else pct(jC / b4s_C - 1)])
        tag = {"B4s": "Base", "M1-noONI": "Weather", "INLA-scaled": "Inla", "S2": "Trends", "S1": "State"}
        if mdl in tag:
            num(f"jsd{tag[mdl]}A", f"{jA:.4f}")
    rows = sep_groups(rows, [GROUP[r[0]] for r in rows])
    write_table("tab_main", ["Model", "Description", "JSD (A)", "95\\% CI (towns)", "vs B4s \\%", "Peak err (wk)",
                             "Onset err (wk)", "JSD (C)", "vs B4s \\% (C)"],
                [[esc(c) for c in r] if r != ["--"] else r for r in rows], "llrrrrrrr",
                "Mean shape error (JSD, lower is better) of every model on design A (expanding window, 12 test years, "
                "3{,}194 municipality-years) and design C (leave one year out, 14 folds). Negative ``vs B4s'' = less "
                "error than the B4s baseline. Scaled INLA was run on design A only.", "tab:main",
                "Upper-bound rows need the same year's weekly counts of other towns in the state, which a truly "
                "missing year does not have; they are reported as a ceiling, not as a usable method.")

    # ---- Figure 1: model comparison
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    f1 = [r for r in fig1 if r[0] != "B0"]
    y = np.arange(len(f1))[::-1]
    for yy, (mdl, j, lo, hi) in zip(y, f1):
        ax.barh(yy, j, color=COL[GROUP[mdl]], height=0.7)
        ax.errorbar(j, yy, xerr=[[j - lo], [hi - j]], color="#222", lw=0.8, capsize=2)
    ax.axvline(b4s_A, color="#222", ls="--", lw=0.8)
    ax.set_yticks(y, [f"{m}  ({DESC[m]})" for m, *_ in f1], fontsize=7.5)
    ax.set_xlim(0.08, 0.135)
    ax.set_xlabel("Mean JSD, design A (lower is better); bars: 95 % municipality bootstrap CI")
    handles = [plt.Rectangle((0, 0), 1, 1, color=COL[k]) for k in ("base", "climate", "inla", "search", "upper")]
    ax.legend(handles, ["Baselines", "Weather", "Scaled INLA", "Search / timing", "Upper bound (state curve)"],
              fontsize=7, loc="lower right", frameon=False)
    fig.savefig(FIG / "fig1_model_comparison.pdf")
    plt.close(fig)

    # ---- Table 2 / Figure 4: ablation (paired, design A)
    abl = [
        ("Annual size terms", "B4s", "B4", "official"),
        ("Local weather", "M1-noONI", "B4s", "timing"),
        ("ONI on top of local weather", "M1", "M1-noONI", "timing"),
        ("Long lags (0--20 vs 0--3 wk)", "M1", "M1-short", "official"),
        ("Boosted learner instead of DM", "M2", "B4s", "official"),
        ("Scaled INLA instead of B4s", "INLA-scaled", "B4s", "inla"),
        ("Shift-only timing (T1)", "T1", "B4s", "timing"),
        ("Google Trends block (S2)", "S2", "B4s", "timing"),
        ("Local weather on top of Trends", "S2C-noONI", "S2", "timing"),
        ("ONI on top of Trends + weather", "S2C", "S2C-noONI", "timing"),
        ("State-curve block (S1)", "S1", "B4s", "timing"),
        ("Trends on top of state curve", "S3", "S1", "timing"),
        ("Weather + ONI on top of S3", "S3C", "S3", "timing"),
        ("State curve as features vs shift", "S1", "T1", "timing"),
        ("State curve from 50 % of towns", "S1-p50", "S1", "timing"),
        ("State curve from 25 % of towns", "S1-p25", "S1", "timing"),
        ("State curve, non-study towns only", "S1-ns", "S1", "timing"),
        ("Wrong-year Trends (refitted)", "S2-wy", "S2", "timing"),
        ("Wrong-year state curve (refitted)", "S1-wy", "S1", "timing"),
        ("Wrong-year signals at test only", "S3-wyTest", "S3", "timing"),
    ]
    pc = pd.read_csv(OFF / "paired_contrasts.csv")
    pc = pc[(pc.run == "designA_power_300") & (pc.metric == "jsd")]
    arows, fig4 = [], []
    for name, a, b, src in abl:
        if src == "timing":
            r = paired(wA, a, b)
            rel, lom, him, loy, hiy = r["rel"], r["rel_lo_muni"], r["rel_hi_muni"], r["rel_lo_year"], r["rel_hi_year"]
            yb, ma, mb = f"{r['years_better']}/{r['n_years']}", r["mean_a"], r["mean_b"]
        elif src == "official":
            q = pc[(pc.model_a == a) & (pc.model_b == b)].iloc[0]
            rel, lom, him, loy, hiy = q.rel_change, q.rel_ci_muni_lo, q.rel_ci_muni_hi, q.rel_ci_year_lo, q.rel_ci_year_hi
            yb, ma, mb = f"{int(q.n_folds_a_better)}/{int(q.n_folds)}", q.mean_a, q.mean_b
        else:   # INLA: per-municipality rows are not committed -> year-level paired comparison only
            ya = inla_y[a].loc[2012:2023]
            yb_ = wA.groupby("year").B4s.mean()
            ma, mb = inla.loc[a, "jsd"], inla.loc[b, "jsd"]
            rel = ma / mb - 1
            rng = np.random.default_rng(SEED)
            n = wA.groupby("year").size().reindex(ya.index).to_numpy()
            da, db = (ya.to_numpy() * n), (yb_.reindex(ya.index).to_numpy() * n)
            wts = rng.multinomial(len(n), np.full(len(n), 1 / len(n)), size=B)
            bt = (wts @ da) / (wts @ db) - 1
            loy, hiy = np.percentile(bt, [2.5, 97.5])
            lom = him = np.nan
            yb = f"{int((ya.to_numpy() < yb_.reindex(ya.index).to_numpy()).sum())}/{len(n)}"
        clear = (not np.isnan(lom)) and (him < 0 or lom > 0) and (hiy < 0 or loy > 0)
        verdict = ("clearly better" if rel < 0 else "clearly worse") if clear else "not clear"
        arows.append([name, esc(f"{a} vs {b}"), f"{ma:.4f}", f"{mb:.4f}", pct(rel),
                      "--" if np.isnan(lom) else ci(lom, him), ci(loy, hiy), yb, verdict])
        fig4.append((name, rel, lom, him, loy, hiy))
    block = [0] * 5 + [1] + [2] * 4 + [3] * 4 + [4] * 3 + [5] * 3
    arows = sep_groups(arows, block)
    write_table("tab_ablation", ["Component changed", "Models", "JSD with", "JSD without", "Change \\%",
                                 "95\\% CI towns", "95\\% CI years", "Years better", "Verdict"],
                [[esc(c) if k == 0 else c for k, c in enumerate(r)] if r != ["--"] else r for r in arows],
                "llrrrrrrl",
                "Ablation study, design A: each row adds, removes or alters one component and reports the paired "
                "change in mean JSD (negative = the component lowers the error). ``Clearly'' requires both the "
                "municipality and the year bootstrap intervals to exclude zero.", "tab:ablation",
                "Scaled INLA: per-municipality rows are not stored, so only the year-level interval is given. "
                "Rows 1, 4 and 5 come from the official climate run (same folds and municipalities).")

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    yy = np.arange(len(fig4))[::-1]
    for y0, (name, rel, lom, him, loy, hiy) in zip(yy, fig4):
        ax.plot([loy * 100, hiy * 100], [y0, y0], color="#b9bec7", lw=3, solid_capstyle="butt")
        if not np.isnan(lom):
            ax.plot([lom * 100, him * 100], [y0, y0], color="#14213d", lw=1.4)
        ax.plot(rel * 100, y0, "o", color="#e07a2e" if rel < 0 else "#2f6fb0", ms=4)
    ax.axvline(0, color="#222", lw=0.8)
    ax.set_yticks(yy, [n.replace("--", "\u2013") for n, *_ in fig4], fontsize=7.5)
    ax.set_xlabel("Change in mean JSD when the component is added (%); left of 0 = less error\n"
                  "dark line: 95 % municipality CI; grey band: 95 % year CI")
    fig.savefig(FIG / "fig4_ablation.pdf")
    plt.close(fig)

    # ---- Table 3: key paired contrasts in both designs (significance of the search result)
    key = [("S2", "B4s"), ("S2", "M1-noONI"), ("S2C-noONI", "S2"), ("T1", "B4s"), ("S1", "B4s"), ("S3", "S1")]
    krows = []
    for a, b in key:
        rA, rC = paired(wA, a, b), paired(wC, a, b)
        krows.append([esc(f"{a} vs {b}"), pct(rA["rel"]), ci(rA["rel_lo_muni"], rA["rel_hi_muni"]),
                      ci(rA["rel_lo_year"], rA["rel_hi_year"]), f"{rA['years_better']}/{rA['n_years']}",
                      pct(rC["rel"]), ci(rC["rel_lo_muni"], rC["rel_hi_muni"]),
                      ci(rC["rel_lo_year"], rC["rel_hi_year"]), f"{rC['years_better']}/{rC['n_years']}"])
        if (a, b) == ("S2", "B4s"):
            num("trendsGainA", pct(rA["rel"], False).lstrip("-"))
            num("trendsGainC", pct(rC["rel"], False).lstrip("-"))
            num("trendsCIyearA", ci(rA["rel_lo_year"], rA["rel_hi_year"]))
            num("trendsCImuniA", ci(rA["rel_lo_muni"], rA["rel_hi_muni"]))
            num("trendsYearsA", f"{rA['years_better']} of {rA['n_years']}")
            num("trendsYearsC", f"{rC['years_better']} of {rC['n_years']}")
        if (a, b) == ("S2C-noONI", "S2"):
            num("weatherOnTrendsA", pct(rA["rel"]))
            num("weatherOnTrendsYearsA", f"{rA['years_better']} of {rA['n_years']}")
    write_table("tab_contrasts", ["Contrast", "A: \\%", "A: CI towns", "A: CI years", "A: years", "C: \\%",
                                  "C: CI towns", "C: CI years", "C: years"], krows, "lrrrrrrrr",
                "Paired bootstrap tests (2{,}000 replicates) of the main contrasts in both designs. \\% = relative "
                "change in mean JSD of the first model vs the second; ``years'' = test years in which the first "
                "model had the lower mean JSD.", "tab:contrasts")

    # ---- Figure 2: JSD by test year
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    by = wA.groupby("year")
    for mdl, lab, c, ls in (("B4s", "B4s (baseline)", COL["base"], "-"), ("M1-noONI", "B4s + local weather",
                            COL["climate"], "-"), ("S2", "B4s + Google Trends", COL["search"], "-"),
                            ("S1", "State curve (upper bound)", "#8a8f98", ":")):
        ax.plot(by[mdl].mean().index, by[mdl].mean().values, marker="o", ms=3, color=c, ls=ls, label=lab)
    ax.plot(inla_y.index, inla_y["INLA-scaled"], marker="s", ms=3, color=COL["inla"], label="Scaled INLA")
    ax.set_xlabel("Test year (design A; training = 2010 to year before)")
    ax.set_ylabel("Mean JSD")
    ax.legend(fontsize=7, frameon=False, ncol=2)
    fig.savefig(FIG / "fig2_jsd_by_year.pdf")
    plt.close(fig)

    # ---- Figure 3: gain distribution and by year
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.8), gridspec_kw={"width_ratios": [1.1, 1]})
    d = (wA.S2 - wA.B4s)
    axs[0].hist(d.clip(-0.15, 0.15), bins=60, color=COL["search"])
    axs[0].axvline(0, color="#222", lw=0.8)
    axs[0].set_xlabel("JSD(S2) - JSD(B4s) per municipality-year")
    axs[0].set_ylabel("Count")
    share_better = (d < 0).mean()
    num("trendsShareBetter", f"{share_better * 100:.0f}")
    axs[0].set_title(f"S2 better in {share_better * 100:.0f} % of 3,194 cases", fontsize=8)
    yrs = sorted(wA.year.unique())
    rel = [(wA[wA.year == y].S2.mean() / wA[wA.year == y].B4s.mean() - 1) * 100 for y in yrs]
    relw = [(wA[wA.year == y]["S2C-noONI"].mean() / wA[wA.year == y].S2.mean() - 1) * 100 for y in yrs]
    x = np.arange(len(yrs))
    axs[1].bar(x - 0.2, rel, 0.4, color=COL["search"], label="Trends vs B4s")
    axs[1].bar(x + 0.2, relw, 0.4, color=COL["climate"], label="+ weather vs Trends")
    axs[1].axhline(0, color="#222", lw=0.8)
    axs[1].set_xticks(x, [str(y)[2:] for y in yrs], fontsize=7)
    axs[1].set_xlabel("Test year (20xx)")
    axs[1].set_ylabel("Change in mean JSD (%)")
    axs[1].legend(fontsize=7, frameon=False)
    fig.savefig(FIG / "fig3_gain_distribution.pdf")
    plt.close(fig)

    # ---- Figure 5: peak and onset error
    fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.6))
    for k_, (frame, lab) in enumerate(((pA, "Peak-week error (weeks)"), (oA, "Onset-week error (weeks)"))):
        for mdl, c, ls in (("B4s", COL["base"], "-"), ("M1-noONI", COL["climate"], "-"), ("S2", COL["search"], "-"),
                           ("S1", "#8a8f98", ":")):
            v = np.sort(frame[mdl].to_numpy())
            axs[k_].plot(v, np.arange(1, len(v) + 1) / len(v), color=c, ls=ls, label=mdl)
        axs[k_].set_xlim(0, 20)
        axs[k_].set_xlabel(lab)
        axs[k_].set_ylabel("Share of municipality-years")
    axs[0].legend(fontsize=7, frameon=False)
    fig.savefig(FIG / "fig5_peak_onset_ecdf.pdf")
    plt.close(fig)
    for mdl, tg in (("B4s", "Base"), ("S2", "Trends"), ("S1", "State")):
        num(f"peak{tg}", f"{pA[mdl].mean():.2f}")
        num(f"onset{tg}", f"{oA[mdl].mean():.2f}")
        num(f"peakWithinTwo{tg}", f"{(pA[mdl] <= 2).mean() * 100:.0f}")

    # ---- Figure 6: learning curve (design B, official)
    lc = pd.read_csv(OFF / "lc_jsd_averaged.csv")
    lc = lc[lc.main.astype(str).str.lower() == "true"]
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    for mdl, c in (("B1", "#b9bec7"), ("B4s", COL["base"]), ("M1", COL["climate"]), ("M2", "#14213d")):
        s = lc[lc.model == mdl].sort_values("k")
        ax.plot(s.k, s.jsd, marker="o", ms=3, color=c, label=mdl)
    ax.set_xlabel("Training years k (test years 2019, 2021, 2023 fixed)")
    ax.set_ylabel("Mean JSD")
    ax.legend(fontsize=7, frameon=False)
    fig.savefig(FIG / "fig6_learning_curve.pdf")
    plt.close(fig)

    # ---- Figure 10 / Table: learning curve with Google Trends (design B + the k = 1 fold)
    lcdir = TIM / "runs_lc"
    lb = pd.read_csv(lcdir / "timing_designB_power_300" / "metrics_detail_timing_designB_power_300.csv")
    l1 = pd.read_csv(lcdir / "timing_designA_power_300" / "metrics_detail_timing_designA_power_300.csv")
    l1 = l1[l1.year == 2011].assign(k=1)
    lcm = pd.concat([lb, l1], ignore_index=True)
    assert lcm.year.max() <= 2023
    lcm["uf"] = (lcm.geocode // 100000).map(UF)
    lrows, curve = [], []
    for k in sorted(lcm.k.unique()):
        g = lcm[lcm.k == k]
        wk = g.pivot_table(index=["geocode", "year", "region", "uf", "N"], columns="model", values="jsd").reset_index()
        ny = wk.year.nunique()
        r1, r2, r3 = paired(wk, "S2", "B4s"), paired(wk, "M1-noONI", "B4s"), paired(wk, "S2C-noONI", "S2")
        yrs = " ".join(str(y) for y in sorted(wk.year.unique()))
        lrows.append([str(k), yrs, f"{wk.B4s.mean():.4f}", f"{wk.S2.mean():.4f}", pct(r1["rel"]),
                      ci(r1["rel_lo_muni"], r1["rel_hi_muni"]), pct(r2["rel"]), pct(r3["rel"])])
        curve.append(dict(k=k, ny=ny, **{m_: wk[m_].mean() for m_ in ("B1", "B4s", "M1-noONI", "S2", "S2C-noONI")},
                          rel=r1["rel"], lo=r1["rel_lo_muni"], hi=r1["rel_hi_muni"]))
    curve = pd.DataFrame(curve)
    lrows = sep_groups(lrows, [0 if int(r[0]) == 1 else (1 if int(r[0]) <= 9 else 2) for r in lrows])
    write_table("tab_lc", ["k", "Test years", "B4s", "S2 (Trends)", "Trends vs B4s \\%", "95\\% CI towns",
                           "Weather vs B4s \\%", "Weather on Trends \\%"], lrows, "llrrrrrr",
                "Learning curve with Google Trends. $k$ = number of training years immediately before the test "
                "year. $k=1$: train 2010, test 2011. $k=2$--9: all three fixed test years (2019, 2021, 2023); "
                "$k \\ge 10$: fewer test years remain, so these rows are not comparable with the others.", "tab:lc",
                "Year-bootstrap intervals are not given: at most three test years per row.")
    main_k = curve[(curve.k >= 2) & (curve.k <= 9)]
    num("lcGainMin", pct(main_k.rel.max(), False).lstrip("-"))
    num("lcGainMax", pct(main_k.rel.min(), False).lstrip("-"))
    num("lcGainTwo", pct(curve[curve.k == 2].rel.iloc[0], False).lstrip("-"))
    num("lcWeatherTwo", pct(curve[curve.k == 2]["M1-noONI"].iloc[0] / curve[curve.k == 2].B4s.iloc[0] - 1))
    num("lcWeatherOnTrendsTwo", pct(curve[curve.k == 2]["S2C-noONI"].iloc[0] / curve[curve.k == 2].S2.iloc[0] - 1))
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.0), gridspec_kw={"width_ratios": [1.25, 1], "wspace": 0.35})
    cm = curve[curve.k <= 9]
    for mdl, c, lab in (("B1", "#b9bec7", "B1 seasonal"), ("B4s", COL["base"], "B4s"),
                        ("M1-noONI", COL["climate"], "B4s + weather"), ("S2", COL["search"], "B4s + Trends"),
                        ("S2C-noONI", "#8c4a14", "Trends + weather")):
        axs[0].plot(cm.k, cm[mdl], marker="o", ms=3, color=c, label=lab)
    axs[0].axvspan(0.5, 1.5, color="#eeeeee", zorder=0)
    axs[0].set_xlabel("Training years k")
    axs[0].set_ylabel("Mean JSD")
    axs[0].legend(fontsize=6.5, frameon=False)
    axs[1].errorbar(curve.k, curve.rel * 100, yerr=[(curve.rel - curve.lo) * 100, (curve.hi - curve.rel) * 100],
                    fmt="o", ms=3, color=COL["search"], capsize=2)
    axs[1].axhline(0, color="#222", lw=0.8)
    axs[1].axvline(9.5, color="#999", ls=":", lw=0.8)
    axs[1].set_xlabel("Training years k")
    axs[1].set_ylabel("Trends vs B4s (%)")
    fig.savefig(FIG / "fig10_learning_curve_trends.pdf")
    plt.close(fig)

    # ---- Figure 7 / Table 4: failure analysis by state and region
    st = wA.groupby("uf").apply(lambda g: pd.Series({"rel": g.S2.mean() / g.B4s.mean() - 1, "n": len(g)}),
                                include_groups=False).sort_values("rel")
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    cols = [COL["climate"] if u in LOW_SEARCH else COL["search"] for u in st.index]
    ax.bar(range(len(st)), st.rel * 100, color=cols)
    ax.axhline(0, color="#222", lw=0.8)
    ax.set_xticks(range(len(st)), st.index, fontsize=7, rotation=90)
    ax.set_ylabel("Trends vs B4s, change in JSD (%)")
    ax.set_xlabel("State (blue = few searches before 2013)")
    fig.savefig(FIG / "fig7_gain_by_state.pdf")
    plt.close(fig)
    worse = st[st.rel > 0]
    num("statesWorse", str(len(worse)))
    num("statesWorseList", ", ".join(worse.index) if len(worse) else "none")
    num("statesTotal", str(len(st)))
    rg = []
    for reg, g in wA.groupby("region"):
        r = paired(g, "S2", "B4s")
        rg.append([reg, str(g.geocode.nunique()), f"{r['mean_b']:.4f}", f"{r['mean_a']:.4f}", pct(r["rel"]),
                   ci(r["rel_lo_muni"], r["rel_hi_muni"]), f"{r['years_better']}/{r['n_years']}"])
    write_table("tab_region", ["Region", "Towns", "B4s", "S2 (Trends)", "Change \\%", "95\\% CI towns",
                               "Years better"], rg, "lrrrrrr",
                "Google Trends gain by region (design A, paired municipality bootstrap).", "tab:region")

    # epidemic-size strata: annual total relative to the municipality's median over 2012-2023 test years
    rs = wA.assign(ratio=wA.N / wA.groupby("geocode").N.transform("median"))
    rs["stratum"] = pd.qcut(rs.ratio, 3, labels=["small year", "typical year", "large year"])
    srows = []
    for s_, g in rs.groupby("stratum", observed=True):
        r1, r2 = paired(g, "S2", "B4s"), paired(g, "M1-noONI", "B4s")
        srows.append([s_, str(len(g)), pct(r1["rel"]), ci(r1["rel_lo_muni"], r1["rel_hi_muni"]), pct(r2["rel"]),
                      ci(r2["rel_lo_muni"], r2["rel_hi_muni"])])
    write_table("tab_size", ["Year size (tercile)", "n", "Trends \\%", "CI towns", "Weather \\%", "CI towns"],
                srows, "lrrrrr",
                "Gain by epidemic size. Strata are terciles of a municipality-year's annual total divided by that "
                "municipality's median total over the test years (a descriptive split, not used by any model).",
                "tab:size")

    # ---- Figure 8: wrong-year checks
    fig, ax = plt.subplots(figsize=(5.4, 2.6))
    names = ["B1", "B4s", "S2", "S2-wy", "S1", "S1-wy", "S3", "S3-wy", "S3-wyTest"]
    vals = [wA[n].mean() for n in names]
    cc = [COL["base"], COL["base"], COL["search"], "#f2c19a", "#8a8f98", "#d7dade", "#8a8f98", "#d7dade", "#c94f4f"]
    ax.bar(range(len(names)), vals, color=cc)
    ax.axhline(b4s_A, color="#222", ls="--", lw=0.8)
    ax.set_xticks(range(len(names)), names, fontsize=7, rotation=30)
    ax.set_ylim(0.08, 0.14)
    ax.set_ylabel("Mean JSD, design A")
    fig.savefig(FIG / "fig8_wrong_year.pdf")
    plt.close(fig)
    num("wyTwo", f"{wA['S2-wy'].mean():.4f}")
    num("wyTest", f"{wA['S3-wyTest'].mean():.4f}")

    # ---- Figure 9: qualitative examples (true vs rebuilt weekly shares)
    ex_csv = TIM / "examples_designA.csv"
    if not ex_csv.exists():
        build_examples(wA, ex_csv)
    if ex_csv.exists():
        ex = pd.read_csv(ex_csv)
        cases = ex[["geocode", "year", "label"]].drop_duplicates().to_numpy()
        fig, axs = plt.subplots(2, 2, figsize=(6.6, 4.4), sharex=True)
        for axx, (g, y, lab) in zip(axs.ravel(), cases):
            e = ex[(ex.geocode == g) & (ex.year == y)].sort_values("week")
            axx.bar(e.week, e.truth, color="#d7dade", width=1.0, label="True weekly share")
            axx.plot(e.week, e.B4s, color=COL["base"], label="B4s")
            axx.plot(e.week, e.S2, color=COL["search"], label="B4s + Trends")
            axx.set_title(f"{lab}: {g} ({e.uf.iloc[0]}), {y}", fontsize=7.5)
        axs[1, 0].set_xlabel("Epi week")
        axs[1, 1].set_xlabel("Epi week")
        axs[0, 0].legend(fontsize=6.5, frameon=False)
        fig.savefig(FIG / "fig9_examples.pdf")
        plt.close(fig)

    # ---- audits table
    arow = []
    for d_ in ("A", "C"):
        j = json.loads((TIM / "runs" / f"timing_design{d_}_power_300" /
                        f"checks_timing_design{d_}_power_300.json").read_text())
        arow.append([d_, str(j["n_folds"]), "pass" if j["official_leakage_audit_all_pass"] else "FAIL",
                     "pass" if j["feature_audit_all_pass"] else "FAIL",
                     "pass" if j.get("exact_sums", True) else "FAIL"])
    write_table("tab_audits", ["Design", "Folds", "Leakage audit", "Feature audit (train-only)", "Exact sums"],
                arow, "lrlll", "Automatic checks of the timing-model runs, from the run check files.", "tab:audits")

    # ---- in-text numbers
    num("nRows", f"{len(wA):,}".replace(",", "{,}"))
    num("nRowsC", f"{len(wC):,}".replace(",", "{,}"))
    num("inlaGainA", pct(inla.loc["INLA-scaled", "jsd"] / b4s_A - 1, False).lstrip("-"))
    num("trendsVsInla", pct(wA.S2.mean() / inla.loc["INLA-scaled", "jsd"] - 1, False).lstrip("-"))
    num("weatherGainA", pct(wA["M1-noONI"].mean() / b4s_A - 1, False).lstrip("-"))
    (OUT / "numbers.tex").write_text("% generated by scripts/make_results_report.py -- do not edit\n" +
                                     "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in sorted(NUM.items())))
    print(f"wrote {len(list(FIG.glob('*.pdf')))} figures, {len(list(TAB.glob('*.tex')))} tables, "
          f"{len(NUM)} numbers into {OUT}")


def build_examples(wA: pd.DataFrame, path: Path):
    """Weekly shares (truth, B4s, S2) for four municipality-years chosen by rule, from the local fold arrays.

    Needs the fold .npz files (written by the timing run, not committed) and the processed dengue data
    (DATA_DIR). Rule: the municipality-years with the largest S2 gain, the median gain, the largest S2 loss,
    and the largest gain among towns in the four low-search states."""
    folds = TIM / "runs" / "timing_designA_power_300" / "folds"
    data_dir = Path(os.environ.get("DATA_DIR", ROOT / "data"))
    dpath = data_dir / "dengue_weekly.parquet"
    if not dpath.exists() or not list(folds.glob("*.npz")):
        print("examples: fold arrays or processed data not found; figure 9 skipped")
        return
    w = wA[wA.N >= 100].assign(d=lambda t: t.S2 - t.B4s).sort_values("d")
    picks = [("Largest gain", w.iloc[0]), ("Median case", w.iloc[len(w) // 2]), ("Largest loss", w.iloc[-1]),
             ("Low-search state", w[w.uf.isin(LOW_SEARCH)].iloc[0])]
    dw = pd.read_parquet(dpath)
    yc = "epi_year" if "epi_year" in dw else "year"
    wc = "epi_week" if "epi_week" in dw else "week"
    cc = [c for c in ("cases", "casos", "y") if c in dw][0]
    out = []
    for lab, r in picks:
        g, y = int(r.geocode), int(r.year)
        assert y <= 2023
        f = next(folds.glob(f"A_T{y}_k*.npz"))
        z = np.load(f)
        i = int(np.where(z["geocodes"] == g)[0][0])
        t = dw[(dw.geocode == g) & (dw[yc] == y)].sort_values(wc)
        truth = t[cc].to_numpy(float)
        n = len(truth)
        for k in range(n):
            out.append(dict(label=lab, geocode=g, year=y, uf=r.uf, week=k + 1, truth=truth[k] / truth.sum(),
                            B4s=z["pred__B4s"][i, k] / z["pred__B4s"][i, :n].sum(),
                            S2=z["pred__S2"][i, k] / z["pred__S2"][i, :n].sum()))
    pd.DataFrame(out).to_csv(path, index=False, float_format="%.6g")


if __name__ == "__main__":
    main()
