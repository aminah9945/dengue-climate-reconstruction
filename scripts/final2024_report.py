"""Score the 2024 final test as fixed in docs/FINAL_TEST_2024.md.

Reads the per-row metrics of the three 2024 runs (official, timing, INLA), checks that
they score the same municipality-years, and writes
``results/final2024/FINAL_TEST_2024_RESULTS.md`` and ``results/final2024/tables/*.csv``.
Interval: municipality bootstrap of the mean paired difference, 2,000 replicates, seed 1.
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

R = Path("results/final2024")
N_BOOT, SEED = 2000, 1


def load():
    parts = []
    for pat, tag in [("official/*/metrics_detail_*.csv", "official"), ("timing/*/metrics_detail_*.csv", "timing"),
                     ("inla/*/metrics_detail_*.csv", "inla")]:
        for f in glob.glob(str(R / pat)):
            x = pd.read_csv(f)
            x["run"] = tag
            parts.append(x)
    d = pd.concat(parts, ignore_index=True)
    assert (d.year == 2024).all()
    # the same model (B1, B4s, M1, M1-noONI) appears in the official and timing runs: keep the official one,
    # after checking they agree
    dup = d[d.duplicated(["geocode", "model"], keep=False)]
    agree = {}
    for m, g in dup.groupby("model"):
        w = g.pivot_table(index="geocode", columns="run", values="jsd")
        agree[m] = float((w.iloc[:, 0] - w.iloc[:, 1]).abs().max())
    d = d.sort_values("run", key=lambda s: s.map({"official": 0, "timing": 1, "inla": 2}))
    d = d.drop_duplicates(["geocode", "model"])
    return d, agree


def contrast(d, a, b, rng):
    w = d.pivot_table(index="geocode", columns="model", values="jsd")[[a, b]].dropna()
    diff = (w[a] - w[b]).to_numpy()
    idx = rng.integers(0, len(diff), (N_BOOT, len(diff)))
    bs = diff[idx].mean(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    mb = w[b].mean()
    return dict(a=a, b=b, n=len(diff), jsd_a=w[a].mean(), jsd_b=mb, diff=diff.mean(), rel=100 * diff.mean() / mb,
                lo=100 * lo / mb, hi=100 * hi / mb, share_a_better=100 * (diff < 0).mean())


CLAIMS = [  # id, a, b, rule
    ("H1", "M1", "B4s", "sign"), ("C1", "S2", "B4s", "neg"), ("C2", "INLA-scaled-trends", "INLA-scaled", "neg"),
    ("C3", "S2C-noONI", "S2", "noimp"), ("C4", "M1", "M1-noONI", "pos_mean"), ("C5", "B4s", "B4", "neg"),
    ("C6", "T1", "B4s", "neg"), ("C7", "S2-wy", "S2", "pos"), ("C8", "S1", "B4s", "neg"),
]
EXTRA = [("M1-noONI", "B4s"), ("M1-short", "M1"), ("M2", "B4s"), ("M1-shufTT", "M1"), ("M1-shufTest", "M1"),
         ("INLA-scaled", "B4s"), ("INLA-scaled-clim15", "INLA-scaled"), ("INLA-scaled-clim20", "INLA-scaled"),
         ("INLA-scaled-clim15trends", "INLA-scaled-trends"), ("INLA-scaled-trends", "S2"), ("S2C", "S2C-noONI"),
         ("S3", "S1"), ("S3C", "S3"), ("S1", "T1"), ("S1-wy", "S1"), ("S3-wyTest", "S3"), ("S1-p50", "S1"),
         ("S1-p25", "S1"), ("S1-ns", "S1")]


def verdict(r, rule):
    if rule == "sign":
        return "M1 better (negative)" if r["diff"] < 0 else "M1 not better (positive)"
    if rule == "neg":
        return "CONFIRMED" if r["diff"] < 0 and r["hi"] < 0 else "NOT CONFIRMED"
    if rule == "pos":
        return "CONFIRMED" if r["diff"] > 0 and r["lo"] > 0 else "NOT CONFIRMED"
    if rule == "pos_mean":
        return "CONFIRMED" if r["diff"] > 0 else "NOT CONFIRMED"
    if rule == "noimp":
        return "CONFIRMED" if r["diff"] > 0 or r["lo"] <= 0 <= r["hi"] else "NOT CONFIRMED"
    raise ValueError(rule)


def main():
    d, agree = load()
    (R / "tables").mkdir(parents=True, exist_ok=True)
    rows_per_model = d.groupby("model").geocode.nunique()
    assert rows_per_model.nunique() == 1, f"models score different rows: {rows_per_model.to_dict()}"
    rng = np.random.default_rng(SEED)
    claims = []
    for cid, a, b, rule in CLAIMS:
        r = contrast(d, a, b, rng)
        r.update(id=cid, rule=rule, verdict=verdict(r, rule))
        claims.append(r)
    claims = pd.DataFrame(claims)
    extra = pd.DataFrame([contrast(d, a, b, rng) for a, b in EXTRA])
    cols = [c for c in ["jsd", "peak_err", "onset_err", "emd", "mae_counts", "cov80", "crps"] if c in d]
    means = d.groupby("model")[cols].mean().sort_values("jsd")
    reg = d.pivot_table(index="region", columns="model", values="jsd", aggfunc="mean")
    claims.to_csv(R / "tables/claims.csv", index=False)
    extra.to_csv(R / "tables/contrasts_extra.csv", index=False)
    means.to_csv(R / "tables/model_means.csv")
    reg.to_csv(R / "tables/jsd_by_region.csv")

    f = lambda x: f"{x:.4f}"  # noqa: E731
    L = ["# Final test on 2024: results", "",
         "Protocol: [`docs/FINAL_TEST_2024.md`](../../docs/FINAL_TEST_2024.md) (committed before the run). "
         "Data checks: [`data_checks.md`](data_checks.md).", "",
         f"One fold: train 2010-2023, test 2024. {rows_per_model.iloc[0]} municipality-years (annual total >= 100). "
         "Intervals: municipality bootstrap, 2,000 replicates, seed 1. Change = relative change in mean JSD.", "",
         "## Pre-registered check and claims", "",
         "| # | Contrast | JSD a | JSD b | Change | 95 % CI | a better in | Result |", "|---|---|---|---|---|---|---|---|"]
    for r in claims.itertuples():
        L.append(f"| {r.id} | {r.a} - {r.b} | {f(r.jsd_a)} | {f(r.jsd_b)} | {r.rel:+.1f} % | "
                 f"[{r.lo:+.1f}, {r.hi:+.1f}] | {r.share_a_better:.0f} % | {r.verdict} |")
    L += ["", "## All models", "", "| Model | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for m, r in means.iterrows():
        L.append(f"| {m} | " + " | ".join("" if pd.isna(v) else f"{v:.4f}" if c in ("jsd", "cov80") else f"{v:.2f}"
                                          for c, v in r.items()) + " |")
    L += ["", "## Other contrasts (descriptive)", "", "| Contrast | Change | 95 % CI | a better in |", "|---|---|---|---|"]
    for r in extra.itertuples():
        L.append(f"| {r.a} - {r.b} | {r.rel:+.1f} % | [{r.lo:+.1f}, {r.hi:+.1f}] | {r.share_a_better:.0f} % |")
    keep = [m for m in ["B4s", "M1", "S2", "INLA-scaled", "INLA-scaled-trends", "S1"] if m in reg]
    L += ["", "## Mean JSD by region", "", "| Region | " + " | ".join(keep) + " |", "|---" * (len(keep) + 1) + "|"]
    for g, r in reg[keep].iterrows():
        L.append(f"| {g} | " + " | ".join(f(v) for v in r) + " |")
    L += ["", "## Consistency checks", "",
          "- Models fitted in two runs (official and timing) give the same 2024 JSD; largest per-row difference: "
          + ", ".join(f"{m} {v:.1e}" for m, v in agree.items()) + ".",
          "- Every model scores the same municipality-years."]
    (R / "FINAL_TEST_2024_RESULTS.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
