"""Checks of the 2023-2025 Google Trends window before the 2024 final test.

1. 2010-2023 inputs unchanged: ``load_trends`` with the new files equals ``load_trends``
   without them (2024 locked).
2. Google's sampling: 2023 within-year shares from the new window vs the original
   2022-2024 window (per geo: correlation and JSD).
3. 2024 completeness: 52 epi weeks per geo, no missing values, number of zero weeks.
4. Plausibility: correlation of each state's 2024 search shares with the state's
   weekly case shares (study municipalities summed), compared with 2019-2023.

Writes ``results/final2024/data_checks.md``. Run with DATA_DIR, TIMING_TRENDS_DIR and
PYTHONPATH=src; 2024 must be unlocked (DENGUE_RUN_FINAL_2024=1) for checks 3-4.
"""
from __future__ import annotations

import importlib
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("results/final2024/data_checks.md")


def jsd(p, q):
    p, q = p / p.sum(), q / q.sum()
    m = (p + q) / 2
    f = lambda a, b: np.sum(np.where(a > 0, a * np.log2(np.where(a > 0, a, 1) / b), 0.0))  # noqa: E731
    return 0.5 * f(p, m) + 0.5 * f(q, m)


def load(trends_dir, final):
    os.environ["TIMING_TRENDS_DIR"] = str(trends_dir)
    os.environ["DENGUE_RUN_FINAL_2024"] = "1" if final else "0"
    import dengue_recon.config as C
    importlib.reload(C)
    from dengue_recon.extensions.timing import signals as SG
    importlib.reload(SG)
    return SG.load_trends()


def main():
    src = Path(os.environ["TIMING_TRENDS_DIR"])
    new = sorted(src.glob("dengue_BR*_2023_2025.csv"))
    lines = ["# 2024 data checks (Google Trends)", "",
             f"New window files: {len(new)} (`dengue_<geo>_2023_2025.csv`).", ""]

    # 1. unchanged 2010-2023
    with tempfile.TemporaryDirectory() as tmp:
        for f in src.glob("dengue_*.csv"):
            if not f.name.endswith("_2023_2025.csv"):
                shutil.copy(f, tmp)
        old = load(tmp, final=False)
    cur = load(src, final=False)
    same = old.reset_index(drop=True).equals(cur.reset_index(drop=True))
    lines += ["## 1. 2010-2023 inputs unchanged", "",
              f"`load_trends()` with and without the new files (2024 locked): identical = **{same}** "
              f"({len(cur)} rows).", ""]

    # 2. 2023 overlap
    rows = []
    for f in new:
        geo = f.name.split("_")[1]
        a = pd.read_csv(src / f"dengue_{geo}_2022_2024.csv")
        b = pd.read_csv(f)
        m = a.merge(b, on="date", suffixes=("_old", "_new"))
        m = m[(m.date >= "2023-01-01") & (m.date <= "2023-12-30")]
        x, y = m.dengue_old.to_numpy(float), m.dengue_new.to_numpy(float)
        ok = x.sum() > 0 and y.sum() > 0
        rows.append(dict(geo=geo, weeks=len(m), corr=np.corrcoef(x, y)[0, 1] if ok and x.std() and y.std() else np.nan,
                         jsd=jsd(x, y) if ok else np.nan))
    ov = pd.DataFrame(rows)
    lines += ["## 2. Google's sampling: 2023 in the old and new window", "",
              "Within-year shape of 2023 from the original `2022_2024` file vs the new `2023_2025` file.", "",
              f"- weekly correlation: median {ov['corr'].median():.3f}, minimum {ov['corr'].min():.3f} "
              f"({ov.loc[ov['corr'].idxmin(), 'geo']})",
              f"- JSD of the two 2023 shapes: median {ov.jsd.median():.4f}, maximum {ov.jsd.max():.4f} "
              f"({ov.loc[ov.jsd.idxmax(), 'geo']})", "",
              "| geo | weeks | corr | JSD |", "|---|---|---|---|"]
    lines += [f"| {r.geo} | {r.weeks} | {r.corr:.3f} | {r.jsd:.4f} |" for r in ov.itertuples()]

    # 3. 2024 completeness
    tr = load(src, final=True)
    t24 = tr[tr.epi_year == 2024]
    comp = t24.groupby("geo").agg(weeks=("epi_week", "nunique"), win=("win", "first"),
                                  na=("value", lambda s: s.isna().sum()), zeros=("value", lambda s: (s == 0).sum()))
    lines += ["", "## 3. 2024 completeness", "",
              f"- geos with 2024 data: {len(comp)}; all from the 2023 window: {bool((comp.win == 2023).all())}",
              f"- weeks per geo: min {comp.weeks.min()}, max {comp.weeks.max()} (2024 has 52 epi weeks); "
              f"missing values: {int(comp.na.sum())}",
              f"- geos with more than 10 zero weeks: {', '.join(comp.index[comp.zeros > 10]) or 'none'}", ""]

    # 4. plausibility vs cases
    import dengue_recon.config as C
    d = pd.read_parquet(C.DATA_DIR / "dengue_weekly.parquet", columns=["uf", "epi_year", "epi_week", "cases"])
    d = d[d.epi_year.between(2019, 2024)].groupby(["uf", "epi_year", "epi_week"]).cases.sum().reset_index()
    d["geo"] = "BR-" + d.uf
    m = d.merge(tr, on=["geo", "epi_year", "epi_week"])
    cor = m.groupby(["geo", "epi_year"]).apply(
        lambda z: np.corrcoef(z.cases, z.value)[0, 1] if z.value.std() > 0 and z.cases.std() > 0 else np.nan,
        include_groups=False).unstack()
    lines += ["## 4. Do 2024 searches follow 2024 cases?", "",
              "Per state, weekly correlation between Google Trends and weekly cases of the study municipalities.", "",
              "| year | median corr | states |", "|---|---|---|"]
    lines += [f"| {y} | {cor[y].median():.3f} | {cor[y].notna().sum()} |" for y in cor.columns]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:12] + lines[-10:]))


if __name__ == "__main__":
    main()
