"""Official report statistics on small synthetic metrics tables (no downloaded data)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from dengue_recon import config as C
from dengue_recon.evaluation.bootstrap import cluster_bootstrap, h3_contrast
from dengue_recon.reporting import official_report as R

REG = ["North", "Northeast", "Center-West", "Southeast", "South"]


def synth(models_shift, years=(2012, 2013, 2014, 2015), n_geo=20, design="A", ks=None, seed=0,
          region_shift=None):
    """Long metrics table. ``models_shift`` maps model -> additive JSD shift vs a base value."""
    rng = np.random.default_rng(seed)
    rows = []
    for yi, y in enumerate(years):
        for k in (ks or [yi + 2]):
            for g in range(n_geo):
                base = 0.1 + 0.02 * rng.standard_normal()
                reg = REG[g % 5]
                for m, s in models_shift.items():
                    extra = (region_shift or {}).get((m, reg), 0.0)
                    rows.append(dict(design=design, fold_id=f"{design}_T{y}_k{k}", geocode=1000 + g, year=y,
                                     region=reg, N=200 + 10 * g, model=m, jsd=base + s + extra + 0.001 * rng.standard_normal(),
                                     peak_err=5.0 + s * 10, jsd_floor=0.01, k=k, climate="power"))
    return pd.DataFrame(rows)


def test_boot_replicates_matches_cluster_bootstrap():
    rng = np.random.default_rng(1)
    d, cl, w = rng.normal(size=200), rng.integers(0, 15, 200), rng.uniform(1, 5, 200)
    est, b = R.boot_replicates(d, cl, 300, 7, w)
    m, lo, hi = cluster_bootstrap(d, cl, 300, 7, w)
    assert est == pytest.approx(m)
    assert R.ci(b) == pytest.approx((lo, hi))


def test_holm_and_boot_p():
    adj = R.holm([0.01, 0.04, 0.03])
    assert adj == pytest.approx([0.03, 0.06, 0.06])
    assert R.boot_p(np.full(999, -1.0)) == pytest.approx(2 / 1000)
    assert R.boot_p(np.linspace(-1, 1, 1001)) == pytest.approx(1.0)


def test_contrast_detects_shift_and_counts_folds():
    df = synth({"B4s": 0.0, "M1": -0.01})
    r = R.contrast(df, "M1", "B4s", n_boot=200)
    assert r["mean_diff"] == pytest.approx(-0.01, abs=1e-3)
    assert r["ci_muni_hi"] < 0 and r["ci_year_hi"] < 0
    assert r["n_folds"] == 4 and r["n_folds_a_better"] == 4
    assert r["headroom_captured"] > 0
    rw = R.contrast(df, "M1", "B4s", weighted=True, n_boot=200)
    assert rw["metric"] == "jsd_nweighted"


def test_hypothesis_decisions():
    good = synth({"B4s": 0.0, "M1": -0.01, "M1-short": 0.0},
                 region_shift={("M1", "North"): 0.009})
    h = R.hypothesis_table(good, n_boot=200).set_index("hypothesis")
    assert h.loc["H1", "decision"].startswith("SUPPORTED") and "2024" in h.loc["H1", "decision"]
    assert h.loc["H2", "decision"].startswith("SUPPORTED")
    assert h.loc["H3", "estimate"] > 0
    assert (h.p_holm >= h.p_both - 1e-12).all()
    null = synth({"B4s": 0.0, "M1": -0.002, "M1-short": 0.0})          # -2 %: below the 5 % threshold
    h0 = R.hypothesis_table(null, n_boot=200).set_index("hypothesis")
    assert h0.loc["H1", "decision"] == "NOT SUPPORTED"


def test_group_contrast_matches_existing_h3_estimate():
    df = synth({"B4s": 0.0, "M1": -0.005}, region_shift={("M1", "North"): 0.004})
    g = R.group_contrast(df, n_boot=100)
    est, _, _ = h3_contrast(df)
    assert g["estimate"] == pytest.approx(est)
    assert g["estimate"] == pytest.approx(0.004, abs=1e-3)


def test_lc_slope_exact_and_composition_guard():
    rows = []
    for y in (2019, 2021):
        for k in range(2, 8):
            for g in range(5):
                rows.append(dict(geocode=g, year=y, k=k, v=0.2 - 0.03 * np.log(k)))
    rows = pd.DataFrame(rows)
    s = R.lc_slope(rows, "v", list(range(2, 8)), n_boot=50)
    assert s["slope_per_log_k"] == pytest.approx(-0.03)
    assert s["ci_lo"] == pytest.approx(-0.03) and s["ci_hi"] == pytest.approx(-0.03)
    with pytest.raises(AssertionError):
        R.lc_slope(rows[~((rows.year == 2019) & (rows.k == 7))], "v", list(range(2, 8)), n_boot=10)


def test_learning_curve_tables_flags_k_with_fewer_test_years():
    a = synth({"B1": 0.0, "B4s": 0.0, "M1": -0.002, "M2": 0.01}, years=(2019,), design="B", ks=list(range(2, 10)), n_geo=10)
    b = synth({"B1": 0.0, "B4s": 0.0, "M1": -0.002, "M2": 0.01}, years=(2021,), design="B", ks=list(range(2, 12)), n_geo=10, seed=1)
    by_year, av, slopes, gap = R.learning_curve_tables(pd.concat([a, b]), n_boot=50)
    assert av[av.k <= 9].main.all() and not av[av.k >= 10].main.any()
    assert set(av[av.k >= 10].n_test_years) == {1}
    assert (slopes.k_max[slopes.scope.str.startswith("main")] == 9).all()
    assert gap[gap.k == 10].main.iloc[0] == False  # noqa: E712


def test_locked_year_guard():
    with pytest.raises(AssertionError):
        R.check_no_locked_years(pd.DataFrame({"year": [2023, C.max_loaded_year() + 1]}))


def test_epidemic_labels_tercile():
    df = synth({"B4s": 0.0, "M1": 0.0}, years=(2013,))
    tot = pd.DataFrame([dict(geocode=1000 + g, year=y, N=100 if y < 2013 else 100 * (1 + g), excluded=False)
                        for g in range(20) for y in (2010, 2011, 2012, 2013)])
    lab = R.epidemic_labels(df, [dict(fold_id="A_T2013_k2", train_years=[2011, 2012])], tot)
    assert (lab.train_median == 100).all()
    assert (lab.stratum == "epidemic").sum() == pytest.approx(len(lab) / 3, abs=1)
    t = R.epidemic_table(df, "x", lab, n_boot=50)
    assert set(t.stratum) == {"epidemic", "quiet"}


def test_coefficient_table_from_fold_json(tmp_path):
    (tmp_path / "run" / "folds").mkdir(parents=True)
    F = len(R.feature_names())
    for i in range(3):
        fit = dict(theta=[0.1 * (i + 1)] * F, delta=[[0.0] * F for _ in C.REGIONS])
        (tmp_path / "run" / "folds" / f"A_T{2012 + i}.json").write_text(json.dumps(dict(fits={"M1": fit})))
    t = R.coefficient_table("run", models=("M1",), base=tmp_path)
    assert len(t) == F
    assert t.theta_mean.iloc[0] == pytest.approx(0.2)
    assert t.folds_positive.iloc[0] == 3


def test_main_end_to_end_on_synthetic_primary_run(tmp_path):
    df = synth({m: s for m, s in zip(["B0", "B1", "B4", "B4s", "M1", "M1-noONI", "M1-short", "M1-shufTT",
                                      "M1-shufTest", "M2"], [0.05, 0.01, 0.01, 0.0, -0.002, -0.003, 0.0, 0.0, 0.004, 0.006])})
    d = tmp_path / R.PRIMARY
    d.mkdir()
    df.to_csv(d / f"metrics_detail_{R.PRIMARY}.csv", index=False)
    out = R.main(base=tmp_path, data_dir=tmp_path, n_boot=50, figures=False)
    assert set(out["hypotheses"].hypothesis) == {"H1", "H2", "H3"}
    assert (tmp_path / "tables" / "paired_contrasts.csv").exists()
    assert (tmp_path / "tables" / "gain_by_region.csv").exists()
