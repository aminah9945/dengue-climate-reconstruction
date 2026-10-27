# results/

Small outputs of the pilot runs: CSV tables, JSON fit summaries, logs and PNG figures. The write-up of the numbers is in [`docs/RESULTS.md`](../docs/RESULTS.md); deviations are in [`docs/DEVIATIONS.md`](../docs/DEVIATIONS.md).

All results are **preliminary**: 60 municipalities, LOYO 2010–2023, pilot hyperparameters fixed in advance, plus a 300-municipality confirmation run with the same frozen settings. 2024 was never loaded.

| Path | Content | Produced by (step) |
|---|---|---|
| `data_foundation/` | EDA (`eda.md`), smoke-test baselines (`baseline_smoke_test_*.csv`), JSD noise floor, peak-week variability, source validation against Cox et al., figures | `dengue_recon.data.eda`, `data.validate_sources` (1) |
| `repro_lim/table_repro.csv`, `repro_lim/README.md` | Reproduction of Lim et al. (2026): per fold and pooled, with the published row. Verdict: reproduced | `base_paper.reproduce` (2) |
| `repro_lim/cancellation_check.csv` | Max share difference between held-out years of the same country (about 1e-16) | (2) |
| `repro_lim/fit_diagnostics.csv`, `repro_lim/folds/*.json` | NUTS diagnostics (R-hat, ESS, divergences), MAP convergence, raw per-fold outputs | (2) |
| `subset60.csv` | The 60 pilot municipalities (12 per region, including Belo Horizonte) | `experiments.brazil_loyo` |
| `brazil/metrics_detail_{weekly_60,monthly_60}.csv` | One row per model × municipality-year with every metric, region, year, N and the noise floor | `evaluation.scoring` (6) |
| `brazil/table_main_{weekly,monthly}[_300].csv` | Mean, median and N-weighted values per model (all years and without 2020) | `reporting.tables` (9) |
| `brazil/table_abs_means_ci.csv` | Mean JSD and peak error with municipality-bootstrap 95 % CIs | (9) |
| `brazil/table_paired_{weekly,monthly}[_300].csv` | Paired differences with municipality and year-block bootstrap CIs, years with a gain, headroom | (6, 9) |
| `brazil/table_by_region_weekly[_300].csv` | M1 − B4s and M1 − B1 by region, H3 contrast | (9) |
| `brazil/table_gain_by_year_weekly[_300].csv` | Mean gain (B4s − M1) by held-out year | (9) |
| `brazil/sanity_checks.csv` | All sanity checks (all pass) | `checks.sanity` (7) |
| `brazil/fitted_params_*.json`, `brazil/convergence_*.csv` | φ, γ, θ, δ and convergence per model and fold | (5) |
| `brazil/leakage_audit_*.csv`, `brazil/checks_*.json` | Per-fold leakage audit; zero-climate check; climate NaN counts | (7) |
| `runtime_log.csv`, `logs/` | Wall time per fit; console logs of the pilot runs | – |
| `figures/` | Main figures (reproduction, baselines vs climate, JSD by model, gain by region and year, example reconstructions, shuffle test, lag coefficients) | `reporting.figures` (9) |
| `weather_validation/` | NASA POWER vs CHIRPS rainfall (weekly, monthly, raw and anomalies, by region) and both vs Cox et al. ERA5-Land 2013–2020: `summary.csv`, `summary.md`, `per_municipality.csv`, PNG figures; write-up in [`docs/WEATHER_VALIDATION.md`](../docs/WEATHER_VALIDATION.md) | `dengue_recon.data.validate_weather` (1) |
| `rain_extension/` | Pre-registered follow-up: `PREREGISTRATION.md`, `RESULTS_RAIN.md`, tables, figures, audits, per-row metrics | `extensions.rain_lags`, `reporting.rain_report` (8, 9) |

## Files omitted from the repository

Large or binary intermediate files are not committed (every committed file is below 2 MB). Regenerate them with the pipeline (paths assume the processed data are in `DATA_DIR`):

| Omitted file | Size | Regenerate with |
|---|---|---|
| `brazil/preds_weekly_60.npz` | 4.4 MB | `python scripts/run_pipeline.py --step brazil --subset 60` (about 22 min on 4 CPUs) |
| `brazil/preds_monthly_60.npz` | 0.9 MB | `python scripts/run_pipeline.py --step brazil --res monthly --subset 60` (about 18 min) |
| `brazil/preds_weekly_300.npz` | 9.0 MB | `python scripts/run_pipeline.py --step brazil --subset 300` (about 41 min) |
| `brazil/metrics_detail_weekly_300.csv` | 4.0 MB | same run as above (written by the scoring at the end) |
| `rain_extension/preds_weekly_60.npz`, `preds_weekly_60_shuf.npz` | 1.1 MB, 0.4 MB | `python scripts/run_pipeline.py --step extension --subset 60` (about 8 min) |
| `rain_extension/preds_weekly_300.npz` | 3.2 MB | `python scripts/run_pipeline.py --step extension --subset 300` (about 29 min) |

M2 (optional) adds `pred__M2` to `brazil/preds_weekly_60.npz`: `python scripts/run_pipeline.py --step m2`.

`python scripts/run_pipeline.py --step report` regenerates the tables and figures from whatever is present. Without the omitted files it rebuilds everything for the 60-municipality runs (verified identical, see `docs/VERIFICATION.md`) and skips the 300-municipality tables and the example-reconstruction figure; the committed versions of those come from the original pilot run. Note that the `report` step overwrites the tables in `RESULTS_DIR`; point `RESULTS_DIR` to a copy if you want to compare.

A `--quick` run (`--step brazil --subset 60 --quick`, about 5 min) writes its outputs under the tag `weekly_60_quick`, so it never overwrites the saved pilot files.

## File names in older documents

`repro_lim/README.md`, `rain_extension/RESULTS_RAIN.md` and `rain_extension/PREREGISTRATION.md` are kept exactly as written during the pilot (the pre-registration must not be edited). They refer to the original script names, which map to the package as follows:

| Original script | Package module |
|---|---|
| `src/data_download.py`, `build_dataset.py`, `epiweeks.py`, `validate_sources.py`, `eda_baselines.py` | `dengue_recon.data.download`, `.build_dataset`, `.epiweeks`, `.validate_sources`, `.eda` |
| `src/brazil_data.py` | `dengue_recon.data.panel` |
| `src/lim_repro.py`, `run_part_a.py` | `dengue_recon.base_paper.lim_dm`, `.reproduce` |
| `src/features.py` | `dengue_recon.features.seasonal`, `.size_terms`, `.climate`; `checks.leakage` |
| `src/evaluation.py`, `metrics_ext.py` | `dengue_recon.evaluation.metrics`, `.allocation`, `.bootstrap`; `features.seasonal` |
| `src/dm_brazil.py`, `run_m2.py` | `dengue_recon.models.dirichlet_multinomial`, `.xgb_softmax` |
| `src/run_brazil.py` | `dengue_recon.experiments.brazil_loyo`, `evaluation.scoring`, `evaluation.cv`, `checks.shuffle`, `checks.zero_climate` |
| `src/make_report.py`, `rain_report.py` | `dengue_recon.reporting.tables`, `.figures`, `.report`, `.rain_report`; `checks.sanity` |
| `src/rain_ext.py` | `dengue_recon.extensions.rain_lags` |
| `reports/` | `results/data_foundation/` |
