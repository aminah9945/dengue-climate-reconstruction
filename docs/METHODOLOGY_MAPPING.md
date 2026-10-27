# Methodology-to-code mapping

Each step of the methodology in the final research proposal maps to one sub-package of `src/dengue_recon/`. The numbered steps exist only in this mapping; the folder names describe what the code does. Section numbers (§) refer to the final proposal, *Climate-Informed Reconstruction of Weekly Dengue Curves from Annual Totals* (October 2026). Spec Part A–D refers to the experiment specification that accompanied it.

Run any step with `python scripts/run_pipeline.py --step <step>` (column "CLI step").

## Summary table

| Step | Methodology | File | Main functions | CLI step |
|---|---|---|---|---|
| 1 | Data acquisition: InfoDengue weekly notifications, NASA POWER daily climate, IBGE population, municipal-seat coordinates; municipality selection (§7.1) | `src/dengue_recon/data/download.py` | `download_meta`, `download_infodengue`, `select_municipalities`, `download_power`, `download_cox` | `data` |
| 1 | Epidemiological-week calendar, 53-week years (§7.2) | `src/dengue_recon/data/epiweeks.py` | `epiyear_start`, `weeks_in_epiyear`, `calendar`, `date_to_epiweek` | `data` |
| 1 | Tidy weekly tables, weekly climate aggregation, quality flags (ES 2021, 2020, 2025) (§7.1) | `src/dengue_recon/data/build_dataset.py` | `load_infodengue`, `load_power`, `main` | `data` |
| 1 | Source validation against Cox et al. (annual totals, rainfall anomalies) (§8.1, §8.3) | `src/dengue_recon/data/validate_sources.py` | `main` | `data --with-cox` |
| 1 | EDA, peak-week variability, smoke-test baselines, JSD noise floor (§8.2–8.3) | `src/dengue_recon/data/eda.py` | `main`, `circ_stats`, `com_week` | `data` |
| 1 | Model panel: 53-slot weekly grid with slot 53 masked, monthly variant by the Wednesday rule, 60-municipality pilot subset, continuous climate matrix (§7.2, spec B.1–B.2) | `src/dengue_recon/data/panel.py` | `load_tables`, `select_subset60`, `build_panel`, `climate_matrix` | `brazil` |
| 2 | Reduced reproduction of the base model (Lim et al. 2026, DM_v3_pooled): data arrays, NumPyro NUTS, MAP, posterior-predictive draws, incidence scoring, cancellation check (§3.2, §8.4, spec Part A) | `src/dengue_recon/base_paper/lim_dm.py` | `read_mask`, `build_arrays`, `numpyro_model`, `fit_nuts`, `fit_map`, `dm_ppc_from_alpha`, `score_incidence`, `shape_scores`, `cancellation`, `run_fold` | `reproduce` |
| 2 | Reproduction runner and table vs published numbers | `src/dengue_recon/base_paper/reproduce.py` | `run`, `table`, `main` | `reproduce` |
| 2 | Fetch the base paper's CV mask at the pinned commit (data only, code never executed) | `src/dengue_recon/base_paper/fetch.py` | `fetch_lim_mask` | `reproduce` |
| 3 | Baselines B0 (uniform), B1 (seasonal profile, LOYO), B1p (past years only) (§7.4) | `src/dengue_recon/baselines/simple.py` | `uniform_shares`, `seasonal_shares`, `past_only_shares` | `baselines` |
| 4 | Seasonal offset h_m = log mean training share (53-slot grid, regional fallback) (§7.2) | `src/dengue_recon/features/seasonal.py` | `seasonal_profile`, `adapt_length`, `offsets`, `past_only_profile` | `brazil` |
| 4 | Annual-size terms: centred log total and previous-year total × week harmonics (B4s) (§7.4) | `src/dengue_recon/features/size_terms.py` | `harmonics`, `size_covariates` | `brazil` |
| 4 | Climate anomalies (in-fold harmonic climatology, standardised) and lag blocks 0–3 … 16–20 weeks (§7.2, Eq. 3) | `src/dengue_recon/features/climate.py` | `climatology_design`, `climate_anomalies`, `lag_block_features`, `to_slots`, `climate_features` | `brazil` |
| 4 | ENSO: weekly ONI on the epi-week calendar, standardised with the fold's climatology years (2009 + training years), 5 lag-block means (0–3 … 16–20 weeks), the same for every municipality; with the 15 local features → 20 M1 features | `src/dengue_recon/features/climate.py`, `src/dengue_recon/data/panel.py` | `oni_standardise`, `oni_block_features`, `fold_climate_features`, `short_mask`; `load_oni_weekly`, `oni_vector` | `official` |
| 4 | Climate source switch: NASA POWER (primary, Amendment 1) or CHIRPS rain + POWER temperature/humidity (robustness) | `src/dengue_recon/data/panel.py`, `config.py` | `load_tables(climate=...)`, `CLIMATE_FILES` | `official --climate` |
| 5 | Dirichlet-multinomial model ladder B4 → B4s → M1 (Eq. 2), penalised likelihood, L-BFGS with JAX gradients, warm starts (§7.4) | `src/dengue_recon/models/dirichlet_multinomial.py` | `DMModel.fit`, `DMModel.predict`, `DMModel.with_params`, `param_template` | `brazil` |
| 5 | Ablations M1-short, M1-mult, B4s-eps1e-4, shuffled climate (§7.4) | `src/dengue_recon/models/dirichlet_multinomial.py` | `variant_inputs` | `brazil` |
| 5 | Uncertainty: 500 draws Dirichlet(φπ) → Multinomial(N), exact sums (§7.5) | `src/dengue_recon/models/dirichlet_multinomial.py` | `draw_counts` | `brazil` |
| 5 | Optional M2: boosted trees with a softmax over weeks (§7.4) | `src/dengue_recon/models/xgb_softmax.py` | `group_softmax`, `design`, `main` | `m2` |
| 5 | Official M2: XGBoost softmax over weeks (features of M1 + season + size); grid max_depth {3, 5} × learning rate {0.05, 0.1}, up to 2,000 rounds, early stopping after 50 rounds on the inner validation year (= last training year), refit on all training years for the chosen number of rounds | `src/dengue_recon/models/xgb_softmax.py` | `fit_predict_official`, `group_cross_entropy` | `official` |
| 6 | Validation design: LOYO 2010–2023, training/evaluation exclusions, 2024 lock and final-test path (§7.6) | `src/dengue_recon/evaluation/cv.py` | `row_masks`, `loyo_folds`, `final_test_split`, `assert_final_year_locked` | `brazil` |
| 6 | Official validation designs: **A** expanding window (train 2010 … T−1, test T = 2012–2023; primary), **B** learning curve (test 2019, 2021, 2023; train on the k = 2 … T−2010 years immediately before T), **C** LOYO 2010–2023 (secondary); fold panel without later years; 2024 assertions | `src/dengue_recon/evaluation/cv.py` | `expanding_window_folds`, `learning_curve_folds`, `loyo_fold_specs`, `design_folds`, `climate_years`, `panel_years`, `fold_train_mask`; `config.assert_year_allowed`, `config.max_loaded_year` | `official` |
| 6 | Exact-total integer allocation (largest remainder) (§2, §7.3) | `src/dengue_recon/evaluation/allocation.py` | `allocate_integers` | `brazil` |
| 6 | Metrics: JSD (primary), peak-week error, EMD, week-of-50 %, MAE, coverage, CRPS, noise floor (§7.7) | `src/dengue_recon/evaluation/metrics.py` | `jsd`, `peak_week_error`, `emd_weeks`, `week50_error`, `mae_shares`, `crps_samples`, `interval_coverage`, `jsd_floor` | `brazil` |
| 6 | Per-row scoring of saved predictions | `src/dengue_recon/evaluation/scoring.py` | `score` | `brazil` |
| 6 | Statistical comparison: paired differences, municipality and year-block bootstraps, years with a gain, headroom, H3 contrast (§7.9) | `src/dengue_recon/evaluation/bootstrap.py` | `cluster_bootstrap`, `paired`, `h3_contrast` | `report` |
| 3–7 | Official driver (designs A/B/C): per fold builds in-fold features, B0, B1, the DM ladder, M2, checks; per-fold checkpoints (resume), scoring with per-fold seeds | `src/dengue_recon/experiments/official.py` | `main`, `run_fold`, `assemble`, `run_tag` | `official` |
| 3–7 | LOYO driver: per fold builds features, baselines, fits the ladder, runs per-fold checks, saves and scores | `src/dengue_recon/experiments/brazil_loyo.py` | `main`, `log_runtime` | `brazil` |
| 7 | Check 1, exact sums (§7.8) | `src/dengue_recon/checks/sums.py` (asserted in `evaluation/scoring.py`, `models/dirichlet_multinomial.py`) | `allocation_sum_ok`, `draws_sum_ok` | `brazil` |
| 7 | Check 3, zero-climate equivalence (§7.8) | `src/dengue_recon/checks/zero_climate.py` | `zero_climate_maxdiff` | `brazil` |
| 7 | Checks 4–5, train-and-test and test-only climate shuffles (§7.8) | `src/dengue_recon/checks/shuffle.py` | `derangement`, `train_test_permutations`, `test_only_permutation`, `shuffle_years`, `shuffle_test_year` | `brazil` |
| 7 | Check 6, leakage audit (§7.2, §7.8) | `src/dengue_recon/checks/leakage.py` | `leakage_audit` | `brazil` |
| 7 | Official per-fold checks: leakage audit (offsets, centring, climatology, ONI scaling; no training year ≥ T and no later year in the panel for A/B), zero-climate equivalence of the 20-feature M1, shuffles restricted to the fold's years | `src/dengue_recon/checks/leakage.py`, `checks/zero_climate.py`, `checks/shuffle.py` | `official_leakage_audit`, `embed_params`, `zero_climate_maxdiff_embedded`, `fold_year_permutations`, `fold_test_permutation` | `official` |
| 7 | Check 2 (uniform worst), check 7 (pipeline consistency), final-test unit test, sanity table | `src/dengue_recon/checks/sanity.py` | `sanity_table`, `pipeline_consistency`, `final_path_unit_test` | `report` |
| 8 | Pre-registered extension: dual rain lags M1-R, asymmetric climate M1-RTH, 21–26-week block, new-feature leakage audit | `src/dengue_recon/extensions/rain_lags.py` | `feature_spec`, `split_features_continuous`, `ext_climate_features`, `ext_leakage_audit`, `main` | `extension` |
| 9 | Where and when climate helps: by region, year, without 2020, N-weighted (§7.10); result tables | `src/dengue_recon/reporting/tables.py` | `main_table`, `paired_table`, `region_table`, `gain_by_year`, `abs_means_ci`, `convergence_table` | `report` |
| 9 | Figures | `src/dengue_recon/reporting/figures.py` | `fig_repro`, `fig_main_ci`, `fig_jsd_by_model`, `fig_gain_region_year`, `fig_shuffle`, `fig_lag_coefficients`, `fig_examples` | `report` |
| 9 | Report orchestration; rain-extension tables and figures | `src/dengue_recon/reporting/report.py`, `src/dengue_recon/reporting/rain_report.py` | `main` | `report` |

Configuration that holds the frozen pilot settings (seeds, penalty scales, lag blocks, thresholds, `RUN_FINAL_2024 = False`) lives in `src/dengue_recon/config.py`.

## Validation designs (official experiment)

| Design | Folds | Training years of fold T | Role | Function |
|---|---|---|---|---|
| A | T = 2012 … 2023 (12; T = 2024 only after the go-ahead) | 2010 … T−1 | primary | `evaluation.cv.expanding_window_folds` |
| B | T ∈ {2019, 2021, 2023} × k = 2 … T−2010 (8 + 10 + 12 = 30) | T−k … T−1 | learning curve (H4) | `evaluation.cv.learning_curve_folds` |
| C | T = 2010 … 2023 (14) | all other years 2010–2023 | secondary, pilot comparison | `evaluation.cv.loyo_fold_specs` |

In every design, everything learned (offsets h_m and the B1 profile, size centring, climatology, ONI mean/SD, M2 settings) uses only the fold's training years, plus 2009 climate (no dengue target) for the climatology and ONI scaling. The previous-year total of the test year (known historically) is used as in the pilot. Run with `python scripts/run_pipeline.py --step official --design {A,B,C} --climate {power,chirps} --subset {60,300} [--exclude-year 2020]`.

## Models and where they are built

| ID | Model | Built by |
|---|---|---|
| B0 | Uniform split | `baselines.simple.uniform_shares` |
| B1 | Seasonal profile exp(h_m), LOYO | `features.seasonal.offsets` → `baselines.simple.seasonal_shares` |
| B1p | Seasonal profile, past years only | `features.seasonal.past_only_profile` → `baselines.simple.past_only_shares` |
| B4 | DM, base-paper structure (offset + national/regional/municipal week deviations, random-walk smoothness, φ per region) | `models.dirichlet_multinomial.DMModel` with Z = 0, X = 0 |
| B4s | B4 + annual-size terms (primary comparator) | `DMModel` with `features.size_terms.size_covariates`, X = 0 |
| M1 (pilot) | B4s + 15 lag-block climate anomalies (the extension) | `DMModel` with `features.climate.climate_features` |
| M1 (official) | B4s + 15 local lag-block anomalies + 5 ONI lag-block features = 20 | `experiments.official` with `features.climate.fold_climate_features(..., oni=...)` |
| M1-noONI (official) | the pilot's 15-feature M1 | `experiments.official`, 15-feature `DMModel` |
| M1-short | lags 0–3 weeks only (official: block 0–3 of temperature, rain, humidity and ONI) | `climate_features(..., short=True)`; official: `short_mask` |
| M1-mult | multinomial likelihood | `DMModel(..., likelihood="mult")` |
| M1-shufTT-s{0,1,2} | climate shuffled between years, refitted (official: one seed, climate + ONI, within the fold's years) | `checks.shuffle.shuffle_years`; official: `fold_year_permutations` |
| M1-shufTest | trained M1 given another year's climate | `checks.shuffle.shuffle_test_year` |
| M2 | boosted softmax over weeks (pilot: fixed 300 rounds; official: grid + early stopping on the last training year) | `models.xgb_softmax.main`; official: `fit_predict_official` |
| M1-R, M1-RTH | dual rain lags; asymmetric temperature and humidity | `extensions.rain_lags` |

## Tests that cover the mapping

| Test file | What it checks |
|---|---|
| `tests/test_epiweeks.py` | 53-week years (2014, 2020, 2025), week 1 contains 4 January, calendar contiguity, date mapping |
| `tests/test_allocation.py` | largest-remainder allocation sums exactly and never misses N·p by a full case; every DM draw sums to N |
| `tests/test_metrics.py` | JSD identity, symmetry, bounds, disjoint = 1, zeros; peak/EMD/week-50; CRPS; bootstrap |
| `tests/test_leakage.py` | climatology, offsets and centring constants unchanged when the held-out year is blanked or perturbed; full leakage audit on every synthetic fold |
| `tests/test_official.py` | designs A and B never train on the test year or later years; learning-curve windows are exactly the k preceding years; 2024 blocked while locked (fold builders and loaders); climatology, ONI scaling, offsets and centring use training years only; official leakage audit passes in A, B and C; ONI epi-week alignment and lag-block means; 20-feature M1 with zero climate coefficients equals B4s; shuffles stay within the fold's years; M2 early-stopping metric and prediction |
| `tests/test_models.py` | year-effect cancellation in the softmax; zero-climate M1 equals B4s; shuffles are derangements; 2024 lock and final-test path |
