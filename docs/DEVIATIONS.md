# Deviations from the methodology

This file lists every known difference between what the final proposal (and its experiment specification) planned and what the code in this repository does. Each deviation was checked against the project sources (final proposal, experiment specification, base-paper repository tables, the result logs and `docs/RESULTS.md`). Each one has a justification and a note on its effect.

Status: the results are from a **pilot**. Some deviations are "not yet run" items that belong to the full study.

## A. Data

### A1. NASA POWER is the only climate source; ERA5-Land is used only for validation, and the planned ERA5-Land sensitivity re-run was not done

- **Planned.** Weekly climate from NASA POWER, plus an ERA5-Land re-run of the North analysis for 2013–2020, using the Cox et al. data (§6.4 H3 confound, §7.10, risk table).
- **Done.** NASA POWER daily point data at the municipal seat (MERRA-2 grid, about 0.5° × 0.625°) for all models. ERA5-Land (from the Cox et al. Zenodo file) was used only to validate the rainfall. The ERA5-Land sensitivity re-run was not done.
- **Why.** NASA POWER has a keyless API, gives complete daily series for 2009–2024 at every municipal seat, and needs about 2 s per municipality. ERA5-Land directly from the Copernicus Climate Data Store needs an account and gridded downloads that are far larger than the project's budget. The Cox et al. file covers only 2013–2020, and its weeks are ISO-style rather than SINAN epi weeks, so it cannot simply be swapped in.
- **Validation** (`results/data_foundation/source_validation_precip.csv`, 300 municipalities, 2013–2020, per-municipality medians):
  - monthly rainfall totals: r = 0.86;
  - **monthly rainfall anomalies** (the year-to-year part that matters here): r = **0.71** overall, **0.47 in the North**; 0.72–0.74 in the other regions.
  - (Earlier summaries sometimes called this the "annual anomaly" correlation. The computation in `validate_sources.py` removes each municipality's monthly climatology and correlates the monthly anomalies.)
- **Effect.** The North results, and therefore H3, may partly reflect rainfall data quality. The rain-lag extension (Step 8) names this as the most likely reason it failed. An ERA5-Land or CHIRPS re-run is the next test.

### A2. ENSO (Oceanic Niño Index) not yet added

- **Planned.** ONI lagged 3–12 months as an additional climate input, and an "ONI only" ablation (§7.1, §7.4).
- **Done.** Not added.
- **Why.** The pilot fixed the core design first (temperature, rainfall, humidity anomalies). ONI is one value per month for the whole country, so it can only shift timing nationally, and the 14 LOYO years give little power to estimate it. It is the next feature to add.

### A3. Municipality selection used annual totals, including test years

- **Planned and done** as in §7.1, but stated here for transparency: the 300 municipalities are the highest-burden ones among the 700 most populous, ranked by total notifications 2010–2025. The selection never uses weekly timing, so it cannot favour climate directly. Results describe higher-burden municipalities with more than about 48,000 inhabitants.

## B. Reproduction of the base paper (Step 2)

### B1. Reduced NUTS run

- **Published.** 3 repetitions × 4 chains × 1,000 draws, target_accept 0.99, PyMC → NumPyro on a GPU.
- **Done.** 1 repetition (rep01) × interpolation mask × 3 folds, 2 chains × (500 warm-up + 500 draws), target_accept 0.95, NumPyro directly, on CPU.
- **Why.** The base paper's README gives 3–5 GPU-hours for its full set of fits and about 10× that on CPU. Our reduced run took about 20 min per fold on 4 CPUs. The reduced design was fixed in the specification before the run.
- **Effect.** The reproduction criteria were still met: median-based MAE 13.34 vs 13.4 published (within 0.5 %), 80 % coverage 0.948 vs 0.95. Diagnostics: 0 divergences, max R-hat 1.015, min ESS 130. Posterior medians of the hyperparameters were not stored. The NUTS run was restarted once after a session interruption; nothing was lost except time.

### B2. Reproduction criterion applied to the median-based MAE

- The published table (`CV_tab1_final_models_performance.R` in the base repository) reports the MAE and RMSE of the posterior-predictive **median**. The specification said "mean". We compare like with like (median) and report the mean-based numbers alongside (MAE 11.69, RMSE 76.0).

### B3. We extend the base paper's Dirichlet-multinomial, not its final disaggregation model

- **Fact.** For annual-to-monthly disaggregation, the base paper's pipeline selected **scaled INLA** (interpolation MAE 10.7, 80 % coverage 0.87). The Dirichlet-multinomial (DM) was its Bayesian alternative (MAE 13.4 in the repository table, 13.5 in the paper text, 80 % coverage 0.95). Source: `output/tables/CV_tab1_cv_performance_disaggregation.csv` at commit `00ff7ae` and the paper's Results section ("For annual disaggregation, scaled INLA provided the best overall performance").
- **Why the DM.**
  - It has open Python code and runs without R-INLA.
  - Its 80 % intervals are calibrated (0.95), whereas scaled INLA under-covers (0.87).
  - Climate enters the DM naturally as week-varying terms in the logits. Scaled INLA rescales an imputed series to the totals, so there is no clean nested with/without-climate comparison.
  - Its year-constant effect cancels in the softmax (verified, max difference 1.1e-16), which pinpoints exactly where climate can add information.
- **Effect.** Our "base model" is not the base paper's best point-error model. The point error of the DM is close to the historical mean shares (JSD 0.135 vs 0.132), as the specification expected.

## C. Brazil models (Steps 3–5)

### C1. Penalised likelihood (MAP) instead of full Bayesian NUTS

- **Done.** B4, B4s, M1 and the ablations are fitted by penalised maximum likelihood (MAP): L-BFGS-B (SciPy) with JAX gradients, float64, CPU. NUTS is used only for the reproduction (Step 2).
- **Why.** At municipality-week resolution the models have about 3,700 (60 municipalities) to 16,300 (300) parameters, fitted in 14 LOYO folds for 9 model variants. NUTS on CPU is not feasible for this. The proposal (§3.2 point 4, §7.4) planned penalised likelihood from the start and kept full Bayesian fitting optional.
- **Effect.** Predictive intervals use plug-in parameters and ignore parameter uncertainty. They slightly under-cover: M1 80 % coverage 0.78 (60) and 0.80 (300), 95 % coverage 0.92–0.93.

### C2. Fixed penalty scales, not estimated hierarchical scales

- Penalties use fixed scales equal to the base paper's prior scales (g 0.3, region 0.4, municipality 0.15) plus the specified γ, θ, δ and log φ penalties. Deviations are additive; hierarchical scale parameters are not estimated. **Why:** joint-mode (MAP) estimates of hierarchical scale parameters are unreliable in non-centred models (in the MAP reproduction, `tau_country_month` went to about 1.58 against a posterior median of 0.047), and fixed scales keep the folds comparable. The scales were frozen before the 60-municipality pilot and kept for the 300-municipality run (no tuning).

### C3. Small specification details

| Item | Specified | Done | Reason / effect |
|---|---|---|---|
| Centring of the annual-size covariate | fold-training mean | each **municipality's** fold-training mean of log(1 + N) | Centres within municipality so the size terms describe "bigger than usual for this place"; computed in-fold (leakage audit passes) |
| Previous-year-missing indicator × harmonics | region-specific with shrinkage | national coefficients only | Few such rows (2010, and the year after ES 2021) |
| Convergence | gradient norm 1e-6 | L-BFGS stopped on relative function reduction (ftol 1e-14) after 60–860 iterations; max \|gradient\| 0.2 (pilot), 1.2 (300) on objectives of order 10^6–10^7 | Objective flat to machine precision; logs in `results/brazil/convergence_*.csv` |
| M1-mult | converged fit | hit the 2,000-iteration cap in 7/14 weekly and 1/14 monthly folds | Results for this ablation are approximate |
| Fold execution | parallel (joblib) **and** warm starts | sequential with warm starts | The two requirements conflict; runtime was within budget |
| M1-shufTT with 53-week years | – | slot 53 gets 0 (no anomaly) when a 52-week year's features move into a 53-week year | Needed to define the shuffle |
| Climatology years | training years | 2009 + training years | 2009 has no dengue target, so it cannot leak; more data for the climatology |
| ε guard | 1e-6 or 1e-4 | 1e-6 (base paper) | Difference −0.00002 JSD (B4s-eps1e-4 vs B4s) |
| Headroom captured | – | ratio of means, not mean of ratios | Per-row ratios are unstable when B4s is near the floor |

### C4. Denton / Chow–Lin baselines (B2/B3) not run

- **Planned** as optional classical reference rows (§7.4). **Not run.** The specification allowed skipping them ("only if < 1 h of work"). They need a fine-scale indicator series; with the seasonal profile as indicator, proportional Denton reproduces B1 up to smoothing, so it adds little to the nested climate question.

### C5. M2 (boosted softmax) without early stopping, weekly pilot only

- Fixed hyperparameters (depth 4, learning rate 0.05, 300 rounds), **no early stopping**, weekly 60-municipality pilot only. **Why:** the specified "early stopping on the next fold" would use another held-out year inside LOYO, which leaks information across folds; fixed rounds avoid this. M2 is optional and exploratory. Result: worse than B4s (+4.6 % JSD).

### C6. Ablations not run

- Raw climate instead of anomalies, one variable at a time, ONI only, and the previous-year peak-week variant (§7.4) were not run in the pilot. M1-short (H2) and M1-mult were run. **Why:** pilot scope; the core set was fixed first (§15, open question 5).

### C7. Monthly-to-weekly spline (Morello et al.) not included

- The cubic total-preserving spline of Morello et al. (2026) was listed as an optional baseline for a monthly → weekly side task. The study's task is annual → weekly, where a spline has no monthly anchor points, so it was not included.

## D. Validation and statistics (Step 6)

### D1. The 2024 final test has not been run

- `RUN_FINAL_2024 = False` by design. 2024 is never loaded (`data.panel.load_tables` drops it). The final-test code path (`evaluation.cv.final_test_split`) exists and is unit-tested on a dummy year (`checks.sanity.final_path_unit_test`, also in `tests/test_models.py`). **Why:** H1 requires the 2024 sign check after all choices are frozen; the pilot must not look at it. It is planned for the final study.

### D2. Scope of the 300-municipality run

- The 300-municipality run covers the main models only (B0, B1, B1p, B4, B4s, M1 and the test-only shuffle). The train-and-test climate shuffle (3 seeds) was run only on the 60-municipality pilot. **Why:** compute budget; the shuffle had already passed on the pilot. **Effect:** the 300-municipality gain is not separately shuffle-checked.

### D3. H3 uses the municipality bootstrap only

- The proposal requires both intervals for H3. Only the municipality bootstrap was computed for the (South + Southeast) − North contrast. **Why:** the year-block bootstrap of a difference between region groups needs a joint resampling scheme that was not implemented in the pilot.

### D4. Holm correction and the 5 % threshold

- Holm correction across H1–H3 (§6.4) is not applied, because the pilot does not test the hypotheses. The 5 % practical threshold is reported as specified; it still has to be confirmed with the supervisor.

### D5. Stratified "where and when" analyses only partly done

- Reported: by region, by held-out year, without 2020, N-weighted means. Not yet reported: epidemic vs quiet years, size class, 2010–2016 vs 2017–2023, timing surprise, and stratification by NASA POWER / ERA5-Land agreement (§7.10). The optional 3-year block hold-out (§7.6 item 4) was not run.

### D6. Past-only design

- B1p (past years only) is reported for all LOYO years with at least one earlier training year, not only for 2014–2023 targets. The pipeline-consistency check compares it with the smoke test on 3,457 rows (max |ΔJSD| 1.1e-16).

## E. Rain-lag extension (Step 8)

- Pre-registered before any fit (`results/rain_extension/PREREGISTRATION.md`, 2026-10-21T07:32Z). Two variants against one reference, **no multiplicity correction** (stated in the pre-registration). Penalties were kept at M1's scales and not re-tuned for 24 or 36 features. The climate-shuffle check used 1 seed (as pre-registered) and only at 60 municipalities.

## F. Repository refactor (no change to numerical behaviour)

The research code was reorganised into the `dengue_recon` package for this repository. Algorithms were not changed. Changes:

- modules split by methodology step (for example `evaluation.py` → `evaluation/metrics.py`, `evaluation/allocation.py`, `features/seasonal.py`; `run_brazil.py` → `experiments/brazil_loyo.py`, `evaluation/scoring.py`, `checks/*`);
- imports are package-relative; paths come from `config.py` (environment variables), and output folders are created by the entry points instead of on import;
- the rain extension's scorer passes its output folder explicitly instead of temporarily patching a module global;
- the data-foundation reports moved from `reports/` to `results/data_foundation/` (`REPORTS_DIR`);
- `validate_sources.py` and the EDA script were wrapped in `main()` functions;
- one CLI (`scripts/run_pipeline.py`) replaces the separate driver scripts.

`docs/VERIFICATION.md` shows that the refactored code reproduces the saved pilot outputs exactly (identical predictions and metrics).
