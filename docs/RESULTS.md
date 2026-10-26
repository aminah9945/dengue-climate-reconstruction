# Pilot experiment results: reproducing the base model, then adding climate

**Status:** these are preliminary pilot results: 60 municipalities, leave-one-year-out (LOYO) validation over 2010–2023, and pilot hyperparameters fixed in advance. A confirmatory run on 300 municipalities with the same frozen settings is also reported.

- 2024 was **never loaded** (`RUN_FINAL_2024 = False`).
- No gain reported here is confirmed. The pilot answers one question: does the pipeline work, and is there a signal worth the full study?

All numbers below come from the CSV files in [`results/`](../results/). Every figure is in [`results/figures/`](../results/figures/). Paths below are relative to `results/`. This file is the pilot write-up, copied with module names updated to the package layout; the rain-lag follow-up is condensed in Section 8.

---

## 1. What was run

| Part | What | Settings | Wall time |
|---|---|---|---|
| A | Re-implementation of the Lim et al. (2026) Dirichlet-multinomial (DM_v3_pooled) on their own CV data (interpolation mask, rep01, 3 folds, 3,648 test country-months) | NUTS (NumPyro): 2 chains × (500 warm-up + 500 draws), target_accept 0.95. MAP: L-BFGS, 3 restarts | NUTS about 20 min per fold, 3 folds in parallel. MAP 23 s in total |
| B/C monthly | B0, B1, B1p, B4, B4s, M1, M1-short, M1-mult, B4s-eps1e-4, M1-shufTT (3 seeds), M1-shufTest. 14 LOYO folds, 60 municipalities | Penalised DM, L-BFGS, JAX float64, warm starts | 18 min |
| B/C weekly | Same set of models, plus M2 (boosted softmax, XGBoost) | 53-slot grid with slot 53 masked in 52-week years | 22 min, plus 1 min for M2 |
| C weekly, 300 municipalities | B0, B1, B1p, B4, B4s, M1, M1-shufTest. Settings frozen from the pilot | The same code | 41 min |

Fits per model and fold are logged in `runtime_log.csv`. The total compute was about 1 h 45 min of wall-clock time on 4 CPUs.

**Data:**
- **Pilot:** 60 municipalities, 12 per region, including Belo Horizonte (`subset60.csv`). All 840 municipality-years are evaluation rows with N ≥ 100.
- **Full run:** 300 municipalities, with 3,723 evaluation rows and 4,058 training rows.

---

## 2. Part A: reproduction of the base paper

`repro_lim/table_repro.csv`, `figures/fig_repro_lim.png`

| | MAE* | RMSE* | 80 % coverage | CRPS | JSD (monthly shares) | Peak-month error |
|---|---|---|---|---|---|---|
| **Published DM** (3 reps, 4 × 1,000 draws) | 13.4 | 78.5 | 0.95 | 9.6 | – | – |
| **Ours, NUTS** (rep01) | **13.34** | **78.1** | **0.948** | **9.53** | 0.135 | 1.81 |
| Ours, MAP | 12.18 | 68.3 | 0.958 | 8.66 | 0.135 | 1.66 |
| Historical mean shares (point only) | – | – | – | – | 0.132 | 1.63 |
| Uniform (point only) | – | – | – | – | 0.187 | 3.14 |

\* MAE and RMSE are in incidence per 100,000. They are computed from the posterior-predictive **median**, which is the definition used in the repository's published table. On the predictive mean, our values are 11.69 and 76.0. On the same rows, the historical mean shares give a mean-based MAE of 11.41.

**Verdict: reproduced.** MAE is within 0.5 % of the published value (criterion ±10 %) and coverage within 0.002 (criterion ±0.03). This holds although our run is smaller (1 repetition, 2 chains × 500 draws). NUTS diagnostics: 0 divergences, maximum R-hat 1.015, minimum ESS 130.

Two findings stated in the experiment specification are confirmed:
- **The DM point error is no better than the historical mean shares.** JSD is 0.135 for the DM against 0.132 for the historical shares. The DM's value lies in its intervals: coverage 0.95 and CRPS 9.5.
- **Cancellation.** For the 76 country-folds with two or more test years, the predicted share vectors of different held-out years of the same country are identical to within 1.1e-16 (`cancellation_check.csv`). The year effect cancels in the softmax, so the base model cannot change the predicted shape from year to year. This is the gap that climate is meant to fill.

---

## 3. Parts B and C on Brazil: main results

Notes for all tables in this section:
- All values are means of per-municipality-year scores (equal weight).
- Brackets are 95 % cluster-bootstrap intervals (2,000 replicates) that resample **municipalities**.
- The JSD noise floor is 0.006 for the 60-municipality pilot and 0.015 for the 300-municipality run.

### 3.1 Absolute performance, weekly resolution

Source: `brazil/table_abs_means_ci.csv`, `table_main_weekly*.csv`; figure `fig_baseline_vs_climate_weekly_60.png`.

| Model | JSD (60 mun.) | Peak-week error, weeks (60) | JSD (300 mun.) | Peak-week error (300) |
|---|---|---|---|---|
| B0 uniform | 0.137 [0.126, 0.148] | 11.7 [10.9, 12.5] | 0.189 [0.182, 0.195] | 11.7 [11.4, 12.0] |
| B1p seasonal, past years only | 0.094 [0.085, 0.104] | 6.20 [5.69, 6.72] | 0.136 [0.130, 0.142] | 6.33 [6.10, 6.57] |
| B1 seasonal profile (LOYO) | 0.082 [0.074, 0.091] | 5.97 [5.45, 6.54] | 0.120 [0.115, 0.125] | 6.04 [5.82, 6.28] |
| B4 DM, base-paper structure | 0.082 [0.074, 0.090] | 5.80 [5.34, 6.30] | 0.119 [0.114, 0.124] | 5.94 [5.72, 6.17] |
| **B4s** DM + annual size (primary comparator) | **0.0735** [0.066, 0.082] | 5.97 [5.45, 6.57] | **0.111** [0.106, 0.116] | 5.99 [5.78, 6.22] |
| **M1** B4s + climate | **0.0727** [0.065, 0.081] | 5.92 [5.38, 6.50] | **0.108** [0.103, 0.113] | 5.91 [5.68, 6.15] |
| M1-short (lags 0–3 weeks only) | 0.0737 | 6.03 | – | – |
| M1-mult (multinomial likelihood) | 0.0819 | 6.48 | – | – |
| M2 boosted softmax (optional) | 0.0769 [0.069, 0.085] | 6.08 [5.60, 6.60] | – | – |

Interval calibration of the weekly counts (500 draws each from Dirichlet(φπ) → Multinomial(N); every draw sums exactly to N):

| | 80 % coverage (60 / 300) | 95 % coverage (60 / 300) | CRPS (60 / 300) |
|---|---|---|---|
| B4 | 0.80 / 0.81 | 0.93 / 0.94 | 69.8 / 29.8 |
| B4s | 0.80 / 0.81 | 0.92 / 0.93 | 62.0 / 26.9 |
| M1 | 0.78 / 0.80 | 0.92 / 0.93 | 58.6 / 25.6 |

At monthly resolution with 60 municipalities, the JSD values are:

| B0 | B1 | B4 | B4s | M1 |
|---|---|---|---|---|
| 0.123 | 0.070 | 0.070 | 0.061 | 0.061 |

### 3.2 Paired differences

Source: `brazil/table_paired_{weekly,monthly}{,_300}.csv`.

- The difference is the JSD of the first model minus the JSD of the second; a negative value means the first model is better.
- "mun." is the municipality bootstrap and "year" is the LOYO year-block bootstrap (14 blocks).
- "Years better" counts the LOYO years in which the mean difference is below 0.

| Contrast | Mean diff. | Rel. | 95 % CI (mun.) | 95 % CI (year) | Years better |
|---|---|---|---|---|---|
| **Weekly, 60: M1 − B4s (H1 contrast)** | −0.00084 | −1.1 % | [−0.00197, +0.00022] | [−0.00426, +0.00267] | 9/14 |
| Weekly, 60, without 2020: M1 − B4s | −0.00024 | −0.3 % | [−0.00135, +0.00082] | [−0.00346, +0.00335] | 8/13 |
| **Weekly, 300: M1 − B4s** | **−0.00335** | **−3.0 %** | **[−0.00404, −0.00267]** | [−0.00660, +0.00011] | 10/14 |
| Weekly, 300, without 2020: M1 − B4s | −0.00342 | −3.1 % | [−0.00411, −0.00274] | [−0.00698, +0.00012] | 9/13 |
| Monthly, 60: M1 − B4s | −0.00012 | −0.2 % | [−0.00120, +0.00092] | [−0.00317, +0.00301] | 9/14 |
| Weekly, 60: M1 − B1 | −0.00935 | −11.4 % | [−0.0122, −0.0066] | [−0.0125, −0.0063] | 13/14 |
| Weekly, 60: B4s − B4 (annual-size terms) | −0.00816 | −10.0 % | [−0.0107, −0.0059] | [−0.0107, −0.0056] | 14/14 |
| Weekly, 300: B4s − B4 | −0.00804 | −6.8 % | [−0.0093, −0.0069] | [−0.0134, −0.0012] | 12/14 |
| Weekly, 60: B4 − B1 | −0.00035 | −0.4 % | [−0.00064, −0.00008] | [−0.0018, +0.0013] | 7/14 |
| **Weekly, 60: M1 − M1-short (H2)** | −0.00106 | −1.4 % | [−0.00202, −0.00016] | [−0.00385, +0.00173] | 10/14 |
| Weekly, 60: M2 − B4s | +0.00342 | +4.6 % | [+0.0016, +0.0054] | [−0.0008, +0.0075] | 3/14 |
| Weekly, 60: M1-mult − M1 | +0.00922 | +12.7 % | [+0.0074, +0.0111] | [+0.0053, +0.0126] | 2/14 |

- **Peak-week error.** No climate contrast changes peak-week error detectably. Weekly 60, M1 − B4s: −0.05 weeks, CI [−0.25, +0.15]. Weekly 300: −0.08 weeks, CI [−0.18, +0.02].
- **Headroom captured** (M1 vs B4s, relative to the noise floor) is 1.2 % for the 60-municipality pilot and 3.5 % for the 300-municipality run.
- **N-weighted JSD** shows a larger relative gain, which suggests that climate helps more in large epidemics:
  - weekly 60: M1 0.061 vs B4s 0.067;
  - weekly 300: 0.078 vs 0.085.

### 3.3 Where and when

Source: `brazil/table_by_region_weekly*.csv`, `table_gain_by_year_weekly*.csv`; figure `fig_gain_by_region_year*.png`.

Gain = JSD(B4s) − JSD(M1); a positive value means climate helps. Brackets are municipality-bootstrap 95 % intervals.

| Region | Gain, 60 municipalities | Gain, 300 municipalities |
|---|---|---|
| North | −0.0013 [−0.0029, +0.0004] | −0.0020 [−0.0029, −0.0012] |
| Northeast | −0.0025 [−0.0039, −0.0008] | −0.0015 [−0.0023, −0.0006] |
| Center-West | +0.0023 [+0.0003, +0.0044] | +0.0038 [+0.0022, +0.0054] |
| Southeast | +0.0037 [+0.0016, +0.0059] | +0.0058 [+0.0048, +0.0067] |
| South | +0.0019 [−0.0004, +0.0047] | +0.0070 [+0.0049, +0.0091] |
| **H3 contrast: (South + Southeast) − North** | **+0.0041 [+0.0018, +0.0064]** | **+0.0080 [+0.0068, +0.0092]** |

**By held-out year (weekly, 60 municipalities).** The climate gain is positive in 9 of 14 years. The largest gains are in 2016 (+0.010), 2020 (+0.009), 2013 and 2019. Clear losses occur in 2012, 2017, 2021 and 2022, and 2021 is the worst year (−0.012).

**Belo Horizonte** (`fig_example_bh.png`):

| Year | B1 | B4s | M1 |
|---|---|---|---|
| 2016 | 0.139 | 0.110 | 0.080 |
| 2019 | 0.209 | 0.108 | 0.054 |

Values are JSD. M1 moves the reconstructed peak towards the observed one in both years. The same figure also shows the largest single loss with N ≥ 2,000 (Maringá 2021), where climate made the reconstruction worse.

---

## 4. Sanity checks

Source: `brazil/sanity_checks.csv`. **All checks pass.**

| Check | Result |
|---|---|
| 1. Exact sums | Pass. Asserted for every model and municipality-year (largest-remainder allocation, `evaluation.allocation.allocate_integers`), and for every one of the 500 probabilistic draws (`models.dirichlet_multinomial.draw_counts`) |
| 2. Uniform is worst | Pass at both resolutions. Weekly: B0 0.137 against the next worst, B1p, at 0.094 |
| 3. Zero-climate equivalence | Pass. M1's prediction function, run with B4s parameters and θ = δ = 0, matches B4s to a maximum absolute difference of 0.0 |
| 4. Train-and-test climate shuffle (M1-shufTT, 3 seeds, refitted in every fold) | Pass. The gain vanishes; the shuffled model is slightly **worse** than B4s: +0.00044 [+0.00022, +0.00068] weekly and +0.00046 monthly. Seeds: +0.00016, +0.00071, +0.00047. Real-climate M1 beats the shuffled fits by about 0.0012 JSD |
| 5. Test-only shuffle | Pass. Giving the trained M1 another year's climate raises its JSD: weekly 0.0761 vs 0.0727 (60 municipalities) and 0.1133 vs 0.1077 (300 municipalities) |
| 6. Leakage audit | Pass in every fold. Recomputing with the held-out year set to NaN leaves unchanged h_m, the climatology coefficients and SDs, the centring constants and the training weights |
| 7. Pipeline consistency with the smoke test | Pass. B1p recomputed for 3,457 rows (2023 or earlier) matches with a maximum absolute JSD difference of 1.1e-16 |
| Final-test path | Pass. Unit-tested on a dummy year with synthetic data. The 2024 scoring path (`evaluation.cv.final_test_split`) exists but is disabled |
| Climate gaps | 0 NaN anomalies, so no zero-filling was needed |

---

## 5. What H1, H2 and H3 currently suggest

These readings come from the pilot and do not test the hypotheses.

- **H1 (M1 beats B4s).** Not supported in the 60-municipality pilot. The gain is −1.1 %, and both intervals include 0.
  - In the 300-municipality run with frozen settings, the gain is −3.0 %. The municipality interval excludes 0, but the year-block interval only just includes it (upper bound +0.0001), and 10 of 14 years show a gain.
  - The gain falls short of the proposed 5 % practical threshold. The required 2024 test has not been run.
  - The shuffle tests show that the small gain is tied to the correct year's climate. It is therefore a real but small timing signal, not an artefact.
  - At monthly resolution, climate adds nothing beyond B4s.
- **H2 (long lags beat short lags).** Partly supported.
  - M1 vs M1-short: −1.4 %, with a municipality interval below 0 but a year interval that includes 0.
  - Short lags alone add nothing: M1-short vs B4s is +0.0002.
  - The fitted coefficients (`fig_lag_coefficients.png`) put the signal in rainfall 4–7 weeks before and temperature 12–20 weeks before. The national θ for the 16–20-week temperature block is about +0.09 per SD of anomaly in every fold. The shuffled fits give coefficients near 0.
- **H3 (South and Southeast gain more than the North).** Consistent with the hypothesis on the municipality bootstrap: +0.0041 [0.0018, 0.0064] in the pilot and +0.0080 [0.0068, 0.0092] in the 300-municipality run.
  - Climate slightly **hurts** in the North and Northeast. The North is also where NASA POWER rainfall anomalies are least reliable, so the planned ERA5-Land check is needed before H3 can be interpreted.
  - No year-block interval was computed for the H3 contrast.
- **Strongest non-climate finding.** Annual-size information (B4s vs B4) cuts JSD by 7–10 %, consistently in 12–14 of 14 years. The base-paper structure without year-varying terms (B4) is essentially the seasonal profile (B1), which matches the cancellation result in Part A.
- **Other findings:**
  - The multinomial likelihood is clearly worse than the DM.
  - The boosted model (M2, untuned) is worse than B4s.
  - The past-only profile (B1p) is about 14 % worse than the LOYO profile, which shows how much the "future years" in LOYO help.

---

## 6. What was scaled down, and deviations from the experiment specification

All deviations, with justifications, are collected in [`DEVIATIONS.md`](DEVIATIONS.md). The list below is the original pilot record.

**Part A**
- 1 repetition × the interpolation mask × 3 folds, with 2 chains × 500 draws (as the spec planned). The published run used 3 repetitions × 4 chains × 1,000 draws at target_accept 0.99.
- The reproduction criterion was applied to the **median-based** MAE, because that is how the published table defines it. The mean-based MAE (11.69) is reported alongside.
- Posterior medians of the NUTS hyperparameters were not stored.
- The NUTS run was restarted once after a session interruption killed the first attempt. Nothing was lost except time.

**Brazil models**
- Penalties use **fixed** scales equal to the base paper's prior scales (g 0.3, region 0.4, municipality 0.15, plus the specified γ, θ, δ and log φ penalties). Deviations are additive. Hierarchical scale parameters are not estimated.
- The annual-size covariate n_mt is centred on each **municipality's** fold-training mean of log(1 + N), not on a global mean.
- The previous-year-missing indicator × harmonics (for 2010, and for the year after ES 2021) has national coefficients only.
- **Convergence.**
  - L-BFGS stopped on relative function reduction (ftol 1e-14) after 60–860 iterations. The gradient-norm target of 1e-6 was not reached: the maximum |gradient| was 0.2 for the pilot and 1.2 for the 300-municipality run, on objectives of order 10^6–10^7.
  - M1-mult hit the 2,000-iteration cap in 7 of 14 weekly folds and 1 of 14 monthly folds. Its results are approximate.
  - Logs are in `brazil/convergence_*.csv`.
- Folds ran **sequentially with warm starts**, not in parallel with joblib. The spec asks for both, and they conflict; the runtime was well within budget either way.
- In M1-shufTT, when a 52-week year's features are moved into a 53-week year, slot 53 gets 0 (no anomaly).
- The climatology regression uses 2009 plus the training years.

**Scope**
- **B2 (Denton) was skipped**, as the spec allows (optional).
- **M2:** fixed hyperparameters (depth 4, learning rate 0.05, 300 rounds), with **no early stopping**. It was run on the weekly pilot only.
- **The 300-municipality run** covers the main models only (B4, B4s, M1, with B0, B1, B1p and the test-only shuffle). The train-and-test shuffle was run only on the 60-municipality pilot.
- **Headroom captured** is computed as a ratio of means, not as a mean of per-row ratios.
- **The H3 contrast** uses the municipality bootstrap only.
- **ε guard (1e-4 vs 1e-6).** The difference is negligible (−0.00002 JSD), so the base-paper ε = 1e-6 is used throughout.

---

## 7. Limitations

- **Pilot sample.** 60 municipalities, chosen among those with many large years.
  - The 300-municipality evaluation reuses the 60 pilot municipalities.
  - The hyperparameters were fixed in advance, not tuned on the pilot.
  - LOYO uses future years, so the results are mildly optimistic; B1p shows the size of that effect.
- **Intervals ignore parameter uncertainty.** Plug-in MAP parameters slightly under-cover: M1's 80 % coverage is 0.78–0.80 and its 95 % coverage 0.92–0.93.
- **Climate data.** NASA POWER point data at the municipal seat (MERRA-2 grid). Rainfall agreement with ERA5-Land is weak in the North, so the regional results may partly reflect data quality.
- **Notifications are not infections.** 2020 has COVID-era reporting effects, and the main H1 contrast in the 60-municipality pilot shrinks without 2020: −0.3 % instead of −1.1 %.
- **The year-block bootstrap is wide.** Climate, ENSO and reporting shocks are shared across municipalities within a year, so the 14 years are the effective sample for H1.
- **Circularity.** Reconstructed series must not be used to estimate climate–dengue effects.

---

## 8. Follow-up: rain-lag extension (condensed)

Full write-up: [`results/rain_extension/RESULTS_RAIN.md`](../results/rain_extension/RESULTS_RAIN.md); design: [`PREREGISTRATION.md`](../results/rain_extension/PREREGISTRATION.md) (written before any new model was fitted).

- **M1-R** ("dual rain"): lags extended to 26 weeks, rain anomaly split into wet and dry parts (24 features). **M1-RTH**: the same split also for temperature and humidity (36 features).
- **Result: no improvement.** Both variants are slightly worse than M1 at both sample sizes, and no pre-registered improvement criterion is met.

| Contrast | 60 municipalities | 300 municipalities |
|---|---|---|
| M1-R vs M1 | +1.7 % JSD | +0.6 % JSD |
| M1-RTH vs M1 | +4.5 % JSD | +1.9 % JSD |
| M1 vs B4s (reference) | −1.1 % | −3.0 % |
| M1-R vs B4s | +0.5 % | −2.5 % |
| M1-RTH vs B4s | +3.3 % | −1.2 % |

| Model | JSD 60 | JSD 300 |
|---|---|---|
| B4s | 0.0735 | 0.1110 |
| M1 | **0.0727** | **0.1077** |
| M1-R | 0.0739 | 0.1083 |
| M1-RTH | 0.0759 | 0.1097 |
| M1-R climate-shuffled (60) | 0.0739 | – |

- The fitted rain coefficients are consistent across the 14 folds (wet anomalies 0–7 weeks earlier shift cases slightly towards a week; dry anomalies 0–11 weeks earlier shift them away, with a 2–5× larger effect), but they do not improve out-of-sample reconstructions: at 60 municipalities M1-R with shuffled climate scores the same as M1-R with the correct climate.
- All leakage audits pass; the M1 refit reproduced the saved pilot M1 exactly (max |Δshare| = 0.0).
- Most likely limitation: NASA POWER rainfall anomalies, especially in the North (Section 7 and `DEVIATIONS.md` A1).
