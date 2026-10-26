# Rain lag extension: do dual rain lags improve the climate gain?

**Follow-up to the pilot (`results/RESULTS.md`).**

- The design was pre-registered in `PREREGISTRATION.md` (2026-10-21T07:32Z) before any new model was fitted. No variant was added after the results were seen.
- Weekly resolution, leave-one-year-out (LOYO) 2010–2023.
- 2024 was never loaded (`RUN_FINAL_2024 = False`).

## In plain language

**Did it help? No.**

**The extended models are not better than the original climate model (M1).**
- Two models were tested:
  - **M1-R** ("dual rain") lets wetter-than-usual and drier-than-usual rainfall have different effects and extends the lags to 26 weeks.
  - **M1-RTH** adds the same split for temperature and humidity.
- Both are slightly **worse** than M1 at both sample sizes:

| Contrast | 60 municipalities | 300 municipalities |
|---|---|---|
| M1-R vs M1 | +1.7 % JSD | +0.6 % JSD |
| M1-RTH vs M1 | +4.5 % JSD | +1.9 % JSD |

- No pre-registered improvement criterion is met.

**Against the seasonal DM model with annual size (B4s), the extension gives back part of M1's small gain.**

| Model | Change vs B4s, 300 municipalities | Change vs B4s, 60 municipalities |
|---|---|---|
| M1 | −3.0 % | −1.1 % |
| M1-R | −2.5 % | +0.5 % |
| M1-RTH | −1.2 % | +3.3 % |

- At 300 municipalities, every variant is below the 5 % practical threshold, and no year-block interval excludes 0.

**Where.**
- The pattern by region is unchanged. Climate helps in the South, the Southeast and partly the Center-West, and hurts in the North and Northeast.
- The extra terms cost the most in the Northeast and the Center-West. They are neutral in the Southeast.

**What the models learned.**
- The fitted rainfall coefficients are **consistent across the 14 folds**. They partly resemble the literature:
  - **Wet effect at short lags.** Wetter-than-usual rain 0–7 weeks earlier shifts cases slightly towards that week. This is small, but positive in every fold.
  - **Dry effect at short lags.** Drier-than-usual rain 0–11 weeks earlier shifts cases away from that week. This effect is 2–5× larger than the wet effect.
  - **Long lags (21–26 weeks).** Wet anomalies shift cases away (negative in 14/14 folds). Dry anomalies shift cases slightly towards the week (negative dry coefficient in 13/14 folds).
- How this compares with the literature:
  - The long-lag dry signal is the direction Lowe et al. (2021) describe (drought raises risk 3–5 months later), but it is small.
  - The short-lag pattern is **monotone and concave** (deficits matter more than excesses), **not U-shaped** as in the Bangkok study.
- **These consistent coefficients do not translate into better out-of-sample reconstructions.** On the 60 municipalities, M1-R with climate shuffled between years scores the same as M1-R with the correct climate. The extra terms add variance rather than usable year-specific timing information.

**Bottom line.** The simpler M1 (5 lag blocks, linear anomalies) remains the best climate model. With this data (NASA POWER point climate) and this model class (penalised Dirichlet-multinomial on weekly shares), a more realistic rain effect does not raise the climate gain over B4s.

---

## 1. What was run

| Run | Models | Wall time (4 CPUs) |
|---|---|---|
| Weekly, 60 municipalities (`results/subset60.csv`) | M1 (refit check), M1-R, M1-RTH | 5.6 min (fits 5.2 min) |
| Weekly, 60 municipalities, climate shuffle | M1-R-shufTT-s0 (M1-R was the better variant at 60, as pre-registered) | 2.4 min |
| Weekly, 300 municipalities | M1-R, M1-RTH (main comparisons only) | 28.6 min (fits 26.3 min), run in parallel with the 60-municipality runs |
| Tables and figures (`src/rain_report.py`) | – | about 1 min |

**Total compute** was about 35 min of wall-clock time, well within the 3 h budget.

**Reference models** (B4s, M1, B1) come from the saved pilot runs (`results/brazil/metrics_detail_weekly_{60,300}.csv`). The M1 refit with the new driver reproduced the saved M1 predictions exactly (maximum |Δshare| = 0.0), so the saved M1 and B4s results were used at both 60 and 300 municipalities, as pre-registered.

**Features.**
- **Anomaly A.** Value minus a harmonic climatology fitted in-fold on 2009 plus the training years, divided by the training residual SD. Computed for temperature, log(1 + rain) and humidity.
- **Lag-block mean Ā.** The mean of A within each lag block. The split is applied to Ā: wet = max(Ā, 0) and dry = min(Ā, 0), so wet + dry = M1's feature.
- **Model sizes.** M1-R has 24 features (6 lag blocks); M1-RTH has 36.
- **Unchanged settings:** penalties (national θ sd 0.5, regional δ sd 0.25), DM likelihood, L-BFGS, warm starts, exclusions (train N ≥ 20, eval N ≥ 100) and metrics.

## 2. Main results (JSD, primary)

Notes for the tables in sections 2 and 3:
- Rows are means of per-municipality-year scores.
- A difference is the first model minus the second; a negative value means the first model is better.
- "mun." is the municipality cluster bootstrap and "year" is the LOYO year-block bootstrap. Both use 2,000 replicates and give 95 % intervals.

### 2.1 Absolute performance

`table_main_weekly_{60,300}.csv`

| Model | JSD 60 | Peak-week error 60 | 80 % coverage 60 | 95 % coverage 60 | CRPS 60 | JSD 300 | Peak-week error 300 | 80 % coverage 300 | 95 % coverage 300 | CRPS 300 |
|---|---|---|---|---|---|---|---|---|---|---|
| B1 | 0.0820 | 5.97 | – | – | – | 0.1197 | 6.04 | – | – | – |
| B4s | 0.0735 | 5.97 | 0.800 | 0.923 | 62.0 | 0.1110 | 5.99 | 0.810 | 0.932 | 26.9 |
| **M1** | **0.0727** | **5.92** | 0.782 | 0.917 | **58.6** | **0.1077** | 5.91 | 0.805 | 0.929 | **25.6** |
| M1-R | 0.0739 | 6.11 | 0.776 | 0.914 | 59.3 | 0.1083 | **5.89** | 0.803 | 0.927 | 25.7 |
| M1-RTH | 0.0759 | 6.18 | 0.766 | 0.908 | 60.4 | 0.1097 | 5.96 | 0.797 | 0.925 | 26.0 |
| M1-R climate-shuffled | 0.0739 | 6.03 | 0.791 | 0.919 | 62.1 | – | – | – | – | – |

### 2.2 Paired differences and the pre-registered decision

`table_paired_weekly_{60,300}.csv`

| Contrast | n | Mean diff. | Rel. | 95 % CI (mun.) | 95 % CI (year) | Years better | Decision |
|---|---|---|---|---|---|---|---|
| **60: M1-R − M1** | 840 | +0.00122 | +1.7 % | [+0.00087, +0.00154] | [+0.00049, +0.00196] | 2/14 | does not improve (worse) |
| **60: M1-RTH − M1** | 840 | +0.00324 | +4.5 % | [+0.00270, +0.00380] | [+0.00183, +0.00485] | 2/14 | does not improve (worse) |
| 60: M1 − B4s (reference) | 840 | −0.00084 | −1.1 % | [−0.00197, +0.00022] | [−0.00426, +0.00267] | 9/14 | below 5 % |
| 60: M1-R − B4s | 840 | +0.00039 | +0.5 % | [−0.00090, +0.00158] | [−0.00340, +0.00414] | 8/14 | below 5 % |
| 60: M1-RTH − B4s | 840 | +0.00241 | +3.3 % | [+0.00114, +0.00365] | [−0.00161, +0.00647] | 8/14 | below 5 % (worse) |
| 60: M1-RTH − M1-R | 840 | +0.00202 | +2.7 % | [+0.00170, +0.00237] | [+0.00119, +0.00301] | 1/14 | descriptive |
| **300: M1-R − M1** | 3,723 | +0.00062 | +0.6 % | [+0.00042, +0.00081] | [−0.00001, +0.00138] | 5/14 | does not improve |
| **300: M1-RTH − M1** | 3,723 | +0.00200 | +1.9 % | [+0.00172, +0.00226] | [+0.00091, +0.00323] | 1/14 | does not improve (worse) |
| 300: M1 − B4s (reference) | 3,723 | −0.00335 | −3.0 % | [−0.00404, −0.00267] | [−0.00660, +0.00011] | 10/14 | below 5 % |
| 300: M1-R − B4s | 3,723 | −0.00273 | −2.5 % | [−0.00350, −0.00198] | [−0.00611, +0.00081] | 10/14 | below 5 % |
| 300: M1-RTH − B4s | 3,723 | −0.00135 | −1.2 % | [−0.00215, −0.00058] | [−0.00477, +0.00228] | 10/14 | below 5 % |
| 300: M1-RTH − M1-R | 3,723 | +0.00138 | +1.3 % | [+0.00122, +0.00152] | [+0.00072, +0.00211] | 1/14 | descriptive |

**Without 2020** the conclusions are the same:
- 300 municipalities: M1-R − M1 is +0.6 % and M1-RTH − M1 is +2.0 %.
- 60 municipalities: M1-R − M1 is +1.9 % and M1-RTH − M1 is +4.9 %.

**N-weighted JSD** (large epidemics count more) tells the same story. M1-R − M1 is +0.0011 (60) and +0.0006 (300).

**Secondary metrics against M1** (all years):
- **Peak-week error.**
  - 60 municipalities: worse for M1-R (+0.19 weeks; municipality CI [+0.02, +0.38], year CI [+0.04, +0.33]).
  - 300 municipalities: unchanged for M1-R (−0.01 weeks; [−0.08, +0.05] and [−0.11, +0.08]). M1-RTH: +0.05 weeks, with both CIs including 0.
- **EMD.** Worse for both variants at both sizes; all CIs exclude 0.
- **MAE on shares.** Essentially unchanged.
- **Coverage.** Coverage of M1-R and M1-RTH is 0.002–0.016 below M1's.

## 3. Where and when

### 3.1 By region

Source: `table_by_region_weekly_{60,300}.csv`; figure `fig_rain_gain_by_region.png`. Values are mean JSD differences, 300 municipalities, with the municipality-bootstrap 95 % CI.

| Region | M1 − B4s | M1-R − B4s | M1-RTH − B4s | M1-R − M1 |
|---|---|---|---|---|
| North | +0.0020 [+0.0012, +0.0028] | +0.0025 [+0.0016, +0.0035] | +0.0038 [+0.0028, +0.0049] | +0.0005 [−0.0000, +0.0010] |
| Northeast | +0.0015 [+0.0006, +0.0023] | +0.0028 [+0.0017, +0.0038] | +0.0036 [+0.0024, +0.0047] | +0.0012 [+0.0009, +0.0016] |
| Center-West | −0.0038 [−0.0053, −0.0022] | −0.0018 [−0.0034, −0.0000] | +0.0001 [−0.0016, +0.0020] | +0.0020 [+0.0017, +0.0024] |
| Southeast | −0.0058 [−0.0067, −0.0048] | −0.0059 [−0.0070, −0.0048] | −0.0043 [−0.0054, −0.0031] | −0.0001 [−0.0004, +0.0002] |
| South | −0.0070 [−0.0091, −0.0049] | −0.0063 [−0.0087, −0.0041] | −0.0056 [−0.0080, −0.0033] | +0.0007 [+0.0002, +0.0011] |

- **Southeast.** The dual-rain model ties with M1, the only region where it does not lose.
- **Northeast and Center-West.** It loses the most here.
- **North.** The pilot finding that climate hurts here gets slightly worse with the extra terms, both in the North and the Northeast. NASA POWER rainfall is weakest in the North.

### 3.2 By year

`table_by_year_weekly_{60,300}.csv`

- At 300 municipalities, M1-R beats M1 in 5 of 14 years: 2014, 2015, 2016, 2017 and 2019, each by at most 0.0008 JSD.
- Its largest losses are in:
  - 2022: +0.0034;
  - 2010: +0.0027;
  - 2013: +0.0023.
- No year shows a large gain from the dual-rain terms.

## 4. Fitted coefficients

Source: `table_coefficients.csv`; figures `fig_rain_coefficients.png` (60) and `fig_rain_coefficients_300.png` (300).

- The numbers are national θ: the change in the logit share of a week per 1 SD of block-mean anomaly. They are means over the 14 folds; "+k/14" counts the folds with θ > 0.
- **The model predicts within-year timing, not incidence.** A positive θ means an anomaly of the same sign shifts cases towards that week, relative to the other weeks of the same year.

**Rain, M1-R, 300 municipalities:**

| Lag (weeks) | 0–3 | 4–7 | 8–11 | 12–15 | 16–20 | 21–26 |
|---|---|---|---|---|---|---|
| wet part, θ_wet | +0.023 (14/14) | +0.019 (14/14) | −0.003 (5/14) | 0.000 (7/14) | +0.009 (11/14) | **−0.044 (0/14)** |
| dry part, θ_dry | **+0.054 (14/14)** | **+0.088 (14/14)** | +0.040 (14/14) | +0.012 (10/14) | −0.008 (3/14) | −0.020 (1/14) |

The 60-municipality fit has the same shape. Its dry part is larger: +0.11 at 4–7 weeks, and still positive at 12–20 weeks.

**Rainfall, read against the literature:**
- **Short lags (0–11 weeks).** A dry block (Ā < 0) times a positive θ_dry lowers the share. Drier-than-usual weeks 0–11 weeks earlier therefore push cases away from that week. Wetter-than-usual weeks push cases towards it, with a coefficient 2–5× smaller.
  - This is the "excess rain raises risk 0–3 months later" direction of Lowe et al. (2021). But the response is concave: a rain deficit matters more than an excess.
  - It is **not** the U-shape of the Bangkok DLNM, which has risk at both low and high rainfall.
- **Long lags (21–26 weeks).**
  - θ_wet is negative in every fold: a wet half-year earlier shifts cases away.
  - θ_dry is mostly negative: a dry period 5–6 months earlier shifts cases slightly towards the week.
  - This matches the sign of Lowe's "drought raises risk 3–5 months later", but the effect is small (about 0.02–0.03 per SD). At 16–20 weeks it is inconsistent across folds.

**Temperature (M1-RTH, 300):**
- The cool part is positive at every lag (+0.03 to +0.10, 14/14 folds), so cooler-than-usual periods push cases away.
- The warm part is near 0 at 0–11 weeks, but +0.05 to +0.10 at 12–26 weeks (14/14 folds).
- This is asymmetric: cool is protective at all lags, while warmth matters mainly 3–6 months ahead. It is loosely J-like, and consistent with M1's strongest block (temperature 16–20 weeks).

**Humidity (M1-RTH, 300).** The sign changes over lags:
- 0–11 weeks: positive, so more humid shifts cases towards the week.
- 16–20 weeks: negative in both parts (0/14 folds positive).
- 21–26 weeks: positive again.

This resembles the "risk, then protective, then rise" pattern of the Bangkok study, but on a weekly lag scale, and the variant that uses it (M1-RTH) predicts worse.

**Climate-shuffled fit (M1-R-shufTT-s0, 60).** All coefficients lie within about ±0.04, with no clear structure (`table_coefficients.csv`). The structure in the real fits is therefore tied to the correct years.

## 5. Checks

| Check | Result |
|---|---|
| Leakage audit, all 14 folds of every run: existing checks (h_m, centring, climatology/SD, training arrays) plus the new-feature audit (climatology unchanged with the held-out year's climate set to NaN; features unchanged with the held-out year's counts and totals set to NaN; wet + dry = M1 block mean; no NaN in the 21–26 block for evaluation weeks) | **pass** (`table_leakage_audit_summary.csv`, `leakage_audit_*.csv`) |
| M1 refit reproduces the saved pilot M1 | pass: max \|Δshare\| = 0.0 |
| Exact sums (allocation and every DM draw) | pass: asserted in the unchanged `run_brazil.score` |
| 2024 untouched | pass: `RUN_FINAL_2024 = False`; `load_tables` drops epi_year ≥ 2024 |
| Climate shuffle, train and test, 1 seed, on M1-R (the better variant at 60) | **Formally passes the pre-registered rule, but the rule is uninformative here.** Shuffled − B4s = +0.00041 (mun. [−0.00013, +0.00093], year [+0.00008, +0.00068]), so the shuffle shows no gain. Real − shuffled = −0.00002 (mun. [−0.00143, +0.00129], year [−0.0038, +0.0038]): the real-climate M1-R is **no better** than its shuffled version. For comparison, in the pilot the real M1 beat its shuffled fits by about 0.0012 |

## 6. Limitations

- **Multiple variants.** Two variants were tested against one reference, with no multiplicity correction. This matters little here, because neither improves.
- **Climate data.** NASA POWER point data at the municipal seat (MERRA-2 grid, about 50 km). Rainfall is the variable it reproduces least well, especially in the **North**, where agreement with ERA5-Land is weakest.
  - A dual rain effect needs accurate rain anomalies, so this is the most likely reason the extension fails.
  - An ERA5-Land or CHIRPS rerun would be the natural next test. It was not run here.
- **Bangkok DLNM.** One city, **monthly** data, lags 0–12 months, modelling incidence level. It was used only to motivate the model shape (U-shaped rain, biphasic humidity, J-shaped temperature), not its numbers or lags.
- **A different target from the DLNM literature.** Lowe et al. 2021 and the Caribbean study also model incidence, not within-year shares. Our coefficients describe *timing* given the annual total, so agreement or disagreement in sign is suggestive only.
- **A crude asymmetry.** The split is applied to block means of standardised anomalies, which is a piecewise-linear hinge at 0. A smooth spline in Ā, a cross-basis like a DLNM, or compound wet-hot-dry interactions might behave differently. These were not pre-registered and were not run.
- **Penalties.** Fixed at M1's scales and not re-tuned for 24 or 36 features. Stronger shrinkage might recover M1's performance, but cannot exceed it in expectation if the extra terms carry no out-of-sample signal, which is what the shuffle check suggests.
- **Validation design.** LOYO over 14 years. The year-block intervals are wide, and 2024 (the final test year) was not used.
- **No causal reading.** Reconstructed weekly series must not be used to estimate climate–dengue effects, because that would be circular. The coefficients above come from fits to observed data only.

## Files

| File | Content |
|---|---|
| `PREREGISTRATION.md` | Variants, decision rule and checks, written before any run |
| `table_main_weekly_{60,300}.csv` | Mean metrics per model (all years and without 2020) |
| `table_paired_weekly_{60,300}.csv` | Paired differences (JSD, peak error, MAE on shares, EMD) with both bootstrap CIs, years better, pre-registered decision flags |
| `table_by_region_weekly_{60,300}.csv` | By region |
| `table_by_year_weekly_{60,300}.csv` | By held-out year |
| `table_coefficients.csv` | θ by feature: mean, SD, min, max, folds positive, regional θ + δ |
| `table_leakage_audit_summary.csv`, `leakage_audit_*.csv`, `checks_rain.json` | Audits and checks |
| `fig_rain_gain_vs_M1_B4s.png` | JSD change vs M1 and vs B4s with both CIs, at 60 and 300 |
| `fig_rain_gain_by_region.png` | Gain over B4s by region |
| `fig_rain_coefficients[_300].png` | Wet/dry, warm/cool and humid/dry-air coefficients by lag block |
| `metrics_detail_*.csv`, `preds_*.npz`, `fitted_params_*.json`, `runtime_*.csv`, `run_info_*.json`, `logs/` | Raw outputs |
