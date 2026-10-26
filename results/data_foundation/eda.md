# EDA and baseline smoke test (phase A: data foundation)

The tables and numbers below are produced by `src/eda_baselines.py` and `src/validate_sources.py`. Raw output is in `reports/eda_baselines_stdout.txt`.

## 1. What we have

* **300 municipalities in all 27 states.** Center-West 45, North 45, Northeast 45, South 45, Southeast 120. They were chosen by dengue burden among the 700 most populous municipalities, with at least 45 per region.
* **Weekly dengue notifications by epi week, 2010 W1 – 2025 W53** (InfoDengue `casos`).
  * There are no missing weeks.
  * 23.7 M cases in total.
  * 2025 is provisional. Espírito Santo 2021 is a reporting artefact and is flagged.
* **Weekly climate from NASA POWER, 2009–2025.** Variables are precipitation (weekly total and rainy days), mean/min/max temperature and relative humidity. The record is complete for 2010–2024.
* **Second temperature/humidity source from InfoDengue.** About 5 % of weeks are missing, with a large gap in 2017.
* **53-week years are 2014, 2020 and 2025.** They are handled explicitly through the `n_weeks_year` column. Profiles are estimated on a 53-slot grid; in 52-week years, slot 53 is imputed as the mean of weeks 52 and 1, and the profile is then renormalised to the length of the target year.

Figure: `figures/coverage_heatmap.png`. The left panel shows InfoDengue station-climate coverage and the right panel shows log annual cases. The ES 2021 hole is visible in the Southeast block. South municipalities had very low counts before 2015.

## 2. Seasonality by region

Figure: `figures/seasonal_curves_by_region.png`. It shows weekly shares (median and IQR) across municipality-years with at least 100 cases, for 2010–2024.

* **Southeast and South:** a sharp peak around epi weeks 14–16 (April), with a weekly share of about 0.045–0.05 at the peak.
* **Center-West:** a broader peak around weeks 10–16.
* **North:** the earliest peak, around weeks 5–10. It also has a secondary rise in weeks 45–52, so the season crosses the calendar year boundary.
* **Northeast:** the latest and flattest peak, around weeks 14–20, with a long tail into the middle of the year.
* The IQR bands are wide (peak share about 0.02–0.06). Individual years deviate strongly from the average curve.

## 3. How much does the peak week vary between years? (the "room for climate" question)

This analysis covers municipality-years with at least 100 cases (2010–2024, ES 2021 excluded). It uses 4,023 municipality-years in 293 municipalities, each with at least 5 such years. Full results are in `peak_week_variability.csv` and `figures/peak_week_variation.png`.

| Region | Circular SD of peak week (weeks, median) | Median abs. deviation from the municipality's mean peak (weeks) | Share of years with peak more than 4 weeks off | Circular SD of the curve's centre-of-mass week |
|---|---|---|---|---|
| Center-West | 6.2 | 4 | 0.47 | 4.7 |
| North | 6.9 | 4 | 0.50 | 6.1 |
| Northeast | 8.0 | 5 | 0.53 | 6.8 |
| Southeast | 5.9 | 4 | 0.40 | 4.9 |
| South | 5.9 | 4 | 0.35 | 4.4 |
| **All (median)** | **6.4** | **4** | **0.47** | **5.1** |

**Interpretation:** a typical municipality's peak shifts by about ±6 weeks from year to year, and in almost half of the years it falls more than a month away from its usual week. The timing of the epidemic is far from fixed. A fixed seasonal profile cannot capture this, so there is substantial room for a year-specific signal such as climate to help.

Two caveats apply:

* The argmax peak is noisy when counts are small. The centre-of-mass week is more stable (SD about 5 weeks) and gives the same conclusion.
* Variability is highest in the Northeast and North, which are also the regions where precipitation data agree least between sources (see section 5).

## 4. Smoke test of the evaluation: simple baselines

The task is to predict a municipality-year's weekly shares given only its annual total. Integer counts are then allocated with largest-remainder rounding, so they sum exactly to the annual total; this is asserted for every case. The evaluation set is 3,757 municipality-years (2011–2024, at least 100 cases, at least one earlier year available, ES 2021 and 2025 excluded).

The three baselines are:

* **uniform:** every week gets the same share.
* **seasonal_LOYO:** the municipality's mean weekly share over all *other* complete years (strict leave-one-year-out).
* **seasonal_past_only:** the same profile, but using only years before the test year (no future information).

| Model | JSD mean | JSD median | Peak-week error mean (weeks) | MAE share | MAE counts (cases per week, mean / median) |
|---|---|---|---|---|---|
| uniform | 0.188 | 0.168 | 11.7 | 0.0156 | 95.6 / 19.2 |
| seasonal_past_only | 0.134 | 0.113 | 6.3 | 0.0125 | 60.9 / 15.3 |
| seasonal_LOYO | **0.117** | **0.100** | **6.0** | **0.0117** | **59.1** / **14.2** |
| *sampling-noise floor* | *0.014* | *0.007* | | | |

The noise floor is the JSD between the true shares and a multinomial resample of the same total. Even a perfect model of the expected curve cannot beat it. By size of annual total it is 0.049 for 100–300 cases, 0.017 for 300–1,000, 0.005 for 1,000–5,000 and 0.001 above 5,000. Files: `baseline_smoke_test_summary.csv`, `_by_region.csv`, `_by_year.csv`, `_detail.csv` and `jsd_noise_floor.csv`.

Sanity checks pass:

* The models rank uniform worst, then past-only, then LOYO. LOYO uses more training years.
* The seasonal baseline still misses the peak by about 6 weeks, close to the natural year-to-year spread.
* The gap between the best baseline (JSD 0.117) and the noise floor (0.014) is large. It is the headroom any climate model has to work in.

By region (JSD, LOYO): the North is hardest (0.144), followed by the South (0.126); the Southeast is easiest (0.107). The North and Northeast have the largest peak errors (about 7 weeks).

## 5. Source validation (2013–2020, against Cox et al., Zenodo 10.5281/zenodo.13286671)

* **Annual case totals:** InfoDengue and Cox (SINAN-derived) have a log-scale correlation of 0.990. The median absolute relative difference is 0.6 % (90th percentile 3.5 %).
* **Monthly precipitation:** NASA POWER (0.5° point) against Cox ERA5-Land (population-weighted).
  * The median per-municipality correlation is 0.86.
  * For **anomalies** (the year-to-year part) it is 0.71: Center-West 0.74, Southeast 0.73, South 0.73, Northeast 0.72, **North 0.47**.
* Cox weekly rows use ISO-style week labels, not SINAN epi weeks. Do not join them directly to our weekly table.

## 6. Notes for phase B

* Use `exclude_suggested` (ES 2021 and 2025). Report 2020 separately as a sensitivity check, since COVID disrupted reporting.
* Evaluate on municipality-years with at least 100 cases, or weight the metrics by total. For small totals the JSD noise floor is large (0.05 at 100–300 cases).
* Use `week_index` in `climate_weekly.parquet`, which starts in 2009, to build lags of up to about 20 weeks that cross year boundaries. The Southeast season starts building in December of the previous year.
* For the main baseline, use the past-only seasonal profile as the "realistic" baseline and LOYO as the "strong" one. Note that LOYO uses future years, so any climate model must be evaluated under the same split for a fair comparison.
* Precipitation uncertainty is a real limitation in the North. As a robustness check, re-run that region with Cox ERA5-Land (2013–2020), or with CHIRPS if it is reachable.
