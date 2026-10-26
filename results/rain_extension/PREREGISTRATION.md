# Pre-registration: rain (and humidity/temperature) lag extension

**Written:** 2026-10-21T07:32:20Z (UTC), before any new model was fitted or any new result was seen.
**Scope:** follow-up experiment on the finished pilot (`results/RESULTS.md`). Existing models (B4, B4s, M1, ...) are not changed. `RUN_FINAL_2024 = False`; 2024 is never loaded or scored.

## Question

Can a more realistic rainfall effect (and, secondarily, asymmetric temperature and humidity effects) increase the climate gain over B4s?

## Evidence the variants encode

- Lowe et al. 2021 (Brazil, DLNM): excess rain raises risk 0–3 months later; drought raises risk 3–5 months later (opposite-sign anomalies matter at different lags).
- Caribbean compound extremes (Lancet Planet Health 2025): long-lag dry, mid-lag hot, short-lag wet gives the highest risk.
- Bangkok DLNM (BMC Public Health 2025; monthly data, one city, lags 0–12 months): U-shaped rainfall, biphasic humidity over lags, J-shaped mean temperature. Used for model **shape** only, not for numbers.

## Variants (fixed; no variant will be added after results are seen)

Common to all: the existing M1 pipeline. Anomaly A = (value − harmonic climatology with intercept + 3 harmonics, fitted per municipality on 2009 + training years only) / training residual SD, for temp_mean, log(1 + precip), humid_mean. Lag-block feature = mean of A over the lags in the block on the continuous weekly timeline (crossing year boundaries). Same B4s structure (offset h, g, r, c, annual-size terms), same national θ + regional δ coefficients with Gaussian penalties at M1's scales (θ sd 0.5, δ sd 0.25), same DM likelihood, same L-BFGS settings, sequential LOYO folds 2010–2023 with warm starts from the previous fold of the same model, same exclusions (train N ≥ 20, eval N ≥ 100, `exclude_suggested`).

| Variant | Lag blocks (weeks) | Features per block | # features |
|---|---|---|---|
| **M1** (reference, existing) | 0–3, 4–7, 8–11, 12–15, 16–20 | temp Ā, rain Ā, humid Ā | 15 |
| **M1-R** ("dual rain") | 0–3, 4–7, 8–11, 12–15, 16–20, **21–26** | temp Ā; rain **wet = max(Ā, 0)** and rain **dry = min(Ā, 0)**; humid Ā | 24 |
| **M1-RTH** | same 6 blocks | temp **pos = max(Ā,0)**, temp **neg = min(Ā,0)**; rain wet, rain dry; humid **pos**, humid **neg** | 36 |

Definition detail: the split is applied to the **block-mean** anomaly Ā (the same quantity M1 uses), so wet + dry = Ā exactly and M1-R nests M1 plus a 21–26 block. A wet coefficient θ_wet > 0 means a wetter-than-usual block shifts dengue towards that week; a dry coefficient θ_dry < 0 means a drier-than-usual block (Ā < 0) shifts dengue towards that week (risk from drought). Both on the logit-share scale (within-year timing, not incidence level).

M1 reference: M1 is refitted in the 60-municipality run with identical code; if its predictions reproduce the saved `results/brazil/preds_weekly_60.npz` M1 predictions (max |Δshare| < 1e-6), the saved M1/B4s metrics are used as reference at both 60 and 300 municipalities (300: saved `preds_weekly_300.npz` / `metrics_detail_weekly_300.csv`, same code path). Otherwise M1 is refitted at 300 as well.

## Runs

1. Weekly, 60 municipalities (`results/subset60.csv`): M1 (refit check), M1-R, M1-RTH.
2. Climate shuffle check (train + test, 1 seed = seed 0 of the existing M1-shufTT derangement): whichever of M1-R / M1-RTH has the lower mean JSD on run 1. Pass if shuffled − B4s is not negative with a municipality CI excluding 0 (same criterion as the existing check 4), and the real variant beats the shuffled one.
3. Weekly, 300 municipalities: M1-R and M1-RTH (main comparisons only), if the total runtime stays within about 3 h on 4 CPUs.

## Metrics and comparisons

- Metrics: JSD (**primary**), peak-week error, MAE on shares, EMD, 80 %/95 % interval coverage and CRPS (500 DM draws), exactly as in `run_brazil.score`.
- Paired comparisons (per municipality-year, a − b): M1-R − M1, M1-RTH − M1, M1-R − B4s, M1-RTH − B4s, M1-RTH − M1-R; each with 95 % municipality cluster bootstrap and 95 % LOYO year-block bootstrap (2,000 replicates, seed 1), number of years better, all years and without 2020.
- By-region table (variant − M1 and variant − B4s, municipality bootstrap).
- Fitted national θ (wet/dry for rain; pos/neg for temperature and humidity) by lag block: mean and range across the 14 folds.

## Decision rule (pre-registered)

- A variant **improves** on M1 only if the mean JSD change vs M1 is negative **and both** the municipality-bootstrap and the year-block-bootstrap 95 % CIs exclude 0. This is evaluated separately at 60 and at 300 municipalities; the 300-municipality result is the more informative one.
- Versus B4s, the relative JSD change is reported against the **5 %** practical threshold (gain ≥ 5 % with both CIs excluding 0 would meet it).
- Two variants are tested; no multiplicity correction is applied, and this is stated as a limitation. M1-RTH − M1-R is descriptive.

## Leakage

The existing per-fold leakage audit (h_m, centring constants, climatology coefficients and SD, training arrays) is run for every fold, plus a new-feature audit: (a) climatology/SD unchanged when the held-out year's climate is set to NaN; (b) new features unchanged when the held-out year's weekly counts and totals are set to NaN; (c) wet + dry (and pos + neg) equals the M1 block mean for the 5 original blocks; (d) no NaN in the 21–26 block for any evaluation week.
