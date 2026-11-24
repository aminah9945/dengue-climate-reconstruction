# Final test on 2024 (protocol)

**Written and committed on 24 Nov 2026, before any model saw 2024.** 2024 had never been
loaded by any model, feature or report before this run. The protocol is fixed here so that the
2024 numbers confirm or refute the 2010–2023 findings, and cannot be used to pick them.

## 1. What is run

One fold: **train 2010–2023, test 2024** (the next step of design A, the expanding window).
300 study municipalities (the same list, selected on 2010–2023 only), NASA POWER weather, ONI.
All code, hyperparameters, penalties, seeds, lags and evaluation rules are those of the
2010–2023 runs. Nothing is tuned on 2024.

| Run | Models | Output |
|---|---|---|
| Official climate study | B0, B1, B4, B4s, M1, M1-noONI, M1-short, M1-shufTT, M1-shufTest, M2 | `results/final2024/official/` |
| Timing models | B1, B4s, M1-noONI, M1, S1, S2, S3, S3C, T1, the controls (S1-wy, S2-wy, S3-wy, S3-wyTest, S1-ns), the mixed-reporting variants, S2C, S2C-noONI | `results/final2024/timing/` |
| Scaled INLA | base, clim15, clim20, trends, clim15trends | `results/final2024/inla/` |

Unlock: the environment variable `DENGUE_RUN_FINAL_2024=1` (the default in `config.py` stays
locked, so every other run and test still cannot load 2024).

```bash
export DENGUE_RUN_FINAL_2024=1 DATA_DIR=<data> TIMING_TRENDS_DIR=data_snapshots/google_trends PYTHONPATH=src
python scripts/run_pipeline.py --step official --design A --climate power --subset 300 --years 2024 --out results/final2024/official
python -m dengue_recon.extensions.timing.run --design A --subset 300 --years 2024 \
    --models B1,B4s,M1-noONI,M1,S1,S2,S3,S3C,T1,S1-wy,S2-wy,S3-wy,S3-wyTest,S1-ns,S1-p25,S1-p50,S3-p25,S3-p50,T1-p25,T1-p50,S2C,S2C-noONI \
    --out results/final2024/timing
python -m dengue_recon.extensions.inla run --design A --subset 300 --climate power --years 2024 \
    --variants base,clim15,clim20,trends,clim15trends --threads 4 --out results/final2024/inla
```

## 2. Data checks done before the run

- **Cases (InfoDengue, 2024):** 300 municipalities × 52 epi weeks, no missing weeks, no
  provisional or flagged rows. 2024 is a record epidemic year (6.77 million notifications in the
  study municipalities, 3.5× 2023). Checked against the Ministry of Health's probable cases
  (OpenDengue V1.3.1, Admin2 weekly): the notified/probable ratio is 1.62 in 2024, inside the
  1.40–2.09 range of 2014–2023; weekly correlation median 0.98; weekly-shape JSD between the two
  sources median 0.016 (0.033–0.038 in 2022–2023). No sign of a reporting break.
- **Weather and ONI:** NASA POWER complete for all 300 municipalities and 52 weeks of 2024; ONI
  available through 2025.
- **Google Trends:** the 2009–2024 snapshot ended on 14 Jan 2024, so one more window
  (`2023-01-01` to `2025-01-15`) was downloaded on 24 Nov 2026 for Brazil and the 27 states
  (`dengue_<geo>_2023_2025.csv`). Only within-year shares enter the models, so the window's own
  0–100 scaling does not matter. `load_trends` keeps the earlier window on ties, so 2010–2023
  still come from the original files (checked: identical to before). The new window's 2023
  values were compared with the original snapshot's 2023 values as a check of Google's sampling
  (see `results/final2024/data_checks.md`).

## 3. What counts as confirmation

Each claim is a paired comparison on the same municipality-years of 2024 (annual total ≥ 100).
Change = relative difference in mean JSD. With a single test year there is no year-level
interval; the 95 % interval is the municipality bootstrap (2,000 replicates, seed 1).

| # | Claim from 2010–2023 | Contrast | Confirmed if |
|---|---|---|---|
| H1 | Pre-registered: climate beats the baseline | M1 − B4s | Reported as the pre-registered 2024 sign check. H1 is already *not supported* on 2010–2023, so 2024 cannot change that decision. |
| C1 | Google Trends improves the DM | S2 − B4s | mean < 0 and interval below 0 |
| C2 | Google Trends improves scaled INLA | INLA-scaled-trends − INLA-scaled | mean < 0 and interval below 0 |
| C3 | Weather adds nothing on top of Trends | S2C-noONI − S2 | interval includes 0 or mean > 0 |
| C4 | ONI makes the climate model worse | M1 − M1-noONI | mean > 0 |
| C5 | Annual-size terms help | B4s − B4 | mean < 0 and interval below 0 |
| C6 | Same-year timing beats the fixed shape | T1 − B4s | mean < 0 and interval below 0 |
| C7 | The gain comes from the right year (control) | S2-wy − S2 | mean > 0 and interval above 0 |
| C8 | Upper bound: state curve | S1 − B4s | mean < 0 and interval below 0 |

Final model under the rule written before 2024 (lowest design-A JSD, confirmed in design C,
the simpler of two within 1 %): **S2**. Its 2024 JSD is reported as the headline number.

Everything is reported whatever the outcome, including claims that fail. Secondary metrics
(peak-week error, onset error, EMD, coverage, CRPS) and results by region are descriptive.

## 4. Not part of this test

Mexico 2024 (needs DGE 2024 data and Mexican Trends for 2024), the learning curve and design C.
The one-fold run has no warm start from the 2023 fold (those checkpoints are not on this
machine); the optimiser starts from its default, which affects only convergence speed.
