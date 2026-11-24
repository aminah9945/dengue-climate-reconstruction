# Final test on 2024, Mexico (protocol)

**Written and committed on 24 Nov 2026, before any model saw Mexico 2024.** The Mexico work
(design C, 2017–2023) is exploratory. This single 2024 fold checks its findings on a year no
model, feature or report has used. Brazil's 2024 test is in `docs/FINAL_TEST_2024.md`.

## 1. What is run

One fold: **train 2017–2023, test 2024** (design A, fold `A_T2024_k7`). The units are the 32
states. The code, penalties, seeds, lags and evaluation rules are the same as in the Mexico
design-C runs (`results/extensions/mexico/`). Nothing is tuned on 2024.

| Run | Models | Output |
|---|---|---|
| Timing models | B1, B4s, M1-noONI, M1, S1, S2, S3, S3C, T1, the controls S1-wy and S2-wy, S2C, S2C-noONI | `results/final2024_mx/timing/` |
| Scaled INLA | base, trends, clim15trends | `results/final2024_mx/inla/` |

The model sets are the ones run in Mexico design C. The wrong-year controls and the two S2C
variants are added from the Brazil test.

```bash
export DENGUE_COUNTRY=MX DENGUE_RUN_FINAL_2024=1 DATA_DIR=C:/dr_data_mx_2024 \
       TIMING_TRENDS_DIR=data_snapshots/google_trends_mx PYTHONPATH=src
python -m dengue_recon.extensions.timing.run --design A --subset 300 --years 2024 \
    --models B1,B4s,M1-noONI,M1,S1,S2,S3,S3C,T1,S1-wy,S2-wy,S2C,S2C-noONI --out results/final2024_mx/timing
python -m dengue_recon.extensions.inla run --design A --subset 300 --climate power --years 2024 \
    --variants base,trends,clim15trends --threads 4 --out results/final2024_mx/inla
```

## 2. Data, and the checks done before the run

**Cases.** All of 2024 comes from the DGE open data year-end snapshot
`historicos/2024/datos_abiertos_dengue_191224.zip`, the last file of 2024; the next files
(January 2025) contain only 2025 onsets. The case definition, state and week rules are the
same as for 2023: every `ESTATUS_CASO`, `ENTIDAD_ASIG`, Sunday epi week of `FECHA_SIGN_SINTOMAS`.
The data folder is built with
`scripts/build_mexico_data.py --last-year 2024` into a separate folder (`C:/dr_data_mx_2024`).

- **Date format.** The 2024 snapshot writes dates as `dd/mm/yyyy`, while 2020–2023 use
  `yyyy-mm-dd`. The build script now parses both formats strictly. Checked: at least 99.9 % of
  onset dates parse, and the 2020–2023 snapshots are unaffected.
- **2017–2023 unchanged.** Cases, weather and the state table are identical to `C:/dr_data_mx`,
  the folder used by the design-C runs. The state table's totals exclude 2024.
- **Totals.** 2024 has 549,153 notified cases in all statuses, 2.0× 2023 (a record year), and
  123,141 confirmed cases. That is consistent with the published national figure of more than
  120,000 confirmed cases in 2024. 30 of the 32 states have ≥ 100 cases and are scored.
- **End-of-year truncation (known limitation).** The snapshot was updated on 16 Dec 2024, so
  week 50 is partial and weeks 51–52 have 0 cases. Weeks ≥ 50 hold 0.5 % of the 2024 cases,
  against 1.4–5.4 % in the complete years 2017–2020. The training years carry the same kind of
  truncation from their own year-end snapshots (2021 and 2022 weeks 50–51 are zero; 2023 week 52
  is zero). 2024 is therefore scored exactly as the data source reports it, consistent with
  training. This affects every model the same way.
- **Weather and ONI.** NASA POWER at the 32 state capitals: complete for all 52 weeks of 2024.
  ONI is available through 2025.
- **Google Trends.** The 2022–2024 window ended on 14 Jan 2024, so the `2023-01-01` to
  `2025-01-15` window was downloaded on 24 Nov 2026 (`dengue_<geo>_2023_2025.csv`). The same
  checks as for Brazil are run: 2010–2023 unchanged, the 2023 overlap, 2024 completeness, and
  2024 searches against 2024 cases. Results are in `results/final2024_mx/data_checks.md`.

## 3. What counts as confirmation

Each claim is a paired comparison on the same state-years of 2024 (annual total ≥ 100,
30 states). Change = relative difference in mean JSD. With one test year there is no
year-level interval. The 95 % interval is the state bootstrap (2,000 replicates, seed 1).
With 30 states the intervals are wide, so a claim that is not confirmed may simply be
underpowered.

Only contrasts that were *clearly* better or worse in Mexico design C become claims.

| # | Claim from Mexico 2017–2023 (design C) | Contrast | Design C | Confirmed if |
|---|---|---|---|---|
| C1 | Google Trends improves the DM | S2 − B4s | −3.1 % | mean < 0 and interval below 0 |
| C2 | Google Trends improves scaled INLA | INLA-scaled-trends − INLA-scaled | −6.9 % | mean < 0 and interval below 0 |
| C3 | Upper bound: regional curve | S1 − B4s | −6.1 % | mean < 0 and interval below 0 |
| C4 | Upper bound: regional curve + Trends | S3 − B4s | −7.9 % | mean < 0 and interval below 0 |
| C5 | A pure timing shift hurts in Mexico | T1 − B4s | +10.3 % | mean > 0 and interval above 0 |
| C6 | Weather + ONI hurt on top of S3 | S3C − S3 | +6.9 % | mean > 0 and interval above 0 |
| C7 | Control (from Brazil): Trends from the wrong year gives no gain | S2-wy − S2 | – | mean > 0 and interval above 0 |

Descriptive only (not clear in design C): M1 − B4s, M1-noONI − B4s, S2C-noONI − S2,
INLA-scaled − B4s, INLA-scaled-clim15trends − INLA-scaled-trends, INLA-scaled-trends − S2.

The final model under the same rule as Brazil: in design C, S2 and INLA + Trends + weather
tied (0.0815). The simpler model is **S2**, and its 2024 JSD is the headline number. INLA +
Trends + weather is reported next to it.

Everything is reported whatever the outcome, including claims that fail. Secondary metrics
are descriptive.

## 4. Not part of this test

Design C, the learning curve, and climate-study hypotheses (no pre-registered climate study
was run for Mexico).
