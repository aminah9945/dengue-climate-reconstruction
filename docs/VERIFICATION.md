# Verification of the refactored code

The code in `src/dengue_recon/` is a reorganisation of the research code that produced the reported pilot results. The algorithms were not changed. This file records the checks that show the refactored package reproduces those results, and states what was **not** re-run.

**Environment:** Linux container, 4 CPUs, 15 GB RAM, no GPU; Python 3.13.16, jax/jaxlib 0.12.0, numpy 2.5.3, scipy 1.18.1, pandas 3.0.5, numpyro 0.22.0, pyreadr 0.5.7, matplotlib 3.11.2, pytest 9.1.1. Date: 2026-10-26.

**Inputs:** the processed data from the original project, read in place through `DATA_DIR` (never modified). The outputs were written to a scratch `RESULTS_DIR` and compared with the saved pilot outputs.

## (a) Unit tests

```
$ python -m pytest -q
....................                                                     [100%]
20 passed in about 5 s (no downloaded data needed)
```

Covered: largest-remainder allocation sums exactly (500 random cases); every Dirichlet-multinomial draw sums to N; JSD identity/symmetry/bounds/disjoint/zero weeks; year-effect cancellation in the softmax (max share difference < 1e-12; a week-varying term does change the shape); leakage (climatology, offsets and centring constants unchanged when the held-out year is blanked or perturbed, and the control that they *do* change when that year is included; full leakage audit in every synthetic fold); zero-climate M1 = B4s exactly on synthetic data; shuffles are derangements; 2024 lock and final-test path unit test; 53-week epi years (2014, 2020, 2025).

## (b) Real pipeline on the existing processed data

### Weekly, 60 municipalities, main models (B0, B1, B1p, B4, B4s, M1 and the test-only shuffle)

```
DATA_DIR=<original project>/05_experiment/data RESULTS_DIR=<scratch> \
python scripts/run_pipeline.py --step brazil --subset 60 --quick
```

Wall time 265 s (14 folds). Mean JSD over 840 municipality-years (B1p: 780), compared with the saved pilot file `metrics_detail_weekly_60.csv`:

| Model | JSD, refactored code | JSD, saved pilot | Reported in RESULTS.md | max \|row difference\| |
|---|---|---|---|---|
| B0 | 0.136568 | 0.136568 | 0.137 | 0 |
| B1 | 0.082017 | 0.082017 | 0.082 | 0 |
| B1p | 0.093542 | 0.093542 | 0.094 | 0 |
| B4 | 0.081666 | 0.081666 | 0.082 | 0 |
| **B4s** | **0.073505** | 0.073505 | 0.0735 | 0 |
| **M1** | **0.072669** | 0.072669 | 0.0727 | 0 |
| M1-shufTest | 0.076132 | 0.076132 | 0.0761 | 0 |

- Predicted shares: max |new − saved| = **0.0** for every model above (bit-identical).
- Fitted parameters (θ, φ) and L-BFGS iteration counts identical in all 14 folds for B4, B4s and M1.
- Peak-week error and EMD identical for every row. Interval coverage and CRPS identical too (B4 cov80 0.7956, CRPS 69.84; B4s 0.7998, 62.01; M1 0.7822, 58.60).
- Per-fold checks: zero-climate equivalence max difference 0.0 in all folds; 0 NaN climate anomalies; leakage audit passes in all 14 folds.

The required tolerance was about 1e-4 against RESULTS.md (B4s 0.0735, M1 0.0727, B1 0.082); the refactored code matches the saved results exactly.

### Rain extension, M1-R, weekly, 60 municipalities

`dengue_recon.extensions.rain_lags.main(60, ("M1-R",), ...)`, 74 s: mean JSD **0.073892** (saved 0.073892, reported 0.0739); max |Δshare| = 0.0; the extended leakage audit (including the new-feature audit) passes in all 14 folds.

### Reproduction of Lim et al. (2026), MAP

```
DATA_DIR=<original project>/05_experiment/data RESULTS_DIR=<scratch> \
python scripts/run_pipeline.py --step reproduce
```

The CV mask was read as data with pyreadr from the copy in the original project's `data/raw/lim_repo/` (taken from `OD_gap_filling_public` at commit `00ff7aeadcaa83f96be44fbc5197bedd88dc00cd`; MD5 `1443878503fb7aedfdb8862306931ef1`, identical to the file in a local clone of that repository at the same commit). No code from that repository was executed.

| | MAE (mean) | MAE (median) | RMSE (median) | 80 % coverage | CRPS | JSD |
|---|---|---|---|---|---|---|
| MAP, refactored code (3 folds pooled) | 10.5994 | 12.1775 | 68.335 | 0.9581 | 8.6626 | 0.1347 |
| MAP, saved pilot | 10.5994 | 12.1775 | 68.335 | 0.9581 | 8.6626 | 0.1347 |
| Published DM (NUTS, 3 reps) | – | 13.4 | 78.5 | 0.95 | 9.6 | – |

Per-fold MAE, coverage, CRPS and JSD are identical to 6 decimals. Cancellation check: 76 country-fold cases, max share difference 1.1e-16. Wall time 273 s; the first fold includes JAX compilation.

### Report regeneration from the committed files

`python scripts/run_pipeline.py --step report` on a copy of `results/` (14 s). The regenerated tables are identical to the committed ones (max absolute numeric difference 0): `table_main_weekly`, `table_main_monthly`, `table_paired_weekly`, `table_paired_monthly`, `table_by_region_weekly`, `table_gain_by_year_weekly`, `convergence_weekly_60`, and the rain-extension `table_main_weekly_60`, `table_paired_weekly_60`, `table_by_region_weekly_60`, `table_coefficients` (60). `sanity_checks.csv` reproduces every pass/fail and statistic (check 7: 3,457 rows, max |ΔJSD| 1.1e-16; final-test path unit test passes). Without the large files that are not committed, the 300-municipality tables and the example figure are skipped, as designed (`results/README.md`).

### Notebooks

All six notebooks (00–05) were executed with `jupyter nbconvert --execute` against a scratch copy of `results/` and the original processed data: 0 errors, no missing figures. Notebook 01 re-ran the MAP folds (same numbers as above); notebook 02's live demo gives JSD 0.153 (uniform) and 0.089 (seasonal profile) for the 2019 fold. The committed notebooks have their outputs cleared.

### Data download

A live smoke test of `dengue_recon.data.download` with a scratch `DATA_DIR` fetched the municipality metadata (5,571 rows), InfoDengue weekly data for Belo Horizonte 2023 (52 weeks, 65,991 cases, the same total as in the processed data) and one month of NASA POWER daily data. The full 300-municipality download (30–45 min) was not repeated.

## Official-experiment code (designs A/B/C, ONI, M2 early stopping), 2026-10-27

Same environment as above, plus xgboost 3.4.1. Processed data were read in place through `DATA_DIR` and not modified. Outputs went to a scratch `RESULTS_DIR`.

### Unit tests

`python -m pytest -q`: **37 passed** in about 7 s. These are the 23 earlier tests plus 14 in `tests/test_official.py`:

- designs A and B never train on the test year or a later year;
- learning-curve windows are exactly the k preceding years (8 + 10 + 12 folds);
- 2024 is refused by the fold builders and is not returned by the loaders (synthetic parquet files that contain 2024 and 2025);
- climatology, ONI mean/SD, offsets and centring do not change when non-training years are perturbed or blanked, and do change when a training year is perturbed;
- the official leakage audit passes in every fold of A, B and C;
- ONI is aligned by epi year/week, and its lag-block means are exact and identical across municipalities;
- the 20-feature M1 with zero climate coefficients equals B4s exactly;
- shuffles stay within the fold's years and equal the pilot's derangements when all years are used;
- M2's early-stopping metric includes the base margin, and its predictions are valid shares.

### Pilot regression: weekly LOYO, 60 municipalities, NASA POWER

| Run | B4s mean JSD | M1 (pilot) / M1-noONI mean JSD | max \|row ΔJSD\| vs saved pilot | Wall time |
|---|---|---|---|---|
| `--step brazil --subset 60 --quick` (pilot driver, after the loader changes) | **0.073505** | **0.072669** | 0 (840 rows; peak-week and EMD differences also 0) | 154 s |
| `--step official --design C --subset 60 --models B4s,M1-noONI` (new driver) | **0.073505** | **0.072669** | 0 (840 rows; peak-week 0, EMD ≤ 4.4e-16) | 125 s |

Both reproduce the reported pilot numbers exactly. In design C the zero-climate equivalence gave a maximum difference of 0.0 and the leakage audit passed in all 14 folds. Coverage and CRPS in the new driver use per-fold random seeds, so they are not compared with the pilot draw by draw.

### Resume

Two folds of design A (60 municipalities, all models) were run. One fold's log was then deleted and the command rerun. The rebuilt `metrics_detail` file was byte-identical (same MD5), including M2 with 4 threads.

### End-to-end check (not a result)

Designs A (12 folds) and B (30 folds) were run at 60 municipalities with NASA POWER and the models B1, B4s, M1 and M2, only to confirm that the code runs end to end. In every fold the leakage audit passed, zero-climate equivalence was 0.0, there were no missing climate anomalies, every L-BFGS fit converged, and no 2024 row was read. The metrics of this run are not reported.

The first attempt flagged the size-centring audit in folds with 7 or more panel years. The cause was a memory-layout artefact: the fold panel's arrays were not C-contiguous, so row sums differed by about 1e-15. It was not leakage. It was fixed (commit "Keep fold panel arrays C-contiguous"), and the audit now passes in all 56 folds of A, B and C at 300 municipalities.

## Official runs (step 4), 300 municipalities, 2010-2023, 2026-10-27

Run with the committed code, unchanged (no bug fixes were needed), `DATA_DIR` pointing at the processed data, 4 CPUs. Each run was started as a background process; its console log is in `results/official/logs/<tag>.log`. After each run, `python scripts/official_postcheck.py results/official/<tag>` re-checked it from the saved files and wrote `postcheck_<tag>.json`, `fold_summary_<tag>.csv` (metric means per fold and model, plus the N-weighted JSD) and `model_summary_<tag>.csv` (means per model over all scored municipality-years); for design B also `k_summary_<tag>.csv` (fold means averaged over the test years at each k). The exact-sum check recomputes the largest-remainder allocation from every saved prediction (`folds/*.npz`) and compares it with the annual total; it is also asserted during scoring. "No 2024" means no scored row, no fold test year and no fold panel year is 2024. `metrics_detail_<tag>.csv` (per row) and `folds/` stay local (gitignored); the commands in the README regenerate them.

Convergence: "success" is SciPy L-BFGS-B's own flag (gtol 1e-6, ftol 1e-14, maxiter 2000). The largest final gradient norms are of the same order as in the pilot's 300-municipality run (max 1.17 there). M2 rounds are the early-stopping best iteration + 1 on the inner validation year; a value of 1 means the validation loss did not improve after the first round, and the refit then uses 1 round, as pre-specified.

Consistency with the pilot: in design C the mean JSD of B0, B1, B4, B4s and M1-noONI over the 3,723 scored municipality-years equals the saved pilot 300-municipality LOYO table (`results/brazil/table_main_weekly_300.csv`, B0/B1/B4/B4s/M1) to within 3e-7, the rounding of the 6-decimal summary file.

| Run | Folds | Wall time | Leakage audit | Zero-climate max diff | Exact sums | 2024 rows | Missing climate | DM fits converged (max iterations / max \|grad\|) | M2 rounds (min-max) |
|---|---|---|---|---|---|---|---|---|---|
| 1. Design A, NASA POWER (primary), all models (`designA_power_300`) | 12/12 | 27 min 11 s | pass in 12/12 folds | 0.0 | 31940/31940 exact | none | 0 | 72/72 (846 / 0.78) | 1-454 (12 fits) |
| 2. Design B, NASA POWER, models B0, B1, B4s, M1, M2 (config default) (`designB_power_300`) | 30/30 | 27 min 54 s | pass in 30/30 folds | 0.0 | 40550/40550 exact | none | 0 | 60/60 (976 / 0.84) | 1-454 (30 fits) |
| 3. Design C (LOYO), NASA POWER, all models (`designC_power_300`) | 14/14 | 53 min 20 s | pass in 14/14 folds | 0.0 | 37230/37230 exact | none | 0 | 84/84 (1132 / 1.17) | 1-509 (14 fits) |

## Not verified in this pass

| Item | Why | How to run |
|---|---|---|
| Full weekly 60 pilot with ablations and 3 shuffle seeds (M1-short, M1-mult, B4s-eps1e-4, M1-shufTT) | about 22 min; the main models were verified bit-for-bit, and the ablations use the same functions | `--step brazil --subset 60` |
| Monthly 60 run | about 18 min; same code path with `weekly=False` | `--step brazil --res monthly --subset 60` |
| Weekly 300 run | about 41 min | `--step brazil --subset 300` |
| NUTS reproduction | about 20 min per fold | `--step reproduce --nuts` |
| M2 (XGBoost) | optional model | `--step m2` |
| M1-RTH and the 300-municipality rain runs | about 30 min | `--step extension --subset 300` |
| Full data download and rebuild | 30–45 min of API calls | `--step data` |
| `validate_sources` | needs the 1.1 GB Cox et al. file | `--step data --with-cox` |

These steps use the same refactored modules as the verified ones; their saved outputs in `results/` come from the original pilot runs.
