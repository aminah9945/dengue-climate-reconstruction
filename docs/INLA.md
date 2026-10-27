# INLA disaggregation model (base paper's "scaled INLA") — EXPLORATORY

**Status: exploratory extension, not pre-registered.** It does not change the pre-registered H1–H4 results in `results/official/`. Development and comparison use 2010–2023 only; `RUN_FINAL_2024` stays `False` and 2024 is never loaded.

## 1. Why

For annual-to-monthly disaggregation, the base paper (Lim et al. 2026) chose **scaled INLA** (Supplementary Table 5: interpolation MAE 10.7 vs 13.5 for the Dirichlet-multinomial). Our study extends the Dirichlet-multinomial (DM) instead (see `docs/DEVIATIONS.md` B3). This extension fits the base paper's best model on our data so that B4s and M1 can be compared with it directly.

## 2. Model

Source: base-paper repository, `script/CV/00_imp_model_inla_spec.R` (`inla_m_hier_shared_formula`, "Model 4", the disaggregation model of `03_down_inla_full_repeatedCV_jupyter.R`), and the supplement (INLA candidates; Supplementary Tables 5 and 12). Our version (`src/dengue_recon/extensions/inla/R/fit_inla.R`) maps country → municipality, latitudinal band → Brazilian region and month → 53 epi-week slots:

```
y[m,w] ~ NegBin(mu, size)                      size prior: loggamma(5, 0.1)
log mu = b0
       + AR1(time | municipality)              continuous week index; PC priors rho (0.8, 0.9), prec (2, 0.01); sum-to-zero
       + RW1cyc(slot)                          national weekly backbone, 53 slots; PC prec (2, 0.01)
       + RW1cyc(slot | region)                 region deviations; PC prec (0.7, 0.01)
       + RW1cyc(slot | municipality)           municipality deviations; PC prec (0.25, 0.01)
       + AR1(year | municipality)              PC rho (0.8, 0.9), prec (1, 0.01)
       + iid(municipality)                     PC prec (0.5, 0.01)
       [+ climate' beta]                       extension only: 20 (M1) or 15 (M1-noONI) features, beta ~ N(0, 1)
```

Inference is as in the base paper: `strategy = "simplified.laplace"`, `int.strategy = "eb"`, `config = TRUE`, and R-INLA 26.8.7 in its default compact mode. The weeks of the held-out year are in the data with `y = NA`, and their posterior means are the "unscaled" prediction. **Scaled INLA** rescales a municipality-year's predicted weekly counts so that they add up to its known annual total (the base paper's `add_rebased_point_predictions`). We then round with the largest-remainder rule, as for every other model. Uncertainty: 500 posterior draws of the linear predictor and the NB size (`inla.posterior.sample`), then negative-binomial counts. For scaled INLA each draw is rescaled to N (the base paper's `rebase_draws_to_targets`) and integer-allocated, so every draw sums to N exactly.

**Models in `metrics_detail`:**

| Model | Description |
|---|---|
| `INLA-unscaled` | Posterior mean (counts do not sum to N). Shape metrics use the normalised curve; `mae_counts`, coverage and CRPS use raw counts. |
| `INLA-scaled` | The same curve rescaled to N. **This is the base paper's model.** |
| `INLA-scaled-clim20` | Scaled INLA plus the 20 M1 features (15 local lag-block anomalies + 5 ONI). |
| `INLA-scaled-clim15` | Scaled INLA plus the 15 local features only (as M1-noONI). |
| `INLA-seasonal-only` | Diagnostic only: exp(seasonal terms) of the same fit, without the AR1(time) carry-over. |

Rescaling by a constant does not change shape. So `INLA-unscaled` and `INLA-scaled` have identical JSD, peak-week error, EMD and MAE on shares; they differ only in the count metrics.

### Deviations from the base paper (computational)

- **Sum-to-zero constraints.** The base paper constrains every replicate of every term. With 300 municipalities that adds about 900 dense linear constraints, and the fit ran out of the 14 GB of memory on the first optimiser step. We keep the constraint on AR1(time | municipality), which governs the forecast into the held-out year. We drop it on:
  - AR1(year | municipality), which is constant within a year and so cancels after scaling;
  - the municipality RW1 deviations, which are made proper with `diagonal = 0.01` instead.

  On the 60-municipality 2012 fold the predicted shapes moved by a mean JSD of 0.00004 (max 0.00015) relative to the fully constrained fit, and the hyperparameters were almost unchanged.

  Dropping the AR1(time) constraint as well would be 6× faster, but it changed the shapes noticeably (mean JSD 0.0017; JSD to the truth 0.0884 instead of 0.0859). It was therefore not used.
- **Batches (300 municipalities).** Even the reduced set of 300 constraints needs about 6.5 GB and about 2 s per optimiser step at k = 2. The 300 municipalities are therefore fitted in 5 region-stratified batches of 60 (12 per region; `municipality_batches`). Each batch has its own national and regional curves and hyperparameters. The 60-municipality run is one batch.
- **Warm starts.** The optimiser starts from the hyperparameter mode of the previous fold (same batch and variant), or, for the climate variants, from the same fold's base fit. It then re-optimises to convergence.
- **Explicit index sets** (`values = 1:n`) on the AR1 and RW1 terms, so that a week or year with no observations is never treated as adjacent to the next one.
- **Rows.** Training rows follow the official rules: `exclude_suggested` is false and N ≥ 20, in the fold's training years. Test rows are the evaluated municipality-years of the test year (N ≥ 100). Training years outside these rules are simply absent.

## 3. Design and leakage

The design is the official design A: an expanding window, test T = 2012 … 2023, train on 2010 … T−1, at 300 municipalities (plus a 60-municipality run). Everything else is shared with `experiments/official.py`:

- the fold panel contains no year after T;
- climatology and ONI scaling use 2009 plus the training years;
- the climate features are those of M1;
- the official leakage audit runs in every fold and passed in all folds;
- every test-year count is NA in the INLA data (checked per fold);
- scoring uses `score_rows` with the same per-fold seeds, so the `metrics_detail` rows pair one-to-one with the official ones on (fold_id, geocode). At 300 municipalities the noise-floor draws are also identical.

The paired statistics reuse `reporting/official_report.contrast`: municipality-cluster and year-block bootstraps with 2,000 replicates.

## 4. Running

```bash
export DATA_DIR=...                      # processed data (as for the official runs)
python -m dengue_recon.extensions.inla run --subset 60
python -m dengue_recon.extensions.inla run --subset 300 --variants base          # resumable, per-fold checkpoints
python -m dengue_recon.extensions.inla run --subset 300                          # adds clim20, clim15
OFFICIAL_DIR=<folder with designA_power_300/metrics_detail_*.csv and folds/> \
python -m dengue_recon.extensions.inla report --subset 300
```

Requirements: R ≥ 4.3 with `INLA` (from `https://inla.r-inla-download.org/R/stable`), `data.table` and `sn`.

Outputs:

- `results/extensions/inla/designA_power_<n>/`: `metrics_detail_inla_*`, `shape_diag_inla_*`, `fold_log_inla_*` and `checks_inla_*`. The per-fold checkpoints and the per-row files at 300 are not committed.
- `results/extensions/inla/tables/*_<n>.csv` and `results/extensions/inla/figures/*_<n>.png`.

## 5. Results, 60 municipalities (720 municipality-years)

The DM rows are from the official 300-municipality run, restricted to the same 60 municipalities (the DM models were fitted on 300).

| Model | JSD | 95 % CI (muni) | N-weighted JSD | Peak error (weeks) | EMD | 80 % coverage | CRPS |
|---|---|---|---|---|---|---|---|
| B1 | 0.0864 | 0.0780–0.0950 | 0.0853 | 6.10 | 4.87 | – | – |
| B4 | 0.0841 | 0.0757–0.0926 | 0.0852 | 6.11 | 4.75 | 0.823 | 69.8 |
| B4s | 0.0782 | 0.0703–0.0868 | 0.0710 | 6.36 | 4.36 | 0.811 | 65.7 |
| M1-noONI | 0.0793 | 0.0714–0.0879 | 0.0671 | 6.21 | 4.42 | 0.785 | 62.9 |
| M1 | 0.0810 | 0.0730–0.0896 | 0.0698 | 6.10 | 4.42 | 0.772 | 64.7 |
| INLA-unscaled | 0.0790 | 0.0722–0.0861 | 0.0753 | 5.91 | 4.60 | 0.730 | 110.8 |
| **INLA-scaled** | **0.0790** | 0.0722–0.0861 | 0.0753 | 5.91 | 4.60 | 0.710 | 67.2 |
| INLA-scaled-clim20 | 0.0794 | 0.0726–0.0863 | 0.0747 | 5.80 | 4.61 | 0.701 | 67.0 |
| INLA-scaled-clim15 | 0.0784 | 0.0716–0.0853 | 0.0731 | 5.86 | 4.60 | 0.707 | 65.8 |
| INLA-seasonal-only | 0.0826 | 0.0751–0.0907 | 0.0791 | 5.85 | 4.80 | – | – |

Paired JSD differences (a − b; negative means a is better), with 95 % CIs from the municipality and year-block bootstraps:

| Contrast | Difference | Relative | CI (municipality) | CI (year) | Years a better |
|---|---|---|---|---|---|
| INLA-scaled − B4s | +0.0008 | +1.0 % | −0.0025 to +0.0045 | −0.0063 to +0.0076 | 6/12 |
| INLA-scaled − M1 | −0.0020 | −2.4 % | −0.0055 to +0.0019 | −0.0090 to +0.0040 | 5/12 |
| INLA-scaled − M1-noONI | −0.0003 | −0.4 % | −0.0041 to +0.0038 | −0.0077 to +0.0062 | 5/12 |
| INLA-scaled − B4 | −0.0051 | −6.0 % | −0.0075 to −0.0026 | −0.0092 to −0.0008 | 9/12 |
| INLA-scaled − B1 | −0.0074 | −8.5 % | −0.0101 to −0.0047 | −0.0108 to −0.0036 | 10/12 |
| INLA climate 20 − INLA-scaled | +0.0004 | +0.5 % | −0.0004 to +0.0012 | −0.0017 to +0.0028 | 7/12 |
| INLA climate 15 − INLA-scaled | −0.0006 | −0.8 % | −0.0013 to +0.0001 | −0.0025 to +0.0010 | 5/12 |
| INLA-scaled − INLA-seasonal-only | −0.0036 | −4.3 % | −0.0056 to −0.0015 | −0.0067 to +0.0002 | 9/12 |

Peak-week error: INLA-scaled − B4s = −0.45 weeks (CI municipality −0.89 to −0.03; year −0.79 to −0.07).

At 60 municipalities scaled INLA is as good as B4s on JSD (no detectable difference) and clearly better than the base paper's DM structure B4 and than the seasonal profile B1. It has a smaller peak-week error than B4s. It is more stable across years: early folds are better, late folds worse (see the figure). Adding climate to INLA changes nothing detectable, which matches the DM result. INLA's 80 % intervals under-cover (0.71), as in the base paper (0.87 scaled monthly), and less well than the DM (0.77–0.82).

## 6. Does INLA's shape vary by held-out year?

Yes, but only through the AR1(time) carry-over, and only weakly.

- In the DM, the year effect is constant within a year and cancels in the softmax. In INLA, the AR1(year) effect cancels the same way after scaling.
- The AR1(time) term does not cancel. In design A the held-out year comes after the training years, so its AR1 posterior is a forecast: it starts from the state of the last training weeks (late T−1) and decays towards zero with ρ ≈ 0.97 per week (half-life about 22 weeks; posterior ρ 0.967–0.972 in every fold).
- The forecast's growing variance also tilts the posterior mean exp(η) slightly upward later in the year.

So the predicted shape of year T depends on the epidemic state at the end of T−1, which is information that is specific to the held-out year. It does not depend on anything that happens within T itself.

Size of the effect (`tables/shape_variation_60.csv`, `figures/fig_inla_shape_variation_60.png`):

- JSD between the full prediction and the seasonal-only shape of the same fit: mean **0.0071** per municipality-year (by test year from 0.0033 in 2017 to 0.0165 in 2022).
- The predicted peak moves in 42 % of municipality-years, by 1.5 weeks on average.
- The carry-over moves the shape towards the truth in 58 % of municipality-years. Paired, it lowers JSD by 4.3 % (CI municipality −0.0056 to −0.0015; year −0.0067 to +0.0002).
- Year-to-year change of the *predicted* shape of the same municipality (mean JSD between consecutive test years, `tables/shape_year_to_year_60.csv`):

  | Shape | Mean JSD |
  |---|---|
  | INLA-seasonal-only | 0.0009 |
  | B4 | 0.0036 |
  | INLA-scaled | 0.0069 |
  | INLA-scaled-clim20 | 0.0102 |
  | B4s | 0.0290 |
  | M1 | 0.0472 |
  | **observed curves** | **0.1441** |

  So INLA's year-specific variation is about twice that of the DM without size terms, but much smaller than that of B4s (whose annual-size terms change the shape) and M1, and a small fraction of the real year-to-year variation.
- The municipality seasonal deviations are shrunk strongly by the base paper's prior (posterior precision 180–3,800). Municipality-specific timing is therefore carried mostly by the regional curve plus the AR1 carry-over.

## 7. Results, 300 municipalities

See section 8 (added when the 300-municipality run finished).

## 8. Runtime

- **60 municipalities**, 12 folds × 3 variants, 4 CPUs: about 81 min in total (fits 73 min, 500 posterior draws per fit 10 min).
- **300 municipalities**: see section 7.
