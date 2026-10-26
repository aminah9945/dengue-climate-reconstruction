# Part A: reproduction of Lim et al. (2026), Dirichlet-multinomial disaggregation

**Verdict: reproduced** on the metric definitions used in the published table, for repetition 1 only. Our own re-implementation of DM_v3_pooled (NumPyro NUTS, 2 chains × 500 warm-up + 500 draws, target_accept 0.95, interpolation mask rep01, 3 folds, 3,648 test country-months) gives MAE 13.34, RMSE 78.1, 80 % coverage 0.948 and CRPS 9.53. The published values are MAE 13.4, RMSE 78.5, coverage 0.95 and CRPS 9.6, pooled over 3 repetitions with 4 chains × 1,000 draws. MAE is within 0.5 % and coverage within 0.002, so both acceptance criteria are met (±10 % and ±0.03).

**How the published MAE is defined.** The repository's table script (`CV_tab1_final_models_performance.R`) reports the MAE and RMSE of the posterior-predictive **median**, so that is what we compare against. Measured on the posterior-predictive **mean**, our numbers are MAE 11.69 and RMSE 76.0. Both versions are in `table_repro.csv` (`MAE`/`RMSE` use the mean; `MAE_median`/`RMSE_median` use the median).

**MAP version.** The MAP fit (L-BFGS on the same non-centred log posterior, 3 restarts) takes about 8 s per fold, against about 20 min per fold for NUTS. It gives a median-based MAE of 12.18, coverage 0.958 and CRPS 8.66. Its mode puts most of the country-level deviation scale on `tau_country_month` (about 1.58, against a posterior median of 0.047 in the paper). This is a known artefact of taking the joint mode in non-centred hierarchical models, but the predicted shares are almost the same as with NUTS: mean JSD is 0.135 for both.

**Expected finding, confirmed.**
- On the same test rows, the DM point error is close to that of the historical-mean-share reference: mean-based MAE 11.69 (DM, NUTS) vs 11.41 (historical mean), and JSD 0.135 vs 0.132.
- The added value of the DM is in its probabilistic output: 80 % coverage of 0.95 and a CRPS of 9.5.

**Cancellation check.** The country-year effect is constant across months, so it cancels in the softmax. For every country with two or more test years in a fold (76 country-fold cases), the predicted mean share vectors of its different test years are identical to within 1.1e-16, for both MAP and NUTS (`cancellation_check.csv`). The base model therefore gives every held-out year of a country the same shape. This is the gap that the week-varying climate terms in Parts B and C are designed to fill.

**Diagnostics.** NUTS had 0 divergences. Maximum R-hat over the key hyperparameters was 1.015 and minimum bulk ESS was 130 (`fit_diagnostics.csv`). Wall time was 18.5–21 min per fold, with the 3 folds run in parallel on 4 CPUs. Posterior medians of the hyperparameters were not saved in this run.

**Differences from the published run.**
- One repetition instead of three.
- 2 chains × 500 draws instead of 4 × 1,000.
- target_accept 0.95 instead of 0.99.
- Sequential chains on CPU.
- NumPyro directly instead of PyMC → NumPyro. The multinomial coefficient is a constant, so this does not change the posterior.
