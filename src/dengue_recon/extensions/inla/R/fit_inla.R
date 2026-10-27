# ============================================================================
# INLA disaggregation model (base paper's "scaled INLA"), municipality-week
# version. EXPLORATORY extension, not pre-registered.
#
# Called by dengue_recon.extensions.inla.driver once per fold and variant:
#   Rscript fit_inla.R <fold_dir> <variant> <n_samples> <seed> <n_threads> [<theta_file>]
# <theta_file> (optional): hyperparameter mode of an earlier fit used as the
# starting value of the optimiser (warm start; the result is re-optimised).
#
# Input  <fold_dir>/data.csv  one row per municipality-week:
#          row, muni, region, slot (1..53), tseq (continuous week, 1..), year (1..),
#          y (count; NA for the held-out test year and for non-training rows),
#          test (1 = held-out test-year row to predict), x01..x20 (climate features)
#        <fold_dir>/meta.json is not read here (the driver keeps it).
# Output <fold_dir>/<variant>_pred.csv     test rows: fitted mean / median, eta mean,
#                                           AR1(time) mean, seasonal sum, climate part
#        <fold_dir>/<variant>_hyper.csv    hyperparameter + fixed-effect summaries
#        <fold_dir>/<variant>_eta.bin      posterior linear-predictor draws (float64,
#                                           n_test x n_samples, column-major)
#        <fold_dir>/<variant>_size.bin     NB size draws (float64, n_samples)
#        <fold_dir>/<variant>_theta.txt    hyperparameter mode (internal scale)
#        <fold_dir>/<variant>_done.json    run info (written last = checkpoint)
#
# Model (base paper, script/CV/00_imp_model_inla_spec.R, inla_m_hier_shared_formula,
# with country -> municipality, latitudinal band -> Brazilian region, month -> 53
# epi-week slots):
#   y ~ NegBin(mu, size)
#   log mu = b0 + AR1(time | muni) + RW1cyc(slot) + RW1cyc(slot | region)
#            + RW1cyc(slot | muni) + AR1(year | muni) + iid(muni) [+ climate' beta]
# Same PC priors, NB size prior, simplified Laplace + empirical Bayes as the base
# paper. Computational change (docs/INLA.md): the base paper puts a sum-to-zero
# constraint on every replicate of every term. With 300 municipalities that is
# ~900 dense linear constraints, which did not fit in 14 GB of memory. We keep the
# constraint on AR1(time | muni), which matters for the forecast, and drop it on
# AR1(year | muni) (constant within a year, so it cancels in the scaled shape) and
# on the municipality RW1 deviations (made proper with a small diagonal 0.01
# instead). On a 60-municipality fold this changed the predicted shapes by a
# JSD of 4e-5 (mean) relative to the fully constrained fit. With 300
# municipalities the 300 remaining constraints still need ~6 GB and ~2 s per
# optimiser step, so the driver fits 300 municipalities in 5 region-stratified
# batches of 60 (each batch has its own national/regional seasonal curves).
# "Scaled": predictions of a municipality-year are rescaled to its known annual
# total (done in Python, per point prediction and per posterior draw).
# ============================================================================

suppressPackageStartupMessages({
  library(INLA)
  library(data.table)
})

args <- commandArgs(trailingOnly = TRUE)
fold_dir <- args[1]
variant <- args[2]                       # "base", "clim20" or "clim15"
nsamp <- as.integer(args[3])
seed <- as.integer(args[4])
nthr <- as.integer(args[5])
theta_file <- if (length(args) >= 6) args[6] else "none"

t0 <- Sys.time()
set.seed(seed)
INLA::inla.setOption(num.threads = sprintf("%d:1", nthr))

d <- fread(file.path(fold_dir, "data.csv"))
setorder(d, row)
n_time <- max(d$tseq)
n_year <- max(d$year)
n_muni <- max(d$muni)
n_reg <- max(d$region)

# copies of the slot index for the three seasonal levels (as in the base paper)
d[, slot_shared := slot]
d[, slot_dev := slot]
d[, slot_dev_muni := slot]

pc_ar1 <- function(prec_u) list(
  rho = list(prior = "pc.cor1", param = c(0.8, 0.9)),
  prec = list(prior = "pc.prec", param = c(prec_u, 0.01))
)

form <- y ~ 1 +
  # within-municipality serial dependence over weeks (continuous week index)
  f(tseq, model = "ar1", replicate = muni, constr = TRUE, values = seq_len(n_time),
    hyper = pc_ar1(2)) +
  # national weekly seasonal backbone (cyclic RW1 over 53 slots)
  f(slot_shared, model = "rw1", cyclic = TRUE, constr = TRUE, scale.model = TRUE,
    values = 1:53, hyper = list(prec = list(prior = "pc.prec", param = c(2, 0.01)))) +
  # region deviations from the national curve
  f(slot_dev, model = "rw1", cyclic = TRUE, constr = TRUE, scale.model = TRUE,
    replicate = region, values = 1:53,
    hyper = list(prec = list(prior = "pc.prec", param = c(0.7, 0.01)))) +
  # municipality deviations from the region curve (strong shrinkage)
  f(slot_dev_muni, model = "rw1", cyclic = TRUE, constr = FALSE, diagonal = 0.01,
    scale.model = TRUE, replicate = muni, values = 1:53,
    hyper = list(prec = list(prior = "pc.prec", param = c(0.25, 0.01)))) +
  # slow across-year structure per municipality
  f(year, model = "ar1", replicate = muni, constr = FALSE, values = seq_len(n_year),
    hyper = pc_ar1(1)) +
  # municipality intercepts
  f(muni, model = "iid", constr = TRUE, values = seq_len(n_muni),
    hyper = list(prec = list(prior = "pc.prec", param = c(0.5, 0.01))))

xcols <- character(0)
if (variant == "clim20") xcols <- sprintf("x%02d", 1:20)
if (variant == "clim15") xcols <- sprintf("x%02d", 1:15)
if (length(xcols)) {
  form <- update(form, as.formula(paste(". ~ . +", paste(xcols, collapse = " + "))))
}

cmode <- list()
if (theta_file != "none" && file.exists(theta_file)) {
  cmode <- list(theta = scan(theta_file, quiet = TRUE), restart = TRUE)
}

fit <- inla(
  form, data = as.data.frame(d),
  family = "nbinomial",
  control.family = list(hyper = list(size = list(prior = "loggamma", param = c(5, 0.1)))),
  # climate coefficients: N(0, 1) per standardised lag-block feature (extension only)
  control.fixed = list(mean = 0, prec = 1, mean.intercept = 0, prec.intercept = 0.001),
  control.predictor = list(compute = TRUE, link = 1),
  control.compute = list(dic = FALSE, waic = FALSE, config = TRUE),
  control.inla = list(strategy = "simplified.laplace", int.strategy = "eb"),
  control.mode = cmode,
  verbose = FALSE
)
t_fit <- as.numeric(difftime(Sys.time(), t0, units = "secs"))

te <- which(d$test == 1)
fv <- fit$summary.fitted.values
lp <- fit$summary.linear.predictor
rt <- fit$summary.random
ar <- rt$tseq$mean[(d$muni[te] - 1) * n_time + d$tseq[te]]
seas <- rt$slot_shared$mean[d$slot[te]] +
  rt$slot_dev$mean[(d$region[te] - 1) * 53 + d$slot[te]] +
  rt$slot_dev_muni$mean[(d$muni[te] - 1) * 53 + d$slot[te]]
clim <- rep(0, length(te))
if (length(xcols)) {
  b <- fit$summary.fixed[xcols, "mean"]
  clim <- as.numeric(as.matrix(d[te, ..xcols]) %*% b)
}
pred <- data.table(row = d$row[te], fit_mean = fv$mean[te], fit_median = fv$`0.5quant`[te],
                   eta_mean = lp$mean[te], ar1_time = ar, seasonal = seas, climate = clim)
fwrite(pred, file.path(fold_dir, paste0(variant, "_pred.csv")))

hy <- fit$summary.hyperpar
hyper <- rbind(
  data.table(parameter = rownames(fit$summary.fixed), type = "fixed", mean = fit$summary.fixed$mean,
             q025 = fit$summary.fixed$`0.025quant`, q50 = fit$summary.fixed$`0.5quant`,
             q975 = fit$summary.fixed$`0.975quant`),
  data.table(parameter = rownames(hy), type = "hyper", mean = hy$mean, q025 = hy$`0.025quant`,
             q50 = hy$`0.5quant`, q975 = hy$`0.975quant`)
)
fwrite(hyper, file.path(fold_dir, paste0(variant, "_hyper.csv")))
writeLines(format(fit$mode$theta, digits = 10), file.path(fold_dir, paste0(variant, "_theta.txt")))

t1 <- Sys.time()
if (nsamp > 0) {
  samp <- inla.posterior.sample(nsamp, fit, seed = seed, selection = list(Predictor = te),
                                num.threads = "1:1")
  eta <- vapply(samp, function(z) as.numeric(z$latent[, 1L]), numeric(length(te)))
  hn <- names(samp[[1]]$hyperpar)
  i_size <- grep("size", hn, ignore.case = TRUE)[1]
  size <- vapply(samp, function(z) as.numeric(z$hyperpar[i_size]), numeric(1))
  con <- file(file.path(fold_dir, paste0(variant, "_eta.bin")), "wb")
  writeBin(as.numeric(eta), con, size = 8); close(con)
  con <- file(file.path(fold_dir, paste0(variant, "_size.bin")), "wb")
  writeBin(as.numeric(size), con, size = 8); close(con)
}
t_samp <- as.numeric(difftime(Sys.time(), t1, units = "secs"))

ok <- is.null(fit$ok) || isTRUE(fit$ok)
info <- sprintf(
  paste0('{"variant": "%s", "n_rows": %d, "n_obs": %d, "n_test": %d, "n_samples": %d, ',
         '"fit_s": %.1f, "sample_s": %.1f, "inla_version": "%s", "mode": "%s", "ok": %s, ',
         '"mlik": %.4f, "warm_start": %s, "fn_calls": %d}'),
  variant, nrow(d), sum(!is.na(d$y)), length(te), nsamp, t_fit, t_samp,
  as.character(packageVersion("INLA")), as.character(fit$.args$inla.mode), tolower(as.character(ok)),
  fit$mlik[1, 1], tolower(as.character(length(cmode) > 0)),
  as.integer(sum(fit$misc$nfunc, na.rm = TRUE))
)
writeLines(info, file.path(fold_dir, paste0(variant, "_done.json")))
cat(info, "\n")
