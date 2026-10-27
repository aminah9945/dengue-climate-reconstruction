"""EXPLORATORY extension (not pre-registered): the base paper's INLA disaggregation
model ("unscaled" and "scaled" INLA) at municipality-week resolution, with and
without climate, on validation design A. See ``docs/INLA.md``.

* :mod:`.driver`  per-fold data export, R-INLA fit (``R/fit_inla.R``), scoring
* :mod:`.report`  paired comparisons with the official DM runs, tables, figures

CLI: ``python -m dengue_recon.extensions.inla {run,report} ...``
"""
