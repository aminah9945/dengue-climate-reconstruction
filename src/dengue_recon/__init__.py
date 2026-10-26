"""dengue_recon: climate-informed reconstruction of weekly dengue curves from annual totals.

Package layout (each sub-package is one step of the methodology; see
``docs/METHODOLOGY_MAPPING.md``):

- :mod:`dengue_recon.data`        Step 1  data acquisition and preparation
- :mod:`dengue_recon.base_paper`  Step 2  reproduction of Lim et al. (2026)
- :mod:`dengue_recon.baselines`   Step 3  non-model baselines (B0, B1, B1p)
- :mod:`dengue_recon.features`    Step 4  in-fold offsets, size terms, climate anomalies
- :mod:`dengue_recon.models`      Step 5  Dirichlet-multinomial ladder (B4, B4s, M1) and M2
- :mod:`dengue_recon.evaluation`  Step 6  LOYO folds, allocation, metrics, bootstraps
- :mod:`dengue_recon.checks`      Step 7  sanity checks and leakage audit
- :mod:`dengue_recon.extensions`  Step 8  pre-registered rain-lag extension
- :mod:`dengue_recon.reporting`   Step 9  tables and figures
- :mod:`dengue_recon.experiments` drivers that run Steps 3-7 over all folds
"""
__version__ = "0.1.0"
