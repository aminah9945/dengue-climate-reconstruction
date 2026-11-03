# Data snapshots

Small external inputs that cannot be re-downloaded identically.

## google_trends/

Weekly Google Trends interest for the search term "dengue", downloaded on
3 Nov 2026 with `python -m dengue_recon.extensions.timing.trends_download`
(113 files from the cloud machine; the last 27, RR 2019-2024 and RS, SC, SE,
SP, TO, from Colab with notebook 08 after a rate limit, same day and code).
A copy of all 140 files is on Google Drive (`dengue_recon_data/google_trends`).

- Geos: `BR` and `BR-<UF>` for all 27 states. Files:
  `dengue_<geo>_<start>_<end>.csv` (columns `date, dengue`; `date` is the
  Sunday that starts the epi week).
- Windows: 4-year, overlapping. Each file has its own 0-100 scaling;
  `signals.load_trends` uses one window per (geo, year).
- Low-search states (e.g. RR, AP, AC, and SE before 2013) have many zero
  weeks and isolated spikes: Google rounds small volumes to 0. Treat their
  search signal as weak.
- Rows after 2023 are dropped before parsing (2024 is locked).

Google samples its Trends data, so a new download differs slightly. Use these
files to reproduce the timing-model results.
