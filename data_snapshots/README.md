# Data snapshots

Small external inputs that cannot be re-downloaded identically.

## google_trends/

Weekly Google Trends interest for the search term "dengue", downloaded on
3 Nov 2026 with `python -m dengue_recon.extensions.timing.trends_download`.

- Geos: `BR` and `BR-<UF>` for all 27 states. Files:
  `dengue_<geo>_<start>_<end>.csv` (columns `date, dengue`; `date` is the
  Sunday that starts the epi week).
- Windows: 4-year, overlapping. Each file has its own 0-100 scaling;
  `signals.load_trends` uses one window per (geo, year).
- `empty_geos.txt`: geo-windows for which Google returned no data (too few
  searches); the model uses the national series there.
- Rows after 2023 are dropped before parsing (2024 is locked).

Google samples its Trends data, so a new download differs slightly. Use these
files to reproduce the timing-model results.
