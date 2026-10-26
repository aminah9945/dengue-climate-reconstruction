# Data

**No data files are stored in this repository.** All inputs are public and are downloaded from the original sources by Step 1 of the pipeline into `DATA_DIR` (default: this `data/` folder; it is git-ignored apart from this README):

```bash
python scripts/run_pipeline.py --step data              # about 30-45 min, resumable
python scripts/run_pipeline.py --step data --with-cox   # + optional Cox et al. cross-check (1.1 GB download)
```

or `notebooks/00_setup_and_data.ipynb` in Colab. Each download is cached under `DATA_DIR/raw/`, so an interrupted run continues where it stopped. Set `DATA_DIR=/path/to/folder` to keep the data elsewhere (for example on Google Drive).

## Sources

| Source | What we use | URL | Licence / terms | How it is downloaded |
|---|---|---|---|---|
| **InfoDengue** (Codeço et al. 2018), based on SINAN notifications of the Brazilian Ministry of Health | Weekly notified dengue cases (`casos`) per municipality and epi week, 2010 W1 – 2025 W53; InfoDengue's own weekly temperature and humidity (check only) | API: `https://info.dengue.mat.br/api/alertcity?geocode=<IBGE code>&disease=dengue&format=csv&ew_start=1&ew_end=53&ey_start=2010&ey_end=2025` | Public surveillance aggregates; the API states no explicit licence. Cite InfoDengue and SINAN | `dengue_recon.data.download.download_infodengue`, one CSV per municipality (700 candidates), about 2 s each |
| **NASA POWER** (Prediction Of Worldwide Energy Resources), MERRA-2 based, about 0.5° × 0.625° grid | Daily `PRECTOTCORR` (bias-corrected precipitation), `T2M`, `T2M_MAX`, `T2M_MIN`, `RH2M` at the municipal seat, 2009-01-01 – 2025-12-31 | API: `https://power.larc.nasa.gov/api/temporal/daily/point?parameters=PRECTOTCORR,T2M,T2M_MAX,T2M_MIN,RH2M&community=AG&longitude=<lon>&latitude=<lat>&start=20090101&end=20251231&format=CSV` | NASA open data; free to use. Acknowledge the NASA Langley Research Center POWER Project | `download_power`, one request per selected municipality (3 parallel workers) |
| **IBGE** SIDRA table 6579 | 2021 population estimates (used to define the 700-candidate pool) | `https://apisidra.ibge.gov.br/values/t/6579/n6/all/v/all/p/2021` | Public data of the Brazilian Institute of Geography and Statistics; cite IBGE | `download_meta` |
| **kelvins/municipios-brasileiros** (GitHub) | IBGE codes, names, latitude/longitude of municipal seats, capital flag | `https://raw.githubusercontent.com/kelvins/municipios-brasileiros/main/csv/municipios.csv` | MIT licence | `download_meta` |
| **Cox et al. (2024)**, Zenodo, optional | SINAN-derived weekly dengue and ERA5-Land climate, all municipalities, 2013–2020; used only to validate annual totals and rainfall | `https://doi.org/10.5281/zenodo.13286671` (file `denv_report.RDS`) | CC BY 4.0 | `download_cox` (streams 1.1 GB, keeps a 20-column parquet subset, deletes the RDS); `validate_sources.main` |
| **Lim et al. (2026)** base-paper repository | Interpolation CV mask (rep01) and the published CV table, for the reproduction only | `https://github.com/ahyoung-lim/OD_gap_filling_public`, commit `00ff7aeadcaa83f96be44fbc5197bedd88dc00cd` | Licence "to be specified" in its README: the files are read as data and are not redistributed; its code is never executed | `dengue_recon.base_paper.fetch.fetch_lim_mask` (blob-less clone, copies 2 files into `raw/lim_repo/`) |

### How to cite the data

- InfoDengue: C. Codeço, F. Coelho, O. Cruz, S. Oliveira, T. Castro, L. Bastos. *Infodengue: A nowcasting system for the surveillance of arboviruses in Brazil.* Revue d'Épidémiologie et de Santé Publique 66:S386, 2018. https://doi.org/10.1016/j.respe.2018.05.408 . Underlying data: SINAN, Ministério da Saúde, Brazil.
- NASA POWER: "These data were obtained from the NASA Langley Research Center (LaRC) POWER Project funded through the NASA Earth Science/Applied Science Program." https://power.larc.nasa.gov/ . MERRA-2: Gelaro et al. 2017, J. Climate 30:5419–5454.
- IBGE: Instituto Brasileiro de Geografia e Estatística, SIDRA table 6579 (population estimates 2021).
- Municipal seats: kelvins/municipios-brasileiros, https://github.com/kelvins/municipios-brasileiros (MIT).
- Cox et al.: V. M. Cox. *Dengue, chikungunya, and Zika case notification data from Brazil at the weekly- and municipality-level over 2013–2020, with matched socioeconomic, human connectivity, climate, and meteorological conditions.* Zenodo, 2024. https://doi.org/10.5281/zenodo.13286671
- Base paper: A. Lim, K. Joshi, P. Lakshminarayanan, K. M. Susong, J. Clarke, et al., O. J. Brady. *Rapid global growth and intensification of dengue since 1990.* Research Square preprint, 2026. https://doi.org/10.21203/rs.3.rs-8976788/v1

## Processing (Step 1)

1. `download.py meta` – municipality table (5,571 municipalities) with region from the IBGE state code.
2. `download.py infodengue` – InfoDengue for the 700 most populous municipalities plus all state capitals.
3. `download.py select` – rank candidates by total notifications 2010–2025; at least 45 per macro-region, remaining slots to the highest burden nationally → **300 municipalities in all 27 states** (Center-West 45, North 45, Northeast 45, South 45, Southeast 120).
4. `download.py power` – NASA POWER daily data for the 300 municipal seats.
5. `build_dataset.py` – epi-week calendar (`epiweeks.py`: Sunday–Saturday weeks, week 1 contains 4 January, 53 weeks in 2014, 2020, 2025); keeps the latest InfoDengue model version per week; checks week starts against InfoDengue `data_iniSE` (0 mismatches); aggregates daily climate to complete epi weeks (7 valid days required); adds quality flags; writes the processed files below.
6. `eda.py` – EDA figures, peak-week variability, smoke-test baselines and the JSD noise floor (`results/data_foundation/`).
7. `validate_sources.py` (optional) – annual totals and monthly rainfall against Cox et al.

Model-ready arrays are built on the fly by `dengue_recon.data.panel` (53-slot weekly grid with slot 53 masked in 52-week years; monthly variant assigns each epi week to the month of its Wednesday). 2024 is not loaded while `RUN_FINAL_2024 = False`.

## Processed files (in `DATA_DIR`)

| File | Rows | Content |
|---|---|---|
| `dengue_weekly.parquet` (and `.csv.gz`) | 250,500 = 300 municipalities × 835 epi weeks (2010 W1 – 2025 W53) | Main table, one row per (geocode, epi_year, epi_week) |
| `climate_weekly.parquet` | 265,800 | NASA POWER weekly climate, 2009 W1 – 2025 W52 (2009 kept so lags exist for 2010) |
| `year_summary.csv` | one per (geocode, epi_year) | Annual total, peak week, completeness, flags |
| `municipalities.csv` | 300 | Selected municipalities with metadata |

### `dengue_weekly` columns

| Column | Type | Description |
|---|---|---|
| geocode | int | IBGE 7-digit municipality code |
| name, uf, region | str | Municipality name, state (UF), macro-region (North, Northeast, Center-West, Southeast, South) |
| epi_year, epi_week | int | SINAN/MMWR epidemiological year and week |
| week_start | date | Sunday that starts the epi week |
| n_weeks_year | int | 52 or 53 (2014, 2020, 2025 have 53 weeks; week 53 is a real week) |
| week_index | int | Consecutive week counter (0 = 2010 W1) |
| cases | int | InfoDengue `casos`: notified dengue cases in the week (0 missing weeks) |
| cases_est | float | InfoDengue nowcast `casos_est`; identical to `cases` here (all weeks final) |
| year_total | int | Sum of `cases` over the epi year |
| share | float | cases / year_total (NaN if year_total = 0, about 0.1 % of rows) |
| year_complete | bool | All weeks of the year present (True everywhere) |
| provisional | bool | True for 2025 |
| data_flag | str | `ES_2021_reporting_gap`, `covid_2020_reporting_disruption`, `provisional_2025`, or empty |
| exclude_suggested | bool | True for Espírito Santo 2021 and all of 2025; excluded from training and evaluation |
| temp_mean, temp_min, temp_max | float °C | NASA POWER 2 m temperature: weekly mean of daily mean, minimum, maximum |
| humid_mean | float % | NASA POWER 2 m relative humidity, weekly mean |
| precip_mm | float mm | NASA POWER `PRECTOTCORR`, weekly total (needs all 7 days) |
| precip_days | int | Days with at least 1 mm |
| id_temp_mean/min/max, id_humid_mean/min/max | float | InfoDengue's own weekly temperature and humidity (about 5 % missing; 24 % in 2017); used only as a check |
| pop | int | Population used by InfoDengue for that week |
| pop2021 | float | IBGE 2021 estimate |
| lat, lon | float | Municipal seat (the NASA POWER request point) |
| capital | int | 1 = state capital |

`climate_weekly` columns: `geocode, epi_year, epi_week, week_start, n_weeks_year, week_index` (0 = **2009** W1, different from the main table), `precip_mm, precip_days, temp_mean, temp_min, temp_max, humid_mean`. The models join climate on (geocode, epi year, epi week), never on `week_index` across tables.

`year_summary` columns: `geocode, epi_year, n_weeks_year, weeks_observed, year_total, peak_week, precip_missing, id_temp_missing, year_complete, provisional, data_flag, exclude_suggested`.

`municipalities` columns: `geocode, total_cases_2010_2025, name, lat, lon, capital, pop2021, uf_code, uf, region`.

## Quality flags and how they are handled

| Flag | Rows | Handling |
|---|---|---|
| Espírito Santo 2021 (about 100 notifications state-wide vs 10^4–10^5 in other years; reporting artefact) | 468 weekly rows (9 municipalities) | `exclude_suggested`: excluded from training and evaluation |
| 2020 (COVID-19 reporting disruption) | 15,900 weekly rows | Kept; every result is also reported without 2020 |
| 2025 (provisional, still being back-filled) | 15,900 weekly rows | Excluded (never loaded by the models) |
| 2024 | – | Final test year, never loaded in the pilot (`RUN_FINAL_2024 = False`) |
| 53-week years (2014, 2020) | – | 53-slot grid; slot 53 masked in 52-week years; profiles use a pseudo week 53 = mean of weeks 52 and 1 |
| Small totals | – | Training rows need an annual total ≥ 20, evaluation rows ≥ 100 |
| Missing climate | 0 in 2009–2024 | Would be set to 0 anomaly and counted (`checks_*.json`: 0 occurrences) |

## Data quality notes

- **InfoDengue has no rainfall**, so all three climate variables come from NASA POWER (one consistent source).
- **Rainfall validation against ERA5-Land** (Cox et al., 2013–2020, monthly, per-municipality medians): totals r = 0.86; **anomalies r = 0.71** overall, **0.47 in the North** (0.72–0.74 elsewhere); POWER is about 8 % drier on average.
- POWER vs InfoDengue: r = 0.84 for temperature, 0.74 for humidity.
- Annual totals agree with Cox et al. (SINAN-derived): log-scale r = 0.990, median absolute relative difference 0.6 % (90th percentile 3.5 %, 2,394 municipality-years).
- Cox et al. weeks are ISO-style, not SINAN epi weeks; use that file only for annual or monthly comparisons.
- Notifications are not infections (testing, reporting capacity, co-circulating chikungunya and Zika). The target is the **timing of notifications**.
- Selection limitation: municipalities with fewer than about 48,000 inhabitants were never candidates.
