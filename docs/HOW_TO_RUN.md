# How to run this project

A step-by-step guide for running everything yourself, in Google Colab or on your own computer. You do not need a GPU or any API key. Every data source is public.

Contents:

1. [Google Colab](#1-google-colab)
2. [Local computer (VS Code or Jupyter)](#2-local-computer-vs-code-or-jupyter)
3. [Reproducing the official results and figures](#3-reproducing-the-official-results-and-figures)
4. [Expected runtimes](#4-expected-runtimes)
5. [Troubleshooting](#5-troubleshooting)

Background: the notebooks below replace the older notebooks that were shared through Google Drive. Use the GitHub versions only.

---

## 1. Google Colab

### 1.1 Open a notebook

Click a link. Each one opens the notebook straight from GitHub (branch `main`):

| Notebook | What it does | Open |
|---|---|---|
| 00 Setup and data | Downloads and builds the data (needed once) | [Open in Colab](https://colab.research.google.com/github/aminah9945/dengue-climate-reconstruction/blob/main/notebooks/00_setup_and_data.ipynb) |
| 01 Base paper reproduction | Lim et al. (2026) Dirichlet-multinomial | [Open in Colab](https://colab.research.google.com/github/aminah9945/dengue-climate-reconstruction/blob/main/notebooks/01_reproduce_base_paper.ipynb) |
| 02 Brazil baselines | B0, B1, B1p, B4, B4s | [Open in Colab](https://colab.research.google.com/github/aminah9945/dengue-climate-reconstruction/blob/main/notebooks/02_brazil_baselines.ipynb) |
| 03 Climate extension | M1, ablations, sanity checks (pilot) | [Open in Colab](https://colab.research.google.com/github/aminah9945/dengue-climate-reconstruction/blob/main/notebooks/03_climate_extension.ipynb) |
| 04 Results and figures | Pilot tables and figures | [Open in Colab](https://colab.research.google.com/github/aminah9945/dengue-climate-reconstruction/blob/main/notebooks/04_results_figures.ipynb) |
| 05 Rain-lag extension | Pre-registered extension (Step 8) | [Open in Colab](https://colab.research.google.com/github/aminah9945/dengue-climate-reconstruction/blob/main/notebooks/05_rain_lag_extension.ipynb) |
| 06 Official experiments | Official results (designs A, B, C) and re-runs | [Open in Colab](https://colab.research.google.com/github/aminah9945/dengue-climate-reconstruction/blob/main/notebooks/06_official_experiments.ipynb) |

**Before pull request #1 is merged:** notebook 06 and the official results exist only on the branch `official-run`. Until the merge, use these links instead (replace `main` with `official-run` in the address for any other notebook), and in the first code cell change `BRANCH = "main"` to `BRANCH = "official-run"`:

- [00 Setup and data (official-run)](https://colab.research.google.com/github/aminah9945/dengue-climate-reconstruction/blob/official-run/notebooks/00_setup_and_data.ipynb)
- [06 Official experiments (official-run)](https://colab.research.google.com/github/aminah9945/dengue-climate-reconstruction/blob/official-run/notebooks/06_official_experiments.ipynb)

### 1.2 Run it

1. *Runtime → Run all* (or run the cells one by one from the top).
2. The first cell clones the repository (the branch in `BRANCH`) into `/content/dengue-climate-reconstruction` and installs it. This takes about 1 minute.
3. Each Colab notebook runs in its own fresh machine, so **run notebook 00 first in the same session** if a notebook says that processed data are missing, or keep the data on Google Drive (next step).

### 1.3 Keep the data on Google Drive (recommended)

Colab deletes `/content` when the session ends. To download the data only once:

1. In the first cell set `USE_DRIVE = True`. Do this in notebook 00 **and** in every notebook you run later.
2. Allow access to your Drive when asked. The data then go to `MyDrive/dengue_recon_data`.

### 1.4 Order

- Only want to **see** the results? Open notebook 04 (pilot) or 06 (official). They show the committed tables and figures without any data.
- Want to **re-run** models? Run notebook 00 first (about 30–45 minutes the first time), then 01–06. In notebooks 02, 03 and 05 set `RERUN = True` or `QUICK = True`. In notebook 06 set `RUN = True` in section 3.

---

## 2. Local computer (VS Code or Jupyter)

You need Python 3.10 or newer and `git`. Windows users: use WSL, or replace `source .venv/bin/activate` with `.venv\Scripts\activate`.

### 2.1 Install (once)

```bash
git clone https://github.com/aminah9945/dengue-climate-reconstruction
cd dengue-climate-reconstruction
git checkout official-run          # only until pull request #1 is merged into main
python -m venv .venv
source .venv/bin/activate
pip install -e ".[all]"
python -m pytest                   # 48 tests, about 10 s, no data needed
```

Conda instead of venv: `conda env create -f environment.yml && conda activate dengue-recon`.

### 2.2 Get the data (once)

```bash
python scripts/run_pipeline.py --step data                  # InfoDengue + NASA POWER + ONI, 30-45 min
python scripts/run_pipeline.py --step data --with-chirps    # the same plus CHIRPS rainfall (adds 1-2 h)
```

The files go to `data/` (change with `DATA_DIR=/some/folder`). Downloads are cached in `data/raw/`; if the command stops, run it again and it continues. You can also run notebook 00.

Only need the CHIRPS or ONI tables? `python -m dengue_recon.data.oni` (seconds); for CHIRPS: `python -m dengue_recon.data.download weather --source chirps`, then `python -m dengue_recon.data.build_dataset --source chirps`.

Note: `--step data` also rewrites the small EDA files in `results/data_foundation/`. Use `git checkout results/` to undo this if you want the committed versions back.

### 2.3 Notebooks in VS Code

1. Open the repository folder in VS Code (install the *Python* and *Jupyter* extensions).
2. Open a notebook in `notebooks/`, click *Select Kernel* and choose the `.venv` interpreter.
3. Run the cells from the top. The first cell finds the repository by itself; nothing is cloned locally.

With Jupyter instead: `pip install jupyterlab`, then `jupyter lab` in the repository folder.

To run a whole notebook from the terminal: `jupyter nbconvert --to notebook --execute notebooks/06_official_experiments.ipynb --output-dir /tmp`.

---

## 3. Reproducing the official results and figures

All official results are for test years **2010–2023**. **2024 is locked** (`RUN_FINAL_2024 = False` in `src/dengue_recon/config.py`): it is never loaded and is kept for one final test after the final model is fixed. Do not change this setting.

The committed tables and figures are in `results/official/tables/` and `results/official/figures/`, with the write-up in `results/official/OFFICIAL_RESULTS.md`. To recompute them, you need the data (Section 2.2, including `--with-chirps` for the CHIRPS run), then:

| Result | Command | Main outputs |
|---|---|---|
| H1–H3 (primary, design A) | `python scripts/run_pipeline.py --step official --design A --climate power --subset 300` | `tables/hypotheses.csv`, `paired_contrasts.csv`, `gain_by_region.csv`, `gain_by_test_year.csv`; figures A, C, D, E, G |
| H4 learning curve (design B) | `python scripts/run_pipeline.py --step official --design B --climate power --subset 300` | `tables/lc_*.csv`; figure B |
| LOYO, comparison with the pilot (design C) | `python scripts/run_pipeline.py --step official --design C --climate power --subset 300` | `tables/designC_vs_pilot.csv` |
| Robustness: CHIRPS rainfall | `python scripts/run_pipeline.py --step official --design A --climate chirps --subset 300` | `tables/cross_run_same_model.csv`; figure F |
| Robustness: without 2020 | `python scripts/run_pipeline.py --step official --design A --climate power --subset 300 --exclude-year 2020` | `tables/cross_run_same_model.csv` |

After the five runs:

```bash
python scripts/run_pipeline.py --step official-report                       # all tables and figures A-G
python scripts/official_postcheck.py results/official/designA_power_300     # re-check one run (repeat per run)
```

Notes:

- Each run writes to `results/official/<tag>/` (for example `designA_power_300`) and saves every finished fold in `folds/`. If a run stops, run the same command again: it resumes and gives identical numbers.
- `official-report` needs the per-row files `metrics_detail_<tag>.csv` of all five runs. They are written by the runs but are not in the repository, so on a fresh clone you must run the five commands first.
- To try a design without overwriting anything, add `--out results/official_rerun` and use `--subset 60` (a quick check; the numbers will differ from the 300-municipality results).
- Pilot results (Section 7.2 of the README): `python scripts/run_pipeline.py --step brazil --subset 300`, `--step extension`, then `--step report`. See README Section 4.2.

---

## 4. Expected runtimes

Measured on a 4-CPU machine with 15 GB RAM (memory stays below 3 GB). Free Colab has 2 CPUs, so expect roughly twice as long there.

| Task | Time |
|---|---|
| Install (`pip install -e ".[all]"`) | 1–2 min |
| Tests (`pytest`) | about 10 s |
| Data download and build (`--step data`) | 30–45 min |
| CHIRPS rainfall download (`--with-chirps`) | 1–2 h |
| Viewing notebooks 00–06 with committed results | seconds each |
| Official design A, 300 municipalities | about 27 min |
| Official design B, 300 municipalities | about 28 min |
| Official design C, 300 municipalities | about 53 min |
| Official design A with CHIRPS / without 2020 | about 27 / 25 min |
| All five official runs | about 2 h 40 min |
| `official-report` | about 10 s |
| Notebook 06 re-run, design A, `SUBSET = 60` | about 18 min |
| Notebook 01 (MAP reproduction, default settings) | about 2 min |
| Pilot weekly run, 60 municipalities (`--step brazil --subset 60`) | about 22 min (`--quick`: about 5 min) |

---

## 5. Troubleshooting

| Problem | Fix |
|---|---|
| `Processed data not found in DATA_DIR` | Run notebook 00 or `python scripts/run_pipeline.py --step data`. In Colab, run 00 in the same session or use `USE_DRIVE = True` in both notebooks. |
| `FileNotFoundError: .../oni_weekly.parquet` | `python -m dengue_recon.data.oni` (or rerun `--step data`). |
| `FileNotFoundError: .../climate_weekly_chirps.parquet` | Needed only for `--climate chirps`: `python scripts/run_pipeline.py --step data --with-chirps`. |
| `official-report`: `metrics_detail_designA_power_300.csv not found` | Run the five official commands of Section 3 first. The committed tables and figures do not need this. |
| Notebook 06 says the file or folder does not exist in Colab | You opened the `main` version before the merge. Use the `official-run` links and set `BRANCH = "official-run"`. |
| Colab cloned the wrong branch | Colab keeps `/content/dengue-climate-reconstruction` while the session lives. *Runtime → Disconnect and delete runtime*, set `BRANCH`, run again. |
| `ModuleNotFoundError: dengue_recon` | Locally: activate the virtual environment and run `pip install -e ".[all]"` in the repository folder. In Colab: rerun the first cell. |
| A download stops with HTTP 429/5xx or a timeout | Run the same command again. Downloads are cached and continue where they stopped. |
| A long official run stops (Colab disconnect, laptop sleep) | Run the same command again: finished folds are loaded from `folds/`. In Colab, put `--out` (or `OUT` in notebook 06) on Google Drive so the folds survive. |
| CHIRPS download is very slow | ClimateSERV processes one job per municipality (about 70 s each). Let it run; it resumes. |
| `git status` shows changed files in `results/` | Some steps rewrite committed outputs: `--step data` the EDA files in `results/data_foundation/`, notebook 01 (`RUN_MAP = True`) the MAP files in `results/repro_lim/`. `git checkout results/` restores them. A MAP refit in a different environment can differ slightly in the 3rd–4th decimal (for example pooled median MAE 12.14 instead of 12.18); this is optimiser and library-version noise, far inside the reproduction criteria. |
| The numbers of a `SUBSET = 60` run differ from the official ones | Expected: the official results use 300 municipalities. |
