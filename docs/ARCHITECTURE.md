# Architecture

## Pipeline

Every box names the module that implements it. Steps 3–7 run once per leave-one-year-out (LOYO) fold inside `experiments/brazil_loyo.py`.

```mermaid
flowchart TD
    subgraph S1["Step 1: data (data/)"]
        A1["InfoDengue API<br/>weekly notifications"] --> D1["download.py"]
        A2["NASA POWER API<br/>daily climate"] --> D1
        A3["IBGE + municipal seats<br/>population, coordinates"] --> D1
        D1 --> D2["build_dataset.py<br/>epi weeks, QC flags"]
        D2 --> D3[("dengue_weekly.parquet<br/>climate_weekly.parquet")]
        D3 --> D4["panel.py<br/>53-slot panel, climate matrix"]
        D3 --> D5["eda.py, validate_sources.py<br/>EDA, smoke test, Cox et al. check"]
    end

    subgraph S2["Step 2: base paper (base_paper/)"]
        L1["fetch.py<br/>Lim et al. CV mask, data only"] --> L2["lim_dm.py<br/>DM re-implementation, NUTS and MAP"]
        L2 --> L3["reproduce.py<br/>table vs published"]
    end

    subgraph LOYO["Per LOYO fold 2010-2023 (experiments/brazil_loyo.py)"]
        F0["evaluation/cv.py<br/>hold out year t, exclusions, 2024 lock"]
        F0 --> F1["features/seasonal.py<br/>offset h_m"]
        F0 --> F2["features/size_terms.py<br/>log total x harmonics"]
        F0 --> F3["features/climate.py<br/>anomalies, lag blocks"]
        F1 --> B["baselines/simple.py<br/>B0, B1, B1p"]
        F1 --> M["models/dirichlet_multinomial.py<br/>B4, B4s, M1, ablations"]
        F2 --> M
        F3 --> M
        F3 --> X["models/xgb_softmax.py<br/>M2, optional"]
        M --> K["checks/<br/>zero-climate, shuffles, leakage audit"]
    end

    D4 --> F0
    L3 -.->|"cancellation: year effect cannot change shape"| M
    B --> E["evaluation/scoring.py<br/>allocation, JSD, peak, EMD, CRPS, coverage"]
    M --> E
    X --> E
    E --> ST["evaluation/bootstrap.py<br/>municipality and year-block bootstraps"]
    ST --> R["reporting/<br/>tables, sanity checks, figures"]
    S8["extensions/rain_lags.py<br/>M1-R, M1-RTH, pre-registered"] --> E
    F0 --> S8
```

## Model ladder

Each rung adds one block of terms to the logits of the same Dirichlet-multinomial model, so every comparison is nested. The primary contrast for the research question is **M1 vs B4s**.

```mermaid
flowchart LR
    B0["B0<br/>uniform 1/W"] --> B1["B1<br/>seasonal profile exp(h_m)"]
    B1 --> B4["B4<br/>DM: h_m + g + r + c<br/>phi per region<br/>(base-paper structure)"]
    B4 --> B4s["B4s<br/>+ annual-size terms<br/>gamma x n x H(w)<br/>primary comparator"]
    B4s --> M1["M1<br/>+ climate anomalies<br/>theta x X(lag blocks)<br/>the extension"]
    M1 --> M1R["M1-R / M1-RTH<br/>dual rain lags,<br/>asymmetric climate<br/>(Step 8)"]
    M1 -.-> AB["ablations: M1-short, M1-mult,<br/>shuffled climate"]
```

Logit of municipality m, year t, week s (Eq. 2 of the proposal):

```
logit[m,t,s] = h[m,s] + g[s] + r[R(m),s] + c[m,s]                       (B4)
             + sum_k (gamma[k] + gamma_reg[R(m),k]) * Z[m,t,s,k]          (+ B4s)
             + sum_f (theta[f] + delta[R(m),f]) * X[m,t,s,f]              (+ M1)
pi[m,t,:]    = softmax over the real weeks of year t
y[m,t,:]     ~ DirichletMultinomial(N[m,t], phi[R(m)] * pi[m,t,:])
```

The base paper's year-constant effect is left out. It is the same for every week, so it cancels in the softmax (`tests/test_models.py::test_year_effect_cancels_in_softmax`; maximum share difference 1.1e-16 in the reproduction).

## Data flow and file layout

```
DATA_DIR (default data/, git-ignored)
  raw/infodengue/<geocode>.csv, raw/power/<geocode>.csv, raw/meta/, raw/lim_repo/
  dengue_weekly.parquet, climate_weekly.parquet, municipalities.csv, year_summary.csv
RESULTS_DIR (default results/)
  data_foundation/   EDA, smoke test, source validation           (Step 1)
  repro_lim/         reproduction table, folds, cancellation        (Step 2)
  brazil/            preds_*.npz*, metrics_detail_*, tables, checks (Steps 3-7, 9)
  rain_extension/    pre-registration, tables, figures             (Steps 8, 9)
  figures/           main figures                                   (Step 9)
* large files are not committed; see results/README.md
```

## Design rules enforced in code

- **No leakage.** Every statistic estimated from data (offsets, climatology, SDs, centring constants) is computed from training rows of the fold only; `checks/leakage.py` recomputes each with the held-out year blanked and requires identical values, in every fold.
- **Exact totals.** Allocation and every probabilistic draw are asserted to sum to the annual total.
- **2024 untouched.** `config.RUN_FINAL_2024 = False`; `data.panel.load_tables` drops 2024 and later; `evaluation.cv.assert_final_year_locked` guards the drivers.
- **Determinism.** Fixed seeds (`config.SEED`, bootstrap seed 1, NUTS seed `20250811 + 10000 + 100·fold`) and float64 JAX. The refactored code reproduces the saved pilot predictions exactly (`docs/VERIFICATION.md`).
- **Base-paper code is never executed.** Its repository is cloned at a pinned commit only to read the CV mask and the published table.
