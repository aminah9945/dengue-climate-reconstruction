# 2024 data checks (Google Trends, MX)

New window files: 33 (`dengue_<geo>_2023_2025.csv`).

## 1. 2010-2023 inputs unchanged

`load_trends()` with and without the new files (2024 locked): identical = **True** (12045 rows).

## 2. Google's sampling: 2023 in the old and new window

Within-year shape of 2023 from the original `2022_2024` file vs the new `2023_2025` file.

- weekly correlation: median 0.981, minimum -0.020 (MX-ZAC)
- JSD of the two 2023 shapes: median 0.0219, maximum 1.0000 (MX-ZAC)

| geo | weeks | corr | JSD |
|---|---|---|---|
| MX-AGU | 52 | 0.981 | 0.0282 |
| MX-BCN | 52 | 0.862 | 0.1208 |
| MX-BCS | 52 | 0.980 | 0.0342 |
| MX-CAM | 52 | 0.995 | 0.0219 |
| MX-CHH | 52 | 0.680 | 0.2532 |
| MX-CHP | 52 | 0.999 | 0.0003 |
| MX-COA | 52 | 0.872 | 0.0978 |
| MX-COL | 52 | 0.839 | 0.1361 |
| MX-DIF | 52 | 0.997 | 0.0010 |
| MX-DUR | 52 | 0.299 | 0.7250 |
| MX-GRO | 52 | 0.999 | 0.0009 |
| MX-GUA | 52 | 0.998 | 0.0089 |
| MX-HID | 52 | 0.875 | 0.1110 |
| MX-JAL | 52 | 0.989 | 0.0012 |
| MX-MEX | 52 | 0.996 | 0.0006 |
| MX-MIC | 52 | 0.987 | 0.0098 |
| MX-MOR | 52 | 1.000 | 0.0004 |
| MX-NAY | 52 | 0.911 | 0.0489 |
| MX-NLE | 52 | 0.885 | 0.0279 |
| MX-OAX | 52 | 0.999 | 0.0003 |
| MX-PUE | 52 | 0.999 | 0.0024 |
| MX-QUE | 52 | 0.959 | 0.0515 |
| MX-ROO | 52 | 0.999 | 0.0002 |
| MX-SIN | 52 | 0.957 | 0.0031 |
| MX-SLP | 52 | 0.942 | 0.0605 |
| MX-SON | 52 | 0.572 | 0.1145 |
| MX-TAB | 52 | 0.998 | 0.0009 |
| MX-TAM | 52 | 0.825 | 0.0505 |
| MX-TLA | 52 | 0.778 | 0.2310 |
| MX-VER | 52 | 1.000 | 0.0002 |
| MX-YUC | 52 | 1.000 | 0.0006 |
| MX-ZAC | 52 | -0.020 | 1.0000 |
| MX | 52 | 0.998 | 0.0003 |

## 3. 2024 completeness

- geos with 2024 data: 33; all from the 2023 window: True
- weeks per geo: min 52, max 52 (2024 has 52 epi weeks); missing values: 0
- geos with more than 10 zero weeks: MX-AGU, MX-BCN, MX-CAM, MX-CHH, MX-COA, MX-DUR, MX-QUE, MX-SON, MX-TLA, MX-ZAC

## 4. Do 2024 searches follow 2024 cases?

Per state, weekly correlation between Google Trends and weekly cases of the study municipalities.

| year | median corr | states |
|---|---|---|
| 2017 | 0.363 | 27 |
| 2018 | 0.512 | 27 |
| 2019 | 0.823 | 32 |
| 2020 | 0.504 | 32 |
| 2021 | 0.211 | 30 |
| 2022 | 0.667 | 30 |
| 2023 | 0.830 | 32 |
| 2024 | 0.923 | 32 |

## Reading

- The new files leave every 2017-2023 input unchanged (check 1).
- Google's sampling: for most states the two downloads of 2023 agree closely (median correlation
  0.981). States with few searches (Zacatecas, Durango, Chihuahua, Sonora, Tlaxcala) differ a lot,
  because their series are mostly zeros and Google samples them anew on each request. The
  training years carry the same noise, so 2024 is no different in kind.
- 2024 is complete (52 weeks, no missing values) and its searches follow 2024 cases more closely
  than in any earlier year (median state correlation 0.923).
- Cases: see docs/FINAL_TEST_2024_MX.md, section 2 (DGE 191224 snapshot, date format fixed,
  2017-2023 identical, weeks 51-52 truncated as in 2021-2023).

Data judged fit for the run.
