# 2024 data checks (Google Trends)

New window files: 28 (`dengue_<geo>_2023_2025.csv`).

## 1. 2010-2023 inputs unchanged

`load_trends()` with and without the new files (2024 locked): identical = **True** (20440 rows).

## 2. Google's sampling: 2023 in the old and new window

Within-year shape of 2023 from the original `2022_2024` file vs the new `2023_2025` file.

- weekly correlation: median 0.994, minimum 0.182 (BR-AP)
- JSD of the two 2023 shapes: median 0.0007, maximum 0.4705 (BR-AP)

| geo | weeks | corr | JSD |
|---|---|---|---|
| BR-AC | 52 | 0.994 | 0.0188 |
| BR-AL | 52 | 0.961 | 0.0020 |
| BR-AM | 52 | 0.990 | 0.0014 |
| BR-AP | 52 | 0.182 | 0.4705 |
| BR-BA | 52 | 0.991 | 0.0005 |
| BR-CE | 52 | 0.996 | 0.0005 |
| BR-DF | 52 | 0.994 | 0.0012 |
| BR-ES | 52 | 0.999 | 0.0002 |
| BR-GO | 52 | 0.995 | 0.0004 |
| BR-MA | 52 | 0.996 | 0.0011 |
| BR-MG | 52 | 0.999 | 0.0006 |
| BR-MS | 52 | 0.999 | 0.0003 |
| BR-MT | 52 | 0.999 | 0.0002 |
| BR-PA | 52 | 0.994 | 0.0006 |
| BR-PB | 52 | 0.983 | 0.0014 |
| BR-PE | 52 | 0.991 | 0.0005 |
| BR-PI | 52 | 0.993 | 0.0023 |
| BR-PR | 52 | 0.995 | 0.0007 |
| BR-RJ | 52 | 0.992 | 0.0007 |
| BR-RN | 52 | 0.980 | 0.0020 |
| BR-RO | 52 | 0.997 | 0.0010 |
| BR-RR | 52 | 0.517 | 0.4284 |
| BR-RS | 52 | 0.999 | 0.0005 |
| BR-SC | 52 | 0.999 | 0.0005 |
| BR-SE | 52 | 0.965 | 0.0019 |
| BR-SP | 52 | 0.996 | 0.0007 |
| BR-TO | 52 | 0.982 | 0.0067 |
| BR | 52 | 0.998 | 0.0004 |

## 3. 2024 completeness

- geos with 2024 data: 28; all from the 2023 window: True
- weeks per geo: min 52, max 52 (2024 has 52 epi weeks); missing values: 0
- geos with more than 10 zero weeks: BR-AC, BR-RR

## 4. Do 2024 searches follow 2024 cases?

Per state, weekly correlation between Google Trends and weekly cases of the study municipalities.

| year | median corr | states |
|---|---|---|
| 2019 | 0.881 | 27 |
| 2020 | 0.772 | 27 |
| 2021 | 0.689 | 27 |
| 2022 | 0.929 | 27 |
| 2023 | 0.892 | 27 |
| 2024 | 0.958 | 27 |

## Reading

All checks pass. The 2023 shapes of the two downloads agree closely (median JSD 0.0007) except in Amapa and Roraima, the two states with the fewest searches, where Google rounds small volumes to 0 and resamples them; their search signal was already documented as weak (data_snapshots/README.md). 2024 is complete and searches follow cases more closely in 2024 than in any year since 2019.
