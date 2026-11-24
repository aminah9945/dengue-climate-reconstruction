# Final test on 2024, Mexico: results

Protocol: [`docs/FINAL_TEST_2024_MX.md`](../../docs/FINAL_TEST_2024_MX.md) (committed before the run). Data checks: [`data_checks.md`](data_checks.md).

One fold: train 2017-2023, test 2024. 30 state-years (annual total >= 100). Intervals: state bootstrap, 2,000 replicates, seed 1. Change = relative change in mean JSD.

## Claims fixed before the run

| # | Contrast | JSD a | JSD b | Change | 95 % CI | a better in | Result |
|---|---|---|---|---|---|---|---|
| C1 | S2 - B4s | 0.0788 | 0.0857 | -8.0 % | [-14.3, -3.3] | 83 % | CONFIRMED |
| C2 | INLA-scaled-trends - INLA-scaled | 0.0852 | 0.0934 | -8.8 % | [-12.7, -5.3] | 87 % | CONFIRMED |
| C3 | S1 - B4s | 0.0836 | 0.0857 | -2.5 % | [-15.7, +8.9] | 53 % | NOT CONFIRMED |
| C4 | S3 - B4s | 0.0769 | 0.0857 | -10.3 % | [-24.2, +2.1] | 67 % | NOT CONFIRMED |
| C5 | T1 - B4s | 0.0797 | 0.0857 | -7.1 % | [-25.3, +14.4] | 73 % | NOT CONFIRMED |
| C6 | S3C - S3 | 0.1051 | 0.0769 | +36.6 % | [+24.5, +49.9] | 10 % | CONFIRMED |
| C7 | S2-wy - S2 | 0.0827 | 0.0788 | +4.9 % | [-0.9, +11.3] | 40 % | NOT CONFIRMED |

## All models

| Model | jsd | peak_err | onset_err | emd | mae_counts | cov80 | crps |
|---|---|---|---|---|---|---|---|
| S3 | 0.0769 | 4.40 | 5.87 | 3.36 | 176.66 | 0.7205 | 128.90 |
| S2 | 0.0788 | 4.40 | 5.07 | 3.50 | 170.41 | 0.7141 | 123.09 |
| INLA-scaled-clim15trends | 0.0794 | 4.70 |  | 4.41 | 146.50 | 0.8744 | 107.62 |
| T1 | 0.0797 | 3.73 | 5.87 | 3.37 | 167.91 | 0.7647 | 117.61 |
| S2-wy | 0.0827 | 4.47 | 5.43 | 3.67 | 176.59 |  |  |
| S1 | 0.0836 | 4.53 | 6.30 | 3.59 | 184.12 | 0.6788 | 135.71 |
| INLA-scaled-trends | 0.0852 | 5.90 |  | 4.69 | 154.78 | 0.8686 | 112.94 |
| S1-wy | 0.0855 | 4.43 | 5.93 | 3.77 | 178.51 |  |  |
| B1 | 0.0855 | 4.67 | 8.90 | 4.50 | 177.36 |  |  |
| B4s | 0.0857 | 4.40 | 5.87 | 3.76 | 179.27 | 0.6955 | 130.28 |
| INLA-seasonal-only | 0.0871 | 5.90 |  | 4.85 | 165.22 |  |  |
| S2C-noONI | 0.0878 | 4.27 | 4.13 | 3.66 | 192.92 |  |  |
| M1-noONI | 0.0931 | 4.30 | 4.40 | 3.78 | 198.94 | 0.6500 | 150.05 |
| INLA-scaled | 0.0934 | 5.87 |  | 5.05 | 163.33 | 0.8545 | 117.13 |
| INLA-unscaled | 0.0934 | 5.87 |  | 5.05 | 268.20 | 0.5333 | 248.75 |
| S2C | 0.0977 | 4.97 | 4.53 | 4.09 | 209.12 |  |  |
| M1 | 0.1036 | 5.00 | 5.00 | 4.26 | 214.49 | 0.6032 | 162.37 |
| S3C | 0.1051 | 5.30 | 6.43 | 4.33 | 220.76 | 0.6026 | 168.12 |

## Other contrasts (descriptive)

| Contrast | Change | 95 % CI | a better in |
|---|---|---|---|
| M1 - B4s | +20.8 % | [+11.8, +30.1] | 20 % |
| M1-noONI - B4s | +8.5 % | [-0.3, +18.0] | 40 % |
| S2C-noONI - S2 | +11.3 % | [+2.7, +20.5] | 37 % |
| S2C - S2C-noONI | +11.3 % | [+6.4, +16.8] | 17 % |
| INLA-scaled - B4s | +9.0 % | [-22.0, +42.1] | 40 % |
| INLA-scaled-clim15trends - INLA-scaled-trends | -6.8 % | [-12.3, -1.6] | 67 % |
| INLA-scaled-trends - S2 | +8.1 % | [-25.6, +41.1] | 43 % |
| INLA-scaled-clim15trends - S2 | +0.7 % | [-33.0, +36.7] | 47 % |
| S3 - S1 | -8.0 % | [-14.8, -3.0] | 77 % |
| S1-wy - S1 | +2.2 % | [-9.2, +16.7] | 47 % |
| M1 - M1-noONI | +11.3 % | [+6.3, +16.9] | 20 % |

## Mean JSD by region

| Region | B4s | M1 | S2 | INLA-scaled | INLA-scaled-trends | INLA-scaled-clim15trends | S1 |
|---|---|---|---|---|---|---|---|
| Center | 0.0227 | 0.0373 | 0.0217 | 0.0672 | 0.0632 | 0.0495 | 0.0241 |
| Northeast | 0.0607 | 0.0653 | 0.0469 | 0.0564 | 0.0541 | 0.0483 | 0.0508 |
| Northwest | 0.0810 | 0.0813 | 0.0607 | 0.1042 | 0.0956 | 0.0890 | 0.0832 |
| South-Southeast | 0.0954 | 0.1210 | 0.0933 | 0.1192 | 0.1083 | 0.1032 | 0.0636 |
| West | 0.1381 | 0.1721 | 0.1369 | 0.0947 | 0.0834 | 0.0832 | 0.1682 |

## Consistency checks

- Every model scores the same state-years.
