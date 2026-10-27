# Weather source validation (CHIRPS vs NASA POWER, 2009-2024)

Median of per-municipality Pearson r; anomaly = deviation from the municipality's week-of-year (weekly) or month-of-year (monthly) climatology. Mean ratio = mean of second source / mean of first source.

| comparison | variable | res. | region | n | r raw | r anomaly | anom. IQR | mean ratio (B/A) |
|---|---|---|---|---:|---:|---:|---|---:|
| CHIRPS vs Cox ERA5-Land | precip | monthly 2013-20 | All | 300 | 0.82 | 0.63 | 0.53-0.68 | 0.92 |
| CHIRPS vs Cox ERA5-Land | precip | monthly 2013-20 | **North** | 45 | 0.87 | 0.49 | 0.39-0.54 | 0.96 |
| CHIRPS vs Cox ERA5-Land | precip | monthly 2013-20 | Northeast | 45 | 0.78 | 0.62 | 0.59-0.68 | 0.83 |
| CHIRPS vs Cox ERA5-Land | precip | monthly 2013-20 | Center-West | 45 | 0.89 | 0.67 | 0.57-0.71 | 0.86 |
| CHIRPS vs Cox ERA5-Land | precip | monthly 2013-20 | Southeast | 120 | 0.82 | 0.63 | 0.55-0.67 | 0.92 |
| CHIRPS vs Cox ERA5-Land | precip | monthly 2013-20 | South | 45 | 0.71 | 0.67 | 0.62-0.70 | 0.98 |
| POWER vs CHIRPS | precip | monthly | All | 300 | 0.90 | 0.77 | 0.72-0.83 | 1.16 |
| POWER vs CHIRPS | precip | monthly | **North** | 45 | 0.90 | 0.63 | 0.53-0.73 | 1.19 |
| POWER vs CHIRPS | precip | monthly | Northeast | 45 | 0.91 | 0.82 | 0.76-0.85 | 1.12 |
| POWER vs CHIRPS | precip | monthly | Center-West | 45 | 0.93 | 0.75 | 0.68-0.79 | 1.21 |
| POWER vs CHIRPS | precip | monthly | Southeast | 120 | 0.90 | 0.77 | 0.72-0.83 | 1.16 |
| POWER vs CHIRPS | precip | monthly | South | 45 | 0.88 | 0.85 | 0.81-0.91 | 1.08 |
| POWER vs CHIRPS | precip | weekly | All | 300 | 0.79 | 0.71 | 0.62-0.75 | 1.16 |
| POWER vs CHIRPS | precip | weekly | **North** | 45 | 0.79 | 0.61 | 0.53-0.65 | 1.19 |
| POWER vs CHIRPS | precip | weekly | Northeast | 45 | 0.75 | 0.68 | 0.59-0.72 | 1.12 |
| POWER vs CHIRPS | precip | weekly | Center-West | 45 | 0.83 | 0.73 | 0.67-0.75 | 1.21 |
| POWER vs CHIRPS | precip | weekly | Southeast | 120 | 0.80 | 0.71 | 0.67-0.74 | 1.16 |
| POWER vs CHIRPS | precip | weekly | South | 45 | 0.79 | 0.78 | 0.71-0.82 | 1.08 |
| POWER vs Cox ERA5-Land | humid_mean | monthly 2013-20 | All | 300 | 0.86 | 0.80 | 0.73-0.83 | 1.00 |
| POWER vs Cox ERA5-Land | humid_mean | monthly 2013-20 | **North** | 45 | 0.87 | 0.64 | 0.54-0.74 | 1.01 |
| POWER vs Cox ERA5-Land | humid_mean | monthly 2013-20 | Northeast | 45 | 0.91 | 0.82 | 0.80-0.83 | 1.02 |
| POWER vs Cox ERA5-Land | humid_mean | monthly 2013-20 | Center-West | 45 | 0.94 | 0.81 | 0.73-0.83 | 1.00 |
| POWER vs Cox ERA5-Land | humid_mean | monthly 2013-20 | Southeast | 120 | 0.86 | 0.81 | 0.76-0.84 | 0.99 |
| POWER vs Cox ERA5-Land | humid_mean | monthly 2013-20 | South | 45 | 0.77 | 0.77 | 0.73-0.81 | 0.98 |
| POWER vs Cox ERA5-Land | precip | monthly 2013-20 | All | 300 | 0.86 | 0.71 | 0.61-0.76 | 1.09 |
| POWER vs Cox ERA5-Land | precip | monthly 2013-20 | **North** | 45 | 0.87 | 0.47 | 0.42-0.56 | 1.20 |
| POWER vs Cox ERA5-Land | precip | monthly 2013-20 | Northeast | 45 | 0.83 | 0.72 | 0.66-0.76 | 0.94 |
| POWER vs Cox ERA5-Land | precip | monthly 2013-20 | Center-West | 45 | 0.92 | 0.74 | 0.59-0.78 | 1.08 |
| POWER vs Cox ERA5-Land | precip | monthly 2013-20 | Southeast | 120 | 0.87 | 0.73 | 0.67-0.77 | 1.08 |
| POWER vs Cox ERA5-Land | precip | monthly 2013-20 | South | 45 | 0.78 | 0.73 | 0.70-0.77 | 1.07 |
| POWER vs Cox ERA5-Land | temp_mean | monthly 2013-20 | All | 300 | 0.96 | 0.86 | 0.81-0.88 | 0.99 |
| POWER vs Cox ERA5-Land | temp_mean | monthly 2013-20 | **North** | 45 | 0.86 | 0.72 | 0.65-0.80 | 0.98 |
| POWER vs Cox ERA5-Land | temp_mean | monthly 2013-20 | Northeast | 45 | 0.97 | 0.83 | 0.80-0.87 | 0.98 |
| POWER vs Cox ERA5-Land | temp_mean | monthly 2013-20 | Center-West | 45 | 0.95 | 0.82 | 0.78-0.86 | 0.99 |
| POWER vs Cox ERA5-Land | temp_mean | monthly 2013-20 | Southeast | 120 | 0.97 | 0.88 | 0.85-0.89 | 0.99 |
| POWER vs Cox ERA5-Land | temp_mean | monthly 2013-20 | South | 45 | 0.99 | 0.90 | 0.87-0.91 | 1.00 |

## Triple collocation, monthly rainfall anomalies 2013-2020

Median per-municipality estimated correlation with the (unknown) true anomaly.

| region | n | NASA POWER | CHIRPS | Cox ERA5-Land |
|---|---:|---:|---:|---:|
| All | 300 | 0.91 | 0.81 | 0.77 |
| **North** | 45 | 0.82 | 0.78 | 0.62 |
| Northeast | 45 | 0.92 | 0.81 | 0.79 |
| Center-West | 45 | 0.91 | 0.83 | 0.82 |
| Southeast | 120 | 0.92 | 0.80 | 0.79 |
| South | 45 | 0.98 | 0.88 | 0.75 |
