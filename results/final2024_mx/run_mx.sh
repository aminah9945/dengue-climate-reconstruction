#!/bin/bash
# Mexico 2024 final test, as fixed in docs/FINAL_TEST_2024_MX.md
cd /d/dengue-climate-reconstruction
export DENGUE_COUNTRY=MX DENGUE_RUN_FINAL_2024=1 DATA_DIR='C:\dr_data_mx_2024' TIMING_TRENDS_DIR="D:/dengue-climate-reconstruction/data_snapshots/google_trends_mx" PYTHONPATH=src PYTHONUNBUFFERED=1
export PATH="/c/Program Files/R/R-4.6.2/bin:$PATH"
echo "$(date -Is) timing start"
.venv/Scripts/python.exe -W ignore -m dengue_recon.extensions.timing.run --design A --subset 300 --years 2024 \
  --models B1,B4s,M1-noONI,M1,S1,S2,S3,S3C,T1,S1-wy,S2-wy,S2C,S2C-noONI \
  --out results/final2024_mx/timing > results/final2024_mx/timing.log 2>&1
echo "$(date -Is) timing exit $?"
.venv/Scripts/python.exe -W ignore -m dengue_recon.extensions.inla run --design A --subset 300 --climate power --years 2024 \
  --variants base,trends,clim15trends --threads 4 --out results/final2024_mx/inla > results/final2024_mx/inla.log 2>&1
echo "$(date -Is) inla exit $?"
