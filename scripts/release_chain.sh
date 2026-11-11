#!/bin/bash
# The release forecast chain: climate to 2100, the canonical forecast, uncertainty runs, disaster analyses, climate risk.
# Called by scripts/release.sh (step 3); about 1.5 hours of GPU. Waits for 12 GB of free GPU memory before each step.
# Appends to outputs/lim2_chain.log and ends it with CHAIN_DONE or CHAIN_FAILED.
cd "$(dirname "$0")/.."
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=outputs/lim2_chain.log
rm -f outputs/disaster/knockouts.json   # knock resumes from this file: a stale one skips every run
echo "chain start $(date)" >> $L
for step in "2100:scripts/climate_2100.py:climate_2100" "forecast:scripts/rerun_forecast.py:forecast_fix" "usims:scripts/uncertainty_sims.py 12:uncertainty_sims" "uattr:scripts/uncertainty_attribution.py:uncertainty_attribution" "uranges:scripts/uncertainty_ranges.py:uncertainty_ranges" "dsim:scripts/disaster_rca.py sim:disaster_sim" "dknock:scripts/disaster_rca.py knock:disaster_knock" "danalyze:scripts/disaster_rca.py analyze:disaster_analyze" "crisk:scripts/climate_risk.py:climate_risk"; do
  IFS=: read name cmd log <<< "$step"
  until [ $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1) -gt 12000 ]; do sleep 60; done
  python3 -P $cmd > outputs/$log.log 2>&1; rc=$?; echo "$name done $rc $(date)" >> $L
  if [ $rc -ne 0 ]; then echo CHAIN_FAILED >> $L; exit 1; fi
done
echo CHAIN_DONE >> $L
