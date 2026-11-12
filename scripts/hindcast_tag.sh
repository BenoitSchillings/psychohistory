#!/bin/bash
# The adopt rule's hindcast set for one tag (docs/adopt_rule.md, 12 November 2026): an origin for every year
# 1970-2009, 15 years, one ensemble refit (SEEDS="0 1 2" for more).
#   bash scripts/hindcast_tag.sh TAG [world_hindcast options, e.g. --consts '{"simulate.X": 1}']
# Writes outputs/world_hindcast/TAG_s0_*.npz and outputs/adopt/TAG_s0.log (runs already on disk are kept); then score with
#   python -P scripts/adopt_score.py BASE TAG
set -u
cd "$(dirname "$0")/.."
TAG=$1; shift
O=$(seq -s " " 1970 2009)
mkdir -p outputs/adopt
for s in ${SEEDS:-0}; do
  until [ $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1) -gt 12000 ]; do sleep 60; done
  PYTHONPATH=. python3 -P scripts/world_hindcast.py ${TAG}_s$s 15 $O --ens-seed $s "$@" >> outputs/adopt/${TAG}_s$s.log 2>&1
  echo "$TAG s$s rc=$? $(date +%H:%M)"
done
echo "HINDCAST_DONE $TAG"
