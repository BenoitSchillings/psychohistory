#!/bin/bash
# The adopt rule's hindcast set for one tag (docs/adopt_rule.md): 16 origins, 15 years, three ensemble refits.
#   bash scripts/hindcast_tag.sh TAG [world_hindcast options, e.g. --consts '{"simulate.X": 1}']
# Writes outputs/world_hindcast/TAG_s{0,1,2}_*.npz and outputs/adopt/TAG_s*.log; then score with
#   python -P scripts/adopt_score.py BASE TAG        (or --floor TAG for a baseline's noise floor)
set -u
cd "$(dirname "$0")/.."
TAG=$1; shift
O="1970 1973 1976 1979 1980 1983 1986 1989 1990 1993 1996 1999 2000 2003 2006 2009"
mkdir -p outputs/adopt
for s in 0 1 2; do
  until [ $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1) -gt 12000 ]; do sleep 60; done
  PYTHONPATH=. python3 -P scripts/world_hindcast.py ${TAG}_s$s 15 $O --ens-seed $s "$@" > outputs/adopt/${TAG}_s$s.log 2>&1
  echo "$TAG s$s rc=$? $(date +%H:%M)"
done
echo "HINDCAST_DONE $TAG"
