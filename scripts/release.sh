#!/bin/bash
# Model release, whenever something better is ready (docs/adopt_rule.md release policy; docs/release_checklist.md).
# Run from the repo root:
#   bash scripts/release.sh VERSION "DATE TEXT"     e.g. bash scripts/release.sh 57 "1 December 2026"
# Stops at the first failing step. Publishing (push, Pages, Netlify, artifact, methodology document) is done after a
# human look at the report: commit, push, then bash scripts/publish.sh (both sites, checked live).
set -euo pipefail
V=$1; D=$2
export PYTHONPATH=. PYTORCH_ALLOC_CONF=expandable_segments:True
echo "== 1. AI arrival assumption rebuilt from data/ai_ledger/sources.csv (enter any new source rows first)"
python3 -P scripts/ai_ledger.py --apply
echo "== 2. tests"
python3 -P -m pytest -q tests
echo "== 3. forecast and analyses (about 1.5 hours of GPU)"
bash scripts/release_chain.sh
grep -q CHAIN_DONE <(tail -1 outputs/lim2_chain.log)
echo "== 4. monthly ledgers: state-based (direct 12-month model) and civilian violence, newest data"
python3 -P scripts/monthly_ledger.py issue || true
read Y M < <(python3 -P -c "from psychohistory import monthly as MO; print(*MO.last_observed())")
python3 -P scripts/monthly_ledger.py summary "$Y" "$M"
python3 -P scripts/monthly_ledger.py civilian "$Y" "$M"
echo "== 5. version, headline and every number in the prose"
sed -i "s/^VERSION = (\"[0-9.]*\", \"[^\"]*\")/VERSION = (\"$V\", \"$D\")/" psychohistory/report.py
python3 -P scripts/headline_sync.py > /dev/null
python3 -P scripts/number_parity.py --fix > /dev/null; python3 -P scripts/number_parity.py
python3 run.py report
echo "== 6. freeze and name the release's canonical forecast"
H=$(python3 -P scripts/annual_ledger.py issue | sed -n 's/^logged annual_[0-9]*_\([0-9a-f]*\)\.json$/\1/p')
python3 -P scripts/annual_ledger.py canonical "$H" "Version $V ($D)"
git add -f outputs/ledger     # the forward record is tracked: staged here so the release commit cannot miss it
echo "Release $V built. Read the report, update docs/methodology.md (what changed), commit and push, then bash scripts/publish.sh."
