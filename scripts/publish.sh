#!/bin/bash
# Publish a release to both public sites and check that each one shows the released version.
# Run after scripts/release.sh, a human look at the report, and the commit and push:
#   bash scripts/publish.sh
# The report artifact and the shared methodology document are republished separately (docs/release_checklist.md).
set -euo pipefail
cd "$(dirname "$0")/.."
V=$(sed -n 's/^VERSION = ("\([0-9.]*\)", "\([^"]*\)").*/Version \1 · \2/p' psychohistory/report.py)
[ -n "$V" ] || { echo "no VERSION in psychohistory/report.py"; exit 1; }
grep -q "$V" outputs/seldon_report.html || { echo "outputs/seldon_report.html is not '$V': run python run.py report"; exit 1; }
[ -z "$(git status --porcelain)" ] || echo "warning: uncommitted changes in the working tree"
bash scripts/deploy_pages.sh
bash scripts/deploy_netlify.sh
rc=0
for u in https://benoitschillings.github.io/psychohistory/ https://seldon-psychohistory.netlify.app/; do
  ok=no
  for i in $(seq 1 12); do          # GitHub Pages takes a minute or two to rebuild
    # (count, not grep -q: grep -q closes the pipe at the first match, curl fails on it and pipefail fails the check)
    if [ "$(curl -fsS "$u" | grep -c "$V")" -gt 0 ]; then ok=yes; break; fi
    sleep 15
  done
  echo "$u: $([ $ok = yes ] && echo "$V" || echo "NOT showing $V")"
  [ $ok = yes ] || rc=1
done
exit $rc
