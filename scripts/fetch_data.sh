#!/bin/bash
# Restore data/raw from the GitHub release archives (the raw data is too large for git).
#   bash scripts/fetch_data.sh            # all parts of the latest data release
#   bash scripts/fetch_data.sh events     # only archives whose name contains "events"
# Needs the GitHub CLI (gh) or curl. Sources whose terms forbid redistribution are not in the archives;
# DATA_LICENSES.md lists them with their download pages.
set -e
cd "$(dirname "$0")/.."
REPO=BenoitSchillings/psychohistory
TAG=${DATA_TAG:-data-2026-10-29}
mkdir -p data/raw .data_dl
if command -v gh >/dev/null; then
  gh release download "$TAG" --repo "$REPO" --dir .data_dl --pattern "*${1:-}*" --skip-existing
else
  for u in $(curl -s "https://api.github.com/repos/$REPO/releases/tags/$TAG" | grep -o '"browser_download_url": *"[^"]*"' | cut -d'"' -f4 | grep -- "${1:-}"); do
    curl -L -C - -o ".data_dl/$(basename "$u")" "$u"
  done
fi
if [ -f .data_dl/SHA256SUMS ]; then (cd .data_dl && sha256sum -c --ignore-missing SHA256SUMS); fi
# each archive may be split into .partNN pieces: join and extract
for base in $(ls .data_dl | grep '\.tar\.gz' | sed -E 's/\.part[0-9]+$//' | sort -u); do
  if ls .data_dl/"$base".part* >/dev/null 2>&1; then
    cat $(ls .data_dl/"$base".part* | sort) | tar -xzf - -C data/raw
  else
    tar -xzf ".data_dl/$base" -C data/raw
  fi
  echo "extracted $base"
done
echo "data/raw restored; remove .data_dl/ to free space"
