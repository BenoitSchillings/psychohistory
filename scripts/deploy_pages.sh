#!/bin/bash
# Publish the report to https://benoitschillings.github.io/psychohistory (public, GitHub Pages from the gh-pages
# branch). Run after `python run.py report`. Netlify (deploy_netlify.sh) was retired when its free credits ran out.
set -e
cd "$(dirname "$0")/.."
SITE=outputs/netlify_site
mkdir -p $SITE
grep -v 'name="claude:viewport-fit-sides"' outputs/seldon_report.html \
  | sed 's#https://benoitschillings.github.io/psychohistory/\(methodology\|data_sources\)\.html#\1.html#g' > $SITE/index.html
python3 -P scripts/draw_dependencies.py
cp docs/data_dependencies.svg $SITE/
python3 -P scripts/build_site_pages.py   # methodology.html, data_sources.html
TMP=$(mktemp -d)
cp $SITE/index.html $SITE/methodology.html $SITE/data_sources.html $SITE/ethics.html $SITE/data_dependencies.svg $TMP/
touch $TMP/.nojekyll
git -C $TMP init -q -b gh-pages
git -C $TMP add -A
git -C $TMP -c user.name="Benoit Schillings" -c user.email="5769705+BenoitSchillings@users.noreply.github.com" \
  commit -qm "Site $(date -u +%Y-%m-%dT%H:%MZ)"
git -C $TMP push -qf "$(git remote get-url origin)" gh-pages
rm -rf $TMP
