#!/bin/bash
# Publish the report to https://seldon-psychohistory.netlify.app (public). Run after `python run.py report`.
# The Netlify copy carries reader notes (site/notes.js, site/netlify/functions/notes.mjs, stored in Netlify Blobs).
# DEPLOY=draft bash scripts/deploy_netlify.sh   -> a draft deploy URL instead of production
set -e
cd "$(dirname "$0")/.."
SITE_ID=1b90db7f-d19e-442a-aa3f-13d5740e6303
mkdir -p outputs/netlify_site
grep -v 'name="claude:viewport-fit-sides"' outputs/seldon_report.html \
  | sed 's#https://benoitschillings.github.io/psychohistory/\(methodology\|data_sources\)\.html#\1.html#g' > outputs/netlify_site/index.html
python3 -P scripts/draw_dependencies.py
cp docs/data_dependencies.svg outputs/netlify_site/
python3 -P scripts/build_site_pages.py   # methodology.html, data_sources.html
cp site/notes.js outputs/netlify_site/
for f in index methodology data_sources; do
  grep -q 'src="/notes.js"' outputs/netlify_site/$f.html || \
    sed -i 's#</html>\s*$#<script src="/notes.js" defer></script>\n</html>#' outputs/netlify_site/$f.html
  grep -q 'src="/notes.js"' outputs/netlify_site/$f.html || echo '<script src="/notes.js" defer></script>' >> outputs/netlify_site/$f.html
done
cd site
[ -d node_modules ] || npm install --silent
if [ "${DEPLOY:-prod}" = draft ]; then
  npx --yes netlify-cli deploy --dir ../outputs/netlify_site --functions netlify/functions --site $SITE_ID
else
  npx --yes netlify-cli deploy --prod --dir ../outputs/netlify_site --functions netlify/functions --site $SITE_ID
fi
