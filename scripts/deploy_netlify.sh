#!/bin/bash
# Publish the report to https://seldon-psychohistory.netlify.app (public). Run after `python run.py report`.
set -e
cd "$(dirname "$0")/.."
mkdir -p outputs/netlify_site
grep -v 'name="claude:viewport-fit-sides"' outputs/seldon_report.html > outputs/netlify_site/index.html
python3 -P scripts/draw_dependencies.py
cp docs/data_dependencies.svg outputs/netlify_site/
python3 -P scripts/build_site_pages.py   # methodology.html, data_sources.html
cd outputs/netlify_site
npx --yes netlify-cli deploy --prod --dir . --site 1b90db7f-d19e-442a-aa3f-13d5740e6303
