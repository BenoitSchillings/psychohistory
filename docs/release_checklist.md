# Release checklist

Since 10 November 2026 a model version is released whenever something better is ready (docs/adopt_rule.md); this checklist applies to every release.
Between releases only data refreshes and labelled patch versions are published.

## Before the release day
1. **AI sources.** Add any new rows to `data/ai_ledger/sources.csv` (Metaculus quartiles, the newest AI Impacts
   wave if any, the newest FRI LEAP wave), with the retrieval date. Nobody edits the knots by hand.
2. **AI indicators.** Add any new quarter to `data/ai_ledger/indicators.csv`: US productivity (FRED OPHNFB), labour share
   (FRED PRS85006173), the four companies' capital spending from their own releases, the newest LBNL / EIA data-centre
   electricity figure. Mark anything from a secondary source as unverified.
3. **Changes to adopt.** Only changes that passed their pre-registered test since the last release, bug fixes, and
   assumptions the director asked for, each with its entry in docs/world_system.md. Nothing is held back for a date.
   An assumption (not a test result) is labelled as one in the assumptions list. If a change alters the hindcast
   outcomes or fixes a bug, re-baseline the adopt-rule hindcast tags after it. Every bug fixed gets a test in
   tests/test_regressions.py. Run `python -P scripts/world_audit.py` on the release's forecast and publish its checks.
4. **UCDP.** Newest candidate files in `data/raw/outcomes/ucdp` (scripts/prospective.py downloads them monthly).

## Release day
`bash scripts/release.sh <version> "<date>"` runs, in order: the AI rebuild, the tests, the forecast and analyses (scripts/release_chain.sh),
the monthly ledgers, the version, headline and prose-number checks, the report, and the canonical frozen forecast.

## After a human look at the report
1. `docs/methodology.md`: a dated section on what changed, and the changes table.
2. Commit (including the new files in outputs/ledger) and push; `bash scripts/publish.sh` deploys both sites (GitHub
   Pages and Netlify) and checks that each shows the new version; republish the report artifact.
3. Update the shared methodology document (its link is in the director's notes): headline, changes table, limits.
