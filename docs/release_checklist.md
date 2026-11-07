# Quarterly release checklist

Model versions are released quarterly (docs/adopt_rule.md). The next is planned for **February 2027** (version 55).
Between releases only data refreshes and labelled patch versions are published.

## Before the release day
1. **AI sources.** Add the quarter's rows to `data/ai_ledger/sources.csv` (Metaculus quartiles, the newest AI Impacts
   wave if any, the newest FRI LEAP wave), with the retrieval date. Nobody edits the knots by hand.
2. **AI indicators.** Add the quarter to `data/ai_ledger/indicators.csv`: US productivity (FRED OPHNFB), labour share
   (FRED PRS85006173), the four companies' capital spending from their own releases, the newest LBNL / EIA data-centre
   electricity figure. Mark anything from a secondary source as unverified.
3. **Changes to adopt.** Only changes that passed `scripts/adopt_score.py` (decision rule, noise floor, guardrails) since
   the last release, each with its entry in docs/world_system.md. None are pending for February as of 7 November 2026.
4. **UCDP.** Newest candidate files in `data/raw/outcomes/ucdp` (scripts/prospective.py downloads them monthly).

## Release day
`bash scripts/quarterly_release.sh 55 "<date>"` runs, in order: the AI rebuild, the tests, the forecast and analyses,
the monthly ledgers, the version, headline and prose-number checks, the report, and the canonical frozen forecast.

## After a human look at the report
1. `docs/methodology.md`: a dated section on what changed, and the changes table.
2. Commit and push; `bash scripts/deploy_pages.sh`; `bash scripts/deploy_netlify.sh`; republish the report artifact.
3. Update the shared methodology document (claude.ai/artifact/1XZXktLR7M6sH1bd5Pof4J): headline, changes table, limits.

## Known in advance for February
- The AI arrival curve will be rebuilt mechanically. With October 2026 source values it gives 68% by 2065 (now 72%) and
  19% by 2030 (now 18%); the February values decide.
- Version 54's canonical forecast (737e06bf69cd is version 53; version 54 is 29bf57da62a4) stays the record until then.
