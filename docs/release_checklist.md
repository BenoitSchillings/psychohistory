# Quarterly release checklist

Model versions are released quarterly (docs/adopt_rule.md). The next is planned for **February 2027** (version 55).
Between releases only data refreshes and labelled patch versions are published.

## Before the release day
1. **AI sources.** Add the quarter's rows to `data/ai_ledger/sources.csv` (Metaculus quartiles, the newest AI Impacts
   wave if any, the newest FRI LEAP wave), with the retrieval date. Nobody edits the knots by hand.
2. **AI indicators.** Add the quarter to `data/ai_ledger/indicators.csv`: US productivity (FRED OPHNFB), labour share
   (FRED PRS85006173), the four companies' capital spending from their own releases, the newest LBNL / EIA data-centre
   electricity figure. Mark anything from a secondary source as unverified.
3. **Changes to adopt.** Only changes that passed their pre-registered test since the last release, each with its
   entry in docs/world_system.md. Pending for February: `monthly.NEWS_TOPICS = True` (news topics in the 12-month
   model; passed 8 November 2026), after refreshing `data/raw/conflictforecast/conflictforecast_ons_armedconf_12.csv`
   from conflictforecast.org (cite Mueller and Rauh 2022).
   From the third debate (9 November 2026), each passed its pre-registered test:
   - `data.CONFLICT_CODING = "deaths"` (civil conflict where the deaths occur; scripts/conflict_coding_test.py). The
     panel cache keeps the location coding; the switch recodes on load. Re-baseline the adopt-rule hindcast tags after
     it, since the hindcast outcomes change with it.
   - `pipeline.COUP_RECAL = True` (5-year coup risks shifted by the backtests' calibration; scripts/coup_recal_test.py;
     needs outputs/backtest_predictions.parquet from the release's backtest).
   - `monthly.ANNUAL_MIN_CHILD = 2400` (12-month model leaves in country-years; scripts/monthly_cluster_test.py).
     Rerun scripts/monthly_topics_test.py with it before switching NEWS_TOPICS on, and publish its year-by-year table.
   - `simulate.WORKFORCE_FALL = (0.57, 1.22)`: an assumption, not a test result (scripts/workforce_fall_test.py shows
     its effect); label it in the assumptions list.
   From the data debate (10 November 2026), bug fixes under rule 7, already in the code (re-baseline tags v55f_base):
   - the working-age term splits on the raw change and removes workforce_beta x WORKFORCE_MEAN once;
   - civil-war deaths come off the population;
   - the explosive-growth exploration seeds its generator with seed + 929 (was 919, shared with the falling-share draw).
   Run `python -P scripts/world_audit.py` on the release's forecast and publish its five checks (world totals failed
   at 2.7% in November: no near-free AI with the largest climate-damage family).
   `simulate.PRESTON`: its adopt-rule rerun was neutral (-0.00%); switch it on only as a stated consistency judgement.
   `simulate.CATASTROPHE_TAILS = True` (nuclear winter after a large exchange, eruptions larger than Pinatubo): stated
   assumptions with published ranges, not test results; label them in the assumptions list and publish
   scripts/catastrophe_tails.py's off/on table with the release.
   `pipeline.COUP1Y_RECAL = True` (12-month coup risks, Platt scaling; scripts/coup1y_recal_test.py passed 10 November
   2026; needs outputs/backtest_predictions.parquet from the release's backtest).
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
