# AI ledger: how the AI assumption is updated (pre-registered 6 November 2026)

When near-free cognition arrives (world.ARRIVAL_CDF) explains 38% of the spread in 2065 world output, and history
cannot test it. This ledger turns it into the part of the model that can be checked soonest. It was written before
any data was entered, and the rule below is followed whatever the data say. Changing the rule needs a dated entry at
the bottom, with a reason that does not come from the data.

## 1. The arrival assumption is recomputed each quarter, mechanically

ARRIVAL_CDF stays an equal mix of the same three sources (world.py comments):

| Source | What is read | Where |
| --- | --- | --- |
| Forecasters | Metaculus "first general AI system devised, tested and publicly known": community quartiles | Metaculus question page (or a dated secondary write-up if the site refuses automated access) |
| Researchers | AI Impacts survey of machine-learning researchers, high-level machine intelligence | the latest published wave (2023 until a new one appears), conditioned on no arrival before the update date |
| Expert panel | Forecasting Research Institute LEAP, the AGI question closest to near-free cognition | the latest wave |

At each quarterly model release, the newest value of each source is entered in `data/ai_ledger/sources.csv`. Then
`scripts/ai_ledger.py` rebuilds the mixture with the same method, and the new knots replace ARRIVAL_CDF. Nobody chooses
the knots by hand. If a source stops being published, its last value is kept and marked stale.

## 2. Four indicators are scored, and do not move the assumption

These test the simulator's AI channel against the economy, quarter by quarter. Each is entered with its source and
retrieval date in `data/ai_ledger/indicators.csv`:

| Indicator | Source | Why |
| --- | --- | --- |
| US nonfarm business labour productivity, growth over 8 quarters | BLS Productivity and Costs | the AI dividend should show here first |
| Hyperscaler capital spending (Microsoft, Alphabet, Amazon, Meta), share of US GDP | company filings, BEA GDP | the investment the dividend needs |
| US data-centre electricity use, share of total | EIA, LBNL | the physical footprint |
| US labour share of nonfarm business income | BLS | where the gains go (rent capture) |

Each indicator is compared with the simulator's distribution for the same quarter, in the worlds where near-free
cognition has not yet arrived. The comparison uses its percentile: a value outside the 5–95% band two quarters running
is reported as a miss of the AI channel (too slow or too fast). It is reported, and fixed only at a quarterly release
under docs/adopt_rule.md.

## 3. What would change the headline between releases

Nothing. The headline changes only at quarterly releases (docs/adopt_rule.md, release policy).

## Changes to this rule

(none yet)
