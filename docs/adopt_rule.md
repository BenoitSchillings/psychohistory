# The adopt rule for simulator changes (fixed 5 November 2026)

This file is written before any decision is taken under it. Changing it later needs a dated entry at the bottom and
a reason that does not come from the result of a pending test.

## Why it changed

An exchange of six independent reviews (5 November 2026) found four problems with the old rule ("tune on 1970–1998,
confirm on 2000–2012, two network seeds"):

- **The test period was used up.** 134 of the 315 runs in `outputs/sim_eval` had been scored on the 2000–2012
  origins, about 60 distinct variants. Decisions on those origins were shaped by what happened in them.
- **The harnesses did not run the forecast.** The 10-year harness used a stand-alone 8-member network without the
  draw of which risk-model family is right; the full-system cache used a 3-member ensemble where the forecast uses 10.
  A change (the ensemble's central offset) passed one configuration and was withdrawn in another, which was itself
  not the forecast's.
- **Pairing removes only Monte Carlo noise.** Paired runs share random draws, so a small consistent difference is real
  *for those networks and origins*. It says nothing about another refit of the ensemble or another decade, and two
  seeds were too few to see that.
- **No single objective.** Twelve statistics × two periods × three horizons × two seeds made almost every change
  "worse on something", and the rare adoptions were exposed to the choice of which comparison counted.

## The rule

1. **Configuration.** Every test builds the simulator with `pipeline.build_sim_env`, the forecast's own construction
   (the risk ensemble from 1860, 10 members, both network seeds pooled, the family draw). `scripts/config_parity.py`
   must show no difference from the forecast other than those a hindcast needs (origin, horizon, number of worlds,
   AI off, extrapolated demography) and the variant's own change.
2. **Harness.** Full-system hindcasts (`scripts/world_hindcast.py`), 15 years, 2,000 worlds, 16 origins:
   1970, 1973, 1976, 1979, 1980, 1983, 1986, 1989, 1990, 1993, 1996, 1999, 2000, 2003, 2006, 2009. Three ensemble
   refits (`--ens-seed 0 1 2`), baseline and variant on the same refits.
3. **One objective.** The composite is the weighted mean of the variant's relative change in each statistic's score
   (CRPS against what happened; Brier for each country's years at war), averaged over the refits:

   | Statistic | Weight |
   | --- | --- |
   | Each country's years at war (Brier) | 0.25 |
   | World share of countries at war | 0.15 |
   | New civil wars | 0.10 |
   | Coups | 0.10 |
   | World GDP per person growth | 0.10 |
   | Spread of countries' growth | 0.10 |
   | Democracy change (population-weighted) | 0.05 |
   | Wars between states | 0.05 |
   | Oil price change | 0.05 |
   | World growth volatility | 0.05 |

   The safest countries' wars and the frequency of falling world GDP are reported as diagnostics only: they count
   rare events and their misses come from one era.
4. **Decision.** A change is adopted only if all of these hold:
   - the composite improves in at least 3 of the 4 decades of origins (leave-one-decade-out: no decade can carry it);
   - the pooled composite improves by more than the noise floor: twice the standard deviation of the composite
     between refits of the baseline itself (`scripts/adopt_score.py --floor`). A statistic whose own floor is above
     10% gets no weight;
   - no weighted statistic is worse by more than 3% pooled.
   `scripts/adopt_score.py BASE VAR` applies the rule and writes `outputs/adopt/VAR_vs_BASE.json`.
5. **What counts as out of sample.** Nothing in 1970–2024 is out of sample any more for the simulator; the rule
   protects against selection, it does not create a clean test. The only clean test is the prospective ledger, and no
   decision may cite an outcome from 2026 on.
6. **Fixes aimed at known post-2000 misses** (the 2010s rise in wars between states, the democratic recession, the
   safest countries' calm since 1995) are fitted to the outcomes that would test them. They go in as labelled
   scenarios, not into the headline, unless they pass this rule on all four decades.
7. **Bug fixes** (code that does not do what it says) are made without the rule, and their effect on the baseline is
   recorded.

## Prospective ledgers

- One canonical frozen forecast per published release, named before any of its outcomes is known
  (`scripts/annual_ledger.py canonical VERSION "release"`; `outputs/ledger/canonical.jsonl`). Other frozen versions are
  scored but are not the release's record.
- Annual ledger: each country's yearly P(in conflict) in Brier score against persistence whose two rates (staying
  at war, starting one) are estimated on the same proxy over the previous ten years. The earlier fixed 0.9 / 0.1
  baseline gave every country at peace a 10% chance of war and flattered the model; it is still reported for
  continuity.
- Power: two to five new civil wars a year means the ledgers will not separate simulator versions for years. They
  test honesty (calibration, coverage), not which version is better.

## Changes to this rule

(none yet)
