The code was given an independent adversarial review (read-only, from the code alone). It raised 24 findings. Here is what was done with each.

### Fixed

- **Autocratization impossible below 0.10 (2.1).** A 0.10 drop cannot happen when the democracy index is already below 0.10, yet those rows counted as "no event", flattering the scores. Closed autocracies are now outside the risk set.
- **Censoring dropped the worst collapses (2.2).** When a state collapses, reporting often stops, and the row was discarded. An event seen in any observed year now counts; only a non-event needs the full five years.
- **Pre-independence "peace" (2.5).** Territories such as South Sudan before 2011 were coded peaceful because their wars were recorded under the predecessor state. Conflict is now unknown before state entry (taken from the start of the Powell and Thyne coup series).
- **Simulated crises counted drawdowns from before 2025 (2.4).** The simulator compared against a five-year peak that reached back before the origin. Only simulated years count now.
- **Hindcasts saw the real future demography (1.3).** Hindcasts now extrapolate population and age structure from the origin year; only forecasts use UN projections.
- **Future survey waves leaked into religion data (1.4).** The five-yearly religion data were interpolated between waves; they are now held forward.
- **Top-decile threshold pooled across origins (1.5).** It is now set within each backtest origin.
- **GDP and democracy shocks could pair different years (3.1).** Both equations now share one fixed year grid.
- **Missing GDP became $0 fans; temperature baseline jumped (3.2, 3.3).** Initial states bridge short reporting gaps, missing paths stay empty, and the temperature trend is anchored at the origin.
- **Residual pools over-sampled alphabetically early countries (3.6).** Sampling is now uniform within each pool.
- **Single-observation volatility read as zero (3.5).** It is now treated as unknown.
- **Missing-data effects hidden in the drivers (4.2).** The bracket now spans all three model families, and missing-data effects appear as their own driver group.
- **Onsets confused with ongoing wars (4.3).** Hindcasts now also report a new-onset AUC for countries at peace at the origin, and 1946 is no longer counted as a year without onsets.
- **The world-mood random walk drifted upward (5.1).** It is now a stationary process with a convexity correction, re-calibrated (σ = 0.2, ρ = 0.9).
- **The short 2020s decade was over-sampled (5.2).** Historical decades are drawn in proportion to the years they contain.
- **Frozen conditions in the simulator (5.3).** Inflation and unemployment revert to each country's norm, refugee flows fade in peace, war's health penalty is transient, and rights and rule-of-law indicators move with simulated democracy.
- **Onset risk reported for countries already at war (2.3).** Those countries are shown as "ongoing".

### Disputed

- **"Backtest feeds realised future features" (1.1) and "multi-task leakage" (1.2).** In this backtest a model is frozen at year T and then applied, year by year, to the data as it is published, the standard expanding-window design. Every label used in training was observed by T, and every feature used to forecast was observable on the date of the forecast. Fitting the standardiser on features (never labels) up to T is likewise not a leak.
- **"EPR double-counts North and South Vietnam / Yemen" (2.5b).** The southern states are not mapped to any code, and the maximum excluded share in the panel is 0.98.

### Acknowledged, not fixed

- **Dispersion tuned on the hindcasts (4.1).** The two dispersion settings were chosen with the same hindcasts that check them; this is stated in the limits.
- **Direct risks and simulator can disagree (5.4).** The two estimates come from different model chains; both are shown side by side.
- **Slow variables still frozen (5.3, part).** Ideology, religion, inequality and debt are still held at their last value.

## Second review (after the model-improvement round)

The same reviewer was asked to audit the improvement round: new features, training windows, the evaluation protocol and the new simulator dynamics.

### Fixed (second review)

- **The rejected macro-finance and oil features were still in the production feature set** (only blanked in experiments). They are now removed from the code.
- **Spain was placed at the Canary Islands** in the neighbour weights (overseas territories share country codes), and Namibia's code "NA" was read as missing. Each country now uses its own mainland centroid.
- **Leader data after August 2021 only registered coups.** Tenure now comes from V-Dem's year-end leader names, which record every turnover through 2025. REIGN's other attributes are carried forward only while the same leader stays in office, and are otherwise unknown.
- **The production quantile network treated missing values as zeros** when estimating quantiles. It now ignores them, as in the evaluated version.
- **Simulator:**
  - Closed autocracies were given elections; now only countries that held one in the past 15 years keep holding them.
  - Every coup attempt succeeded; now the historical success rate applies, and only successful coups install military rulers and restart the regime clock.
  - Institutional indicators stayed frozen while democracy changed; they now move with it, and their 5-year changes are recomputed from the simulated levels.
  - Only one neural seed drove the hazards; all seeds are now pooled.
  - These corrections, together with the richer feature set, left the network's average hazard levels off. In the 2004 hindcast, simulated wars ended too fast and world conflict drained away. The hazards are now aligned at each origin to the previous decade's observed onset, ending and coup rates, which restored hindcast coverage.
- **Brier skill baselines** in the experiment harness now always use post-1955 climatology.
- **Reported numbers:** the 2020–2023 test numbers, rounding and the attribution of the autocratization gain were corrected (see the improvement log).

### Disputed (second review)

- **"Realised future features in multi-year test windows."** As in the first review: each model is frozen at its origin and applied year by year to data as it is published.

### Acknowledged (second review)

- **The evaluation splits overlap.** Development, lockbox and fresh test windows share outcome years. For example, events in 2011–2014 count both for the last development origin and the first lockbox origin, and 2021–2024 events also fall inside the 2015 lockbox window. None of the three is fully independent of the others.
- **Survivorship and merged states.** The country set is chosen by 2024 survival, and pre-unification Vietnam and Yemen conflicts are merged into the modern country.
