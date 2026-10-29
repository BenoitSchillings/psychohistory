The predictive models were improved in a single 12-hour round, judged by one number: the mean, over the four five-year crises, of AUC plus Brier skill score. All tuning used forecasts made from 1975–2005 (the development set). Forecasts made from 2010 and 2015 were held back as a lockbox, and one-year forecasts for 2020–2023 were kept as a fresh test.

### Result

All numbers are from the final, corrected code (after the second independent review).

- **Development set:** 0.849 → 0.933. That is the honest, fully nested estimate, where combination weights are chosen without the origin being scored. The final policy scores 0.946 there, but its two task exceptions were chosen on these same data.
- **Locked 2010–2015 set:** 0.987 → 1.014.
  - Ranking (AUC) improved for every crisis: civil conflict 0.911 → 0.919, economic collapse 0.792 → 0.805, autocratization 0.724 → 0.767, coups 0.822 → 0.830.
  - Brier skill improved for autocratization (+0.02 → +0.08) and economic collapse (+0.13 → +0.15), but slipped for civil conflict (+0.28 → +0.27) and coups (+0.26 → +0.24).
  - The lockbox was consulted at three milestones (eight scorings), once to help choose the autocratization rule, so it is mildly contaminated.
- **2020–2023 one-year forecasts:** 0.865 → 0.892.
  - Conflict onset: AUC 0.859 → 0.901, but Brier skill fell (+0.14 → +0.07).
  - Conflict ending: AUC 0.638 → 0.707, Brier skill +0.02 → +0.05.
  - Coup attempts: AUC 0.871 → 0.883, Brier skill +0.06 → +0.07.
  - These samples are tiny (10–14 events each). This test was also used once to reject a recalibration step.
- **The three test sets overlap.** Their outcome windows share years (development 2006–2009 rows look ahead to 2014; the 2015 lockbox origin looks ahead to 2024), so none is fully independent of the others.

### What helped

- **Institutional detail from V-Dem** (+0.025): judicial and legislative constraints, media censorship, repression, protest mobilization for and against democracy, the military's role in the regime, election quality, election timing, and five-year erosion of those institutions. Autocratization AUC rose from 0.74 to 0.80.
- **Geographic contagion** (+0.007): distance-weighted neighbour conflict, coups and democratic change, plus war intensity and interstate involvement. Later, neighbours' growth, recession depth and inflation (+0.006) mainly helped economic collapse.
- **Tuning the gradient-boosted trees** (+0.03): shallow trees (seven leaves), strong regularization, slow learning and feature subsampling.
- **Leader data** (+0.002): tenure from V-Dem's leader names (every turnover, through 2025), plus REIGN's age, military background, and delayed and anticipated elections.
- **A different blend for autocratization:** gradient-boosted trees and the logistic model, averaged in log-odds and without the neural network, raised the development score for this crisis from 0.93 to 0.98. Adding trees trained on history back to 1860 added only a little more (+0.005).
- **Specialized combinations:** a plain average of GBM and neural network for most crises; a log-odds blend of short- and long-history GBMs with logit for autocratization; a quantile-input neural network with logit for economic collapse. Each choice held up when the weights were picked on six development origins and tested on the seventh.

### What did not help

- **Macro-financial indicators** (exchange rates, current account, interest rates, money growth) and **oil-price cycles** added noise.
- **Cumulative long-memory counts** of past coups, war years and collapses added nothing beyond the recent-history features.
- **Recalibrating on the most recent eight years** of each training window was too noisy.
- **Recalibrating with maps fitted on the development forecasts** looked good there (coups: Brier skill +0.20 → +0.22) but failed on fresh 2020–2023 data. Fitted on models trained with less data, the maps inflated conflict-onset probabilities (Brier skill +0.04 → −0.15), so all such maps were removed.
- **Feeding the neural network a GBM-selected subset of features** made it worse.
- **Freely searched weights** for civil conflict and coups did not beat a plain average out of sample.
- **UCDP violence intensity** (battle deaths, fighting between non-state groups, and killings of civilians, 1989+) did not help either.
- **A single-task network** for economic collapse did no better than the multi-task one.
- **Distilling the trees into the simulator's neural network** improved the five-year scores, but not the one-year hazards the simulator needs. It was kept out of the forecast.
