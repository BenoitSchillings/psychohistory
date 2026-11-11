# Psychohistory

A statistical model of political and economic crises for every country, after Asimov's
*Foundation*. It learns from 80 years of country-level history and simulates thousands of
alternative futures to 2065. Like Seldon's science, it speaks only in probabilities about
large populations; it cannot foresee individual events or leaders (the "Mule").

A study conceived and directed by Benoit Schillings.

- Interactive report: https://benoitschillings.github.io/psychohistory (a second copy, with reader notes: https://seldon-psychohistory.netlify.app)
- Methodology: https://benoitschillings.github.io/psychohistory/methodology.html (source: `docs/methodology.md`)
- Data sources: https://benoitschillings.github.io/psychohistory/data_sources.html
- Engineering log of every test, change and rejected idea: `docs/world_system.md`

## Data

The repository holds the code, the processed panel (`data/panel.npz`, `data/extra_*.npz`,
`data/interstate_df.parquet`) and the main results in `outputs/` (forecast, report, validation and
benchmark files). The raw downloads (about 4 GB) are attached to the GitHub release
`data-2026-10-29`; restore them with

```bash
bash scripts/fetch_data.sh          # everything, into data/raw/
bash scripts/fetch_data.sh events   # one folder only
```

The release predates the sources added in November 2026 (Ember electricity, the UN age-group file and the
conflictforecast.org news topics that the 12-month model uses since version 56): until a newer data release is
made, rebuilding the current version needs those downloaded by hand (`DATA_LICENSES.md`, last section).

Some sources do not allow redistribution, or publish no licence; they are not in the release.
`DATA_LICENSES.md` gives each source's licence and attribution, and the download page of every
source left out. `python run.py data` re-downloads what it can automatically.

## Quick start

```bash
pip install --user -r requirements.txt   # the versions the published forecast was built with
python run.py all          # download data, backtest, hindcast, forecast, build the report
xdg-open outputs/seldon_report.html
```

`python run.py all --quick` is a faster, lower-fidelity run. Data is cached in `data/raw/`;
`--refresh` re-downloads everything. A CUDA GPU is used when available (the 10,000-world
simulation takes ~10 s on an RTX 5060 Ti); CPU works but is slower.

| Command | What it does | Main outputs |
|---|---|---|
| `run.py data` | download sources, build the country-year panel | `data/panel.npz` |
| `run.py backtest` | rolling-origin backtest of the crisis models | `outputs/backtest_summary.csv`, `calibration.csv` |
| `run.py hindcast` | run the simulator from past origins and score it (the testing rule uses `scripts/hindcast_tag.sh`: 16 origins 1970–2009, three refits; `docs/adopt_rule.md`) | `outputs/hindcast_summary.csv` |
| `run.py forecast` | backtest + hindcast + interstate module + 2025→2065 world simulation (climate, AI and alliance scenarios) | `outputs/forecast.json`, `country_forecasts.csv` |
| `run.py states` | US state module (backtest, 2025–29 risks, stress index, climate) | `outputs/us_states.json` |
| `scripts/run_panel.py` | ask two AI analysts to review the five-year forecasts (needs their command-line tools; see psychohistory/llm_panel.py) | `outputs/llm_panel.json`, appends to `outputs/llm_ledger.jsonl` |
| `scripts/score_2025.py` | score the 2024-origin models on 2025 outcomes (UCDP 26.1, V-Dem v16) | `outputs/score_2025.json` |
| `scripts/monthly_nowcast.py [--vintage]` | monthly conflict model vs ViEWS on 2025 (`--vintage`: only data available in Jan 2025) | `outputs/monthly_2025*.json` |
| `scripts/chronicles.py` | narrate four simulated worlds from their event logs (needs an LLM command-line tool; see the script) | `outputs/chronicles.json` |
| `scripts/paired_bootstrap.py` | paired cluster-bootstrap test of a feature family | console |
| `python -m psychohistory.us_history` / `.seshat` | long-cycle tests (US 1790–2025; Seshat crises) | `outputs/us_history.json`, `outputs/seshat.json` |
| `run.py report` | build the interactive report | `outputs/seldon_report.html` |
| `scripts/ablation.py` | test whether each data family improves prediction | `outputs/ablation.csv` |

The report inlines `outputs/narrative.md` (written commentary) and `outputs/review_response.md`
(the independent review and what was done about it) when present.

## Tests and releases

```bash
python -P -m pytest -q tests                     # smoke tests and a regression test for every bug found (about 2 minutes)
bash scripts/release.sh 57 "11 November 2026"    # AI rebuild, tests, forecast chain, ledgers, number checks, report
bash scripts/publish.sh                          # after reading the report and pushing: both sites, checked live
```

A new version is released whenever a tested improvement or a bug fix is ready (`docs/adopt_rule.md`,
`docs/release_checklist.md`). Every release freezes its forecast in `outputs/ledger/` for later scoring.

## What it predicts

Four "Seldon crises", each within five years, for every country:

| Crisis | Definition | Source |
|---|---|---|
| Civil conflict onset | a new intrastate armed conflict (≥25 battle deaths/yr) in a country at peace | UCDP/PRIO 25.1 |
| Coup attempt | any successful or failed coup | Powell & Thyne via V-Dem |
| Economic collapse | GDP per person falls ≥10% below its starting level | Maddison + World Bank |
| Autocratization | V-Dem electoral democracy index falls by ≥0.10 | V-Dem v15 |

and, through simulation, the joint trajectories of conflict, coups, GDP per person and
democracy for all countries to 2065.

## How it works

```
data.py      33 OWID/World Bank/UN series + V-Dem (ideology, religion, coercion, coups),
             UCDP conflict, EPR ethnic exclusion, COW World Religion Project,
             Global Macro Database (debt, crises)  →  175 countries × 1940–2100
features.py  70 structural features in 9 groups (one torch implementation, shared by
             training and simulation) + 7 outcome labels
models.py    SeldonEnsemble: tuned LightGBM (post-war and since-1860), PsychoNet GPU ensembles
             (plain and quantile inputs), extra trees, logit; per-crisis combination rules and
             recalibration from ensemble_config.json · naive Fearon–Laitin benchmark
backtest.py  rolling-origin evaluation: train only on outcomes observed by year T
simulate.py  the "Prime Radiant": Monte Carlo simulation of all countries jointly
pipeline.py  backtest → hindcast → forecast → outputs/forecast.json
climate.py   ERA5 observations + CMIP6 projections (4 SSPs, ensemble p10/median/p90) per country
us_states.py state-level module: BEA, BLS, FHFA, CDC, Mother Jones, MIT, Shor–McCarty, NOAA
report.py    outputs/forecast.json (+ us_states.json) → outputs/seldon_report.html
```

Feature groups follow structural-demographic theory (Goldstone, Turchin): immiseration,
demography (youth bulge, urbanisation), elite overproduction and inequality, state and
regime, violence and contagion, climate shocks, ideology and religion, state capacity and
coercion, fiscal stress and coups. Variables that trend world-wide (internet and mobile
adoption, world conflict share, temperature level) are excluded because they act as
calendar dummies.

**Simulator.** Each scenario steps all countries forward a year at a time:

1. features are recomputed from the simulated state;
2. conflict onset, conflict termination and coup attempts are drawn from the PsychoNet hazards
   (aligned at the origin so their average levels match the previous decade's observed rates;
   about half of coup attempts succeed, as historically)
   (each scenario uses one ensemble member, so model uncertainty enters the fan), shifted
   by slowly varying AR(1) "world mood" terms at global and regional level (with a convexity
   correction so they widen the fan without raising the average hazard);
3. GDP per person and democracy follow ridge-regression transition equations
   (conditional convergence to the rich-world frontier, conflict and coup effects) plus
   residuals resampled from history at four levels: world (a whole historical decade),
   region (booms and lost decades), country (decade-long growth regimes) and year;
4. demography follows the UN WPP medium variant; urbanisation, child mortality and
   education follow damped trends; inflation and unemployment revert to each country's norm;
   rights and rule-of-law indicators move with simulated democracy; slow variables (ideology,
   religion, inequality, debt) persist; coup and financial-crisis memories decay and are
   regenerated by simulated events. Hindcasts extrapolate demography from the origin instead
   of using the realised (or UN) path.

## World system

The simulator couples countries through a model of 2,117 politically relevant pairs of states
(`psychohistory/interstate.py`), alliance cascades, food, energy, pandemics and an AI
"near-free cognition" scenario layer (`psychohistory/world.py`). The design, the split between
learned and assumed mechanisms, and the world-war retrodictions are in `docs/world_system.md`.

## Climate (to 2065)

Country climate comes from the World Bank Climate Change Knowledge Portal: ERA5 annual
temperature, precipitation, consecutive dry days and hottest day (1950–2022), and CMIP6
multi-model projections for SSP1-2.6, SSP2-4.5, SSP3-7.0 and SSP5-8.5 (median, p10, p90),
joined to the observations with the delta method. In the simulator each world draws one
climate sensitivity (shared by all countries) inside the model spread, plus year-to-year
weather noise. Weather shocks enter the hazard models (they added no backtest skill), and the
Burke–Hsiang–Miguel (2015) temperature–growth function, evaluated at population-weighted
temperature, lowers reported GDP for warming beyond 2024. The forecast runs under every
scenario, with and without damage.

## US states

`us_states.py` forecasts five state-level crises five years ahead (real income shock,
unemployment surge, housing bust, overdose surge, public mass shooting) with a logit + GBM
pair, a rolling backtest, Platt recalibration and an explicit skill verdict per outcome
(overdose surges showed no skill and are not forecast). It also builds a Turchin-style
structural stress index and state warming paths (national CMIP6 change scaled by each
state's observed warming trend). The Census API now needs a key, and FBI crime downloads
were unavailable, so education, inequality, age structure and crime are not yet included.

## Model improvement workflow

`psychohistory/evaluate.py` scores any model on forecasts made from the 1975–2005 development origins
(score = mean of AUC + Brier skill over the four crises); forecasts made from 2010 and 2015 are a
lockbox used only at milestones. Experiments are logged to `outputs/experiments.csv`:

* `scripts/exp.py NAME [--drop group] [--model ...] [--first-year 1860]`: one experiment
* `scripts/tune_gbm.py`, `scripts/tune_net.py`: random hyperparameter search
* `scripts/cache_preds2.py SPLIT TAG --leaders` + `scripts/search_weights2.py TAG`: cache each model
  family's predictions and choose per-crisis combinations with nested leave-one-origin-out selection
* `scripts/fresh_eval.py`: one-year-ahead check on 2020–2023
* the chosen combination lives in `psychohistory/ensemble_config.json` (used by `SeldonEnsemble`)

## Validation

All numbers are out-of-sample. See the report's *Validation* section or the CSVs.

* **Backtest** (origins 1980–2015, five-year horizon): the ensemble is compared with
  each component and with a naive model (wealth, size, conflict history).
* **Calibration**: predicted versus observed frequency, by probability bin.
* **Hindcasts**: the simulator, trained only on data before 1975, 1985, 1995 or 2004, is
  run forward 20 years and checked against what happened (ranking AUC and band coverage).
* **Ablation**: each new data family (ideology and religion; state capacity; fiscal stress
  and coups) is added to the structural core and kept only because it improved AUC.

## Known limits

* No "Mules": new technologies (AI), great-power war, pandemics or a leader's choices
  appear only as generic shocks resampled from 1950–2024.
* Interstate war (e.g. Russia–Ukraine) is not an outcome; the conflict target is civil conflict.
* In hindcasts the median GDP path ran ~12% too high after 20 years (−8% to +36% by era); rich
  economies grow too fast and growth miracles (China 1995–2015) too slowly. Fans cover most outcomes.
* Population follows the UN WPP 2024 probabilistic projections (each world draws its own fertility and
  migration path); slow variables (education, urbanisation, institutions) evolve by fitted equations.
* Since the 1990s countries drift apart more in the model than they did (economies became calmer than any earlier
  history suggested), world recessions are too rare, and the post-2000 democratic decline sits in the low tail of the
  simulated range (docs/world_system.md, limits rounds 1 and 2).
* The GDP fan width (`growth_regime_scale`) and the world-mood volatility were tuned on the
  same hindcasts used to check them, so their coverage is mildly optimistic.

## Data sources

Our World in Data (Maddison Project, UN WPP 2024, World Bank WDI, V-Dem, UNHCR, UNODC,
FAO, WID, Copernicus ERA5), V-Dem v15 full dataset, UCDP/PRIO Armed Conflict Dataset 25.1,
ETH Zürich Ethnic Power Relations 2021, Correlates of War World Religion Project,
Global Macro Database (Müller et al.), REIGN (country codes only). Please cite the original
sources when using the outputs.

## Data licences

Most sources are open (CC BY or public domain). Some restrict reuse; the report and outputs contain only derived
statistics, never their raw records:
- ICOW territorial claims (tested, not used in the model): the authors ask that the data not be redistributed.
- Seshat Equinox / CrisisDB: non-commercial use with attribution (Turchin et al. 2015); used only for the
  long-cycle analysis.
- ICEWS (tested, not used): Dataverse terms give limited usage information.
- ViEWS forecasts: used only for scoring.
Cite the original sources when using the outputs.
