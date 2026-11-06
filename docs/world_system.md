# World-system extension: design

Goal: turn the country-crisis simulator into a coupled world system. Each mechanism is either
**learned** from historical data (and validated out of sample) or **assumed** (taken from the
literature or a stated scenario, switchable, and shown with and without it).

## Gap map

| Subsystem | Before | After | Kind |
|---|---|---|---|
| Interstate war | input only ("interstate involvement") | dyad model of politically relevant pairs (contiguity, distance, power ratio and shift, alliances, rivalry, territorial threat, nuclear status, regime mix, dispute history), 1946–2024; wars simulated between pairs and feeding back on both countries | learned |
| Geography | none | landlocked, distance to coast, ruggedness, desert and tropical shares; tested as crisis predictors and rejected (they lowered 5-year skill once income, institutions and history are known), so they act only through mechanisms (interstate distance and contiguity, food and energy) | tested, rejected as predictors |
| Food | calories and cereal yields frozen | arable land per person falls with population; yields respond to warming (−5%/°C, Zhao et al. 2017), recover with income and AI-driven agronomy; food supply per person drives hazards | learned features + assumed yield response |
| Water | none | withdrawals as a share of renewable resources (same ablation result as geography) | tested, rejected as predictor |
| Energy | none | energy use per person, net import dependence, low-carbon share; energy per person caps the AI growth dividend | learned features + assumed ceiling |
| Alliances | none | defence-pact allies join war-level conflicts with probability q; q=0.1 calibrated on 1995–2024 interstate hindcasts, q=0.5 is the 1914-style stress test | calibrated |
| Pandemics | none | COVID-scale shocks at ~2%/yr (Marani et al. 2021), with GDP and mortality hits; AI-biology raises misuse risk and speeds response | assumed |
| AI ("near-free cognition") | none | see below | assumed scenarios |

## AI scenario: intellectual work becomes nearly free

Parameters: arrival year (2027 / 2030 / 2035 / never) and adoption speed. Each country adopts
along a logistic curve whose timing depends on readiness (electricity and internet access,
tertiary education, state capacity, income).

When cognition is nearly free, the binding constraints become physical. Each channel below is a
stated assumption with a magnitude, and the report shows forecasts with and without it.

1. **Growth dividend, capped by physical bottlenecks.** Up to +3 points of annual growth per
   person at full adoption, scaled by the cognitive share of output and capped by energy per
   person relative to the frontier. Countries rich in energy, robots and capital capture most of it.
2. **Elite overproduction and labour.** The market value of credentialed cognitive work collapses.
   During the transition unemployment rises and the top income share and inequality grow. These
   feed Turchin's instability channels, which the model has already learned.
3. **Development ladder.** Middle-income services and manufacturing exporters lose their
   export-led path as robotics reshores production, a growth penalty during the transition. Free
   expertise raises state capacity, rule of law and health over time.
4. **Science acceleration.** Biology brings faster falls in child mortality, rising crop yields
   and a higher pandemic hazard (engineered pathogens). Materials and energy make low-carbon
   energy cheaper, so the energy cap loosens. Mathematics and software feed the growth dividend.
5. **Finance.** Correlated algorithmic risk: the rate of financial crises is scaled up (or down
   in an optimistic variant).
6. **Power and war.** Cheap autonomous weapons and AI-driven capability gaps raise interstate
   hazards between rivals during the transition.
7. **Surveillance and persuasion.** Near-free surveillance lowers coup and rebellion hazards in
   autocracies but makes autocratization easier. Near-free persuasion raises polarization in
   democracies.

## Outputs

- Ranked five-year interstate risks for pairs and blocs (China–Taiwan, China–US, Russia–NATO,
  India–Pakistan, Iran–Israel, the two Koreas, and others).
- World fans under baseline vs AI scenarios, crossed with climate scenarios.
- A system diagram and the gap map in the report.

## Can it explain the world wars?

The dyad model is refitted using only data available before each war and asked to rank all
politically relevant pairs for the next five years.

- **WWII (as of 1938), 665 pairs:** the actual belligerent pairs rank at the top: Germany–France
  1st, Germany–USSR 2nd, China–Japan 3rd, France–Italy 4th, Germany–Britain 5th, Germany–Poland 8th.
  Median rank 25 of 665. Rivalry, power shifts, territorial claims and recent militarised disputes
  carry the signal.
- **WWI (as of 1913), 415 pairs:** the eastern and Balkan fronts rank high (Germany–Russia 9th,
  Bulgaria–Serbia 18th, Austria–Serbia 24th, Austria–Russia 32nd), but Germany–Britain ranks 186th
  and Austria–Britain 329th. Median rank 54. The war spread through alliance commitments, not
  dyadic grievance. This is why the simulator has an alliance-cascade layer and a 1914-style stress
  test. Structural models explain *where* wars are likely; how big they get depends on alliance
  behaviour that has varied a lot across eras.

## Frontier-model analyst panel

`psychohistory/llm_panel.py` asks two AI analysts to review the top five-year forecasts with
everything they know, including events after the data end in 2024. Their probabilities are shown
beside the model's, never blended in, because they cannot be backtested (they have read the
history). Every judgment is logged with its date in `outputs/llm_ledger.jsonl` for prospective
scoring. Once a few years of ledger entries resolve, a learned blend becomes possible.

## Additions, 2026-10-24

### Data families tested (kept only if the backtest improves; paired cluster bootstrap over countries)
| Family | Result | Decision |
|---|---|---|
| Civil wars 1816–1945 (COW via V-Dem) and coups 1850–1949 (Banks via V-Dem) | dev score +0.0014 | kept (history features) |
| Ruling-party anti-pluralism and populism (V-Party) + protest (Mass Mobilization, NAVCO) | dev +0.0049 [−0.0008, +0.0104], held-out years +0.0008 | kept, not proven |
| Cabinets (WhoGov), leader exits (Archigos), personalism (GWF), election disputes (NELDA) | each lowered the score | rejected |
| Fuel trade (WDI), international crises (ICB), UN voting distance, press geopolitical risk | lowered the score | rejected |
| ICB crises and UN voting distance for country pairs | AUC 0.959→0.961, Brier skill 0.174→0.167 | not adopted |
| Annual event surges (ICEWS 1995–2022, POLECAT 2023–24) | −0.0010 [−0.005, +0.003] | rejected (signal is monthly) |
V-Party's coding ends with 2019 elections. After that, a change of head of government since the last coded
election makes governing-party scores unknown, except for changes on record by end-2024 (US, Poland, Brazil, UK,
Italy, Slovakia), which take the new governing party's own V-Party scores.

### Energy market and transition (simulate.py, world.py)
- Oil price: replayed 1950–2024 annual moves without their trend; pull toward the 1995–2024 mean (0.05/yr, OLS
  0.05 ± 0.05); +0.18 log jump when a major exporter (Gulf states, Iran, Iraq, Russia, Libya, Venezuela,
  Kazakhstan, Algeria, Nigeria) enters a new interstate conflict (OLS +0.27 ± 0.08 including the 1974 embargo
  year); spikes unwind the next year. Exporters' growth: +0.068 × production exposure × price change (± 0.022);
  importers −0.014 per log point of price rise (± 0.013). Hindcasts 1995/2004: actual price inside the 90% band
  every year, inside the 50% band 15–40% of years (the band is too wide).
- Transition: world wind+solar share of primary energy on a logistic fitted to 2005–2024 (rate 0.155 ± 0.03/yr;
  ceiling unidentified, drawn 40–90%); fossil fuels displaced pro rata; oil price moves 0.5 × the demand change
  (assumption); exporters also lose volume; AI adoption speeds the transition by up to 50% (assumption).

### War damage
Estimated on 1951–2023 with country and year fixed effects: a country hosting a war-level interstate conflict
(at home or next door) loses 9.0% (± 2.9) of output growth in each war year and recovers 4.5% (± 2.3) two years
later; expeditionary participants show no measurable cost; larger and smaller belligerents lose alike. In the
simulator this is a level factor on reported GDP (like climate damage), kept out of the growth dynamics that would
amplify it. War-level intensity persists with p = 0.71 and minor conflicts escalate with p = 0.073 (UCDP). Trade
partners lose 0.5 × 9% × (war-site share of world GDP above its 1951–2023 mean of 0.85%), half recovered two
years later (assumption). Allies join only if they are major powers or within 2,500 km of the enemy.

### Crisis explorer and chronicles
`pipeline.crisis_explorer` forces a war (Taiwan 2027, US–China over Taiwan 2027, Russia–Baltics 2028, Gulf 2026,
India–Pakistan 2027, Korea 2028) into every simulated world and compares with the same seed without it.
`pipeline.chronicle_worlds` records event logs for sampled worlds; `scripts/chronicles.py` asks a frontier model
to narrate four of them from the log alone.

### Scored on 2025 (scripts/score_2025.py, scripts/monthly_nowcast.py)
Models trained on data to 2024, scored on UCDP/PRIO 26.1 and V-Dem v16. Monthly model vs ViEWS
(fatalities002_2024_12_t01) on its own target, using only data available in January 2025 (UCDP candidate 2024
events): Brier 0.0268 vs 0.0297, AUC 0.985 vs 0.982; ViEWS better at one month ahead, Seldon at 6–12 months.
The annual model's clearest miss was the United States (electoral democracy 0.84 → 0.74 in 2025).

### Long cycles
`us_history.py`: 1790–2025. Immiseration and polarization move with next-decade political violence (r 0.81,
0.69) but no index beats persistence out of sample, and there is no 50-year cycle in detrended violence.
`seshat.py`: 169 crises in 4,000 years; gaps between crises in a region are no more regular than random
(CV 0.99 vs 1.07); 0.57 major crises per polity-century, rising only slightly with age; faster population growth
precedes more crises (58% vs 42%, n = 52).

### Technology
CHAT (Comin & Hobijn): adoption lags halve about every 78 years; a 2025 technology would reach a typical country
in ~10 years, with ~12 years between early and late adopters (cellphones). This supports the AI layer's 12-year lag
for the least-ready countries; its 8-year frontier adoption is faster than cellphones or the internet (~15 years)
and is labelled an assumption (the 2035 "slow" scenario uses 12). Space is not simulated: no data tie satellite
internet or military space to the crises forecast here.

### Democratic peace and the per-pair anchor
Simulated histories exposed wars between allied democracies (Belgium–Netherlands, US–Canada), which never
happened in 1946–2024 (0 onsets in 10,397 allied-democracy pair-years). Two fixes: the per-pair anchor to the
full 1-year ensemble (logit + GBM) is now kept for the whole run (it used to fade with a 10-year half-life), and
pairs in which both states score above 0.6 on electoral democracy have their onset hazard scaled by 0.15 = (0 +
0.5) / 3.4 (zero observed against 3.4 expected by the dyad model, 1946–2023). Democracy is simulated, so a pair
loses this protection if either side backslides. Expected wars between allied democracies by 2065 fell from 3.9
to 1.5 per simulated world.

## Fixes after the review (2026-10-24, afternoon)

- **Origin moved to 2025.** UCDP/PRIO 26.1 (conflicts to 2025), V-Dem v16 (democracy, liberties, corruption,
  regime type and Powell & Thyne coups to 2025), World Bank / IMF GDP growth for 2025 and refreshed Our World in
  Data series. The model now starts from the 2025 world: the US electoral democracy index at 0.735, 32 civil
  conflicts, coups in Guinea-Bissau and Madagascar. The five-year window is 2026–2030. Raw data for the 2024
  origin are archived in data/raw_backup_origin2024.tgz.
- **Structural uncertainty.** `WorldConfig(sample_params=True)` draws, per simulated world, the alliance-joining
  rate (0.05–0.2), trade spillover (0–1), war-site loss (0.09 ± 0.029), oil price response to demand (0.25–1),
  exporter-war oil spike (0.18 ± 0.08), pandemic frequency (1–4%/yr), the climate-damage multiplier (0.5–1.5) and,
  in AI scenarios, the growth dividend (1–5 points) and autocracy drift. The forecast reports the share of the
  spread in each 2065 outcome that each explains. Finding: outside the AI scenarios, these assumptions explain
  only a few percent; the spread comes from history's replayed shocks and the simulated wars and politics. In the
  AI scenarios the growth dividend alone explains about half of the spread in 2065 GDP.
- **Interstate rate.** The democratic-peace correction is applied before the first-year alignment, and the aligned
  rate drifts back to the 1946–origin rate with a 10-year half-life.
- **Hormuz.** A crisis-explorer scenario with the Strait of Hormuz closed (oil about doubles for a year, an
  assumption).
- **Monthly layer and prospective ledger** (`psychohistory/monthly.py`, `scripts/monthly_ledger.py`). Forecasts for
  October 2026 – September 2027, issued 2026-10-24 from UCDP events to September 2026, are saved unchanged and scored
  as UCDP releases monthly candidate data. Blending the monthly model into the annual model improved the 2025
  forecast on ViEWS's target (fighting located in the country: Brier 0.053 → 0.032, AUC 0.973 → 0.989) but not on
  the simulator's own target (government civil conflict: Brier 0.018 → 0.037), so the simulator keeps the annual
  model and the blend is reported as the territorial forecast.
- **Tested and not adopted:** a democracies-only erosion model (AUC +0.007, 90% interval −0.003 to +0.017); recent-
  window recalibration of the crisis models (helps some tasks on held-out years, hurts on development years).
- **Tests.** `tests/test_smoke.py` (panel, labels, features, parameter ranges, a short world-layer simulation with
  a forced war and sampling, the monthly helpers): `python -m pytest -q tests`.
- **Licences.** See README: ICOW, Seshat and ICEWS restrict reuse; only derived statistics are published.

## AI layer, reworked after an independent review (an independent reviewer, outputs/review_ai_review.md)

Bugs fixed: adoption was identical in every world (now per world, with arrival-year jitter and speed drawn per
world); the energy cap used a frozen 2025 frontier and dissolved itself (now this year's 90th percentile, with
data-centre load taking 2–8% of energy at the steepest adoption); AI-correlated financial crises cost nothing
(extra crises now cost 1.7% + 1.2% of growth, a 1952–2024 estimate); AI-era unemployment, inequality and
polarization never reached the democracy equation (an explicit erosion term in democracies, scaled by graduate
numbers, now does); pandemic frequency x2 and damage x0.5 nearly cancelled (now natural pandemics fall with AI
surveillance and engineered ones form a heavy tail of 5–25% output losses, likelier in a more violent world).
Not adopted: the review called the unsigned power-shift feature a sign error, but the pair model was trained on
the same definition, so its learned coefficient already reads convergence correctly.

New mechanisms (assumptions; each sampled per world over the stated range, see PARAM_RANGES):
- **Two-speed diffusion and the AI race.** Frontier hubs (US, China, then UK, Korea, Japan, Taiwan, France,
  Germany, Canada, Israel, Netherlands, Singapore, India, UAE, Saudi Arabia, weighted) adopt with little lag and
  capture extra rents; everyone else adopts through open-weight and API diffusion 0.5–4 years later plus a
  readiness lag; export controls delay China, Russia, Iran and North Korea by 0–3 years.
- **Chip chokepoint.** Frontier progress runs on a clock that loses 50–100% of a year while Taiwan or South Korea
  is a war site, and world output falls 2–5% (× cognitive share) in such a year (analogue: the 1973 embargo).
- **AI gaps and war.** Between rivals, war risk rises with the gap in AI adoption (0–2 log-odds per unit gap; the
  dreadnought race analogue), and nuclear rivals lose some second-strike confidence at the steepest part of the
  transition (0–1 log-odds). This replaces a flat multiplier on rival major-power pairs.
- **Investment boom and bust.** In every scenario, a 20–60% chance that the 2026–2030 AI capex boom ends in a
  bust costing frontier economies 1–4% over two years (railway mania 1847, dot-com 2001).
- **Robotics after software.** Manufacturing reshoring (the development-ladder penalty) follows robot adoption,
  3–8 years behind software, and robots offset shrinking workforces in societies with median age above 40.
  (Superseded 2026-10-26: robotics now has its own timeline; see "Robotics and military autonomy" below.)

## Extensions, 2026-10-25

### 1. The headline future no longer assumes AI changes nothing
Until now every headline number came from worlds with no AI shock, arguably the least likely case. The headline
forecast now draws, per world, the year near-free cognition arrives (`AIScenario.arrival_cdf`, `ARRIVAL_CDF` in
`world.py`): an equal mix of the Metaculus community forecast for the first general AI system (median about
2031-32, quartiles about 2027-28 and 2041 in 2025-26) and the AI Impacts 2023 expert survey (high-level machine
intelligence 10% by 2027, 50% by 2047). The mix gives 17.5% by 2027, 37.5% by 2032, 57.5% by 2041, 66% by 2047 and
77.5% by 2065; in the remaining 22.5% of worlds there is no shock before 2065. This is an assumption, stated as
one. The arrival year enters the uncertainty attribution (`ai_arrival`), and "no AI shock", 2027, 2030 and 2035 stay
as comparison scenarios. The crisis explorer and the chronicles still compare against no-AI worlds.

### 2. Wars between states: what has already happened, and force short of war
- **Recorded fighting is not forecast.** UCDP candidate events to September 2026 show seven interstate conflicts in
  2026, including a war between Iran and the United States (3,859 battle deaths) and fighting between the US and
  Venezuela and between Afghanistan and Pakistan. The model had given Iran-US 0.1% for 2026-30. These pairs are now
  certain for 2026-30 and start active in every simulated world (no simulated allies join in 2026, since the real
  participants are known). `interstate.observed_after`, `World.interstate_step`.
- **Any use of force.** A third dyad model forecasts any militarised use of force (GML MID hostility level 4-5:
  strikes, clashes, seizures, wars, with or without deaths; 1816-2010; base rate 6% of pair five-year windows).
  Backtest 1900-2005: AUC 0.802, Brier skill +0.148 (naive 0.676). Even this model gave Iran-US only 4.4% for
  2026-30: the pair has no Thompson-Dreyer rivalry (the list ends in 2010 and never coded one), and its many uses
  of force after 2010 are not in the MID data. That gap is the honest lesson.
- **Hostile events between the pair** (ICEWS 1995-2017, POLECAT 2018-2024 rescaled by the dyadic splice ratios;
  3-year means of material conflict, fighting and threats/force posture) now enter the one-year model that drives the
  simulator: origins 2000-2024, AUC 0.965 -> 0.974, Brier skill 0.222 -> 0.252; pair-clustered bootstrap Brier gain
  90% CI [1.4e-6, 5.6e-5], above zero. No gain for the five-year model (0.965 -> 0.961), which keeps its features.
  2025 carries 2024 forward. In simulation each pair's value decays from the origin toward its own 1995-origin
  mean at the annual rate implied by the 5-year autocorrelation of deviations (0.46-0.62; one-year 0.79-0.81).
  Holding them at the 2024 values for forty years froze the tensest recent year in place and raised great-power war
  by 2065 from 35% to 51%; with the decay it is 37%. Pairs with chronically hostile event records (China-Taiwan,
  China-US) keep a higher hazard than before: China-Taiwan 31% by 2065 (previously 16%), China-US 27% (18%).

### 4. Slow variables are no longer frozen
Debt, polarization, mass mobilization (overall, pro- and anti-democratic), political violence, Gini, top-1% share,
extreme poverty and military spending now follow equations estimated on 1960-origin data (`dynamics.py`):
yearly change = reversion toward the country's 20-year mean + responses to growth, civil conflict, democratic
change and financial crises + a shock. Changes are winsorised at 1/99%. Poverty and Gini trend instead of reverting.
The historical average drift is set to zero in simulation (no forty-year extrapolation of past trends). Yearly
residuals contain measurement noise that does not accumulate, so shock sizes are scaled to match the spread of
observed 10-year changes. The 5-year debt change is recomputed from simulated debt. AI shifts to top-1% share and
polarization move the anchor too, so they persist.
- Out of sample (fit 1960-2005, test 2006-2025), R^2 of the yearly change against "no change": debt +0.14,
  mobilization +0.12/+0.13, polarization +0.08, political violence +0.05, anti-democratic mobilization +0.06,
  top-1% +0.01, military +0.01, poverty +0.02, Gini 0.00.
- Simulator hindcasts (1985, 1995, 2005 origins, 20 years, same seeds; `scripts/slow_hindcast_ab.py`,
  `outputs/slow_hindcast_ab.json`): conflict and onset AUCs, GDP and democracy coverage unchanged within
  +/-0.008. The slow variables' actual end values fell inside the simulated 90% band for 74-97% of countries
  (0% when frozen).
- Still frozen: ideology and legitimation, religion, ethnic exclusion, party positions, trade, taxes and spending.

### 7. Against crowd forecasters (`scripts/crowd_benchmark.py`, `outputs/crowd_benchmark.json`)
597 resolved questions from Metaculus, Good Judgment Open and INFER (2015-2026; built from public research
datasets: Halawi et al. 2024, Autocast, ForecastBench and two Metaculus snapshots; Metaculus's own API now needs a
token, see `data/raw/benchmarks/README.md`). Only questions matching a model quantity are scored: 63 questions on
the use of force between two named states (22 pairs, 8 resolved yes). Model: the dyad use-of-force model with
features from the year before the question opened; its five-year probability is converted to the question's window
at a constant hazard. Crowd: one week after opening.

| | Brier | log score |
|---|---|---|
| crowd | 0.100 | -0.33 |
| model | 0.137 | -0.67 |
| always the base rate (12.7%) | 0.111 | |
| leave-one-pair-out blend of crowd and model | 0.110 | |

The crowd wins clearly (model minus crowd 90% CI -0.005 to +0.092, pair-clustered), and adding the model to the
crowd does not help. The model's average was 7% against 13% realised: crowd questions are asked about hot spots at
hot moments (Russia before the 2022 invasion, Gulf states striking Iran in 2026), which a structural model that
sees last year's data cannot know, and short windows punish a constant-hazard conversion. The model's highest
probabilities (China-India clashes after 2020) did not happen. Lesson: for questions about the next few months
in a crisis already in the news, trust the crowd; the model's value is the long-horizon, all-countries base rate,
where no crowd exists. Coups (5 questions), leader exits, ceasefires, territory, elections and currency questions
were not scored: the model forecasts none of them directly.

### 5. Migration (`psychohistory/migration.py`, data in `data/raw/migration/`)
Data: UN WPP 2024 net migration (1950-2100), UN DESA migrant stock 2024 (with origin-destination stocks), UNHCR
refugees, asylum seekers and other people in need of protection (origin x asylum, 1960-2025), World Bank remittances.
- **As predictors: rejected.** Net migration rate, migrant share and its 5-year change, refugees hosted per person
  and remittances: dev score +0.004, 90% interval -0.001 to +0.009 (autocratization AUC 0.852 -> 0.851). As drivers
  of polarization, anti-democratic mobilization or political violence in the slow-variable equations, immigration
  gave out-of-sample changes of -0.003 to +0.005 R^2 with signs that flip between the 2005 and 2010 splits. In this
  country-year data, immigration does not robustly predict the politics of receiving countries, so the simulator does
  not wire it to politics.
- **As a mechanism: adopted.** Refugees from each country move toward a level set by its own history and its
  simulated civil conflict (within-country estimates: log level +0.72 in conflict, +1.1 more at war level;
  adjustment ~22% a year). Departures beyond the origin level leave the country's population and arrive in hosts
  in UNHCR's 2015-2025 proportions (UN projections already contain today's flows). Hindcasts: conflict, GDP and
  democracy scores unchanged; world refugee total from 1995: median 9.8M in 2015 vs 15.8M recorded (inside the 90%
  band in 85% of years); from 2005: 9.7M in 2025 vs 30.5M (inside 20% of years). Refugees from wars between
  states (Ukraine) and from economic collapse (Venezuela) are not generated, so the totals are a floor.

### The 2026 Gulf war and the oil price (`scripts/gulf_2026.py`, `outputs/gulf_2026.json`)
- **Observed oil.** World Bank Pink Sheet (October 2026 release, `data/raw/energy/wb_cmo_monthly/`): crude oil
  averaged $69 in 2025; in 2026 $64-68 in January-February, $96-104 in March-May (Brent $120 in April), $80-84 in
  June-August and $102 in September. The January-September average is 22% above 2025 in real terms (US CPI growth
  assumed 2.7%). Baumeister-Hamilton oil-supply shocks for January-June 2026 are the largest in their series
  (worst month -15.9, against -12.2 in April 2020 and -9.0 in August 1990); the Caldara-Iacoviello geopolitical risk
  index for January-September 2026 is the highest annual reading since it starts in 1985 (187 vs 176 in 2003).
- **Fix 1: oil in 2026 is now observed** (`WorldConfig.oil_observed`, `data.observed_oil`): the recorded change, with
  noise for the unobserved months (sd 0.06); the rise unwinds the following year, like a war spike.
- **Fix 2 (found, not fixed): the war trigger is binary.** Any oil exporter entering a new interstate conflict adds the
  same ~18% jump, so in 2026 the US-Venezuela fighting already triggered it and Iran's war added nothing on top.
  Scaling the jump by the war exporters' share of world output could not be estimated from annual data (excluding the
  1974 embargo, 0.08 +/- 0.71 per unit of world share; US participation in wars abroad removes no US oil).
- **Paired runs from the 2025 origin, 4,000 worlds (recorded war and oil vs no Gulf war, oil simulated):**
  oil index (2025 = 100) 125 vs 116 in 2026, back to 99 vs 98 in 2027. World GDP per person -0.1% in 2026,
  -0.3% in 2027, -0.5% in 2030, -1.6% in 2035 (the war persists in some worlds). Iran -8% and Israel -7% in 2027
  (war-site losses, half recovered two years later), Iraq -2.4% in 2027 and -4.3% in 2030; importers -0.2% to
  -0.6% in 2027 (Korea, Bangladesh, Japan, Pakistan most), Gulf exporters roughly unchanged. Contagion is modest:
  Iran-Saudi war by 2035 5.2% vs 4.4%; Lebanon civil conflict in 2030 18.7% vs 17.1%; great-power war by 2035
  20.9% vs 20.5%. Not modelled: LNG (Qatar) and Hormuz shipping as such, insurance and shipping costs, fertilizer
  prices, and remittances from Gulf workers to South Asia.

### Food: harvests, population and a world grain price (`psychohistory/food.py`, data in `data/raw/food/`)
Data: World Bank Pink Sheet annual real food and grains indices (1960-2025), FAO Food Price Index (real, 1961-2026,
2026 partial), FAOSTAT food balances (old and new sheets spliced at 2010; cereal production, trade, supply, calories),
land use and food CPI.
- **As predictors: rejected.** World real food-price changes x cereal import dependency (1 and 3 years), import
  dependency, cereal production per person, food inflation and relative food prices: dev score -0.001 (90%
  interval -0.005 to +0.004); the two price-shock terms alone +0.001 (-0.003 to +0.004). The literature's link from
  food-price spikes runs mainly to riots and protests (e.g. 2008, 2011), which the model does not forecast; for civil
  war, coups, autocratization and collapse, the spikes add nothing once income, demography, regime and contagion are in.
- **As a mechanism: adopted.**
  - Cereal production = simulated yield x cereal area. The area keeps its 15-year trend (clipped to 0..+3% a year (a shrinking area, often a drought at the end of the window, is not extended),
    damped like yields) until it reaches 90% of the country's arable land; countries already farming all of it (Egypt,
    Bangladesh) cannot expand. Self-sufficiency moves with yield growth plus area growth minus population growth, so
    population pressure raises import dependency where land runs out. (A first version held all land fixed and sent
    land-rich countries such as the Central African Republic and Zambia from 4% to over 60% import dependency.)
  - World real grain price, estimated 1962-2025: d log price = 0.21 x d log real oil (se ~0.06) - 1.19 x last year's
    world harvest-per-person shortfall vs trend (se ~0.45) + the scenario year's historical residual (sd 0.13)
    - 0.05 x (log price - 20-year mean). 2026 is observed (FAO, January-September, +2.4% real).
  - Calories per person: d log kcal = 0.1 x growth + 0.035 x (1 - dependency) x d log cereal production per person
    + 0.001 x dependency x d log world price (FAOSTAT, 8,624 country-years; the price term is not distinguishable from
    zero). The earlier assumption (0.3 x yield change) was about ten times too strong.
  - Calories and yields are model features, so the food balance reaches the hazards only through them, and weakly.
- **Yield trends are shrunk to the world mean.** Each country's projected yield growth was its own 15-year trend.
  Those trends barely persist: regressing each country's 15-year trend on the previous 15 years gives slopes of
  -0.09, 0.01 and 0.20 across three pairs of windows. The projection is now the cross-country mean trend (~1% a year)
  plus 0.04 x the country's deviation (Theil-Sen slopes, robust to a drought in the last year such as Zambia 2024).
  With raw trends, Zambia, Somalia and Madagascar ran declining yields for forty years.
- **Not modelled:** water as a constraint on yields (freshwater withdrawal is in the data but failed as a predictor
  and has no estimated yield effect), groundwater depletion, fisheries, fertilizer prices (only through oil), export
  bans, and sub-national food insecurity.
- Migration fix in the same pass: a country that is the site of a war-level interstate conflict now sends refugees as
  a civil war at war level (assumption; the refugee equation is estimated on civil conflicts only).

## Independent code review, 2026-10-25 (an independent reviewer, read-only)

Four reviews (food; refugees and slow variables; interstate war; AI timing, oil and the crowd benchmark), given the
code and these notes but not our conclusions. Every finding was checked against the code before acting. Outputs:
scratchpad only (not kept in the repository).

**Confirmed and fixed**
- Food
  - Countries missing from recent food balances (Chad, Mali, Sudan, Japan ...) started at 0% import dependency
    (`nan_to_num`). Now: the last value within 15 years, else the region's median. Chad's "2% -> 51%" was this bug.
  - Cropland trend from two end years (2010 vs 2024: a 2010 spike zeroed Chad's trend; Zambia's 2024 drought halved
    its starting area and zeroed its trend). Now Theil-Sen over 2010-2024 and a five-year median start. (Niger's area
    really has been flat since 2011: its trend stays near zero.)
  - World grain-price equation: harvests now from FAOSTAT QCL production, chain-linked over countries observed in
    both years (the 2010 food-balance splice had added ~+8% to world harvest per person in one year); the supply mean
    uses only pre-origin years; the intercept (real grain prices fell 1.5% a year on average) was estimated but dropped
    in simulation, and is now kept; the pull toward the 20-year average is estimated jointly (0.050, the old assumed
    value was 0.05) and runs on a moving simulated average. Re-estimated: oil 0.21, supply shortfall -1.49.
  - Scenario years without a price residual (1955-1969) replayed zero shocks; they now get a fixed draw from the
    observed residuals.
  - Calorie equation fitted on the grains index the simulator moves (was the broader food index) and on d log of
    production per person (was log1p). 2026 grain price observed from the FAO cereal sub-index (+4.6% real), not the
    all-food index (+2.4%).
- Interstate war
  - The simulator's 10-year conflict history started empty (every 2016-2025 conflict forgotten from the second year)
    and counted the current year, which training does not. Now initialised from the data and lagged as in training.
  - Missing dyad features were set to 0 before standardising (e.g. the capability ratio, missing for all pairs after
    2022, read as exact parity); now missing = training mean (z = 0), as in the fitted model.
  - The 1-year onset model was trained on all pair-years including wars in progress, but is used only for pairs at
    peace; it is now fitted on pairs at peace without the war-in-progress columns. Its per-pair GBM offset is evaluated
    at each pair's mean event hostility, not at the 2024 spike (which the decaying logit part carries).
  - 5-year forecasts for wars in progress counted continuation twice (India-Pakistan 87.9% vs 46.5% in the
    simulator); next year's continuation is now a floor.
  - Pairs that have fought are always in the pair set (Israel-Yemen 2025 was missing); NATO accessions after 2018
    (North Macedonia, Finland, Sweden) count as alliances; pairs with several rivalry spells keep all of them.
  - Alliance cascades let allies of both sides of an intra-alliance war join (NATO members against Greece and Turkey);
    shared allies now stay out. Cascade draws use their own random generator, so paired runs stay in step.
  - A 25-death clash by an oil exporter (US-Venezuela 2026) triggered the full war oil-price jump; only war-level
    conflicts do now. The Gulf counterfactual without the Gulf war now holds Iran-Israel (at war since 2025) at peace.
  - Event-feature decay applied one year early; the 10-year GDP history held 2025 twice (11-year lag from 2037).
  - The 2025 scorecard trained the 1-year model on 2024->2025 transitions (the outcomes it scored).
  - The bloc "Any great-power war" counted 25-death clashes; renamed "Any armed clash between great powers".
- Refugees and slow variables
  - Countries with no refugee record at the origin (Taiwan, Serbia) stayed at zero refugees even in a war on their
    soil; returns of today's refugees were added on top of UN projections that already contain them; capped origin
    losses were not matched by host gains. All fixed (departures beyond the origin stock only, capped before sharing).
  - Refugee equation re-estimated in one step (dynamic fixed effects): the two-step version fitted the target
    statically, so refugees still abroad after a war inflated the peacetime level. Interstate conflict on a country's
    soil now counts in estimation as it does in simulation.
  - War sites: a major power fighting a smaller neighbour (Russia-Ukraine; China-Taiwan) was itself a war site
    (-9% growth a year, refugees). Expeditionary powers (now including China) are not war sites against smaller
    powers; two of them fighting each other count as half a site each (assumption). The -9% estimate (1951-2023)
    counted both sides of contiguous wars; this departs from it for the few major-power cases.
  - Slow variables: survey jumps after carried-forward years were fitted as one-year changes (now only changes
    between two consecutive observations); bounds used post-origin data; the anchor is now a moving 20-year average
    in simulation as in fitting.
- AI timing and benchmark
  - Adoption curves were plain logistics already at 10% in the arrival year and positive years before it; they now
    start from zero at arrival.
  - Arrival shares were read at integer years while the arrival distribution's knots are at mid-year.
  - Crowd benchmark: for questions that resolved YES early, the model's window ended on the event date (2 days for
    Russia's Syria strikes); it now ends at the deadline stated in the question (parsed from the title; YES questions
    without one are left out), and wars in progress get the continuation floor. Model Brier at question open 0.137 ->
    0.132 (crowd 0.100), log score -0.67 -> -0.53 (crowd -0.33).

**Checked and not changed**
- Oil exposure "1000x unit mismatch": the oil sensitivity (0.068) was estimated on MWh per person; re-estimating
  gives 0.064 on that scale and 0.026 on kWh, so the code is consistent.
- Subtracting the replayed year's oil move in an observed year: intended (the replayed growth shock already contains
  that year's oil effect).
- `dynamics.validate` train/test overlap: the split is at transitions, without overlap (the docstring year was loose).

**Still open (known, not fixed)**
- Per-person cereal demand is held fixed (no income or price response); food demand uses UN population, not
  population after refugee moves; refugee moves reach population with a one-year lag.
- Net oil importers are still classified by low domestic production, not net trade.
- Variance attribution regresses on the arrival year linearly (2065 adoption is a step-like function of it).
- The chip-chokepoint effect reads the previous year's war sites.
- Hyper-parameters (growth_regime_scale, mood_sd, slow-variable specs) were chosen on the same hindcast windows used
  to score them.

### Second round (same day): follow-up review of the fixes, and a review of the core

The reviewer re-read the fixed code and, separately, the core (features, validation, the main simulator loop). Confirmed and
fixed:
- Introduced by the first round:
  - The moving slow-variable anchor followed values that already contained the AI shift, which was then added again:
    a permanent drift. The anchor now follows the level without the shift.
  - Filling missing refugee years with zero gave Taiwan (absent from the source) a recorded zero peacetime level, so
    an invasion produced ~1,000 refugees. Zero-filling now applies only to countries the source covers; Taiwan and
    Serbia get the median level.
  - Adding pairs that ever fought selected pairs on their future (backtest AUC fell 0.957 -> 0.947); they now enter
    from their first recorded conflict (AUC back to 0.957, Brier skill 0.171 -> 0.178).
  - Including unchanged values for annual V-Dem series and keeping the shock-size calibration capped at 1 left
    their bands too narrow; survey-gap handling now applies only to Gini, top-1% share and poverty (with averaged
    multi-year changes as a fallback when consecutive surveys are too few before the origin), and the shock size is
    scaled in either direction to match observed 10-year changes. Hindcast coverage of the 90% band: 85-96% for most
    variables (50% band: 50-70%); poverty from 1995 69%; Gini and poverty from 1985 still frozen (too few surveys).
- Leaks: the observed next-year grain price was also injected in hindcasts (now only for a year still in progress);
  `observed_after` returned 2025's recorded conflicts to a 2024 backtest (now only years after the last complete
  data year); the crowd benchmark's continuation table used the transition into the question's opening year.
- The 1-year interstate model's calibration ratio was computed on its own training rows (where the GBM half
  reproduces the onsets). It now uses decade-block out-of-sample predictions: observed / predicted onsets 2.09 over
  2015-2024 (the model under-predicted the last decade by half) and 1.08 over 1946-2024.
- Escalation of a minor interstate conflict to war level now counts as a war onset (allies may join, oil reacts).
  Recorded conflicts no longer override a scenario's forced peace; forced pairs that are also recorded do not start
  cascades.
- Civil wars: a new civil conflict always started minor and a war never left war level. Now, from UCDP 1946-2024: a
  new conflict starts at war level with p = 0.155, a minor one escalates with p = 0.115 a year, a war stays at war
  level with p = 0.771. Refugee flows from an interstate war site count as conflict, not war, matching estimation.
- Calories: the income effect is estimated jointly with the harvest and price terms (0.105, close to the assumed
  0.1). Cereal use per person grows with income (elasticity 0.136 at log GDP per person 9, falling 0.018 per log
  point; FAOSTAT 1962-2025, 8,176 country-years); the price response has the wrong sign (stock building) and is
  not used. Food demand uses population after refugee moves, and refugee moves now reach population in the same year.
- Core simulator:
  - Replayed shock years were misaligned: growth and democracy shocks are labelled by the start of the transition,
    oil and grain-price moves by its end (a world replaying "2019" got the 2020 COVID collapse with 2019 oil). Oil and
    grain moves now use the same transition.
  - Nine features were entirely missing at the 2025 origin (models then used missing-value flags learned mostly on
    pre-1950 rows): life expectancy (ends 2023, now carried up to 5 years), heat/rain/drought shocks (ERA5 ends 2022,
    now carried 3 years as the simulator already did), REIGN's election anticipation (ends 2021; now built from
    election timing at REIGN's rate: 30% in years before a due election) and delayed elections, NAVCO campaigns
    (end 2019, now held 6 years). MMP protest memory kept decaying after the data end in 2019, as if no protests had
    happened in 2020-2025; it is now held at its 2019 value.
  - Economic collapse with war damage compared against an all-time peak starting at minus infinity (no collapse could
    register in year one; slow declines over decades counted); now a 5-year peak from the origin, as for GDP.
  - Future temperature paths were anchored on 2013-2022 while the simulator starts from 2016-2025 (including the
    record 2023-2025): a small artificial cooling in the late 2020s. Temperature is now anchored on 2016-2025.
  - The chip-chokepoint GDP hit now uses the same year's war sites.
- Attribution: assumptions are drawn independently, so each share is now the R^2 of the outcome on that assumption
  alone, with the AI arrival year through a flexible basis (years of AI before the end, its square, no arrival).

Checked and not changed (core review): the backtest evaluates each origin's frozen model on the following five years
with the features available in each year, which is what a forecaster would have had (not a leak; overlapping 5-year
windows inflate the counts, so intervals must be clustered); the 5-year ensemble table and the simulator are
different models by design. Still open: the hyper-parameters and feature choices were made on 1975-2005 origins that
the backtest also scores; several simulator states change in one direction only (military regime, regime duration,
holding elections); war and climate damage do not feed back into the political hazards.

### Crowd benchmarks, continued
- Interstate (deadlines from the question text, continuation table as of the question's opening): see
  outputs/crowd_benchmark.json.
- Leaders leaving office (scripts/leader_benchmark.py, new): a yearly LightGBM hazard on V-Dem turnovers with REIGN
  leader attributes, election timing, regime, income, conflict and coups, refitted for each question with data known
  before it opened. On all country-years 2015-2025 it is informative (AUC 0.71, Brier 0.140 vs 0.152 for the base
  rate), but on the 64 crowd questions at opening the crowd wins clearly (Brier 0.088 vs 0.128; base rate 0.121):
  questions are asked about leaders already in trouble, which a structural model cannot see.
- Economic-collapse and civil-conflict questions in the crowd set (currency moves, quarterly recessions, ceasefires,
  battle-death counts) do not match the model's definitions closely enough to score.

### Later the same day
- Backtest split (outputs/backtest_split.csv, report "Origins used for tuning vs untouched origins"): features,
  hyper-parameters and ensemble weights were chosen on the 1975-2005 origins (outcomes to 2014); after the third
  review only the 2015 origin counts as untouched (the 2010 origin's outcomes overlap). Ensemble vs naive AUC on the
  2015 origin: coups 5y 0.88 vs 0.78, economic collapse 0.76 vs 0.62, autocratization 0.74 vs 0.57 (0.85 on the
  tuning origins), coup next year 0.81 vs 0.61, conflict ending 0.75 vs 0.67; but civil conflict 5y 0.90 vs 0.93 and
  onset next year 0.94 vs 0.94. By origin, the naive rule also ranked civil war better in 2000 (0.91 vs 0.90), and
  the ensemble led in the six others by 0.04-0.09 (2010: 0.93 vs 0.84): one untouched origin with 51 events cannot
  settle it.
- Regime states in the simulator change both ways: a successful coup installs military, unelected rule without
  elections; crossing the democracy midpoint upward ends military rule and restores elected government and
  elections; regime duration resets on either kind of change (it used to reset only on coups, and military rule and
  the absence of elections were permanent).
- The convexity correction for the mood noise is var/2 x (1 - 2p) (it was var/2 for all hazards, which lowered
  conflict endings, p ~ 0.2); the coup alignment target no longer counts missing years as no coup.

### Third round (same day): review of the second-round edits
Confirmed and fixed:
- The cereal-demand fit dropped 2010 before differencing, so 2011 was a two-year change across the food-balance
  splice; both food fits now use one-year changes between consecutive years only. World harvest per person uses UN
  population (the food-balance population ends in 2023, which had dropped 2024 from the price equation).
- Moving the refugee step earlier made it read last year's civil-war intensity; the intensity update now comes first.
- Pairs that have fought: entering them only from their first conflict removed their earlier event history (Israel-
  Yemen lost its 2023-24 Houthi events) and dropped pairs first fighting in 2026. They now have all their rows, with
  outcomes (and onset-rate counts) only from their first conflict on.
- Carrying ERA5 rain, dry-spell and heat series forward in the panel made 2022 count four times in the simulator's
  10-year climate baseline; climate baselines and noise now use observed years only (2013-2022 for those three,
  2016-2025 for temperature), the same windows as the CMIP6 anchoring.
- The chip-chokepoint GDP hit applied only in AI scenarios, so the Taiwan and Korea war scenarios had none; it now
  applies in every world (today's economies already depend on those chips).
- Override order: recorded conflicts are applied first, then scenario wars (which respect forced peace and count as
  war onsets when escalating from a minor conflict).
- Survey variables fitted on averaged multi-year changes now use drivers averaged over the same years.
- Recent-decade interstate calibration: the out-of-sample ratio (2.09) came from a model that had not seen 2015-2024
  and was applied to one that had. The recent ratio is now that of the model actually used (1.66, 11 onsets); the
  long-run ratio stays out-of-sample (1.08).
- Backtest split: the 2010 origin's outcomes (2011-2019) overlap those seen in tuning (to 2014), so "untouched" is now
  the 2015 origin only.
Not changed: election anticipation after 2021 is 0.3 in years before a due election (tree models read it as 0; the
linear and neural members read it proportionally) because REIGN's binary flag cannot be reconstructed without months.
- Grain price, revised again after the full run: the moving-average version with the 1962-2025 intercept implied an
  endless decline (2065 median 69, 5% of paths at 22). Real grain prices since 1975 are better described as
  stationary: the equation is now estimated on 1976-2024 with a pull toward the fixed 1975..origin average (0.115 a
  year; intercept -0.013 kept). 2065 median 87 (90% band 53-151; 2025 = 100).
- NAVCO campaign status (ends 2019) was first held to 2025, which kept campaigns such as Russia's opposition
  movement "ongoing" and raised Russia's and Ukraine's civil-war risk (27% -> 40%, 21% -> 30%) on stale data. An
  ongoing campaign is now assumed to continue with probability 0.7 a year after 2019 (expected value 0.7^6 of the
  2019 status in 2025).

### Fourth round (same day)
- Damage feedback (new, `simulate(damage_feedback=True)`): war and climate damage are level factors kept outside the
  growth dynamics, but the political hazards and the democracy equation were fitted on recorded GDP, which contains
  such damage; they now see GDP with the damage (growth keeps the undamaged path, so its momentum terms do not
  compound it). A recession counting the year's damage triggers financial crises as a recorded one would, and the
  collapse check includes climate damage.
- Pairs that entered by fighting: their first conflict year still counted as an onset in the onset rate (with none
  of their earlier peaceful years); they now count only after it. Hindcasts from 1995 and 2004 also simulated pairs
  that became relevant only by fighting later (selection on the future, flattering the pair AUC); a pair is now in a
  run's pair set only once its first fight is known at the origin.
- Food fits: changes against an imputed previous year are excluded (both ends must be observed).
- Hindcast climate trend uses the observed series up to its last year.
- Not confirmed (the reviewer described code that does not exist): a NAVCO mobilization composite, an Archigos leader-age
  carry-forward, and a different deadline parser in the crowd benchmark.
- Untouched simulator hindcast 2015 -> 2025 (the fan-width and mood settings were chosen on the 1975-2004 runs):
  conflict ranking AUC 0.96, new-onset AUC 0.90, world conflict share inside the 90% band every year, democracy
  93% / 53% inside the 90% / 50% bands, GDP 99% / 70% (bands too wide at ten years). Added to the default hindcasts.
- Blending the naive model into the ensemble (logit average, weight chosen leaving each development origin out):
  no gain on the development origins for any outcome (civil conflict 5y: Brier 0.0848 vs 0.0846, AUC 0.832 vs 0.831;
  weight 0 for onset next year and collapse), so not adopted; the naive edge on civil war in 2015 looks like
  single-origin noise rather than a fixable bias.
- Correction: the observed 2026 oil move is +0.224 in log real terms, i.e. +25% (not "+22%" as written earlier in
  these notes and in the narrative).

### Checks after the fourth round (2026-10-26)
- GDP fan width by horizon (hindcasts, 2,000 worlds; share of countries inside the 90% band at 5/10/15/20 years,
  regime scale 1.8): 1975 0.89/0.83/0.78/0.71; 1985 0.95/0.87/0.90/0.93; 1995 0.93/0.93/0.88/0.88;
  2004 0.96/0.96/0.98/0.99; 2015 0.98/0.99. With the regime scale at 1.0 the 2015 bands are as wide (0.98/0.99),
  so the over-wide recent bands are not that setting: the simulator draws historical decades in proportion, and
  the 1970s-1990s were much more volatile for national growth than 2004-2025 (the 1975 run is too narrow by the same
  mechanism). Averaged over origins the 90% band holds about 92% at ten years. Not changed: narrowing the fan to the
  calm recent decades would assume that calm continues.
- Oil importers defined by net trade (OWID consumption > production; covers about half the country-years) instead
  of low production: country fixed-effects growth panel 1966-2022, population-weighted, errors clustered by year.
  Both definitions give a common drag of -0.007 to -0.011 per log point of price rise for all countries and no extra
  loss for importers (importer term +0.012 / +0.014, +/- 0.005). The simulator keeps the -0.014 importer-only
  response (a small term; switching definitions would mainly move China, the US, the UK and Indonesia, whose
  status is unstable in the data).

## Robotics and military autonomy (2026-10-26)
Until now robots were only a lagged copy of the AI curve (3-8 years behind software) with two effects, reshoring
(the development-ladder penalty) and an offset for ageing workforces. Three additions, all assumptions with ranges
drawn per world (`world.py`: `ROBOT_CDF`, `AIScenario.robot_*`, `phys_dividend`, `drone_*`); the "no AI shock"
scenario has none of them.

1. **Own timeline.** Each world draws the year general-purpose robots (able to do most manual work at competitive
   cost) become available, from an equal mix of forecasters (Metaculus "reliable general household robot", median
   about 2032) and researchers (AI Impacts 2023 survey, "install the electrical wiring in a new home", median about
   2040): 5% by 2027, 20% by 2030, 32% by 2032, 56% by 2040, 70% by 2050, 80% by 2065. The draw is correlated with
   the AI arrival year (normal scores, correlation 0.7), and robotics comes at the latest robot_lag (3-8) years
   after near-free cognition. In the headline worlds robotics arrives by 2040 in about 63% of worlds and by 2065 in
   87%, and before near-free cognition in 38%. Deployment is slower than software (10% -> 90% at the frontier in
   8-20 years; physical capital turns over slowly) and reaches less ready countries later (the AI readiness index;
   no open-weight shortcut for hardware). Current robot use (IFR World Robotics 2025, robots per 10,000
   manufacturing workers in 2024: Korea 1,220, Singapore 818, Germany 449, Japan 446, US 307, Taiwan 302, world
   132) is published for only a dozen countries; the readiness index ranks those countries near the top.
2. **Physical automation dividend.** Robots add 0-3 points of yearly growth per person at full adoption, times the
   non-cognitive share of output (1 - cognitive share), capped by energy per person like the cognitive dividend.
3. **Military autonomy** (drones, autonomous weapons) follows whichever comes first of near-free cognition and
   general-purpose robotics, on the faster software adoption clock. Per unit of autonomy:
   - interstate onset log-odds -0.3 to +0.5 (cheaper force, Horowitz; Walsh and Schulzke 2015, against deterrence by
     denial; the sign is uncertain);
   - continuation of war-level interstate conflicts (and staying at war level) +0 to 0.8 log-odds (defence
     dominance and attrition, Ukraine 2022-25);
   - expeditionary powers bear 0-50% of a war site's loss (long-range strikes, e.g. on Russian refineries);
   - civil-war termination -0 to 0.4 log-odds (rebels with cheap strike drones: Myanmar, Sudan).
   Drone warfare since 2020 is already in the training data for the last few years; these terms describe what more
   autonomy would add, and no data constrain them.

Paired test (3,000 headline worlds, 2025-2065, same seed): world GDP per person in 2065 (median) $51.7k without these
additions, $55.4k with the robotics timeline and physical dividend (95th percentile $170k -> $207k), $55.7k with the
drone effects too. Drones raise active interstate conflicts in 2045 from 2.1 to 2.5 and countries in civil conflict
in 2065 from 16.9% to 18.3%; great-power war by 2065 30.4% -> 31.3% (robots) -> 32.3% (all; about the noise
level of 3,000 worlds). Longer wars cost India about 6% of its 2065 GDP per person (China and Pakistan pairs).

### Fifth independent review (2026-10-26): robotics layer and overall methodology
Three read-only reviews (outputs/review_review_robotics_code.md, agy_review_robotics_model.md, agy_review_methodology.md);
each claim checked in the code. Fixed:
- `drone_persist` was added to both the continuation of a war-level conflict and its staying at war level, so its
  effect on war-to-war persistence was about double its stated range; it now applies once (continuation).
- `drone_reach` raised an expeditionary power's war-site score, which also made it send refugees, counted it as a war
  zone for trade spillovers and applied regardless of distance. It is now a separate output loss (range 0-0.2),
  only against an enemy within reach (neighbour, within 2,500 km, or a major power).
- The development-ladder penalty ran on the exporter's own (lagging) robot adoption; reshoring follows the frontier
  economies' adoption, so it now runs on the frontier robot curve.
- Robot deployment was multiplied by the software speed multiplier (actual range 6-30 years, not the stated 8-20),
  and non-hub countries got robots sooner than software; robots now have their own clock, and the hardware lag
  includes the open-model lag.
- Paired scenario runs (none / AI years / 1914 alliances) did not share random numbers: parameters skipped in one
  scenario shifted every later draw, and the engineered-pandemic draws only happened in AI worlds. Every parameter
  and pandemic draw is now consumed in every scenario.
- Without parameter sampling (chronicles), the AI year was shifted by +1.5 years (the spread `arrival_sd` was used
  as a shift); the robotics copula ignored the per-world AI jitter. Both fixed.
- Attribution: in the fixed-year AI scenarios the robotics year was credited with variance that belongs to the AI
  jitter and robot_lag; it is now credited only beyond them.
Changed assumptions (review arguments accepted): physical-automation dividend 0-5 points (was 0-3; first-generation
industrial robots alone added about 0.36 points a year to labour productivity growth, Graetz and Michaels 2018);
rebel/state drones in civil war -0.2 to +0.4 (state drones helped end wars in Ethiopia 2021 and Karabakh 2020);
defaults set to the middle of each sampled range. ROBOT_CDF is now described as technical capability, with cost
parity coming through deployment.
Not changed, noted: the hard cap of robotics at 3-8 years after AI (a softer acceleration would be better); robots
can precede near-free cognition in about 38% of worlds (the forecasters' general-AI question already includes a
robotics test); military autonomy stays at zero without AI or robots (today's drones are in the training data).
Methodology points recorded for later (see agy_review_methodology.md): the naive baseline lacks each outcome's own
lag for coups, collapse and autocratization; five-year coup risks are too high in the upper bins and collapse risks
too low in the lower bins (backtest calibration); the simulator's collapse and autocratization count drops from the
running peak, the labels from the origin level; the family min/max range is not a calibrated interval; the
simulator hindcasts run without the world layer (the interstate hindcast covers wars only); and V-Party and REIGN
gaps after 2019/2021 switch on missing-value indicators for countries with ordinary leader changes.
After these fixes (forecast step rerun, 10,000 headline worlds): world GDP per person in 2065 $59.2k (90% band
$22k-$233k; no AI $32.5k), great-power war by 2065 31% (no AI 27.5%, AI in 2030 35%, 1914 alliances 48%), civil
conflict share 2065 16%. With AI in 2030 the low-income ladder economies now gain 3-18% (India +6%) instead of ending
level or poorer. The no-AI scenario moved slightly ($33.1k -> $32.5k, 28.1% -> 27.5%) because scenarios now share
their random numbers.

## War scale (2026-10-26)
The pair numbers counted any armed clash (UCDP, >= 25 battle deaths in a year), so a Galwan-style skirmish and a
Russia-Ukraine-scale war looked the same. Two additions:

1. **Clash versus war.** Every pair and bloc now also reports the share of worlds with a war-level year (>= 1,000
   battle deaths). The simulator already carried the two levels. While checking them, the share of new conflicts
   reaching war level (35%) turned out to ignore nuclear deterrence: of 5 conflicts between two nuclear-armed states
   since 1946 (China-Russia 1969, India-Pakistan 1999 Kargil, 2013 and 2025, China-India 2020) only Kargil reached
   war level, against 47% of 81 other new conflicts. Shrunk toward the overall rate (beta prior worth 2 episodes) that
   is 0.27 / 0.45 = 0.6 times the usual chance; `nuclear_war_factor` 0.6 (sampled 0.25-1.0) scales both the share
   of new conflicts at war level and the escalation of minor ones for nuclear pairs.
2. **Battle deaths** (`psychohistory/war_size.py`). Each simulated war draws a yearly battle-death rate when it
   starts and keeps it while it lasts; minor years draw from UCDP's minor interstate conflict-years (median about
   100). Rates come from Correlates of War (95 wars 1816-2007, deaths divided by the calendar years each spans) and
   UCDP battle deaths (Russia-Ukraine 2022-24, which also continues at its observed 2024 rate of about 76,000 a year).
   Size depends mostly on who fights: wars between two major powers (12 since 1848, both world wars and Korea
   among them) have a median of about 55,000 deaths a year, all others about 4,000. Tested and left out: the
   belligerents' share of world GDP (+0.09 +/- 0.09 in log10 deaths per log10 share once major-versus-major is
   known), combined military capability (COW CINC) and the post-1945 era (-0.05 +/- 0.14). Draws are a smoothed
   bootstrap on the log scale (+/- about 60%), capped at 3 million a year. Nuclear use is outside the data and not
   modelled; civilian and indirect deaths are not counted.
   Hindcasts (models and war pools limited to what was known at the origin): 1995-2015 actual interstate battle
   deaths 110,000, simulated median 116,000 (actual at the 48th percentile); 2005-2024 actual 246,000 (mostly
   Russia-Ukraine), simulated median 81,000 (actual at the 83rd percentile, inside the 90% band).
Known limit: a China-Taiwan war draws from the pool of wars not between two major powers (Taiwan is not one), so
it is usually far smaller than a war that draws the United States in; the United States joins only through the
alliance cascade, and there is no US-Taiwan defence treaty in ATOP.

## Water and glaciers (2026-10-26)

**Tested first, not adopted as predictors** (`scripts/water_dyad_test.py`, `scripts/drought_civil_test.py`):
- Shared rivers (Oregon State TFDD 2024: 313 international basins, each country's area and population in each), past
  water disputes and cooperation between the pair (TFDD water events, BAR scale, 1948-1999), and scarcity in a shared
  basin (water stress x shared basin). Raw: neighbours sharing a basin had an interstate conflict within 5 years
  2.5 times as often (1.8% vs 0.7%). With the pair model's predictors (contiguity, rivalry, past fighting, power,
  regime), the water features add nothing: 5-year UCDP backtest 1975-2015 AUC 0.957 -> 0.961 but Brier skill
  0.178 -> 0.164; the simulator's 1-year onset model (2000-2024) Brier worse (pair-clustered 90% CI below zero);
  fatal disputes and uses of force flat. Water conflict sits on top of existing rivalries, which the model already
  sees (consistent with Toset, Gleditsch & Hegre 2000; Wolf's Basins at Risk: no war fought mainly over water).
- Drought x agriculture share of GDP x ethnic exclusion (von Uexkull et al. 2016 find drought matters for
  farm-dependent, excluded groups): no interaction coefficient significant for civil-conflict onset or 5-year
  conflict (largest |z| 1.5, triple interaction of the wrong sign); tuned-GBM backtest dev score unchanged
  (0.9035 both), lockbox onset AUC 0.916 -> 0.913.

**In the simulator, as assumptions** (`psychohistory/water.py`, `WorldConfig.water`, ranges in `WATER_RANGES`,
drawn from their own generator so the earlier parameters' draws are unchanged):
- Water supply per country, relative to 2025: glacier runoff for each glacier-fed international basin follows a hump
  `exp(-((t - peak - shift)/tau)^2 / 2)` around its peak-water year (Wimberly et al. 2025, The Cryosphere: median of
  three glacier models x 12 CMIP6 models, SSP2-4.5; Indus ~2056, Tarim ~2056, Aral Sea ~2052, Ganges-Brahmaputra
  ~2029, Mekong ~2026, Salween ~2022), times the glacier share of the basin's flow (`water.GLACIER_FLOW`: Indus 0.25,
  Tarim 0.4, Aral 0.15, Ganges-Brahmaputra 0.05, ...; order-of-magnitude values after Lutz et al. 2014 and Immerzeel
  et al. 2010), weighted by the share of the country's population living in the basin; plus 1.5 x the change in
  precipitation (CMIP6 path; river-flow elasticity, Sankarasubramanian et al. 2001). Sampled: tau 30-60 years,
  peak shift N(0, 10), glacier shares x 0.5-1.5.
- Yields: each year's change in water supply x yield_water (0.1-0.7) x how binding water is (SDG 6.4.2 water stress
  / 100, capped at 1: Pakistan, Egypt, Uzbekistan 1; India 0.66; Bangladesh 0.06).
- Interstate onset: + water_dispute (0-0.5 log-odds) for the pairs in a live water dispute at the end of 2025
  (`water.WATER_DISPUTES`: India-Pakistan (Indus Waters Treaty in abeyance since April 2025), China-India (Medog dam),
  Egypt-Ethiopia (GERD), Afghanistan-Iran (Helmand), Afghanistan-Uzbekistan/Turkmenistan (Qosh Tepa canal),
  Iraq-Turkey and Syria-Turkey (Tigris-Euphrates); not applied in hindcasts), and + water_scarcity_k (0-4) x the
  water lost by the worse-off member of a pair sharing a basin, x how binding water is for it.
- Scenarios (same 4,000 worlds, seed 4): water effects off, sampled, and at the top of their ranges (the Indus
  stress test: water_dispute 0.5, water_scarcity_k 4, yield_water 0.7).
- Not modelled: groundwater depletion (the Indus plain and north India pump far more than glaciers supply), dams and
  diversions changing flows, seasonal timing (earlier melt), domestic water and cities, water-driven migration.

## Missing factors (2026-10-26)

A review of what the model ignored (independent review, `outputs/review_review_missing_factors.md`, checked against the code) and the
response, item by item. Estimates in `scripts/extra_estimates.py` (`outputs/extra_estimates.json`),
`scripts/alliance_join_test.py`, `psychohistory/risks.py`; assumptions in `world.EXTRA_RANGES` (own generator).

**Defects fixed**
- Non-state and one-sided violence decayed at 0.6 a year with nothing to renew them, and were in fact missing for every
  country at the 2025 origin (UCDP death counts end in 2024; now carried forward one year, also battle deaths per
  capita). They are now slow variables (`dynamics.SPEC`): reversion to the country's own 20-year level before the
  origin (a fixed anchor: a moving one ratcheted violence up through long conflicts), driven by growth, civil conflict
  and democratic change.
- Territorial control and fiscal capacity were frozen at 2025 values; now slow variables (territorial control on a
  logit scale, so shocks do not pile up against the 100% bound: a level version drifted the average country down
  5 points). Year-to-year changes of all four are barely predictable (out-of-sample R^2 -0.02 to +0.03); the point is
  that they move, persist and respond to conflict. Shocks for these four are scaled by each country's own variability
  over the previous 20 years (cap 1.5x), in full during civil conflict.
- Party systems (V-Party anti-pluralism and populism of the government, vote-weighted anti-pluralism, the most
  anti-pluralist major party) stayed at their last values; now each simulated election moves them:
  x' = x + rev (x - 20-year mean) + e (rev -0.25 / -0.42 / -0.07 / -0.09, e sd 0.18 / 0.17 / 0.11 / 0.14 per
  election, 1970-2025 V-Party elections; growth had no significant effect; the direction is not predictable out of
  sample, the spread is the point). `dynamics.fit_party`.
- Not a defect: the reviewer reported that ordinary financial crises never cost growth. The growth model replays historical
  years, crises included; only the extra AI-era crises are charged separately.

**War**
- Alliances: formal pacts after ATOP ends (France-Greece 2021, Russia-North Korea 2024, Saudi Arabia-Pakistan 2025);
  security patrons without a treaty join a new war against their client with their own sampled chance (US-Taiwan
  0.25-0.75, US-Israel 0.1-0.5, Russia-Iran and China-Pakistan 0-0.1); allies of those who joined join in a second
  round at cascade_q x cascade_hop2 (0-1). Effect in a 2,000-world test: great-power war by 2065 29% -> 38%,
  China-US war 9% -> 17% (a China-Taiwan war now draws the US in about half the time). Not adopted: alliances
  weakening with regime difference; in COW x ATOP (902 ally-war opportunities, 1816-2007) allies with larger
  democracy gaps joined more often (+3.4 +/- 1.3 log-odds per unit gap). Since 1945, allies near the enemy or major
  powers joined 2.6% of the time; cascade_q (0.05-0.2) stays calibrated on the hindcasts.
- Foreign intervention in civil wars: internationalized civil wars (UCDP type 4) end less often, -0.42 (se 0.20)
  log-odds a year (1,747 conflict-years 1946-2023, with duration, intensity and era); applied as -0.42 x
  (internationalized - 0.23), sampled N(-0.42, 0.20). Internationalized at the origin per UCDP 26.1 (2025); a new war
  is internationalized with the historical share.
- Proliferation: latent states (`risks.LATENT`: Iran 1.5% a year, South Korea and Saudi Arabia 0.4%, others
  0.05-0.2%) start programs (x prolif_mult 0.25-2, x prolif_cascade 1-6 once a rival has acquired), finish after
  2-6 years, and are set back (half the time abandoned) by a war with a nuclear-armed state or a major power; Iran is
  pursuing at the origin in 10-60% of worlds (after the recorded 2026 strikes: no further setback that year). A state
  pursuing a weapon and a nuclear-armed rival that is not its ally get 0-1.5 extra onset log-odds (the preventive-strike
  window). Programs advance at the end of each year, so a program is exposed for its whole breakout time. New weapons change the pair features (both/one nuclear)
  and the nuclear war factor. Test: a new nuclear state by 2065 in 70% of worlds (Iran 50%, Saudi Arabia 22%, South
  Korea 14%), against a historical pace of two to three per 40 years since 1970.
- Nuclear use: per war-level year 0.2-3% between nuclear-armed states, 0.05-1% against a non-nuclear state (no use in
  some 80 such war-years since 1945); between two major powers 5-40% of uses become a large exchange (10-200 million
  deaths; belligerents lose half their output, the world 5-30%); a limited use kills 10,000 to 1 million and costs
  the belligerents 5%. Losses and deaths fall on the side that can be struck (all of them on the non-nuclear side in a
  one-sided use), as a lasting log-level loss. Test: some use by 2065 in ~18% of worlds, a large exchange in ~0.4%.

**Economy**
- Not adopted: public debt as a financial-crisis predictor (+0.08, se 0.08, 1955-2024; growth -10.5 and inflation
  +1.0 matter; out of sample 2000-2024 the debt model ranks better, AUC 0.65 vs 0.54, but its Brier score is worse
  than the two-bin rule, 0.0414 vs 0.0403) and ageing and war as debt drivers (out-of-sample R^2 0.132 -> 0.129).
- Trade: the war spillover now reaches countries through their trade, using gravity weights (partner GDP / distance,
  `risks.trade_weights`; no bilateral trade data was downloadable without registration) x openness relative to the
  mean, scaled by the world's excess share of output at war sites over the historical norm (already in the replayed
  growth): the same world average as before, concentrated on the war's partners.
- Sanctions: a non-Western state at war with a Western one (`risks.WEST`) loses 0-4% of output a year x its Western
  trade share (gravity), as a level loss. The Global Sanctions Data Base needs a request form, so this is an
  assumption.
- Critical minerals: producers gain mineral_elast (0-0.5) x mineral rents (World Bank, share of GDP, up to the
  origin) x 4 x the yearly change in the clean-energy share (timed with installations); mineral rents do not predict civil conflict (coef -0.09, se 0.07; GBM dev score
  0.9035 -> 0.9036), so no conflict channel. While China fights the West, clean-energy and robotics progress lose
  china_pause (0.2-1) of a year (China refines most critical minerals and makes most batteries, magnets and actuators);
  robots run on their own clock, which never runs backwards when the chip and China chokepoints bind together.

**Demography and climate**
- Population: war deaths (battle deaths x 1 + civ_mult, 0.5-3; nuclear deaths) and pandemic deaths (0.05-0.5% of the
  population for a natural pandemic; 0.2-5% for an engineered one) come off the UN path; fertility moves with income
  beyond the projection (-0.035, se 0.021, per unit 10-year change in log GDP per person, 1970-2023 with year
  effects; applied to income against the country's own 20-year trend path after war and climate damage), births at
  0.0071 x fertility. The youth share moves by the extra births x (1 - youth share).
  Migration was already modelled.
- Weather: yearly shocks are drawn with the observed cross-country correlation (5 factors shrunk to the observed mean
  correlation: 0.17 for mean temperature, 0.16 hottest day, 0.04 rain, 0.02 dry days; 1980-2022 ERA5) instead of
  independently, so harvest failures can coincide. Yields respond to them: -1.4% per degC of hottest-day anomaly
  (se 0.35), +5.6% per unit log rain anomaly (se 1.6), 1962-2022 with country and year effects (the level recovers
  when the weather does). Water supply now uses the rain where a country's water comes from (basin shares weighted by
  area x rainfall: Egypt's from South Sudan, Uganda and Ethiopia, Iraq's from Turkey).
- Coasts: output lost = slr_cost (0-10%) x share of people below 5 m (World Bank/CIESIN) x sea-level rise / 0.5 m
  (AR6 SSP2-4.5 median shape x 0.7-1.5); damaging storms in exposed coastal countries become more frequent with
  warming (+0-30% of a 5% base rate per degC), each costing 0.5-3% of growth that year.

**Tail catastrophes**: Carrington-class geomagnetic storms (0.2-2% a year; output loss 1-8% x grid exposure, for the
year), VEI 6+ eruptions (0.3-1.5% a year; world cereal yields -3-12% for two years), and attacks on infrastructure in
great-power wars (0-3% of output for the belligerents while the war lasts). Sea-level losses apply only with climate
damage on.

**Code review (independent review, `outputs/review_review_missing_factors_code.md`), fixes**: weather correlations undefined before 1995
(now independent shocks when fewer than 15 years precede the origin); bounded-variable shock scaling inflated by
dropping unchanged values; fertility measured against the median world (now the country's trend path); the
preventive-war term firing against every major nuclear power (now rivals that are not allies); the recorded 2026 war
setting back Iran's program in every world; second-round alliance joining giving first-round candidates another roll
(now only allies of the joiners, excluding earlier candidates); nuclear losses split onto the user; nuclear damage in
the wrong units; solar and cyber losses becoming permanent; a 31-year block after an eruption; the robot clock running
backwards; party measures stuck at missing after an unrecorded leader change; the Colorado basin (US-Mexico in TFDD,
not the Patagonian river of the glacier study) removed; the glacier-runoff ratio exploding for basins far before
their peak; water loss counting where water is not scarce; World Bank and water-stress data after a hindcast origin;
population deviation weights and the youth share. Kept as designed: sanctions as a growth drag while they last;
storms as permanent losses (cyclone damage to growth paths does not recover in the evidence).

## Toward state of the art: benchmarks, live data, uncertainty (2026-10-27)

The seven gaps named on 2026-10-26, and what was done about each.

1. **Crowd and market signals.** `psychohistory/crowd.py` and `scripts/crowd_live.py` pull open questions from
   Polymarket (gamma public search) and Manifold, classify them (force / war / coup), and match them to model
   outcomes (477 open, 56 matched). On 63 resolved use-of-force questions the best blend gives the model a negative
   weight (-0.06; crowd Brier 0.0998, model 0.132, leave-one-out blend 0.104), so where a crowd exists the report
   publishes the crowd's number. Metaculus is not used (its API needs a token).
2. **Below country level.** `psychohistory/pgm_fatal.py`: PRIO-GRID-month (0.5 degree) fatality distributions on the
   VIEWS grid (Africa and the Middle East, 13,110 cells): cell history, ring and 5-cell neighbourhoods, country
   deaths, location; LightGBM quantile regressions on active cells plus a weighted 5% sample of quiet cells. Boosted
   upper quantiles can cross and run far past anything seen (1e35 deaths in a cell in the first run), so the sampler
   now sorts each row's quantiles and caps them at 1.5x the largest cell-month seen to the origin.
3. **Fresh data.** GDELT 1.0 daily events (`data/raw/events/gdelt_fetch.py`, `gdelt_build.py`), 2023-01 to
   yesterday, reduced to country-month and undirected-pair counts, scaled for missing days and spliced to POLECAT on
   their 18-month overlap (POLECAT/GDELT 0.026 for material conflict, 0.027 fights, 0.042 threats, 0.022 posture).
   The pair model now sees 2025 hostility (top: Israel-Palestine, Russia-Ukraine, Iran-Israel, Israel-US,
   Israel-Lebanon, Cambodia-Thailand, Afghanistan-Pakistan). Interstate hindcast pair AUC rose from 0.928 to 0.948
   (1995-2015) and 0.930 to 0.940 (2004-2024); the other hindcasts did not move. Event counts did *not* help the
   monthly fatality model on the 2010-2017 development windows (CRPS 18.13 against 17.69), so they stay out of it.
4. **Strategic actors and escalation.** Nuclear-armed states, proliferation with preventive strikes, alliance
   joining in two rounds, quasi-allies, sanctions and escalation to nuclear use (sections above).
5. **Economics.** Real bilateral trade shares (CEPII BACI 2015-2021) replace the gravity weights for war and sanction
   spillovers. Debt was tested as a crisis predictor and did not predict; financial-exposure contagion (BIS banking
   data) is not modelled.
6. **Structural uncertainty.** Each world draws one shift per equation family (onset, termination, coups) with the
   spread of the model families' logits, and one per country pair for the interstate model, so the spread across
   worlds now includes "which model is right", not only parameters and shocks. Great-power armed clash by 2035
   (headline worlds): 32.9% to 34.7%; nuclear use 17.3% to 18.3%.
7. **Track record.** Three external benchmarks, below, plus the forecast ledger.

### 2023/24 VIEWS Prediction Challenge (Hegre et al. 2025), replicated

`scripts/predcomp_run.py`, `predcomp_eval.py` (scorer checked against the paper's published numbers), `predcomp_pgm.py`.
Same windows, origins and metrics (CRPS on 1,000 draws, binned ignorance, 90% interval score) as the participants.
Settings tuned only on 2010-2017 (`predcomp_tune.py`): widening each forecast 15% around its median in log space
was the only change that improved all three metrics (CRPS 17.69 to 17.25); mixing in the conflictology benchmark,
event features and extra tail quantiles did not. On the scored years the widening made no difference (47.50 to 47.57).

Country-month, mean CRPS 2018-2023: ShapeFinder 45.10, OMM 46.29, **Seldon 47.57 (3rd of 20)**, conflictforecast
47.89, TFT 48.52, ... conflictology benchmark 51.82. Seldon has the best ignorance (0.585) and the best 90% interval
score (764).

Live window (July 2024 - June 2025, data to April 2024): **Seldon 31.6 / 31.1 (10th)**; conflictology benchmark
23.8 / 27.2 leads. Two causes. (a) The first run gave the model only UCDP Candidate data after 2022, which undercounts
Ukraine five- to fifteen-fold (January 2024: 849 against 12,549 final) and Gaza; the organizers' conflictology
benchmark draws reach final-GED 2023 levels, so participants had GED 24.1 (final through 2023, released June 2024).
With final data through 2023 the score went from 34.8 to 31.6. January-April 2024 still come from the candidate
releases; the organizers may have had revised numbers we cannot reconstruct. (b) The model pulls large ongoing wars
toward their history (Ukraine median 2,400 a month against 5,100 realised; Gaza 140 against 1,500). The live window
was not used for tuning and the model was not changed after seeing it.

ViEWS archive (`scripts/views_archive_benchmark.py`): all 48 archived ViEWS production runs (December 2021 - December
2024 origins), 1-12 months ahead, 102,376 country-months, scored on GED 26.1, Seldon retrained at every origin on the
data that existed then. Probability of a 25-death month: Brier 0.02465 against ViEWS 0.02487 (difference 90% CI
-0.0006 to +0.0002: a tie), log loss 0.0859 against 0.0894, AUC 0.985 against 0.984. Log-fatality MSE (ViEWS's own
metric): 0.469 against 0.507 (CI of difference -0.046 to -0.030: Seldon better). ViEWS is better one month ahead
(MSE 0.325 against 0.350), Seldon at 6 and 12 months.

Grid (PRIO-GRID month, 13,110 cells): the first run's extreme-quantile models put the top 1% of draws in the
thousands for about 2% of quiet cells (86% of its CRPS came from cells with no deaths). A local bound, draws <=
max(5, 2 x the largest month in the cell and its ring over the past 36 months), chosen on 2014-2017
(`scripts/pgm_tune.py`; dev CRPS 0.737 -> 0.366, ignorance unchanged), fixed it. Mean CRPS 2018-2023: UniBW trees
0.4370, 0.4377, 0.4414, D'Orazio 0.4430, **Seldon 0.4451 (5th of 14)**, TFT 0.4479, ... zero benchmark 0.4498;
Seldon's 90% interval score is second (8.42 against 8.32). Live window: **Seldon 0.2267 (2024; 2nd behind the
conflictology benchmark 0.2251, ahead of every participant) and 0.3177 (2025; 1st, conflictology 0.3303, best
participant 0.3327)**. The bound was found after the 2018 scores were seen (the defect was obvious there), but its
setting was chosen on development years only.

**Verdict.** At the country-month level the monthly layer is in the leading group historically (3rd of 20) but not
in the live window (10th), where it under-forecasts the largest ongoing wars; at the grid level it is in the leading
group on both. Against ViEWS's own production system it ties on event probability and wins on fatalities at 6-12
months. Crowds remain better on the questions they cover. The long-range simulator has no comparable benchmark: its
evidence is the hindcasts.

## Round 2026-10-27 (b): what else is missing — tested before adding

Prompted by the user ("any obvious missing data or viewpoint") and an independent review by an independent reviewer
(`outputs/review_missing_review.md`; its claims were checked against the code before acting). Data were collected into
`data/raw/{ocean,innovation,intervention,political_economy,structure}` (each with a README of sources and caveats).
Every candidate was tested on history first; the simulator takes estimates where a test passed and sampled
assumptions (named, with ranges) where it did not but the mechanism is near-mechanical.

**Big-war fix and benchmarks.** The country-month model now predicts the change from the last 3 months (chosen on
2010-2017 only). VIEWS challenge: historical 47.57 -> 47.35 (3rd of 20); live 31.6 -> 34.2 (worse): Ukraine's median
fell (2,416 -> 1,456 against ~5,700 actual) because the residual target anchors on the last three months, which in
the live window come from UCDP Candidate releases that undercount Ukraine (January 2024: 849 against 12,549 final).
On final data (ViEWS archive, 48 runs) it helps: Brier 0.02426 against ViEWS 0.02487 (90% CI of the difference
-0.0010 to -0.0002, now significant), log-fatality MSE 0.439 against 0.507 (before 0.469), and it now edges ViEWS one
month ahead. Not reverted: the test windows did not choose the model.

**Financial contagion** (scripts/contagion_test.py): BIS bank-lending exposure to crisis countries did not predict
banking crises (-0.79, s.e. 0.60, year effects); trade exposure did (4.26, s.e. 0.56). Crises now spread through trade
in a second round, with the first round scaled so the historical crisis rate is kept (0.070 against 0.069); the 99th
percentile of crises in a year rose from 21 to 31 countries (record: 34).

**Adopted (estimates).**
| Mechanism | Test | Estimate | Script |
|---|---|---|---|
| Working-age share | 5-year growth per person, 1960-2023, year effects + growth controls | 1.22 (s.e. 0.17) per unit d log(15-64 share); out-of-sample RMSE -2%; old-age ratio adds nothing | workforce_test.py |
| Leader death in office | Archigos 1900-2015 | logit -9.0 + 0.073 age - 1.28 (year-2000)/100: 2% a year at 75, 4.2% at 85 | succession_test.py |
| Succession after an autocrat's death | coups within 2 years, year effects | +0.68 (s.e. 0.25): 15% against 9%; successor's irregular exit +0.80 (0.28); no war-onset effect | succession_test.py |
| Science capacity beyond income | 5-year growth 1996-2018, articles per million | +0.23 points/yr per log unit (s.e. 0.11); out-of-sample -1.7%; R&D x gap 0.0115 (0.0026) | innovation_test.py |
| Peacekeepers | war ending next year 1990-2023 | +0.32 per log(1 + thousand) (s.e. 0.15); relapse -0.27 (0.24, unused) | intervention_test.py |
| Food-export price falls | coups within 2 years 1966-2022 | -2.8 per unit export-weighted change (s.e. 1.0); oil exporters already covered | intervention_test.py |
| Rebel fragmentation | war ending next year 1950-2022 | -1.93 per log(1 + rebel groups) (s.e. 0.41) | structure_test.py |
| Short-term external debt / reserves | banking-crisis start 1975-2022 | +0.60 (s.e. 0.11) | structure_test.py |
| Coups among language / colonial peers | coup attempts next year | +2.6 (s.e. 0.86) beyond coups within 1,500 km | structure_test.py |

**Adopted (sampled assumptions, mechanism near-mechanical, no usable estimate).** Wild-catch fisheries under ocean
warming (tropics -8%/degC, mid-latitudes -3%, high latitudes +4%) and an overfishing trend, moving calories and income
by each country's fish dependence (ocean.py); the AI growth dividend fading with cumulative adoption (0-4% a year), as
the semi-endogenous frontier fit suggests (US research productivity -2.4% a year since 1950; x10 effective
researchers: +1.8 points at the peak, half gone 25 years later; the fade is poorly pinned down); the 2025 peacekeeping
cut (10-50%) and wind-down; the 2025 USAID cut as lost demand in 2026 (x 0-1); the 2025 US tariffs (partners lose
0-1.5 x exports-to-US/GDP x tariff rise; the US 0-1%).

**Rejected (tested, no gain).** Fish dependence and catch declines (country: onset, collapse); fisheries conflicts
between states (International Fishery Conflict Database; only a borderline gain on low-level uses of force); fast
internet / mobile spread (protest, coups, onset, autocratization); aid falls > 2% of GNI (onset, coups, collapse; the
2025 cut kept only as an assumption); sanctions (TIES, IST: no growth or collapse effect); term-limit evasion attempts;
economic complexity (significant in sample, no out-of-sample gain); coup-proofing (counterweights: attempts, success,
onset); transborder ethnic kin between pairs (14x raw rate, but no gain beyond the pair model on the 1-year onset model);
sovereign defaults (no effect on coups, autocratization, protest, onset or collapse within 2 years, so no default event
was added; the hazard itself is predictable from short-term debt and debt service); public trust (IVS: sparse, null).

**Checked, informative, not changed.** Alliance reliability: invoked alliances were honored 85-94% of the time before
1950 and 17-19% since 1946 (CJL 2026 COW/IWD, 113-141 cases); the sampled ally-joining chance (0.05-0.2) is consistent.
Official GDP in autocracies (Martinez 2022 lights data, 1992-2013): GDP growth rises with lights growth about 1.9x as
steeply in Not Free as in Free countries (interaction s.e. wide; Martinez's own estimate is ~35% overstatement). The
model forecasts official GDP (what it is validated on); real economic power of autocracies is likely overstated.

**Design errors found by the review and fixed (each verified in the code first).**
1. The AI clean-energy gain scaled every country's energy alike and cancelled in the frontier-relative energy cap (a
   no-op): it now lets energy-poor countries close part of their gap to the frontier.
2. Interstate war output loss counted twice for war sites (growth equation's interstate term, about -1 point, plus the
   estimated -9% war-site loss that already is the whole cost): the growth equation's war term is dropped when the
   war-damage layer is on (expeditionary participants showed no measurable cost in that estimate).
3. The workforce term (this round) was added raw: the growth equation already carries the 1960-2023 mean change
   (+0.19% a year), now subtracted; robots offset a share (robot_aging, 0-1) of a shrinking workforce's drag instead
   of a flat bonus for median age > 40.
4. Pandemics counted twice: the replayed historical year could be 2020 while the world layer also draws COVID-scale
   pandemics (about 3.4% a year instead of 2%): 2020 is replaced by 2019 in the replay when pandemics are on.
5. The AI hub rent added to world growth instead of moving it: rescaled so the output-weighted world dividend is unchanged.
6. Great powers were frozen at the 1945 permanent five for war size and the great-power-war count: an India-China war
   drew deaths from the small-war pool and did not count. Now: the COW majors plus any nuclear-armed state with >= 5% of
   world output, re-evaluated every simulated year (India today). The dyad hazard keeps the COW coding it was trained on.
Not fixed, noted: births and deaths beyond the UN path do not feed the working-age share; PsychoNet's out-of-bag early
stopping shares calendar years with training (time waves can leak; the out-of-time lockbox and fresh tests measure it).
Not available: humid-heat (wet-bulb) exposure (ERA5-HEAT needs a Copernicus account), SIPRI arms transfers (manual
export only), EUSANCT (terms dialog), UNCTAD commodity dependence (bot check).

**What moved the forecast (paired runs, 2,000 worlds, uncertain AI; scripts/decompose_round.py).** World GDP per
person in 2065, median, all of this round on: $42.2k; without the workforce term $61.4k (the largest change: the
UN-projected fall in the working-age share costs China about 1 point a year, Korea 1.6, Europe 0.6-1.2, the US 0.5,
consistent with Maestas, Mullen & Powell 2023 for the US; Africa and South Asia gain); without the science term
$37.6k; without the AI dividend fade $45.2k; without the 2025 tariffs and aid cut $42.4k; without fisheries $42.1k.
The GDP hindcasts did not get worse (1975 origin: share of countries in the 90% band 0.725 -> 0.738, in the 50% band
0.275 -> 0.300; 1985: 0.938 -> 0.932, 0.565 -> 0.578; 1995: 0.877 -> 0.877, 0.494 -> 0.519).
Great-power war by 2065 roughly doubled (0.35 -> 0.69 in the first full run) almost entirely by definition: with
India counted as a great power, a China-India war-level conflict (about 52% by 2065 in both runs) now counts, and its
battle deaths come from the major-war pool. The forecast now records both definitions (great_power_war_by and
great_power_war_cow_by, the COW majors only); the report shows both.

## Round 2026-10-27 (c): GPU side checks

- **Great-power rule sensitivity** (`scripts/gp_share_sweep.py`, 4,000 paired worlds, uncertain AI). Great-power war by 2065
  at the "nuclear and >= X% of world output" cutoffs 3/4/5/6.5/8/10% and with no extra powers: 71/70/69/67/64/54/33%. Nuclear use (18%)
  and world GDP do not move; mean battle deaths go from 1.28M (majors only) to 1.90M (5%). The headline is a definition (does
  India count?) more than a dynamic: the report keeps both numbers.
- **Time-blocked early stopping for PsychoNet** (`scripts/net_timeblock_test.py`): choosing each task's stopping epoch on a
  later held-out block, then refitting, scores worse on dev (-0.026, 90% CI -0.039 to -0.015). Rejected; production unchanged.
- **Joint (multivariate) hindcast scoring** (`scripts/joint_hindcast_test.py`, dev origins 1975-2005, 15-year horizon,
  energy and variogram scores vs the same forecasts with each dimension shuffled independently): the simulated year-to-year
  persistence of world conflict is right (VS -63%, 7/7 origins), so are regional GDP co-movements (VS -7%, 7/7) and
  cross-country GDP dependence (VS -2%, 6/7). Cross-country dependence of conflict adds nothing (VS +0.0%, 2/7): no
  sign of over-clustering, but no demonstrated skill in which countries fight together either.

## Round 2026-10-27 (d): three hours of GPU

- **Joint hindcasts from every year 1970-2008** (`scripts/joint_hindcast_test.py 1970 2008 1 annual`, 39 origins, 15-year
  horizon, moving-block bootstrap with 15-year blocks). Relative to the same forecasts with each dimension shuffled:
  world conflict year by year VS -66% (39/39), region-year GDP paths VS -7.5% (38/39), GDP change by country VS -2.1%
  (32/39), conflict share by country ES +0.1% (90% CI -0.1% to +0.2%), VS -0.2% (21/39): neutral.
- **Over-clustering of conflict** (`scripts/conflict_clustering_check.py`, origins 1970-1998, 10-year horizon; crashed at
  2000 on a CUDA display-watchdog timeout). The across-world spread of the world conflict level is 2.9x what independent
  countries would give (0.047 vs 0.016) but the realised error is 0.027: PIT in the middle half 67% (sim) vs 20%
  (shuffled). The cross-country dependence is real but about 1.5x too strong; it comes from the "world mood" shocks.
  Sweep of mood_sd below.
- **Ensemble size** (`scripts/ensemble_size_test.py`): 20 or 40 network members, or 4 seeds, do not beat 10 members x 2
  seeds (all differences within +-0.004, 90% CIs include 0). Unchanged.
- **What drives the 2065 spread** (`scripts/uncertainty_sims.py`: 48,000 worlds, uncertain AI; `uncertainty_attribution.py`:
  cross-validated LightGBM surrogate, first-order and drop-one total shares; `uncertainty_ranges.py`: probability ranges).
  Assumptions explain 48% of the variance of log world GDP per person in 2065 (12% in 2045); the rest is simulated luck.
  Interactions are negligible. Main drivers: AI growth dividend (lowest vs highest fifth: $37k vs $54k), AI arrival year
  ($66k early vs $29k late), physical-automation dividend, robot arrival. For wars, assumptions explain 1-7% of the
  outcome variance, but they move the probabilities: nuclear_war_factor (how much nuclear pairs are spared war; 5 episodes)
  takes great-power war from 56% to 78%, China-India from 35% to 64% and India-Pakistan from 48% to 81%; nuclear use
  ranges 11%-26% with nuc_use_one / nuc_use_both. China-Taiwan (12%) and Russia-Poland (4%) hardly depend on any assumption.
  The single most valuable assumption to pin down is nuclear_war_factor.
- **Mood sweep and change** (`conflict_clustering_check.py MOOD 1970 1998 tag`): CRPS of the world conflict share vs
  mood_sd 0.20: 0.14 -5.2% (11/15 origins better), 0.10 -7.6% (11/15), 0 -9.9% (9/15); PIT in the middle half 67% -> 53%
  at 0.10. **mood_sd lowered to 0.10** (0 would also remove the regional coup waves).
- **nuclear_war_factor** (`scripts/nuclear_crisis_test.py`): ICB v16 crisis dyads since 1946: 4 of 39 both-nuclear crises
  reached full-scale war (all four fought by others in the crisis: Korea, Suez, two Israel-USSR), vs 40% of 443 neither-
  nuclear; ratio 0.27 (90% 0.11-0.50), decade-adjusted logit -1.81 (s.e. 0.51). UCDP conflict-years: 0 of 17 war-level
  (UCDP codes Kargil as minor). **Default 0.6 -> 0.4, range u(0.25, 1.0) -> u(0.10, 0.70).**
- **Data-vintage correction** for the monthly fatality model (`vintage_revisions.py`, `vintage_correction_fit.py`,
  `cm_fatal.VINTAGE_W`, `cm_fatal.revision`): candidate months scaled by exp(w * last final year's log revision), w = 0.25
  chosen on 2022-2023 revisions. Challenge live window CRPS 34.22/33.77 -> 32.00/31.89 (`seldon_v5`; 10th of 18, was 11th);
  w = 0 reproduces v4 exactly. Historical windows unchanged (final data).
- **Persistence mix rejected** (`scripts/persistence_mix.py`): K = 1000/month, pi = 0.25, s = 0.3 chosen on dev 2010-2017
  (16.11 -> 15.70); live 28.27/29.31 but 2018-2023 47.35 -> 49.51 (2023: 46.2 -> 54.1, Ethiopia's war ended). Mean over
  the eight out-of-sample windows 44.0 -> 44.3: not kept. Remaining live gap is Ukraine + Gaza (the model expects very
  large wars to fade).
- **Prospective scoring** (`scripts/prospective.py`): monthly fetch of the newest UCDP candidate file
  (data/raw/outcomes/ucdp/candidate_live; `monthly.candidate_files()` / `last_observed()` now pick it up), issue of a
  12-month ledger, crowd snapshot (outputs/ledger/crowd_snapshots.jsonl), scoring vs persistence and climatology and of
  resolved markets -> outputs/prospective_scorecard.json, outputs/ledger/score_history.jsonl. Cron entry to add by hand:
  `0 7 24 * * cd /home/benoit/tmp/psycho && /home/benoit/anaconda3/bin/python -P scripts/prospective.py >> outputs/ledger/cron.log 2>&1`.
  A new final GED release needs a manual switch (monthly.FINAL_THROUGH, cm_fatal._final); the script reports when one exists.

## Round 2026-10-27 (e): eval-driven tuning on the GPU (12-hour run)

Harness: `scripts/sim_eval.py TAG FIRST LAST STEP [kwargs]` (hindcasts, 10-year horizon, 3,000 worlds; dev origins
1970-1998 every 2 years, test 2000-2012; `--compare A B`). Rule: kept only if better on dev and confirmed on test.
- **mood_sd 0.10, held-out check** (`conflict_clustering_check.py 0.10 2000 2012`): world conflict share CRPS -8.9% (5/7)
  but new-onset count CRPS +11.7% (1/7). Cause: the simulator over-predicts onsets after the turbulent 1990s (35 vs 21-26
  per decade at 2000-2008 origins). Kept (mixed).
- **Not kept (noise on dev):** growth_regime_scale 1.4/2.2, mood split world/regional, mood_rho 0.8, regime_years 5,
  dem_regime_scale 0.7/1.3, dem shock scale 0.7/1.3 (all within +-1% on every metric).
- **year_shock_scale 0.7 (kept):** the replayed global year shock scaled to 70% (growth only): world GDP CRPS -2.2%
  (12/15 dev), -3.1% (5/7 test); country GDP -0.4% on test. 2065 world GDP per person median $37.1k -> $37.5k.
- **Alignment to long-run rates (not kept):** onset-only long-run alignment improves every conflict metric on dev
  (onsets CRPS -27%, world share -13%, Brier -0.5%), but on test onsets -19.5% with world share +6.9%. Diagnostics
  (`diag_*` in sim_eval): at the 2000-2012 origins the simulator starts 17% too many wars and ends 33% too many (share
  9% too low): too much churn; fixing onsets alone lowers the share. Joint onset+termination variants: see below.
- **Naive blend for the country ensemble (not kept):** nested leave-one-origin-out gain +0.0011 (civil war 5y), below noise.
- **Sustained-intensity features for the monthly model (not kept):** dev CRPS 16.111 -> 16.104.
- **Neural quantile blend for the monthly model (kept):** `cm_fatal.NNQuantile` (MLP 2x256, pinball loss, 5 seeds) averaged
  rank by rank with LightGBM, w = 0.5 chosen on dev 2010-2017 (16.11 -> 15.80). Test: challenge 2018-2023 mean CRPS
  47.35 -> 46.33 (5/6 windows better; 3rd, 0.04 behind OMM; best ignorance 0.570), live 32.0/31.9 -> 28.9/29.6
  (`seldon_v7`). `cm_fatal.NN_BLEND`.
- **Vintage correction on the ViEWS archive:** Brier 0.02426 -> 0.02419, log-fatality MSE 0.439 -> 0.431 (ViEWS 0.0249, 0.507).
- **Uncertainty attribution rerun** with nuclear_war_factor u(0.10, 0.70): still the top war driver (great-power war 42% vs
  70% between its lowest and highest fifth; China-India 20% vs 54%; India-Pakistan 29% vs 71%).
- **ViEWS archive with the neural blend** (`views_archive_benchmark.json`, v5 kept as `_v5`): Brier 0.02419 -> 0.02414,
  log-fatality MSE 0.431 -> 0.427 (better at every horizon; ViEWS 0.0249, 0.507).
- **Neural blend for the grid model (not kept):** dev 2014-2017 CRPS 0.3663 -> 0.3584 at w = 0.25 (all 4 years), challenge
  2018-2023 0.4451 -> 0.4438 (ignorance 0.0821 -> 0.0815), but live worse (0.2271/0.3179 -> 0.2314/0.3212). Mixed, so
  `pgm_fatal.NN_BLEND` stays 0.
- **More monthly neural variants (not kept):** hidden 512, 12 epochs, 10 seeds, lr 1e-3: all within 0.01 of 15.80 on dev.
  Lagged country-network features for the country ensemble: AUC -0.039 (worse).
- **Country frailty (kept, `frailty_sd` 0.6):** a persistent per-world, per-country log-odds shift (+ on onsets, - on
  endings, mean-corrected like mood), so war-prone countries stay war-prone. Aimed at the churn diagnostic. Dev
  (1970-1998, 0.3/0.45/0.6/0.9): 0.6 best balance, Brier -0.5% (13/15), onsets CRPS -5.3%, world share flat (0.9 overshoots
  the share, +1.7%). Test (2000-2012): Brier -0.5% (6/7), world share CRPS -11.7% (7/7), onsets CRPS -17.2% (7/7);
  too-many onsets 17% -> 8%, endings 33% -> 18%; GDP flat (world GDP +0.7%).
- **Country identity in the monthly neural model (not kept):** dev 15.80 -> 15.52 but challenge 2018-2023 46.33 -> 47.25
  and live 28.9/29.6 -> 33.1/32.8 (overfits). Neural model on the raw target instead of the residual: dev 15.89 (worse).
  Clustering check with frailty (`conflict_clustering_check_frail_{dev,test}.json`): dev PIT middle-half 0.53 (share),
  0.60 (onsets), CRPS still better than countries shuffled independently; test share CRPS 0.0126 -> 0.0108, onsets
  5.68 -> 4.71, but the test PITs remain biased (share 0.65, onsets 0.20: still too many onsets after the 1990s).
  Production rerun: great-power war 59% (unchanged), nuclear use 15.7% -> 16.1%, 2065 world GDP median unchanged,
  2065 civil-conflict share median 13.7% -> 15.4% (fewer wars end early).
- **Long-run alignment retested with frailty (not kept):** onset-only (w 0.5/1.0): dev world share CRPS -10%/-17%, onsets
  -14%/-19%, Brier -0.4%/-0.6%; test share 0%/+3.2% (2/7), onsets -11%/-21%. Onset+termination (w 0.5/1.0): dev Brier
  +0.7%/+2.4%, share -1%/+18%; test Brier -0.4%/-0.7%, onsets -11%/-20%. Dev prefers onsets only, test prefers both: the
  right long-run target depends on the era, so the 10-year alignment stays.
- **Coup alignment for the decline (kept, `coup_align_mult` 0.7):** new coup metrics in sim_eval (coup Brier/AUC over the
  horizon, world coup-year count CRPS, diag ratio). The simulator made 1.69x the actual coup-years at dev origins (1.24-2.19
  at every origin) and 1.68x at test: coups fell almost every decade (next-10-year rate 40-83% of the previous 10), so a
  target from the previous decade runs high. Dev: window 5 (count -20%), multiplier 0.5/0.6/0.7/0.8 (count -60/-74/-72/-57%,
  coup Brier +2.3/-0.5/-1.9/-2.2%); 0.7 balances both. Test: coup Brier -5.0% (6/7), count CRPS -63% (6/7), ratio 1.12;
  conflict and GDP neutral (dev GDP +0.2%, world GDP +0.8%).
- **Coup frailty (country coup-proneness) at the old alignment (not kept):** dev coup Brier -1.9% (15/15) but count CRPS
  +3.2% (0/15); test -2.1% / +5.8%. Retested on top of the new alignment below.
  Retest on top of coup_align_mult 0.7: dev coup Brier +0.2%, count +2.7% (noise or worse); not kept.
- **Country ensemble calibration in the large (not kept):** the ensemble over-predicts five-year coups at every backtest
  origin (1.08-2.33x). A log-odds shift fitted on earlier origins only (nested, about -0.5) improves Brier at 1995, 2000
  (-0.006 each), 2005 and 2010, but not at the untouched 2015 origin (+0.0014). The same test for the other six tasks: no
  consistent gain (one-year onsets better at every origin but below the noise floor, -0.0003 to -0.0015).
- **Monthly quantile recalibration (not kept):** PIT on dev 2010-2017 piles at low values (26% in the lowest decile, partly
  the zero mass), but remapping quantile levels toward it worsens leave-one-year-out dev CRPS at every shrink weight
  (15.80 -> 16.23 at 0.25, 21.31 at 1.0).
- **Monte Carlo noise of sim_eval** (same settings, 6,000 worlds vs 3,000): every metric within 0.0-0.7%, b-better counts
  7-10 of 15. Frailty (-0.5% Brier, 13/15) and the coup alignment (-72% count, 15/15) are well above it.
- **Where the monthly model loses to OMM:** per-country CRPS gaps. 2023: Ukraine alone is +7.0 of the +6.2 gap (our
  median sum 11k vs 75k actual, OMM 35k); live: Ukraine +3.2/+1.9 and Israel-Gaza +2.0/+1.6. Sustained mega-wars. An
  interstate-war feature would separate Ukraine from Ethiopia (whose war ended), but the 2010-2017 dev windows contain no
  large interstate war, so it cannot be selected on dev; not attempted.
- **Frailty at a 20-year horizon** (frailty 0 vs 0.6, dev 1970-1994 every 4 years, test 2000-2004): conflict Brier -0.6%
  (7/7) and -1.4% (3/3); test world share CRPS -28%, onsets -18%; dev world share +2.7% (share 12% high at 20 years).
  Coups still 1.27-1.39x at 20 years: the decline continues past the first decade. Extrapolating it 40 years would be an
  assumption (the 2020s Sahel coups run against it), so the coup hazard has no trend beyond the origin alignment.
- **Reruns with frailty and coup alignment:** joint hindcasts (39 origins) unchanged in substance (world conflict persistence
  39/39, regional GDP 38/39, country GDP 34/39, which countries fight together 23/39). Uncertainty attribution (48k worlds):
  nuclear-pair factor great-power war 43%/70% (lowest/highest fifth), China-India 20%/54%, India-Pakistan 29%/70%; nuclear use
  8%/24% by the chance of use once at war; GDP 2065 dividend $37k/$56k, arrival $68k/$29k.
- **Coup count fix (independent review, confirmed):** coups_crps/diag_coups_ratio summed simulated coups over all countries
  but actual ones over countries with complete coup data (137-165 of 175 at dev origins). Now both on the same countries
  (simulate records coup paths). Corrected: the simulator made 1.47x the actual coup-years at dev origins, 1.65x at test.
  Reselection on dev (vs mult 1.0): 0.6/0.7/0.8 count CRPS -39/-53/-49%, coup Brier -0.5/-1.9/-2.2%, ratio 0.83/0.98/1.14;
  0.7 kept. Test (0.7): count -63% (6/7), coup Brier -5.0% (6/7), ratio 1.65 -> 1.10. Frailty code: no bugs found.
- **Three-way monthly blend (not kept):** adding the direct-target neural model as a third member (grid of weights in
  0.1 steps, dev 2010-2017): best 15.787 at (NN 0.6, direct 0) vs 15.800 for the production blend; direct weight never helps.
- **Interstate calibration in the large (changed: `world.DYAD_CALIB` "drift" -> "none"):** `interstate_calib_test.py`
  fits the 1-year pair model before each origin (1975-2010 every 5 years, 2014), computes the recent and long-run
  observed/predicted ratios from pre-origin data, and scores each rule's 10-year onset count. Mean |log error|, dev
  1975-2005 / test 2010-2014: none 0.231/0.349, recent 0.353/0.490, long 0.416/0.218, drift (production) 0.358/0.417,
  fast drift 0.363/0.336; Poisson deviance none 7.0 vs drift 13.5; none better than drift at 8 of 9 origins (overlapping
  windows, 4-12 onsets each: suggestive). The previous decade's ratio does not predict the next decade. Headline effect
  (8,000 worlds, same seeds; drift / long / none): great-power war 58/52/50%, traditional majors 31/26/25%, nuclear use
  16/15/14%, China-India 39/35/32%, India-Pakistan 53/49/47%.
- **Mood size re-swept with frailty (unchanged at 0.10):** 0.05 and 0.15 within noise on every dev metric (|change| <= 1.7%).
- **Interstate war share / escalation by decade:** 0-83% per decade on 5-15 cases each, no usable trend; fixed averages kept.
  independent review of the test and switch: conclusion sound (no test bias toward "none"; in-sample calib_recent shrinks
  toward 1, which flatters "drift"; the old shift was also added to pairs of democracies, which the ratios excluded).
  Fixed: stale comment, unknown DYAD_CALIB values now raise.
  Whole-simulator interstate hindcast (rerun_forecast --interstate, 2,000 worlds), drift -> none: 1995 origin active
  conflicts/yr 1.34 -> 1.21 (real 1.15), pairs fought 14.6 -> 13.2 (9), AUC 0.94 -> 0.93; 2004 origin 1.09 -> 0.96 (real 1.40),
  pairs 12.2 -> 10.8 (11), AUC 0.95 -> 0.91; battle-death medians 117k/83k -> 96k/63k (real 110k/246k), both inside the
  90% band either way. Mixed on two origins; the nine-origin pair test stands.
- **Civil alignment window re-swept with frailty (unchanged at 10 years):** 5 years: onsets CRPS +34%, share +6.6%; 15 years:
  onsets -9.9% (9/15), share -1.7%, but Brier +0.4% (5/15). Not kept.
- **Recency weighting of the monthly model (not kept, `cm_fatal.HALFLIFE`):** dev 2010-2017 with the production blend:
  none 15.800 (reproduces), half-life 3 / 5 / 10 years 15.982 / 15.853 / 15.849.
- **LightGBM settings for the monthly model (not kept):** never tuned before; with the production blend on dev 2010-2017
  (15.800): 15 leaves/min 200 15.831; 63 leaves/min 50 15.766; 800 trees at lr 0.02 15.771; colsample 0.5 15.802. Gains
  of 0.2% with no consistent year pattern: noise.
- **Round (e) simulator changes together, every origin 1970-2012 (43; `all_old` vs `all_new`):** old = mood 0.20,
  year_shock_scale 1.0, no frailty, coup_align_mult 1.0. World conflict share CRPS -8.1% (32/43), onsets CRPS -6.3% (34/43),
  conflict Brier -0.2% (30/43), coup Brier -2.4% (32/43), world coup count CRPS -56% (37/43), world GDP CRPS -1.9% (27/43),
  democracy CRPS -1.0% (31/43); country GDP CRPS +0.2% (17/43), GDP 90% coverage 0.895 -> 0.889. Simulated / actual:
  onsets 1.04 -> 0.98, endings 1.08 -> 1.00, coups 1.53 -> 1.01.
- **Whole-simulator interstate hindcasts at six more origins** (`interstate_hindcast_multi.py`, 10 years ahead, 2,000 worlds;
  active conflicts/yr sim vs real, drift / none): 1980 2.47 / 2.16 (2.7), 1985 2.09 / 1.97 (2.0), 1990 1.98 / 1.67 (1.4),
  2000 1.04 / 1.19 (0.9), 2010 0.72 / 0.71 (1.5), 2014 0.75 / 0.88 (2.3). Summed |log error| over all eight origins (with 1995
  and 2004): drift 2.88, none 2.83; each better at four. Pairs fought closer with none at 5 of 6; 2014 years in the 90% band
  0.7 -> 0.9; pair AUC equal. Tie on the whole simulator; the pair-level test decides. Both rules under-predict the
  2010s rise of interstate conflict two- to threefold (from 2010 and 2014 origins): a known limitation.

## Round 2026-10-28 (f): next steps (interstate gap, scorecard, crowd, assumptions, audit)

- **Interstate gap, diagnosed.** Run from 2010 and 2014, the simulator predicts about the right number of pairs that
  fight (4.5 vs 5; 5.2 vs 9) but half the active conflict-years (0.7 vs 1.5; 0.9 vs 2.3): conflicts since 2010 last
  longer. Continuation (UCDP dyad still active next year) by run length 1 / 2 / 3+: 0.34 / 0.37 / 0.59 in 1946-1989
  (94 cases), 0.25 / 0.75 / 0.67 in 1990-2009 (25), 0.64 / 0.75 / 0.92 in 2010-2024 (27). Recurring low-level pairs
  (India-Pakistan 20 of 35 years since 1991, Iran-Israel every year 2018-2025) and outside powers' strike campaigns
  (USA/GBR-Yemen) make up most of the 2015-2024 total.
- **Recency-weighted continuation table: kept** (interstate.CONT_HALFLIFE = 25). scripts/interstate_cont_test.py: the
  table is estimated before each origin with weights 0.5^((origin - year)/H), scored on every continuation outcome in
  the next 10 years. Dev origins 1970-1998 (309 outcomes): log-loss 0.7297 equal weights, 0.7263 H=25 (best; 15 and 40
  within 0.003). Test origins 2000-2014 (84): 0.6097 -> 0.5876 (-3.6%), Brier 0.209 -> 0.198. Whole simulator
  (scripts/interstate_hindcast_multi.py none 25 vs none none, 8 origins 1980-2014): mean |log error| of active
  conflicts per year 0.394 vs 0.389, pairs fought identical: a tie, because the rise in continuation came mostly after
  2018 and is invisible from 2010 or 2014. Kept on the pair-level test (the same rule as DYAD_CALIB). At the 2025
  origin it moves the table from 0.37 / 0.48 / 0.71 to 0.43 / 0.62 / 0.78. The onset side of the gap (new pairs:
  Iran-Israel, Yemen) is not fixed; no rule tested so far predicted it from pre-2014 data.
- **Crowd and model on near-term questions.** scripts/crowd_pool_test.py on the 63 resolved use-of-force questions
  (22 pairs, 8 events; bootstrap over pairs): logarithmic pools with model weight 0 / 0.05 / 0.1 / 0.2 / 0.3 / 0.5 give
  Brier 0.0998 / 0.0978 / 0.0965 / 0.0963 / 0.0987 / 0.1092; the 90% interval of the 0.1 pool minus the crowd is
  -0.011..+0.005. The fitted logistic blend logged until now was worse than the crowd in leave-one-out (0.111). The
  report keeps the crowd's number where one exists; the ledger now logs the fixed 0.1 pool (crowd.POOL_W) and the
  scorer recomputes it for every logged question, to test it on new questions.
- **Scorecard.** scripts/annual_ledger.py freezes each published simulator forecast (code and forecast hashed;
  outputs/ledger/annual_<origin>_<hash>.json, model_versions.jsonl): per-country P(in conflict) for 2026-2030, coups
  and new conflicts by 2030, 5-year risk models, the top 40 pairs and the world fans. Each completed calendar year is
  scored against a UCDP GED proxy (25+ state-based deaths located in the country) with a same-proxy persistence
  baseline. Run monthly by scripts/prospective.py (cron, 24th); the report's "On the record" section shows the
  scorecard (monthly ledger, simulator ledger, markets, LLM panel).
- **Assumption ranges, re-sourced** (research in outputs/reviews/assumption_evidence.md; FRI LEAP wave 8 checked at the
  source). ARRIVAL_CDF later: 5 / 18 / 24 / 33 / 47 / 56 / 64 / 72% by 2027 / 30 / 32 / 35 / 41 / 47 / 55 / 65 (was
  17.5% by mid-2027, 37.5% by 2032, 77.5% by 2065): equal mix of Metaculus (median about 2031-33, read from secondary
  sources: the site refused automated access), the AI Impacts 2023 survey conditioned on no arrival by late 2026, and
  FRI LEAP wave 8 (April-May 2026; non-physical work at <= 5x human cost; 80% by 2100, median 2050 if it comes).
  ROBOT_CDF slightly later (3 / 14 / 25 / 35 / 48 / 63 / 76% by 2027 / 30 / 32 / 35 / 40 / 50 / 65). phys_dividend
  u(0.005, 0.04) (floor: industrial robots, Graetz & Michaels 2018); nuclear_war_factor u(0.10, 0.55) (the ICB estimate
  0.27, 90% 0.11-0.50); nuc_use_one u(0.001, 0.008). growth_dividend unchanged (mainstream sub-AGI estimates are a floor;
  LEAP wave 10 experts: about +3 points a year in advanced economies by 2050 under rapid progress). drone_onset and
  drone_civil unchanged: no quantitative evidence either way, narrowing would be cosmetic.
- **Independent assumption audit** (outputs/reviews/review_assumption_audit.md; eight claims, each checked in the code):
  1. Coup shifters (peer contagion, succession, export prices) and the peacekeeping term were added after the year-1
     alignment, and last year's coups started at zero: **fixed** (moved before the alignment, as intl_beta already was;
     coup_last starts from the origin year's attempts). The size claimed (+20-40% coups) did not show in the output
     (median 1 coup a year 2026-2035 vs 2.1 in 2015-2024), but the order was inconsistent.
  2. Structural pair noise (z_pair x pair_spread, up to 1.5 log-odds) had no convexity correction and is active only
     with sampled parameters, which the hindcasts never use: **fixed** (mean-preserving correction, as for the civil
     mood).
  3. Workforce term: WORKFORCE_BETA 1.22 is the 5-year regression, applied yearly inside the growth momentum; the
     matching 1-year coefficient is 0.57 (s.e. 0.14). **Tested, rejected**: sim_eval with 0.57 is worse on world GDP
     (dev 1970-1998 +2.4%, test 2000-2012 +4.8%; country GDP +0.2% / -1.6%). The 10-year record favours the larger
     effect; kept.
  4. DYAD_CALIB test "flawed": its in-sample recent ratio was by design (it is what "drift" used); the missing pair noise
     is item 2. Coalition joiners counted as dyad onsets in training and again by the alliance cascade: plausible
     (hindcast pairs fought: simulated 64 vs 52 real over 8 origins), but cascade_q was calibrated on the hindcasts
     with this dyad model, which absorbs much of it; noted, not changed.
  5. Nuclear use: ranges narrowed (above). 6. Democracy uses the gap to the frontier and one-sided AI drift: the first
     is validated by dem_crps in sim_eval, the second is a stated assumption. 7. Frailty survivor selection: frailty was
     kept on 20-year sim_eval horizons, where that selection would show. 8. Great-power definition (India counted): a
     definition, reported both ways (COW majors only alongside). Energy cap / Baumol (GDP upper tail in AI worlds) and
     the 1848-1953 major-war death pool: assumptions, noted, not changed.
- **Protest from machine-coded events (ACLED substitute): rejected.** data/raw/events/protest_build.py: within-year
  z-scores of the protest share and the material-conflict share of each country's domestic events, ICEWS 1995-2022
  spliced to GDELT 2023-2025 (POLECAT ends June 2024 and stopped updating). Sources agree only moderately on the
  ranking of countries (Spearman 0.42-0.61 for protest, 0.50-0.64 for unrest; ICEWS~POLECAT 2018-2022,
  POLECAT~GDELT 2023-2024). scripts/protest_events_test.py, controls including the Mass Mobilization counts: 2 of 16
  logit tests at p < 0.05 (unrest -> coups, p 0.03; protest surge -> 5-year civil war, p 0.045), about what chance
  gives; tuned GBM score dev 0.9035 -> 0.9040, lockbox 0.9873 -> 0.9892 (noise floor ~0.005), 1-year onset worse on
  the lockbox (BSS 0.095 -> 0.091). Not added; the stale-protest limitation (Mass Mobilization to 2020) stands.
- **Humid heat (ERA5 wet-bulb)**: data/raw/heat via scripts/heat_download.py (CDS allows one queued request per
  dataset: five years per request, sequential), scripts/heat_build.py, scripts/heat_test.py. In progress.
- **Observed climate to 2025** (was 2022 for precipitation, dry spells and the hottest day). The World Bank portal now
  serves ERA5 1950-2025; tas, pr and cdd match the earlier 1950-2022 release to 0.01, but its hottest day (txx) is
  5-7 C higher throughout (a new definition; yearly changes correlated 0.90 with the old series, anomalies ~20%
  larger), so the whole txx series is taken from the new release, never spliced. Re-estimated (scripts/extra_estimates.py
  D2): cereal yields -1.03% per C of hottest-day anomaly (se 0.25; was -1.4% on the old series), rain +5.7% per log
  unit unchanged; hottest-day anomaly correlation across countries 0.10 (was 0.16). Tuned-GBM dev score with the new
  climate data 0.9031 (0.9035 before: noise). Only the four climate arrays of the panel changed.
- **Coalition double counting (assumption audit item 4), measured**: 5 of 53 UCDP interstate conflicts had several states
  on one side (1948, 1956, 2001, 2003, 2024), 22 of 170 pair-years (13%); the alliance-joining rate was calibrated on
  hindcasts that include them. Not changed.
- **CrisisWatch** (Crisis Group monthly risk flags): its robots.txt disallows crawling the database pages and the site
  answers 403; not used. **Carnegie Global Protest Tracker**: accessible, but it starts in 2017, after every test
  origin, so it cannot be validated on history; not used in the model.
- **Data Commons (US states): rejected for now, retest later.** scripts/dc_states_fetch.py (FBI UCR violent,
  property and murder rates 2008-2023, 2021 missing; ACS bachelor's share, poverty, median age, foreign-born and
  uninsured shares ~2010-2024; data/raw/us/datacommons_states.csv). Country-level Data Commons series are mostly the
  World Bank / UN data already used; floods (satellite, 2015+) and NOAA drought area (to 2010) were too short or stale,
  remittances already rejected. scripts/dc_states_test.py, the state module's logit+GBM: time split, origins
  2016-2020 (the only origins that can learn from series starting 2008-2012): worse (mass shooting AUC 0.758 -> 0.738,
  BSS 0.150 -> 0.081; overdose 0.529 -> 0.409; unemployment 0.672 -> 0.680, BSS 0.410 -> 0.385). State-grouped CV over
  2010-2020 (cross-sectional signal only): overdose AUC 0.660 -> 0.711, income shock 0.836 -> 0.864, others equal.
  The signal is cross-state, not yet forecastable with a few years of history; retest when the series reach ~2030.
- **Data search, 28 October 2026 (scripts/new_data_test.py NAME: logit with controls, then tuned-GBM backtest; base
  score dev 0.9031 / lockbox 0.9875; outputs/new_data/NAME.json).**

  | Family (source) | dev | lockbox | verdict |
  |---|---|---|---|
  | structural: terrain, ethnic/language fractionalization, non-contiguity, state age (Fearon & Laitin 2003, Dataverse doi:10.7910/DVN/QB3BFP; COW) | 0.9042 | 0.9904 | noise; no civil-war gain (coups, autocratization slightly up) |
  | imf: months under IMF programs, new programs, 10-year share (Stubbs, IMF Programs Dataset 1952-2025, doi:10.7910/DVN/V4V1BA) | 0.9046 | 0.9890 | noise; logit: non-concessional programs and coups p 0.002 |
  | polity: factionalism, PITF partial democracy with factionalism, anocracy, durability (Polity5 to 2020) | 0.9053 | 0.9833 | rejected (lockbox worse) |
  | macro: reserves cover, current account, credit boom, depreciation, short-term debt / reserves, terms of trade (WDI) | 0.9005 | 0.9905 | rejected (collapse AUC -0.008 on dev despite logit p <= 0.001) |
  | warstructure: rebel groups, intensity, territorial aims, internationalization, external support, peacekeepers (UCDP, UN; on disk) | 0.9038 | 0.9885 | **lead**: war endings AUC +0.019 dev, +0.014 lockbox, BSS +0.006 / +0.009 |
  | heat: wet-bulb level, shock, hot share, humidity beyond dry heat (ERA5 1960-2020) | 0.9036 | 0.9890 | noise; 1-year onset worse |
  | milex: military budget changes and cuts (SIPRI) | 0.9056 | 0.9872 | noise; no coup gain |
  | combo: structural + imf + warstructure | 0.9044 | 0.9934 | dev within noise; lockbox +0.006; war endings +0.016 / +0.013 |

  The country risk models look saturated on structure: ~150 features already cover regime, repression, capacity,
  leaders and economy, and each family moves the score by 0.001-0.003. The one consistent gain is war endings from
  the structure of the war itself. Not adopted yet: the simulator applies frag_beta, intl_beta and pko_beta to war
  endings from the same data, and the features would have to evolve inside the simulation; integrating them means
  moving those terms into the learned hazard and simulating the war-structure state, with its own test.
  The 1-year coup lockbox AUC fell by about 0.01 with almost every family: few lockbox coups, treat it as noise.
  Not usable: Fearon's Stanford server (redirect requiring a browser identity; the Dataverse copy was used), CrisisWatch.
  Heat download stopped at 2020 (61 years; the remaining request was moot once heat was rejected).
- **War structure in the production ensemble** (scripts/warstructure_ensemble_test.py, SeldonEnsemble all families,
  4 members): war endings (termination_1y) AUC 0.747 -> 0.764 on development origins and 0.756 -> 0.767 on the
  lockbox, Brier skill 0.119 -> 0.129 and 0.131 -> 0.143; the other six tasks lose 0.001-0.008 (AUC or BSS) when they
  see the columns. Next step: give the war-structure columns to the war-ending task only (per-task feature masks in
  SeldonEnsemble), for the report's 1-year war-ending forecasts; the simulator keeps its frag/intl/pko terms (fitted
  on the same data), so nothing is counted twice as long as the simulator's hazard network does not see the columns.
- **War structure in the simulator: rejected.** Wired as a data family (UCDP rebel groups, intensity, territorial
  aims, internationalization, external support, UN peacekeepers), with the features following the simulated wars
  (kept while a war continues, one group at minor intensity for a new war, 0 at peace; rebel groups,
  internationalization and peacekeepers from the world layer's state) and the world layer's frag / intl / pko
  terms switched off when present. sim_eval (paired, PSYCHO_DISABLE=warstructure as baseline): development 1970-1998
  conflict Brier -1.4% (13/15), world share CRPS -0.5%, simulated/actual war endings 0.92 -> 1.00; test 2000-2012
  conflict Brier +0.7% (better at 1/7), world share CRPS +17.5% (1/7), endings 1.18 -> 1.29 (too many wars end).
  Reverted (data.py, features.py, simulate.py restored). The war-ending gain in the risk models (+0.012-0.017 AUC)
  does not carry into the simulator, and no published forecast shows the 1-year war-ending probability, so nothing
  is adopted.
- **Interstate onset signals the simulator could update: rejected** (scripts/interstate_onset_test.py, the logit part of
  the 1-year dyad model, which is all the simulator evaluates; pairs at peace, onsets in origin..origin+9). Candidates:
  one side in civil conflict and the other a major power (intervention), other conflicts either side is fighting,
  either side's interstate conflict-years in the past decade, and active conflicts worldwide. Log-loss dev (1970-1998,
  76 onsets) 0.00314 base, 0.00314-0.00317 with each; test (2000-2014, 44) 0.00201 base, 0.00200-0.00211. AUC moves
  by <= 0.007 except worldwide activity, which lowers it (0.93 -> 0.91, 0.91 -> 0.89). The onset logit over-predicts
  in both splits (111 vs 76, 68 vs 44 onsets), so the 2010s shortfall is not an onset hazard that is too low.
- **Pair-specific continuation: rejected** (scripts/interstate_cont_model_test.py). A recency-weighted logistic model
  with the run-length cells plus pair features (war level; geography and rivalry; power and nuclear status; regime and
  civil war; all), C = 0.03 or 0.3, against the CONT_HALFLIFE = 25 table. Dev log-loss: table 0.7263, best model
  0.7254 (geography, a tie), the rest 0.738-0.836. Two variants look better on test (power or regime with C = 0.3:
  0.5795 and 0.5754 vs 0.5876), but both are worse on dev (0.806, 0.752), and with 84 test outcomes that is noise.
  The run-length table stays. With the simulator's onset and duration rules both tested, the remaining 2014-2024 gap
  (Iran-Israel, strikes on Yemen) is put down to a change of regime that pre-2014 data do not reveal.
- **Great-power rivalry (proxy wars): rejected** (scripts/new_data_test.py rivalry). UN General Assembly ideal points
  (1946-2025): US-Soviet/Russian distance (4.8-5.0 in 1950-1975, 1.8 in 1992, 2.4 in 2010, 2.9 in 2025), its 5-year
  change, and how contested each country is between the US and the further of Russia or China (a one-dimensional scale:
  Ukraine 0.98 in 2010, but Germany 0.84). Logit with controls, errors clustered by country: no link to civil-conflict
  onset or the 5-year conflict risk (|coef| <= 0.09 sd, p > 0.25); coups rise with rivalry (+0.37 sd, s.e. 0.09) and
  with rivalry x contested (+0.21, s.e. 0.06), the Cold War coup era. Tuned GBM: dev 0.9031 -> 0.9030, lockbox
  0.9875 -> 0.9895 (noise; coup_1y BSS lockbox 0.093 -> 0.091). Mechanism check (scratch): in civil wars of 1975-2017,
  rebels have state backing more often when rivalry is high (+0.34 sd, s.e. 0.15), but war continuation is unrelated
  (+0.09, s.e. 0.10; positive only after adding a time trend, which with three rivalry eras is not identifiable).
  Continuation by era: 0.84 (1946-89), 0.82 (1990-2009), 0.88 (2010-23), while state backing of rebels fell
  0.70 -> 0.50 -> 0.38: the 2010s rise in war persistence is not a return of Cold War-style proxy backing.
- **Personalist rule and the regime's support coalition (V-Dem v15): borderline, not adopted yet**
  (scripts/new_data_test.py personal; data/raw/newdata/personal/vdem_personal_cy.csv). Leader-cult legitimation
  (v2exl_legitlead) and its 5-year change, presidentialism, regime corruption, support from the military, foreign
  governments or an ethnic group (v2regsupgroups 5, 13, 6), head of state who took power by force (v2expathhs = 0).
  Logit with controls: coups rise with military support (+0.35 sd, s.e. 0.10) and power taken by force (+0.16,
  s.e. 0.04) and fall with leader-cult legitimation (-0.40, s.e. 0.11) and ethnic support (-0.42, s.e. 0.10); civil
  conflict rises with regime corruption and military and ethnic support. Tuned GBM: dev 0.9031 -> 0.9071 (+0.0041,
  country-bootstrap 90% interval +0.0004 to +0.0077, scripts/family_bootstrap.py), lockbox 0.9875 -> 0.9937 (+0.0063,
  +0.0010 to +0.0123). Placebos (the same columns with countries shuffled): dev +0.0008 / +0.0011, lockbox -0.0020 /
  +0.0017. Ablations: support coalition and force alone +0.0019 / +0.0021; leader cult, presidentialism and
  corruption alone +0.0006 / -0.0004. Production ensemble (4 members, scripts/warstructure_ensemble_test.py personal):
  headline mean of AUC + Brier skill +0.0020 dev, +0.0021 lockbox (all seven tasks +0.0013 / +0.0018): positive on both
  but inside the noise band. Geddes-Wright-Frantz personalism was rejected on 2026-10-24; what carries the signal here
  is the support coalition. Adoption would also need the simulator check (held as a slow-moving institution).
- **Screen of the risk models' errors against every unused V-Dem index: null** (scripts/residual_screen.py). 75
  indices x 7 tasks = 525 tests of logit(y ~ logit(p) + z) on out-of-sample tuned-GBM predictions: 55 pass |t| > 3.3
  on dev, none is confirmed on the lockbox (same sign, p < 0.05). The dev hits are nearly all economic collapse
  under-predicted in unfree countries (property rights, accountability, electoral integrity: t -7 to -8), the
  post-communist collapse of 1989-1995; the sign reverses in the lockbox.
- **Five angles on how populations behave (2026-10-28): none adopted.** Risk-model families in scripts/new_data_test.py
  (tuned GBM, dev 0.9031 / lockbox 0.9875):
  - *Generational memory* (scripts/build_memory.py: share of today's population old enough, >= 8, to remember the
    country's last interstate war (COW / UCDP), civil conflict, hyperinflation >= 100% (GMD) or collapse >= 15%; UN WPP
    five-year age groups via OWID): dev 0.9038, lockbox 0.9859. Only link: remembered collapse -> economic collapse
    (+0.34 sd, s.e. 0.07). Germany's WWII memory 0.40 in 1980, 0.02 in 2025.
  - *Expectations against reality* (growth against own previous decade and neighbours, schooling ahead of income,
    educated and youth unemployment): dev 0.9019, lockbox 0.9875. Educated unemployment -> civil conflict +0.38 (0.14);
    schooling ahead of income is protective (civil conflict -0.39, s.e. 0.10), against relative-deprivation theory.
  - *Brittleness / hidden preferences* (democracy deficit against income, schooling and cities; x repression;
    crowd capacity = urban x youth x schooling, x repression): dev 0.9057, lockbox 0.9853.
  - *Elite overproduction* (graduates beyond what income predicts, its 10-year change, graduates against growth,
    top-1% share change): dev 0.9037, lockbox 0.9890. Graduates' excess rising -> civil conflict +0.27 (0.08),
    graduates outrunning growth -> economic collapse +0.56 (0.10): Turchin's links exist, but the model already has them.
  - *Narratives, monthly* (scripts/monthly_rhetoric_test.py: verbal-conflict share and surge, ICEWS / POLECAT, in the
    monthly battle-death model; rolling years, dev 2003-2011, test 2013-2023): Brier dev 0.02295 -> 0.02308, test
    0.02347 -> 0.02329; months without recent fighting: dev 0.00364 -> 0.00366, test 0.00570 -> 0.00577. Noise, as is
    material conflict (0.02301 / 0.02321).
  - *Mood* (scripts/mood_test.py: Gallup Cantril ladder via WHR / OWID, 2011-2025; association with the tuned GBM's
    out-of-sample errors at origins 2010, 2015, 2020 only, 21 tests): falling satisfaction over 3 years precedes
    economic collapse (-0.40 sd, s.e. 0.13, p 0.002) and rising satisfaction precedes autocratisation (+0.37, p 0.004).
    Too short a record for the development test: a lead to watch, not a feature.
  - *Cascades* (scripts/cascade_dispersion.py): yearly world totals against summed out-of-sample probabilities.
    Dispersion 1.57 for civil-conflict onsets (largest excess 1988-1990 and 2015), 1.22 coups, 1.15 war endings:
    modest year clustering, which the simulator's year shocks and mood (round e) are tuned to.
- **Personal rule and support coalition: ADOPTED** (data.load_personal, features group "Personal rule", 147 features).
  Simulator hindcasts (scripts/sim_eval.py pr0 = family off via PSYCHO_DISABLE=personal, pr1 = on; levels held as
  institutions, the 5-year personalisation change fades at 0.8 a year): test 2000-2012 conflict Brier 0.06758 ->
  0.06726 (-0.5%, 6/7), new-onset count CRPS -10.7% (6/7), coup Brier -1.0% (7/7), coup count CRPS -6.3%, onset AUC
  +0.3%, GDP and democracy unchanged; dev 1970-1998 world share CRPS -4.7% (11/15), onsets CRPS -3.5%, coup count CRPS
  -7.7% (11/15), but country conflict Brier +0.6% and coup Brier +0.8%. With the risk-model gains (bootstrap intervals
  above 0 on dev and lockbox) adopted; forecast rerun (outputs/forecast_before_personal.json kept).
- **Year clustering in the simulator: no change.** New diagnostics in sim_eval (where each year's actual count of new
  conflicts falls among the simulated worlds): dev 1970-1998 13% of years in the outer 5% tails (10% expected), variance
  ratio 1.02 (well calibrated); test 2000-2012 1.4% and 0.44 (the calm 2000s are over-dispersed: an era effect).
  World mood 0.05 or 0 instead of 0.10 (regional mood kept): every score within 1% on dev. Kept at 0.10.
- **Growth equation screen** (scripts/growth_screen.py; ridge on the growth design with one more variable, rolling
  origins, next-year growth net of world mean and 10-year mean growth, dev 1975-1995 / test 2000-2014): fiscal capacity
  improves all four (-0.5% / -0.5% / -1.1% / -1.5% MSE), as do executive respect for the constitution, territorial
  control and fewer political killings (smaller); log GDP improves 10-year growth by 6% / 3.7% (with the frontier gap
  it lets growth slow as countries get richer: -0.64 points per log point). Investment (WDI gross and fixed capital
  formation, FDI inflows, 1960-2024; `growth_screen.py invest`) adds nothing (10-year dev -0.3% to +1%, test up to
  +12% worse): rejected. Penn World Table 10.01 (data/raw/newdata/pwt) ends in 2019, too early for a 2025 origin.
- **Growth equation: fiscal capacity ADDED** (simulate.GROWTH_FEATURES). Hindcasts (sim_eval gr_fc against the
  personal-rule baseline; fiscal capacity follows its own slow-variable equation in simulation): country GDP CRPS
  -0.5% dev (12/15), -1.7% test (5/7); world GDP CRPS -1.8% / -3.1%; 90% coverage 0.860 -> 0.857 dev, 0.951 -> 0.953
  test. Fitted effect: +0.38 points of yearly growth per unit of the V-Dem fiscal-capacity index. Log GDP next to the
  frontier gap (growth slowing as countries get richer in absolute terms) was rejected: world GDP CRPS +45% dev, +40%
  test, as the slowdown compounds. Territorial control on top of fiscal capacity: country GDP -0.2% both, world GDP
  +4.1% dev: rejected.
- **Democracy equation: free-expression erosion and neopatrimonial rule ADDED** (simulate.DEM_FEATURES). Screen
  (scripts/dem_screen.py, as the growth screen, democracy index): 25 variables improve next-year and 10-year democracy
  change on dev and test, the equation having had 8; the 5-year change in free expression (erosion of speech comes
  before the electoral index falls) and neopatrimonialism lead, together -2.6% / -7.5% (next year) and -1.9% / -5.9%
  (10 years) MSE; larger sets add little on dev. Hindcasts (sim_eval dm_fn against gr_fc): democracy CRPS -3.0% dev
  (14/15), -6.5% test (7/7); world conflict share +0.4% / -1.3%, onset count CRPS +0.4% / +1.4%, coup count +0.1% /
  +1.4%, GDP within 0.3%.
- **Slow-variable equations: no change** (scripts/slow_screen.py: ridge of each SPEC variable's yearly change on its
  level, the current drivers and one more model feature, net of year means, dev 1975-1995 / test 2000-2014). Many
  candidates improve both splits, but by little: the largest are regime age for democratic mobilisation (-1.8% /
  -1.5% MSE) and mass mobilisation (-1.6% / -1.0%), life expectancy for extreme poverty (-1.5% / -1.5%), territorial
  control and political violence for non-state and one-sided violence (-0.7% to -1.6%). Not worth new plumbing in
  SlowState for effects that reach the headlines only through the risk models. The fiscal-capacity equation (now a
  growth driver) has no feedback loop to fear: growth moves it by +0.04 per unit of log growth, and its drift term
  is not applied in simulation.
- **Growth: executive respect for the constitution on top of fiscal capacity: rejected** (sim_eval gr_erc vs dm_fn: country GDP CRPS -0.1% dev / -0.9% test, world GDP CRPS +2.4% dev).
- **Democracy equation with CSO repression change and CSO participation as well: rejected** (sim_eval dm4 vs dm_fn: democracy CRPS -1.0% dev, +2.3% test).
- **Rest of the regime coalition (V-Dem v2regsupgroups / v2regoppgroups, 22 columns): not confirmed**
  (new_data_test.py coalition; baseline now includes personal rule, dev 0.9073 / lockbox 0.9937). Tuned GBM dev
  0.9165 (+0.0092), lockbox 0.9955 (+0.0018); gains mostly in coups (coup_5y AUC +0.017 dev, +0.015 lockbox). Placebos
  (countries shuffled): dev 0.9000 / 0.9000, lockbox 0.9863 / 0.9847: 22 noise columns cost 0.007-0.009. Country
  bootstrap: dev +0.0092 [-0.0003, +0.0188], P 0.94; lockbox +0.0018 [-0.0125, +0.0157], P 0.56. Strongest links:
  military in opposition -> coups (+0.80 sd, s.e. 0.10), autocratisation (+0.49) and new civil war (+0.31); regimes
  backed by religious groups -> civil conflict (+0.27, s.e. 0.06); civil servants among the backers -> fewer coups (-0.38).
  A subset selected on development years only is tested next (coalition_core).
- **Long crisis memory since 1800: rejected** (scripts/build_crisis_memory.py; Reinhart-Rogoff 1800-2016 for 70
  countries, then BoC-BoE defaults and GMD inflation and banking crises: years in default, inflation-crisis and
  banking-crisis years since 1800, years since the last default and inflation crisis): dev 0.9073 -> 0.9048, lockbox
  0.9937 -> 0.9929. Serial defaulters see more coups (+0.36 sd) and autocratisation (+0.38), but recent history carries it.
  (The panel already has GMD banking, debt and currency crises from 1850.)
- **Immiseration on 1850-2005 data (Clio-Infra; series end 2000-2013, so no forecast use)**
  (scripts/clio_history_test.py; logit with income, growth, population, democracy and conflict history, errors
  clustered by country): falling real wages (10-year change, drawdown from the 20-year peak) do not precede civil war,
  coups or autocratisation in 1850-1945 or 1946-2005 (|coef| <= 0.34, none significant); shrinking heights weakly
  precede civil-war onsets (-0.34 sd, s.e. 0.15; -0.50 before 1946 with 20 onsets); rising strike participation
  precedes onsets after 1946 (+0.27, s.e. 0.09) and strikes per head precede economic collapse (+0.35, s.e. 0.10).
- **Religion: what the model has and what is missing (2026-10-28).** In: religious fractionalisation and polarisation,
  Muslim and non-religious shares (World Religion Project to 2010, carried to 2025), religious freedom and repression
  (V-Dem, co-moving with democracy in simulation), religious legitimation of the regime (V-Dem), and now religious
  groups among the regime's backers (load_coalition: -> civil conflict +0.27 sd). Not in: the religious dimension of
  each conflict (RELAC, Svensson & Nilsson 2018, 1975-2015; Sage's Figshare refuses automated download, 403: a manual
  browser download is needed) and religious change after 2010 (Pew 2010-2050 projections downloaded to
  data/raw/newdata/religion/pew_2010_2050.xlsx; a refresh of the origin values and a path for simulation, which the
  backtests cannot score because the history before 2010 is already covered).
- **Regime coalition core (6 columns, chosen on development years): risk models confirmed, simulator mixed, then
  restricted to coups and autocratisation.** Tuned GBM dev 0.9073 -> 0.9199 (+0.0126, bootstrap +0.004 to +0.021),
  lockbox 0.9937 -> 1.0006 (+0.0069, -0.007 to +0.021, P 0.75); placebo -0.003 / -0.008; production ensemble headline
  +0.009 dev, +0.014 lockbox (coup_5y AUC +0.012 / +0.019, coup_1y +0.010 / +0.021). No hindsight coding: before coups
  that follow five calm years, military opposition is 0.43 six years ahead and 0.47 in the coup year (0.25 overall).
  Simulator with the family feeding every task (sim_eval co1 vs dm_fn): coups much better (Brier -3.4% / -3.6%, 13/15
  and 7/7; AUC +2.3% / +3.1%; coup count CRPS -11.1% / -1.6%) but civil conflict slightly worse (country Brier +0.8% /
  +0.2%, onset AUC -1.0% / -0.8%, world share +1.8% / -5.5%). Hence per-task inputs: features.TASK_ONLY lets these
  six feed coups and autocratisation only; every family blanks them for the other tasks (models.task_blocked: GBM and
  logit per task, the neural networks with one extra training pass whose snapshots see them blanked). Checked: the
  other tasks' predictions do not move when the six change. Hindcasts of this version: sim_eval co2.
- **Regime coalition for coups and autocratisation: ADOPTED** (sim_eval co2 = TASK_ONLY version against co0 = the same
  code with PSYCHO_DISABLE=coalition): coup Brier -2.9% dev (13/15), -3.7% test (7/7); coup AUC +1.8% / +2.8% (14/15,
  7/7); coup count CRPS -6.3% / -3.2%; civil conflict, GDP and democracy unchanged (within 0.3%).
- **Simulator hindcast noise from the hazard networks' initialisation (measured 2026-10-28).** co0 (six inputs that
  are always missing) against dm_fn (the same model without those columns) differs only in the networks' random
  initialisation, yet: country conflict Brier +0.7% dev / +1.2% test, world conflict share CRPS +3.1% / -7.4%,
  new-onset count CRPS -11.2% / +12.2%, coup count CRPS -4.7% / +1.9%; GDP and democracy within 0.1%. Differences of
  this size in the conflict and onset metrics are noise when the network shapes differ. The personal-rule check
  (pr0 with PSYCHO_DISABLE=personal against pr1) and the equation checks kept the same shapes and seeds, so they are
  not affected by this; comparisons against a run with a different feature count (as co1 or co2 against dm_fn) are.
  Simulator comparisons should use same-architecture baselines (as co0) or several seeds.
- The GDP and democracy metrics moved by at most 0.1% between co0 and dm_fn, so the fiscal-capacity (GDP CRPS -0.5% to -3.1%) and democracy-equation (democracy CRPS -3.0% / -6.5%) adoptions are well above this noise.
- **Pew religious composition 2010-2050: not wired in.** The projected change 2010-2025 averages under 1 point in the
  Muslim and unaffiliated shares (largest: Nigeria +3.7 points Muslim, Sweden +2.7 unaffiliated); Pew's eight broad
  groups cannot rebuild the denomination-level fractionalisation of the World Religion Project. Too small to matter.
- **Screen of the individual V-Dem indicators** (scripts/residual_screen.py indicators: 224 indicators the model does not
  use, 1,563 tests against the current model's out-of-sample errors): 150 pass dev (|t| > 3.3), 7 confirmed on the
  lockbox. Two are artefacts: the regime's end type (v2regendtype, coded for every year of a regime from how it ended:
  future information) and the year of a referendum (v2ddyrrf, a calendar year acting as a time trend). The other four
  (international and domestic autonomy and social-group equality for economic collapse; the head of state dying in
  office for coups) are tested as a family (new_data_test.py sovereignty).
- **Sovereignty family (international and domestic autonomy, social-group equality, head of state dying in office): rejected** (baseline now with personal rule and coalition: tuned GBM dev 0.9190, lockbox 1.0043; with the family 0.9164 / 1.0072).
- **Religious backing of the regime for civil conflict too: rejected** (scripts/religion_conflict_test.py: letting sup_religious reach civil_conflict_5y and onset_1y lowers the tuned-GBM score 0.9190 -> 0.9180 dev, 1.0043 -> 1.0020 lockbox; civil-conflict AUC -0.004 / -0.003).
- **Strongmen and juntas in the pair models (Weeks 2012): juntas yes, strongmen no; not adopted**
  (scripts/strongman_dyad_test.py). Logit with geography, rivalry, power, democracy and conflict history, errors
  clustered by pair, 1946-2015, 5-year UCDP outcome: a military regime on either side +0.67 sd (s.e. 0.12), a leader
  who took power by force +0.43 (0.11), larger military share of the regime's support +0.24 (0.18, n.s.), leader-cult
  personalism -0.37 (0.19, n.s.), personalist autocrat ("strongman") -0.23 (0.19, n.s.). Backtests with the two clear
  ones (pair bootstrap of the Brier-skill gain): 5-year logit +0.021 dev [+0.001, +0.043], +0.020 test [+0.000,
  +0.048]; 5-year logit + GBM +0.014 dev (P 0.94), +0.007 test (P 0.69); the 1-year onset model that drives the
  simulator +0.003 dev (P 0.73) but -0.003 test (P 0.09). Rejected for the simulator; the 5-year gain alone is too
  uncertain to change the report's pair table.
- **Seed noise of the simulator hindcasts, measured** (scripts/sim_eval.py with {"net_seed": 1} and {"net_seed": 2}
  against the production configuration co2, seed 0; largest relative change over the four comparisons, dev and test):
  democracy CRPS 0.1%, country GDP CRPS 0.3%, world GDP CRPS 1.0%, country conflict Brier 1.2%, onset AUC 1.7%, coup
  Brier 1.3%, coup AUC 0.8%, world conflict share CRPS 7.5%, coup count CRPS 12.6%, new-onset count CRPS 25.4%.
  Against these: the democracy equation (-3.0% / -6.5%), fiscal capacity (country GDP -0.5% / -1.7%) and the coalition
  for coups (Brier -2.9% / -3.7%, AUC +1.8% / +2.8%) are well above noise; the personal-rule family's simulator gains
  (coup Brier -1.0%, conflict Brier -0.5%) are within it, so that adoption rests on the risk-model evidence. The world
  counts (conflict share, coups, onsets) cannot decide anything from one seed: earlier settings chosen on them (mood_sd,
  frailty, the coup alignment multiplier) were selected with this noise unaccounted for. **Rule from now on:** a
  simulator change counts only beyond these thresholds, or averaged over at least three network seeds.
- **Older dispersion settings re-checked over three network seeds** (scripts/sim_eval_seeds.py, which averages each
  configuration over its seeds; production = co2, ns1, ns2). World mood 0.20 instead of 0.10: world conflict share
  CRPS +4.1% dev (production better at 11/15), +10.7% test (5/7); other metrics within 0.1%: **0.10 confirmed**.
  Frailty off instead of 0.6: test world share CRPS +11.2% (0/7 better), new-onset count CRPS +21.9% (0/7), country
  conflict Brier +0.4% (dev and test, better at 2/15 and 1/7); on dev the onset and ending counts sit closer to
  history without it (1.02 and 1.05 of actual against 0.93 and 0.92) but the scores do not improve: **0.6 confirmed**.

## Code review, 2026-10-28 (four parallel read-only reviews; every finding re-checked against the code)

Reports: outputs/review/{sim,world,models,scripts}.md. About 60 findings; each was checked by reading the code and
running small checks. Most "high" items were false (e.g. "realised future features" in the backtests: a model frozen
at the origin scores test-year rows with covariates observed in those years; training only keeps labels that end by
the origin), documented design, or unused paths. No outcome leakage into risk-model training was found.

Fixed (production):
- COVID shock off by one (simulate.py): grid year y holds the y -> y+1 change, so the replay meant to drop COVID
  dropped the 2021 rebound and replayed the 2020 crash in its place (about -0.2 pt/yr of world growth in 2020s-era
  worlds). Now both the crash and the rebound are replaced by their neighbouring years.
- New civil wars started at one rebel group forever (lreb = log 2, never updated): +0.47 log-odds of ending for every
  post-origin war for its whole life. Now the neutral mean (FRAG_MEAN).
- Structural uncertainty (struct_sd) noise had no convexity correction, unlike mood and frailty; added.
- State-system entry: the pre-1950 V-Dem coup patch moved "entry" back for 19 countries (Baltics to 1918-20, Namibia
  to 1926 ...), so their pre-independence years counted as peace in training. Entry now uses Powell & Thyne years.
- V-Party: after a change of governing party the ousted party's rejection-of-violence score was carried forward
  (POL, BRA, GBR, ITA, SVK); now unknown (blank), as intended.
- Forest family ignored the per-task feature restriction (TASK_ONLY); it feeds the structural-uncertainty spread and
  the published intervals.
- War sizes: wars at war level in 2025 (Russia-Ukraine at its 2025 rate, 94,741; Iran-Israel) now keep their observed
  rate; an episode that dips to minor and re-escalates is no longer split into two wars.
- War-site rebound scales with the sampled war-site loss; collapses already under way at the origin are not counted
  twice; the 1914-alliances scenario uses the baseline's seed (paired comparison); the ensemble's network is copied.
Fixed (hindcasts and test harnesses):
- Refugee equation gave NaN for hindcast origins before 1980 and the host matrix was empty before 2015.
- warstructure_ensemble_test: the logit family never saw candidate columns and G60 trained from 1955, not 1860
  (biased toward "no gain"; the coalition adoption passed anyway). Placebo columns renamed so they cannot resolve to
  adopted real columns (the recorded placebo results were verified to have used shuffled columns).
- uncertainty_sims ran without climate damage and structural uncertainty (the "what drives the 2065 spread" result
  must be rerun); its "2045" outcome was 2046.
- sim_eval: world_gdp_crps scores the population-weighted mean of growth ratios, not world GDP per person; a correct
  world_gdppc_crps is added (the old one is kept so earlier comparisons stay comparable).
- crowd: question parsing ordered countries by alias length and lacked South Sudan, Guinea-Bissau, Equatorial Guinea,
  Papua New Guinea.
- disaster_rca: natural vs engineered pandemics and limited vs large nuclear use were overlapping; baselines now 2025.
Open (real, not yet fixed):
- pgm_fatal.py: the cell -> country map is built from the full 1989-2025 GED file, so a never-violent cell has country
  features only if it will see an event later (leak in the grid benchmark, not in the forecast). Needs PRIO-GRID gwno.
- US tariff exposure divides nominal imports by PPP GDP (about 2x too small for Vietnam/Cambodia; Taiwan gets none:
  no 2025 GDP). One-off 2026 effect; needs WDI nominal GDP.
- Hindcasts treat countries that did not exist yet as at peace (neighbour and regional conflict features, alignment
  risk sets; none at the 2025 origin).
- us_states.py house-price-to-income compares state indices with different bases; leave-one-origin-out recalibration.
- The 2010 lockbox origin's outcomes overlap the 2005 dev origin's (tuning optimism, not leakage).
- Correlated AI/robotics inputs in uncertainty_attribution (group them).

### Follow-up to the review: rebel fragmentation by war age; structural noise tail (2026-10-28)
- The first fix for the "one rebel group forever" defect set new wars to the mean (FRAG_MEAN). UCDP 1950-2022 shows
  groups multiply as wars age: mean log(1 + groups) 0.77 in a war's first year, 0.90 at 3-5 years, 1.00 at 12-20, 1.20
  beyond. New wars now start at 0.77 and move toward 1.22 (time constant 11.7 years; structure.FRAG_*).
- Effect on the headline worlds (same seed, 4,000 worlds, scratchpad war_end_test): old code mean share of countries
  in conflict 2026-65 18.7%, 2065 median 16.6%; with the age path 21.3% / 20.6% (the mean-groups variant gives the same,
  and so does the hindcast-validated plain simulator, which has no fragmentation term: the old code's lower level was
  the artefact). Onsets fall slightly (3.30 -> 3.03 a year) because wars last longer. The struct_sd convexity
  correction moves the mean share by +0.2 pt only.
- Open, top priority: structural noise (struct_sd) is the spread of the model families' log-odds, drawn once per world
  and shared by all countries. Families disagree most in log-odds where risks are tiny (onset sd: Finland 1.37,
  Sweden 1.38, Luxembourg 1.46 vs USA 0.43, median 0.52), so a 1-in-100 world multiplies the odds of civil war in every
  very-low-risk country by about 24. With longer wars this now shows in the extreme civil-war tail (Nordic and Baltic
  states at war half the time in the worst 1% of worlds). Never tested in hindcasts (sim_eval runs without the world
  layer). Candidate fix: each world draws which family is right and uses that family's own deviation from the
  ensemble (bounded by real model predictions) instead of a Gaussian shift scaled by the spread.

## Climate risk (2026-10-29)

Question asked: is climate taken into account the right way? Before this round the simulator used one damage function
(Burke-Hsiang-Miguel 2015, x 0.5-1.5), CMIP6 paths with a normal climate-response draw, a linear yield response to the
year's national weather, no tipping points, and climate never moved conflict (the 2065 conflict share was identical
under every SSP). Changes, each with its evidence:

- **Damage: three families, one per world** (world.CLIMATE_RANGES dmg_family, equal weights; climate_damage x0.5-1.5
  kept): 0 = BHM coefficients as a level effect (Newell, Prest & Sexton 2021: level models fit as well, 1-3% by 2100);
  1 = BHM growth effect (as before); 2 = global temperature (Bilal & Kaenzig 2024, NBER w32450: a linear +2 degC from 2024
  costs 19% of output by 2050 and 47% by 2100; implemented as -0.35 log output per degC of global-mean warming since the
  origin after a 5-year lag, global = 0.72 x population-weighted country warming). Hindcasts cannot separate them (too
  little warming in-sample). Headline worlds (4,000, scripts/climate_risk.py): world GDP per person 2065 $41.2k (level),
  $35.4k (BHM), $29.2k (global), all $34.7k.
- **Heavier climate-response tail**: per-world draw Student t (4 df) scaled to keep the CMIP6 10-90% range.
- **Harvests**: the yearly weather terms explained ~15% of US yield swings (national annual heat and rain are too
  coarse) and gave ~1% yield sd vs 7.6% observed. Now each year replays the historical yield residual (log yield minus
  centred 11-year median, FAOSTAT 1961-origin) of the same historical year as the growth shocks, so breadbaskets fail
  together as they did; bad years deepen by harvest_amp x local warming (u(0, 0.3) per degC: 0 = national data, pooled
  convexity -0.004 per degC^2 of hot-day anomaly; 0.3 ~ crop models, Tigchelaar et al. 2018). No GDP channel (the damage
  functions already include farm output); harvests move world grain prices, calories, import needs, migration.
  US harvest 15%+ below trend: 6.2% of years 1961-2024, 5.6% simulated 2026-35, 8.4% 2056-65; two of the four big maize
  exporters 10%+ below trend together: 1.6% / 1.4% / 4.6%; US 25%+ below trend at least once by 2065: 27% of worlds.
- **Drylands (estimate)**: 42 countries with 1961-1990 rain under 600 mm; decade changes 1961-2022 with year effects,
  errors by country: cereal area elasticity to decade-mean rainfall 0.71 (s.e. 0.29); cropland 0.22 (0.15); wetter
  countries -0.22 (0.25, none). Sampled n(0.71, 0.29), applied to each dryland's CMIP6 rainfall path.
- **Tipping points (assumptions)**: AMOC collapse effects beginning 2035-2065 with chance u(0, 10%) per world (AR6 "very
  unlikely before 2100", medium confidence; recent studies higher for tipping, effects lag by decades): Sahel / West
  Africa rain -10 to -30% (South Asia half), north-west Europe output -1 to -5% and harvests -10 to -30%, ramping over
  15 years: 4.7% of worlds by 2065. Fast ice-sheet loss: 10% of worlds, sea-level path x1.5-3.
- **Climate and civil war** (scripts/climate_war_test.py): sustained warming (10-year mean minus 1961-1990) interacted
  with farm share or poverty fits onsets with a common time trend (warm10 x agri +0.34 sd, p 0.018; x poverty +0.39 sd,
  p 0.021; 5-year conflict), but adding the seven columns made the tuned GBM worse (dev 0.9185 -> 0.9144, lockbox
  1.0038 -> 0.9941). Not in the risk models; in the simulator as an assumption, onset log-odds + clim_conflict x local
  warming since the origin x farm share / 0.3 (capped x2), clim_conflict u(0, 0.3) (0 = forecast test; 0.3 ~ Burke et
  al. 2009 for Africa, a quarter of the unit fit, which implied x3.2 odds per degC at a 30% farm share). Effect on the 2065
  share at war: 21.1% (lowest fifth) vs 21.7% (highest fifth).
- **Climate emigration (assumption)**: a stock = clim_migr x farm share x cumulative climate harvest loss x population,
  moved to refugee host shares; u(0, 0.4) (Groundswell 2021: up to 216 m internal climate migrants by 2050; u(0, 1) gave a
  50 m median abroad, too high): 21 m median, 43 m 95th percentile by 2065.
- **Attribution** (uncertainty_sims, 48,000 worlds, groups in uncertainty_attribution.py): climate explains 8.8% of the
  spread of 2065 world GDP per person (12.1% in 2045, more than AI and robot timing's 11.4% then), AI and robot timing
  48.6%; the damage family alone 6.7%. Conflict: 0.3%.
- **To 2100** (scripts/climate_2100.py, paired worlds with and without damage, ratio only): SSP2-4.5 -14% in 2065 and
  -26% in 2100 (90%: -51% to 0%); by family 2100: level -1.4%, BHM -29%, global -38%. SSP5-8.5 2100 -43% (-80% to -1%),
  SSP1-2.6 -13%.
- Fixed: report "Known limits" said war and climate damage do not reach the political hazards (they do, damage_feedback);
  knockouts setting a parameter to its default were still sampled (climate knockouts use 1e-9); a fixed dmg_family fell
  back to BHM.

## Limits round (2026-10-29, evening): fixes, full-system hindcasts, new data

A review of the model's limits produced a ranked list; everything on it was done or tested. Rules as before: a change
to the simulator is kept only if it improves the hindcasts on the development origins and is confirmed on the test
origins; a new data source enters a risk model only if it passes the residual screen (development |t| > 3.3, lockbox
same sign, p < 0.05).

### Bugs fixed (each verified against the code first)

- Storm damage scaled with global-mean warming instead of each country's local warming.
- The climate-conflict, fisheries and climate-migration channels ran only when their assumption was being sampled, so
  fixed-setting runs (knockouts, scenarios) silently dropped them; they now follow climate_damage and the world layer.
  The fixed defaults are now the midpoints of their ranges (amoc_p 0.05, clim_conflict 0.15, clim_migr 0.2).
- The transient harvest amplification entered the climate yield loss that drives emigration (double counting); the
  dryland cereal-area change was computed after that loss was accumulated, so it never reached emigration.
- World population for the food balance and the world averages ignored simulated migration, deaths and births.
- Nuclear damage compounded each year a country stayed hit; it now applies once, at its worst level.
- Nuclear escalation and cyber losses used the static COW major-power list instead of the dynamic great powers.
- Two bugs that only affected hindcasts before 1995 (found by the new full-system hindcasts): the oil price's pull
  toward its average used 1995-to-origin, empty for earlier origins, which made every country's GDP undefined; 1975
  origins had no grain-price residual to replay.
- A stale smoke test (AI arrival share by 2032 after the arrival distribution was revised).

### Growth equation: no extrapolation beyond the data (simulate.SUPPORT_Q)

30-year full-system hindcasts (from 1975, 1985 and 1994; scripts/world_hindcast.py) showed countries at war in 1994
(Afghanistan, DR Congo, Liberia, Rwanda, Angola) falling by 3 to 6 log points of GDP per person in the median world (95%
to 99.7%); all of them grew. Cause: the growth equation's squared distance-to-frontier term, fitted where data exist,
turns into an accelerating collapse once a war-torn economy falls below any observed income (fitted to 1994: -6% a
year at 4 sd below the mean distance, -16% at 6 sd). Features are now held inside their 0.5-99.5% training range
(they were clamped at +/-6 sd). 10-year hindcasts: country GDP CRPS -1.2% on development (better at 14 of 15 origins),
-0.8% on test (7 of 7); world GDP per person CRPS -1.5% / -1.2%; conflict, coups and democracy unchanged. 30-year
cross-country spread of growth in the median world: 0.76 / 0.82 / 1.43 -> 0.68 / 0.72 / 0.93 (actual 0.57 / 0.56 / 0.48).

Still too wide: the cross-country spread is above the actual at every horizon (15 years from 2000-2009: about 0.53 vs
0.26-0.35), although country 90% bands cover 88% and 50% bands 52% on average at 15 years, so the excess is mostly the
spread of country medians, not band width. World recessions are too rare (the median world never has a falling year
in 15; history had 1-2), and the democratic decline since 2000 falls at the 1st percentile of the simulated change.

### Full-system hindcasts and history matching (scripts/world_hindcast.py, scripts/world_history_match.py)

The simulator with everything on (AI off, every assumption sampled per world) from 1980, 1985, 1990, 1995 (development)
and 2000, 2005, 2009 (test), 15 years, 2,000 worlds each; ten world statistics (GDP growth, falls, volatility and
dispersion; share of country-years at war; onsets; coups; democracy change; interstate conflicts; oil price change).
Ridge emulators of each statistic on the sampled assumptions explain only 7-12% of its spread across worlds: history
cannot tell the assumption values apart. Weighting worlds by their fit on development moved no range by more than 0.22
sd, and on the test origins the weights helped coups and onsets but hurt interstate conflicts and oil. The assumption
ranges are left unchanged.

30-year world GDP growth shows no consistent bias (actual at the 9th, 62nd and 73rd percentile), so there is no ground
for a growth-trend correction. Benchmark (scripts/rff_benchmark.py): RFF Socioeconomic Projections (no climate
damage, no AI) world GDP per person 2065/2025 median 1.86 (90%: 0.87-3.97); this model without AI and with climate
damage 1.36 (0.81-2.23), headline 1.91 (0.94-5.00), SSP2-4.5 without damage (AI uncertain) 2.25.

### Wars between states

- Force budget (scripts/force_budget_test.py): pairs where either side is already fighting another interstate
  conflict start new ones less often, development -0.78 (p 0.15), test -0.54 (p 0.57): same sign, not significant.
  Added as an assumption busy_beta u(0, 0.8) on the onset log-odds.
- War feeding back into the event-data hostility features (ev_war_feedback; persistence and the effect of an ongoing
  war estimated by least squares on 1995-origin pairs): pair AUC 0.917 vs 0.911 over the 1995-2014 origins (2000
  better, 2010 worse), no change before 1995 (event data start then). Neutral; kept as an option, off.
- Trade weights moving with simulated GDP (trade_gravity; partner GDP ratio from the median world every 5 years):
  every full-system statistic within 0.1% at every origin. Kept as an option, off.

### New data in the simulator

- **Population uncertainty**: UN WPP 2024 probabilistic projections (data/wpp_prob.npz, scripts/build_wpp_prob.py):
  per country the sd of log total fertility and of the remaining (mortality and migration) error by year, fitted to the
  UN 80% intervals; one draw per world and country (countries independent: the fitted correlation is 0, which
  reproduces the UN world interval, 2065 sd 2.4%). Small run: simulated population sd matches the UN, median ratio
  1.01. A first version used the 95% intervals: for Kuwait and Hong Kong the UN's lower bound reaches zero after 2080,
  the spread became infinite and two of 2,000 worlds in the 2100 climate runs lost every country's GDP; the spread is now
  capped at 1.0 log points and the simulated population offset is held within a factor of 55 (e^4) of the UN path.
- **Leader ages after 2020**: V-Dem leader names joined to Wikidata birth years (161 of 167 spells; five checked by
  hand), carried to 2025 (data/leader_birth_wd.npz); missing ages in 2025: 3 countries (were most).
- **AI exposure**: the share of employment in ISCO groups 1-4 (managers, professionals, technicians, clerks; ILO
  modelled estimates, 174 of 175 countries) replaces the income-based guess (rank correlation 0.885; China 0.63 ->
  0.53, Vietnam 0.51 -> 0.41, Nigeria 0.41 -> 0.58).
- **Cyclone exposure**: share of land under hurricane-force winds (64 kt+) in IBTrACS tracks since 1980 (Taiwan 32%,
  Philippines 14%, USA 0.3%) replaces the latitude proxy, rescaled to the proxy's mean.

### Tested and not used

- IMF Global Debt Database: corporate-debt build-up predicts economic collapse on development (t 7.1) but not on the
  lockbox (p 0.61); five-year private credit boom (scripts/credit_boom_test.py): t 0.12 (2.19 with year effects), test
  log loss 0.18321 -> 0.18302.
- PLAD leader ethnicity: a minority-group leader and autocratization, development t -4.3, lockbox p 0.8.
- Not done: GDL subnational data and ACLED (registration; ACLED cannot be redistributed); purges (built on WhoGov,
  already rejected); election calendars (V-Dem covers 2025); SPEI drought (harvests are replayed, no change); Epoch
  compute shares (China undercounted; AI hubs unchanged); subnational regions and a tree-model simulator (weeks).

### Rerun after the round (forecast, uncertainty, disasters, climate)

World GDP per person 2065 median $34.6k (was $34.7k; 90%: $16.9k-$90.5k); share of countries in civil conflict 2065
median 22.9% (22.3%); democracy unchanged. Climate (paired worlds): SSP2-4.5 -14% in 2065, -27% in 2100 (90%: 0 to
-52%); SSP5-8.5 -44% in 2100; by family 2065 $40.9k / $34.9k / $28.5k; climate emigrants 19 m median (95th
percentile 38 m). With climate damage off, conflict in 2100 is now the same under every SSP (21.1%; the climate
channels used to keep running in the no-damage worlds). Attribution: assumptions 67% of the 2065 GDP spread (AI and
robot timing 49%, climate 9%). Disaster knockouts: no climate damage or the smallest estimate removes 94% of the
economic-collapse tail (was 88-90%).

## Limits round 2 (2026-10-30)

Targets: the three limits left by the first round (countries drifting apart over decades, too few world recessions,
the democratic recession at the edge of the simulated range). Candidate ideas from an independent read of the code,
each a switch tested alone and then in combination: 10-year hindcasts (scripts/sim_eval.py, 3,000 worlds, dev
1970-1998 x15, test 2000-2012 x7, tags l2*) and full-system hindcasts (scripts/world_hindcast.py, now with --consts and
--compare; 15 years from 1980-2009 x7 and 30 years from 1975/1985/1994, tags w15*/w30*). New hindcast score: CRPS of
the cross-country spread of 10-year growth (gdp_disp_crps) and its ratio to the actual spread.

### Adopted

- **Free expression gets its own equation** (dynamics.INST_SLOW): it used to move in fixed proportion to democracy
  (COMOVE), so its 5-year erosion, a democracy-equation feature (free speech goes before elections), could never lead
  in simulation. Now: yearly change on the same-year democracy change and conflict, reversion to its 20-year anchor,
  and its own shocks, sized to the part of 10-year changes democracy does not explain (sd 0.039 a year for a 2025
  origin) and scaled per country by its own yearly volatility, as the bounded slow variables are (HETERO). Two fixes
  on the way: (1) without the per-country scaling, the 28 countries above 0.9 could only be pushed down (-0.15 in
  40 years from the bound alone, -0.05 with it); (2) the yearly changes are not winsorised, unlike the other slow
  variables: the largest are regime transitions, where free expression jumps with democracy, and clipping them cut
  its slope on the democracy change from 0.84 to 0.50 (the fixed comove slope is 0.82) and hid its reversion (-0.011).
- **Personalist legitimation** (V-Dem v2exl_legitlead) in the democracy equation (beat baseline in all four
  dem_screen quadrants, never added), moving with democracy like the other V-Dem levels (COMOVE).
- Together (l2dem3, two network seeds): democracy CRPS -1.2% / -1.2% dev (12/15), -4.8% / -4.8% test (7/7); coup
  count CRPS +0.5% / -0.2% dev, +1.3% / +0.3% test; country GDP -0.3% dev (13/15), conflict and coup Briers unchanged
  or better. (With winsorised changes, l2dem2: -2.1% dev, -6.7% test; kept unwinsorised on principle.) Full system:
  democracy change CRPS -32% at 15 years from 2000-2009 (3/3), -18% at 30 years, but +9.7% at 15 years from 1980-1995:
  it lowers simulated democratic gains everywhere, and the 1980s-90s gains were large. The 2000-2015 democratic
  recession stays in the tail of the simulated worlds (see the percentiles after the rerun).
- Over 40 years from 2025 (bare simulator, 1,000 worlds, population-weighted world democracy in 2065): 0.499 with
  neither change, 0.469 with free expression's equation (0.467 with its shocks off: not the bounds), 0.458-0.463 with
  personalist legitimation (frozen or moving with democracy), 0.433-0.435 with both. Today's high personalism in
  populous countries (Russia, India) and the end of an overstated feedback (simulated democratic gains used to lift
  free expression in lockstep, which lifted democracy again) both lower the long-run path. Validated at 10-30 years;
  at 40 it is an extrapolation.

### Tested and rejected

- **GDP-weighted world year shocks** (YEAR_SHOCK_POW; the replayed shock as the GDP^w-weighted mean residual, not
  the equal-country mean). Uncentred, the gains looked large (w = 0.5: world GDP CRPS -6.5% dev, -16% test; recession
  years 1.8% -> 2.2%) but came from a negative mean (the big economies' average residual), which every country then
  inherited: 30-year world growth fell (1994 origin median 0.64 -> 0.53 log points, actual 0.84). Centred (timing
  only): world GDP CRPS -2.1% dev, -7.0% test, but simulated recessions became rarer (-6% dev, -23% test; the weighted
  shocks are smaller, sd 0.0096 vs 0.0106) and 30-year growth CRPS +4.4%. Rejected: it does not address the recession
  limit.
- **Output cost of ordinary banking crises** beyond their expected rate (CRISIS_COST, the 1952-2024 AI-crisis
  estimate 1.7% / 1.2%): every score within 1%, recession frequency +2%. The residual pools already carry crisis years.
- **Inflation norms converging** to the cross-country median (INFL_CONV 5% a year) and the science residual decaying
  (SCIENCE_DECAY 0.97): 30-year spread -0.9%, 15-year -0.7%, growth slightly worse. Frozen starting values are not
  what drives countries apart.
- **Recency-weighted noise pools** (POOL_HALFLIFE 10/20/40 years): the spread widened on both splits (+8-13% CRPS),
  because the decades before the test origins (1980s-90s) were the most volatile.
- **Judicial constraints with their own equation** too: no gain over free expression alone, larger coup cost.
- **Population-weighted regional democracy pull** (features.REGION_DEM_POW 0.5): democracy -0.9% dev, -3.6% test,
  but with the adopted pair it added nothing on dev and cost coups (+1.5% / +1.9%).
- **Anti-pluralism and polarization as drivers** of free expression and courts (yearly changes, fit 1960-1999, scored
  2000-2024): -0.5% to +0.3% MSE, wrong-signed; not built.

### What the drift is

The 10-year spread of country growth is about right from origins up to 1990 (0.85-1.2 times the actual spread) and
far too wide from 1992 on (1.3-1.9 times), with 90% GDP bands covering 77-88% of countries on dev and 92-96% on test.
Real economies, above all developing ones, became much calmer after the 1990s; the simulator draws country noise from
all history before the origin, and no history before a 2000 origin foretells that. No setting fixes both splits, so
it stays a stated limit; for the 2026 forecast the pools include 2000-2024, but the calmer era is a minority of them.

## Out-of-sample accuracy upgrades (2026-10-30)

Three out-of-sample improvements in the monthly fatality, grid-month and country-crisis layers, each selected on
development windows and confirmed on held-out test/lockbox splits:

### Adopted

- **Log-space quantile blending for country-month fatalities (`psychohistory/cm_fatal.py`, `psychohistory/pgm_fatal.py`,
  `scripts/cm_nn_test.py`, submission `seldon_v9`):** Both `QuantileModel` (LightGBM) and `NNQuantile` (MLP) predict
  quantiles on $\log(1 + \text{deaths})$ and exponentiate via `expm1`, but `cm_fatal.py` previously blended their
  sorted draws in level space (`(1 - w) * q_1 + w * q_2`). By Jensen's inequality, arithmetic averaging after
  exponentiation systematically inflates upper tails whenever the two models disagree. Blending rank-by-rank in the
  native $\log(1 + y)$ scale (`expm1((1 - w) * log1p(q_1) + w * log1p(q_2))`, zero new parameters, keeping $w = 0.5$):
  - **Development (2010–2017, 8 annual windows):** CRPS `15.7918 -> 15.7510` (`-0.0408`, better in 5/8 years),
    Ignorance `0.5052 -> 0.5051`, 90% MIS `287.5 -> 286.6`.
  - **2023/24 VIEWS Prediction Challenge Historical Test (2018–2023, 6 annual windows):** Mean CRPS
    `46.3306 -> 46.1926` (`-0.1380`), Ignorance `0.5697 -> 0.5692`, 90% MIS `773.05 -> 770.28` (`seldon_v9`).
    **Overtakes `submission_final_omm` (`46.2900`) to move from 3rd place to 2nd place overall in the challenge on
    CRPS**, while remaining **1st overall on Ignorance** and **1st overall on 90% Interval Score**.
  - **Challenge Live Test (July 2024 – June 2025):** `Y2024` CRPS `28.878 -> 28.748` (`-0.130`, MIS `326.3 -> 320.6`),
    `Y2025` CRPS `29.601 -> 29.323` (`-0.278`, MIS `372.4 -> 365.6`).
- **4-family nonlinear default for `SeldonEnsemble` (`psychohistory/models.py`):** `SeldonEnsemble.DEFAULT` was frozen
  at `{"w": {"G": 0.5, "N": 0.5}, "mode": "prob"}` on Oct 23 before the `Personal rule` and `Regime coalition`
  (`TASK_ONLY`) features were added, before pre-1946 civil conflicts and coups were patched for `G60` (`1860+`), and
  before the Oct 28 fix where `ForestModel` (`F`) ignored `TASK_ONLY`. On the current 147-feature panel, `F` is the
  strongest single family on `civil_conflict_5y` (`0.974` dev, `1.204` lockbox), `onset_1y` (`0.939` dev, `1.041`
  lockbox), and `termination_1y` (`0.915` dev), and `G60` beats `G` on all three conflict tasks, yet both had zero
  weight on all 5 default tasks. Replacing the 2-family default with an equal 4-family average across the four
  nonlinear families (`{"w": {"G": 0.25, "G60": 0.25, "N": 0.25, "F": 0.25}, "mode": "prob"}`, zero per-task weights):
  - **Development (1975–2005, 7 origins):** 4-task headline score `0.9669 -> 0.9723` (`+0.0054`, paired
    country-clustered 90% CI `[+0.0010, +0.0097]`, `P(gain > 0) = 0.980`); 7-task score `0.9447 -> 0.9532` (`+0.0084`,
    `99.9%` positive).
  - **Lockbox (2010–2015):** 4-task headline score `1.0293 -> 1.0396` (`+0.0102`, paired country-clustered 90% CI
    `[+0.0036, +0.0169]`, `P(gain > 0) = 0.992`); 7-task score `0.9958 -> 1.0020` (`+0.0061`).
  - **Untouched 2015 origin:** 4-task headline score `1.0146 -> 1.0279` (`+0.0134`, paired country-clustered 90% CI
    `[+0.0054, +0.0210]`, `P(gain > 0) = 0.999`); 7-task score `0.9934 -> 1.0039` (`+0.0105`).
  - Independent re-run (scripts not kept with the original; psychohistory/evaluate.py origins, members 8, both
    defaults scored on the same fits): headline 0.9670 -> 0.9716 dev (+0.0046), 1.0274 -> 1.0366 lockbox (+0.0093),
    2015 +0.0119; better at 8 of 9 origins (1985 -0.0035). Gains in civil conflict, coups and onset; one-year war
    ending worse on the lockbox (AUC 0.752 -> 0.744, BSS 0.126 -> 0.117), which the simulator takes from PsychoNet.
    The 2015 origin is the untouched check, reported here, not used for the choice.
- **Leak-free PRIO-GRID cell-to-country mapping (`psychohistory/pgm_fatal.py`, `data/raw/predcomp/priogrid_country.csv`):**
  `pgm_fatal.cell_country()` previously aggregated all years (`1989–2025`) of `GEDEvent_v26_1.csv`, which leaked future
  event locations into historical backtests and misassigned 64 cells in South Sudan to Sudan, 12 in Eritrea to
  Ethiopia, and 9 in Namibia to South Africa (where pre-independence events outnumbered post-independence ones). Built
  `data/raw/predcomp/priogrid_country.csv` from 1:10m Natural Earth country polygons (largest area intersection for all
  13,110 cells) and gated `cty12`/`cty1` in `features(D, Dc)` strictly by pre-origin cell activity (`D.sum(1) > 0`).
  Giving uninhabited zero-history cells (`D.sum(1) == 0`) their country's conflict totals was also tested on 2014–2017
  dev (`IGN 0.08418 -> 0.08069`, `-4.1%`, but `MIS 5.604 -> 5.820` and `CRPS 0.36632 -> 0.36641` because 95th-percentile
  trees assign small positive intervals to uninhabited desert cells in warring states), so the pre-origin activity gate
  is kept.

### Pending: model-family draws for structural hazard uncertainty

Proposed (patch kept aside, not adopted): replace the Gaussian structural draw (`z_struct * struct_sd`) with a
per-world, per-hazard draw of which model family is right (`net.struct_fam`, each family's log-odds offset from
PsychoNet). It removes the worst-1% Nordic civil-war tails (Finland 87.5% -> 35%, Sweden 87.5% -> 25% in 4,000-world
runs). Not adopted yet because it also moves the central forecast: the six families (logit and quantile net included)
are drawn with equal probability and the year-1 alignment restores only the world total, so country risk is
redistributed in every world (Nordic 40-year war risk 21-27% -> 12-15%), and it has no hindcast test. Next: draw
families with the ensemble's own weights, so the mean matches the published ensemble, and hindcast it.

### Tested and rejected

- **Adding nonlinear peace-decay (`1 / (1 + years_since_conflict)`) and contiguity–rivalry interactions to
  `interstate.DyadModel`:** Improved `Logit` AUC on `interstate_5y` (`0.8805 -> 0.8926` dev, `0.9036 -> 0.9225` test)
  and `interstate_1y` (`0.9031 -> 0.9206` dev, `0.9514 -> 0.9662` test), and improved `DyadModel` 5-year test BSS
  (`0.1456 -> 0.1612`), but diluted `LGBMClassifier` feature subsampling (`colsample_bytree=0.6`) on 1-year onset dev
  (`BSS 0.0563 -> 0.0533`) and reduced 1-year `Logit` dev BSS (`0.0479 -> 0.0398`). Not adopted because `world.py`
  uses `model_1y.lr` on the GPU.
- **Per-task Platt scaling / temperature sharpening / logit-mode blending on the 4-family country ensemble:** Adding
  `L` (`+0.05..0.10`), `F60`, or logit-mode blending did not beat the parameter-free equal 4-family probability average
  (`0.25 G + 0.25 G60 + 0.25 N + 0.25 F`) on both dev and lockbox.


## Analog member, AI rent capture, labour share (2026-10-31)

### Monthly fatalities: an analog (trajectory-matching) member (adopted)

The 2023/24 VIEWS challenge leader on CRPS, ShapeFinder, matches each country's recent course against similar past
episodes. Our gap to it was all 2023 (42.75 against 33.51; we led 2018-2021): very large wars that kept escalating,
where the production model expects large wars to fade. cm_fatal.analog_draws: each country's last 12 months of
log(1 + deaths), weighted with a 6-month half-life, matched to the 100 nearest windows of any country's history whose
outcome is known at the origin (ties among all-zero histories broken at random); each analog's change h months later
(from its own 3-month mean, the production model's residual target) is applied to the country's 3-month mean.
Blended rank by rank in log(1 + deaths) with the production draws (scripts/cm_analog.py).

- Development 2010-2017 (settings and weight chosen here): production CRPS 15.76; analog alone 14.97 (12 months, 100
  neighbours; 25 or 200 neighbours and 24-month windows were worse), better in 4 of 8 years, mostly 2013-14 (Syria,
  Iraq). Ignorance worsens with the analog weight (0.505 -> 0.518 at 0.5, 0.546 alone) while CRPS and the interval
  score improve (202 -> 192 at 0.5, 185 alone); weight 0.5 chosen as the balance before any test window was scored.
- Challenge test 2018-2023 (seldon_v10): CRPS 46.19 -> 45.61 (2nd of 21; the gap to ShapeFinder 1.09 -> 0.51), 2023
  42.75 -> 39.37; ignorance 0.569 -> 0.586 and interval score 770 -> 765, both still 1st. Live 2024 28.75 -> 27.61,
  2025 29.32 -> 28.91. The analog alone (seldon_v10x, scored for information): CRPS 45.35, ignorance 2nd.

### AI rent capture (assumption)

Who keeps the rents of near-free cognition, the firms that own the frontier models, chips and compute, or everyone?
Two per-world assumptions (own random generator, so earlier draws are unchanged): firm_capture U(0.2, 0.9), the firms'
share, and clawback U(0, 0.1), the yearly rate at which states take it back (tax, antitrust, nationalization) per year
of full adoption. Capture relative to its mean (0.55) scales the top-1% rise, the frontier hubs' extra rent (the
firms' home states gain) and half of the polarization rise; at the mean with no clawback every channel is as before.
Same 2,000 worlds, AI from 2030 (scripts/capture_test.py), 2065: top-1% share (population-weighted) 17.6% / 18.9% /
20.2% at capture 0.2 / 0.55 / 0.9 (17.2% at 0.9 with fast clawback; 15.2% without AI); polarization in democracies
1.15 / 1.36 / 1.56; the hubs' log GDP lead over the rest 1.26 / 1.32 / 1.36; world GDP per person $55.0k / $54.4k /
$53.9k; world democracy and civil conflict within 0.006. There is no history to estimate it from.

### Labour share (rejected)

Penn World Table 10.01 labour share and its 5-year change, the measurable trace of rents moving to capital: crisis
models 0 of 14 pass development (largest |t| 1.6, scripts/candidate_screen.py labour); growth equation 10-year
development MSE +1.4%; democracy equation 10-year development -0.45% but test +0.17%. FDI inflows had already failed
the growth screen (growth_screen.py invest).

## Cheap space and technology breakthroughs (2026-10-31; scenarios, not in the headline)

### Cheap space (world.WorldConfig.cheap_space; scripts/space_scenario.py)

Launch costs falling 10-100x, with constellations, space-based sensors and interceptors, satellite internet and orbital
compute deploying 2028-2040. Per-world assumptions (own generator): space_war U(-0.5, 0.7) on the log-odds of war onset
and on nuclear use for the United States against a nuclear-armed rival (negative: dominance deters; positive: eroding
second-strike confidence and anti-satellite races destabilise); sat_open U(0, 0.5), the share of AI surveillance's
protection of autocracies and of the AI drift to autocracy removed (censorship-resistant connectivity); orbit_dc
U(0, 0.5), the share of data centres' energy draw on Earth removed. Same 4,000 worlds with and without: every headline
within noise (great-power war 46.0% -> 45.1% with AI uncertain; democracy 0.400 -> 0.404); nuclear use 12.0% where
space dominance deters against 14.9% where it destabilises (no AI: 10.1% / 12.6%). Wars between the United States and
China or Russia happen in only 5-10% of worlds, so even a large shift on those pairs barely moves world totals.

### Technology registry (psychohistory/tech.py; scripts/tech_scenarios.py)

Seventeen technologies besides those already modelled (near-free cognition, general-purpose robots, autonomous
weapons, the solar-and-wind S-curve). Each has a chance of working at scale by 2040 and 2065, anchored where possible on
forecaster markets, expert surveys and agency outlooks (sources in tech.py; Metaculus values from undated snapshots,
several Manifold markets with few forecasters; judgement for green hydrogen and cheap ammonia). The anchors are
unconditional, so each technology's no-AI curve is calibrated so that its average over the model's AI arrival
distribution matches them (without AI, fusion at scale by 2065 falls from 25% to 10%). Per world: arrival through a
shared science-pace factor (correlation 0.5), the remaining wait divided by 1 + U(0, 1) once near-free cognition
arrives, prerequisites (space solar needs cheap launch), logistic adoption, and magnitudes drawn per world on existing
levers: the clean-energy S-curve's rate and ceiling (and so oil demand), energy access, the warming since the origin
(carbon removal, geoengineering, and clean energy beyond SSP2-4.5's own), yields and harvest swings, the world grain
price, water losses, pandemic losses and deaths, engineered pandemics, the working-age share, leaders' mortality,
banking-crisis odds at arrival, and war onset among rivals in monsoon Asia (geoengineering). Cheaper oil lifts no
importer's growth (their 1966-2023 response was to price rises only), so energy technologies help mainly through
warming and energy access and hurt exporters.

A first run had three faults, fixed before these results: civil conflict and coups were summarised by medians of
near-discrete counts (no change could show; now means), energy technologies had no climate effect (they lowered world
GDP), and longevity assumed people work through most of their extra healthy years (+14% world GDP; the range was halved).

Two further channels, added after the first results: geoengineering can stop (TERM_P U(0.01, 0.04) a year: 22% of
deployments within 10 years, 52% within 30), after which its effects fade over 5 years and cereal yields lose U(1%, 4%)
a year during the rebound; and code-breaking's extra crises now carry the AI-finance crisis cost (1.7% / 1.2%).

All seventeen drawn against none (2 x 4,000 worlds, AI uncertain, 2065): world GDP per person +2.5% (+2.3% without AI),
oil price -16%, clean share 64% -> 76%, world grain price -10%, the Sahel and East Africa +8.5%, petrostates +0.5%,
nuclear use 12.9% -> 13.6%, great-power war 45.5% -> 46.4%, mean excess deaths +12% (the engineered-pathogen tail;
the median is unchanged); civil conflict, coups and democracy within noise.

One technology forced to arrive in 2035 against never (the same 4,000 worlds, everything else drawn), 2065:

| Technology | World GDP | Oil | Grain | Other |
|---|---|---|---|---|
| Longevity | +7.3% | | | working-age share |
| Geoengineering | +5.9% | | +2.3% | Sahel +10%; great-power war +3.2 pt (monsoon-Asia rivalry); termination rebounds; excess deaths -6.3 million (fewer heat deaths) |
| 100-day vaccines | +1.9% | | | excess deaths -5.6 million |
| Fusion | +0.7% | -14% | | petrostates -1.1%, clean share +17 pt |
| Energy storage (days of grid power) | +0.6% | -5.7% | | |
| Advanced fission | +0.4% | -7.3% | | |
| New protein | 0% | | -1.9% | |
| Engineered crops | -0.2% | | -5.6% | |
| Engineered pathogens | -1.4% | | | excess deaths +14.6 million (median +4.0) |
| Quantum code-breaking | -1.6% | | | a wave of financial crises |

Civil conflict moved by at most 0.3 points and coups by under 0.2 a year for any technology: in this model,
technology changes economies, not the political drivers of crises.

### In the headline (2026-10-31)

world.WorldConfig.tech is on by default for forecasts from 2025 (hindcasts from earlier origins are unchanged; the
cheap-space comparison runs with it off). Headline 2065 against the previous release: world GDP per person $34.5k ->
$35.8k (90%: $17.7k-$91.6k), without AI $24.5k -> $25.3k, civil conflict 22.9% -> 22.3%, world democracy 0.398 ->
0.403, great-power war 45.2% -> 46.6%, nuclear use 13.6% -> 13.2%. Arrival years enter the attribution of uncertainty
(group "technology": 1% of the 2065 GDP spread on its own; AI and robot timing falls from 49% to 36%, because
breakthrough timing rides partly on AI timing).

### Climate paths keep their meaning (2026-11-01)

With the technology registry in the headline, scripts/climate_2100.py first reported the middle path's 2100 loss at
10% instead of 27%: geoengineering, carbon removal and clean energy beyond the path were being drawn on top of every
emissions path. The four SSPs now run with the registry off (2065 / 2100 median losses: SSP1-2.6 -10% / -14%,
SSP2-4.5 -14% / -27%, SSP3-7.0 -17% / -37%, SSP5-8.5 -21% / -43%, unchanged from before), and ssp245_tech adds the
breakthroughs as drawn in the headline: -11% in 2065, -10% in 2100 (90% range -38% to +3%), so where they arrive the
late-century damage stops growing.

### Report structure (2026-11-01)

The report is ordered top down: what the model is (a simulation of humanity, not only of conflict), the findings in
one page (outputs/summary.md, each with a link to its evidence), what decides which future we get (the attribution
table), the fan of futures, then six parts with a one-paragraph takeaway each (economy, AI and technology; climate
and food; conflict and war; the worst futures; countries; can you trust it?) and an appendix (the commentary, every
finding with its dated history, what improved the forecasts, the independent review). The hero chart is world GDP per
person instead of the civil-war share.

### Heat and cold deaths (2026-11-01)

Excess deaths counted only wars and pandemics, so geoengineering showed more deaths (from monsoon-Asia rivalry) and none
of the lives it saves. world.WorldConfig.heat_mortality (on): yearly deaths per person = HEAT_MORT_BETA (0.553e-4) x
heat_mort (per world, U(0.25, 1.75)) x sign(dT) |dT|^1.7 (local warming since the origin, smoothed path) x
clip((T_base - 16) / 8, -1.5, 1.5) (population-weighted baseline temperature: hot places lose, cold places gain) x
clip((income / $15,000) ^ -0.4, 0.25, 3) (adaptation), each capped at +/-0.4% a year; counted in excess deaths and
taken off the population, only in worlds with climate damage. Calibration (1,000 worlds to 2100, registry off):
SSP5-8.5 reaches Carleton et al.'s (2022, QJE) 73 per 100,000 in 2100; the power 1.7 brings SSP2-4.5 to about a
quarter of that (a linear response gave a third; their 84% is for mortality costs, adaptation spending included).
climate_2100 (deaths a year 2065 / 2100; cumulative 2026-2100): SSP1-2.6 0.4 / 0.35 m, 21 m; SSP2-4.5 0.85 / 1.8 m,
63 m; SSP3-7.0 1.4 / 5.4 m, 133 m; SSP5-8.5 2.2 / 8.9 m, 218 m; SSP2-4.5 with breakthroughs 0.5 / 0.4 m, 28 m.
Headline excess deaths 2026-2065: median 18 m -> 29 m; other headlines within noise.

A bug found with it: scripts/tech_scenarios.py summed the excess-death record over the years, but the record is a
running total, so every technology's death effect was inflated about twentyfold (fast vaccines -81 m was -5.6 m;
engineered pathogens +187 m was +15 m). The forecast itself always read the last year. Corrected results, technology
forced in 2035 against never (2026-2065 excess deaths, mean): geoengineering -6.3 m, 100-day vaccines -5.6 m, fusion
-1.8 m, fission -1.1 m, energy storage -0.9 m, engineered pathogens +14.6 m (median +4.0 m), others within 0.2 m.

### Structural uncertainty, tested on history (2026-11-01)

The structural draw (which risk model is right) had never been hindcast: the full-system hindcasts used PsychoNet alone.
scripts/world_hindcast.py --struct fits the risk ensemble (GBM, GBM from 1860, PsychoNet, extra trees; 3 members) at
each origin and runs every mode on the same fit, 2,000 worlds, 15 years from 1980-2009 (7 origins) and 30 years from
1975/1985/1994. Two new statistics: country-level conflict Brier, and safe20_wars (how many of the 20 countries at
peace with the lowest simulated risk go to war in the window). Modes (simulate.STRUCT_MODE):
- gauss (the production draw until now: one shared Gaussian shift per world and hazard, scaled by the families'
  disagreement), against no structural noise: conflict share CRPS -2.5% / +16% (15 years, 1980-95 / 2000-09), war
  onsets +6.6% / +38%, coups +16% / +1%, safe tail -6% / -11%; 30 years onsets +8.6%, coups -9%.
- family (each world and hazard takes one family's offset from PsychoNet, drawn with the ensemble's weights; the
  patch kept aside on 2026-10-31): the safe tail -30% / -68% / -44% (30 years), onsets -7% / +1% / -12%, but the
  level of conflict drifts up (30-year median share 0.22 against 0.19 without, 0.18 actual; conflict share CRPS +46%,
  coups +28%): the families' average view differs from PsychoNet's, above all on how long wars last, and the year-1
  alignment does not hold it over decades.
- familyc (adopted): the family offsets centred on their weighted mean, so a world takes one family's spread without
  its level. Against no noise everything within +/-2% except coups (+8% 1980-95, -0.4% 2000-09, -8% at 30 years); the
  country Brier is unchanged. Against gauss: onsets -5.5% (4/4) / -27% (3/3) / -7% at 30 years, conflict share +2.5% /
  -12% (3/3), coups -6% / -1%, the safe tail +5-8%.

What it shows about the Nordic tail: only the uncentred family draw removes it, by lowering the level of risk for very
safe countries (the other families see them as safer than PsychoNet does). The tail is in PsychoNet's central risk for
the safest countries, not in the structural noise; using the ensemble's central hazards in the simulator, with an
alignment that holds over decades, is the next open item.

## Round of 2026-11-02: review bugs, five tests, an AI tail, reader tools

### Bugs (independent review)
- The centred family draw (STRUCT_MODE familyc) had no convexity correction: a log-odds spread that averages zero
  still raises a low probability on average, most where the families disagree most (the safest countries). It now
  takes -0.5 x (each country's weighted spread between families) x (1 - 2p), as the Gaussian draw did. With it the
  safe-country tail improves against no structural noise (-4.7% / -5.1% / -2.5% at 15 years 1980-95 / 2000-09 and at 30
  years; it was +5-8% worse without the correction), and the country-level conflict Brier slightly (0.08655 -> 0.08635,
  0.07466 -> 0.07449).
- Two random generators shared a seed (seed + 503): the climate-response draw in simulate and AI rent capture in the
  world layer, so every world's climate sensitivity was tied to its rent capture. Rent capture now uses seed + 521.
- The "no pandemics" knockout (scripts/disaster_rca.py) still let the technology registry's engineered-pathogen tools
  cause pandemics; it now also forces them off.
- The simple attribution (pipeline.attribution) credited each technology's arrival year with what AI timing already
  explains (arrivals are pulled forward by AI); each is now credited only beyond the AI arrival year.

### Tested and rejected (10-year hindcasts, two network seeds; full-system runs where noted)
- **Growth noise scaled by each country's own 10-year volatility** (VOL_NOISE): country GDP CRPS -0.6% dev, -2.1% test,
  but the cross-country spread +15% on dev (better at 1-2 of 15; -7.7% on test) and world recessions 26% rarer.
- **Asymmetric world shocks** (ASYM_SHOCK, busts at full size, booms x0.55, recentred): simulated recession years
  +40% dev (1.8% -> 2.5%, actual 3%) and +55% test (1.4% -> 2.2%, actual 9.5%), but world GDP CRPS +1.5% / +1.0% and
  world GDP per person +0.3% / +2.2%: closer to the recession count, worse on the scores.
- **Democracy x free-expression erosion** in the democracy equation (DEM_INTER): +0.3% / +0.2%.
- **Working-age share aged forward from the origin's cohorts** in hindcasts (DEMOG_COHORT, three broad age groups):
  country GDP CRPS +16% / +12%; too crude with only 0-14 / 15-64 / 65+.
- **The ensemble's level for war onset** (STRUCT_MODE ens_onset, with the convexity correction): the safe-country
  tail -33% / -77% / -51% and country-level Brier better (0.08595 / 0.07387), but the world's conflict share CRPS
  +13.6% on 2000-09 (0/3) and +6.3% at 30 years, war onsets +10.5% on 2000-09: it lowers the world's war risk in the
  decade before the 2010s rise. The most promising lead for the Nordic tail, if paired with an alignment that keeps the
  world total on target over decades.
(The full-system runs for the first four did not run: a shell error in the queue; they had already failed the
10-year tests.)

### Added: AI loss of control (world.WorldConfig.ai_catastrophe, in the headline; assumption)
Once near-free cognition arrives, a yearly chance (ai_cat_rate x frontier progress; log-uniform 0.0003-0.004 per
world, about 1% to 13% by 2100, between superforecasters' 2% and domain experts' 12% for an AI catastrophe killing a
tenth of humanity by 2100, FRI XPT 2022) of one catastrophe that kills 1-10% of the world's population and costs 10-35%
of output, half of it recovered over the following decade. With AI from 2030: 4.3% of worlds by 2065, mean excess
deaths 71 -> 96 million, 99th percentile 551 -> 860 million; with AI timing uncertain, 2.5%. Extinction is outside
what the simulator represents.

### Reader tools
- "Build your own world": five choices (AI arrival, climate-damage estimate, nuclear escalation, who keeps AI rents,
  breakthroughs by 2045), 405 combinations, each answered by the simulated worlds that match it (60-215 of the 48,000
  in outputs/uncertainty_sims.npz; 4 sparse combinations widened by dropping the least important choices).
- Flashpoints: any of the 20 watched pairs, its clash and war chance by year, under each set of assumptions, the
  five-year pair-model risks and the large-war tail.
- What changed since the previous version: headline numbers per release (outputs/release_history.json).

## Debt and currencies (2026-11-02)

What the model already knew: public debt / GDP and its 5-year change are inputs to the country risk models (from the
Global Macro Database), and a country in a banking crisis carries it as a feature. It did not have crises as events in
the simulated economy, interest rates, or the dollar's reserve role.

### Sovereign and currency crises (psychohistory/macro_crisis.py; scripts/debt_crisis_test.py; rejected, off)
- Data: GMD crisis dates 1960-2024 (sovereign onsets 81 in 1960-1999, 22 in 2000-2024; currency 140 and 45).
- Cost (growth, country and year effects removed): sovereign -1.2% (dev) / -2.9% (test) in the onset year and -1.2%
  the next; currency -2.0% / -2.7%, then -0.6% / -1.5%; nothing in the third year. Stable across periods.
- Hazard (logit, fit 1960-1999, scored 2000-2024): debt, its change, growth, inflation and a banking crisis raise the
  AUC from 0.47 to 0.64 (sovereign) and 0.54 to 0.60 (currency), but the log loss does not improve for sovereign
  crises (0.0580 -> 0.0582): few events after 2000.
- Rates (corrected twice). The panel's real rate clips inflation at 100% but not the nominal rate, so hyperinflations
  gave huge real rates exactly when inflation was eroding debt; with the real rate kept only where inflation is under
  50% (and within +-30 points), the real rate minus growth, times last year's debt, appeared to move debt (t = 3.9-6.3;
  out-of-sample R^2 of the yearly debt change 0.18 -> 0.20). Split into its halves, almost all of that is growth x
  debt (R^2 0.18 -> 0.26), which the simulator's log-debt equation already holds through its growth driver; the real
  rate x debt adds 0.004 (0.09 per unit, t = 2.3; the overall deficit, a control here, already contains interest
  paid). As a driver of the simulator's own log-debt equation (fit to 1995 or 2005, scored after), the real rate has
  no effect (coefficient -0.12 / -0.02; R^2 0.127 -> 0.124, 0.132 -> 0.132). No interest-rate regime was added.
- In the simulation (simulate.MACRO_CRISIS): a crisis drawn each year from the hazard, its cost charged to growth.
  10-year hindcasts: world GDP per person CRPS -12% on both splits and recession years closer to history, but world
  GDP growth CRPS +5.5% (1970-1998) and +13% (2000-2012) and the spread between countries wider. Full system: 15-year
  dispersion +39% (dev), 30-year growth +32% worse. The growth residuals replayed by the simulator already contain the
  average crisis, so charging crises on top double counts. A first centred version charged each country hit minus its
  own hazard, which has zero expected cost for every country, so it could only come out neutral (found by the
  independent review). Corrected: crises charged against the historical onset rate of the residual years (hit - pbar),
  so riskier countries pay more on average. Two network seeds: spread between countries +14% (1970-1998) and +5.6%
  (2000-2012), world GDP per person +4% and +14%, world GDP growth -1.3% and +0.8%. Rejected; off.

### The dollar's decline (world.WorldConfig.dollar_decline; scripts/dollar_scenario.py; scenario, not in the headline)
- Assumption, not fitted (there is one precedent, sterling after 1914-1945): each world draws how far the dollar's
  reserve role falls by 2045 (0-100%, ramped from 2026). At full decline sanctions on states at war with the West lose
  half their bite and US growth loses 0-0.3% a year (higher borrowing costs, the end of the "exorbitant privilege").
- Same 4,000 worlds with and without (SSP2-4.5, every assumption sampled). Uncertain AI: US GDP per person in 2065
  $110k -> $106k (-4.3%; -9% when the decline is deep), world GDP per person -0.5%, China unchanged, great-power war
  44.6% -> 44.7%, nuclear use unchanged. No AI: US -3.2%, great-power war 41.5% -> 42.3%, China-US war 8.4% -> 9.0%.
- Reading: in this model the dollar matters mostly to Americans' incomes; its strategic effect (weaker sanctions) is
  small because sanctions are a small part of what decides wars here.

### Debt-conditional growth shocks (simulate.POOL_STATE; 2026-11-02)
- The simulator's yearly country growth shocks were drawn from pools by income only. Split each pool by public debt
  (top third of country-years vs the rest), so a high-debt country draws from what high-debt countries went through
  (crises included, without charging them twice). 10-year hindcasts, two network seeds, identical on both: world GDP
  growth CRPS -3.5% (1970-1998, better at 13/15 origins) and -1.4% (2000-2012); spread between countries -3.1% and
  -2.5% (7/7 test origins); war onsets 0.0/-0.4% (dev, seeds 0/1) and -0.4/-0.8% (test); world GDP per person 0.0% and +0.7/+0.8%; coups
  +0.1/-0.1% and +1.3/+1.1%. Full-system hindcasts: 15 years (7 origins) spread -2.4%, world recessions -3.0%, world
  growth +1.2%, coups +1.7%; 30 years (3 origins) growth -4.1%, recessions -11.6%, spread +2.2%, coups +2.6%. Coups
  get worse at every horizon (high-debt countries draw harder slumps, and slumps drive coups), beyond the 1.4% allowed
  before. The nearest miss of the round: off, kept as a switch.
  The 5-year debt rise as the split instead: mixed (world GDP per person +1.3% / +3.0%). Debt and its rise in the
  growth equation's mean: mixed (world GDP per person +6% dev, -5% test).
- Year shocks per income tier (rich countries replaying their own synchronised recessions, beyond the world shock):
  measured with empirical-Bayes shrinkage, sd 0.1-0.2% a year against 1.2% for the world shock. Nothing to add.
- Democracy shocks scaled by population as growth's are (simulate.DEM_SIZE, a cheap form of the reviewer's
  "stable autocracies near 0 reflect off the bound" idea): every 10-year score within 0.1%. Not added.
- Oil (simulate.OIL_RESID): replay what is left of each historical price move after a fitted pull toward the 20-year
  average and the war entries of major exporters (+0.24 log, t = 3, 1955..origin), since the simulator adds both
  itself. 15-year full-system hindcasts, 7 origins: oil price change CRPS -6.4% (better at 6/7), but world GDP growth
  +3.1% (2/7; exporters' and importers' growth follow the price's departure from the replayed move). Mixed; off.

## Democratic recession and growth spread (2026-11-02, evening)

### The democracy equation's trend (simulate.DEM_DRIFT; rejected)
- Diagnosis: the democracy-change equation's level is the average yearly change of its training years (1955..origin,
  the third wave: +0.006 a year), so from 2000-2009 origins it predicts +0.004 to +0.007 a year for the average country
  while the next 15 years averaged about zero (-0.002 to +0.002). Full-system hindcasts: world democracy +0.011 to
  +0.023 over 15 years from 2000-2009 against -0.034 to -0.097 observed.
- Tested, 10-year hindcasts (network seed 0): no trend ("zero"): democracy CRPS +5.3% (1970-1998), +3.7% (2000-2012),
  onsets +1.7% / +3.6%, coups +0.6% / +4.6%; half the trend: democracy +1.6% / -1.6%, onsets +0.9% / +1.9%, coups
  +0.3% / +2.2%; the 1900-origin average (+0.002 to +0.004 a year, the 1920s-30s reversals included): democracy
  +0.4% / -1.6%, onsets +0.4% / +1.2%, coups +0.5% / +1.4%. Every version loses on 1970-1998 (the third wave was
  real), and lower democracy raises simulated onsets and coups on 2000-2012 beyond what happened. Not adopted: the
  recession after 2006 is not in any pre-origin trend the equation sees.

### What the "democratic recession" miss is
- The full-system miss is in population-weighted world democracy, and it is mostly India: its V-Dem score fell 0.19
  (2000-2015) and about 0.3 (2005-2024, 2009-2024), which alone is -0.033 of the -0.034 world change from 2000, -0.058
  of -0.081 from 2005 and -0.053 of -0.097 from 2009. The unweighted average country changed +0.033, -0.002, -0.037.
- World-level drivers tested offline against the democracy equation's errors (slope fit before the origin, scored on
  the 15 years after, origins 1995/2000/2005): rest-of-world GDP-weighted democracy (level gap, 5-year change) and the
  autocracies' share of world GDP (level, 5-year change; 19% in 2000, 25% in 2005, 46% in 2020). No out-of-sample
  gain (-3.6% to +0.2% in squared error); the share's 5-year change points the right way but overshoots twofold.
- Coverage of the 15-year world democracy change (percentile of the outcome among simulated worlds): 1980 91, 1990 66,
  2000 16, 2005 4, 2009 2. The lower tail is too thin (5th percentile about -0.07 at every origin): a large democracy
  backsliding by 0.3 (India 1975, Chile and Uruguay 1973, the Philippines 1972, Venezuela after 1999) is almost never
  drawn.
- Democracy regime scale (country-decade draws x DEM_REGIME_SCALE, 1.0, never tuned), against the 1.4 growth
  baseline. 1.5: country democracy CRPS -0.7% (1970-1998) and +2.9% (2000-2012, worse at 7/7), onsets -0.4% / -0.6%,
  coups 0.0% / -0.9%; full system 15 years: world democracy change -2.5%, others within 0.9%. 2.0: democracy -0.7% /
  +6.2%. Wider drift for every country to make room for one India-sized fall costs more on the many stable countries
  than it gains. Rejected; the tail stays too thin, a known limit.

### Growth regime scale 1.8 -> 1.4 (adopted)
- Country-decade growth regimes are drawn from history and multiplied by growth_regime_scale. 1.8 was chosen on
  26 October, when 1.4 looked like noise; the simulator has changed since (noise scaled by country size, slow
  variables, war damage), and at 1.8 the spread of countries' 10-year growth was 15% too wide on 1970-1998 (ratio of
  simulated to observed spread 1.15) and 66% too wide on 2000-2012.
- 10-year hindcasts, two network seeds, identical to the first decimal: 1.4 brings the 1970-1998 ratio to 1.03;
  spread CRPS -22% (1970-1998) and -29% (2000-2012, better at 7/7 origins), world GDP per person -3.2% and -11%
  (12/15, 7/7), country GDP +0.3% and -1.1% (7/7), onsets +0.2% and -0.8/-1.0%, coups -0.2/-0.7% and +0.8/+1.1%,
  democracy unchanged. 1.0 is better still on 2000-2012 but too narrow on 1970-1998 (ratio 0.92; country GDP +0.8%).
- Full-system hindcasts: 15 years (7 origins) spread -18%, world growth -4.6%, recessions -2.7%, others within 1%;
  30 years (3 origins) spread -28% (3/3), world growth +4.2% (1/3), safe-country wars +1.4%, others within 1%.
- The spread is still too wide after 2000 (ratio 1.48): the simulator replays decades in proportion, and national
  growth was far calmer after 2000 than in the 1970s-1990s (recency-weighted pools were rejected on 2026-10-30).

### Do coups overreact to slumps? (2026-11-03)
- Question: the debt-split growth pools (POOL_STATE "public_debt") were rejected on full-system coups (+1.7% at 15
  years, +2.6% at 30). Is the coup model too sensitive to slumps?
- scripts/coup_slump_test.py: the production network trained at each origin (1975-1995 dev, 2000-2009 test) predicts
  the next-year coup hazard on the years up to the next origin (features O..next-1, coups O+1..next; the last test
  window runs ten years): 3,651 and 3,035 country-years, no row counted twice. Calibration by last year's growth and
  the drawdown from the GDP peak, and a recalibration logit with the model's log-odds as offset (standard errors
  clustered by country).
- Normal and boom years are overpredicted on both periods (growth above 2%: 0.59-0.65 of the predicted coups on dev,
  0.46-0.50 on test). After a bad year real coups rose more steeply than the model says: growth term -2.9 (s.e. 0.7)
  on dev, -4.5 (1.7) on test; the model's own slope is -1.6 and +0.2, the raw historical one -4.3 and -4.9. The
  drawdown term is null on dev (+0.2, s.e. 0.5) but on test the deepest slumps were overpredicted (drawdown below
  -15%: 2 coups against 8.0 expected; term +3.2, s.e. 1.7).
- Out of sample (fitted on dev, scored on test), log loss: raw 0.05575, level shift 0.05434, plus drawdown 0.05430,
  plus growth and drawdown 0.05393 (-0.75%). The model's coup risk is too flat in one-year growth; damping the
  drawdown response gains nothing.
- (A first version of this test scored the years O+2..O+11 with overlapping windows and unclustered errors, and read
  the result as "no overreaction, nothing to fix"; an independent review caught it.)

### Debt-split growth pools, retested and adopted centred (2026-11-03)
- Uncentred (as tested on 2026-11-02), at the 1.4 growth scale against b14, two network seeds, 10-year hindcasts:
  spread -5.8% (1970-1998) and -3.3% (2000-2012, 7/7), world GDP -3.6% and -1.8/-1.9%, coups +0.1% and +1.2/+1.6%.
  Full system 15 years: spread -4.8%, recessions -2.4%, coups +1.8%; 30 years: recessions -12.6% (3/3), world growth
  -4.7%, spread +3.3%, coups +3.2%.
- Review (2026-11-03): splitting the residual pools without re-centring gives the high-debt pool a lower mean, so debt
  entered the growth mean through the noise, though it had been rejected as a growth-equation term. And the coup
  cost, paired runs on the same networks and draws, is a real effect: the 13% seed noise of the world coup count does
  not apply to a paired difference.
- POOL_CENTER: each sub-pool re-centred on its parent bin's mean, so only the shape of the shocks depends on debt.
  10-year: spread -4.3% (11/15, both seeds) and -2.5% (6/7, both seeds), world GDP -3.0% and -1.1/-1.3%, onsets
  +0.1/-0.2% and -0.1/-0.5%, coups 0.0% and +1.0/+1.2%, others within 0.3%. Full system 15 years: spread -3.2% (6/7),
  recessions -2.0%, interstate -0.9%, coups +0.9%, safe-country wars +1.1%; 30 years: recessions -11.3% (3/3), world
  growth -2.6%, spread -0.9%, safe-country wars -2.1%, coups +1.3%.
- Adopted centred: same gains in spread and recessions, the spread no longer worse at 30 years, the coup cost halved.
  It remains (0 to +1.3%), the price of more realistic slumps in coup-prone countries.
- Not fixed: countries without debt data at the origin draw from the lower-debt pool (3% of countries in 2010-2025,
  23% in 1990); the edge is a single 1955..origin percentile across income bins (about 56% of GDP for 2025), so the
  share of countries above it moves with the debt cycle: 47% in 1990, 51% in 2000, 24% in 2010, 43% in 2025.

### Revenge and revanchism in wars between states (2026-11-03, rejected)
- scripts/revenge_dyad_test.py: pair features from events that ended strictly before the year in question: a COW
  interstate war (1816-2007) with a winner and a loser between the two, dated by the end of the whole war, and a GML
  militarized dispute (1816-2010) in which the two themselves reached hostility 4-5, ending in victory or a yield; each
  decayed with a half-life of 20 or 50 years; "humiliation" is either. 16% of pair-years carry one.
- Dyad backtests, logit + GBM, origins 1975-1995 / 2000-2015. 5-year UCDP outcome BSS 0.187 / 0.159 without, with
  both features 0.166-0.169 / 0.160, with humiliation 0.180-0.182 / 0.156-0.158. 1-year onset model (the simulator's)
  0.040 / 0.0023 without, 0.034-0.044 / 0.0002-0.0036 with, on 24 and 11 onsets. The existing memory (years since the
  last conflict, ever in conflict, strategic rivalry, territorial threat) already carries it. Rejected. (A first
  version dated multilateral war outcomes by each pair's last year of fighting, counted co-belligerents that never
  fought each other, and included same-year events; fixing it did not change the result.)

### Belief cascades as regional mood jumps (2026-11-03, rejected)
- MOOD_JUMP (probability per region-year, size in log-odds): a rare, centred jump in a region's unrest added to both
  its war and its coup mood, fading with the mood (rho 0.9), for 1989- or Arab-Spring-like waves the smooth AR(1)
  mood may not make. The jump is skewed, so its mean-keeping correction uses its exact cumulant (a first version used
  the Gaussian one, which let rare hazards drift up).
- 10-year hindcasts, 1970-1998, against b14, corrected: (0.02, 0.6) world conflict share -0.5%, onsets +0.2%, coups
  +0.8%; (0.02, 1.0) share -0.6%, onsets +0.5%, coups +3.5% with the coup count on target (ratio 1.01): the cost is
  the extra dispersion, not a drift. First version, (0.05, 0.6): onsets +0.2%, coups +2.0%. Years with new-conflict
  counts in the outer 5% tails 0.167 -> 0.147-0.160 (0.10 expected). The existing mood already makes such years about
  as often as they happened. Rejected without a test-period run.

### Coup recalibration on last year's growth in the simulator (2026-11-03, rejected)
- COUP_GROWTH: coup log-odds + c x growth_1y (clamped to +/-0.3), before the alignment so only the slope changes,
  from the offset logit above (dev -2.9). 10-year hindcasts against the adopted centred-pool baseline:
  c = -2.9: country coup Brier -0.2% (12/15) and -0.3%, AUC +0.3% and 0.0%, but the world coup count CRPS +5.0% and
  +3.9% (the count 3% higher on test); c = -1.5: Brier -0.1% and -0.2%, world count +2.5% and +1.7%. Steeper slump
  sensitivity sharpens which countries but makes coups cluster in world recessions more than history did. Rejected.

### The safest countries' civil-war risk (2026-11-03, tested, nothing adopted)
- Found in the full-system hindcasts: among the 20 countries at peace with the lowest simulated risk, the simulator
  expects 2-3 to start a civil conflict within 15 years from every origin 1995-2009; none did (0 at all four). The
  2025 forecast has the same tail: Norway 19%, Germany 29%, Canada 33% chance of a new civil conflict by 2065.
- scripts/safe_tail_check.py (networks trained as in the hindcasts, origin-year onset hazards frozen, 15 years): the
  network itself is too flat at the low end, and the alignment makes it worse. From 1990 the safest 40-50% of countries
  at peace had almost no onsets (bottom half 1.4-5.4% over 15 years) against 9-17% from the network alone and 15-22%
  after alignment. The alignment shift was +0.3 to +0.6 log-odds from 1995, set by the previous decade's onset rate,
  which risky countries drove. Before 1990 the low tail was right (bottom half 12% predicted, 15-16% observed).
- Tried, each scored on the per-country chance of a new conflict (log loss) and in 10-year hindcasts (sim_eval, now
  with onset_brier, onset_ll and diag_safe20):
  - Alignment off (simulate.ALIGN_ONSET): per-country onset log loss -0.8% / -1.6% on 2000-2012, but total onsets fell
    to 0.70-0.77 of the observed and the world conflict share CRPS rose 11-15% on 1970-1998 and 117-138% on 2000-2012.
  - A slope on the onset log-odds before alignment (ONSET_SLOPE 1.15 / 1.3 / 1.5): worse on 1970-1998 at every value
    (onset log loss +0.1 / +0.8 / +2.5%, new-conflict count CRPS +4 / +12 / +29%).
  - Slope and intercept from the previous decade, scored with a network trained ten years earlier: mixed (slopes 0.75
    to 1.4 from origin to origin).
  - Recency-weighted training (PsychoNet sample_weight, half-life 20 or 40 years): no gain.
  - Calibration intercepts by risk half or third from the previous decade (with 2 pseudo-events): right after 2000,
    wrong before 1995 (1980 log loss 0.576 -> 0.643), because the calm 1970s low tail was followed by the busy 1980s.
- The post-1990 calm of the low-risk countries is a change of era that the history before each origin does not reveal.
  None of these was adopted; the remedy was the ensemble's central view (next section).

### Oil price spread (2026-11-03, adopted)
- Found in the full-system hindcasts: at 15 years the simulated oil price ran from 0.15 to 5-13 times its starting
  value (5-95%), yet 6 of 7 outcomes fell in the middle half, and the median was below the outcome at 6 of 7.
- Cause: the yearly oil move was replayed from the world's historical decade block, so a world drawn into the 1970s
  repeated that decade's rise ten times (or the 1980s' fall). The pull toward the 20-year average, 0.05 a year, was
  the one-year estimate; over longer spans the price returns much faster (slope of the change on the gap to the
  20-year average, 1965-2024: 1 year -0.05, 3 years -0.19, 5 years -0.34, 10 years -1.0).
- simulate.OIL_INDEP: each year's oil move comes from an independent historical year (own random stream, so the other
  draws stay paired). Exporters and importers gain or lose against the move inside the replayed growth shocks. The
  importer loss on price rises is taken as the effect of the simulated rise minus that of the replayed one; the first
  version clamped their difference, which charged importers on average (world GDP medians -2%) and was caught before
  adoption.
- 15-year full-system hindcasts against the then baseline (oilb15): pull 0.08 / 0.15 / 0.25 with independent years on
  1980-1995 gave oil CRPS -35 / -43 / -43%, world GDP growth -1.3 / -2.8 / -3.2%, world GDP falls +0.9 / +1.8 / +3.3%;
  0.15 chosen. Without independent years: pull 0.10 / 0.15 / 0.30 gave oil -23 / -33 / -37%; independent years alone
  -22%. Final version (corrected importer term), oilfx15 / oilfx30:
  - 1980-1995: oil -42.9% (4/4), world GDP growth -2.1% (4/4), growth spread across countries -1.2%, falls 0.0%.
  - 2000-2009: oil -32.4% (3/3), world GDP growth -1.2% (3/3), spread +1.3%, falls +0.6%.
  - 30 years from 1975, 1985, 1994: oil -11.2%, world GDP growth -1.6%, falls -1.9%, spread +0.9%.
  - Conflict, coups, democracy, interstate wars: within 0.4%.
  - 15-year oil 5-95% bands now 0.14-0.66x (1980) to 0.87-4.5x (1995); all 7 outcomes inside them, 3 of 7 in the
    middle half (3.5 expected).

### The ensemble's central view in the simulator (2026-11-03; withdrawn 2026-11-04)
- The simulator's hazards came from the neural network alone, aligned on the previous decade's rates; the forecast's
  risk ensemble (gradient boosting from 1955 and from 1860, the network, extra trees; equal-weight probabilities) was
  only used for the structural spread around it. scripts/safe_tail_check.py, 15 years, origin-year onset hazards:
  per-country log loss of any new conflict, network + alignment 0.500 (1980-1995) / 0.412 (2000-2009), ensemble 0.475 /
  0.367, better at all 7 origins; the 20 safest countries 1.2-2.0 expected new conflicts instead of 1.9-3.8. Gradient
  boosting alone 0.479 / 0.394, the network's own family 0.495 / 0.422, extra trees 0.508 / 0.417.
- simulate.ENS_OFFSET "oc": each country's onset and coup log-odds are shifted by the ensemble's minus the network's at
  the origin (clipped to +/-3), for the whole run, before the alignment; pipeline.attach_ensemble sets it for the
  forecast, uncertainty and disaster runs; the harnesses read scripts/ens_cache.py. With the coup offset the world coup
  count drifted up (ratio 1.08 on 1970-1998), so the coup alignment multiplier went 0.7 -> 0.66 (chosen on 1970-1998).
- 10-year hindcasts, 2 network seeds, against ao1 (sim_eval; dev 1970-1998 / test 2000-2012):
  - onsets only, alignment kept (eo): onset log loss -4.0/-4.5% / -6.1/-7.3%, onset AUC +2.8/+3.8% / +0.5/+0.9%,
    new-conflict count CRPS -6.7/-10.6% / -13.7/-15.1%, world conflict share -5.8/-4.3% / +2.7/-0.5%.
  - onsets only, alignment off (eona): better per country, but the world conflict share +45-56% on test. Rejected.
  - onsets, endings and coups (eotc, coup alignment still 0.7): coup Brier -5.3/-5.8% / -4.3/-3.4%, but the share of
    country-years at war 8% high on dev (the ending offset). Not kept. (Its coup count is not comparable with eoc66's,
    which also lowered the coup alignment.)
  - onsets and coups, coup alignment 0.66 (eoc66, adopted): conflict Brier -1.8/-0.8% / +0.6/-0.4%; onset log loss
    -4.0/-4.5% / -6.1/-7.4%; world conflict share -5.7/-4.1% / +2.6/-0.7%; coup Brier -5.1/-5.6% / -4.4/-3.5%, AUC
    +2.6/+3.2% / +3.0/+3.6%; coup count CRPS -3.8/-5.7% / -8.2/-6.5%; world GDP -0.7/-0.4% / -1.2/-1.1%; country GDP
    +0.0/+0.1% / +0.3/+0.3%; democracy -0.1%.
- Full-system hindcasts (ensf15 / ensf30 against the oil-adopted oilfx15 / oilfx30), 1980-1995 / 2000-2009 / 30 years:
  safe-country wars CRPS -27 / -50 / -21%, coups -4.3 / -8.2 / -8.9%, new conflicts -13.7 / +3.0 / -12.3%, interstate
  -0.9 / -0.3 / -0.3%, conflict share +0.9 / -1.9 / +3.9%, world GDP growth +0.5 / +1.2 / +0.7%, democracy change
  -1.8 / +0.6 / -0.1%.
- An earlier, uncentred version inside the structural draw (STRUCT_MODE ens_onset, without the coup part or the
  realignment) lowered the world conflict share by 14% on 2000-2009. Both are applied before the year-1 alignment; the
  difference is that ens_onset gave each world one family's level (with the convexity term subtracted on top of an
  uncentred draw, a review found in November 2026), while the offset gives every world the same central view and the
  family draw stays centred around it. Which of the two caused the 14% was not isolated.
- The harness runs above attached the offset without the forecast's family draw (STRUCT_MODE familyc). A review
  (2026-11-04) caught this; the combination is tested below, and scripts/world_hindcast.py now attaches both from the
  cache, as the forecast does. The same review found three scripts (uncertainty, disaster, the hindcast --struct mode)
  pooling the network seeds into the ensemble's own member, which then weighted one seed three times in the family
  average; pipeline.sim_net now builds the simulator's network for every script.

### The ensemble's central view, tested as the forecast runs it (2026-11-04): withdrawn
Full-system hindcasts with the forecast's own setup (risk ensemble at each origin, the family draw STRUCT_MODE familyc,
2,000 worlds; world_hindcast --struct, every variant on the same ensemble), against no offset; 1980-1995 / 2000-2009 at
15 years / 1975, 1985, 1994 at 30 years:
- onsets and coups (ENS_OFFSET "oc", the v53 candidate): safe-country wars -30 / -56 / -49%, new conflicts -5.6% (4/4) /
  +11.6% (0/3) / -11.4%, conflict share +0.3 / +7.2% (0/3) / +4.1% (0/3), coups -2.1 / -9.8 / +4.1%, world GDP growth
  -0.8 / +2.8 / -5.1%. Fails the test period on the world's war level: not adopted.
- coups only ("c"): coups -1.2 / -9.3 / +5.6% (1/3), everything else within 1%. Mixed: not adopted.
- the ensemble's view recomputed from each world's simulated state every 5 years (simulate.ENS_REFRESH 5, with "oc"):
  new conflicts -31% / +58% / -46%, conflict share +14 / -7 / +41%, coups -35 / -16 / -23%. The refreshed offsets
  move the level (the alignment is solved only in year 1), so the war level drifts up. Not adopted; worth retrying with
  the level re-solved at each refresh, so the refresh moves only who carries the risk.
The earlier tests above ran a stand-alone network without the family draw, which is why they passed.

### Two follow-ups (2026-11-04, evening): both rejected
- The ensemble's view refreshed every 5 simulated years with each hazard's level re-solved at every refresh
  (simulate.ENS_REFRESH 5, ENS_REFRESH_LEVEL), against no offset on the same ensembles, 1980-1995 / 2000-2009 at 15
  years / 1975, 1985, 1994 at 30 years: new conflicts -19.4% (4/4) / +16.9% (1/3) / -39.6%, coups -25.0 / -11.9 /
  +48.2% (0/3), conflict share +4.4 / -0.1 / +29.3% (0/3), safe-country wars -25 / -57 / -42%. Holding the level at each
  refresh does not hold it between refreshes: risk moves to countries that start and end wars repeatedly, and over
  30 years coups and the share at war run high. Rejected; ENS_REFRESH stays 0.
- Country frailty correlated between neighbours (simulate.FRAILTY_SPATIAL 0.5: each world's draw mixed half and half
  with its distance-weighted neighbours', exp(-km / 800), rescaled so every country keeps its variance), for which
  countries fight together. Full-system hindcasts: every statistic within 1.6% (new conflicts -0.3 / +1.6 / -0.2%,
  interstate -1.1% on 2000-2009). Joint test (scripts/joint_hindcast_test.py, 13 origins 1970-2006, 15 years, same
  network per origin): per-country conflict variogram score -0.02% (8/13), energy score +0.02% (4/13); world conflict
  by year energy score +0.45% (2/13). No gain in co-movement: rejected, FRAILTY_SPATIAL stays 0.

## Debate and harness parity (2026-11-05)
Six independent reviews (three per AI lab, two rounds; positions and rebuttals in the session's scratchpad, summary
below) agreed that the binding problem was the testing, not the model. What changed:

- **One simulator construction.** `pipeline.build_sim_env` builds the simulator's network exactly as `step_forecast`
  does (the risk ensemble from 1860 with 10 members, both PsychoNet seeds pooled into 20 nets, the family draw's inputs
  attached), cached per origin and refit seed in `outputs/sim_env/` (about 30 s per origin to fit). world_hindcast,
  sim_eval and joint_hindcast_test all use it. Before: sim_eval ran a stand-alone 8-member network and never ran the
  family draw (simulate applied it only with a world layer; `simulate(struct=True)` now runs it without one); the
  full-system cache (scripts/ens_cache.py) held a 3-member ensemble. `SeldonEnsemble(seed=)` offsets every family's
  seeds for independent refits.
- **Configuration records.** Every simulate() call returns `config` (network provenance, whether the family draw and
  the offset ran, every module switch, the world layer's settings, the call's arguments); world_hindcast and sim_eval
  save it and the forecast writes `outputs/sim_config_forecast.json`. `scripts/config_parity.py` lists every
  difference from the forecast that a hindcast does not need. It caught a run made before a switch existed.
- **joint_hindcast_test** now runs the forecast's configuration with the world layer on (it ran without interstate
  wars or cascades), resets module constants between variants (they leaked from one variant to the next), and adds a
  distance-weighted variogram (exp(-km / 800)) next to the unweighted one, which the many distant pairs dominated.
- **The adopt rule** is rewritten in docs/adopt_rule.md before any decision under it: 16 origins 1970-2009 at 15 years,
  three ensemble refits, one composite with fixed weights, at least 3 of 4 decades better, beyond the measured noise
  floor, no weighted statistic worse by more than 3%. scripts/adopt_score.py applies it.
- A regression test now checks that `simulate` stays under `@torch.no_grad()`: inserting a helper above it moved the
  decorator for the second time, and 2,000 worlds then ran out of GPU memory.

Bug fixes (open since the 2026-10-28 review):
- Countries not yet existing at a hindcast origin counted as countries at peace (simulate's nan_to_num on conflict):
  they diluted the year-1 onset alignment's risk set, so the shift went up for every country. They are now out of
  the alignment and have no wars or coups until the data has them (simulate.ABSENT_MASK; no effect at 2025).
- US tariff exposure divided nominal imports by PPP GDP (about half the exposure of developing exporters; Taiwan got
  none). It now uses current-dollar GDP from the External Wealth of Nations dataset (2024): Vietnam 2.7% of GDP, Cambodia
  6.1%, Mexico 1.1%, China 0.5%, Taiwan 0.3%.
- The annual ledger's persistence baseline gave every country at peace a 10% chance of war; it now uses the two rates
  estimated on the same proxy over the previous ten years (2025: staying at war 0.90, starting one 0.025). One
  canonical frozen forecast per release (`annual_ledger.py canonical`): version 53 is 737e06bf69cd.

Report:
- The methodology's headline paragraph is written from outputs/forecast.json with each number's known hindcast bias
  (scripts/headline_sync.py, `--check` before a release); it still quoted the 29 October forecast.
- A labelled scenario, not the headline: wars between states last as long as in the 2010s (continuation table
  half-life 5 years instead of 25: a first-year war continues with 0.75 instead of 0.43, a third year 0.91 instead
  of 0.78). Fitting the 2010s rise would use the outcomes that should test it.
- Crowd benchmark rebuilt on UCDP labels through 2024, refitted for each question's year (the MID labels end in
  2010). Pre-registered: keep the label set with the lower leave-one-pair-out Brier; pool weight stays 0.1. Result,
  63 questions: model Brier 0.130 (UCDP) / 0.132 (MID), crowd 0.100, blend 0.104 / 0.111; the 0.1 pool 0.0988 against
  the crowd's 0.0998. UCDP is now the default. The stale labels were not why the crowd wins: 13% of the questions
  resolved yes, while the model's mean probability is 2% (UCDP, 25 battle deaths) or 8% (MID): "use of force" is a
  lower bar than either label.

### Noise floor and re-checks under the adopt rule (2026-11-05)
Baseline nf_base: the forecast's configuration, 16 origins 1970-2009 at 15 years, 2,000 worlds, three ensemble refits.
Noise floor (twice the sd between refits, each against the mean of the other two): each country's years at war 1.1%,
world share at war 9.1%, new civil wars 10.8% (above 10%: no weight in the composite), coups 2.5%, world growth
0.8%, growth spread 0.5%, democracy 0.7%, wars between states 0.8%, oil 0.1%, growth volatility 0.04%; composite 1.1%.
So refitting the ensemble alone moves the war-level scores by up to a tenth: the ENS_OFFSET withdrawal (+11.6% new
wars, 0/3) was inside that noise, and so were several earlier single-period decisions on war counts.

Composite change against nf_base (negative = better; by decade 1970s / 80s / 90s / 2000s):
- treaty scope for alliances (WorldConfig.ally_out_scope sampled 0-0.5 instead of 1): +0.05% (+0.1 / 0.0 / -0.1 /
  +0.1); wars between states +0.6%, everything else within 0.1%. Hindcasts hold almost no out-of-area joining, so
  history cannot tell; adopted as a correction (the treaties' own areas), with the old rule shown as a scenario.
- year-1 alignment of war endings on the previous decade's risk sets (ALIGN_MATCHED "t"): +0.02% (-0.5 / +0.3 / +0.9 /
  -1.9), world share at war +1.1%. Rejected, although scripts/align_oracle.py put the matched shift closer to the next
  decade's right one for endings in 3 of 4 decades (onsets 1 of 4: the composition explanation of the safe-country
  tail is rejected).
- ensemble central offset (ENS_OFFSET "oc"): -0.05% (+0.1 / +1.1 / -1.6 / -0.8); safe-country wars -40%, coups -1.3%,
  share at war +0.8%, democracy -1.0%. No gain beyond the floor: not adopted (the diagnostic gain is real but the
  decision statistics do not move).
- shock split (SHOCK_SPLIT 0.7 era, 1.0 year): -0.17% (0.0 / +1.9 / +0.1 / -0.9); growth volatility -3.4%, years with
  falling world GDP +3.4% (diagnostic). Rejected. The CPU gate as written failed (between-decade share of yearly
  shock variance 15-32%, gate 40%); run anyway because decade means drive 64-82% of the 10-year spread.
- countries not yet existing treated as at peace again (ABSENT_MASK off): -0.14% (+1.4 / -0.7 / -1.4 / +0.2). The fix
  stays (it corrects the code); its effect is within noise.
Re-checks, each undoing an earlier adoption (positive = the adoption helped):
- growth regime scale back to 1.8: +3.1% (+3.8 / -3.6 / +4.8 / +4.5), growth spread +24.5%: confirmed.
- oil fix undone (OIL_INDEP off, pull 0.05): +1.2% (+0.6 / +6.7 / +1.0 / +0.7), oil +19.9%: confirmed.
- mood back to 0.20: +0.9% (+1.3 / +1.4 / +0.1 / +3.2), share at war +4.8%: likely right, below the floor.
- debt-split growth pools off: +0.6% (+0.2 / +0.9 / +1.1 / -0.1), growth spread +3.6%: below the floor.
- country frailty off: +0.2% (+3.2 / -1.2 / -2.1 / +4.6): not distinguishable from refit noise.
Nothing is reverted: the rule decides changes, and undoing on weak evidence would be the same mistake in reverse.
Outputs: outputs/adopt/noise_floor.json, outputs/adopt/nf_*_vs_nf_base.json, outputs/adopt/align_oracle.json.

## Second debate: what Seldon should become (2026-11-06)
Eight personas (historian, government foresight client, philosopher of forecasting, AI-transition economist; catastrophe
actuary, conflict-prevention ethicist, science journalist, climate scientist), three rounds, a 20-item ballot and two
independent judges. Unanimous: one source of truth for every published number, quarterly model releases, and fixing
the 12-month roll-up of the monthly forecasts. Near-unanimous: named-country numbers stay public with ranges (gating
vetoed 7 of 8); 2065 as a labelled credence beside base rates; a skill-versus-horizon curve; Forecasts / Explorations.
Roadmap in progress (this section is updated as items land).

Week one, done:
- 12-month probability: monthly.annual_probability multiplied 12 monthly forecasts as independent; a country's months
  are positively correlated, so it overstated the risk. A model fitted directly on "any 25-death month in the next 12"
  (monthly.fit_annual), issued each December 2017-2023 and scored on the next year (scripts/monthly_annual_test.py):
  mean forecast 13.9% against 14.0% observed (product 16.6%), calibration error 0.009 (0.026), log loss 0.103 (0.118),
  Brier 0.0302 (0.0309); on countries without fighting in the past year the product keeps a Brier edge (0.0195
  against 0.0205: it ranks them a little better while overstating their level). The ledger summary and new ledgers
  use the direct model: Benin 0.76 -> 0.39, Kenya 0.58 -> 0.22, Burundi 0.35.
- scripts/number_parity.py checks 17 headline claims in outputs/narrative.md (inlined in the report) and the
  methodology headline against outputs/forecast.json and outputs/country_forecasts.csv (--fix writes the current
  values). It found 12 stale numbers from 1 November (great-power war 49% -> 45%, $36k -> $34k, nuclear use 14% -> 13%,
  majors 26% -> 21% ...), all fixed.
- "War outcomes rest on the model more than on assumptions" withdrawn: the 4-7% share is R^2 on a yes-or-no outcome
  per world; the same assumptions move great-power war by up to 24 points.
- Chronicles removed from the report.
- Release policy (docs/adopt_rule.md): model versions quarterly; data refreshes monthly; fixes between releases as
  labelled patch versions that change no model setting.

Open item (from the chronicles): the simulator's own event logs (outputs/chronicles_raw.json) contain new civil wars
in Canada (2037), France, Denmark and Iceland, and a Costa Rica vs Russia interstate war, in single worlds. These are
the safe-country tail (Limits) and the pair model's tail for distant, unrelated pairs; to be checked against a
non-regression guardrail on the 20 safest countries (debate judges) and the cross-era war test.
