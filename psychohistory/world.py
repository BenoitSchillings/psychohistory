"""World-system layer for the simulator: interstate wars between country pairs (with alliance
cascades), food and energy dynamics, pandemics, and an AI scenario in which intellectual work
becomes nearly free.

Learned components: the one-year interstate hazard (psychohistory.interstate) and how war
involvement changes growth and democracy (transition equations). Everything else is an explicit,
switchable assumption (see docs/world_system.md); the report shows results with and without it.
"""
from __future__ import annotations

import functools

from dataclasses import dataclass, field

import numpy as np
import torch

from .interstate import FEATS_1Y as STATE_FEATS, FEATS_ON as DYAD_FEATS


# ---------------------------------------------------------------------------------------------
# interstate calibration in the large (scripts/interstate_calib_test.py): "drift" = previous decade's observed/predicted
# ratio drifting to the long-run ratio (10-year half-life); "long" = long-run ratio only; "none" = the model as fitted
DYAD_CALIB = "none"
PAIR_NOISE_CORR = 1.0   # 0 = no convexity correction of the structural pair noise (before 2026-10-28)

@dataclass
class AIScenario:
    """Near-free cognition. `arrival`: year it is available at the frontier (None = no AI shock).
    Magnitudes are assumptions, chosen to be large but physically bounded; see docs."""
    name: str = "No AI shock"
    arrival: int | None = None
    adoption_years: float = 8.0       # frontier: 10% -> 90% adoption
    lag_years: float = 12.0           # extra delay for the least-ready country
    growth_dividend: float = 0.03     # extra growth per person at full adoption, frontier, energy-unconstrained
    displacement_unemp: float = 6.0   # unemployment points added at the steepest part of adoption (fades as labour reallocates)
    top1_rise: float = 8.0            # points of top-1% income share at full adoption (halved by strong fiscal capacity)
    polarization_rise: float = 1.0    # V-Dem polarization units in democracies at full adoption
    # who captures AI rents (assumption, 2026-10-31): the share kept by the firms that own the frontier models, chips
    # and compute, and the yearly rate at which states take it back (tax, antitrust, nationalization), per year of full
    # adoption. Capture scales the top-1% rise, the frontier hubs' extra rent (the firms' home states gain) and half of
    # the polarization rise; at the mean capture (0.55, no clawback) every channel is as before
    firm_capture: float = 0.55
    clawback: float = 0.04
    surveillance: float = 0.4         # log-odds cut in coup / rebellion hazards in autocracies at full adoption
    autocracy_drift: float = 0.004    # yearly democracy-index erosion in non-democracies at full adoption
    ladder_penalty: float = 0.012     # growth lost by middle-income export economies at the steepest part of adoption
    capacity_gain: float = 0.01       # yearly gain toward 1 in rule of law / impartial administration at full adoption
    health_gain: float = 0.03         # extra yearly fall in log child mortality at full adoption
    yield_gain: float = 0.01          # extra yearly cereal-yield growth at full adoption
    clean_energy_gain: float = 0.02   # extra yearly growth of energy per person (cheap clean energy) at full frontier adoption
    bio_risk: float = 2.0             # pandemic probability multiplier at full frontier adoption (misuse)
    bio_response: float = 0.5         # pandemic damage multiplier at full frontier adoption (faster countermeasures)
    finance_mult: float = 1.3         # financial-crisis probability multiplier at full adoption
    military_mult: float = 2.0        # (superseded by ai_war_beta / nuc_beta, kept for reference)
    # --- added after the independent review (2026-10-24); assumptions unless noted, sampled per world (PARAM_RANGES)
    arrival_sd: float = 1.5           # per-world jitter of the arrival year
    speed_mult: float = 1.0           # per-world multiplier on adoption_years
    open_lag: float = 2.0             # years by which open-weight/API diffusion trails the frontier hubs
    export_lag: float = 1.5           # extra years for countries under US-led export controls (CHN, RUS, IRN, PRK)
    hub_rent: float = 0.25            # extra share of the dividend captured by frontier hubs (and lost by others)
    chip_pause: float = 0.75          # share of a year of frontier progress lost while Taiwan or Korea is a war site
    chip_gdp: float = 0.035           # world output hit (x cognitive share) in such a year (analogue: 1973 embargo)
    dc_share: float = 0.05            # energy diverted to data centres at the steepest part of adoption
    robot_lag: float = 5.5            # years by which general-purpose robotics follows near-free cognition (if not earlier)
    robot_aging: float = 0.5          # share of a shrinking workforce's growth drag that robots offset at full adoption
    # general-purpose robotics on its own timeline (added 2026-10-26; assumptions unless noted): each world draws the
    # year robots able to do most manual work become available (ROBOT_CDF), correlated with the AI arrival year
    # (Gaussian copula, robot_corr); robotics comes at the latest robot_lag years after near-free cognition
    robot_cdf: tuple | None = None    # None -> ROBOT_CDF
    robot_corr: float = 0.7
    robot_years: float = 14.0         # frontier deployment 10% -> 90% (physical capital turns over slower than software)
    phys_dividend: float = 0.025      # extra growth per person at full robot adoption x non-cognitive share of output
    # military autonomy (drones, autonomous weapons), following whichever comes first of near-free cognition and
    # general-purpose robotics, on the software adoption clock. Effects in log-odds per unit of autonomy:
    drone_onset: float = 0.1          # interstate onset (cheaper force vs deterrence by denial; sign uncertain)
    drone_persist: float = 0.4        # continuation of war-level interstate conflicts (defence dominance, Ukraine 2022-25);
                                      # applied once, to the conflict continuing (its intensity keeps the estimated rates)
    drone_reach: float = 0.1          # share of a war site's loss reaching an expeditionary power from a nearby or major
                                      # enemy (long-range strikes); output loss only, no refugees or trade spillover
    drone_civil: float = 0.1          # cut in civil-conflict termination (rebels with cheap strike drones: Myanmar, Sudan;
                                      # negative = states' drones end wars faster: Ethiopia 2021, Karabakh 2020)
    dem_erosion: float = 0.003        # yearly democracy erosion in democracies at the steepest adoption, scaled by elite overproduction
    ai_war_beta: float = 1.0          # log-odds of rival onset per unit gap in AI adoption between the two
    nuc_beta: float = 0.5             # log-odds added for nuclear rivals at the steepest part (second-strike erosion)
    eng_pandemic: float = 0.006       # yearly probability of an engineered pandemic at full frontier adoption
    natural_cut: float = 0.4          # cut in natural pandemic frequency at full adoption (AI surveillance)
    # uncertain arrival (the headline forecast): each world draws its own arrival year from this CDF, given as
    # (year, P(arrival <= year)) knots, linear in between; worlds drawing above the last knot get no AI shock by 2065.
    # When set, it replaces `arrival` + `arrival_sd`.
    arrival_cdf: tuple | None = None


# Arrival-year distribution for near-free cognition (assumption; revised 2026-10-28, sources in
# outputs/reviews/assumption_evidence.md): an equal mix of
#  - forecasters: Metaculus "first general AI system devised, tested and publicly known" (community median about
#    2031-33 in 2026, lower quartile about 2028; read from secondary write-ups, the site refused automated access);
#  - researchers: AI Impacts 2023 survey (Grace et al. 2024), high-level machine intelligence 50% by 2047, conditioned
#    on nothing having arrived by late 2026 (no later wave of the survey exists);
#  - FRI LEAP wave 8 (April-May 2026): "AGI" = beats the 90th-percentile professional on 90% of non-physical tasks at
#    no more than 5x human cost, the closest definition to near-free cognition; 80% by 2100, median year 2050 if it
#    comes (quartiles 2039 / 2065), superforecasters 2047.
# The 2025 version gave 17.5% by mid-2027, a figure from the 2023 survey that no 2026 source supports. Note that in
# FRI's 2022 tournament both experts and superforecasters underestimated AI progress.
# Mix: 5% by 2027, 18% by 2030, 24% by 2032, 33% by 2035, 47% by 2041, 56% by 2047, 64% by 2055, 72% by 2065;
# 28% no shock before 2065. (Previous: 17.5 / - / 37.5 / - / 57.5 / 66 / - / 77.5%.)
ARRIVAL_CDF = ((2026.75, 0.0), (2027.5, 0.05), (2030.5, 0.18), (2032.5, 0.24), (2035.5, 0.33), (2041.5, 0.47), (2047.5, 0.56),
               (2055.5, 0.64), (2065.5, 0.72))

# Arrival-year distribution for general-purpose robotics (robots technically able to do most manual work; cost
# parity comes with deployment, robot_years; assumption, 2026-10-26): an equal mix of
#  - forecasters: Metaculus "reliable general household robot" (community median about 2032 in 2025-26; quartiles
#    taken as about 2029 and 2042, 12% after 2065);
#  - researchers: AI Impacts 2023 survey, "install the electrical wiring in a new home" median about 2040 (17 years
#    from the 2023 survey), with the spread of the survey's other milestones (about 10% by 2030, 28% after 2065;
#    full automation of all occupations is much later, median 2116, but that is deployment, not capability).
# Mix: 5% by 2027, 20% by 2030, 32% by 2032, 56% by 2040, 70% by 2050, 80% by 2065; 20% none before 2065.
# Current robot use (IFR World Robotics 2025, robots per 10,000 manufacturing workers in 2024: Korea 1,220,
# Singapore 818, Germany 449, Japan 446, US 307, Taiwan 302; world 132) is only published for a dozen countries;
# deployment across countries follows the AI readiness index, which ranks those countries near the top.
# Revised 2026-10-28: slightly later (56% by 2040 exceeded the AI Impacts all-tasks forecast for that date, which
# includes physical work; no survey of roboticists exists): 3% by 2027, 14% by 2030, 25% by 2032, 35% by 2035, 48% by
# 2040, 63% by 2050, 76% by 2065. (Previous: 5 / 20 / 32 / - / 56 / 70 / 80%.)
ROBOT_CDF = ((2026.75, 0.0), (2027.5, 0.03), (2030.5, 0.14), (2032.5, 0.25), (2035.5, 0.35), (2040.5, 0.48), (2050.5, 0.63),
             (2065.5, 0.76))


AI_SCENARIOS = {
    "none": AIScenario(),
    "2030": AIScenario("Near-free cognition from 2030", 2030),
    "2027": AIScenario("Near-free cognition from 2027 (fast)", 2027, adoption_years=5),
    "2035": AIScenario("Near-free cognition from 2035 (slow)", 2035, adoption_years=12),
    "uncertain": AIScenario("Near-free cognition, arrival uncertain", 2030, arrival_cdf=ARRIVAL_CDF),
}


@functools.lru_cache(maxsize=1)
def _observed_oil():
    from .data import observed_oil
    return observed_oil()


@dataclass
class WorldConfig:
    ai: AIScenario = field(default_factory=AIScenario)
    interstate: bool = True
    cascade_q: float = 0.1            # P(a defence-pact ally joins a war-level conflict), calibrated on 1995 and 2004 hindcasts (0.5 = pre-1945 alliance behaviour)
    dem_peace_factor: float = 0.15    # see interstate_step
    anchor_decay: float = 1.0         # per-pair ensemble (logit+GBM) anchor kept for the whole run; with 0.93 (half-life
                                      # 10 years) allied democracies started fighting each other, which never happened 1946-2024
    war_persist: float = 0.71         # P(war level | war level last year, still active), UCDP 1946-2024
    war_escalate: float = 0.073       # P(war level | minor last year, still active)
    war_share: float = 0.35           # share of new interstate conflicts that reach war level (>= 1000 deaths)
    # between two nuclear-armed states, war level is rarer: of 5 conflicts since 1946 (China-Russia 1969, India-
    # Pakistan 1999 Kargil, 2013 and 2025, China-India 2020) only Kargil reached it, against 47% of 81 other new
    # conflicts; shrunk toward the overall rate (beta prior worth 2 episodes): 0.27 / 0.45 = 0.6. ICB v16 crisis dyads
    # (scripts/nuclear_crisis_test.py) add more evidence: 4 of 39 both-nuclear crises reached full-scale war (all four
    # wars fought by others in the crisis, not between the pair) vs 40% of 443 non-nuclear ones, ratio 0.27 (90% 0.11-0.50);
    # UCDP conflict-years: 0 of 17 at war level (it codes Kargil as minor). Central value lowered to 0.4
    nuclear_war_factor: float = 0.4
    food: bool = True
    yield_per_degC: float = -0.05     # cereal yield change per degC of local warming (Zhao et al. 2017, global mean of crops)
    # water (psychohistory/water.py, docs/world_system.md "Water and glaciers"): glacier runoff (a hump around each
    # basin's peak-water year) and precipitation change the water each country gets; where water is scarce this
    # moves cereal yields, and pairs sharing a basin whose worse-off member loses water, or locked in a live water
    # dispute, get extra conflict risk. Tested first (2026-10-26): shared rivers, water-event history and scarcity add
    # nothing to the pair model's backtests, and drought x farming x ethnic exclusion nothing to civil conflict
    # forecasts, so every effect here is an assumption whose range includes zero (WATER_RANGES)
    water: bool = True
    # cheap space (scenario, off by default; 2026-10-31): launch costs fall 10-100x and constellations, space-based sensors
    # and interceptors, satellite internet and orbital compute deploy from 2028 to 2040. Per-world assumptions
    # (SPACE_RANGES): space_war shifts war onset and nuclear use for the United States against a nuclear-armed rival
    # (negative: dominance deters; positive: eroding second-strike confidence and anti-satellite races destabilise);
    # sat_open removes that share of AI surveillance's protection of autocracies and of the AI drift to autocracy
    # (censorship-resistant connectivity); orbit_dc removes that share of data centres' energy draw on Earth
    # deaths from heat and cold as the climate warms (2026-11-01; Carleton et al. 2022, QJE): yearly deaths per person =
    # HEAT_MORT_BETA x heat_mort x local warming since the origin x a heat/cold factor from the country's baseline
    # temperature (negative where milder winters save lives) x an income factor (richer countries adapt); counted in
    # excess deaths and taken off the population, in worlds with climate damage on
    heat_mortality: bool = True
    # AI loss of control (2026-11-02; assumption): once near-free cognition arrives, a yearly chance (ai_cat_rate x
    # frontier progress; log-uniform per world, about 1% to 13% by 2100: superforecasters' 2% to domain experts' 12% for
    # an AI catastrophe killing a tenth of humanity by 2100, FRI XPT 2022) of a catastrophe that kills ai_cat_deaths of
    # the world's population and costs ai_cat_loss of output, half of it recovered over the following decade.
    # Extinction is outside what the simulator can represent and is stated, not simulated
    ai_catastrophe: bool = True       # (in the headline since 2026-11-02)
    ai_cat_rate: float = 0.0012
    ai_cat_deaths: float = 0.04
    ai_cat_loss: float = 0.2
    heat_mort: float = 1.0
    # the dollar's decline (scenario, off by default; 2026-11-02): how far its reserve role falls by 2045 (dollar_shift,
    # ramped in from 2026), sanctions losing up to half their bite (much of it runs through dollar clearing) and US
    # growth losing up to usd_cost a year at full decline (higher borrowing costs, the end of the "exorbitant
    # privilege"); assumptions, with no history to test them
    dollar_decline: bool = False
    dollar_shift: float = 0.5
    usd_cost: float = 0.0015
    cheap_space: bool = False
    # technology breakthroughs (tech.py; in the headline since 2026-10-31, for forecasts from 2025): each world draws
    # which arrive and when, and their conditional effects; tech_force = {name: year or None} fixes an arrival in
    # every world (paired comparisons)
    tech: bool = True
    tech_force: dict | None = None
    space_war: float = 0.1
    sat_open: float = 0.25
    orbit_dc: float = 0.25
    water_dispute: float = 0.25       # onset log-odds for pairs in a live water dispute (water.WATER_DISPUTES)
    water_scarcity_k: float = 2.0     # onset log-odds per unit water lost by the worse-off member of a basin pair
    yield_water: float = 0.4          # cereal yield change per unit change in water supply, x how binding water is
    glacier_tau: float = 45.0         # width (sd, years) of the glacier-runoff hump around peak water
    peak_shift: float = 0.0           # shift of every basin's peak-water year (climate-model spread)
    glacier_flow_mult: float = 1.0    # multiplier on water.GLACIER_FLOW
    runoff_elast_pr: float = 1.5      # river-flow change per unit precipitation change (Sankarasubramanian et al. 2001: ~1-2.5)
    # alliances (2026-10-26): allies of allies join at cascade_q x cascade_hop2; security patrons without a treaty
    # (interstate.QUASI_ALLIES) join a new war against their client with their own chance (assumptions, EXTRA_RANGES).
    # Tested and not adopted: alliances weakening with the regime gap between partners (COW 1816-2007 x ATOP, 902
    # ally-war opportunities: allies with larger democracy gaps joined MORE often, +3.4 +/- 1.3 log-odds per unit gap,
    # e.g. US defence of South Korea and South Vietnam)
    cascade_hop2: float = 0.5
    quasi_USA_TWN: float = 0.5
    quasi_USA_ISR: float = 0.3
    quasi_RUS_IRN: float = 0.05
    quasi_CHN_PAK: float = 0.05
    # missing factors added 2026-10-26 (psychohistory/risks.py; ranges in EXTRA_RANGES)
    intl_beta: float = -0.42          # civil-war termination log-odds when internationalized (UCDP 1946-2023, se 0.20)
    prolif_mult: float = 1.0          # x risks.LATENT yearly chance of starting a weapons program
    prolif_cascade: float = 3.5       # x that chance once a rival has acquired weapons
    iran_pursuit0: float = 0.35       # share of worlds in which Iran is pursuing a weapon at the origin
    breakout_years: float = 4.0       # years from the start of a program to a weapon
    prevent_beta: float = 0.75        # onset log-odds between a state pursuing a weapon and a nuclear-armed rival or patron
    busy_beta: float = 0.4            # onset log-odds lost when either side is already fighting another interstate conflict
                                      # (force budget; scripts/force_budget_test.py: -0.78 in 1946-94, -0.54 in 1995-2023, not
                                      # significant either time; range 0 to 0.8 in EXTRA_RANGES)
    nuc_use_both: float = 0.01        # yearly chance of nuclear use in a war-level conflict between nuclear-armed states
    nuc_use_one: float = 0.003        # ... between a nuclear-armed state and one without
    nuc_escalate: float = 0.2         # chance a use between two major powers becomes a large exchange
    nuc_gdp_large: float = 0.15       # world output lost (level) after a large exchange; belligerents lose 0.5
    sanction_loss: float = 0.02       # yearly output loss x Western trade share for a non-Western state at war with the West
    mineral_elast: float = 0.25       # extra output x mineral rents (share of GDP) per unit log growth of the clean share
    china_pause: float = 0.6          # share of a year of clean-energy and robotics progress lost while China fights the West
    civ_mult: float = 1.5             # civilian and indirect war deaths per battle death (for population)
    pandemic_mort: float = 0.002      # population share killed by a natural pandemic
    slr_mult: float = 1.0             # x the AR6 SSP2-4.5 median sea-level path
    slr_cost: float = 0.05            # output level lost per 0.5 m of sea-level rise x share of people below 5 m
    storm_trend: float = 0.15         # rise in damaging-storm frequency per degC of warming (share of the base rate)
    storm_loss: float = 0.015         # growth lost in a damaging-storm year in exposed (tropical, coastal) countries
    solar_p: float = 0.008            # yearly chance of a Carrington-class geomagnetic storm
    solar_loss: float = 0.04          # output lost that year x grid exposure (AI readiness)
    volc_p: float = 0.008             # yearly chance of a VEI 6+ eruption (Pinatubo-class or larger)
    volc_yield: float = 0.06          # world cereal-yield loss for two years after one
    cyber_loss: float = 0.015         # yearly output lost by belligerents in a great-power war (infrastructure attacks)
    # climate (2026-10-29, docs/world_system.md "Climate risk"; ranges in CLIMATE_RANGES). The damage function is a
    # model choice the hindcasts cannot settle (too little warming in-sample), so each world draws one of three
    # published families; the rest are tails and channels with their evidence noted at the range
    dmg_family: float = 1.0           # 0 level effect (BHM coefficients on the level), 1 BHM growth effect, 2 global temperature (Bilal-Kaenzig)
    clim_tail_df: float = 4.0         # Student-t degrees of freedom of the per-world climate-response draw (0 = normal)
    amoc_p: float = 0.05              # chance that an AMOC collapse's effects begin before 2065
    amoc_eu_loss: float = 0.03        # north-west European output level lost once its effects are complete
    amoc_eu_yield: float = 0.2        # north-west European cereal-yield loss
    amoc_rain: float = 0.2            # rainfall loss in the Sahel and West Africa (half in South Asia)
    slr_ice_p: float = 0.1            # chance of fast ice-sheet loss (sea-level path x slr_ice_mult)
    slr_ice_mult: float = 2.0
    harvest_amp: float = 0.15         # bad harvests deepen by this share per degC of local warming
    dry_area_elast: float = 0.71      # dryland cereal area per unit log change in decade-mean rainfall
    clim_conflict: float = 0.15       # civil-war onset log-odds per degC of local warming since the origin, x farm share / 0.3
    clim_migr: float = 0.2            # emigration as a share of the climate harvest loss x farm share (stock)
    ev_war_feedback: bool = False     # simulated conflicts raise the pair's event hostility (interstate.sim_inputs ev_war)
    demog_uncertainty: bool = True    # draw fertility within the UN WPP 2024 probabilistic intervals (data/wpp_prob.npz)
    ai_exposure_data: bool = True     # cognitive share of output from occupation data (data/ai_exposure.npz)
    trade_gravity: bool = False       # every 5 years, trade shares move with partners' simulated GDP (World.update_trade)
    # financial contagion through trade (scripts/contagion_test.py, BIS + Laeven-Valencia 1985-2024, year effects):
    # a banking-crisis start is likelier the larger the share of a country's trade with partners in crisis that
    # year, logit coefficient 4.26 (s.e. 0.56); exposure to crisis-hit *bank lenders* added nothing (-0.79, s.e. 0.60)
    fin_contagion: float = 4.26
    # workforce: growth per person moves with the change in the log working-age share (15-64 / all, UN WPP), 5-year
    # panel estimate 1.22 (s.e. 0.17) with year effects and the growth controls (scripts/workforce_test.py)
    workforce_beta: float = 1.22
    # ocean food (ocean.py): wild catch, log change per degC of warming by latitude band (tropics, mid, high) and a
    # global overfishing / management trend per year; moves calories and income by each country's dependence
    fish_trop: float = -0.08
    fish_mid: float = -0.03
    fish_high: float = 0.04
    wild_trend: float = -0.003
    # innovation (innovation.py): catch-up from science capacity (estimate) and a fading AI dividend (assumption)
    articles_beta: float = 0.0023
    dividend_fade: float = 0.02
    # peacekeeping, aid and commodity prices (intervention.py)
    pko_beta: float = 0.315           # war-ending log-odds per log(1 + thousand peacekeepers) (estimate)
    pko_cut: float = 0.25             # share of 2025 deployments cut (UN liquidity crisis)
    pko_decay: float = 0.05           # yearly wind-down of the remaining missions
    aid_mult: float = 0.5             # output lost per unit of the 2025 USAID cut (% of GNI), 2026 only
    commod_coup: float = -2.8         # coup log-odds per unit of export-weighted commodity price change (estimate)
    # structure.py estimates: rebel fragmentation (war ending), short-term debt (banking crises), peer coups
    tariff_pass: float = 0.75         # partner output lost per unit of (exports to US / GDP) x tariff rise, 2026
    tariff_us: float = 0.005          # US output lost to its own 2025 tariffs, 2026
    frag_beta: float = -1.93
    st_debt_beta: float = 0.596
    peer_coup: float = 2.645
    pandemics: bool = True
    pandemic_p: float = 0.02          # COVID-scale pandemics per year (Marani et al. 2021 order of magnitude)
    pandemic_gdp: tuple = (-0.06, -0.02)
    # economic cost of war on a country's own territory, estimated on 1951-2023 (country and year fixed effects,
    # 66 war-site years): growth -9.0% (+/- 2.9) in each year of a war-level interstate conflict fought at home or
    # next door, +4.5% (+/- 2.3) two years later; no measurable cost for expeditionary participants. Comparable to
    # Federle et al. (2024, "The Price of War"). The spillover to other countries is an assumption.
    war_damage: bool = True
    war_site_loss: float = 0.09       # growth loss in each war year at a war site
    war_rebound: float = 0.045        # partial rebound two years later
    war_share_norm: float = 0.0085    # mean share of world GDP at war sites, 1951-2023 (already in replayed history)
    war_spillover: float = 0.5        # others lose this x (war-site share of world GDP) x war_site_loss (assumption)
    # not war sites when fighting a smaller power, even a neighbour (Russia's economy grew in 2023-24 while Ukraine
    # was the war site; China would not be the site of a Taiwan war). Two expeditionary powers fighting each other:
    # half a war site each (assumption; the 1951-2023 estimate has no such case)
    expeditionary: tuple = ("USA", "GBR", "FRA", "RUS", "AUS", "CAN", "CHN")
    ally_reach_km: float = 2500.0     # allies join only if major powers or within this distance of the enemy
    # treaty scope (2026-11-05): a permanent Security Council member joins an ally's war at the full cascade_q only if
    # the enemy is near the ally it defends or near itself (NATO's Article 6 area, the US Pacific pacts); outside that
    # area at this share of it (sampled 0-0.5; 1.0 = the earlier rule, in which France could join any US war at the
    # full rate, so a US-China war over Taiwan drew France in as often as a Russian attack on Estonia)
    ally_out_scope: float = 0.25
    explosive_p: float = 0.0          # share of worlds in the explosive AI-growth regime (exploration only; 0 = headline)
    param_block: int = 1               # worlds per block sharing every sampled assumption (1 = each world its own)
    major_era_mult: float = 1.0       # onset odds of major-power pairs x this (1.3 = the pre-1945 era; scenario only)
    # clean-energy transition: world wind+solar share of primary energy on an S-curve fitted to 2005-2024
    # (OWID/Energy Institute; growth rate 0.155 +/- 0.03 a year), with an unidentified ceiling drawn per world.
    # Clean energy displaces fossil fuels pro rata; oil demand follows, the oil price moves oil_eta x the demand
    # change (assumption), and exporters lose volume as well as price. AI speeds the transition.
    transition: bool = True
    trans_ceiling: tuple = (0.4, 0.9)
    trans_rate: tuple = (0.155, 0.03)
    trans_share_0: float = 0.064      # wind+solar share of world primary energy, 2024
    fossil_share_0: float = 0.813     # fossil share, 2024
    oil_eta: float = 0.5
    ai_transition_boost: float = 0.5  # transition speed x (1 + boost x AI adoption)
    # structural uncertainty: each world draws its own value of the assumed or weakly estimated parameters
    # (ranges below); the forecast then reports how much of the spread in each outcome each one explains
    sample_params: bool = False
    # AI investment boom: chance the 2026-2030 capex boom ends in a bust (railway mania 1847, dot-com 2001 analogues),
    # and its cost to frontier economies (x cognitive share, spread over two years). Assumptions.
    bubble_p: float = 0.4
    bubble_cost: float = 0.025
    fin_crisis_cost: tuple = (0.017, 0.012)   # growth lost in the year of a financial crisis and the next (1952-2024 FE estimate, 638 crisis-years)
    # forced interstate wars for crisis scenarios: ((iso3_a, iso3_b, year), ...)
    forced: tuple = ()
    peace: tuple = ()                 # (a, b, year): pairs held at peace that year (counterfactuals)
    # energy market (estimated on 1950-2024 annual real oil prices, see docs/world_system.md)
    energy: bool = True
    oil_war_jump: float = 0.18        # log real oil price jump in a year a major exporter enters a new interstate conflict
                                      # (OLS +0.27 +/- 0.08 incl. 1974; +0.18 without the embargo year)
    oil_reversion: float = 0.15       # yearly pull toward the 20-year average. 0.05 until 2026-11-03 (one-year OLS 0.05 +/- 0.05);
                                      # over longer spans the price returns faster (slope of the change on the gap to the
                                      # 20-year average: 3 years -0.19, 5 years -0.34, 10 years -1.0, 1965-2024). With
                                      # simulate.OIL_INDEP, 15-year hindcasts chose 0.15 on 1980-1995 (oil CRPS -43%) and
                                      # confirmed it on 2000-2009 (-32%; world GDP growth -2.8% / -2.1%)
    oil_beta: float = 0.068           # growth response to oil price changes per unit of production exposure (+/- 0.022)
    oil_importer: float = -0.014      # growth response of net importers to price rises (pop-weighted, 1966-2023; +/- 0.013)
    # observed oil years after the origin: ((year, change in log real price vs the previous year, sd of the
    # unobserved months), ...), from the World Bank monthly Pink Sheet (data.observed_oil)
    oil_observed: tuple = field(default_factory=lambda: _observed_oil())


# ---------------------------------------------------------------------------------------------
PARAM_RANGES = {
    # name: (kind, a, b)  uniform(a, b) or normal(mean a, sd b) clipped to the "clip" range
    "cascade_q": ("u", 0.05, 0.2),            # calibrated 0.1 on 1995-2024 hindcasts
    "nuclear_war_factor": ("u", 0.10, 0.55),   # 5 UCDP episodes + 39 ICB crisis dyads (0.27, 90% 0.11-0.50); 0.70 -> 0.55 2026-10-28
    "war_spillover": ("u", 0.0, 1.0),         # assumption
    "war_site_loss": ("n", 0.09, 0.029),      # estimate and standard error
    "oil_eta": ("u", 0.25, 1.0),              # assumption
    "oil_war_jump": ("n", 0.18, 0.08),        # estimate range (0.18 without 1974, 0.27 with)
    "pandemic_p": ("u", 0.01, 0.04),          # order of magnitude (Marani et al. 2021)
    "climate_damage": ("u", 0.5, 1.5),        # multiplier on the BHM damage function
    "growth_dividend": ("u", 0.01, 0.05),     # AI scenarios only
    "autocracy_drift": ("u", 0.0, 0.008),     # AI scenarios only
    "bubble_p": ("u", 0.2, 0.6),
    "bubble_cost": ("u", 0.01, 0.04),
    # AI scenarios only (see AIScenario)
    "arrival_sd": ("n", 0.0, 1.5), "speed_mult": ("u", 0.75, 1.5), "open_lag": ("u", 0.5, 4.0), "export_lag": ("u", 0.0, 3.0),
    "hub_rent": ("u", 0.0, 0.5), "chip_pause": ("u", 0.5, 1.0), "chip_gdp": ("u", 0.02, 0.05), "dc_share": ("u", 0.02, 0.08),
    "robot_lag": ("u", 3.0, 8.0), "robot_aging": ("u", 0.0, 1.0), "dem_erosion": ("u", 0.0, 0.006),
    "ai_war_beta": ("u", 0.0, 2.0), "nuc_beta": ("u", 0.0, 1.0), "eng_pandemic": ("u", 0.002, 0.01), "natural_cut": ("u", 0.2, 0.6),
    "robot_years": ("u", 8.0, 20.0), "phys_dividend": ("u", 0.005, 0.04),   # floor: industrial robots ~0.36 pp/yr (Graetz & Michaels 2018)
    "drone_onset": ("u", -0.3, 0.5), "drone_persist": ("u", 0.0, 0.8), "drone_reach": ("u", 0.0, 0.2), "drone_civil": ("u", -0.2, 0.4),
}
USD_RANGES = {"dollar_shift": ("u", 0.0, 1.0), "usd_cost": ("u", 0.0, 0.003)}   # dollar decline only
AICAT_RANGES = {"ai_cat_rate": ("logu", 0.0003, 0.004), "ai_cat_deaths": ("u", 0.01, 0.10), "ai_cat_loss": ("u", 0.10, 0.35)}
HEAT_RANGES = {"heat_mort": ("u", 0.25, 1.75)}      # spans Carleton et al.'s wide range (interquartile 6-101 per 100,000)
CAPTURE_RANGES = {"firm_capture": ("u", 0.2, 0.9), "clawback": ("u", 0.0, 0.1)}   # AI scenarios only (assumptions)
SPACE_RANGES = {"space_war": ("u", -0.5, 0.7), "sat_open": ("u", 0.0, 0.5), "orbit_dc": ("u", 0.0, 0.5)}   # cheap space only
SPACE_START, SPACE_FULL = 2028, 2040
# water assumptions: drawn from their own generator, so adding them left every earlier parameter's draws unchanged
WATER_RANGES = {
    "water_dispute": ("u", 0.0, 0.5), "water_scarcity_k": ("u", 0.0, 4.0), "yield_water": ("u", 0.1, 0.7),
    "glacier_tau": ("u", 30.0, 60.0), "glacier_flow_mult": ("u", 0.5, 1.5),
    "peak_shift": ("n", 0.0, 10.0),           # a shift, may be negative (the climate-model spread of peak years)
}
# further assumptions added 2026-10-26 (alliances, proliferation, nuclear use, sanctions, minerals, demography,
# coasts, tail catastrophes): drawn from a third generator, so the earlier draws are unchanged
EXTRA_RANGES = {
    "cascade_hop2": ("u", 0.0, 1.0), "busy_beta": ("u", 0.0, 0.8),
    "quasi_USA_TWN": ("u", 0.25, 0.75), "quasi_USA_ISR": ("u", 0.1, 0.5),
    "quasi_RUS_IRN": ("u", 0.0, 0.1), "quasi_CHN_PAK": ("u", 0.0, 0.1),
    "intl_beta": ("n", -0.42, 0.20),
    "prolif_mult": ("u", 0.25, 2.0), "prolif_cascade": ("u", 1.0, 6.0), "iran_pursuit0": ("u", 0.1, 0.6),
    "breakout_years": ("u", 2.0, 6.0), "prevent_beta": ("u", 0.0, 1.5),
    "nuc_use_both": ("u", 0.002, 0.03), "nuc_use_one": ("u", 0.001, 0.008), "nuc_escalate": ("u", 0.05, 0.4),
    "nuc_gdp_large": ("u", 0.05, 0.3),
    "sanction_loss": ("u", 0.0, 0.04), "mineral_elast": ("u", 0.0, 0.5), "china_pause": ("u", 0.2, 1.0),
    "civ_mult": ("u", 0.5, 3.0), "pandemic_mort": ("u", 0.0005, 0.005),
    "slr_mult": ("u", 0.7, 1.5), "slr_cost": ("u", 0.0, 0.1), "storm_trend": ("u", 0.0, 0.3), "storm_loss": ("u", 0.005, 0.03),
    "solar_p": ("u", 0.002, 0.02), "solar_loss": ("u", 0.01, 0.08), "volc_p": ("u", 0.003, 0.015),
    "volc_yield": ("u", 0.03, 0.12), "cyber_loss": ("u", 0.0, 0.03),
    "fin_contagion": ("n", 4.26, 0.56),
    "workforce_beta": ("n", 1.22, 0.17),
    "fish_trop": ("n", -0.08, 0.04), "fish_mid": ("n", -0.03, 0.03), "fish_high": ("n", 0.04, 0.05),
    "wild_trend": ("u", -0.01, 0.003),
    "articles_beta": ("n", 0.0023, 0.0011), "dividend_fade": ("u", 0.0, 0.04),
    "pko_beta": ("n", 0.315, 0.152), "pko_cut": ("u", 0.1, 0.5), "pko_decay": ("u", 0.0, 0.1),
    "aid_mult": ("u", 0.0, 1.0), "commod_coup": ("n", -2.8, 1.0),
    "tariff_pass": ("u", 0.0, 1.5), "tariff_us": ("u", 0.0, 0.01),
    "frag_beta": ("n", -1.93, 0.41), "st_debt_beta": ("n", 0.596, 0.112), "peer_coup": ("n", 2.645, 0.86),
    # (appended last so every earlier draw is unchanged) a global power joining an ally's war outside the treaty's
    # area, as a share of cascade_q: an assumption (Korea 1950: Britain and France fought China under a UN mandate)
    "ally_out_scope": ("u", 0.0, 0.5),
}
# climate assumptions and estimates (2026-10-29): drawn from a fourth generator, so the earlier draws are unchanged
CLIMATE_RANGES = {
    "dmg_family": ("u", 0.0, 3.0),            # one damage family per world, equal weights (floor of the draw)
    "amoc_p": ("u", 0.0, 0.10),               # IPCC AR6: collapse before 2100 very unlikely (medium confidence); recent
                                              # studies 25-70% by 2100 for tipping, effects lag tipping by decades
    "amoc_eu_loss": ("u", 0.01, 0.05), "amoc_eu_yield": ("u", 0.1, 0.3), "amoc_rain": ("u", 0.1, 0.3),
    "slr_ice_u": ("u", 0.0, 1.0), "slr_ice_mult": ("u", 1.5, 3.0),   # AR6 low-likelihood, high-impact storyline
    "harvest_amp": ("u", 0.0, 0.3),           # 0: national 1962-2022 data (no clear amplification); 0.3 ~ crop models
                                              # (Tigchelaar et al. 2018: synchronised maize losses ~0 now, 7% at 2 degC)
    "dry_area_elast": ("n", 0.71, 0.29),      # estimate: 42 drylands, decade changes 1961-2022, errors by country
    "clim_conflict": ("u", 0.0, 0.3),         # 0: no forecast gain (scripts/climate_war_test.py); 0.3 ~ Burke et al.
                                              # 2009 for Africa, a quarter of this panel's own (untested) fit
    "clim_migr": ("u", 0.0, 0.4),             # assumption: World Bank Groundswell (2021) projects up to 216 m internal climate
                                              # migrants by 2050; a 10-20% cross-border share gives tens of millions, which the
                                              # upper end reaches by 2065 (u(0, 1) gave a 50 m median, above that)
}
GLOBAL_PER_LAND = 0.72   # global-mean warming per degC of population-weighted country warming (CMIP6 SSP2-4.5, 2016-25 -> 2061-70: 1.38 vs ~1.0 AR6 GSAT)
BK_PER_DEG = 0.35        # log output lost per degC of global warming since the origin, after a 5-year lag (Bilal & Kaenzig 2024:
                         # -19% by 2050 and -47% by 2100 for a linear +2 degC from 2024)
BK_LAG = 5.0
NW_EUROPE = ("GBR", "IRL", "FRA", "BEL", "NLD", "LUX", "DEU", "DNK", "NOR", "SWE", "FIN", "ISL")
WEST_AFRICA_SAHEL = ("SEN", "GMB", "MLI", "BFA", "NER", "TCD", "MRT", "NGA", "GHA", "CIV", "GIN", "SLE", "LBR", "TGO",
                     "BEN", "SDN", "CMR", "GNB")
SOUTH_ASIA = ("IND", "PAK", "BGD", "NPL", "LKA")
GP_SHARE = 0.05     # nuclear-armed and >= 5% of world output counts as a great power (India from 2025)
AI_ONLY = {"growth_dividend", "autocracy_drift", "arrival_sd", "speed_mult", "open_lag", "export_lag", "hub_rent", "chip_pause",
           "dc_share", "robot_lag", "robot_aging", "dem_erosion", "ai_war_beta", "nuc_beta", "eng_pandemic", "natural_cut",
           "robot_years", "phys_dividend", "drone_onset", "drone_persist", "drone_reach", "drone_civil", "dividend_fade"}
# frontier AI hubs (labs, compute, chips; assumption informed by Epoch AI / Stanford AI Index rankings) and export-controlled states
HUBS = {"USA": 1.0, "CHN": 0.8, "GBR": 0.5, "KOR": 0.5, "JPN": 0.5, "TWN": 0.5, "FRA": 0.4, "DEU": 0.4, "CAN": 0.4, "ISR": 0.4,
        "NLD": 0.4, "SGP": 0.4, "IND": 0.3, "ARE": 0.3, "SAU": 0.2}
EXPORT_CONTROLLED = ("CHN", "RUS", "IRN", "PRK")


def _contagion_table(Wt, p, n=4000, seed=5):
    """first-round scale s(beta) with E[crisis rate] = p for the two-round trade-contagion draw"""
    rng = np.random.default_rng(seed)
    C = Wt.shape[0]
    u1, u2 = rng.random((n, C)), rng.random((n, C))
    grid = np.linspace(0.0, 9.0, 19)
    out = []
    for beta in grid:
        lo, hi = 0.3, 1.0
        for _ in range(20):
            sc = (lo + hi) / 2
            p1 = p * sc
            c1 = u1 < p1
            p2 = np.clip(p1 * (np.exp(beta * (c1 @ Wt.T)) - 1) / (1 - p1), 0, 1)
            rate = (c1 | ((u2 < p2) & ~c1)).mean()
            lo, hi = (sc, hi) if rate < p else (lo, sc)
        out.append((lo + hi) / 2)
    return grid, np.array(out)


class World:
    """Holds the state of the world-system layer during one simulate() call."""

    def update_trade(self, lg, lpop):
        """gravity update of the trade network: each partner's share moves with its GDP against the origin (elasticity
        1, Head & Mayer 2014), from the median world (one network for all worlds); rows sum to 1"""
        gdp = torch.exp(lg + lpop).nan_to_num().median(0).values
        r = (gdp / self.trade_gdp0.clamp(min=1e-6)).clamp(0.05, 20)
        r = torch.where(self.trade_gdp0 > 0, r, torch.ones_like(r))
        w = self.trade_w0 * r[None, :]
        self.trade_w = w / w.sum(1, keepdim=True).clamp(min=1e-12)
        self.cont_grid, self.cont_scale = _contagion_table(self.trade_w.cpu().numpy(), 0.065)

    def contagion_scale(self):
        """[S, 1] (or scalar) first-round crisis scale for this world's contagion strength"""
        b = self.prm("fin_contagion")
        bb = b.detach().cpu().numpy().ravel() if torch.is_tensor(b) else np.array([float(b)])
        sc = np.interp(bb, self.cont_grid, self.cont_scale)
        return torch.as_tensor(sc, dtype=torch.float32, device=self.trade_w.device)[:, None] if torch.is_tensor(b) else float(sc[0])

    def space_ramp(self, year):
        """deployment of cheap space, 0 before SPACE_START to 1 from SPACE_FULL (0 without the scenario)"""
        if self.tech is not None and "cheap_launch" in self.tech.arrival:
            return self.tech.adoption("cheap_launch", year)            # [S, 1]: when cheap launch arrives in each world
        if not self.cfg.cheap_space:
            return 0.0
        return float(np.clip((year - SPACE_START) / (SPACE_FULL - SPACE_START), 0.0, 1.0))

    def dollar(self, year):
        """[S, 1] (or 0) how far the dollar's reserve role has fallen: dollar_shift, ramped in 2026-2045"""
        if not self.cfg.dollar_decline:
            return 0.0
        return self.prm("dollar_shift") * float(np.clip((year - 2026) / 19, 0, 1))

    def capture(self):
        """firms' share of AI rents relative to the mean assumption (1 = as before the rent-capture split), falling as
        states claw rents back with every year of full adoption"""
        return self.prm("firm_capture") * torch.exp(-self.prm("clawback") * self.cum_A) / 0.55

    def prm(self, name):
        """per-world draw [S, 1] when sampling, else the configured scalar"""
        if name in self.params:
            return self.params[name]
        if name == "arrival_sd":
            return 0.0                       # a spread across worlds, not a shift: no jitter without sampling
        if hasattr(self.cfg.ai, name):
            return getattr(self.cfg.ai, name)
        return getattr(self.cfg, name)

    def _draw_params(self, S, device, seed):
        self.params = {}
        g = torch.Generator(device=device).manual_seed(seed + 99)
        defaults = WorldConfig()
        for k, (kind, a, b) in (PARAM_RANGES.items() if self.cfg.sample_params else ()):
            # every parameter consumes its random draw whether or not it is used, so scenarios run with the same
            # seed (none / 2030 / 1914 alliances ...) give each world the same values of the parameters they share
            r_ = torch.rand(S, 1, device=device, generator=g) if kind == "u" else torch.randn(S, 1, device=device, generator=g)
            if k in AI_ONLY and not self.cfg.ai.arrival:
                continue
            if hasattr(defaults, k) and getattr(self.cfg, k) != getattr(defaults, k):
                continue                     # a scenario that sets this parameter explicitly keeps it fixed
            if hasattr(defaults.ai, k) and getattr(self.cfg.ai, k) != getattr(defaults.ai, k):
                continue                     # (AI parameters live on the AI scenario)
            if kind == "u":
                v = a + (b - a) * r_
            elif k == "arrival_sd" and self.cfg.ai.arrival_cdf:
                v = torch.zeros(S, 1, device=device)                             # the arrival year is drawn below
            elif k == "arrival_sd":
                v = a + b * r_                                                   # a jitter, may be negative
            else:
                v = (a + b * r_).clamp(min=0)
            self.params[k] = v
        gw = torch.Generator(device=device).manual_seed(seed + 211)
        for k, (kind, a, b) in (WATER_RANGES.items() if self.cfg.sample_params else ()):
            r_ = torch.rand(S, 1, device=device, generator=gw) if kind == "u" else torch.randn(S, 1, device=device, generator=gw)
            if not self.cfg.water or getattr(self.cfg, k) != getattr(defaults, k):
                continue
            self.params[k] = a + (b - a) * r_ if kind == "u" else a + b * r_
        ge = torch.Generator(device=device).manual_seed(seed + 307)
        for k, (kind, a, b) in (EXTRA_RANGES.items() if self.cfg.sample_params else ()):
            r_ = torch.rand(S, 1, device=device, generator=ge) if kind == "u" else torch.randn(S, 1, device=device, generator=ge)
            if getattr(self.cfg, k) != getattr(defaults, k):
                continue
            self.params[k] = a + (b - a) * r_ if kind == "u" else a + b * r_
        gu = torch.Generator(device=device).manual_seed(seed + 1009)      # dollar decline: its own generator
        for k, (kind, a, b) in (USD_RANGES.items() if self.cfg.sample_params else ()):
            r_ = torch.rand(S, 1, device=device, generator=gu)
            if self.cfg.dollar_decline and getattr(self.cfg, k) == getattr(defaults, k):
                self.params[k] = a + (b - a) * r_
        ga = torch.Generator(device=device).manual_seed(seed + 907)       # AI loss of control: its own generator
        for k, (kind, a, b) in (AICAT_RANGES.items() if self.cfg.sample_params else ()):
            r_ = torch.rand(S, 1, device=device, generator=ga)
            if self.cfg.ai_catastrophe and self.cfg.ai.arrival and getattr(self.cfg, k) == getattr(defaults, k):
                self.params[k] = torch.exp(np.log(a) + (np.log(b) - np.log(a)) * r_) if kind == "logu" else a + (b - a) * r_
        gh = torch.Generator(device=device).manual_seed(seed + 811)       # heat mortality: its own generator
        for k, (kind, a, b) in (HEAT_RANGES.items() if self.cfg.sample_params else ()):
            r_ = torch.rand(S, 1, device=device, generator=gh)
            if self.cfg.heat_mortality and getattr(self.cfg, k) == getattr(defaults, k):
                self.params[k] = a + (b - a) * r_
        gs = torch.Generator(device=device).manual_seed(seed + 601)       # cheap space: its own generator too
        for k, (kind, a, b) in (SPACE_RANGES.items() if self.cfg.sample_params else ()):
            r_ = torch.rand(S, 1, device=device, generator=gs)
            if (self.cfg.cheap_space or self.cfg.tech) and getattr(self.cfg, k) == getattr(defaults, k):
                self.params[k] = a + (b - a) * r_
        # rent capture: its own generator (seed + 521; + 503 was already the climate-response draw in simulate, which
        # tied every world's climate sensitivity to its rent capture; review 2026-11-02)
        gr = torch.Generator(device=device).manual_seed(seed + 521)
        for k, (kind, a, b) in (CAPTURE_RANGES.items() if self.cfg.sample_params else ()):
            r_ = torch.rand(S, 1, device=device, generator=gr)
            if not self.cfg.ai.arrival or getattr(self.cfg.ai, k) != getattr(defaults.ai, k):
                continue
            self.params[k] = a + (b - a) * r_
        gc = torch.Generator(device=device).manual_seed(seed + 401)
        for k, (kind, a, b) in (CLIMATE_RANGES.items() if self.cfg.sample_params else ()):
            r_ = torch.rand(S, 1, device=device, generator=gc) if kind == "u" else torch.randn(S, 1, device=device, generator=gc)
            if hasattr(defaults, k) and getattr(self.cfg, k) != getattr(defaults, k):
                continue
            self.params[k] = a + (b - a) * r_ if kind == "u" else a + b * r_
        if "dmg_family" in self.params:
            self.params["dmg_family"] = self.params["dmg_family"].floor().clamp(0, 2)
        if "slr_ice_u" in self.params:
            ice = (self.params.pop("slr_ice_u") < self.cfg.slr_ice_p).float()
            self.params["slr_ice"] = ice
            if "slr_mult" in self.params:
                self.params["slr_mult"] = self.params["slr_mult"] * (1 + ice * (self.params["slr_ice_mult"] - 1))
        # AMOC: the year its effects begin (inf = not before 2066), ramping in over 15 years
        u_a = torch.rand(S, 1, device=device, generator=gc)
        y_a = 2035.0 + 30.0 * torch.rand(S, 1, device=device, generator=gc)
        self.amoc_year = torch.where(u_a < torch.as_tensor(self.prm("amoc_p"), device=device), y_a, torch.full_like(y_a, float("inf")))
        if self.cfg.sample_params:
            self.params["amoc_hit"] = (self.amoc_year < 2066).float()
        if self.cfg.ai.arrival_cdf:
            ys, ps = map(np.asarray, zip(*self.cfg.ai.arrival_cdf))
            u = torch.rand(S, 1, device=device, generator=g).cpu().numpy()
            yr = np.where(u <= ps[-1], np.interp(u, ps, ys), np.inf)
            self.arrival_w = torch.as_tensor(yr, dtype=torch.float32, device=device)
            self.params["ai_arrival"] = self.arrival_w.clamp(max=2070.0)         # for the attribution of uncertainty
        if self.cfg.ai.arrival:
            self._draw_robot_arrival(S, device, g, u if self.cfg.ai.arrival_cdf else None)
        ge_ = torch.Generator(device=device).manual_seed(seed + 929)        # (own generator; 919 is the workforce draw)
        self.explosive = (torch.rand(S, 1, device=device, generator=ge_) < self.cfg.explosive_p)
        self.explosive_mult = 3.0 + 3.0 * torch.rand(S, 1, device=device, generator=ge_)
        if self.cfg.param_block > 1:
            # two-level runs (second debate, 2026-11-06): worlds in blocks of param_block share every assumption, so
            # each block's share of outcomes is a probability given those assumptions, and the spread across blocks is
            # the part of the uncertainty that comes from the assumptions (scripts/credence.py)
            B = self.cfg.param_block
            blk = lambda x: x[::B].repeat_interleave(B, 0)[:S] if torch.is_tensor(x) and x.shape[0] == S else x
            self.params = {k: blk(v) for k, v in self.params.items()}
            for a_ in ("arrival_w", "robot_w", "amoc_year"):
                if hasattr(self, a_):
                    setattr(self, a_, blk(getattr(self, a_)))

    def _draw_robot_arrival(self, S, device, g, u_ai):
        """robotics arrival year per world: the ROBOT_CDF quantile of a normal score correlated (robot_corr) with the
        AI arrival's score, capped at AI arrival + robot_lag. With a fixed AI year, its score is the ARRIVAL_CDF
        quantile of that year."""
        from scipy.stats import norm
        ai = self.cfg.ai
        if u_ai is None:
            ys, ps = map(np.asarray, zip(*ARRIVAL_CDF))
            ai_w = ai.arrival + self.prm("arrival_sd")
            u_ai = np.interp(np.broadcast_to(np.asarray(ai_w.cpu() if torch.is_tensor(ai_w) else ai_w, float), (S, 1)) + 0.5, ys, ps)
        else:
            ai_w = self.arrival_w
        z_ai = norm.ppf(np.clip(u_ai, 1e-4, 1 - 1e-4))
        eps = torch.randn(S, 1, device=device, generator=g).cpu().numpy()
        u_r = norm.cdf(ai.robot_corr * z_ai + np.sqrt(1 - ai.robot_corr ** 2) * eps)
        ys, ps = map(np.asarray, zip(*(ai.robot_cdf or ROBOT_CDF)))
        r = np.where(u_r <= ps[-1], np.interp(u_r, ps, ys), np.inf)
        r = torch.as_tensor(r, dtype=torch.float32, device=device)
        cap = ai_w + self.prm("robot_lag")
        self.robot_w = torch.minimum(r, cap if torch.is_tensor(cap) else torch.full_like(r, float(cap)))
        self.params["robot_arrival"] = self.robot_w.clamp(max=2070.0)

    def __init__(self, cfg: WorldConfig, p, base, origin, S, device, gen, dyads=None):
        self.cfg, self.S, self.dev, self.gen = cfg, S, device, gen
        self.gen_cascade = torch.Generator(device=device)
        self.gen_cascade.manual_seed(int(gen.initial_seed()) + 7919 if gen is not None else 7919)
        self._draw_params(S, device, int(gen.initial_seed()) if hasattr(gen, "initial_seed") else 0)
        self.ai_cat_done = torch.zeros(S, device=device)         # AI catastrophe already happened in this world
        self.ai_cat_back = torch.zeros(S, device=device)         # output still to recover from it
        self.tech = None
        if self.cfg.tech and origin >= 2025:              # (hindcasts from earlier origins are unchanged)
            from .tech import TechState
            ai_w = None
            if self.cfg.ai.arrival:
                ai_w = (self.arrival_w if self.cfg.ai.arrival_cdf else self.cfg.ai.arrival + self.prm("arrival_sd"))
                ai_w = (ai_w * torch.ones(S, 1, device=device)).cpu().numpy().ravel()
            self.tech = TechState(S, device, int(gen.initial_seed()) if hasattr(gen, "initial_seed") else 0, ai_w,
                                  force=self.cfg.tech_force)
            if self.cfg.sample_params:
                for name_, yr_ in self.tech.arrival.items():          # arrival years, for the attribution of uncertainty
                    self.params[f"tech_{name_}"] = torch.as_tensor(np.minimum(yr_, 2070.0), dtype=torch.float32, device=device)[:, None]
        C = len(p.codes)
        t0 = p.yi(origin)
        f = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float32), device=device)
        self.origin = origin
        # --- AI readiness (percentile of a mean z-score; data at the origin)
        def z(a):
            a = np.asarray(a, float)
            return (a - np.nanmean(a)) / (np.nanstd(a) + 1e-9)
        inet = p.x.get("internet", np.full((C, len(p.years)), np.nan))
        last = lambda r: r[np.isfinite(r)][-1] if np.isfinite(r).any() else np.nan
        inet_now = np.array([last(r[:t0 + 1]) for r in inet])
        parts = [z(np.log1p(base["energy_pc"][:, t0])), z(base["tertiary_enrol"][:, t0]), z(inet_now),
                 z(base["rule_of_law"][:, t0]), z(base["lg"][:, t0])]
        R = np.nanmean(np.stack(parts), 0)
        R = np.nan_to_num((np.argsort(np.argsort(np.nan_to_num(R, nan=-9))) + 0.5) / C, nan=0.2)
        self.readiness = f(R)
        cog = 0.3 + 0.5 * R                            # cognitive share of output rises with readiness
        # measured instead where available: employment in the occupations most exposed to generative AI (ISCO 1-4,
        # ILO modelled estimates), mapped to the same mean and spread (scripts/build_ai_exposure.py)
        import os as _os
        _f = _os.path.join(_os.path.dirname(__file__), "..", "data", "ai_exposure.npz")
        if cfg.ai_exposure_data and _os.path.exists(_f):
            _z = np.load(_f)
            if list(_z["codes"]) == list(p.codes):
                cog = np.where(np.isfinite(_z["cog_share"]), _z["cog_share"], cog)
        self.cog_share = f(cog)
        trade = np.nan_to_num(base["trade"][:, t0], nan=60)
        self.exporter = f(np.clip(trade / 100, 0, 1.5))
        self.fiscal = f(np.clip(np.nan_to_num(base["fiscal_capacity"][:, t0]) / 3 + 0.5, 0, 1))
        self.e_frontier = float(np.nanquantile(base["energy_pc"][:, t0], 0.9))
        self.hub = f([HUBS.get(c, 0.0) for c in p.codes])
        self.controlled = f([1.0 if c in EXPORT_CONTROLLED else 0.0 for c in p.codes])
        self.adopt_prev = torch.zeros(S, C, device=device)
        self.tau = torch.full((S, 1), float(origin), device=device)       # frontier-progress clock (slowed by chip wars)
        self.chip_ix = [p.codes.index(c) for c in ("TWN", "KOR") if c in p.codes]
        self.chip_hit = torch.zeros(S, 1, device=device)
        self.A_robot = torch.zeros(S, C, device=device)
        self.robot_frontier = torch.zeros(S, 1, device=device)
        from . import innovation, intervention, ocean, structure
        sv = structure.inputs(p.codes, origin)
        self.lreb0, self.st_debt, self.peer_w = f(sv["lreb"]), f(sv["st"]), f(sv["peers"])
        iv = intervention.inputs(p.codes, origin, np.exp(base["lg"][:, t0] + base["lpop"][:, t0]))
        recent = origin >= 2025                  # the 2025 aid cut, tariffs and peacekeeping cut exist only from 2025
        self.tariff_hit = f(iv["tariff"] * recent)
        self.aid_cut = f(iv["aid_cut"] * recent)
        self.pko_cut_on = float(recent)
        self.is_usa = f([1.0 if c == "USA" else 0.0 for c in p.codes])
        self.pko_n = f(iv["pko"])
        self.fuel_x, self.food_x = f(iv["fuel"]), f(iv["food"])
        self.science = f(innovation.article_capacity(p.codes, p.years, base["lpop"], base["lg"], origin))
        self.cum_A = torch.zeros(S, C, device=device)                   # years of full AI adoption so far
        oc = ocean.inputs(p.codes, base, p.years, origin)
        self.fish_cal, self.fish_wild, self.fish_gdp = f(oc["cal"]), f(oc["wild"]), f(oc["gdp"])
        self.fish_band = torch.as_tensor(oc["band"], device=device).long()
        self.fish_capt = f(oc["gdp"] * np.exp(np.nan_to_num(base["lg"][:, t0] + base["lpop"][:, t0])))   # catch value
        self.mil_auto = torch.zeros(S, 1, device=device)                   # military autonomy, 0..1
        # --- food: country yield trend (damped), calories
        cy = base["cereal_yield"]
        tr = []
        for r in np.log(cy[:, max(0, t0 - 15):t0 + 1]):
            # Theil-Sen (median of pairwise slopes): a drought in the last year (Zambia 2024) does not set the trend
            x = np.arange(len(r))[np.isfinite(r)]
            y = r[np.isfinite(r)]
            sl = [(y[j] - y[i]) / (x[j] - x[i]) for i in range(len(x)) for j in range(i + 1, len(x))]
            tr.append(float(np.median(sl)) if len(x) >= 5 else 0.01)
        # country yield trends barely persist: regressing each country's 15-year trend on the previous 15 years gives
        # slopes of -0.09, 0.01 and 0.20 (windows 1965-80/80-95, 80-95/95-10, 95-10/10-25), so the projection is the
        # cross-country mean trend plus 0.04 x the country's deviation from it
        tr = np.asarray(tr, float)
        m_tr = float(np.nanmean(tr))
        self.yield_trend = f(np.clip(np.nan_to_num(m_tr + 0.04 * (tr - m_tr), nan=m_tr), -0.01, 0.03))
        # --- water: each country's exposure to glacier-fed basins (population share x glacier share of the flow) and
        # how binding water is for its farms; water_change is its water supply relative to the origin (fraction)
        self.water_change = torch.zeros(S, C, device=device)
        if cfg.water:
            from . import water as W_
            bas = list(W_.PEAK_WATER)
            G = np.zeros((C, len(bas)), np.float32)
            for i, c in enumerate(p.codes):
                for b_, s_ in W_.glacier_share(c).items():
                    G[i, bas.index(b_)] = s_ * W_.GLACIER_FLOW[b_]
            self.glac_w = f(G)
            self.peak_year = f([W_.PEAK_WATER[b_] for b_ in bas])
            self.water_bind = f(np.nan_to_num(W_.binding(p.codes, origin), nan=0.2))
        # --- missing factors (risks.py): nuclear status per world, weapons programs, trade links, the West, minerals,
        # low-lying coasts, upstream rain
        from . import risks as R_
        self.gen_risk = torch.Generator(device=device)
        self.gen_risk.manual_seed(int(gen.initial_seed()) + 7777 if gen is not None else 7777)
        warh = np.nan_to_num(np.asarray(p.x["warheads"], float)[:, t0]) if "warheads" in p.x else np.zeros(C)
        self.nuclear = (torch.as_tensor(warh > 0, device=device)[None].repeat(S, 1))
        self.nuclear0 = self.nuclear.clone()
        # weapons programs: risks.LATENT hazards, only from the 2025 origin (a hindcast would not know them)
        self.latent = f([R_.LATENT.get(c, 0.0) if origin >= 2025 else 0.0 for c in p.codes])
        self.pursuit = torch.zeros(S, C, dtype=torch.bool, device=device)
        self.pursuit_t = torch.zeros(S, C, device=device)
        if origin >= 2025 and "IRN" in p.codes:
            u0 = torch.rand(S, 1, device=device, generator=self.gen_risk)
            i0 = list(p.codes).index("IRN")
            self.pursuit[:, i0] = (u0 < self.prm("iran_pursuit0"))[:, 0]
        gdp0 = np.exp(base["lg"][:, t0] + base["lpop"][:, t0])
        Wt = R_.trade_weights(list(p.codes), gdp0)
        self.trade_w = f(Wt)
        self.trade_w0, self.trade_gdp0 = self.trade_w.clone(), f(np.nan_to_num(gdp0, nan=0.0))
        # first-round crisis scale for each contagion strength, solved so that the two-round process keeps the
        # crisis rate it starts from (World.contagion_scale); historical exposure includes contagion's own
        # clustering, so dividing by it would undershoot
        self.cont_grid, self.cont_scale = _contagion_table(Wt, 0.065)
        tr0 = np.nan_to_num(base["trade"][:, t0] / 100, nan=0.6)
        self.trade_rel = f(tr0 / tr0.mean())
        west = np.array([c in R_.WEST for c in p.codes], float)
        self.west = f(west)
        self.west_share = f(Wt @ west)
        self.rents = f(R_.mineral_rents(list(p.codes), origin))
        self.lowpop = f(R_.low_elevation(list(p.codes)))
        mask = lambda names: f(np.array([c in names for c in p.codes], float))
        self.nw_europe, self.sahel, self.south_asia = mask(NW_EUROPE), mask(WEST_AFRICA_SAHEL), mask(SOUTH_ASIA)
        from .food import agri_share, dryland
        self.agri = f(agri_share(p, origin))                      # farm share of GDP at the origin (0-1)
        self.dry = f(dryland(p))                                  # 1961-1990 rainfall below 600 mm a year
        pr_base = np.nanmean(np.asarray(p.x["c_pr"], float)[:, max(0, t0 - 12):t0 - 2], 1) if "c_pr" in p.x else None
        self.upstream = f(R_.upstream_precip(list(p.codes), pr_base))
        self.nuc_hit = torch.zeros(S, C, device=device)          # this year's output level lost to nuclear use
        self.deaths_year = torch.zeros(S, C, device=device)      # this year's war deaths (battle + civilian + nuclear)
        self.sanctioned = torch.zeros(S, C, device=device)
        self.china_hit = torch.zeros(S, 1, device=device)
        self.tau_r = torch.full((S, 1), float(origin), device=device)    # robot-hardware clock
        self.cyber = torch.zeros(S, C, device=device)
        self.codes = list(p.codes)
        # --- interstate dyads
        self.dy = dyads if (cfg.interstate and dyads is not None) else None
        if self.dy is not None:
            self._init_dyads(p, base, t0, f)
        self.rec = {"ai_catastrophe": [], "clim_deaths": [], "wars": [], "active": [], "great_power_war": [], "pandemic": [], "ai_adoption_world": [], "oil": [], "clean_share": [], "war_site_share": [], "war_sites": [], "bubble": [],
                    "eng_pandemic": [], "food_price": [], "import_dep": [], "robot_adoption_world": [], "mil_autonomy": [], "battle_deaths": [],
                    "war_deaths_major": [], "water_change": [], "nuclear_use": [], "nuclear_large": [], "nuclear_deaths": [],
                    "nuclear_states": [], "sanctioned": [], "solar": [], "volcano": [], "pop_dev": [], "excess_deaths": [], "wild_catch": [], "great_power_war_cow": [],
                    "amoc": [], "harvest_us": [], "harvest_bad4": [], "clim_migrants": []}
        self.new_country = torch.zeros(S, len(p.codes), device=device)
        self.new_war_country = torch.zeros(S, len(p.codes), device=device)
        self.watch_hits = None

    # ---------------------------------------------------------------- water
    def water_step(self, year, pr_ratio):
        """update each country's water supply relative to the origin [S, C] from glacier runoff and precipitation
        (pr_ratio: precipitation / its origin-decade mean); returns this year's change (for yields)."""
        tau, sh = self.prm("glacier_tau"), self.prm("peak_shift")
        hump = lambda t: torch.exp(-0.5 * ((t - self.peak_year[None] - sh) / tau) ** 2)     # [S or 1, B]
        # glacier runoff vs the origin (denominator floored: a basin far before its peak at the origin would otherwise
        # multiply its glacier share many times over)
        rel = (hump(float(year)) - hump(float(self.origin))) / hump(float(self.origin)).clamp(min=0.5)
        glac = self.prm("glacier_flow_mult") * (rel @ self.glac_w.T)
        # precipitation where the water comes from: own rain and upstream rain in shared basins (risks.upstream_precip)
        pr_eff = pr_ratio.clamp(0.5, 1.5) @ self.upstream.T
        wc = glac + self.cfg.runoff_elast_pr * (pr_eff - 1)
        if self.tech is not None:
            wc = torch.where(wc < 0, wc * (1 - self.tech.lever("water_cut").clamp(max=1)), wc)   # desalination
        d = wc - self.water_change
        self.water_change = wc.expand_as(self.water_change).clone()
        self.rec["water_change"].append(self.water_change.median(0).values)
        return d

    # ---------------------------------------------------------------- dyads
    def _init_dyads(self, p, base, t0, f):
        d = self.dy
        ci = {c: i for i, c in enumerate(p.codes)}
        rows = d["now"]
        self.ia = torch.as_tensor([ci[a] for a in rows.a], device=self.dev)
        self.ib = torch.as_tensor([ci[b] for b in rows.b], device=self.dev)
        P = len(rows)
        self.P = P
        # missing static features stay missing and take the training mean when standardized
        self.static = {k: f(rows[k].to_numpy(float)) for k in STATE_FEATS}
        self.k_now = 0
        self.ev_decay = {c: (torch.where(torch.isfinite(f(d["ev_mean"][c])), f(d["ev_mean"][c]), self.static[c]), float(d["ev_rho"][c]))
                         for c in d.get("ev_mean", {}) if c in self.static}
        S = self.S
        # event hostility per world: deviation from the pair's mean, raised by simulated conflicts (interstate.sim_inputs)
        self.ev_war = {c: d["ev_war"][c] for c in self.ev_decay if c in d.get("ev_war", {})} if self.cfg.ev_war_feedback else {}
        self.ev_dev = {c: (self.static[c] - self.ev_decay[c][0])[None].repeat(S, 1).clone() for c in self.ev_war}
        rep = lambda k: torch.nan_to_num(self.static[k])[None].repeat(S, 1).clone()
        self.ongoing = rep("ongoing")
        self.war = rep("war_now")
        self.run = rep("conflict_run_years")
        self.past10 = rep("conflict_past_10y")
        self.since = torch.expm1(rep("log_years_since_conflict"))
        self.ever = rep("ever_conflict")
        # conflict years origin-10 .. origin-1; at the end of each step the year just simulated's predecessor is
        # appended, so past10 / since / ever always describe the years before the row's year, as in training
        h10 = d.get("hist10")
        self.hist10 = (f(h10)[None].repeat(S, 1, 1) if h10 is not None else torch.zeros(S, P, 10, device=self.dev))
        # log total GDP of origin-10 .. origin-1 (missing stays missing: the feature then takes its training mean)
        lgdp0 = torch.as_tensor(base["lg"][:, t0 - 10:t0] + base["lpop"][:, t0 - 10:t0], dtype=torch.float32, device=self.dev)
        self.lgdp_hist = lgdp0[None].repeat(S, 1, 1)
        m = d["model_1y"]
        self.lr_w = f(m.lr.coef_[0])
        self.lr_b = float(m.lr.intercept_[0])
        self.mu, self.sd = f(np.nan_to_num(m.mu)), f(np.nan_to_num(m.sd, nan=1.0))
        self.pair_off = f(d["pair_offset"]) if d.get("pair_offset") is not None else torch.zeros(P, device=self.dev)
        # structural uncertainty: one error per world, scaled by each pair's model-family disagreement
        self.pair_spread = f(d["pair_spread"]) if (d.get("pair_spread") is not None and self.cfg.sample_params) \
            else torch.zeros(P, device=self.dev)
        g_ps = torch.Generator(device=self.dev).manual_seed((int(self.gen.initial_seed()) if self.gen is not None else 0) + 5151)
        self.z_pair = torch.randn(self.S, 1, device=self.dev, generator=g_ps)
        if self.cfg.param_block > 1:
            self.z_pair = self.z_pair[::self.cfg.param_block].repeat_interleave(self.cfg.param_block, 0)[:self.S]
        self.onset_target = d.get("onset_rate")
        self.dyad_shift = 0.0
        cont = d["continuation"]
        self.cont = f([0.0, cont[1], cont[2], cont[3]])
        A = np.zeros((len(p.codes), len(p.codes)), np.float32)
        for c, al in d["allies"].items():
            if c in ci:
                for x in al:
                    if x in ci:
                        A[ci[c], ci[x]] = 1
        self.ally = f(A)
        self.ci = ci
        self.pair_names = list(zip(rows.a, rows.b))
        # who can realistically join a war through an alliance, and where a war is fought
        maj = set(np.asarray(rows.a)[rows.major_pair.to_numpy() > 0]) | set(np.asarray(rows.b)[rows.major_pair.to_numpy() > 0])
        near = (rows.log_cap_dist.to_numpy() < np.log1p(self.cfg.ally_reach_km)) | (rows.contig_land.to_numpy() > 0) | \
               (rows.contig_sea.to_numpy() > 0)
        self.near = f(near) > 0
        # global reach (joining wars far from home): the permanent Security Council members only
        self.is_major_c = torch.as_tensor([c in ("USA", "RUS", "CHN", "GBR", "FRA") for c in p.codes], device=self.dev)
        self._near_cc = None                                        # [C, C] near pairs (built with pair_index)
        # great powers for war size and the great-power-war count: the COW majors (as in the dyad model's features),
        # plus any nuclear-armed state with at least GP_SHARE of world output, re-evaluated every simulated year
        # (India today; others if they grow or arm). The dyad hazard keeps the COW coding it was trained on.
        self.cow_major_c = torch.as_tensor([c in maj for c in p.codes], device=self.dev)
        t0_ = t0
        self._great_powers(torch.as_tensor(np.nan_to_num(base["lg"][:, t0_]), device=self.dev, dtype=torch.float32)[None],
                           torch.as_tensor(np.nan_to_num(base["lpop"][:, t0_]), device=self.dev, dtype=torch.float32)[None])
        contig = (rows.contig_land.to_numpy() > 0) | (rows.contig_sea.to_numpy() > 0)
        exa, exb = rows.a.isin(self.cfg.expeditionary).to_numpy(), rows.b.isin(self.cfg.expeditionary).to_numpy()
        both = exa & exb
        self.site_a = f(np.where(both, 0.5, np.where(exa, 0.0, 1.0)))
        self.site_b = f(np.where(both, 0.5, np.where(exb, 0.0, 1.0)))
        self.war_site = torch.zeros(S, len(p.codes), device=self.dev)
        self.strike_site = torch.zeros(S, len(p.codes), device=self.dev)
        self.pair_index = torch.full((len(p.codes), len(p.codes)), -1, dtype=torch.long, device=self.dev)
        self.pair_index[self.ia, self.ib] = torch.arange(P, device=self.dev)
        self.pair_index[self.ib, self.ia] = torch.arange(P, device=self.dev)
        self.rival_major = torch.nan_to_num(self.static["rival"] * torch.clamp(self.static["major_pair"] + self.static["one_major"], max=1))
        self.major_pair = torch.nan_to_num(self.static["major_pair"])
        watch = d.get("watch_pairs", [])
        self.watch = [(nm, int(self.pair_index[ci[a], ci[b]])) for nm, a, b in watch if a in ci and b in ci
                      and int(self.pair_index[ci[a], ci[b]]) >= 0]
        self.watch_ever = torch.zeros(S, len(self.watch), device=self.dev)
        self.bloc = d.get("blocs", {})
        self.bloc_idx = {k: torch.as_tensor([int(self.pair_index[ci[a], ci[b]]) for a, b in v
                                             if a in ci and b in ci and int(self.pair_index[ci[a], ci[b]]) >= 0],
                                            device=self.dev) for k, v in self.bloc.items()}
        self.bloc_ever = {k: torch.zeros(S, device=self.dev) for k in self.bloc}
        self.pair_ever = torch.zeros(P, device=self.dev)       # P(pair fights at some point), summed over worlds
        # --- war size (psychohistory/war_size.py): battle deaths per war-year, drawn when a war starts; a separate
        # generator keeps paired scenario runs paired
        from .war_size import BW, CAP, FLOOR, pools
        P_ = pools(self.origin)
        self.ws_major, self.ws_other = torch.log10(f(P_["major"])), torch.log10(f(P_["other"]))
        self.ws_minor, self.ws_bw, self.ws_cap, self.ws_floor = f(P_["minor"]), BW, CAP, FLOOR
        self.gen_size = torch.Generator(device=self.dev)
        self.gen_size.manual_seed(int(self.gen.initial_seed()) + 104729 if self.gen is not None else 104729)
        self.war_rate = torch.where(self.war > 0.5, self._draw_rate(), torch.zeros(S, P, device=self.dev))
        for (a_, b_), v_ in P_["known"].items():          # wars under way at the origin keep their observed rate
            if a_ in ci and b_ in ci and int(self.pair_index[ci[a_], ci[b_]]) >= 0:
                self.war_rate[:, int(self.pair_index[ci[a_], ci[b_]])] = v_
        self.ep_total = self.war_rate * (self.war > 0.5)
        self.max_ep = torch.zeros(S, device=self.dev)
        self.watch_war_ever = torch.zeros(S, len(self.watch), device=self.dev)
        self.watch_ep_max = torch.zeros(S, len(self.watch), device=self.dev)
        self.watch_war_hits = None
        self.bloc_war_ever = {k: torch.zeros(S, device=self.dev) for k in self.bloc}
        self.pair_any = torch.zeros(S, P, dtype=torch.bool, device=self.dev)
        from . import water as W_
        pb = W_.pair_basins()
        self.shared_basin = f([float((a_, b_) in pb) for a_, b_ in self.pair_names])
        # (disputes as of 2025: not applied in hindcasts from earlier origins)
        # security patrons without a treaty: for each client pair (client, enemy) the patron-enemy pair it would create
        from .interstate import QUASI_ALLIES
        self.quasi = {}
        for (pat, cli) in QUASI_ALLIES:
            if pat not in ci or cli not in ci:
                continue
            lst = []
            for k_, (a_, b_) in enumerate(self.pair_names):
                if cli in (a_, b_):
                    en = b_ if a_ == cli else a_
                    pe = int(self.pair_index[ci[pat], ci[en]]) if en != pat else -1
                    if pe >= 0 and self.ally[ci[pat], ci[en]] == 0:
                        lst.append((k_, pe))
            self.quasi[(pat, cli)] = (f"quasi_{pat}_{cli}", lst if self.origin >= 2025 else [])
        self.water_dispute = f([float((a_, b_) in W_.WATER_DISPUTES and self.origin >= 2025) for a_, b_ in self.pair_names])

    def _consequences(self, war, active, deaths):
        """nuclear use, war deaths by country, sanctions, the China supply chokepoint and wartime cyber attacks
        (assumptions, world.EXTRA_RANGES)"""
        S, dev, g = self.S, self.dev, self.gen_risk
        Cn = self.nuclear.shape[1]
        wf = war.float()
        both_n = (self.nuclear[:, self.ia] & self.nuclear[:, self.ib]).float()
        one_n = (self.nuclear[:, self.ia] ^ self.nuclear[:, self.ib]).float()
        # nuclear use: limited (10,000 to 1 million deaths, the belligerents lose 5% of output) or, between two great
        # powers (the changing set, _great_powers), a large exchange (10 to 200 million deaths, belligerents lose half their output, the world
        # nuc_gdp_large through trade, finance and climate effects)
        p_use = wf * (both_n * self.prm("nuc_use_both") + one_n * self.prm("nuc_use_one"))
        R_sp = self.space_ramp(self.origin + getattr(self, "k_now", 0))
        if torch.is_tensor(R_sp) or R_sp:
            p_use = p_use * torch.exp(self.prm("space_war") * R_sp * both_n * (self.is_usa[self.ia] + self.is_usa[self.ib])[None])
        use = torch.rand(S, self.P, device=dev, generator=g) < p_use
        large = use & (self.gp_pair > 0) & (both_n > 0) & (torch.rand(S, self.P, device=dev, generator=g) < self.prm("nuc_escalate"))
        lim_d = 10 ** (4 + 2 * torch.rand(S, self.P, device=dev, generator=g))
        big_d = 10 ** (7 + 1.3 * torch.rand(S, self.P, device=dev, generator=g))
        nd = torch.where(large, big_d, torch.where(use, lim_d, torch.zeros_like(lim_d)))
        hit_pair = torch.where(large, torch.full_like(nd, 0.5), torch.where(use, torch.full_like(nd, 0.05), torch.zeros_like(nd)))
        # (a side is hit only if the other side has weapons: in a one-sided use the target bears it all)
        nb_s, na_s = self.nuclear[:, self.ib].float(), self.nuclear[:, self.ia].float()
        hit = torch.zeros(S, Cn, device=dev).index_add_(1, self.ia, hit_pair * nb_s).index_add_(1, self.ib, hit_pair * na_s).clamp(max=0.7)
        any_large = large.any(-1, keepdim=True).float()
        self.nuc_hit = torch.maximum(hit, any_large * self.prm("nuc_gdp_large"))
        self.rec["nuclear_use"].append(use.any(-1).float())
        self.rec["nuclear_large"].append(any_large[:, 0])
        self.rec["nuclear_deaths"].append(nd.sum(-1))
        # deaths by country (for population): battle deaths x (1 + civilian multiplier), split between the two sides,
        # plus nuclear deaths
        # (deaths where the war is fought: war-site shares, so an expeditionary power loses soldiers, not civilians;
        # nuclear deaths on the side that can be struck)
        wa_ = self.site_a / (self.site_a + self.site_b)
        conv = deaths * (1 + self.prm("civ_mult"))
        ns_ = (na_s + nb_s).clamp(min=1)
        da = conv * wa_ + nd * nb_s / ns_
        db = conv * (1 - wa_) + nd * na_s / ns_
        self.deaths_year = torch.zeros(S, Cn, device=dev).index_add_(1, self.ia, da).index_add_(1, self.ib, db)
        # sanctions: a non-Western state at war (war level) with a Western one (risks.WEST)
        wa, wb = self.west[self.ia][None], self.west[self.ib][None]
        sa = wf * wb * (1 - wa)                            # a is sanctioned
        sb = wf * wa * (1 - wb)
        self.sanctioned = (torch.zeros(S, Cn, device=dev).index_add_(1, self.ia, sa).index_add_(1, self.ib, sb) > 0).float()
        self.rec["sanctioned"].append(self.sanctioned.sum(-1))
        # China at war with the West: clean-tech and robot hardware supply stalls
        if "CHN" in self.ci:
            self.china_hit = self.sanctioned[:, self.ci["CHN"]:self.ci["CHN"] + 1]
        # a weapons program is set back to its start by a war with a nuclear-armed state or a major power
        nb_, na_ = self.nuclear[:, self.ib].float(), self.nuclear[:, self.ia].float()
        mb, ma = self.is_major_c[self.ib].float()[None], self.is_major_c[self.ia].float()[None]
        struck = torch.zeros(S, Cn, device=dev).index_add_(1, self.ia, wf * (nb_ + mb).clamp(max=1)) \
            .index_add_(1, self.ib, wf * (na_ + ma).clamp(max=1)) > 0
        # (and abandoned half the time)
        struck &= self.k_now > 1          # (iran_pursuit0 describes programs after the recorded 2026 strikes)
        drop = struck & self.pursuit & (torch.rand(S, Cn, device=dev, generator=g) < 0.5)
        self.pursuit_t = torch.where(struck & self.pursuit, torch.zeros_like(self.pursuit_t), self.pursuit_t)
        self.pursuit &= ~drop
        # great-power war: infrastructure and cyber attacks on both sides
        gp = wf * self.gp_pair
        self.cyber = (torch.zeros(S, Cn, device=dev).index_add_(1, self.ia, gp).index_add_(1, self.ib, gp) > 0).float()

    def _join(self, new_war, q, joiners_only=False, exclude=None):
        """pairs (ally, enemy) created when defence allies join the war-level conflicts in `new_war` [S, P], and the
        countries that had a chance to join [S, C]. joiners_only: `new_war` holds (joiner, enemy) pairs from a first
        round, and only allies of the joiner (not of the enemy) are candidates; exclude [S, C]: countries that may not
        join. The random draws come from a separate generator, so a different set of wars in one run does not shift
        every later draw of the main generator (paired scenario runs stay comparable)"""
        dev = self.dev
        C = self.ally.shape[0]
        joined = torch.zeros_like(new_war)
        cand = torch.zeros(self.S, C, dtype=torch.bool, device=dev)
        if not new_war.any():
            return joined, cand
        s_idx, p_idx = torch.nonzero(new_war, as_tuple=True)
        worlds, s_loc = torch.unique(s_idx, return_inverse=True)       # only worlds with a new war this year
        Nw = torch.zeros(len(worlds), C, C, device=dev)
        if joiners_only:
            # Nw[w, enemy, joiner]: allies of the joiner (columns) may join against the enemy (rows)
            a_, b_ = self.ia[p_idx], self.ib[p_idx]
            jn = self.joiner_side(s_idx, p_idx)
            en_, jo_ = torch.where(jn, b_, a_), torch.where(jn, a_, b_)
            Nw[s_loc, en_, jo_] = 1
        else:
            Nw[s_loc, self.ia[p_idx], self.ib[p_idx]] = 1
            Nw[s_loc, self.ib[p_idx], self.ia[p_idx]] = 1
        # J[w, i, c] > 0: country i is at war with some j, and c is a defence ally of j but not of i (allies
        # of both sides of an intra-alliance war stay out)
        J = (torch.matmul(Nw, self.ally) > 0) & (self.ally[None] == 0)
        if exclude is not None:
            J &= ~exclude[worlds][:, None, :]
        cand[worlds] = J.any(1)
        q = q[worlds][:, :, None] if torch.is_tensor(q) else q
        # treaty scope: in scope when the enemy i is near a partner j that c defends, or near c itself; a global power
        # (is_major_c) outside that area joins at ally_out_scope x q (others are held to nearness below, as before)
        if self._near_cc is None:
            pi = self.pair_index
            self._near_cc = (pi >= 0) & self.near[pi.clamp(min=0)]
        nc = self._near_cc.float()
        scope = (torch.matmul(Nw * nc[None], self.ally) > 0) | self._near_cc[None]
        out = self.prm("ally_out_scope")
        out = out[worlds][:, :, None] if torch.is_tensor(out) else out
        q = q * torch.where(scope, torch.ones((), device=dev), out * torch.ones((), device=dev))
        J &= torch.rand(J.shape, device=dev, generator=self.gen_cascade) < q
        w2, i2, c2 = torch.nonzero(J, as_tuple=True)
        s2 = worlds[w2]
        pidx = self.pair_index[i2, c2]
        ok = (pidx >= 0) & (i2 != c2)
        ok &= self.near[pidx.clamp(min=0)] | self.is_major_c[c2]
        joined[s2[ok], pidx[ok]] = True
        if not joiners_only:
            self._last_joiner = (s2[ok], pidx[ok], c2[ok])       # who joined (for the second round)
        return joined, cand

    def joiner_side(self, s_idx, p_idx):
        """for first-round joined pairs: True where side a (ia) of the pair is the joiner"""
        js, jp, jc = getattr(self, "_last_joiner", (None, None, None))
        out = torch.zeros(len(p_idx), dtype=torch.bool, device=self.dev)
        if js is None or not len(js):
            return out
        key = s_idx * self.P + p_idx
        jkey = js * self.P + jp
        m = torch.isin(key, jkey)
        if m.any():
            # the joiner is the country of the pair recorded as c
            lut = torch.full((int(jkey.max()) + 1,), -1, dtype=torch.long, device=self.dev)
            lut[jkey] = jc
            cj = lut[key[m]]
            out[m] = self.ia[p_idx[m]] == cj
        return out

    def _join_quasi(self, new_war):
        """security patrons without a treaty (interstate.QUASI_ALLIES) joining a new war-level conflict against their
        client, each with its own chance (QUASI_RANGES)"""
        joined = torch.zeros_like(new_war)
        rec = []
        for (patron, client), (nm, pidx_list) in self.quasi.items():
            for pi, enemy_pair in pidx_list:          # pi: the client's pair with an enemy; enemy_pair: patron-enemy
                hit = new_war[:, pi]
                if hit.any():
                    q = self.prm(nm)
                    u = torch.rand(self.S, device=self.dev, generator=self.gen_cascade)
                    j = hit & (u < (q[:, 0] if torch.is_tensor(q) else q))
                    joined[:, enemy_pair] |= j
                    w_ = torch.nonzero(j, as_tuple=True)[0]
                    if len(w_):
                        rec.append((w_, torch.full_like(w_, enemy_pair), torch.full_like(w_, self.ci[patron])))
        if rec:                                      # the patron is the joiner (for the second round)
            js, jp, jc = getattr(self, "_last_joiner", (None, None, None))
            parts = ([(js, jp, jc)] if js is not None else []) + rec
            self._last_joiner = tuple(torch.cat([p_[i] for p_ in parts]) for i in range(3))
        return joined

    def _great_powers(self, lg, lpop):
        gdp = torch.exp(lg + lpop).nan_to_num()
        share = gdp / gdp.sum(-1, keepdim=True).clamp(min=1e-9)
        gp = self.cow_major_c[None] | (self.nuclear & (share >= GP_SHARE))
        self.gp_c = gp.expand(self.S, -1) if gp.shape[0] == 1 else gp
        self.gp_pair = (self.gp_c[:, self.ia] & self.gp_c[:, self.ib]).float()        # [S, P]

    def _draw_rate(self):
        """battle deaths per war-year [S, P]: smoothed bootstrap from the pool of each pair's type"""
        S, P, dev, g = self.S, self.P, self.dev, self.gen_size
        def draw(pool):
            i = torch.randint(len(pool), (S, P), device=dev, generator=g)
            return pool[i] + self.ws_bw * torch.randn(S, P, device=dev, generator=g)
        lv = torch.where(self.gp_pair > 0, draw(self.ws_major), draw(self.ws_other))
        return (10 ** lv).clamp(self.ws_floor, self.ws_cap)

    def _dyad_features(self, lg, lpop, dem, conf, coup, sl=slice(None)):
        """standardised dyad features for the worlds in `sl` (chunked to bound GPU memory)."""
        lg, lpop, dem, conf, coup = lg[sl], lpop[sl], dem[sl], conf[sl], coup[sl]
        g = lg + lpop                                     # [s, C] log total GDP
        ga, gb = g[:, self.ia], g[:, self.ib]
        ratio = (ga - gb).abs()
        h = self.lgdp_hist[sl]
        ratio10 = (h[:, self.ia, 0] - h[:, self.ib, 0]).abs()
        da, db = dem[:, self.ia], dem[:, self.ib]
        n = lg.shape[0]
        feats = {k: v[None].expand(n, -1) for k, v in self.static.items()}
        feats.update({
            "abs_power_ratio": ratio, "power_shift_10y": ratio - ratio10, "log_joint_gdp": torch.logaddexp(ga, gb),
            "dem_min": torch.minimum(da, db), "dem_max": torch.maximum(da, db), "dem_gap": (da - db).abs(),
            "civil_war_any": ((conf[:, self.ia] + conf[:, self.ib]) > 0).float(),
            "coup_recent_max": torch.maximum(coup[:, self.ia], coup[:, self.ib]),
            "conflict_past_10y": self.past10[sl], "log_years_since_conflict": torch.log1p(self.since[sl].clamp(max=80)),
            "ever_conflict": self.ever[sl], "ongoing": self.ongoing[sl], "war_now": self.war[sl],
            "conflict_run_years": self.run[sl],
        })
        # nuclear status can change in simulation (proliferation): per world
        nuc = self.nuclear[sl]
        na, nb = nuc[:, self.ia], nuc[:, self.ib]
        feats["both_nuclear"], feats["one_nuclear"] = (na & nb).float(), (na ^ nb).float()
        for c_, (m_, rho_) in self.ev_decay.items():      # event hostility fades toward the pair's own mean
            if c_ in self.ev_dev:
                feats[c_] = m_[None] + self.ev_dev[c_][sl]
            else:
                feats[c_] = (m_ + (self.static[c_] - m_) * rho_ ** (self.k_now - 1))[None].expand(n, -1)
        # missing values take the training mean (z = 0), as in DyadModel._z
        X = torch.stack([feats[k] for k in DYAD_FEATS], -1)
        return torch.nan_to_num(((X - self.mu) / self.sd).clamp(-6, 6), nan=0.0)

    def proliferation_step(self):
        """weapons programs start (risks.LATENT x prolif_mult, x prolif_cascade once a rival has weapons it did not have
        at the origin) and end in a weapon after breakout_years; a war with a nuclear-armed state or a major power
        sets a program back to its start (Iran 2025)"""
        S, dev = self.S, self.dev
        if not bool((self.latent > 0).any()):
            return
        newn = (self.nuclear & ~self.nuclear0).float()
        riv = torch.nan_to_num(self.static["rival"])[None]
        rival_new = torch.zeros_like(newn).index_add_(1, self.ia, newn[:, self.ib] * riv).index_add_(1, self.ib, newn[:, self.ia] * riv)
        haz = self.latent[None] * self.prm("prolif_mult") * (1 + (self.prm("prolif_cascade") - 1) * (rival_new > 0).float())
        u = torch.rand(S, self.latent.shape[0], device=dev, generator=self.gen_risk)
        # (end of the year: running programs advance and may finish; new ones start, exposed from next year on)
        self.pursuit_t = torch.where(self.pursuit, self.pursuit_t + 1, torch.zeros_like(self.pursuit_t))
        acquire = self.pursuit & (self.pursuit_t >= self.prm("breakout_years"))
        self.nuclear |= acquire
        self.pursuit &= ~acquire
        start = (u < haz) & ~self.pursuit & ~self.nuclear
        self.pursuit |= start
        self.rec["nuclear_states"].append((self.nuclear & ~self.nuclear0).float().sum(-1))

    def interstate_step(self, k, lg, lpop, dem, conf, coup, mil_mult):
        """one year of interstate conflict. Returns per-country involvement [S, C] (0/1)."""
        S, P, dev = self.S, self.P, self.dev
        self.k_now = k
        self._great_powers(lg, lpop)
        both_n = (self.nuclear[:, self.ia] & self.nuclear[:, self.ib]).float()       # [S, P]
        one_n = (self.nuclear[:, self.ia] ^ self.nuclear[:, self.ib]).float()
        logit_onset = torch.empty(S, P, device=dev)
        for c0 in range(0, S, 1000):
            sl = slice(c0, min(S, c0 + 1000))
            logit_onset[sl] = self._dyad_features(lg, lpop, dem, conf, coup, sl) @ self.lr_w + self.lr_b
        logit_onset += self.pair_off * self.cfg.anchor_decay ** (k - 1)  # full-ensemble anchor per pair
        # (mean-preserving: the noise would raise the average probability of rare onsets by up to exp(sd^2 / 2), 3x at
        # the largest spread; the second-order correction, as for the civil-conflict mood, keeps the fitted mean.
        # Added 2026-10-28 after the Independent assumption audit: the hindcasts run without sampled parameters, so this
        # inflation of the production forecast was never tested)
        logit_onset += self.z_pair * self.pair_spread[None] \
            - PAIR_NOISE_CORR * 0.5 * self.pair_spread[None] ** 2 * (1 - 2 * torch.sigmoid(logit_onset))
        # democratic peace: pairs of democracies (both above 0.6) had no interstate conflict onset in 1946-2023
        # although the dyad model expected 3.4, so their hazard is scaled by (0 + 0.5) / 3.4 = 0.15 (with DYAD_CALIB
        # "drift" or "long" the calibration shift below is added on top, to all pairs)
        dmin = torch.minimum(dem[:, self.ia], dem[:, self.ib]).nan_to_num(0)
        logit_onset = logit_onset + float(np.log(self.cfg.dem_peace_factor)) * (dmin > 0.6).float()
        if k == 1 and self.onset_target:
            # calibration in the large (DYAD_CALIB): "drift" scales the 1-year hazards by observed / predicted onsets
            # over the previous decade, drifting to the 1946-origin ratio with a 10-year half-life (log-odds ~ log
            # ratio for small probabilities); "none" (production) uses the model as fitted
            self.shift_recent = float(np.log(self.dy.get("calib_recent", 1.0)))
            self.shift_long = float(np.log(self.dy.get("calib_long", 1.0)))
        if self.onset_target:
            if DYAD_CALIB == "none":
                self.dyad_shift = 0.0
            elif DYAD_CALIB == "long":
                self.dyad_shift = self.shift_long
            elif DYAD_CALIB == "drift":
                self.dyad_shift = self.shift_long + (self.shift_recent - self.shift_long) * 0.93 ** (k - 1)
            else:
                raise ValueError(f"DYAD_CALIB must be none, long or drift, not {DYAD_CALIB!r}")
        logit_onset = logit_onset + self.dyad_shift
        if self.cfg.major_era_mult != 1.0:
            # era scenario (scripts/era_test.py, 2026-11-06): the pair model fitted on 1946-2005 and scored on
            # 1816-1938 is calibrated overall (observed / predicted 1.03) but under-predicts fatal disputes between two
            # major powers by 30% (1.30); this multiplies the onset odds of major-power pairs (a scenario, not the headline)
            logit_onset = logit_onset + float(np.log(self.cfg.major_era_mult)) * self.major_pair[None]
        if mil_mult is not None:
            # AI and war (assumptions): rivals drift toward war when one pulls ahead in AI adoption (a window of
            # preventive advantage, as in the dreadnought race), and nuclear rivals lose some second-strike
            # confidence during the steepest part of the transition (AI-enabled tracking and cyber)
            A, fr = mil_mult
            rivals = torch.nan_to_num(self.static["rival"])[None]
            gap = (A[:, self.ia] - A[:, self.ib]).abs()
            logit_onset = logit_onset + self.prm("ai_war_beta") * gap * rivals \
                + self.prm("nuc_beta") * 4 * fr * (1 - fr) * both_n * rivals
        M = self.mil_auto                                  # [S, 1] military autonomy (0 without an AI scenario)
        logit_onset = logit_onset + self.prm("drone_onset") * M if self.cfg.ai.arrival else logit_onset
        if self.cfg.water:
            # live water disputes, and basin partners whose worse-off member is losing water (last year's supply)
            wcb = self.water_change * self.water_bind[None]          # a loss matters where water is scarce
            lost = torch.relu(-torch.minimum(wcb[:, self.ia], wcb[:, self.ib]))
            logit_onset = logit_onset + self.prm("water_dispute") * self.water_dispute[None] \
                + self.prm("water_scarcity_k") * lost * self.shared_basin[None]
        # the breakout window: a state pursuing a weapon and a nuclear-armed rival, or a nuclear-armed major power, are
        # likelier to fight (preventive strikes: Israel and the United States on Iran in 2025; assumption)
        pa, pb = self.pursuit[:, self.ia].float(), self.pursuit[:, self.ib].float()
        na_, nb_ = self.nuclear[:, self.ia].float(), self.nuclear[:, self.ib].float()
        # (rivals only, never allies: the US, Britain or France are not a preventive threat to an ally's program)
        riv_ = torch.nan_to_num(self.static["rival"])[None] * (1 - torch.nan_to_num(self.static["allied"])[None])
        prev_ = (pa * nb_ + pb * na_).clamp(max=1) * riv_
        logit_onset = logit_onset + self.prm("prevent_beta") * prev_
        # a state already fighting another interstate conflict starts new ones less often (force budget, assumption)
        busy_c = torch.zeros(S, self.nuclear.shape[1], device=dev).index_add_(1, self.ia, self.ongoing).index_add_(1, self.ib, self.ongoing)
        busy_p = ((busy_c[:, self.ia] + busy_c[:, self.ib] - 2 * self.ongoing) > 0).float()
        logit_onset = logit_onset - self.prm("busy_beta") * busy_p
        R_sp = self.space_ramp(self.origin + k)
        if torch.is_tensor(R_sp) or R_sp:
            # cheap space: the United States against a nuclear-armed rival (scenario assumption)
            logit_onset = logit_onset + self.prm("space_war") * R_sp * both_n * (self.is_usa[self.ia] + self.is_usa[self.ib])[None]
        if self.tech is not None:
            # unilateral climate intervention: rivals in monsoon Asia (assumption)
            if not hasattr(self, "monsoon_pair"):
                ms = torch.as_tensor([c in SOUTH_ASIA + ("CHN",) for c in self.codes], device=dev).float()
                self.monsoon_pair = (ms[self.ia] * ms[self.ib])[None] * torch.nan_to_num(self.static["rival"])[None]
            logit_onset = logit_onset + self.tech.lever("geo_tension") * self.monsoon_pair
        p_on = torch.sigmoid(logit_onset)
        p_cont = self.cont[self.run.clamp(0, 3).long()]
        lgt = lambda q_: torch.logit(torch.as_tensor(q_, device=dev).clamp(1e-6, 1 - 1e-6))
        if self.cfg.ai.arrival:
            # drones and autonomy favour the defence: war-level conflicts last longer
            p_cont = torch.where(self.war > 0.5, torch.sigmoid(lgt(p_cont) + self.prm("drone_persist") * M), p_cont)
        u = torch.rand(S, P, device=dev, generator=self.gen)
        active = torch.where(self.ongoing > 0.5, u < p_cont, u < p_on)
        forced_now = [int(self.pair_index[self.ci[a_], self.ci[b_]]) for a_, b_, y_ in self.cfg.forced
                      if y_ == self.origin + k and a_ in self.ci and b_ in self.ci and int(self.pair_index[self.ci[a_], self.ci[b_]]) >= 0]
        if forced_now:
            fi = torch.as_tensor(forced_now, device=dev)
        # conflicts already recorded for this year (UCDP candidate events after the origin) happen in every world
        obs = [(int(self.pair_index[self.ci[a_], self.ci[b_]]), w_) for a_, b_, y_, w_ in (self.dy or {}).get("observed", ())
               if y_ == self.origin + k and a_ in self.ci and b_ in self.ci and int(self.pair_index[self.ci[a_], self.ci[b_]]) >= 0]
        if obs:
            oi = torch.as_tensor([i for i, _ in obs], device=dev)
            ow = torch.as_tensor([bool(w_) for _, w_ in obs], device=dev)
            active[:, oi] = True
        if forced_now:                        # scenario overrides win over the record
            active[:, fi] = True
        peace_now = [int(self.pair_index[self.ci[a_], self.ci[b_]]) for a_, b_, y_ in self.cfg.peace
                     if y_ == self.origin + k and a_ in self.ci and b_ in self.ci and int(self.pair_index[self.ci[a_], self.ci[b_]]) >= 0]
        if peace_now:
            active[:, torch.as_tensor(peace_now, device=dev)] = False
        new = active & (self.ongoing < 0.5)
        nf_ = self.prm("nuclear_war_factor")
        nuc = 1 + (nf_ - 1) * both_n                                                        # [S, P]
        new_war = new & (torch.rand(S, P, device=dev, generator=self.gen) < self.cfg.war_share * nuc)
        # intensity of continuing conflicts (UCDP 1946-2024): war-level stays war-level with p = 0.71 (n = 24),
        # a minor conflict escalates to war level with p = 0.073 (n = 41)
        u_w = torch.rand(S, P, device=dev, generator=self.gen)
        cont_war = torch.where(self.war > 0.5, u_w < self.cfg.war_persist, u_w < self.cfg.war_escalate * nuc)
        war = torch.where(new, new_war, cont_war) & active
        # a minor conflict escalating to war level is also a war onset (allies may join; markets react)
        new_war = new_war | (war & active & (self.ongoing > 0.5) & (self.war < 0.5))
        if obs:
            war[:, oi] = ow[None].expand(S, -1) & active[:, oi]
            new_war[:, oi] = False           # the recorded participants are known: no simulated allies join that year
        if forced_now:                       # a scenario's war: at war level, a war onset unless already at war
            war[:, fi] = active[:, fi]
            new_war[:, fi] = active[:, fi] & (self.war[:, fi] < 0.5)
            if obs:
                rec = torch.isin(fi, oi)
                new_war[:, fi[rec]] = False
        # alliance cascade: defence-pact allies of each side of a new war-level conflict may join. The random draws
        # come from a separate generator, so a different set of wars in one run does not shift every later draw of
        # the main generator (paired scenario runs stay comparable)
        if new_war.any() and (self.cfg.cascade_q > 0 or "cascade_q" in self.params):
            # first round: formal allies (cascade_q) and security patrons without a treaty (their own q); second round:
            # allies of those who joined, at cascade_q x cascade_hop2 (allies of allies)
            q = self.prm("cascade_q")
            self._last_joiner = (None, None, None)
            j1, cand1 = self._join(new_war, q)
            joined = j1 | self._join_quasi(new_war)
            if obs:
                joined[:, oi] = False        # recorded conflicts keep their recorded level
            if peace_now:
                joined[:, torch.as_tensor(peace_now, device=dev)] = False
            joined &= ~active
            if joined.any():
                # second round: only the allies of those who just joined, and not countries that already had their
                # chance in the first round (a NATO member that stayed out does not get a second roll)
                j2, _ = self._join(joined, q * self.prm("cascade_hop2"), joiners_only=True, exclude=cand1)
                j2 &= ~active & ~joined
                if obs:
                    j2[:, oi] = False
                if peace_now:
                    j2[:, torch.as_tensor(peace_now, device=dev)] = False
                joined |= j2
            active |= joined
            war |= joined
            new = active & (self.ongoing < 0.5)
        nf = new.float()
        self.last_new = new
        self.new_country = torch.zeros_like(self.new_country).index_add_(1, self.ia, nf).index_add_(1, self.ib, nf)
        # new war-level conflicts per country (what moves oil markets: a 25-death clash does not)
        nw = (war & active & (self.war < 0.5)).float()      # new war-level conflicts, including escalations
        self.new_war_country = torch.zeros_like(self.new_country).index_add_(1, self.ia, nw).index_add_(1, self.ib, nw)
        self.last_war = war
        af = active.float()
        self.run = torch.where(active, self.run + 1, torch.zeros_like(self.run))
        # war-site exposure: war-level conflicts on the territory of each side
        hot = war.float()
        Cn = self.war_site.shape[1]
        self.war_site = (torch.zeros(S, Cn, device=dev).index_add_(1, self.ia, hot * self.site_a)
                         .index_add_(1, self.ib, hot * self.site_b)).clamp(max=1)
        # long-range drones erode the sanctuary of expeditionary powers when the enemy is near or a major power
        # (Ukraine's strikes on Russian refineries): an output loss only, kept apart from the war site (no refugees,
        # no trade spillover, no chip-chokepoint effect)
        self.strike_site = torch.zeros(S, Cn, device=dev)
        if self.cfg.ai.arrival:
            reach = self.prm("drone_reach") * M * hot
            ok_a = (self.near | self.is_major_c[self.ib]).float()          # b can reach a
            ok_b = (self.near | self.is_major_c[self.ia]).float()
            self.strike_site = (self.strike_site.index_add_(1, self.ia, reach * (1 - self.site_a) * ok_a)
                                .index_add_(1, self.ib, reach * (1 - self.site_b) * ok_b)).clamp(max=1)
        # battle deaths: war-level years at the war's drawn rate, minor years from UCDP's minor conflict-years
        # an episode is a continuous spell of activity: its rate is drawn when it first reaches war level and kept if it
        # dips to minor and comes back (code review 2026-10-28: a re-escalation used to split one war into two)
        new_ep = war & (self.war_rate == 0)
        if bool(new_ep.any()):
            self.war_rate = torch.where(new_ep, self._draw_rate(), self.war_rate)
        dw = war.float() * self.war_rate
        mi = torch.randint(len(self.ws_minor), (S, P), device=dev, generator=self.gen_size)
        dmin = (active & ~war).float() * self.ws_minor[mi]
        self.ep_total = torch.where(active, self.ep_total + dw, torch.zeros_like(dw))
        self.war_rate = torch.where(active, self.war_rate, torch.zeros_like(self.war_rate))
        self.max_ep = torch.maximum(self.max_ep, self.ep_total.max(-1).values)
        self.rec["battle_deaths"].append((dw + dmin).sum(-1))
        self._consequences(war, active, dw + dmin)
        self.proliferation_step()
        self.rec["war_deaths_major"].append((dw * self.major_pair[None]).sum(-1))
        prev = self.ongoing > 0.5                    # last year's state enters the history of this year's row
        self.hist10 = torch.cat([self.hist10[..., 1:], prev.float()[..., None]], -1)
        self.past10 = self.hist10.sum(-1)
        self.since = torch.where(prev, torch.ones_like(self.since), self.since + 1)
        self.ever = torch.maximum(self.ever, prev.float())
        self.ongoing, self.war = af, war.float()
        for c_, (rho_, b0_, b1_) in self.ev_war.items():
            self.ev_dev[c_] = rho_ * self.ev_dev[c_] + b0_ * af + b1_ * prev.float()
        self.lgdp_hist = torch.cat([self.lgdp_hist[..., 1:], (lg + lpop)[..., None]], -1)
        # records
        self.rec["wars"].append((war.float().sum(-1)))
        self.rec["active"].append(af.sum(-1))
        self.pair_any |= active
        self.rec["great_power_war"].append(((war.float() * self.gp_pair).sum(-1) > 0).float())
        self.rec["great_power_war_cow"].append(((war.float() * self.major_pair).sum(-1) > 0).float())   # COW majors only
        if self.watch:
            idx = torch.as_tensor([i for _, i in self.watch], device=dev)
            self.watch_ever = torch.maximum(self.watch_ever, af[:, idx])
            self.watch_war_ever = torch.maximum(self.watch_war_ever, war.float()[:, idx])
            self.watch_ep_max = torch.maximum(self.watch_ep_max, self.ep_total[:, idx] * war.float()[:, idx])
            if k in (5, 10, 25, 40):
                self.watch_hits = self.watch_hits or {}
                self.watch_hits[k] = self.watch_ever.mean(0).cpu().numpy()
                self.watch_war_hits = self.watch_war_hits or {}
                self.watch_war_hits[k] = self.watch_war_ever.mean(0).cpu().numpy()
        for kk, idx in self.bloc_idx.items():
            if len(idx):
                self.bloc_ever[kk] = torch.maximum(self.bloc_ever[kk], (af[:, idx].sum(-1) > 0).float())
                self.bloc_war_ever[kk] = torch.maximum(self.bloc_war_ever[kk], (war.float()[:, idx].sum(-1) > 0).float())
        # per-country involvement
        inv = torch.zeros(S, lg.shape[1], device=dev)
        inv.index_add_(1, self.ia, af)
        inv.index_add_(1, self.ib, af)
        return (inv > 0).float()

    # ---------------------------------------------------------------- AI
    def update_chip_hit(self):
        """1 in worlds where Taiwan or South Korea is a war site this year (chip chokepoint)"""
        if self.chip_ix and hasattr(self, "war_site"):
            self.chip_hit = (self.war_site[:, self.chip_ix].sum(-1, keepdim=True) > 0).float()

    def adoption(self, year):
        """per-world adoption A [S, C], its change dA, and frontier progress fr [S, 1].
        Two speeds: frontier hubs adopt with little lag; everyone else through open-weight/API diffusion, which
        trails by open_lag years plus a readiness lag; export-controlled states lose export_lag more. Frontier
        progress runs on a clock that slows while Taiwan or South Korea is a war site (chip chokepoint)."""
        ai = self.cfg.ai
        S, C = self.S, self.readiness.shape[0]
        if ai.arrival is None:
            return torch.zeros(S, C, device=self.dev), torch.zeros(S, C, device=self.dev), torch.zeros(S, 1, device=self.dev)
        pr = self.prm
        # frontier clock: slowed by last year's chip-chokepoint war (this year's GDP hit uses this year's wars, set
        # after the interstate step by update_chip_hit)
        self.tau = self.tau + 1 - pr("chip_pause") * self.chip_hit
        ad = ai.adoption_years * pr("speed_mult")
        scale = ad / 4.4                                  # logistic: 10%->90% spans 4.4 scale units
        mid = (self.arrival_w if ai.arrival_cdf else ai.arrival + pr("arrival_sd")) + ad / 2
        lag = (1 - self.hub) * (pr("open_lag") + ai.lag_years * (1 - self.readiness)) + pr("export_lag") * self.controlled
        # logistic curves that start from zero at the arrival year (a plain logistic would already be at 10% then and
        # above zero years before: AI effects before AI arrives)
        s0 = float(torch.sigmoid(torch.tensor(-2.2)))
        cut = lambda z: ((torch.sigmoid(z) - s0) / (1 - s0)).clamp(min=0)
        A = cut((self.tau - (mid + lag)) / scale)
        # robots: deployed over robot_years from the robotics arrival year, reaching less ready countries later (no
        # open-weight shortcut for hardware); military autonomy follows the first of AI and robotics on the faster
        # software clock
        r_ad = pr("robot_years")
        r_lag = (1 - self.hub) * pr("open_lag") + ai.lag_years * (1 - self.readiness)
        # (robot hardware also waits while China, which makes most actuators, batteries and magnets, fights the West)
        # (its own clock, never running backwards when both chokepoints bind)
        self.tau_r = self.tau_r + (1 - pr("chip_pause") * self.chip_hit - self.prm("china_pause") * self.china_hit).clamp(min=0)
        tau_r = self.tau_r
        self.A_robot = cut((tau_r - (self.robot_w + r_ad / 2 + r_lag)) / (r_ad / 4.4))
        self.robot_frontier = cut((tau_r - (self.robot_w + r_ad / 2)) / (r_ad / 4.4))   # [S, 1]
        self.mil_auto = torch.maximum(cut((self.tau - mid) / scale), cut((self.tau - (self.robot_w + ad / 2)) / scale))
        dA = A - self.adopt_prev
        self.adopt_prev = A
        frontier = cut((self.tau - mid) / scale) * torch.ones(S, 1, device=self.dev)
        return A, dA, frontier
