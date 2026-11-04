"""Data-dependency graph of the whole engine: raw sources -> processing -> fitted models and estimates -> simulation
-> validation and benchmarks -> outputs. Layered layout, nodes placed near the mean height of their inputs.
python scripts/draw_dependencies.py  -> docs/data_dependencies.svg
"""
import os
from html import escape
from textwrap import wrap

ROOT = os.path.join(os.path.dirname(__file__), "..")

LAYERS = [("Raw data", "#5b6b82"), ("Processing", "#0f8a8a"), ("Models & estimates", "#7247b8"),
          ("Simulation", "#c2410c"), ("Validation & benchmarks", "#2f7d32"), ("Result files", "#1d5fbf"),
          ("Published", "#334155")]
ASSUME = "#b7791f"

# id: (layer, title, detail, colour override)
N = {
    # raw data
    "owid": (0, "OWID / World Bank / UN series", "GDP (Maddison, WB), population & UN projections, age, fertility, schooling, inequality, gov. spending, tax, military, health, internet, warheads"),
    "vdem": (0, "V-Dem v15", "democracy, liberties, corruption, coups; pre-1946 history"),
    "gmd": (0, "Global Macro Database", "public debt, rates; banking, sovereign-debt and currency crisis dates (Laeven-Valencia)"),
    "epr": (0, "EPR · WRP religion · REIGN", "ethnic power relations, religious composition, leaders"),
    "pol": (0, "Political datasets", "Archigos, GWF, MMP, NAVCO, NELDA, V-Party, WhoGov"),
    "ucdp": (0, "UCDP ACD 26.1 · BRD / NS / OS 25.1", "armed-conflict years, battle deaths, non-state & one-sided violence"),
    "ged": (0, "UCDP GED 26.1 + Candidate monthly", "geolocated events; data vintages as forecasters had them"),
    "events": (0, "Event data: ICEWS, POLECAT, GDELT", "country-year, country-month and dyad event counts (GDELT through 2025)"),
    "cow": (0, "COW, ATOP, MID, rivalries, NMC, ICB, UNGA", "states, majors, contiguity, alliances, disputes, capabilities, crises, UN voting alignment"),
    "cowwar": (0, "COW Inter-State War v4", "battle deaths per war; war winners and losers (revenge test)"),
    "energy": (0, "Energy & geopolitical risk", "WDI fuel trade, Caldara-Iacoviello GPR, EI oil, WB Pink Sheet prices"),
    "food": (0, "Food: FAO balances, grain prices", "calories, imports, food CPI, WB grain index"),
    "mig": (0, "Migration: UNHCR, UN DESA", "refugees by origin-destination, migrant stock, remittances"),
    "trade": (0, "CEPII BACI trade + Gravity", "bilateral trade 2015-2021"),
    "bis": (0, "BIS consolidated banking statistics", "bilateral bank claims 1983-2025"),
    "wb": (0, "World Bank WDI extras", "mineral rents, population below 5 m elevation"),
    "clim": (0, "Climate: ERA5 + CMIP6 (WB CCKP)", "observed and SSP1-2.6…5-8.5 temp, precip, dry spells, heat; BHM damage function"),
    "water": (0, "Water: TFDD basins & events", "shared basins, Basins-at-Risk events; glacier peak water (Wimberly 2025)"),
    "ai": (0, "AI & robotics timelines", "Metaculus, AI Impacts survey; IFR robot counts", ASSUME),
    "ocean": (0, "Ocean food: FAO fisheries (OWID), IFCD", "catch, aquaculture, fish in diets; fishery conflicts 1974-2016"),
    "innov": (0, "Innovation: PWT 11, WDI R&D, OECD MSTI, NSF, Epoch AI", "TFP, R&D, researchers, articles, patents; US frontier since 1929; AI compute"),
    "interv": (0, "UN peacekeeping, OECD/US aid, commodity exports", "UN Data Hub personnel 1990-2026; ODA; 2025 USAID cut; export shares; tariffs (Census, Yale)"),
    "struct": (0, "UCDP dyadic & ESD, EPR-TEK, CCP, SSF, ECI, WDI IDS", "rebel groups, outside support, ethnic kin, term limits, coup-proofing, external debt"),
    "pe": (0, "BoC-BoE defaults, Martinez lights, alliance reliability, IVS", "sovereign defaults, night lights vs GDP, honored alliances, public trust"),
    "us": (0, "US states: BEA, BLS, FHFA, CDC, NOAA…", "income, jobs, housing, overdoses, shootings, elections, climate"),
    "ushist": (0, "US history: USPVD, Voteview, JST", "political violence 1780-, polarisation, macro history"),
    "seshat": (0, "Seshat CrisisDB", "crises of historical polities"),
    "out25": (0, "2025 outcomes", "UCDP 2025, V-Dem v16, GDP 2025: held out for scoring"),
    "bench": (0, "VIEWS Prediction Challenge 2023/24 · ViEWS archive", "actuals, benchmark & participant draws; 48 archived ViEWS runs"),
    "crowd": (0, "Crowd forecasts", "Metaculus / GJOpen / INFER archive; Polymarket & Manifold live"),
    # added after the 29 October 2026 release
    "wpp": (0, "UN WPP 2024 probabilistic projections", "fertility and population uncertainty by country (80% intervals, PPgp/wpp2024)"),
    "wikid": (0, "Wikidata leaders", "birth years of national leaders after 2020 (ages and deaths in office)"),
    "ilo": (0, "ILO occupation shares (DBnomics)", "managers, professionals, technicians, clerks: exposure to near-free AI"),
    "ibtracs": (0, "IBTrACS cyclone tracks", "share of land under hurricane-force winds since 1980"),
    "natearth": (0, "Natural Earth borders (1:10m)", "55 km grid cells mapped to countries"),
    "mort": (0, "Climate mortality: Carleton et al. 2022", "deaths per degree of warming, cold-climate gains, adaptation with income", ASSUME),
    "anchors": (0, "Technology and catastrophe forecasts", "Metaculus, Manifold, GRI quantum timeline, IEA, CEPI, FRI XPT: chances of 17 breakthroughs and of an AI catastrophe", ASSUME),
    "rff": (0, "RFF Socioeconomic Projections", "world GDP to 2065 without climate damage (benchmark)"),
    "rejected": (0, "Tested and not used", "IMF Global Debt Database, PLAD leaders' ethnicity, Penn World Table labour share"),
    # processing
    "panel": (1, "data.py → country-year panel", "~200 countries, 1850-2025: economy, demography, regime, conflict, crises, events, climate"),
    "feat": (1, "features.py", "11-year windows + current state → X; labels for 7 tasks; spatial neighbour weights"),
    "climp": (1, "climate.py", "bias-corrected SSP paths, population-weighted"),
    "dyadb": (1, "interstate.build → dyad-years", "every pair of states from 1850: power, distance, alliances, rivalry, alignment, events, regime"),
    "waterp": (1, "water.py", "pair basins, binding scarcity, glacier-fed share"),
    "risks": (1, "risks.py", "trade weights, distances, correlated weather factors, internationalised wars, sea-level rise"),
    "warsize": (1, "war_size.py", "battle-death pools: major-vs-major, other wars, minor clashes"),
    "foodp": (1, "food.py", "world food price, calories, demand"),
    "migp": (1, "migration.py", "refugee flows and host shares"),
    "dyn": (1, "dynamics.py", "slow variables (debt, education, age, urban…) and party systems"),
    "bisb": (1, "bis_build.py", "bank claims by lender → borrower"),
    "oceanp": (1, "ocean.py", "fish share of calories, wild share, catch value / GDP, latitude band"),
    "newp": (1, "innovation.py · intervention.py · structure.py", "science beyond income, peacekeepers, aid & tariff hits, food-export exposure, rebel groups, short-term debt, language peers"),
    "cmdata": (1, "cm / pgm deaths by vintage", "country-month and 55 km grid-month deaths, final GED + Candidate months"),
    # models & estimates
    "ens": (2, "SeldonEnsemble", "GBM, PsychoNet, Forest, Logit → civil conflict, economic collapse, autocratization, coup (5 y); onset, termination, coup (1 y); structural spread"),
    "trans": (2, "Growth & Δdemocracy transitions", "fitted yearly dynamics; growth noise drawn from countries of similar debt (top third vs rest)"),
    "dyad": (2, "DyadModel (logit + GBM)", "interstate conflict 1 y / 5 y, fatal MID, use of force; alliance cascade calibration"),
    "cont": (2, "contagion_test.py", "trade contagion β = 4.26 ± 0.56; bank-lending link rejected"),
    "extra": (2, "extra_estimates.py", "financial-crisis hazard & cost, mineral rents, fertility vs income, yields vs weather"),
    "assume": (2, "Sampled assumptions", "PARAM / WATER / EXTRA ranges, AI & robot arrival CDFs, nuclear, cyber, tail catastrophes, dollar-decline scenario", ASSUME),
    "cmq": (2, "Fatality quantile models (LightGBM)", "cm: change from 3-month level; pgm: local tail bound; 1,000 draws each"),
    "newtests": (2, "History tests of candidates", "workforce, succession, ocean, innovation, intervention, structure, political economy, ethnic kin, sovereign & currency crises (costs kept, simulation channel rejected), interest rates (untestable), coups after slumps (no overreaction), revenge after lost wars and disputes (rejected): estimates kept, failures rejected"),
    "monthly": (2, "monthly.py", "P(≥25 deaths) per country-month"),
    "side": (2, "US & long-cycle models", "US state crisis models, US structural stress, Seshat cycle tests"),
    # simulation
    "sim": (3, "Monte Carlo world simulator", "simulate.py + world.py, 10,000 worlds 2025→2065 on GPU: country hazards, interstate onsets & alliance cascades, war size, oil, food, water, climate damage, AI & robots, trade contagion, proliferation & nuclear use, tail catastrophes, workforce, leader deaths & succession, peacekeepers, fisheries, tariffs, dynamic great powers"),
    "runs": (3, "Scenario runs", "SSP 1-2.6…5-8.5, AI none / 2027 / 2030 / 2035, 1914 alliances, water low / ref / high"),
    # validation
    "evalv": (4, "Dev / lockbox evaluation", "origins 1975-2005, fresh 2020-23, ablation → model selection"),
    "back": (4, "Backtest & calibration", "origins 1975-2015"),
    "hind": (4, "Hindcasts", "world paths from 1995 / 2004; WWI & WWII retrodiction"),
    "s25": (4, "2025 scorecard", "out-of-sample on 2025 outcomes"),
    "pc": (4, "VIEWS Challenge leaderboards", "cm and pgm, historical + live windows"),
    "arch": (4, "ViEWS archive head-to-head", "Brier & MSE vs 48 runs"),
    "crowdb": (4, "Crowd benchmarks", "archive questions + live markets"),
    "llm": (4, "AI analyst panel", "two independent AI analysts, logged for prospective scoring"),
    # result files
    "fc": (5, "forecast.json", "world, countries, scenarios, AI, interstate, validation"),
    "csv": (5, "country_forecasts.csv", "per-country risks with ranges & drivers"),
    "chron": (5, "chronicles & crisis explorer", "narrated sample worlds"),
    "benchj": (5, "benchmark JSONs", "leaderboards, archive, score_2025, crowd, ledger"),
    "sidej": (5, "us_states / us_history / seshat JSON", ""),
    # published
    "report": (6, "Seldon report (artifact)", "seldon_report_artifact.html"),
    "docs": (6, "Narrative & methodology", "narrative.md, docs/world_system.md, methodology document"),
}

E = [  # (source, target, label or "", dashed)
    ("wpp", "sim"), ("wikid", "panel"), ("ilo", "sim"), ("ibtracs", "sim"), ("natearth", "cmdata"), ("mort", "sim"),
    ("anchors", "assume"), ("rff", "hind"), ("rejected", "newtests"), ("gmd", "extra"), ("gmd", "newtests"), ("cowwar", "newtests"), ("cow", "newtests"),
    ("owid", "panel"), ("vdem", "panel"), ("gmd", "panel"), ("epr", "panel"), ("pol", "panel"), ("ucdp", "panel"),
    ("events", "panel"), ("energy", "panel"), ("food", "panel"), ("mig", "panel"), ("clim", "panel"),
    ("panel", "feat"), ("clim", "climp"),
    ("cow", "dyadb"), ("ucdp", "dyadb"), ("events", "dyadb"), ("panel", "dyadb"), ("waterp", "dyadb"),
    ("water", "waterp"), ("owid", "waterp"),
    ("trade", "risks"), ("wb", "risks"), ("ucdp", "risks"), ("clim", "risks"),
    ("cowwar", "warsize"), ("ucdp", "warsize"),
    ("food", "foodp"), ("energy", "foodp"), ("owid", "foodp"),
    ("mig", "migp"), ("panel", "migp"), ("panel", "dyn"),
    ("bis", "bisb"),
    ("ged", "cmdata"), ("events", "cmdata"), ("panel", "cmdata"),
    # models
    ("feat", "ens"), ("feat", "trans"), ("dyadb", "dyad"),
    ("bisb", "cont"), ("trade", "cont"), ("panel", "cont"),
    ("panel", "extra"), ("wb", "extra"), ("clim", "extra"),
    ("ai", "assume"), ("cmdata", "cmq"), ("ged", "monthly"), ("bench", "monthly"),
    ("ocean", "oceanp"), ("innov", "newp"), ("interv", "newp"), ("struct", "newp"), ("trade", "newp"),
    ("innov", "newtests"), ("interv", "newtests"), ("struct", "newtests"), ("pe", "newtests"), ("ocean", "newtests"),
    ("panel", "newtests"), ("vdem", "newtests"),
    ("oceanp", "sim"), ("newp", "sim"),
    ("us", "side"), ("ushist", "side"), ("seshat", "side"), ("vdem", "side"),
    # simulation
    ("ens", "sim"), ("trans", "sim"), ("dyn", "sim"), ("climp", "sim"), ("dyad", "sim"), ("waterp", "sim"),
    ("risks", "sim"), ("warsize", "sim"), ("foodp", "sim"), ("migp", "sim"), ("cont", "sim"), ("extra", "sim"),
    ("assume", "sim"), ("energy", "sim"), ("sim", "runs"),
    # validation
    ("ens", "evalv"), ("ens", "back"), ("sim", "hind"), ("dyad", "hind"),
    ("ens", "s25"), ("dyad", "s25"), ("monthly", "s25"), ("out25", "s25"),
    ("cmq", "pc"), ("bench", "pc"), ("cmq", "arch"), ("bench", "arch"),
    ("dyad", "crowdb"), ("crowd", "crowdb"), ("sim", "crowdb"), ("sim", "llm"),
    # results
    ("sim", "fc"), ("runs", "fc"), ("dyad", "fc"), ("ens", "fc"), ("back", "fc"), ("hind", "fc"),
    ("ens", "csv"), ("sim", "chron"),
    ("s25", "benchj"), ("pc", "benchj"), ("arch", "benchj"), ("crowdb", "benchj"), ("llm", "benchj"),
    ("side", "sidej"),
    ("fc", "report"), ("csv", "report"), ("chron", "report"), ("benchj", "report"), ("sidej", "report"),
    ("fc", "docs"), ("benchj", "docs"),
]
FEEDBACK = [("newtests", "assume", "estimates & ranges"), ("evalv", "ens", "model selection"), ("cont", "assume", "β → fin_contagion"), ("extra", "risks", "constants")]

BW, COLW, X0, Y0, GAP = 262, 352, 40, 190, 12
TITLE_CH, DET_CH = 34, 44


def size(n):
    t = wrap(N[n][1], TITLE_CH)
    d = wrap(N[n][2], DET_CH) if N[n][2] else []
    return t, d, 14 + 15 * len(t) + 13 * len(d) + (6 if d else 0)


ins = {n: [s for s, t in E if t == n] for n in N}
ORDER = {1: ["panel", "feat", "dyn", "migp", "cmdata", "dyadb", "waterp", "warsize", "foodp", "risks", "bisb", "climp", "oceanp", "newp"],
         2: ["ens", "trans", "dyad", "cmq", "monthly", "cont", "extra", "newtests", "assume", "side"],
         4: ["evalv", "back", "hind", "s25", "crowdb", "llm", "pc", "arch"]}
pos = {}
for _ in range(3):                  # a few passes so nodes fed from their own layer settle after their sources
    for layer in range(len(LAYERS)):
        nodes = ORDER.get(layer) or [n for n in N if N[n][0] == layer]
        if layer == 0:
            y = Y0
            for n in nodes:
                pos[n] = y
                y += size(n)[2] + GAP
            continue
        y = Y0
        for n in nodes:             # pulled towards the mean height of the inputs, order kept, no overlap
            src = [pos[s] + size(s)[2] / 2 for s in ins[n] if s in pos]
            want = sum(src) / len(src) - size(n)[2] / 2 if src else y
            pos[n] = max(want, y)
            y = pos[n] + size(n)[2] + GAP + (8 if layer >= 2 else 0)

H = max(pos[n] + size(n)[2] for n in N) + 140
W = X0 * 2 + COLW * (len(LAYERS) - 1) + BW


def col(n):
    return N[n][3] if len(N[n]) > 3 else LAYERS[N[n][0]][1]


def x(n):
    return X0 + COLW * N[n][0]


# ports spread along each side, ordered by the other end's height
outp, inp = {}, {}
for n in N:
    o = sorted([e for e in E if e[0] == n], key=lambda e: pos[e[1]])
    i = sorted([e for e in E if e[1] == n], key=lambda e: pos[e[0]])
    h = size(n)[2]
    for k, e in enumerate(o):
        outp[e] = pos[n] + h * (k + 1) / (len(o) + 1)
    for k, e in enumerate(i):
        inp[e] = pos[n] + h * (k + 1) / (len(i) + 1)

s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H:.0f}" viewBox="0 0 {W} {H:.0f}" '
     'font-family="Inter, Segoe UI, Helvetica, Arial, sans-serif">',
     '<defs>' + "".join(
         f'<marker id="a{i}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">'
         f'<path d="M0,1 L9,5 L0,9 z" fill="{c}"/></marker>' for i, c in enumerate([l[1] for l in LAYERS] + [ASSUME])) + '</defs>',
     f'<rect width="{W}" height="{H:.0f}" fill="#fbfbf9"/>',
     f'<text x="{X0}" y="52" font-size="28" font-weight="700" fill="#1f2937">Psychohistory engine: data dependencies</text>',
     f'<text x="{X0}" y="80" font-size="14.5" fill="#4b5563">What data goes in, how it is processed and modelled, and which results it produces. '
     'Arrows run left to right; dashed arrows are estimates fed back into earlier stages.</text>']
for i, (name, c) in enumerate(LAYERS):
    cx = X0 + COLW * i
    s.append(f'<text x="{cx}" y="{Y0 - 72}" font-size="15" font-weight="700" fill="{c}" letter-spacing=".4">{escape(name.upper())}</text>')
    s.append(f'<rect x="{cx}" y="{Y0 - 64}" width="{BW}" height="3" rx="1.5" fill="{c}"/>')

palette = [l[1] for l in LAYERS] + [ASSUME]
for e in E:
    a, b = e
    x1, y1 = x(a) + BW, outp[e]
    x2, y2 = x(b) - 2, inp[e]
    c = col(a)
    dx = max(60, (x2 - x1) * 0.45)
    s.append(f'<path d="M{x1},{y1:.1f} C{x1 + dx},{y1:.1f} {x2 - dx},{y2:.1f} {x2},{y2:.1f}" fill="none" stroke="{c}" '
             f'stroke-opacity=".42" stroke-width="1.6" marker-end="url(#a{palette.index(c)})"/>')
for a, b, lab in FEEDBACK:
    ha, hb = size(a)[2], size(b)[2]
    if N[a][0] > N[b][0]:          # backwards: arc over the top from a to b
        x1, y1 = x(a) + BW / 2, pos[a] - 2
        x2, y2 = x(b) + BW * 0.75, pos[b] - 2
        cy = min(y1, y2) - 50
        if pos[a] > Y0 + 40:        # far down the page: leave and enter by the bottom instead
            y1, y2 = pos[a] + ha + 2, pos[b] + hb + 2
            cy = max(y1, y2) + 50
        d = f"M{x1},{y1:.1f} C{x1},{cy:.1f} {x2},{cy:.1f} {x2},{y2:.1f}"
        lx, ly = (x1 + x2) / 2, (cy + min(y1, y2)) / 2 + (-4 if cy < y1 else 14)
    else:                           # same or later layer: bottom of a to top of b
        x1, y1 = x(a) + BW * 0.3, pos[a] + ha
        x2, y2 = x(b) + BW * 0.6, pos[b] - 2
        if y2 < y1:
            x1, y1 = x(a) + BW * 0.3, pos[a]
            y2 = pos[b] + hb + 2
        d = f"M{x1},{y1:.1f} C{x1},{(y1 + y2) / 2:.1f} {x2},{(y1 + y2) / 2:.1f} {x2},{y2:.1f}"
        lx, ly = (x1 + x2) / 2 + 6, (y1 + y2) / 2
    c = col(a)
    s.append(f'<path d="{d}" fill="none" stroke="{c}" stroke-width="1.8" stroke-dasharray="6 4" '
             f'marker-end="url(#a{palette.index(c)})"/>')
    s.append(f'<text x="{lx:.0f}" y="{ly:.0f}" font-size="11.5" font-style="italic" fill="{c}" text-anchor="middle" '
             f'paint-order="stroke" stroke="#fbfbf9" stroke-width="4">{escape(lab)}</text>')

for n in N:
    t, d, h = size(n)
    c = col(n)
    X, Y = x(n), pos[n]
    s.append(f'<g><title>{escape(N[n][1] + ": " + N[n][2])}</title>'
             f'<rect x="{X}" y="{Y:.1f}" width="{BW}" height="{h}" rx="7" fill="#ffffff" stroke="{c}" stroke-width="1.4"/>'
             f'<rect x="{X}" y="{Y:.1f}" width="5" height="{h}" rx="2.5" fill="{c}"/>')
    yy = Y + 19
    for line in t:
        s.append(f'<text x="{X + 14}" y="{yy:.1f}" font-size="13" font-weight="650" fill="#111827">{escape(line)}</text>')
        yy += 15
    yy += 2
    for line in d:
        s.append(f'<text x="{X + 14}" y="{yy:.1f}" font-size="11" fill="#4b5563">{escape(line)}</text>')
        yy += 13
    s.append('</g>')

ly = H - 70
items = [(l[0], l[1]) for l in LAYERS] + [("Sampled assumptions (not estimated)", ASSUME)]
xx = X0
for name, c in items:
    s.append(f'<rect x="{xx}" y="{ly}" width="14" height="14" rx="3" fill="#fff" stroke="{c}" stroke-width="2"/>'
             f'<rect x="{xx}" y="{ly}" width="4" height="14" rx="2" fill="{c}"/>'
             f'<text x="{xx + 22}" y="{ly + 11.5}" font-size="12.5" fill="#374151">{escape(name)}</text>')
    xx += 30 + 7.2 * len(name)
s.append(f'<text x="{X0}" y="{ly + 40}" font-size="11.5" fill="#6b7280">Arrow colour follows the source node. '
         'Generated by scripts/draw_dependencies.py from the code paths in psychohistory/ and scripts/.</text>')
s.append('</svg>')
out = os.path.join(ROOT, "docs", "data_dependencies.svg")
open(out, "w").write("\n".join(s))
print(out, W, round(H))
