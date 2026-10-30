# Data licences

The code in this repository is the authors' own. The data it uses come from many providers, each with its own
terms. The raw data release (GitHub release `data-2026-10-29`, restored with `bash scripts/fetch_data.sh`) contains
only sources whose terms allow public redistribution. Sources whose terms forbid it, or whose terms could not be
verified, are **not included**; they are listed at the end with their download pages, so you can fetch them yourself
(the build scripts expect them at the paths given). The exact exclusion list is `scripts/data_release_exclude.txt`.

This review (October 2026) is a good-faith reading of each provider's published terms, not legal advice. If you are a
data provider and believe a file should not be here, please open an issue.

The processed files in `data/` (the country-year panel, the pair table) and the results in `outputs/` are derived,
transformed data built from all sources, including some that are not redistributed in raw form.

## Included, with the attribution each licence requires

**CC BY 4.0** (attribution)
- World Bank: World Development Indicators, International Debt Statistics, Commodity Price Data (Pink Sheet),
  Climate Change Knowledge Portal (ERA5 observations, CMIP6 projections).
- UCDP/PRIO Armed Conflict, Battle-Related Deaths, Non-State, One-Sided, Dyadic, GED and Candidate datasets
  (Gleditsch et al. 2002; Davies et al.; Sundberg & Melander 2013; Hegre et al. 2020).
- FAO / FAOSTAT and the FAO Food Price Index.
- UNHCR Refugee Population Statistics.
- Penn World Table 11.0 and 10.01 (Feenstra, Inklaar & Timmer 2015).
- Epoch AI data.
- WhoGov (Nyrup & Bramwell).
- Wand & Hoyer 2023 (Seshat).
- Känzig (2021) oil supply news shocks.
- OECD (MSTI, DAC2A, MEI).
- ERA5: "Contains modified Copernicus Climate Change Service information; ERA5 © ECMWF, licensed CC-BY-4.0".
- Data Commons (with the underlying FBI UCR and Census ACS).
- Our World in Data exports, with each chart's original source: Maddison Project, UN World Population Prospects,
  V-Dem, World Inequality Database, UNU-WIDER, SIPRI (via OWID), Energy Institute Statistical Review of World Energy
  2026, IFR via the AI Index (CC BY-ND 4.0: files shipped unmodified), World Happiness Report, UNOOSA and others.

**CC BY 3.0 IGO**: UN DESA World Population Prospects 2024; UN DESA International Migrant Stock 2024.

**CC BY-SA 4.0** (share-alike applies to derived tables): V-Dem v15/v16 and V-Party v2 (Coppedge et al.;
Lindberg et al.); ISO-3166 country list (lukes).

**CC BY-NC-SA 4.0** (non-commercial, share-alike): Global Macro Database (Müller et al. 2025); EPR Core 2021 and
EPR-TEK (Vogt et al. 2015); JST Macrohistory R6 (Jordà, Schularick & Taylor 2017); Seshat Databank Equinox, Polaris,
API extracts and US Political Violence Database (Turchin et al. 2015; Turchin 2012).

**CC BY-NC 4.0**: International Fishery Conflict Database (Spijkers et al. 2019, doi:10.25903/5f22492a64642).

**Other open terms**
- IMF World Economic Outlook: "Source: International Monetary Fund, World Economic Outlook Database, October 2026"
  (values transformed; commercial reuse needs IMF permission).
- BIS consolidated banking statistics: "Source: BIS consolidated banking statistics" (no endorsement implied).
- Bank of Canada – Bank of England Sovereign Default Database (Beers, Ndukwe & Berry 2026), courtesy of the Bank of
  Canada.
- CEPII Gravity V202211, GeoDist and BACI (Conte, Cotterlaz & Mayer 2022): Etalab Open Licence 2.0.
- AidData Global Chinese Development Finance 3.0: ODC-By.
- GDELT 1.0 events: "The GDELT Project", https://www.gdeltproject.org/ (citation and link required).
- world-atlas and us-atlas (Natural Earth, US Census): ISC; world-countries-centroids (gavinr): MIT;
  Wimberly et al. 2025 glacier-runoff notebooks: GPL-3.0.
- Transboundary Freshwater Dispute Database and Basins at Risk events (Oregon State University): credit requested.
- US federal statistics (public domain, citation requested): BEA, BLS, FHFA, CDC, NOAA nClimDiv, Census, NCES,
  NSF/NCSES, ForeignAssistance.gov; series retrieved from FRED, Federal Reserve Bank of St. Louis (Cleveland Fed
  series: Federal Reserve Bank of Cleveland).

**CC0** (citation appreciated): POLECAT (Scarborough et al. 2023); UNGA ideal points (Bailey, Strezhnev & Voeten
2017); Mass Mobilization; NAVCO 1.3/2.1; International Sanctions Termination (Attia & Grauvogel); Fearon & Laitin 2003;
Martinez 2022 replication package (contains Freedom House data); alliance reliability data (Leeds et al. 2000; Leeds
2003; Chiba, Johnson & Leeds 2026; LeVeck & Narang 2017); Atlas of Economic Complexity (Harvard Growth Lab); MIT
Election Lab; Shor–McCarty; Boxell, Gentzkow & Shapiro 2024; Seshat CrisisDB consequences (Hoyer et al. 2023) and Qing
data (Orlandi et al. 2023).

## Added after the data-2026-10-29 release (not in that archive; download pages below)

Used by the 2026-10-29 evening round (docs/world_system.md). The derived files in `data/` built from them are tracked.

| Path under data/raw | Source | Licence | Download | Derived file |
|---|---|---|---|---|
| `demography/wpp2024_prob/` | UN DESA WPP 2024 probabilistic projections (`popproj1dt`, `tfrproj1dt`, `countries` from the wpp2024 R package) | CC BY 3.0 IGO | https://github.com/PPgp/wpp2024 | `data/wpp_prob.npz` |
| `politics/leaders_wikidata.csv` | Wikidata birth dates of heads of government / state (scripts/fetch_leader_ages.py) | CC0 | https://www.wikidata.org/ | `data/leader_birth_wd.npz` |
| `tech/ilo_occupations/` | ILO modelled estimates, employment by occupation (ISCO-08), via DBnomics | CC BY 4.0 | https://db.nomics.world/ILO | `data/ai_exposure.npz` |
| `climate/ibtracs/` | NOAA IBTrACS v04r01 tropical cyclone tracks since 1980 (Knapp et al. 2010) | public domain (US government) | https://www.ncei.noaa.gov/products/international-best-track-archive | `data/cyclone_exposure.npz` |
| `finance_gdd/` | IMF Global Debt Database (tested, not used) | IMF terms (as the WEO above) | https://www.imf.org/external/datamapper/datasets/GDD | — |
| `politics/plad/` | Political Leaders' Affiliation Database (Dreher, Fuchs, Hodler, Parks, Raschky & Tierney 2024; tested, not used) | CC0 | https://doi.org/10.7910/DVN/YUS575 | — |
| `benchmarks/rffsps/` | RFF Socioeconomic Projections v5 (Rennert et al. 2022; benchmark only) | CC BY 4.0 | https://zenodo.org/records/6016583 | `outputs/rff_benchmark.json` |

## Not included: fetch these yourself

Terms forbid redistribution:

| Path under data/raw | Source | Download |
|---|---|---|
| `WRP_national.csv`, `wars/cow_interstate_v4.csv`, `system/cow_*.rda` | Correlates of War (World Religion Project, Inter-State War, states, majors, contiguity, capabilities, trade) | https://correlatesofwar.org/data-sets/ (the `.rda` files come with the peacesciencer R package) |
| `newdata/polity/` | Polity5 | https://www.systemicpeace.org/inscrdata.html |
| `interstate/icow/`, `interstate/icow_dy.csv` | ICOW territorial claims | https://www.paulhensel.org/icowterr.html |
| `intervention/sipri_milex*` | SIPRI Military Expenditure | https://www.sipri.org/databases/milex |
| `events/raw/icews/`, `events/raw/icews_events_slim.parquet` | ICEWS coded events (research use only) | https://doi.org/10.7910/DVN/28075 |
| `us_history/measuringworth/`, `us_history/elite_annual.csv` | MeasuringWorth US GDP and wages | https://www.measuringworth.com/datasets/usgdp/ , https://www.measuringworth.com/datasets/uswage/ |
| `us_history/anes/pew/`, `newdata/religion/pew_2010_2050.xlsx` | Pew Research Center | https://www.pewresearch.org/ |
| `benchmarks/` crowd archives | Metaculus, Good Judgment Open, INFER question archives (platform terms) | https://people.eecs.berkeley.edu/~hendrycks/autocast.tar.gz ; https://huggingface.co/datasets/YuehHanChen/forecasting_raw ; https://huggingface.co/datasets/forecastingresearch/forecastbench-datasets ; then `data/raw/benchmarks/scripts/` |
| papers and reports (`tech/way2022/*.pdf`, `us_history/elite/2024-aba-nlps.pdf`, `innovation/frontier/bjvw/`) | publisher copyright | linked from the folder READMEs |

No licence could be verified (free to download, citation requested; excluded to be safe):

| Path under data/raw | Source | Download |
|---|---|---|
| `REIGN_2021_8.csv` | REIGN (OEF Research) | https://github.com/OEFDataScience/REIGN.github.io |
| `politics/archigos*`, `politics/gwf*`, `politics/nelda*` | Archigos 4.1; Geddes–Wright–Frantz; NELDA | https://www.rochester.edu/college/faculty/hgoemans/data.htm ; https://sites.psu.edu/dictators/ ; https://nelda.co/ |
| `system/atop_alliance.rda`, `system/gml_mid_*.rda`, `system/td_rivalries.rda`, `system/terrthreat.rda`, `system/rugged.rda`, `system/nunn/` | ATOP, GML MID, Thompson–Dreyer rivalries, territorial threat, Nunn–Puga ruggedness | peacesciencer (https://github.com/svmiller/peacesciencer); https://diegopuga.org/data/rugged/ |
| `interstate/icb*` | International Crisis Behavior | https://sites.duke.edu/icbdata/data-collections/ |
| `climate/bhm/` | Burke–Hsiang–Miguel damage projections | https://web.stanford.edu/~mburke/climate/data/ |
| `energy/rr*`, `political_economy/reinhart_rogoff*` | Reinhart–Rogoff crises (HBS) | https://www.hbs.edu/behavioral-finance-and-financial-stability/data/Pages/global.aspx |
| `energy/gpr*` | Caldara–Iacoviello Geopolitical Risk index | https://www.matteoiacoviello.com/gpr.htm |
| `energy/oil_shocks/BH2_*.xlsx`, `market-based_oil_price_shocks.xlsx` | Baumeister–Hamilton oil shocks | https://sites.google.com/site/cjsbaumeister/datasets |
| `events/icews_*.csv`, `events/splice_*.csv` | aggregates of ICEWS counts | rebuilt by `data/raw/events/build_events.py` from ICEWS |
| `innovation/frontier/fernald/`, `innovation/us_tfp_fernald_quarterly.csv` | FRBSF Fernald TFP | https://www.frbsf.org/research-and-insights/data-and-indicators/total-factor-productivity-tfp/ |
| `intervention/un_psdata/`, `intervention/un_pko_*` | UN Peace and Security Data Hub | https://psdata.un.org/ |
| `intervention/ocindex*` | Global Organized Crime Index | https://ocindex.net/downloads |
| `intervention/cgd*`, `intervention/us_aid_cut_2025.csv` | Center for Global Development USAID cut estimates | https://www.cgdev.org/blog/usaid-cuts-new-estimates-country-level |
| `newdata/clio/`, `newdata/imf/` | Clio-Infra; IMF Monitor programmes | https://clio-infra.eu ; http://imfmonitor.org |
| `outcomes/views*`, `predcomp/` | ViEWS forecasts API; VIEWS Prediction Challenge 2023/24 replication data | https://api.viewsforecasting.org/ ; https://viewsforecasting.org/ |
| `political_economy/tariffs/`, `political_economy/us_tariff_etr_budgetlab_*` | Yale Budget Lab tariff tracker | https://budgetlab.yale.edu/ |
| `structure/ccp*`, `structure/ties*`, `structure/sanctions_ties_country_year.csv`, `structure/ssf*`, `structure/ewn*` | Comparative Constitutions Project; TIES; State Security Forces; External Wealth of Nations | https://comparativeconstitutionsproject.org/download-data/ ; https://sanctions.web.unc.edu/ ; https://www.prio.org/journals/jpr/replicationdata ; https://www.brookings.edu/articles/the-external-wealth-of-nations-database/ |
| `tech/chat*`, `tech/robots_cy.csv`, `tech/way2022_parameters.csv` | CHAT (Comin–Hobijn); IFR robots (derived, CC BY-ND upstream); Way et al. 2022 parameters | https://data.nber.org/data-appendix/w15319/ ; https://ourworldindata.org/ ; Joule 2022 SI |
| `us/motherjones_mass_shootings.csv` | Mother Jones mass shootings | https://www.motherjones.com/politics/2012/12/mass-shootings-mother-jones-full-data/ |
| `us_history/voteview*`, `us_history/grumbach*`, `us_history/brightline*`, `us_history/anes_annual.csv` | Voteview; State Democracy Index; Bright Line Watch; trust series | https://voteview.com/data ; https://democracypolicylab.berkeley.edu/state-democracy-index/ ; https://brightlinewatch.org/ |
