"""US state-year series from Data Commons (api.datacommons.org v2; key in DATACOMMONS_API_KEY), for the state module.
For each variable the facet (source) with the most years is kept. Rates are built here:
  violent_crime, property_crime, murder   FBI UCR counts per 100,000 people (2008-2023)
  bachelors_share                          people 25+ with a bachelor's degree or more / people 25+ (ACS, ~2010-2024)
  poverty_rate                             people below the poverty line / people (ACS)
  median_age                               ACS
  foreign_born_share                       foreign-born people / people (ACS)
  uninsured_share                          people without health insurance / people (ACS)
python -P scripts/dc_states_fetch.py -> data/raw/us/datacommons_states.csv (+ datacommons_states_sources.json)"""
import json
import os
import sys
import urllib.request

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from psychohistory.us_states import STATES

OUT = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "us")
VARS = ["Count_CriminalActivities_ViolentCrime", "Count_CriminalActivities_PropertyCrime",
        "Count_CriminalActivities_MurderAndNonNegligentManslaughter", "Count_Person_EducationalAttainmentBachelorsDegreeOrHigher",
        "Count_Person_25OrMoreYears", "Count_Person_BelowPovertyLevelInThePast12Months", "Median_Age_Person",
        "Count_Person_ForeignBorn", "Count_Person_NoHealthInsurance", "Count_Person"]


def fetch(variables, entities):
    body = json.dumps({"date": "", "variable": {"dcids": variables}, "entity": {"dcids": entities},
                       "select": ["entity", "variable", "value", "date"]}).encode()
    req = urllib.request.Request(f"https://api.datacommons.org/v2/observation?key={os.environ['DATACOMMONS_API_KEY']}",
                                 data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)


def main():
    geo = {f"geoId/{fips:02d}": code for code, fips, *_ in STATES}
    d = fetch(VARS, list(geo))
    facets = d.get("facets", {})
    rows, src = [], {}
    for v, x in d["byVariable"].items():
        for e, o in x.get("byEntity", {}).items():
            # annual observations only; the facet with the most years
            best = max(o.get("orderedFacets", []), default=None,
                       key=lambda f: sum(len(ob["date"]) == 4 for ob in f["observations"]))
            if best is None:
                continue
            src.setdefault(v, set()).add(json.dumps(facets.get(best["facetId"], {}), sort_keys=True))
            for ob in best["observations"]:
                if len(ob["date"]) == 4:
                    rows.append({"code": geo[e], "year": int(ob["date"]), "var": v, "value": float(ob["value"])})
    w = pd.DataFrame(rows).pivot_table(index=["code", "year"], columns="var", values="value").reset_index()
    pop = w["Count_Person"]
    out = pd.DataFrame({"code": w.code, "year": w.year,
                        "violent_crime": w["Count_CriminalActivities_ViolentCrime"] / pop * 1e5,
                        "property_crime": w["Count_CriminalActivities_PropertyCrime"] / pop * 1e5,
                        "murder": w["Count_CriminalActivities_MurderAndNonNegligentManslaughter"] / pop * 1e5,
                        "bachelors_share": w["Count_Person_EducationalAttainmentBachelorsDegreeOrHigher"] / w["Count_Person_25OrMoreYears"],
                        "poverty_rate": w["Count_Person_BelowPovertyLevelInThePast12Months"] / pop,
                        "median_age": w["Median_Age_Person"],
                        "foreign_born_share": w["Count_Person_ForeignBorn"] / pop,
                        "uninsured_share": w["Count_Person_NoHealthInsurance"] / pop})
    out = out.dropna(how="all", subset=out.columns[2:]).round(5)
    out.to_csv(os.path.join(OUT, "datacommons_states.csv"), index=False)
    json.dump({k: sorted(v) for k, v in src.items()}, open(os.path.join(OUT, "datacommons_states_sources.json"), "w"), indent=1)
    print(out.groupby("year").count().iloc[:, 1:].to_string())


if __name__ == "__main__":
    main()
