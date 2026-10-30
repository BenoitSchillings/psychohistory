"""Birth years of leaders in office after REIGN ends (August 2021), from Wikidata (CC0).

The effective leader each year is V-Dem's (head of state if the more powerful office, else head of government; as in
psychohistory.data.load_vdem). For every spell in office starting in 2021-2025 (a returning leader counts again), search Wikidata by name and keep the first
candidate that is a human (P31 Q5) with a birth date (P569) and a citizenship (P27) or position (P39) linked to that
country; unmatched names stay missing.
python -P scripts/fetch_leader_ages.py -> data/raw/politics/leaders_wikidata.csv (then scripts/build_leader_ages.py)"""
import json
import os
import time
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

RAW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "raw")
HERE = os.path.join(RAW, "politics")
UA = {"User-Agent": "psychohistory-research/1.0 (leader birth years; https://github.com/BenoitSchillings/psychohistory)"}
API = "https://www.wikidata.org/w/api.php"


def get(params):
    url = API + "?" + urllib.parse.urlencode({**params, "format": "json"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
                return json.load(r)
        except Exception:
            time.sleep(2 + 3 * attempt)
    return {}


def country_qids(codes):
    """ISO alpha-3 -> Wikidata item of the country (P298)"""
    out = {}
    for i in range(0, len(codes), 40):
        q = "SELECT ?c ?iso WHERE { ?c wdt:P298 ?iso . VALUES ?iso { %s } }" % " ".join(f'"{c}"' for c in codes[i:i + 40])
        url = "https://query.wikidata.org/sparql?" + urllib.parse.urlencode({"query": q, "format": "json"})
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
            for b in json.load(r)["results"]["bindings"]:
                out.setdefault(b["iso"]["value"], set()).add(b["c"]["value"].rsplit("/", 1)[1])
        time.sleep(1)
    return out


def claim_ids(ent, prop):
    return {c["mainsnak"].get("datavalue", {}).get("value", {}).get("id") for c in ent.get("claims", {}).get(prop, [])}


def main():
    d = pd.read_parquet(os.path.join(RAW, "vdem_extra_v3.parquet"))
    d = d.rename(columns={"country_text_id": "code"}) if "country_text_id" in d else d
    d = d.sort_values(["code", "year"])
    use_hog = (d.v2ex_hosw < 0.5) & d.v2exnamhog.notna() & (d.v2exnamhog.astype(str).str.len() > 0)
    d["leader"] = np.where(use_hog, d.v2exnamhog, d.v2exnamhos)
    d = d[d.leader.notna() & (d.leader.astype(str).str.len() > 0)]
    # spells in office: a new spell whenever the effective leader changes (a returning leader counts again)
    d["spell"] = (d.leader != d.groupby("code").leader.shift()).astype(int).groupby(d.code).cumsum()
    first = d.groupby(["code", "spell"]).agg(leader=("leader", "first"), year=("year", "min")).reset_index()
    new = first[first.year >= 2021].drop_duplicates(["code", "leader"]).sort_values(["code", "year"])
    old = os.path.join(HERE, "leaders_wikidata.csv")
    done = pd.read_csv(old).set_index(["code", "leader"]) if os.path.exists(old) else None
    cq = country_qids(sorted(new.code.unique()))
    rows = []
    for r in new.itertuples():
        if done is not None and (r.code, r.leader) in done.index and pd.notna(done.loc[(r.code, r.leader), "birth_year"]):
            o = done.loc[(r.code, r.leader)]
            rows.append({"code": r.code, "leader": r.leader, "first_year": int(r.year), "qid": o.qid,
                         "birth_year": int(o.birth_year), "description": o.description})
            continue
        name = str(r.leader).split("(")[0].strip()
        hits = get({"action": "wbsearchentities", "search": name, "language": "en", "type": "item", "limit": 7}).get("search", [])
        ids = [h["id"] for h in hits]
        best = None
        if ids:
            ents = get({"action": "wbgetentities", "ids": "|".join(ids), "props": "claims|descriptions"}).get("entities", {})
            for q in ids:
                e = ents.get(q, {})
                if "Q5" not in claim_ids(e, "P31") or not e.get("claims", {}).get("P569"):
                    continue
                linked = (claim_ids(e, "P27") | claim_ids(e, "P1001")) & cq.get(r.code, set())
                if not linked:
                    continue
                t = e["claims"]["P569"][0]["mainsnak"].get("datavalue", {}).get("value", {}).get("time", "")
                if len(t) > 5:
                    best = (q, int(t[1:5]), e.get("descriptions", {}).get("en", {}).get("value", ""))
                    break
        rows.append({"code": r.code, "leader": r.leader, "first_year": int(r.year), "qid": best[0] if best else None,
                     "birth_year": best[1] if best else None, "description": best[2] if best else None})
        print(rows[-1], flush=True)
        time.sleep(0.3)
    # names Wikidata's search does not find under V-Dem's spelling (birth years from the same Wikidata items)
    manual = {("AFG", "Mawlawi Hibatullah Akhundzada"): ("Q24237654", 1967), ("PAK", "Mian Mohammad Shehbaz Sharif"): ("Q2096387", 1951),
              ("HKG", "John Lee Ka-chiu"): ("Q9051824", 1957), ("KWT", "Mishal Al-Ahmad Al-Jaber Al-Sabah"): ("Q12243161", 1940),
              ("XKX", "Albin Kurti"): ("Q441633", 1975)}
    for r in rows:
        if r["birth_year"] is None and (r["code"], r["leader"]) in manual:
            r["qid"], r["birth_year"] = manual[(r["code"], r["leader"])]
            r["description"] = "(matched by hand)"
    pd.DataFrame(rows).to_csv(os.path.join(HERE, "leaders_wikidata.csv"), index=False)


if __name__ == "__main__":
    main()
