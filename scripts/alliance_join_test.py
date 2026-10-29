"""When do defence allies join a war? COW Inter-State War v4.0 participants x ATOP defence pacts (start year).

An opportunity: a defence ally c of an initial belligerent i (a participant whose entry is within 30 days of the
war's start), c not itself an initial belligerent and not allied with i's enemies. Joined: c enters later on i's side.
Logit of joining on the regime gap |dem_i - dem_c| (V-Dem electoral democracy), c a major power, c within
2,500 km of the enemy (capital distance) and era (after 1945). Writes outputs/alliance_join_test.json.
"""
import json, os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, pandas as pd
import statsmodels.api as sm
from psychohistory import interstate as I
from psychohistory.data import RAW, ROOT, get_panel
from psychohistory.features import base_arrays

p = get_panel(); b = base_arrays(p)
c2i = I._ccode_to_iso(set(p.codes))
w = pd.read_csv(RAW / "wars" / "cow_interstate_v4.csv")
w["start"] = pd.to_datetime(dict(year=w.StartYear1, month=w.StartMonth1.clip(1, 12), day=w.StartDay1.clip(1, 28)), errors="coerce")
atop = I._rda("atop_alliance"); atop = atop[atop.atop_defense == 1]
al = {}
for a_, b_, y in zip(atop.ccode1, atop.ccode2, atop.year):
    al.setdefault(int(y), set()).update({(int(a_), int(b_)), (int(b_), int(a_))})
capt = I._rda("cow_capitals"); cap = {int(c): (la, lo) for c, la, lo in zip(capt.ccode, capt.lat, capt.lng)}
def dist(a, b_):
    if a not in cap or b_ not in cap: return np.nan
    (la1, lo1), (la2, lo2) = map(np.radians, (cap[a], cap[b_]))
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(h))
majors = I._majors(c2i)
is_major = lambda iso, y: any(s <= y <= e for s, e in majors.get(iso, []))
dem = lambda c, y: b["dem"][p.codes.index(c2i[c]), p.yi(y)] if c in c2i and c2i[c] in p.codes and p.years[0] <= y <= p.years[-1] else np.nan
rows = []
for wn, s in w.groupby("WarNum"):
    t0 = s.start.min(); y0 = int(s.StartYear1.min())
    init = s[s.start <= t0 + pd.Timedelta(days=30)]
    later = s[s.start > t0 + pd.Timedelta(days=30)]
    pacts = al.get(min(y0, max(al)), set())
    for _, r in init.iterrows():
        i, side = int(r.ccode), r.Side
        enemies = set(init[init.Side != side].ccode.astype(int))
        for (x, c) in pacts:
            if x != i or c in set(init.ccode.astype(int)) or any((c, e) in pacts for e in enemies):
                continue
            joined = bool(((later.ccode == c) & (later.Side == side)).any())
            dmin = np.nanmin([dist(c, e) for e in enemies]) if enemies else np.nan
            rows.append({"war": wn, "year": y0, "i": i, "c": c, "joined": joined, "dem_gap": abs(dem(i, y0) - dem(c, y0)),
                         "dem_c": dem(c, y0), "major_c": float(c in c2i and is_major(c2i[c], y0)), "near": float(dmin < 2500),
                         "post45": float(y0 > 1945)})
d = pd.DataFrame(rows).drop_duplicates(["war", "i", "c"])
out = {"n": len(d), "join_rate": float(d.joined.mean()), "wars": int(d.war.nunique())}
dd = d.dropna(subset=["dem_gap"])
X = sm.add_constant(dd[["dem_gap", "major_c", "near", "post45"]])
m = sm.Logit(dd.joined.astype(float), X).fit(disp=0, cov_type="cluster", cov_kwds={"groups": dd.war})
out["logit"] = {k: {"coef": round(float(m.params[k]), 3), "se": round(float(m.bse[k]), 3)} for k in X.columns}
out["by_gap"] = dd.groupby(pd.cut(dd.dem_gap, [-.01, .2, .4, 1])).joined.agg(["mean", "size"]).reset_index().astype(str).to_dict("records")
out["near_or_major"] = d[(d.near == 1) | (d.major_c == 1)].joined.agg(["mean", "size"]).to_dict()
out["near_or_major_post45"] = d[((d.near == 1) | (d.major_c == 1)) & (d.post45 == 1)].joined.agg(["mean", "size"]).to_dict()
print(json.dumps(out, indent=1, default=float))
json.dump(out, open(ROOT / "outputs" / "alliance_join_test.json", "w"), indent=1, default=float)
