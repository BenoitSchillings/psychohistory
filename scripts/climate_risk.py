"""Climate risk in the headline worlds, for the report: damage families, harvest failures (against 1961-2024),
tipping points, drylands, climate emigration and the climate-conflict assumption; with the 2100 view
(climate_2100.json), the climate-war test (climate_war_test.json) and the uncertainty attribution.
python -P scripts/climate_risk.py -> outputs/climate_risk.json"""
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

import disaster_rca as D
from psychohistory import pipeline as PL
from psychohistory.food import yield_residuals

FAMILIES = ("Level effect (Newell et al. 2021)", "Growth effect (Burke, Hsiang & Miguel 2015)",
            "Global temperature (Bilal & Känzig 2024)")
X4 = ("USA", "BRA", "ARG", "UKR")


def history(p):
    """frequency of harvest failures 1961-2024 from the same residuals the simulator replays"""
    r = yield_residuals(p, 2025)
    yrs = np.asarray(p.years)
    m = (yrs >= 1961) & (yrs <= 2024)
    idx = {c: i for i, c in enumerate(p.codes)}
    us = r[idx["USA"], m]
    x4 = np.stack([r[idx[c], m] for c in X4 if c in idx])
    bad = (x4 < -0.10).sum(0)
    return {"us_bad15": round(float(np.nanmean(us < -0.15)), 3), "x4_two": round(float(np.mean(bad >= 2)), 3),
            "x4_all": round(float(np.mean(bad >= 4)), 3), "years": int(m.sum()),
            "us_worst": sorted(((int(y), round(float(v), 3)) for y, v in zip(yrs[m], us) if np.isfinite(v)), key=lambda t: t[1])[:4]}


def main():
    env = D.setup()
    p = env[0]
    D.N = 4000
    o = D.run(env, 300, record=False)
    w = o["world"]
    P = w["params"]
    yrs = list(o["years"])
    gdp = o["world_gdp"][:, -1]
    fam = np.asarray(P["dmg_family"]).ravel().astype(int)
    out = {"n": D.N, "history": history(p)}
    out["families"] = [{"name": n, "share": round(float((fam == f).mean()), 3), "gdp_2065": round(float(np.median(gdp[fam == f])))}
                       for f, n in enumerate(FAMILIES)]
    out["gdp_2065"] = round(float(np.median(gdp)))
    hu, b4 = w["harvest_us"], w["harvest_bad4"]
    first, last = slice(0, 10), slice(len(yrs) - 10, len(yrs))
    out["harvest"] = {"us_bad15_early": round(float((hu[:, first] < -0.15).mean()), 3),
                      "us_bad15_late": round(float((hu[:, last] < -0.15).mean()), 3),
                      "us_bad25_any": round(float((hu < -0.25).any(1).mean()), 3),
                      "x4_two_early": round(float((b4[:, first] >= 2).mean()), 3),
                      "x4_two_late": round(float((b4[:, last] >= 2).mean()), 3),
                      "x4_all_any": round(float((b4 >= 4).any(1).mean()), 3),
                      "food_spike_any": round(float((np.diff(w["food_price"], axis=1) > np.log(1.5)).any(1).mean()), 3)}
    out["amoc_share"] = round(float(w["amoc"][:, -1].mean()), 3)
    out["slr_ice_share"] = round(float(np.mean(np.asarray(P["slr_ice"]).ravel())), 3) if "slr_ice" in P else None
    cm = w["clim_migrants"][:, -1]
    out["clim_migrants_m"] = {"median": round(float(np.median(cm)), 1), "p95": round(float(np.quantile(cm, .95)), 1)}
    cc = np.asarray(P["clim_conflict"]).ravel()
    conf = o["world_conflict"][:, -1]
    q = np.quantile(cc, [0.2, 0.8])
    out["clim_conflict"] = {"low": round(float(np.median(conf[cc <= q[0]])), 4), "high": round(float(np.median(conf[cc >= q[1]])), 4)}
    for k, fn in (("y2100", "climate_2100.json"), ("war_test", "climate_war_test.json")):
        f = PL.OUT / fn
        out[k] = json.loads(f.read_text()) if f.exists() else None
    att = PL.OUT / "uncertainty_attribution.json"
    if att.exists():
        a = json.loads(att.read_text())
        out["attribution"] = {k: v.get("groups", {}) for k, v in a.items()}
    json.dump(out, open(PL.OUT / "climate_risk.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k not in ("war_test", "y2100")}, indent=1))


if __name__ == "__main__":
    main()
