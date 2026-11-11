"""Does the simulator share the SSP scenarios' convergence bias? (11 November 2026)

Burgess et al. (2021): for 2005-2017 every SSP baseline over-projected growth of GDP per person in every region, by
about 1 to 3 points a year in developing regions: poor countries were assumed to catch up faster than they did.
Written and committed before its run; a diagnostic of the hindcasts, it adopts nothing.

Design. The adopt rule's hindcast files (16 origins 1970-2009, 15 years) of tag TAG that store each country's
simulated growth quantiles (scripts/world_hindcast.py from 11 November 2026). Error = (simulated median - realised) log
growth of GDP per person, per year, in points. Countries are split at each origin into thirds by income at the origin.
Intervals: 1,000 bootstrap resamples of origins.
Baselines on the same country-windows: the country's own growth over the 15 years before the origin carried forward;
and a convergence line (growth on log income, fitted on the 15 years before the origin) applied to the origin's incomes.

Pre-registered reading.
  "Shares the bias" if the poorest third's mean error is at least +0.5 points a year with a 90% interval above 0, and
  exceeds the richest third's by at least 0.5 with an interval above 0.
  "No convergence bias" if the poorest third's interval includes 0 and the poor-rich gap's does too.
  The simulator's mean absolute error is reported against both baselines; losing to either is stated as a limit.
python -P scripts/convergence_bias_test.py TAG -> outputs/convergence_bias.json"""
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
from psychohistory.data import get_panel              # noqa: E402
from psychohistory.features import base_arrays        # noqa: E402

tag = sys.argv[1]
p = get_panel()
lg = base_arrays(p)["lg"]
codes = list(p.codes)
rows = []
for f in sorted(glob.glob(os.path.join(ROOT, "outputs", "world_hindcast", f"{tag}_s*_*.npz"))):
    z = np.load(f, allow_pickle=True)
    if "gdp_q" not in z.files:
        continue
    O, T = int(z["origin"]), int(z["T"])
    i0 = p.yi(O)
    ix = [codes.index(c) for c in z["gdp_codes"]]
    err = (z["gdp_q"][2] - z["gdp_act"]) / T * 100
    act = z["gdp_act"] / T * 100
    own = (lg[ix, i0] - lg[ix, i0 - 15]) / 15 * 100                     # own past growth carried forward
    allc = np.isfinite(lg[:, i0]) & np.isfinite(lg[:, i0 - 15])
    sl, ic = np.polyfit(lg[allc, i0 - 15], (lg[allc, i0] - lg[allc, i0 - 15]) / 15 * 100, 1)
    conv = ic + sl * z["gdp_lg0"]
    third = np.searchsorted(np.quantile(z["gdp_lg0"], [1 / 3, 2 / 3]), z["gdp_lg0"])
    for k in range(len(ix)):
        rows.append((O, third[k], err[k], abs(err[k]), abs(own[k] - act[k]), abs(conv[k] - act[k])))
R = np.array(rows, float)
assert len(R), f"no hindcast file of tag {tag} stores country growth"
R = R[np.isfinite(R).all(1)]
origins = np.unique(R[:, 0])
rng = np.random.default_rng(0)


def stats(M):
    poor, mid, rich = (M[M[:, 1] == k, 2].mean() for k in (0, 1, 2))
    return np.array([poor, mid, rich, poor - rich, M[:, 3].mean(), M[:, 4].mean(), M[:, 5].mean()])
est = stats(R)
bs = np.array([stats(np.concatenate([R[R[:, 0] == o] for o in rng.choice(origins, len(origins))])) for _ in range(1000)])
lo, hi = np.percentile(bs, [5, 95], 0)
names = ["poor third error", "middle third error", "rich third error", "poor minus rich", "simulator MAE",
         "own past growth MAE", "convergence line MAE"]
out = {"tag": tag, "country_windows": int(len(R)), "origins": [int(o) for o in origins],
       **{n: {"est": round(float(e), 3), "lo": round(float(a), 3), "hi": round(float(b_), 3)} for n, e, a, b_ in zip(names, est, lo, hi)}}
shares = est[0] >= 0.5 and lo[0] > 0 and est[3] >= 0.5 and lo[3] > 0
none = lo[0] <= 0 <= hi[0] and lo[3] <= 0 <= hi[3]
out["verdict"] = "shares the bias" if shares else "no convergence bias" if none else "in between: stated as found"
for n in names:
    print(f"{n:24s} {out[n]['est']:+.2f} [{out[n]['lo']:+.2f}, {out[n]['hi']:+.2f}]")
print(out["verdict"], len(R), "country-windows")
json.dump(out, open(os.path.join(ROOT, "outputs", "convergence_bias.json"), "w"), indent=1)
