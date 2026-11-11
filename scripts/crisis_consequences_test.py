"""Do the consequences of a civil war move together? (11 November 2026)

Hoyer et al. (2024, Seshat crisis database, 168 historical crises) report that a crisis's consequences (civil war,
population loss, reform, collapse) are highly variable and largely unrelated to one another. The simulator draws a
civil war's deaths (psychohistory/civil_deaths.json) independently of its output cost (a fixed loss per war-year by
level) and of what happens to the regime. Written and committed before its run.

Design. Civil-conflict onsets 1950-2018 (a country at peace the year before, with the next five years observed).
Four consequences over the five years after onset:
  deaths   log of mean battle deaths per million over the window's conflict years
  output   change in log GDP per person over five years minus five times the country's average growth in the ten
           years before onset (negative = loss)
  regime   change in the democracy index over five years
  coup     a coup attempt within five years (0 / 1)
Spearman rank correlations for the six pairs, 90% intervals from 1,000 bootstrap resamples of countries.

Pre-registered reading.
  "Independent, as the simulator has it" if every |rho| < 0.2.
  If rho(deaths, output) <= -0.3 with an interval excluding 0, linking a war's output cost to its drawn death rate
  becomes a candidate mechanism (to be tested under the adopt rule, not adopted here).
  Anything in between: stated as found, no change.
python -P scripts/crisis_consequences_test.py -> outputs/crisis_consequences.json"""
import json
import os
import sys
import warnings

import numpy as np
from scipy.stats import spearmanr

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
warnings.simplefilter("ignore")
from psychohistory.data import get_panel              # noqa: E402
from psychohistory.features import base_arrays        # noqa: E402

p = get_panel()
b = base_arrays(p)
yrs = np.asarray(p.years)
conf, lg, dem, coup, bd = b["conflict"], b["lg"], b["dem"], b["coup_attempts"], b["battle_deaths_pc"]
rows = []
for c in range(conf.shape[0]):
    for t in np.where((yrs >= 1950) & (yrs <= 2018))[0]:
        if not (conf[c, t] == 1 and conf[c, t - 1] == 0):
            continue
        w = slice(t, t + 6)
        d = bd[c, w][(conf[c, w] > 0) & (bd[c, w] > 0)]
        pre = (lg[c, t - 1] - lg[c, t - 11]) / 10
        r = (c, np.log(d.mean()) if len(d) else np.nan, lg[c, t + 5] - lg[c, t] - 5 * pre,
             dem[c, t + 5] - dem[c, t], float(np.nansum(coup[c, t + 1:t + 6]) > 0) if np.isfinite(coup[c, t + 1:t + 6]).any() else np.nan)
        rows.append(r)
R = np.array(rows, float)
names = ["deaths", "output", "regime", "coup"]
rng = np.random.default_rng(0)
cs = np.unique(R[:, 0])
out = {"onsets": int(len(R)), "countries": int(len(cs)), "pairs": {}}
for i in range(4):
    for j in range(i + 1, 4):
        ok = np.isfinite(R[:, 1 + i]) & np.isfinite(R[:, 1 + j])
        rho = spearmanr(R[ok, 1 + i], R[ok, 1 + j])[0]
        bs = []
        for _ in range(1000):
            pick = rng.choice(cs, len(cs))
            S = np.concatenate([R[(R[:, 0] == c) & ok] for c in pick])
            if len(S) > 10:
                bs.append(spearmanr(S[:, 1 + i], S[:, 1 + j])[0])
        lo, hi = np.nanpercentile(bs, [5, 95])
        out["pairs"][f"{names[i]}-{names[j]}"] = {"n": int(ok.sum()), "rho": round(float(rho), 3), "lo": round(float(lo), 3), "hi": round(float(hi), 3)}
        print(f"{names[i]:7s} {names[j]:7s} n={ok.sum():4d} rho={rho:+.3f} [{lo:+.3f}, {hi:+.3f}]")
mx = max(abs(v["rho"]) for v in out["pairs"].values())
do = out["pairs"]["deaths-output"]
out["verdict"] = "independent" if mx < 0.2 else ("candidate: link output cost to deaths" if do["rho"] <= -0.3 and do["hi"] < 0 else "in between: stated, no change")
out["medians"] = {n: round(float(np.nanmedian(R[:, 1 + k])), 3) for k, n in enumerate(names)}
print(out["verdict"], out["medians"])
json.dump(out, open(os.path.join(ROOT, "outputs", "crisis_consequences.json"), "w"), indent=1)
