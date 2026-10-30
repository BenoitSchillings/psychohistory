"""Does a state already fighting other interstate conflicts start new ones less often (a force budget), or more often
(war begets war)? Tested before the simulator gets either.

For pairs at peace, the 1-year onset model's out-of-sample prediction (refitted with each decade held out, as
interstate.run's calibration) and the number of other conflicts each side is fighting that year (UCDP ongoing pairs).
logit(onset next year ~ logit(p_oof) + other wars of the busier side), errors clustered by pair; development 1946-1994,
confirmation 1995-2024.
python -P scripts/force_budget_test.py -> outputs/force_budget_test.json"""
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pandas as pd
import statsmodels.api as sm

from psychohistory import interstate as I
from psychohistory.data import ROOT


def main():
    df = I.build(verbose=False)
    origin = 2024
    t1 = df.interstate_1y.notna() & (df.year + 1 <= origin) & (df.ongoing == 0)
    oof = pd.Series(np.nan, index=df.index)
    for lo in range(1946, origin, 10):
        te = t1 & (df.year >= lo) & (df.year < lo + 10)
        if te.any():
            tr = t1 & ~((df.year >= lo - 1) & (df.year < lo + 11))
            m = I.DyadModel().fit(df.loc[tr, I.FEATS_ON].to_numpy(float), df.loc[tr, "interstate_1y"].to_numpy())
            oof[te] = m.predict(df.loc[te, I.FEATS_ON].to_numpy(float))
    on = df[df.ongoing == 1]
    busy = pd.concat([on[["a", "year"]].rename(columns={"a": "c"}), on[["b", "year"]].rename(columns={"b": "c"})]).groupby(["c", "year"]).size()
    d = df[t1 & oof.notna()].copy()
    d["oof"] = oof[d.index].clip(1e-5, 1 - 1e-5)
    d["busy_a"] = [busy.get((a, y), 0) for a, y in zip(d.a, d.year)]
    d["busy_b"] = [busy.get((b, y), 0) for b, y in zip(d.b, d.year)]
    d["busy_max"] = np.maximum(d.busy_a, d.busy_b)
    d["busy_any"] = (d.busy_max > 0).astype(float)
    d["pair"] = d.a + "-" + d.b
    out = {}
    for name, sub in (("dev_1946_1994", d[d.year <= 1994]), ("test_1995_2023", d[d.year >= 1995])):
        r = {"n": len(sub), "onsets": int(sub.interstate_1y.sum()), "share_busy": round(float(sub.busy_any.mean()), 4)}
        for v in ("busy_any", "busy_max"):
            X = sm.add_constant(np.column_stack([np.log(sub.oof / (1 - sub.oof)), sub[v]]))
            m = sm.Logit(sub.interstate_1y.to_numpy(float), X).fit(disp=0, cov_type="cluster", cov_kwds={"groups": pd.factorize(sub.pair)[0]})
            r[v] = {"coef": round(float(m.params[2]), 3), "t": round(float(m.tvalues[2]), 2), "p": round(float(m.pvalues[2]), 4)}
        out[name] = r
    print(json.dumps(out, indent=1))
    json.dump(out, open(os.path.join(ROOT, "outputs", "force_budget_test.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
