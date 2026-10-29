"""Fetch live crowd and market questions, match them to model quantities, write outputs/crowd_live.json.
python scripts/crowd_live.py"""
import datetime as dt
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from psychohistory import crowd
from psychohistory.data import ROOT

fc = json.load(open(ROOT / "outputs" / "forecast.json"))
qs = crowd.fetch()
m, W = crowd.match(qs, fc)
print(f"{len(qs)} open questions, {len(m)} matched; blend weights {W}")
for r in m[:25]:
    print(f"  {r['source']:10s} {r['kind']:5s} crowd {r['p']:.3f} model {r['model']:.3f} blend {r['blend']:.3f} | {r['question']}")
json.dump({"fetched": dt.datetime.now(dt.timezone.utc).isoformat(), "weights": W, "n_open": len(qs), "matched": m},
          open(ROOT / "outputs" / "crowd_live.json", "w"), indent=1)
