"""Whole-simulator interstate hindcasts at more origins (10 years ahead; 2014 to 2024) for one interstate calibration rule
(world.DYAD_CALIB) and, optionally, a continuation half-life (interstate.CONT_HALFLIFE; "none" = equal weights).
python -P scripts/interstate_hindcast_multi.py RULE [HALFLIFE] [--feedback] -> outputs/interstate_hindcast_multi_<RULE>[_h<HALFLIFE>].json"""
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

from psychohistory import interstate as I, pipeline as PL, world as WM

world_kw = {}
if "--feedback" in sys.argv:                 # simulated conflicts raise the pair's event hostility (WorldConfig.ev_war_feedback)
    sys.argv.remove("--feedback")
    world_kw["ev_war_feedback"] = True
rule = sys.argv[1]
WM.DYAD_CALIB = rule
tag = rule
if len(sys.argv) > 2:
    I.CONT_HALFLIFE = None if sys.argv[2] == "none" else float(sys.argv[2])
    tag = f"{rule}_h{sys.argv[2]}"
p, base, X, Yl = PL.prepare(False)
origins = [(1980, 1990), (1985, 1995), (1990, 2000), (1995, 2005), (2000, 2010), (2005, 2015), (2010, 2020), (2014, 2024)]
if world_kw:
    tag += "_evfb"
rows = PL.step_interstate_hindcast(p, base, X, Yl, origins=origins, n_scen=2000, world_kw=world_kw)
for r in rows:
    print(r["origin"], round(r["active_per_year_sim"], 3), r["active_per_year_real"], round(r["pairs_fought_sim"], 1),
          r["pairs_fought_real"], round(r["pair_auc"], 3), r.get("years_in_90"), flush=True)
json.dump(rows, open(os.path.join(os.path.dirname(__file__), "..", "outputs", f"interstate_hindcast_multi_{tag}.json"), "w"),
          indent=1, default=float)
