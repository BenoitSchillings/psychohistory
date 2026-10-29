"""Re-run only the forecast step (scenarios, AI, crisis explorer), keeping the validation block (backtests,
hindcasts) of the existing outputs/forecast.json: those do not use the world layer. --interstate also refreshes
the interstate hindcast, which does; --hindcast the country hindcasts (needed when the slow dynamics change)."""
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pandas as pd

from psychohistory import pipeline as PL

old = json.load(open(PL.OUT / "forecast.json"))
p, base, X, Yl = PL.prepare(False)
fc = PL.step_forecast(p, base, X, Yl)
fc["validation"] = old["validation"]
if "--hindcast" in sys.argv:            # also refresh the country hindcasts (slow dynamics, weather)
    hdf, fans = PL.step_hindcast(p, base, X, Yl)
    fc["validation"]["hindcast"] = hdf.to_dict("records")
    fc["validation"]["hindcast_fans"] = {str(k): v for k, v in fans.items()}
if "--interstate" in sys.argv:          # also refresh the interstate hindcast (uses the world layer)
    fc["validation"]["interstate_hindcast"] = PL.step_interstate_hindcast(p, base, X, Yl, n_scen=2000)
with open(PL.OUT / "forecast.json", "w") as f:
    json.dump(PL._clean(fc), f)
pd.DataFrame([{k: v for k, v in c.items() if not isinstance(v, (list, dict))} for c in fc["countries"]]) \
    .to_csv(PL.OUT / "country_forecasts.csv", index=False)
print("wrote forecast.json")
