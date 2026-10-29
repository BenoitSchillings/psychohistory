"""Fresh, never-tuned-on check: one-year-ahead forecasts for 2020-2023 (conflict onset, conflict
termination, coup attempt), models trained on data observed by 2020. Compares the model at the
start of the improvement round with the final ensemble."""
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from psychohistory.backtest import flatten, train_mask
from psychohistory.evaluate import metrics
from psychohistory.features import FEATURES, GROUPS, TASKS
from psychohistory import models as M
from psychohistory.pipeline import prepare

ONE_YEAR = ["onset_1y", "termination_1y", "coup_1y"]
p, base, X, Yl = prepare()
Xf, Yf, ci, year = flatten(p, X, Yl)
te = (year >= 2020) & (year <= 2023)


def score(pred, name, Ytr):
    b = np.broadcast_to(np.nanmean(Ytr, 0), pred.shape)
    out = {t: metrics(Yf[te, TASKS.index(t)], pred[:, TASKS.index(t)], b[:, TASKS.index(t)]) for t in ONE_YEAR}
    s = float(np.mean([out[t]["auc"] + out[t]["bss"] for t in ONE_YEAR]))
    row = {t: out[t] for t in ONE_YEAR}
    print(f"[fresh] {name:32s} score {s:.4f} | " + " ".join(f"{t} {out[t]['auc']:.3f}/{out[t]['bss']:+.3f} (n={out[t]['n']}, ev={out[t]['events']})" for t in ONE_YEAR), flush=True)
    import pandas as pd
    from psychohistory.evaluate import LOG
    r = {"name": "fresh2020:" + name, "split": "fresh", "score": round(s, 4),
         **{f"{t}_auc": round(out[t]["auc"], 4) for t in ONE_YEAR}, **{f"{t}_bss": round(out[t]["bss"], 4) for t in ONE_YEAR}}
    pd.DataFrame([r]).reindex(columns=pd.read_csv(LOG, nrows=0).columns).to_csv(LOG, mode="a", header=False, index=False)


# session-start model: original feature set and the plain three-model average
old = [f for g in ("Institutions", "Macro-finance", "Commodities", "Leaders", "Economic contagion", "Violence intensity")
       if g in GROUPS for f in GROUPS[g]] + ["nb_conflict", "nb_coups", "nb_dem_gap", "nb_dem_change", "war_now",
                                             "interstate_recent", "pre1946"]
Xo = Xf.copy(); Xo[:, [FEATURES.index(f) for f in old if f in FEATURES]] = np.nan
Ytr, tr = train_mask(Yf, year, 2020)
m0 = M.Ensemble(members=8).fit(Xo[tr], Ytr[tr], ci[tr])
score(m0.predict(Xo[te]), "session-start model", Ytr[tr])

# final ensemble (current features minus rejected groups)
rej = [f for g in ("Macro-finance", "Commodities") for f in GROUPS.get(g, [])]
if "--with-intensity" not in sys.argv and "Violence intensity" in GROUPS:
    rej += GROUPS["Violence intensity"]
Xn = Xf.copy(); Xn[:, [FEATURES.index(f) for f in rej]] = np.nan
Ytr, tr = train_mask(Yf, year, 2020, first_year=1860)
m1 = M.SeldonEnsemble(M.load_combine(), members=8, families=["G", "G60", "N", "Nq", "L"]).fit(Xn[tr], Ytr[tr], ci[tr], year[tr])
score(m1.predict(Xn[te]), "final ensemble" + (" +intensity" if "--with-intensity" in sys.argv else ""),
      Ytr[tr & (year >= 1955)])
