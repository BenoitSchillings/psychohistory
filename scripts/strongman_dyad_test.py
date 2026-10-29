"""Do personalist strongmen and military juntas start more wars (Weeks 2012, "Strongmen and Straw Men")? Pair features
from the country panel (V-Dem via the personal-rule family): the higher leader-cult legitimation of the two, either a
military regime, either leader having taken power by force, the larger military share of the regime's support, and
"strongman" (personalist and autocratic: leader cult above the 75th percentile and democracy below 0.4) on either side.
Dyad backtests (interstate.backtest, logit + GBM) for the 5-year UCDP outcome (origins 1975-1995 dev, 2000-2015 test)
and the 1-year onset model that drives the simulator (pairs at peace, FEATS_ON, same splits), with and without them.
python -P scripts/strongman_dyad_test.py -> outputs/new_data/strongman_dyad.json
("junta": only a military regime on either side and a leader who took power by force, the two with clear effects in a
logit with controls, errors clustered by pair, 1946-2015: +0.67 and +0.43 sd; personalism -0.37, n.s.)"""
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

from psychohistory import interstate as I
from psychohistory.data import get_panel, ROOT
from psychohistory.features import base_arrays

NEW = ["personalist_max", "military_regime_any", "path_force_any", "support_military_max", "strongman_any"]


def add(df):
    p = get_panel()
    b = base_arrays(p)
    ci = {c: i for i, c in enumerate(p.codes)}
    y0 = int(p.years[0])
    get = lambda name: p.x.get("feat__" + name, b.get(name))
    lp, sm, pf = get("legit_personalist"), get("support_military"), get("path_force")
    mil, dem = b["regime_military"], b["dem"]
    q75 = np.nanquantile(lp, 0.75)
    strong = np.where(np.isfinite(lp) & np.isfinite(dem), ((lp > q75) & (dem < 0.4)).astype(float), np.nan)
    t = np.clip(df.year.to_numpy() - y0, 0, lp.shape[1] - 1)
    ia, ib = df.a.map(ci).to_numpy(), df.b.map(ci).to_numpy()
    both = lambda a: (a[ia, t], a[ib, t])
    mx = lambda a: np.fmax(*both(a))
    df["personalist_max"] = mx(lp)
    df["military_regime_any"] = mx(np.nan_to_num(mil, nan=np.nan))
    df["path_force_any"] = mx(pf)
    df["support_military_max"] = mx(sm)
    df["strongman_any"] = mx(strong)
    return df


def main():
    df = add(I.build(False))
    res = {}
    for target, feats0 in (("interstate_5y", I.FEATS), ("interstate_1y", I.FEATS_ON)):
        d = df if target == "interstate_5y" else df[df.ongoing == 0]
        for split, origins in (("dev", range(1975, 1996, 5)), ("test", range(2000, 2016, 5))):
            for name, feats in (("base", feats0), ("strongman", feats0 + NEW),
                                ("junta", feats0 + ["military_regime_any", "path_force_any"])):
                out, _ = I.backtest(d, target, origins, feats)
                res[f"{target}:{split}:{name}"] = {m: {k: round(v, 4) for k, v in out[m].items() if k in ("auc", "bss", "events")}
                                                   for m in ("logit", "gbm", "avg")}
                print(target, split, name, res[f"{target}:{split}:{name}"]["avg"], "logit", res[f"{target}:{split}:{name}"]["logit"], flush=True)
    json.dump(res, open(os.path.join(ROOT, "outputs", "new_data", "strongman_dyad.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
