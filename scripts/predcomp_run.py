"""Our submission to the 2023/24 VIEWS Prediction Challenge protocol (country-month), written in its layout.

Windows Y2018..Y2023: data to October of the year before (final GED), forecasts for the twelve months of the year.
Live window (July 2024 - June 2025, files Y2024 and Y2025): data to April 2024; final GED 23.1 then covered 2022,
so 2023-01 .. 2024-04 come from the UCDP Candidate monthly releases, as for the participants.

python scripts/predcomp_run.py NAME [--spread S]   [--live-only] [--final-through YEAR] [--vintage-w W]
                                                    (S: widen each forecast around its median in log space;
                                                    1.15 chosen on the 2010-2017 dev windows, predcomp_tune.py)      -> outputs/predcomp/NAME/cm/window=Y..../NAME_cm_Y....parquet
"""
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pandas as pd

from psychohistory import cm_fatal as CF

ROOT = os.path.join(os.path.dirname(__file__), "..")


def write(name, window, draws):
    cs = CF.countries()
    rows = []
    for m, X in draws.items():
        n = X.shape[1]
        rows.append(pd.DataFrame({"month_id": m, "country_id": np.repeat(cs.views_country_id.to_numpy(), n),
                                  "draw": np.tile(np.arange(n), len(cs)), "outcome": X.ravel().astype(np.float32)}))
    df = pd.concat(rows).set_index(["month_id", "country_id", "draw"])
    d = os.path.join(ROOT, "outputs", "predcomp", name, "cm", f"window={window}")
    os.makedirs(d, exist_ok=True)
    df.to_parquet(os.path.join(d, f"{name}_cm_{window}.parquet"))


def spread(X, s):
    if s == 1.0:
        return X
    lx = np.log1p(X)
    med = np.median(lx, 1, keepdims=True)
    return np.expm1(np.clip(med + s * (lx - med), 0, None))


def main():
    name = sys.argv[1]
    s = float(sys.argv[sys.argv.index("--spread") + 1]) if "--spread" in sys.argv else 1.0
    windows = [(f"Y{y}", CF.mid(y - 1, 10), [CF.mid(y, m) for m in range(1, 13)], None) for y in range(2018, 2024)]
    live_origin = CF.mid(2024, 4)
    live = [CF.mid(2024, m) for m in range(7, 13)] + [CF.mid(2025, m) for m in range(1, 7)]
    # final GED for the live origin: participants had GED 24.1 (final through 2023, released June 2024) - visible in
    # the organizers' conflictology benchmark, whose Ukraine and Gaza draws reach final-GED 2023 levels that the
    # candidate releases never showed. --final-through 2022 reproduces the stricter (wrong) vintage of seldon_v1/v2.
    vw = float(sys.argv[sys.argv.index("--vintage-w") + 1]) if "--vintage-w" in sys.argv else None
    ft = 2023 if "--final-through" not in sys.argv else int(sys.argv[sys.argv.index("--final-through") + 1])
    windows.append(("LIVE", live_origin, live, CF.mid(ft, 12)))
    if "--live-only" in sys.argv:
        windows = windows[-1:]
    for w, origin, months, ft in windows:
        t0 = time.time()
        draws, _ = CF.train_and_forecast(origin, months, final_through=ft, correct_w=vw)
        draws = {m: spread(np.asarray(v, float), s) for m, v in draws.items()}
        if w == "LIVE":
            write(name, "Y2024", {m: v for m, v in draws.items() if m <= CF.mid(2024, 12)})
            write(name, "Y2025", {m: v for m, v in draws.items() if m > CF.mid(2024, 12)})
        else:
            write(name, w, draws)
        print(f"{w}: {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
