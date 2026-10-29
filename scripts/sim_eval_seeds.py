"""Compare two simulator configurations averaged over network seeds (the single-seed noise is large for the world counts:
docs/world_system.md, "Seed noise of the simulator hindcasts").
python -P scripts/sim_eval_seeds.py "A_TAG1,A_TAG2,A_TAG3" "B_TAG1,B_TAG2,B_TAG3"
Each tag is an outputs/sim_eval/<tag>.json; per origin, each metric is averaged over a configuration's seeds, then the
two configurations are compared as in sim_eval --compare (mean over origins, relative change, origins where B is
better). The seed spread (range of the per-seed means, relative) is printed beside it."""
import json
import os
import sys

import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs", "sim_eval")
LOWER_BETTER = ("conflict_brier", "coup_brier", "coups_crps", "world_share_crps", "onsets_crps", "gdp_crps",
                "world_gdp_crps", "world_gdppc_crps", "dem_crps")


def load(tags):
    runs = [json.load(open(os.path.join(OUT, f"{t}.json")))["by_origin"] for t in tags.split(",")]
    common = sorted(set.intersection(*(set(r) for r in runs)), key=int)
    return runs, common


def main():
    A, ca = load(sys.argv[1])
    B, cb = load(sys.argv[2])
    common = [o for o in ca if o in cb]
    print(f"B ({sys.argv[2]}) vs A ({sys.argv[1]}): {len(common)} origins ({common[0]}-{common[-1]}), "
          f"{len(A)} and {len(B)} seeds")
    for m in A[0][common[0]]:
        if m.startswith("diag_onset_year"):
            continue
        get = lambda runs: np.array([[r[o].get(m, np.nan) for o in common] for r in runs], float)    # [seeds, origins]
        va, vb = get(A), get(B)
        if not np.isfinite(va).any():
            continue
        ma, mb = np.nanmean(va, 0), np.nanmean(vb, 0)
        lower = m in LOWER_BETTER
        better = (mb < ma) if lower else (mb > ma)
        rel = np.nanmean(mb) / np.nanmean(ma) - 1
        spread = lambda v: (np.nanmax(np.nanmean(v, 1)) - np.nanmin(np.nanmean(v, 1))) / abs(np.nanmean(v))
        print(f"  {m:20s} {np.nanmean(ma):>10.5f} -> {np.nanmean(mb):>10.5f}  ({100 * rel:+.1f}%)  B better at "
              f"{int(better.sum())}/{len(common)}   seed spread A {100 * spread(va):.1f}%, B {100 * spread(vb):.1f}%")


if __name__ == "__main__":
    main()
