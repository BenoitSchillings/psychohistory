"""Risk-model ensemble at past origins, for simulator tests (sim_eval ens_offset): per origin, each family's one-year
probabilities (onset, termination, coup) at the origin's features, trained only on data before the origin, as the
forecast's ensemble (families G, G60, N, F; 3 members).
python -P scripts/ens_cache.py FIRST LAST STEP -> outputs/sim_eval/ens_cache/<origin>.npz"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs", "sim_eval", "ens_cache")


def main():
    from psychohistory import pipeline as PL
    from psychohistory.backtest import flatten, train_mask
    from psychohistory.features import TASKS
    from psychohistory.models import SeldonEnsemble, load_combine
    first, last, step = (int(a) for a in sys.argv[1:4])
    os.makedirs(OUT, exist_ok=True)
    p, base, X, Yl = PL.prepare(False)
    Xf, Yf, ci, year = flatten(p, X, Yl)
    k3 = [TASKS.index(t) for t in ("onset_1y", "termination_1y", "coup_1y")]
    for O in range(first, last + 1, step):
        f = os.path.join(OUT, f"{O}.npz")
        if os.path.exists(f):
            continue
        Ytr, tr = train_mask(Yf, year, O, first_year=1860)
        ens = SeldonEnsemble(load_combine(), members=3, families=["G", "G60", "N", "F"]).fit(Xf[tr], Ytr[tr], ci[tr], year[tr])
        P = ens.predict_families(X[:, p.yi(O)])
        np.savez_compressed(f, fams=np.array(sorted(P)), p=np.stack([P[k][:, k3] for k in sorted(P)]))
        print(O, "done", flush=True)


if __name__ == "__main__":
    main()
