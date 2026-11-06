"""Does a hindcast run what the forecast runs? Compares the configuration each simulate() call recorded (simulate.
_config_record: the network's provenance, whether the family draw and the ensemble offset ran, every module switch,
the world layer's settings and the call's arguments) with the forecast's (outputs/sim_config_forecast.json, written by
pipeline.step_forecast). Without a saved forecast configuration the reference is what step_forecast would build now
(pipeline.SIM_ENV, the current module switches, the family draw on).

Differences that are part of a hindcast by design are listed separately: the origin and horizon, the number of
worlds and their seed, extrapolated demography, AI off, an ensemble refit seed, the climate inputs, and (sim_eval
only) the world layer. Anything else is a parity failure: exit status 1.

python -P scripts/config_parity.py [sim_eval TAG ...] [world_hindcast TAG ORIGIN ...]
python -P scripts/config_parity.py --self-test   (checks the comparison itself on synthetic records)"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
BY_DESIGN = {"call.origin", "call.end", "call.n_scen", "call.seed", "call.record_paths", "call.demography",
             "call.chronicle", "call.climate_damage", "call.struct", "net.origin", "net.ens_seed", "world.ai"}
WORLDLESS_OK = {"world"}           # sim_eval: the 10-year harness has no world layer by design


def flat(d, pre=""):
    out = {}
    for k, v in (d or {}).items():
        key = f"{pre}{k}"
        if isinstance(v, dict):
            out.update(flat(v, key + "."))
        else:
            out[key] = v
    return out


def expected():
    """what step_forecast would record now, from the code (no fitting)"""
    from psychohistory import pipeline as PL, simulate as SIM
    from psychohistory.world import AI_SCENARIOS, WorldConfig
    import dataclasses
    cfg = WorldConfig(ai=AI_SCENARIOS["uncertain"], sample_params=True)
    world = {f.name: (getattr(cfg, f.name) if isinstance(getattr(cfg, f.name), (int, float, str, bool, type(None)))
                      else repr(getattr(cfg, f.name))) for f in dataclasses.fields(cfg)}
    mod = {k: (list(v) if isinstance(v, tuple) else v) for k, v in vars(SIM).items()
           if k.isupper() and isinstance(v, (int, float, str, bool, tuple, type(None)))}
    n_nets = PL.SIM_ENV["members"] * 2                 # both PsychoNet seeds pooled
    return {"net": {"members": PL.SIM_ENV["members"], "n_nets": n_nets, "families": sorted(PL.SIM_ENV["families"]),
                    "first_year": PL.SIM_ENV["first_year"]},
            "struct_ran": SIM.STRUCT_MODE in ("family", "familyc", "ens_onset", "gauss"), "offset_ran": bool(SIM.ENS_OFFSET),
            "world": world, "module": mod}


def diff(ref, got, worldless=False):
    """{key: (reference, harness)} for keys present in the reference that differ, split into (failures, by_design)"""
    R, G = flat(ref), flat(got)
    bad, ok = {}, {}
    for k, v in R.items():
        if k.startswith("call."):
            continue                                   # call arguments: only the listed ones are compared, below
        if worldless and k.split(".")[0] in WORLDLESS_OK:
            continue
        g = G.get(k, "<missing>")
        if k == "world.ai" or k in BY_DESIGN:
            if g != v:
                ok[k] = (v, g)
            continue
        if g != v and not (isinstance(v, float) and isinstance(g, float) and np.isclose(v, g)):
            bad[k] = (v, g)
    if worldless and got.get("world") is not None:
        bad["world"] = (None, "world layer present")
    for k in ("call.align", "call.year_shock_scale", "call.frailty_sd", "call.mood_sd", "call.growth_regime_scale",
              "call.coup_align_mult", "call.align_window"):
        if k in R and R.get(k) != G.get(k):
            bad[k] = (R.get(k), G.get(k))
    return bad, ok


def load(kind, tag, origins):
    if kind == "sim_eval":
        d = json.load(open(os.path.join(ROOT, "outputs", "sim_eval", f"{tag}.json")))
        return {o: c for o, c in d.get("config", {}).items()}, True
    out = {}
    for o in origins:
        z = np.load(os.path.join(ROOT, "outputs", "world_hindcast", f"{tag}_{o}.npz"))
        if "config" not in z:
            out[o] = None
        else:
            out[o] = json.loads(str(z["config"]))
    return out, False


def self_test():
    ref = {"net": {"members": 10, "n_nets": 20}, "struct_ran": True, "module": {"STRUCT_MODE": "familyc"},
           "world": {"ai": "x", "sample_params": True}, "call": {"align": True, "origin": 2025}}
    same = json.loads(json.dumps(ref)); same["call"]["origin"] = 1990; same["world"]["ai"] = "none"
    assert diff(ref, same) == ({}, {"world.ai": ("x", "none")}), diff(ref, same)
    small = json.loads(json.dumps(same)); small["net"]["members"] = 3; small["net"]["n_nets"] = 6
    assert set(diff(ref, small)[0]) == {"net.members", "net.n_nets"}
    nofam = json.loads(json.dumps(same)); nofam["struct_ran"] = False
    assert set(diff(ref, nofam)[0]) == {"struct_ran"}
    worldless = json.loads(json.dumps(same)); worldless["world"] = None
    assert diff(ref, worldless, worldless=True)[0] == {}
    print("self-test passed")


def main():
    a = sys.argv[1:]
    if a and a[0] == "--self-test":
        return self_test()
    fref = os.path.join(ROOT, "outputs", "sim_config_forecast.json")
    if os.path.exists(fref):
        ref, src = json.load(open(fref)), "the saved forecast configuration"
    else:
        ref, src = expected(), "the configuration step_forecast would build now"
    print(f"reference: {src}")
    fail = 0
    i = 0
    while i < len(a):
        kind = a[i]
        if kind == "sim_eval":
            tags = []
            i += 1
            while i < len(a) and a[i] not in ("sim_eval", "world_hindcast"):
                tags.append(a[i]); i += 1
            runs = [(t, load("sim_eval", t, [])) for t in tags]
        else:
            tag, origins = a[i + 1], []
            i += 2
            while i < len(a) and a[i] not in ("sim_eval", "world_hindcast"):
                origins.append(a[i]); i += 1
            runs = [(tag, load("world_hindcast", tag, origins))]
        for tag, (cfgs, worldless) in runs:
            for o, c in cfgs.items():
                if c is None:
                    print(f"  {kind} {tag} {o}: no configuration recorded (run before 2026-11-05) - FAIL")
                    fail += 1
                    continue
                bad, ok = diff(ref, c, worldless)
                status = "ok" if not bad else "FAIL"
                fail += bool(bad)
                print(f"  {kind} {tag} {o}: {status}" + (f"  differs: {bad}" if bad else "") + (f"  (by design: {sorted(ok)})" if ok else ""))
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
