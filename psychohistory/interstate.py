"""Interstate conflict between pairs of countries (dyads), 1946-2024.

Unit: undirected, politically relevant country pairs (neighbours by land or sea, capitals within
1000 km of each other, strategic rivals, or pairs involving a major power) per year.

Outcomes
  interstate_5y  a UCDP/PRIO interstate armed conflict (>= 25 battle deaths in a year) between the
                 two within the next five years (consistent 1946-2024)
  fatal_mid_5y   a militarised interstate dispute with > 25 deaths between them within five years
                 (Gibler-Miller-Little MIDs, 1816-2010): more events, auxiliary
  interstate_1y  the same UCDP outcome in the next year (used by the simulator)
  force_5y       any militarised use of force between them within five years (GML MID hostility level 4-5:
                 strikes, clashes, seizures and wars, with or without deaths; 1816-2010): catches the short
                 strike-and-retaliate episodes that never reach 25 battle deaths

Fighting already recorded after the origin year (UCDP candidate events, data/raw/outcomes/ucdp_2025_interstate_dy.csv)
is treated as known: those pairs are certain for the first forecast window, and the simulator starts with them active.

Data via the peacesciencer package (Miller 2022): GML MIDs, COW capitals, contiguity, national
material capabilities, major powers, ATOP alliances, Thompson-Dreyer strategic rivalries,
territorial threat (Miller); plus UCDP/PRIO 25.1, OWID nuclear warheads and the country panel.
"""
from __future__ import annotations

import io
import warnings
import zipfile

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyreadr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from .data import RAW, LAST_OBS_YEAR, get_panel, REIGN_NAME_FIX, EPR_NAME_FIX, UCDP_NAME_FIX
from .features import base_arrays

SYS = RAW / "system"
FIRST = 1850
H = 5
MAJORS = {"USA", "RUS", "CHN", "GBR", "FRA"}          # permanent Security Council members
COW_NAME_FIX = {**REIGN_NAME_FIX, **EPR_NAME_FIX, "United States of America": "United States",
                "German Federal Republic": "Germany", "Yemen Arab Republic": "Yemen", "Republic of Vietnam": "Vietnam",
                "Democratic Republic of the Congo": "Democratic Republic of Congo", "Macedonia": "North Macedonia",
                "Swaziland": "Eswatini", "Czech Republic": "Czechia", "Ivory Coast": "Cote d'Ivoire",
                "Korea": "South Korea"}


def _rda(name):
    return list(pyreadr.read_r(SYS / f"{name}.rda").values())[0]


def _ccode_to_iso(codes):
    st = _rda("cow_states")
    names = pd.read_csv(RAW / "population-with-un-projections.csv")[["entity", "code"]].drop_duplicates()
    n2c = dict(zip(names.entity, names.code))
    m = {}
    for cc, nm in zip(st.ccode, st.statenme):
        iso = n2c.get(COW_NAME_FIX.get(nm, nm))
        if iso in codes:
            m[int(cc)] = iso
    m[713] = "TWN"
    m[365] = "RUS"
    m[300] = "AUT"        # Austria-Hungary -> Austria
    m[260] = "DEU"        # German Federal Republic
    return m


def _existence(c2i):
    """years each panel country existed as a state (COW system membership)."""
    st = _rda("cow_states")
    ex = {}
    for cc, s, e in zip(st.ccode, st.styear, st.endyear):
        iso = c2i.get(int(cc))
        if iso:
            ex.setdefault(iso, []).append((int(s), int(e) if e == e and e < 2011 else 9999))
    return ex


def _majors(c2i):
    mj = _rda("cow_majors")
    out = {}
    for cc, s, e in zip(mj.ccode, mj.styear, mj.endyear):
        iso = c2i.get(int(cc))
        if iso:
            out.setdefault(iso, []).append((int(s), 9999 if e >= 2016 else int(e)))
    return out


def _ucdp_interstate(codes):
    """set of (iso_a, iso_b, year) with an active UCDP interstate conflict (type 2)."""
    names = pd.read_csv(RAW / "population-with-un-projections.csv")[["entity", "code"]].drop_duplicates()
    n2c = dict(zip(names.entity, names.code))
    from .data import UCDP_FILE
    z = zipfile.ZipFile(RAW / UCDP_FILE)
    ac = pd.read_csv(io.BytesIO(z.read([n for n in z.namelist() if n.endswith(".csv")][0])))
    ac = ac[ac.type_of_conflict == 2]
    side = lambda s: [n2c.get(UCDP_NAME_FIX.get(x, x)) for x in
                      [t.replace("Government of ", "").strip() for t in str(s).split(",")]]
    ev = {}
    for a, b, y, inten in zip(ac.side_a, ac.side_b, ac.year, ac.intensity_level):
        for i in side(a):
            for j in side(b):
                if i in codes and j in codes and i != j:
                    k = (min(i, j), max(i, j), int(y))
                    ev[k] = max(ev.get(k, 0), int(inten))      # 1: 25-999 deaths, 2: war (1000+)
    return ev


def build(verbose=True):
    p = get_panel()
    base = base_arrays(p)
    codes = list(p.codes)
    c2i = _ccode_to_iso(set(codes))
    years = np.arange(FIRST, LAST_OBS_YEAR + 1)
    # --- relevance: contiguity, capital distance, rivalry, major power ---------------------------
    cont = _rda("cow_contdir")
    cont = cont[cont.conttype <= 5]   # land border or <= 400 miles of water
    contig = {}
    for a, b, t, s, e in zip(cont.ccode1, cont.ccode2, cont.conttype, cont.stdate, cont.enddate):
        if int(a) in c2i and int(b) in c2i:
            k = tuple(sorted((c2i[int(a)], c2i[int(b)])))
            contig[k] = min(contig.get(k, 9), int(t))
    capt = _rda("cow_capitals")
    cap = {c2i[int(c)]: (la, lo) for c, la, lo in zip(capt.ccode, capt.lat, capt.lng) if int(c) in c2i}
    riv = _rda("td_rivalries")
    rivals = {}
    for a, b, s, e in zip(riv.ccode1, riv.ccode2, riv.styear, riv.endyear):
        if int(a) in c2i and int(b) in c2i:
            k = tuple(sorted((c2i[int(a)], c2i[int(b)])))
            rivals.setdefault(k, []).append((int(s), 9999 if e >= 2007 else int(e)))   # still active at the data's end
    rivals[("CHN", "TWN")] = [(1949, 9999)]                             # not a COW dyad pre-1971 in all sources

    def dist(a, b):
        if a not in cap or b not in cap:
            return np.nan
        (la1, lo1), (la2, lo2) = map(lambda t: np.radians(t), (cap[a], cap[b]))
        h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
        return 6371 * 2 * np.arcsin(np.sqrt(h))

    exist = _existence(c2i)
    majors = _majors(c2i)
    alive = lambda c, y: any(s <= y <= e for s, e in exist.get(c, [(1946, 9999)]))
    is_major = lambda c, y: any(s <= y <= e for s, e in majors.get(c, []))
    ever_major = set(majors) | MAJORS
    pairs = []
    for i, a in enumerate(codes):
        for b in codes[i + 1:]:
            k = (min(a, b), max(a, b))
            dd = dist(a, b)
            if k in contig or k in rivals or (dd < 1000) or a in ever_major or b in ever_major:
                pairs.append(k)
    if verbose:
        print(f"  {len(pairs)} politically relevant pairs")
    # --- outcomes -------------------------------------------------------------------------------
    ucdp = _ucdp_interstate(set(codes))
    # pairs that have fought (UCDP, or recorded after the origin) are relevant whatever their distance. Their rows
    # exist from the start (event history, features), but their outcomes count only from their first recorded
    # conflict on: counting earlier ones would select pairs on their future
    seen = set(pairs)
    first = {}
    for a, b, y in ucdp:
        first[(a, b)] = min(first.get((a, b), 9999), y)
    for a, b, y, _ in observed_after(LAST_OBS_YEAR):
        first[(a, b)] = min(first.get((a, b), 9999), y)
    extra = {k: y for k, y in first.items() if k not in seen and k[0] in codes and k[1] in codes}
    pairs += sorted(extra)
    g = _rda("gml_mid_ddydisps")
    g = g[g.fatality >= 2]          # dispute with more than 25 battle deaths
    fatal = set()
    for a, b, y in zip(g.ccode1, g.ccode2, g.year):
        if int(a) in c2i and int(b) in c2i:
            fatal.add((min(c2i[int(a)], c2i[int(b)]), max(c2i[int(a)], c2i[int(b)]), int(y)))
    gf = _rda("gml_mid_ddydisps")
    gf = gf[gf.hostlev >= 4]        # use of force or war, whatever the death toll
    force = set()
    for a, b, y in zip(gf.ccode1, gf.ccode2, gf.year):
        if int(a) in c2i and int(b) in c2i:
            force.add((min(c2i[int(a)], c2i[int(b)]), max(c2i[int(a)], c2i[int(b)]), int(y)))
    # --- monadic inputs -------------------------------------------------------------------------
    ci = {c: i for i, c in enumerate(codes)}
    yi = lambda y: p.yi(y)
    lgdp = base["lg"] + base["lpop"]                      # log total GDP (PPP), the power proxy
    dem, conf = base["dem"], base["conflict"]
    milex = p.x["military"]
    warh = np.nan_to_num(p.x.get("warheads", np.zeros_like(dem)))
    nmc = _rda("cow_nmc")
    cinc = np.full(dem.shape, np.nan)
    for c, y, v in zip(nmc.ccode, nmc.year, nmc.cinc):
        if int(c) in c2i and p.years[0] <= y <= p.years[-1]:
            cinc[ci[c2i[int(c)]], yi(int(y))] = v
    tt = _rda("terrthreat")
    terr = np.full(dem.shape, np.nan)
    for c, y, v in zip(tt.ccode, tt.year, tt.lterrthreat):
        if int(c) in c2i and p.years[0] <= y <= p.years[-1]:
            terr[ci[c2i[int(c)]], yi(int(y))] = v
    atop = _rda("atop_alliance")
    atop = atop[atop.atop_defense == 1]
    allied = set()
    for a, b, y in zip(atop.ccode1, atop.ccode2, atop.year):
        if int(a) in c2i and int(b) in c2i:
            allied.add((min(c2i[int(a)], c2i[int(b)]), max(c2i[int(a)], c2i[int(b)]), int(y)))
    # after ATOP ends: NATO members (with their accession years) remain allied
    a_end = int(atop.year.max())
    for y in range(a_end + 1, p.years[-1] + 1):
        mem = [c for c in NATO if NATO_JOIN.get(c, 0) <= y]
        for i, a in enumerate(mem):
            for b in mem[i + 1:]:
                allied.add((min(a, b), max(a, b), y))
    ffill = lambda arr, lim: pd.DataFrame(arr).T.ffill(limit=lim).T.to_numpy()
    cinc, terr = ffill(cinc, 6), ffill(terr, 15)
    coup = base["coup_recent"]
    # --- dyad-year rows -------------------------------------------------------------------------
    rows = []
    for a, b in pairs:
        ia, ib = ci[a], ci[b]
        rv = rivals.get((a, b))
        ever_al = False
        last_ev, past = -99, []
        for y in years:
            if not (alive(a, y) and alive(b, y)):
                continue
            t = yi(y)
            ev_u = (a, b, y) in ucdp if y >= 1946 else np.nan
            war_now = float(ucdp.get((a, b, y), 0) == 2) if y >= 1946 else np.nan
            ev_g = (a, b, y) in fatal
            al = (a, b, min(y, 2018)) in allied or (a, b, y) in allied or \
                any({a, b} == {x_, z_} and y >= y_ for x_, z_, y_ in POST_ATOP)   # ATOP to 2018, NATO and later pacts after
            ga, gb = lgdp[ia, t], lgdp[ib, t]
            ra = ga - gb if np.isfinite(ga) and np.isfinite(gb) else np.nan
            ra10 = (lgdp[ia, t - 10] - lgdp[ib, t - 10]) if t >= 10 else np.nan
            rows.append({
                "a": a, "b": b, "year": int(y), "ev_ucdp": ev_u, "ev_gml": ev_g if y <= 2010 else np.nan,
                "ev_force": float((a, b, y) in force or ev_u is True) if y <= 2010 else np.nan,
                "contig_land": float(contig.get((a, b), 9) == 1), "contig_sea": float(1 < contig.get((a, b), 9) <= 5),
                "log_cap_dist": np.log1p(dist(a, b)), "rival": float(rv is not None and any(s_ <= y <= e_ for s_, e_ in rv)),
                "major_pair": float(is_major(a, y) and is_major(b, y)), "one_major": float(is_major(a, y) ^ is_major(b, y)),
                "abs_power_ratio": abs(ra) if np.isfinite(ra) else np.nan,
                "power_shift_10y": (abs(ra) - abs(ra10)) if np.isfinite(ra) and np.isfinite(ra10) else np.nan,
                "log_joint_gdp": np.logaddexp(ga, gb) if np.isfinite(ga) and np.isfinite(gb) else np.nan,
                "cinc_ratio": np.log((max(cinc[ia, t], cinc[ib, t]) + 1e-6) / (min(cinc[ia, t], cinc[ib, t]) + 1e-6))
                if np.isfinite(cinc[ia, t]) and np.isfinite(cinc[ib, t]) else np.nan,
                "both_nuclear": float(warh[ia, t] > 0 and warh[ib, t] > 0), "one_nuclear": float((warh[ia, t] > 0) ^ (warh[ib, t] > 0)),
                "allied": float(al),
                "terr_threat_max": np.nanmax([terr[ia, t], terr[ib, t]]) if np.isfinite([terr[ia, t], terr[ib, t]]).any() else np.nan,
                "dem_min": np.nanmin([dem[ia, t], dem[ib, t]]), "dem_max": np.nanmax([dem[ia, t], dem[ib, t]]),
                "dem_gap": abs(dem[ia, t] - dem[ib, t]),
                "civil_war_any": float(np.nan_to_num(conf[ia, t]) + np.nan_to_num(conf[ib, t]) > 0),
                "coup_recent_max": np.nanmax([coup[ia, t], coup[ib, t]]),
                "milex_max": np.nanmax([milex[ia, t], milex[ib, t]]) if np.isfinite([milex[ia, t], milex[ib, t]]).any() else np.nan,
                "conflict_past_10y": float(np.mean([y - 10 <= e < y for e in past])) * 10 / 10 if past else 0.0,
                "log_years_since_conflict": np.log1p(min(y - last_ev, 80)),
                "ever_conflict": float(ever_al), "war_now": war_now,
            })
            if ev_u is True or ev_g is True:
                last_ev = y
                past.append(y)
                ever_al = True
    df = pd.DataFrame(rows)
    # count of conflict years in the past 10 (strictly before t)
    df = df.sort_values(["a", "b", "year"])
    evt = ((df.ev_ucdp == 1) | (df.ev_gml == 1)).astype(float)
    df["conflict_past_10y"] = evt.groupby([df.a, df.b]).transform(lambda s: s.shift(1).rolling(10, min_periods=1).sum()).fillna(0)
    # labels
    grp = df.groupby(["a", "b"])
    fut = lambda col, h: grp[col].transform(lambda s: s.shift(-1)[::-1].rolling(h, min_periods=h).max()[::-1])
    df["interstate_5y"] = fut("ev_ucdp", H)
    df["interstate_1y"] = grp["ev_ucdp"].shift(-1).astype(float)
    df.loc[df.year < 1946, ["interstate_5y", "interstate_1y"]] = np.nan
    g5 = grp["ev_gml"].transform(lambda s: s.shift(-1)[::-1].rolling(H, min_periods=H).max()[::-1])
    df["fatal_mid_5y"] = g5
    df["force_5y"] = grp["ev_force"].transform(lambda s: s.shift(-1)[::-1].rolling(H, min_periods=H).max()[::-1])
    df.loc[df.year + H > LAST_OBS_YEAR, "interstate_5y"] = np.nan
    df.loc[df.year + 1 > LAST_OBS_YEAR, "interstate_1y"] = np.nan
    df["first_y"] = np.array([extra.get((a, b), -9999) for a, b in zip(df.a, df.b)])
    df["entered"] = df.year >= df.first_y
    df.loc[~df.entered, ["interstate_5y", "interstate_1y", "fatal_mid_5y", "force_5y"]] = np.nan
    # current fighting: UCDP interstate conflict (1946+) and fatal dispute (GML, to 2010) kept separate
    df["ongoing"] = np.where(df.year >= 1946, (df.ev_ucdp == 1).astype(float), np.nan)
    # duration dependence: consecutive years the current conflict has been active (0 if none)
    act = (df.ev_ucdp == 1).astype(int)
    run = act.groupby([df.a, df.b]).transform(lambda s_: s_.groupby((s_ != s_.shift()).cumsum()).cumsum())
    df["conflict_run_years"] = np.where(df.year >= 1946, run * act, np.nan)
    df = _add_crisis_and_alignment(df)
    df = _add_events(df)
    return df


EV_RATIO = {"n_material_conflict": 0.731, "n_fight": 0.737, "n_threats": 3.013, "n_force_posture": 0.619}   # POLECAT/ICEWS, dyads


def _add_events(df):
    """machine-coded hostile events between the pair (ICEWS 1995-2017, POLECAT 2018-2024 and GDELT 2025, each rescaled to
    ICEWS levels by splice ratios on overlapping years; 3-year means, log). Without the GDELT file, 2025 carries 2024."""
    f = RAW / "events"
    try:
        ic = pd.read_csv(f / "icews_dy.csv")
        pc = pd.read_csv(f / "polecat_dy.csv")
    except FileNotFoundError:
        for c in EV_FEATS:
            df[c] = np.nan
        return df
    cols = list(EV_RATIO)
    ic, pc = ic[ic.year <= 2017], pc[pc.year >= 2018].copy()
    for c in cols:
        pc[c] = pc[c] / EV_RATIO[c]
    pc.loc[pc.year == 2024, cols] *= 365 / 181                    # POLECAT ends 2024-06-29
    parts = [ic, pc]
    # 2025 onward from GDELT (still updated; POLECAT stopped in June 2024), rescaled to POLECAT on their overlap
    # (data/raw/events/gdelt_splice.json) and then to ICEWS levels like POLECAT
    gd, gs = f / "gdelt_dy.csv", f / "gdelt_splice.json"
    if gd.exists() and gs.exists():
        import json as _json
        rr = _json.load(open(gs))["polecat_over_gdelt"]
        g = pd.read_csv(gd)
        g = g[g.year == 2025].copy()
        for c in cols:
            g[c] = g[c] * rr[c] / EV_RATIO[c]
        parts.append(g)
    ev = pd.concat(parts)[["iso3_a", "iso3_b", "year"] + cols].rename(columns={"iso3_a": "a", "iso3_b": "b"})
    df = df.merge(ev, on=["a", "b", "year"], how="left")
    last_ev = 2025 if (f / "gdelt_dy.csv").exists() and (f / "gdelt_splice.json").exists() else 2024
    known = (df.year >= 1995) & (df.year <= last_ev)
    for c in cols:
        df[c] = np.where(known, df[c].fillna(0), np.nan)
    df = df.sort_values(["a", "b", "year"])
    g = df.groupby(["a", "b"])
    m3 = lambda c: g[c].transform(lambda s_: s_.rolling(3, min_periods=1).mean())
    df["ev_mat3"], df["ev_fight3"] = np.log1p(m3("n_material_conflict")), np.log1p(m3("n_fight"))
    df["ev_threat3"] = np.log1p(m3("n_threats") + m3("n_force_posture"))
    if last_ev < 2025:                          # (no 2025 event data: carry 2024 forward)
        for c in EV_FEATS:
            df[c] = np.where(df.year == 2025, df.groupby(["a", "b"])[c].shift(1), df[c])
    return df.drop(columns=cols)


def _add_crisis_and_alignment(df):
    """International crises between the pair (ICB 1918-2022) and UN voting distance (1946-2025)."""
    try:
        icb = pd.read_csv(RAW / "interstate" / "icb_dy.csv")[["iso3_a", "iso3_b", "year", "icb_crisis", "icb_war_both"]]
        icb.columns = ["a", "b", "year", "icb_crisis", "icb_war"]
        df = df.merge(icb, on=["a", "b", "year"], how="left")
        known = (df.year >= 1918) & (df.year <= 2022)
        for c in ("icb_crisis", "icb_war"):
            df[c] = np.where(known, df[c].fillna(0), np.where(df.year > 2022, 0.0, np.nan))
        df = df.sort_values(["a", "b", "year"])
        df["crisis_past_10y"] = df.groupby(["a", "b"]).icb_crisis.transform(lambda s_: s_.rolling(10, min_periods=1).sum())
        df["crisis_now"] = df.icb_crisis
        un = pd.read_csv(RAW / "interstate" / "unga_dy.csv", usecols=["iso3_a", "iso3_b", "year", "ideal_absdiff"])
        un.columns = ["a", "b", "year", "un_dist"]
        df = df.merge(un, on=["a", "b", "year"], how="left")
        df["un_dist"] = df.groupby(["a", "b"]).un_dist.transform(lambda s_: s_.ffill(limit=2))
        df["un_dist_change_5y"] = df.un_dist - df.groupby(["a", "b"]).un_dist.shift(5)
        df = df.drop(columns=["icb_crisis", "icb_war"])
    except FileNotFoundError:
        for c in NEW_FEATS:
            df[c] = np.nan
    return df


FEATS = ["contig_land", "contig_sea", "log_cap_dist", "rival", "major_pair", "one_major", "abs_power_ratio",
         "power_shift_10y", "log_joint_gdp", "cinc_ratio", "both_nuclear", "one_nuclear", "allied", "terr_threat_max",
         "dem_min", "dem_max", "dem_gap", "civil_war_any", "coup_recent_max", "milex_max", "conflict_past_10y",
         "log_years_since_conflict", "ever_conflict", "ongoing", "war_now", "conflict_run_years"]
# tested 2026-10-24 and not adopted: ICB crises between the pair and UN voting distance (AUC 0.959 -> 0.961,
# Brier skill 0.174 -> 0.167); kept as columns for analysis
NEW_FEATS = ["crisis_past_10y", "crisis_now", "un_dist", "un_dist_change_5y"]
# hostile events between the pair (ICEWS/POLECAT), adopted 2026-10-25 for the 1-year model that drives the simulator:
# origins 2000-2024, AUC 0.965 -> 0.974, Brier skill 0.222 -> 0.252 (pair-clustered bootstrap: Brier gain 90% CI above 0);
# no gain for the 5-year model (AUC 0.965 -> 0.961), which keeps FEATS. Held at their origin value in simulation.
EV_FEATS = ["ev_mat3", "ev_fight3", "ev_threat3"]
# hostility after conflict, 1995-2024 (rho, b ongoing_t, b ongoing_t-1): used where the event history before an origin is
# too short to estimate it
EV_WAR_DEFAULT = {"ev_mat3": (0.794, 1.237, -0.323), "ev_fight3": (0.813, 1.341, -0.352), "ev_threat3": (0.791, 0.643, -0.053)}
LAST_EVENT_YEAR = 2025 if (RAW / "events" / "gdelt_dy.csv").exists() else 2024   # last year of dyadic event data
FEATS_1Y = FEATS + EV_FEATS
# the 1-year model is the simulator's onset hazard for pairs at peace (wars in progress use the duration table), so it
# is fitted on pairs at peace only and does not see the war-in-progress columns
FEATS_ON = [f for f in FEATS_1Y if f not in ("ongoing", "war_now", "conflict_run_years")]
GBM = dict(n_estimators=500, learning_rate=0.01, num_leaves=7, min_child_samples=40, subsample=0.7, subsample_freq=1,
           colsample_bytree=0.6, reg_lambda=10, verbose=-1)


class DyadModel:
    def fit(self, X, y):
        self.mu, self.sd = np.nanmean(X, 0), np.nanstd(X, 0) + 1e-6
        Z = self._z(X)
        self.lr = LogisticRegression(C=0.3, max_iter=4000).fit(Z, y)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.gb = lgb.LGBMClassifier(**GBM).fit(X, y)
        return self

    def _z(self, X):
        return np.nan_to_num(np.clip((X - self.mu) / self.sd, -6, 6))

    def predict_parts(self, X):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return self.lr.predict_proba(self._z(X))[:, 1], self.gb.predict_proba(X)[:, 1]

    def predict(self, X):
        a, b = self.predict_parts(X)
        return (a + b) / 2


def backtest(df, target="interstate_5y", origins=range(1975, 2020, 5), feats=None):
    feats = feats or FEATS
    rows = []
    for T in origins:
        tr = (df.year + (5 if target.endswith("5y") else 1) <= T) & df[target].notna()
        te = (df.year >= T) & (df.year < T + 5) & df[target].notna()
        if df.loc[tr, target].sum() < 10 or df.loc[te, target].sum() < 1:
            continue
        m = DyadModel().fit(df.loc[tr, feats].to_numpy(float), df.loc[tr, target].to_numpy())
        lr, gb = m.predict_parts(df.loc[te, feats].to_numpy(float))
        rows.append(pd.DataFrame({"origin": T, "y": df.loc[te, target].to_numpy(), "logit": lr, "gbm": gb,
                                  "avg": (lr + gb) / 2, "base": df.loc[tr, target].mean(),
                                  "naive": df.loc[te, "conflict_past_10y"].to_numpy() + df.loc[te, "rival"].to_numpy()}))
    bt = pd.concat(rows)
    out = {}
    for mname in ("logit", "gbm", "avg", "naive"):
        y, p_ = bt.y.to_numpy(), bt[mname].to_numpy()
        out[mname] = {"auc": roc_auc_score(y, p_),
                      "bss": 1 - np.mean((p_ - y) ** 2) / np.mean((bt.base - y) ** 2) if mname != "naive" else np.nan,
                      "events": int(y.sum()), "n": len(y)}
        top = bt.groupby("origin")[mname].transform(lambda s: s >= s.quantile(0.99))
        out[mname]["top1pct_capture"] = float(bt.y[top].sum() / bt.y.sum())
    return out, bt


WATCH = [("CHN", "TWN"), ("CHN", "USA"), ("CHN", "IND"), ("CHN", "JPN"), ("CHN", "PHL"), ("CHN", "VNM"),
         ("IND", "PAK"), ("ISR", "IRN"), ("IRN", "USA"), ("KOR", "PRK"), ("PRK", "USA"), ("RUS", "UKR"),
         ("RUS", "USA"), ("POL", "RUS"), ("EST", "RUS"), ("LVA", "RUS"), ("LTU", "RUS"), ("FIN", "RUS"),
         ("GBR", "RUS"), ("FRA", "RUS"), ("ARM", "AZE"), ("ERI", "ETH"), ("GUY", "VEN"), ("GRC", "TUR"),
         ("SAU", "IRN"), ("EGY", "ETH"), ("SDN", "SSD"), ("AFG", "PAK"), ("THA", "KHM")]
NATO_JOIN = {**{c: 1949 for c in ("USA", "GBR", "FRA", "CAN", "ITA", "NLD", "BEL", "LUX", "NOR", "DNK", "ISL", "PRT")},
             "GRC": 1952, "TUR": 1952, "DEU": 1955, "ESP": 1982, "CZE": 1999, "HUN": 1999, "POL": 1999,
             **{c: 2004 for c in ("BGR", "EST", "LVA", "LTU", "ROU", "SVK", "SVN")}, "ALB": 2009, "HRV": 2009,
             "MNE": 2017, "MKD": 2020, "FIN": 2023, "SWE": 2024}
NATO = {"USA", "GBR", "FRA", "DEU", "POL", "EST", "LVA", "LTU", "FIN", "NOR", "ROU", "TUR", "ITA", "CAN", "SWE",
        "DNK", "NLD", "BEL", "ESP", "PRT", "CZE", "SVK", "HUN", "BGR", "GRC", "HRV", "SVN", "ALB", "MNE", "MKD", "ISL", "LUX"}


WWI = [("DEU", "FRA"), ("DEU", "GBR"), ("DEU", "RUS"), ("AUT", "SRB"), ("AUT", "RUS"), ("BEL", "DEU"),
       ("AUT", "ITA"), ("RUS", "TUR"), ("GBR", "TUR"), ("DEU", "USA"), ("AUT", "FRA"), ("AUT", "GBR"),
       ("BGR", "SRB"), ("DEU", "ITA"), ("ROU", "AUT"), ("DEU", "ROU"), ("GRC", "TUR"), ("FRA", "TUR"), ("JPN", "DEU")]
WWII = [("DEU", "POL"), ("DEU", "GBR"), ("DEU", "FRA"), ("DEU", "RUS"), ("JPN", "USA"), ("GBR", "JPN"), ("CHN", "JPN"),
        ("GBR", "ITA"), ("FRA", "ITA"), ("DEU", "USA"), ("DEU", "NOR"), ("DEU", "NLD"), ("BEL", "DEU"), ("DEU", "GRC"),
        ("GRC", "ITA"), ("FIN", "RUS"), ("DEU", "SRB"), ("ITA", "USA"), ("HUN", "RUS"), ("ROU", "RUS"), ("JPN", "RUS")]


def retrodict(df, as_of, war_pairs, target="fatal_mid_5y"):
    """train only on outcomes observed by `as_of` (rows <= as_of - 5), then score every pair in
    `as_of` for the next five years; report where the real belligerent pairs rank."""
    tr = (df.year + H <= as_of) & df[target].notna()
    m = DyadModel().fit(df.loc[tr, FEATS].to_numpy(float), df.loc[tr, target].to_numpy())
    now = df[df.year == as_of].copy()
    now["p"] = m.predict(now[FEATS].to_numpy(float))
    now = now.sort_values("p", ascending=False).reset_index(drop=True)
    now["rank"] = np.arange(1, len(now) + 1)
    key = {(min(a, b), max(a, b)) for a, b in war_pairs}
    hit = now[[(a, b) in key for a, b in zip(now.a, now.b)]]
    y = now[target].fillna(0).to_numpy()
    auc = roc_auc_score(y, now.p) if 0 < y.sum() < len(y) else np.nan
    return {"as_of": as_of, "n_pairs": len(now), "auc_all_fatal_disputes": auc, "events": int(y.sum()),
            "belligerent_ranks": hit[["a", "b", "p", "rank", target]].to_dict("records"),
            "median_rank": float(hit["rank"].median()) if len(hit) else np.nan,
            "top": now.head(15)[["a", "b", "p", target]].to_dict("records")}


def observed_after(origin=LAST_OBS_YEAR):
    """interstate conflicts already recorded after the origin: [(a, b, year, war_level)] from UCDP candidate events"""
    f = RAW / "outcomes" / "ucdp_2025_interstate_dy.csv"
    if not f.exists():
        return []
    o = pd.read_csv(f)
    # only the year after the origin can be partly observed, and only after the last complete data year (a backtest
    # from 2024 must not see 2025's recorded conflicts)
    o = o[(o.year > max(origin, LAST_OBS_YEAR)) & (o.year <= origin + 1)]
    return [(min(a, b), max(a, b), int(y), int(w)) for a, b, y, w in zip(o.iso3_a, o.iso3_b, o.year, o.war_level)]


def known_at(df, origin):
    """pairs relevant at `origin`: all structural pairs, and pairs that entered by fighting once that fight is known
    (by the origin; for the live forecast also those recorded in the partly observed next year). A hindcast from
    1995 must not include pairs that became relevant only by fighting later."""
    return (df.first_y <= origin) | ((origin >= LAST_OBS_YEAR) & (df.first_y > LAST_OBS_YEAR))


# half-life (years) of the weights on past dyad-years in the continuation table; None = equal weights. Conflicts have
# lasted longer since 2010 (continuation 0.64 / 0.75 / 0.92 by run length 1 / 2 / 3+, against 0.34 / 0.37 / 0.59 in
# 1946-1989); 25 was chosen on origins 1970-1998 and cut the log-loss of continuation by 3.6% on origins 2000-2014
# (scripts/interstate_cont_test.py, 2026-10-28; 84 test outcomes)
CONT_HALFLIFE = 25


def continuation_rates(df, origin=LAST_OBS_YEAR, halflife="default"):
    """P(an active interstate conflict is still active next year) by how long it has run
    (1, 2, 3+ years), from UCDP dyad-years before `origin`, weighted 0.5^((origin - year) / CONT_HALFLIFE).
    Too few cases for the learners, so a transparent duration table is used for wars in progress."""
    halflife = CONT_HALFLIFE if halflife == "default" else halflife
    d = df[(df.ongoing == 1) & (df.year < origin)].copy()
    nxt = df.set_index(["a", "b", "year"]).ev_ucdp
    d["next"] = [nxt.get((a, b, y + 1), np.nan) for a, b, y in zip(d.a, d.b, d.year)]
    d = d[d["next"].notna()]
    w = np.ones(len(d)) if halflife is None else 0.5 ** ((origin - d.year.to_numpy()) / halflife)
    y = d["next"].astype(float).to_numpy()
    out = {}
    for r_ in (1, 2, 3):
        sel = ((d.conflict_run_years >= 3) if r_ == 3 else (d.conflict_run_years == r_)).to_numpy()
        out[r_] = float(np.sum(w[sel] * y[sel]) / np.sum(w[sel])) if sel.sum() >= 5 else 0.5
    return out


# formal defence pacts signed after ATOP ends (2018): (a, b, year signed)
POST_ATOP = [("FRA", "GRC", 2021),     # strategic partnership with a mutual-assistance clause
             ("PRK", "RUS", 2024),     # Comprehensive Strategic Partnership Treaty, art. 4 mutual military assistance
             ("PAK", "SAU", 2025)]     # Strategic Mutual Defence Agreement, September 2025
# security patrons without a defence treaty: (patron, client) -> range of the yearly chance the patron joins a new war
# against the client, sampled per world (assumptions; see world.QUASI_RANGES)
QUASI_ALLIES = {("USA", "TWN"): "Taiwan Relations Act 1979, strategic ambiguity",
                ("USA", "ISR"): "US strikes on Iran in June 2025 and the 2026 Gulf war",
                ("RUS", "IRN"): "2025 Comprehensive Strategic Partnership Treaty (no mutual-defence clause)",
                ("CHN", "PAK"): "'all-weather' strategic partnership, arms supply, no defence treaty"}


def alliances_2024(codes, c2i, origin=LAST_OBS_YEAR):
    """defence-pact partners at `origin`: ATOP defence pacts in force that year (ATOP ends in 2018)
    plus NATO members that had joined by then and the later pacts in POST_ATOP."""
    atop = _rda("atop_alliance")
    atop = atop[(atop.atop_defense == 1) & (atop.year == min(origin, int(atop.year.max())))]
    allies = {c: set() for c in codes}
    for a, b in zip(atop.ccode1, atop.ccode2):
        if int(a) in c2i and int(b) in c2i:
            allies[c2i[int(a)]].add(c2i[int(b)])
            allies[c2i[int(b)]].add(c2i[int(a)])
    nato = [c for c in NATO if c in allies and NATO_JOIN.get(c, 9999) <= origin]
    for c in nato:
        allies[c] |= set(nato) - {c}
    for a, b, y in POST_ATOP:
        if y <= origin and a in allies and b in allies:
            allies[a].add(b)
            allies[b].add(a)
    return {k: sorted(v) for k, v in allies.items()}


def run(verbose=True, origin=LAST_OBS_YEAR, backtests=True, df=None):
    df = build(verbose) if df is None else df
    res = {}
    for target, origins in ((("interstate_5y", range(1975, 2020, 5)), ("fatal_mid_5y", range(1900, 2006, 5)),
                              ("force_5y", range(1900, 2006, 5)))
                            if backtests else ()):
        res[target], _ = backtest(df, target, origins)
        if verbose:
            print(f"  backtest {target}: " + ", ".join(f"{m} AUC {v['auc']:.3f}" + (f" BSS {v['bss']:+.3f}" if v['bss'] == v['bss'] else "")
                                                     + f" top1% {v['top1pct_capture']:.2f}" for m, v in res[target].items()))
    # forecast 2025-2029 from 2024
    tr = df.interstate_5y.notna() & (df.year + H <= origin)
    m5 = DyadModel().fit(df.loc[tr, FEATS].to_numpy(float), df.loc[tr, "interstate_5y"].to_numpy())
    trg = df.fatal_mid_5y.notna() & (df.year + H <= origin)
    mg = DyadModel().fit(df.loc[trg, FEATS].to_numpy(float), df.loc[trg, "fatal_mid_5y"].to_numpy())
    now = df[(df.year == origin) & known_at(df, origin)].copy()
    Xn = now[FEATS].to_numpy(float)
    now["p_conflict_5y"] = m5.predict(Xn)
    # wars in progress: continuation from the duration table, plus renewed fighting if they stop
    cont = continuation_rates(df, origin)
    run_y = now.conflict_run_years.fillna(0).clip(0, 3).astype(int).to_numpy()
    p_cont = np.array([cont.get(r_, 0.5) if r_ > 0 else 0.0 for r_ in run_y])
    # (the 5-year model already sees that the war is in progress; next year's continuation is a floor, not an
    # extra chance on top of it)
    now["p_conflict_5y"] = np.where(run_y > 0, np.maximum(p_cont, now.p_conflict_5y), now.p_conflict_5y)
    now["p_fatal_mid_5y"] = mg.predict(Xn)
    trf = df.force_5y.notna() & (df.year + H <= origin)
    mf = DyadModel().fit(df.loc[trf, FEATS].to_numpy(float), df.loc[trf, "force_5y"].to_numpy())
    now["p_force_5y"] = np.maximum(mf.predict(Xn), now.p_conflict_5y)     # a 25-death conflict is also a use of force
    # fighting already recorded after the origin is not a forecast
    obs = {(a, b) for a, b, y, w in observed_after(origin) if y <= origin + H}
    now["observed"] = [float((a, b) in obs) for a, b in zip(now.a, now.b)]
    lr, gb = m5.predict_parts(Xn)
    lr = np.where(run_y > 0, np.maximum(p_cont, lr), lr)
    gb = np.where(run_y > 0, np.maximum(p_cont, gb), gb)
    now["p_lo"], now["p_hi"] = np.minimum(lr, gb), np.maximum(lr, gb)
    for c in ("p_conflict_5y", "p_lo", "p_hi", "p_force_5y", "p_fatal_mid_5y"):
        now[c] = np.where(now.observed == 1, 1.0, now[c])
    now = now.sort_values("p_conflict_5y", ascending=False)
    blocs = {"Russia vs any NATO member": now[((now.a == "RUS") & now.b.isin(NATO)) | ((now.b == "RUS") & now.a.isin(NATO))]}
    bloc_out = {k: float(1 - np.prod(1 - v.p_conflict_5y.to_numpy())) for k, v in blocs.items()}
    base_rate = float(df.loc[tr, "interstate_5y"].mean())
    # 1-year model for the simulator (logit + GBM average; the simulator uses the logit part on the GPU)
    t1 = df.interstate_1y.notna() & (df.year + 1 <= origin) & (df.ongoing == 0)
    m1 = DyadModel().fit(df.loc[t1, FEATS_ON].to_numpy(float), df.loc[t1, "interstate_1y"].to_numpy())
    ww = {"WWI (as of 1913)": retrodict(df, 1913, WWI), "WWII (as of 1938)": retrodict(df, 1938, WWII)} if backtests else {}
    return {"backtest": res, "world_wars": ww, "top": now.head(40)[["a", "b", "p_conflict_5y", "p_lo", "p_hi", "p_fatal_mid_5y", "p_force_5y", "ongoing", "observed", "rival"]],
            "watch": now.set_index(["a", "b"]).reindex([(min(a, b), max(a, b)) for a, b in WATCH])[["p_conflict_5y", "p_lo", "p_hi", "p_fatal_mid_5y", "p_force_5y", "ongoing", "observed"]].reset_index(),
            "blocs": bloc_out, "base_rate": base_rate, "df": df, "model_1y": m1, "continuation": cont, "origin": origin,
            "n_pairs": int(df[["a", "b"]].drop_duplicates().shape[0])}


WATCH_NAMED = [("China-Taiwan", "CHN", "TWN"), ("China-United States", "CHN", "USA"), ("China-India", "CHN", "IND"),
               ("China-Japan", "CHN", "JPN"), ("China-Philippines", "CHN", "PHL"), ("India-Pakistan", "IND", "PAK"),
               ("Iran-Israel", "IRN", "ISR"), ("Iran-United States", "IRN", "USA"), ("North-South Korea", "PRK", "KOR"),
               ("Russia-Ukraine", "RUS", "UKR"), ("Russia-United States", "RUS", "USA"), ("Russia-Poland", "POL", "RUS"),
               ("Armenia-Azerbaijan", "ARM", "AZE"), ("Ethiopia-Eritrea", "ERI", "ETH"), ("Venezuela-Guyana", "GUY", "VEN"),
               ("Greece-Turkey", "GRC", "TUR"), ("Saudi Arabia-Iran", "IRN", "SAU"),
               # water disputes (water.WATER_DISPUTES)
               ("Egypt-Ethiopia", "EGY", "ETH"), ("Afghanistan-Iran", "AFG", "IRN"), ("Iraq-Turkey", "IRQ", "TUR")]


def sim_inputs(r):
    """everything the simulator's interstate component needs, from a run() result."""
    df = r["df"]
    origin = r.get("origin", LAST_OBS_YEAR)
    now = df[(df.year == origin) & known_at(df, origin)].copy()
    # alignment target: new interstate conflicts per pair-year at peace, over the decade before the origin
    # (pairs that are not both democracies: the democratic-peace correction handles the others, and aligning over
    # all pairs would push their share of wars onto the remaining ones)
    def onset_rate(sub):
        sub = sub.sort_values(["a", "b", "year"])
        # (a pair that entered by fighting counts only after its first conflict, which is not an onset at risk)
        at_peace = (sub.groupby(["a", "b"]).ev_ucdp.shift(1) == False) & ~(sub.dem_min > 0.6) & (sub.year > sub.first_y)
        return float(((sub.ev_ucdp == True) & at_peace).sum() / max(at_peace.sum(), 1))
    new_rate = onset_rate(df[(df.year > origin - 10) & (df.year <= origin) & (df.year >= 1946)])
    # long-run rate (1946 to the origin), toward which the simulated rate drifts back
    long_rate = onset_rate(df[(df.year >= 1946) & (df.year <= origin)])

    # calibration-in-the-large: observed onsets / what the 1-year model predicted for the same pair-years
    # (pairs at peace, not both democracies). Unlike matching today's mean hazard to a historical mean, this is
    # immune to changes in the mix of pairs.
    # out-of-sample predictions for the long-run calibration: the 1-year model refitted with each block of years held
    # out (2026-10-25: observed / predicted 1.08 over 1946-2024; 2.09 for 2015-2024 alone, against 1.66 for the
    # model fitted on all years)
    t1 = df.interstate_1y.notna() & (df.year + 1 <= origin) & (df.ongoing == 0)
    oof = pd.Series(np.nan, index=df.index)
    for lo_y in range(1946, origin, 10):
        te = t1 & (df.year >= lo_y) & (df.year < lo_y + 10)
        if te.any():
            tr_ = t1 & ~((df.year >= lo_y - 1) & (df.year < lo_y + 11))     # one-year buffer either side
            mb = DyadModel().fit(df.loc[tr_, FEATS_ON].to_numpy(float), df.loc[tr_, "interstate_1y"].to_numpy())
            oof[te] = mb.predict(df.loc[te, FEATS_ON].to_numpy(float))

    def calib(sub, pred_of):
        sub = sub[(sub.ongoing == 0) & ~(sub.dem_min > 0.6) & sub.interstate_1y.notna() & oof.loc[sub.index].notna()]
        return float((sub.interstate_1y.sum() + 0.5) / max(pred_of(sub).sum(), 1e-6))
    # recent decade: how far the model actually used (trained on all years, that decade included) falls short of
    # its onsets; long run: out-of-sample, where the in-sample fit would flatter it
    ratio_recent = calib(df[(df.year >= origin - 10) & (df.year < origin)],
                         lambda s: r["model_1y"].predict(s[FEATS_ON].to_numpy(float)))
    ratio_long = calib(df[(df.year >= 1946) & (df.year < origin)], lambda s: oof.loc[s.index])
    p = get_panel()
    c2i = _ccode_to_iso(set(p.codes))
    pairs = lambda a, bs: [(min(a, b), max(a, b)) for b in bs if b != a]
    blocs = {"Russia vs any NATO member": pairs("RUS", NATO),
             "Any armed clash between nuclear-armed states": [(a, b) for a, b in zip(now.a, now.b)
                                                if now.loc[(now.a == a) & (now.b == b), "both_nuclear"].iloc[0] == 1],
             "Any armed clash between great powers": [(a, b) for a, b, m in zip(now.a, now.b, now.major_pair) if m == 1]}
    # per-pair anchor: log-odds gap between the full 1-year ensemble (logit + GBM) and its logit part,
    # which is all the simulator evaluates each year; the gap fades over time in simulation
    # event hostility is a short-lived signal: in simulation each pair's value decays from the origin toward the pair's
    # own 1995..origin mean at the annual rate implied by its 5-year autocorrelation of deviations (0.46-0.62)
    # (event data end in 2024; the origin row of 2025 is a copy and is left out)
    hist = df[(df.year >= 1995) & (df.year <= min(origin, LAST_EVENT_YEAR))].sort_values(["a", "b", "year"])
    ev_mean, ev_rho, ev_war = {}, {}, {}
    for c in EV_FEATS:
        m = hist.groupby(["a", "b"])[c].transform("mean")
        dev = hist[c] - m
        l5 = dev.groupby([hist.a, hist.b]).shift(5)
        ok = l5.notna() & dev.notna()
        den = float(np.sum(l5[ok] ** 2))
        # too little event history before the origin (hindcasts from the 1990s): a typical rate
        ev_rho[c] = float(max(np.sum(dev[ok] * l5[ok]) / den, 0.0) ** 0.2) if ok.sum() > 100 and den > 0 else 0.55
        pm = hist.groupby(["a", "b"])[c].mean()
        ev_mean[c] = np.array([pm.get((a, b), np.nan) for a, b in zip(now.a, now.b)])
        # a conflict raises the pair's hostility (estimated 1995-2024: about 2-3 sd above its mean while it lasts):
        # deviation_t = rho dev_t-1 + b0 ongoing_t + b1 ongoing_t-1, so a simulated war is followed by the hostility
        # that followed real ones, not by the pair's calm average (independent review 2026-10-29)
        on = hist.ongoing.astype(float)
        on1 = on.groupby([hist.a, hist.b]).shift(1)
        l1 = dev.groupby([hist.a, hist.b]).shift(1)
        okw = l1.notna() & dev.notna() & on1.notna()
        if okw.sum() > 1000 and on[okw].sum() >= 30:
            A_ = np.c_[l1[okw], on[okw], on1[okw], np.ones(int(okw.sum()))]
            bw = np.linalg.lstsq(A_, dev[okw].to_numpy(float), rcond=None)[0]
            ev_war[c] = (float(np.clip(bw[0], 0, 0.95)), float(bw[1]), float(bw[2]))
        else:
            ev_war[c] = EV_WAR_DEFAULT[c]
    # evaluated with event hostility at each pair's own mean: today's spike is carried by the decaying logit part,
    # the offset keeps only the pair's lasting non-linear risk
    calm = now.copy()
    for c in EV_FEATS:
        calm[c] = np.where(np.isfinite(ev_mean[c]), ev_mean[c], calm[c])
    lr, gb = r["model_1y"].predict_parts(calm[FEATS_ON].to_numpy(float))
    lo = lambda q: np.log(np.clip(q, 1e-5, 1 - 1e-5) / (1 - np.clip(q, 1e-5, 1 - 1e-5)))
    pair_offset = np.clip(lo((lr + gb) / 2) - lo(lr), -3, 3)
    # structural uncertainty: half the log-odds gap between the two model families (logit, GBM)
    pair_spread = np.clip(np.abs(lo(gb) - lo(lr)) / 2, 0, 1.5)
    # each pair's conflict years origin-10 .. origin-1 (the simulator's rolling 10-year count, as in training)
    oc = df[(df.year >= origin - 10) & (df.year < origin)].pivot_table(index=["a", "b"], columns="year", values="ongoing")
    oc = oc.reindex(columns=range(origin - 10, origin)).reindex(list(zip(now.a, now.b)))
    hist10 = np.nan_to_num(oc.to_numpy(float))
    return {"now": now, "model_1y": r["model_1y"], "continuation": r["continuation"], "pair_offset": pair_offset,
            "pair_spread": pair_spread,
            "hist10": hist10,
            "allies": alliances_2024(list(p.codes), c2i, origin), "onset_rate": new_rate, "onset_rate_long": long_rate,
            "calib_recent": ratio_recent, "calib_long": ratio_long, "observed": observed_after(origin),
            "ev_mean": ev_mean, "ev_rho": ev_rho, "ev_war": ev_war,
            "watch_pairs": [(n, min(a, b), max(a, b)) for n, a, b in WATCH_NAMED], "blocs": blocs}
