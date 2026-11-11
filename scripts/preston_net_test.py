"""Health and income after damage (code audit, 10 November 2026). simulate.PRESTON moves life expectancy and child
mortality with income above the country's trend, but uses income before war and climate damage (lg_next), although the
coefficients were estimated on recorded GDP, which includes such losses, and fertility already uses income after damage
(lg_next + dmg_cum + war_cum). simulate.PRESTON_NET = True uses income after damage for health too.

This is a consistency correction, not a claim of better accuracy (the term itself was hindcast-neutral: -0.00%).
Pre-registered decision (10 November 2026, written and committed before the variant run): PRESTON_NET is switched on in
version 57 if, on the adopt rule's hindcast set against the baseline v57_base (16 origins, three refits),
  (i) no statistic is worse by more than the 3% guardrail, and
  (ii) the composite is not worse than plus the baseline's noise floor (non-inferiority; it need not improve).
Otherwise it stays off and the inconsistency is listed in the limits.
  bash scripts/hindcast_tag.sh v57_base ; python -P scripts/adopt_score.py --floor v57_base
  bash scripts/hindcast_tag.sh v57_pnet --consts '{"simulate.PRESTON_NET": true}'
  python -P scripts/adopt_score.py v57_base v57_pnet ; python -P scripts/preston_net_test.py"""
import json
import os

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs", "adopt")
r = json.load(open(os.path.join(OUT, "v57_pnet_vs_v57_base.json")))
i = not r["guardrail_breaches"]
ii = r["noise_floor"] is not None and r["composite"] <= r["noise_floor"]
res = {"composite": r["composite"], "noise_floor": r["noise_floor"], "guardrail_breaches": r["guardrail_breaches"],
       "i_no_breach": bool(i), "ii_non_inferior": bool(ii), "switch_on": bool(i and ii)}
print(json.dumps(res, indent=1))
json.dump(res, open(os.path.join(OUT, "..", "preston_net_test.json"), "w"), indent=1)
