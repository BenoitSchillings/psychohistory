"""Analyst panel: frontier language models review the structural forecasts.

The statistical engine sees only data up to 2024 and nothing it cannot measure (leaders'
intentions, diplomatic signals, events after the data cut-off). Two model families (Claude via the
Claude Code CLI, Gemini via the Antigravity CLI) receive each question with the structural
probability and its main drivers, and return an adjusted probability with short reasons.

Their judgments cannot be backtested: the models have read the history they would be scored on.
So they are shown beside the structural forecast, not merged into it, and every judgment is
appended with its date to outputs/llm_ledger.jsonl, to be scored as the forecast window unfolds.
"""
from __future__ import annotations

import datetime as dt
import json
import re
import subprocess

from .data import ROOT

OUT = ROOT / "outputs"
LEDGER = OUT / "llm_ledger.jsonl"
MODELS = {
    "Claude": ["claude", "--output-format", "text", "-p"],
    "Gemini": ["agy", "--mode", "plan", "--model", "gemini-launch-candidate-high", "--print-timeout", "900s", "-p"],
}

PROMPT = """You are a superforecaster on a panel reviewing a statistical world-forecasting model.
Today's date is {today}. The model's data end in {origin}, so it knows nothing that happened since;
you may use everything you know, including events in 2025-2026.

For each question below you get the model's probability and its main drivers. Give your own
probability for the stated event and window. Think about base rates first, then adjust for
specific information. Be calibrated: do not round to 0 or 1 unless the event has certainly
happened or become impossible. If the event already occurred within the window, say so.

Questions (JSON):
{questions}

Do not use tools. Reply ONLY with a JSON array, one object per question, in the same order:
[{{"id": "...", "p": 0.xx, "reasons": "at most 30 words"}}, ...]
"""


def questions_from_forecast(fc, n_conflict=10, n_coup=6, n_autoc=6, n_collapse=6, extra=("USA", "CHN", "IND", "RUS")):
    qs = []
    o = fc["meta"]["origin"]
    w0, w1 = o + 1, o + 5
    cs = fc["countries"]
    by = lambda t: sorted([c for c in cs if c.get(t) is not None], key=lambda c: -c[t])
    spec = [("civil_conflict_5y", n_conflict, "a new civil armed conflict (>= 25 battle deaths a year, UCDP definition) starts"),
            ("coup_5y", n_coup, "a coup attempt (successful or failed, Powell & Thyne definition) takes place"),
            ("autocratization_5y", n_autoc, "the V-Dem electoral democracy index falls by at least 0.10 from its {origin} level"),
            ("econ_collapse_5y", n_collapse, "real GDP per person falls at least 10% below its {origin} level in some year")]
    chosen = []
    for t, n, text in spec:
        sel = by(t)[:n] + [c for c in cs if c["code"] in extra and c.get(t) is not None]
        seen = set()
        for c in sel:
            if c["code"] in seen:
                continue
            seen.add(c["code"])
            chosen.append({"id": f"{t}:{c['code']}", "event": f"In {c['name']}, between {w0} and {w1}, {text.format(origin=o)}.",
                           "model_p": round(c[t], 3), "drivers": c.get(t + "_drivers", [])[:3]})
    for w in fc.get("interstate", {}).get("watch", []):
        if w.get("p_conflict_5y") is None:
            continue
        qs.append({"id": f"interstate:{w['a']}-{w['b']}",
                   "event": f"Between {w0} and {w1}, an interstate armed conflict with >= 25 battle deaths in a year "
                            f"occurs between {w['a']} and {w['b']} (ISO codes; UCDP definition).",
                   "model_p": round(w["p_conflict_5y"], 3), "drivers": ["ongoing" if w.get("ongoing") else f"not ongoing in {o}"]})
    return chosen + qs


def ask(model, questions, today, origin, timeout=1800):
    prompt = PROMPT.format(today=today, origin=origin, questions=json.dumps(questions, indent=1))
    r = subprocess.run(MODELS[model] + [prompt], capture_output=True, text=True, timeout=timeout, cwd=str(ROOT))
    txt = r.stdout
    m = re.search(r"\[\s*\{.*\}\s*\]", txt, re.S)
    if not m:
        raise RuntimeError(f"{model}: no JSON in reply: {txt[:300]} {r.stderr[:300]}")
    return json.loads(m.group(0))


def run(fc, models=("Claude", "Gemini"), today=None):
    today = today or dt.date.today().isoformat()
    qs = questions_from_forecast(fc)
    out = {q["id"]: {**q, "panel": {}} for q in qs}
    for mname in models:
        try:
            ans = ask(mname, qs, today, fc["meta"]["origin"])
        except Exception as e:
            print(f"  !! {mname}: {e}")
            continue
        with open(LEDGER, "a") as f:
            for a in ans:
                if a.get("id") in out:
                    out[a["id"]]["panel"][mname] = {"p": float(a["p"]), "reasons": a.get("reasons", "")}
                    f.write(json.dumps({"date": today, "model": mname, "id": a["id"], "p": float(a["p"]),
                                        "event": out[a["id"]]["event"], "model_p": out[a["id"]]["model_p"]}) + "\n")
    return {"date": today, "questions": list(out.values()), "models": list(models)}
