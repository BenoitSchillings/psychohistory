"""Write readable histories of a few simulated worlds (outputs/chronicles_raw.json -> outputs/chronicles.json).

Each history is written by a frontier model from the world's event log only. It is storytelling about one
sampled future, not a forecast: the forecast is the distribution over all simulated worlds.
"""
import json
import os
import re
import subprocess
import sys

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "outputs")

PROMPT = """You are a historian in the year 2065 writing a short history of the world since 2025, in the spirit of
the Encyclopedia Galactica entries in Asimov's Foundation. Your only source is the event log below, produced
by a statistical world simulator for ONE possible future ({kind_text}). Rules:
- Use only events in the log. Do not invent wars, coups, leaders, technologies, or numbers not in it.
- You may explain consequences in general terms (e.g. "the oil shock squeezed importers").
- Mention the biggest turning points with their years; skip minor items (say "a scattering of coups in the Sahel").
- Plain, vivid prose. 230-300 words. A title on the first line (no markdown symbols), then 3-4 paragraphs.
- World GDP per person is in 2011 international dollars (about $18,000 in 2024). Oil index: 2024 = 100.

Event log (JSON, one entry per year; empty fields mean nothing notable):
{log}
"""

KIND = {"typical": "a typical world, near the median of all simulated futures",
        "calm": "the calmest of the simulated worlds",
        "great_power_war": "a world in which the great powers go to war",
        "ai_typical": "a typical world in which intellectual work becomes nearly free from about 2030"}


def compact(events):
    keep = []
    for e in events:
        e = dict(e)
        if e["year"] % 5 and not any(k in e for k in ("onset", "coup", "new_interstate", "pandemic", "slump", "dem_drop")):
            continue
        keep.append(e)
    return json.dumps(keep, separators=(",", ":"))


def main():
    raw = json.load(open(os.path.join(OUT, "chronicles_raw.json")))
    out = []
    for w in raw:
        prompt = PROMPT.format(kind_text=KIND.get(w["kind"], w["kind"]), log=compact(w["events"]))
        r = subprocess.run(["claude", "--output-format", "text", "-p", prompt], capture_output=True, text=True, timeout=900)
        txt = r.stdout.strip()
        if not txt:
            print("!!", w["kind"], r.stderr[:300], file=sys.stderr)
            continue
        title, _, body = txt.partition("\n")
        out.append({"kind": w["kind"], "scenario": w["scenario"], "title": re.sub(r"^[#*\s]+|[*]+$", "", title).strip(),
                    "body": body.strip(), "world_gdp_2065": w["world_gdp_2065"], "great_power_war": w["great_power_war"],
                    "conflict_mean": w["conflict_mean"]})
        print("wrote", w["kind"], "-", out[-1]["title"])
    json.dump(out, open(os.path.join(OUT, "chronicles.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
