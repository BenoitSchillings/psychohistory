"""Ask the frontier-model analyst panel about the latest forecast; writes outputs/llm_panel.json."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from psychohistory import llm_panel  # noqa: E402

fc = json.loads((llm_panel.OUT / "forecast.json").read_text())
res = llm_panel.run(fc)
(llm_panel.OUT / "llm_panel.json").write_text(json.dumps(res, indent=1))
print(f"{len(res['questions'])} questions; answered by", {m: sum(m in q["panel"] for q in res["questions"]) for m in res["models"]})
