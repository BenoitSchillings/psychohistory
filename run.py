#!/usr/bin/env python3
"""Psychohistory command line.

  python run.py data [--refresh]   download sources and build the panel
  python run.py backtest           rolling-origin backtest of the crisis models
  python run.py hindcast           simulate from past origins and score against history
  python run.py forecast           backtest + hindcast + 2024->2050 forecast -> outputs/forecast.json
  python run.py states             US state module -> outputs/us_states.json
  python run.py report             build outputs/seldon_report.html from outputs/forecast.json
  python run.py all                forecast + states + report
add --quick for a faster, lower-fidelity run.
"""
import argparse
import warnings

warnings.filterwarnings("ignore")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["data", "backtest", "hindcast", "forecast", "states", "report", "all"])
    ap.add_argument("--refresh", action="store_true", help="re-download all sources")
    ap.add_argument("--quick", action="store_true", help="fewer origins / scenarios")
    a = ap.parse_args()

    from psychohistory import pipeline
    pipeline.OUT.mkdir(exist_ok=True)
    if a.command == "data":
        from psychohistory.data import get_panel
        p = get_panel(refresh=True) if a.refresh else get_panel()
        print(f"{len(p.codes)} countries, {len(p.x)} series")
    elif a.command == "backtest":
        p, base, X, Yl = pipeline.prepare(a.refresh)
        summ, _, _ = pipeline.step_backtest(p, X, Yl)
        print(summ.round(3).to_string())
    elif a.command == "hindcast":
        p, base, X, Yl = pipeline.prepare(a.refresh)
        print(pipeline.step_hindcast(p, base, X, Yl)[0].round(3).to_string())
    elif a.command in ("forecast", "all"):
        pipeline.run_all(a.refresh, a.quick)
    if a.command in ("states", "all"):
        pipeline.run_states()
    if a.command in ("report", "all"):
        from psychohistory import report
        print("wrote", report.build())


if __name__ == "__main__":
    main()
