#!/bin/sh
# leave-one-group-out with the tuned GBM
for g in "Economy / immiseration" "Demography" "Elites & inequality" "State & regime" "Violence & contagion" "Climate shocks" "Era" "Ideology & religion" "State capacity & coercion" "Fiscal stress & coups" "Institutions" "Leaders" "Economic contagion"; do
  python3 -W ignore scripts/exp.py "abl:-$g" --drop "Macro-finance,Commodities,$g" --model gbm2 2>&1 | grep "^\["
done
