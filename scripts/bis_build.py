"""BIS consolidated banking statistics (data.bis.org bulk WS_CBS_PUB, public): foreign claims of each reporting
country's domestic banks on each counterparty country, immediate-counterparty basis, all sectors, instruments,
maturities and currencies, USD millions, end of year (Q4) -> data/raw/bis/cbs_bilateral.csv (rep, cp, year, claims)
python3 -I scripts/bis_build.py data/raw/bis/cbs.zip
"""
import csv
import io
import sys
import zipfile

z = zipfile.ZipFile(sys.argv[1])
f = io.TextIOWrapper(z.open(z.namelist()[0]), encoding="utf-8")
r = csv.reader(f)
h = next(r)
ix = {k.split(":")[0]: i for i, k in enumerate(h)}
want = {"L_MEASURE": "S", "CBS_BANK_TYPE": "4R", "CBS_BASIS": "F", "L_POSITION": "C", "L_INSTR": "A",
        "REM_MATURITY": "A", "CURR_TYPE_BOOK": "TO1", "L_CP_SECTOR": "A"}
code = lambda v: v.split(":")[0].strip()
out = open(sys.argv[1].replace("cbs.zip", "cbs_bilateral.csv"), "w", newline="")
w = csv.writer(out)
w.writerow(["rep", "cp", "year", "claims"])
n = 0
for row in r:
    if any(code(row[ix[k]]) != v for k, v in want.items()):
        continue
    t, v = row[ix["TIME_PERIOD"]], row[ix["OBS_VALUE"]]
    if not t.endswith("Q4") or v in ("", "NaN"):
        continue
    w.writerow([code(row[ix["L_REP_CTY"]]), code(row[ix["L_CP_COUNTRY"]]), t[:4], v])
    n += 1
print(n)
