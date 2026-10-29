"""ERA5 monthly means by hour of day (00, 06, 12, 18 UTC) of 2 m temperature and dew point, 1 degree grid, one file per
year, for humid-heat (wet-bulb) exposure by country-year. Needs a Copernicus CDS key in ~/.cdsapirc.
python -P scripts/heat_download.py [FIRST LAST] -> data/raw/heat/era5_td_<year>.nc"""
import os
import sys

import cdsapi

OUT = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "heat")
first, last = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (1960, 2025)
os.makedirs(OUT, exist_ok=True)
c = cdsapi.Client(quiet=True)
# the CDS queues one request per user and dataset, so years go five to a request, one request at a time; each request's
# file is split into one file per year
import xarray as xr
todo = [y for y in range(first, last + 1) if not os.path.exists(os.path.join(OUT, f"era5_td_{y}.nc"))]
for i in range(0, len(todo), 5):
    ys = todo[i:i + 5]
    tmp = os.path.join(OUT, f"era5_td_{ys[0]}_{ys[-1]}.part")
    c.retrieve("reanalysis-era5-single-levels-monthly-means", {
        "product_type": ["monthly_averaged_reanalysis_by_hour_of_day"],
        "variable": ["2m_temperature", "2m_dewpoint_temperature"],
        "year": [str(y) for y in ys], "month": [f"{m:02d}" for m in range(1, 13)], "time": ["00:00", "06:00", "12:00", "18:00"],
        "grid": [1.0, 1.0], "data_format": "netcdf", "download_format": "unarchived"}, tmp)
    ds = xr.open_dataset(tmp)
    tdim = "valid_time" if "valid_time" in ds.dims else "time"
    for y in ys:
        sub = ds.sel({tdim: ds[tdim].dt.year == y})
        sub.to_netcdf(os.path.join(OUT, f"era5_td_{y}.nc"))
    ds.close()
    os.remove(tmp)
    print(ys, flush=True)
