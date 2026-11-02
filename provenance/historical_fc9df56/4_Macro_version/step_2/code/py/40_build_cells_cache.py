#!/usr/bin/env python3
"""Build the estimation cell panel once per (period, rung, horizon), and cache it.

Every table producer then reads the same panel off disk instead of rebuilding it
from the year parquets in process, so no two exhibits can differ in their sample
and the construction checks that have to re-read the sources run here rather than
on every run of every producer.

The cache is keyed to the size and mtime of every source file it was derived
from, so rebuilding the cells or the candidate frame invalidates it
automatically. `--force` rebuilds regardless.
"""
import argparse
import importlib.util
import sys
import time
from pathlib import Path

import pandas as pd

H = Path(__file__).resolve()
sys.path.insert(0, str(H.parent))
_s = importlib.util.spec_from_file_location("md", H.parent / "mpgr_disagg.py")
md = importlib.util.module_from_spec(_s)
_s.loader.exec_module(md)

a = argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter)
a.add_argument("--horizon", type=int, nargs="+", default=[2, 3, 4])
a.add_argument("--rung", nargs="+", default=list(md.RUNGS))
a.add_argument("--share", default="fixed5")
a.add_argument("--level", type=float, default=5.0)
a.add_argument("--period", default="pre")
a.add_argument("--source", default="efsy", choices=list(md.SOURCES),
               help="which employment series to build the cells from; the "
                    "cache is keyed on it, so an EFSY cache is never handed "
                    "to a NETS run")
a.add_argument("--force", action="store_true", help="rebuild even if fresh")
g = a.parse_args()

md.dc.LOG_DIR.mkdir(parents=True, exist_ok=True)
tee = md.Tee(md.dc.LOG_DIR
             / f"40_build_cells_cache_{g.period}_{g.source}_{g.share}.log")
t0 = time.time()

rows = []
for rung in g.rung:
    for h in g.horizon:
        t1 = time.time()
        p = md.Panel(h, rung, g.share, g.level, period=g.period,
                     source=g.source, rebuild=g.force)
        d = p.d
        rows.append(dict(rung=rung.replace("_alloc", ""), h=h, source=g.source,
                         county_years=p.n, built=d["n_built"],
                         kept=len(d["cur"]), negative=d["n_neg"],
                         zero_both=d["n_dead"],
                         pct_dropped=100.0 * (1 - len(d["cur"]) / d["n_built"]),
                         seconds=time.time() - t1))

print(f"\n  CELL PANEL CACHE, {g.period}-1998, employment source {g.source}, "
      f"z_income_{g.share}, IQR{g.level:g} by year\n")
print(f"  {'rung':8s}{'h':>3s}{'cty-yrs':>10s}{'built':>12s}{'kept':>12s}"
      f"{'negative':>10s}{'zero both':>12s}{'dropped':>9s}{'sec':>8s}")
for r in rows:
    print(f"  {r['rung']:8s}{r['h']:>3d}{r['county_years']:10,d}{r['built']:12,d}"
          f"{r['kept']:12,d}{r['negative']:10,d}{r['zero_both']:12,d}"
          f"{r['pct_dropped']:8.1f}%{r['seconds']:8.1f}")

out = md.RESULT_DIR / f"cells_cache_{g.period}_{g.source}_{g.share}.csv"
pd.DataFrame(rows).to_csv(out, index=False, float_format="%.15g")
print(f"\n  cache in {md.CACHE_DIR}")
print(f"  wrote {out.relative_to(md.REPO_ROOT)}")
print(f"  {len(md.w22.failures)} checks failed, {time.time() - t0:.1f}s")
tee.close()
