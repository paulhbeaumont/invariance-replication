#!/usr/bin/env python3
"""Render the cached cell panel as Stata datasets, for the Stata port.

ONLY THE PRIMITIVES CROSS OVER. The cell file carries employment in the two
periods and the keys; the frame file carries the shock and the instrument. Every
growth rate, every weight, every county aggregate and every sample mask is
rebuilt on the Stata side from these, so `90_compare_python.do` compares two
independent constructions rather than one construction estimated twice. Exporting
`w_mpgr` or `mpgr` would make the comparison a test of Stata's regression command
and nothing else.

The cells are the pruned support: negative-employment cells excluded, cells
exactly zero in both periods excluded, county totals rebuilt from what survives.
That is the same support the Python producers estimate on, which is the point.

`--poisson` EXPORTS A DIFFERENT SUPPORT, AND IT HAS TO. Table 6 estimates on the
LIVE-CELL support - every county-industry that has employment in at least one
window, keeping the windows where it has none - because Poisson's invariance
needs the per-county cell count to be constant across windows and the pruned
support's is not (37_table_poisson.py's `live_cells` sets out why, and the
falsification run measures what happens if it is ignored). Exporting the pruned
cells here and running ppmlhdfe on them would reproduce the Python's disagreement
with itself rather than test the port. The two exports are written side by side
under different names for that reason, never merged.
"""
import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

H = Path(__file__).resolve()
sys.path.insert(0, str(H.parent))
_s = importlib.util.spec_from_file_location("md", H.parent / "mpgr_disagg.py")
md = importlib.util.module_from_spec(_s)
_s.loader.exec_module(md)
_p = importlib.util.spec_from_file_location("t37", H.parent / "37_table_poisson.py")
t37 = importlib.util.module_from_spec(_p)
_p.loader.exec_module(t37)

a = argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter)
a.add_argument("--horizon", type=int, nargs="+", default=[2, 3, 4])
a.add_argument("--rung", nargs="+", default=list(md.RUNGS))
a.add_argument("--share", default="fixed5")
a.add_argument("--level", type=float, default=5.0)
a.add_argument("--period", default="pre")
a.add_argument("--poisson", action="store_true",
               help="export the LIVE-CELL support for Table 6 instead of the "
                    "pruned one; see the module docstring")
g = a.parse_args()

# Under INVARIANCE_SCRATCH_ROOT this moves off Dropbox; stata/00_launcher.do reads
# the same variable, so $cells follows without a second place to keep in step.
OUT = ((md._SCRATCH / "cells_stata") if md._SCRATCH is not None
       else md.dc.CURRENT_PAPER_DATA_ROOT / "temp" / "cells_stata")
OUT.mkdir(parents=True, exist_ok=True)
md.dc.LOG_DIR.mkdir(parents=True, exist_ok=True)
tag = "_poisson" if g.poisson else ""
tee = md.Tee(md.dc.LOG_DIR
             / f"41_export_cells_stata{tag}_{g.period}_{g.share}.log")
t0 = time.time()

rows = []
for rung in g.rung:
    for h in g.horizon:
        if g.poisson:
            # Table 6's own support, built the way 37 builds it rather than
            # reimplemented, so the two cannot drift apart.
            frame_df = md.t29.build_frame(g.period, h, g.share, g.level)
            yr_a = frame_df["year"].to_numpy()
            d = t37.build_cells(frame_df, rung, h)
            live = t37.live_cells(d, yr_a, rung)
            d = {k: (v[live] if isinstance(v, np.ndarray) and len(v) == len(live)
                     else v) for k, v in d.items()}
            ccg_a = frame_df["ccg"].to_numpy()
            shock_a, z_a = frame_df["shock"].to_numpy(), frame_df["z"].to_numpy()
            n_frame = len(frame_df)
        else:
            p = md.Panel(h, rung, g.share, g.level, period=g.period, quiet=True)
            d = p.d
            ccg_a, yr_a, shock_a, z_a, n_frame = p.ccg, p.yr, p.shock, p.z, p.n
        cy = d["f_cy"]

        # The frame, once per horizon: the shock and the instrument only. The
        # county aggregates are deliberately NOT exported, so Stata has to
        # rebuild them from the cells and A.1.2 is retested on that side.
        frame = pd.DataFrame({
            "ccg": ccg_a.astype(np.int32),
            "year": yr_a.astype(np.int16),
            "shock": shock_a.astype(np.float64),
            "z": z_a.astype(np.float64)})
        fpath = OUT / f"frame_{g.period}_h{h}_{g.share}.dta"
        frame.to_stata(fpath, write_index=False, version=118)

        cells = pd.DataFrame({
            "ccg": ccg_a[cy].astype(np.int32),
            "year": yr_a[cy].astype(np.int16),
            # The code INDEX, not the NAICS code. Only its grouping matters, and
            # f_ky is exported beside it so Stata's industry-by-year factor is
            # the identical partition rather than a reconstruction of it.
            "code": (d["f_ky"] // d["n_year"]).astype(np.int32),
            "fe_ky": d["f_ky"].astype(np.int32),
            "cur": d["cur"].astype(np.float64),
            "lag": d["lag"].astype(np.float64)})
        cpath = OUT / f"cells{tag}_{g.period}_{rung}_h{h}.dta"
        cells.to_stata(cpath, write_index=False, version=118)

        mb = (cpath.stat().st_size + fpath.stat().st_size) / 1e6
        md.check(len(cells) == len(d["cur"]) and len(frame) == n_frame,
                 f"exported every row (h={h}, {md.PRETTY[rung]})",
                 f"{len(cells):,} cells, {len(frame):,} county-years, {mb:.0f} MB")
        rows.append(dict(rung=rung.replace("_alloc", ""), h=h, cells=len(cells),
                         county_years=n_frame, mb=mb))

print(f"\n  STATA EXPORT{tag.upper()}, {g.period}-1998, z_income_{g.share}\n")
print(f"  {'rung':8s}{'h':>3s}{'cells':>12s}{'cty-yrs':>10s}{'MB':>8s}")
for r in rows:
    print(f"  {r['rung']:8s}{r['h']:>3d}{r['cells']:12,d}{r['county_years']:10,d}"
          f"{r['mb']:8.0f}")
print(f"\n  wrote {OUT}")
print(f"  {len(md.w22.failures)} checks failed, {time.time() - t0:.1f}s")
tee.close()
