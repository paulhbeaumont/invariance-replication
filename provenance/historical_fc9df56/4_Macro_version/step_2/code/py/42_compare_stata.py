#!/usr/bin/env python3
"""Compare the Stata port against the Python producers, coefficient by coefficient.

Bit-identity is not the target and is not achievable: the two sides demean with
different implementations, one alternating projections in numpy and one reghdfe,
each stopping on its own tolerance. The target is that no coefficient differs by
more than TOL relative, which is the claim "the Stata code reproduces the Python
code" made checkable.

What this does NOT test is whether the two agree because they share an input:
the Stata side receives only cell employment and the county-level shock and
instrument, and rebuilds every growth rate, weight, aggregate and mask itself.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

TOL = 1e-8
RESULT = Path(__file__).resolve().parents[2] / "result"

a = argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter)
a.add_argument("--period", default="pre", choices=["pre", "post"])
a.add_argument("--horizon", type=int, default=4,
               help="the horizon 34's ladder was run at; 35 and 36 carry theirs "
                    "in the rows rather than the filename")
a.add_argument("--share", default="fixed5")
g = a.parse_args()

P, H, S = g.period, g.horizon, g.share

# The era is part of every filename on both sides, so it is threaded rather than
# spelled out: a comparison that read the pre Python against the post Stata would
# report a disagreement of the wrong kind and say nothing about the port.
PAIRS = (
    ("36 standard growth",
     f"coeff_table_standard_{P}_{S}.csv",
     f"coeff_table_standard_stata_{P}_{S}.csv",
     ["rung", "h"],
     ["county", "cell_w", "cell_eq", "entry", "gap", "county_cap", "cell_cap",
      "gap_cap", "constr_err"]),
    ("35 log-gap decomposition",
     f"coeff_table_decomposition_{P}_{S}.csv",
     f"coeff_table_decomposition_stata_{P}_{S}.csv",
     ["rung", "h"],
     ["county", "cell_naive", "cell_equal", "cell_what", "cell_1n",
      "county_surv", "gap_naive", "gap_equal", "net_ext", "entry", "exit",
      "jensen", "renorm", "within", "across"]),
    ("34 disaggregation ladder",
     f"coeff_tables_payoff_{P}_h{H}_{S}.csv",
     f"coeff_tables_payoff_stata_{P}_h{H}_{S}.csv",
     ["spec"],
     ["county", "n2", "n2_iy", "n4", "n4_iy"]),
    # 37 is the one pair where the two sides run DIFFERENT estimators rather than
    # the same closed form twice: a hand-rolled IRLS against ppmlhdfe, each
    # stopping on its own criterion. Agreement here is a stronger statement than
    # the three above, and a weaker tolerance would be the honest cost of it -
    # so far none has been needed.
    ("37 Poisson ladder",
     f"coeff_table_poisson_{P}_h{H}_{S}.csv",
     f"coeff_table_poisson_stata_{P}_h{H}_{S}.csv",
     ["spec"],
     ["county", "n2", "n2_iy", "n2_cellfe", "n4", "n4_iy", "n4_cellfe"]),
)


def one(label, pyf, stf, keys, cols):
    a = pd.read_csv(RESULT / pyf)
    b = pd.read_csv(RESULT / stf)
    for f in (a, b):
        if "rung" in f.columns:
            f["rung"] = f["rung"].str.replace("_alloc", "", regex=False)
    a = a.sort_values(keys).reset_index(drop=True)
    b = b.sort_values(keys).reset_index(drop=True)
    if len(a) != len(b):
        print(f"  [FAIL] {label}: {len(a)} python rows against {len(b)} stata rows")
        return False, np.inf
    for k in keys:
        if not (a[k].astype(str).to_numpy() == b[k].astype(str).to_numpy()).all():
            print(f"  [FAIL] {label}: the `{k}` keys do not line up")
            return False, np.inf

    A = a[cols].to_numpy(np.float64)
    B = b[cols].to_numpy(np.float64)
    rel = np.abs(A - B) / np.maximum(np.abs(A), 1e-30)
    worst = float(rel.max())
    ok = worst < TOL
    i = np.unravel_index(rel.argmax(), rel.shape)
    print(f"  [{'ok ' if ok else 'FAIL'}] {label}: {A.size} coefficients, "
          f"max {worst:.2e} relative, worst at {cols[i[1]]} "
          f"({'/'.join(str(a.loc[i[0], k]) for k in keys)})")
    if not ok:
        for j, c in enumerate(cols):
            if rel[:, j].max() >= TOL:
                r = int(rel[:, j].argmax())
                print(f"         {c:12s} python {A[r, j]:.15g}  "
                      f"stata {B[r, j]:.15g}  rel {rel[r, j]:.2e}")
    return ok, worst


print(f"\n  STATA PORT AGAINST THE PYTHON PRODUCERS, tolerance {TOL:g} relative")
print(f"  period {P}, h{H} for 34, share {S}\n")
results = [one(*p) for p in PAIRS]
allok = all(r[0] for r in results)
n = sum(1 for _ in PAIRS)
print(f"\n  {n} tables compared, worst disagreement anywhere "
      f"{max(r[1] for r in results):.2e}")
print(f"  {'PASS' if allok else 'FAIL'} at {TOL:g} relative\n")
raise SystemExit(0 if allok else 1)
