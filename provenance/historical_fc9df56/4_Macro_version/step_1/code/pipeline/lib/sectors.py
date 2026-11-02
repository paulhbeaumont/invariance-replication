"""Predetermined EFSY NAICS2 sector shares, the county-composition covariates.

SECTORS / OMITTED define the sector partition; naics2_shares reads the
desuppressed CBP (EFSY) NAICS2 cells producer 11 writes and returns one row per
(ccg, base year).

Lifted verbatim from 21_rotemberg.py (SECTORS 21:76, OMITTED 21:82,
naics2_shares 21:205) when producer 22's balance battery needed the same three
sector shares as 21's Panel F correlates. Same partition on purpose: the two
exhibits then rest on one covariate basis, even though they are NOT the same
test (21 Panel F regresses county exposure on characteristics; 22 Panel A
regresses the national category shock on exposure-weighted characteristics --
different level, different side, different assumption; BHJ fn.37 warns against
reading one as confirming the other).

The fail-loud checks are the point of this file and are preserved exactly: a
missing year, a missing county, a zero denominator, an unclassified code, or a
degenerate omitted residual each stop the run rather than quietly producing a
share that is wrong.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LIB = Path(__file__).resolve().parent
PIPELINE = LIB.parent
CODE = PIPELINE.parent
sys.path.insert(0, str(CODE / "py"))
sys.path.insert(0, str(PIPELINE))
sys.path.insert(0, str(LIB))
import dcads_common as dc  # noqa: E402

NAICS2_DIR = dc.CURRENT_PAPER_DATA_ROOT / "temp" / "employment" / "naics2"

# EFSY NAICS2 sector groups. NAICS 92 (public administration) is deliberately
# ABSENT: EFSY is desuppressed CBP, which excludes government, so the 92 share
# is 0.6% of national employment in 1978 and 0.0% by 2014 - a degenerate
# regressor, collinear with the year effects in the later years. NAICS 54
# carries the RDTE contractors instead.
SECTORS = {
    "Manufacturing share": (31, 32, 33),
    "Prof./technical services share": (54,),
    "Non-tradable share": (44, 45, 72),
}
# Everything else in the 26-code EFSY NAICS2 panel is the omitted residual.
OMITTED = (11, 21, 22, 23, 42, 48, 49, 51, 52, 53, 55, 56, 61, 62, 71, 81,
           92, 95, 99)


def naics2_shares(base_years, units):
    """Predetermined EFSY NAICS2 sector shares, one row per (ccg, base year).

    Fails loud on a missing year, a missing county, a zero denominator, or a
    group total that reaches one - the last would leave no omitted residual.
    """
    keep = set(units.tolist())
    out = []
    for b in sorted(base_years):
        f = NAICS2_DIR / f"y{b}.parquet"
        if not f.exists():
            raise SystemExit(f"  FAIL: no EFSY NAICS2 cells for base year {b}")
        d = pd.read_parquet(f, columns=["ccg", "code", "emp"])
        d = d[d["ccg"].isin(keep)]
        tot = d.groupby("ccg")["emp"].sum()
        miss = keep - set(tot.index.tolist())
        if miss:
            raise SystemExit(f"  FAIL: {len(miss)} counties absent from the "
                             f"EFSY NAICS2 cells in {b}")
        if (tot <= 0).any():
            raise SystemExit(f"  FAIL: {int((tot<=0).sum())} zero-employment "
                             f"NAICS2 denominators in {b}")
        seen = set(d["code"].unique().tolist())
        unclassed = seen - {c for cs in SECTORS.values() for c in cs} - set(OMITTED)
        if unclassed:
            raise SystemExit(f"  FAIL: EFSY NAICS2 codes {sorted(unclassed)} in "
                             f"{b} are in neither a named sector nor the "
                             f"declared omitted residual")
        r = pd.DataFrame(index=tot.index)
        for name, codes in SECTORS.items():
            s = (d[d["code"].isin(codes)].groupby("ccg")["emp"].sum()
                 .reindex(tot.index).fillna(0.0))
            r[name] = (s / tot).to_numpy(np.float64)
        r["base_year"] = b
        out.append(r.reset_index())
    sh = pd.concat(out, ignore_index=True)
    # The three shares are collinear with the fixed effects only if the omitted
    # residual is degenerate. Individual small counties with no residual sector
    # are ordinary and are counted, not fatal.
    res = 1.0 - sh[list(SECTORS)].sum(axis=1)
    if float(res.std()) <= 0 or float(res.max()) <= 0:
        raise SystemExit("  FAIL: the omitted sector residual is degenerate; "
                         "the sector shares would be collinear with the effects")
    n0 = int((res <= 1e-12).sum())
    print(f"  NAICS2 shares: {len(sh):,} county-base-year cells, residual "
          f"sector mean {float(res.mean()):.3f}, {n0} cells with no residual")
    return sh
