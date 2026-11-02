"""Stage 1: contract panels from the parsed per-FY records.

Writes to <MACRO_DATA_ROOT>/temp/pipeline/:
  contracts_ccg_year.parquet           ccg x year real dollars, columns
                                       dollars_net and dollars_pos (floor 0),
                                       rectangular, zero-filled
  contracts_category_ccg_year.parquet  nonzero cells of the harmonized
                                       county x category x year tensors,
                                       long: rule, ccg, cat, year, dollars
                                       (float32, as accumulated)
  national_category_year.parquet       rule, cat, year, national dollars
  cats.csv                             top-NCAT categories, rank order

The county universe is the screened base panel's (3,058 counties). A county
absent from a year has ZERO contracts; the rectangle is asserted.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PIPELINE = Path(__file__).resolve().parent
sys.path.insert(0, str(PIPELINE))
sys.path.insert(0, str(PIPELINE / "lib"))
import locked  # noqa: E402
import contracts  # noqa: E402
import panel  # noqa: E402
import dcads_common as dc  # noqa: E402

PIPE = dc.MACRO_DATA_ROOT / "temp" / "pipeline"


def main():
    t0 = time.time()
    PIPE.mkdir(parents=True, exist_ok=True)
    B = panel.base_panel()
    units = np.array(sorted(B["ccg"].unique()))
    assert len(units) == locked.N_COUNTIES, len(units)

    CON = contracts.contract_matrices(units, locked.FIRST_YEAR, locked.LAST_YEAR)
    net = CON[("net", 0.0)]
    pos = CON[("positive", 0.0)]
    assert net.shape == pos.shape == (locked.N_COUNTIES,
                                      locked.LAST_YEAR - locked.FIRST_YEAR + 1)
    assert not net.isna().any().any() and not pos.isna().any().any()
    long = (net.stack().rename("dollars_net").reset_index()
               .merge(pos.stack().rename("dollars_pos").reset_index(),
                      on=["ccg", "year"]))
    long = long.sort_values(["ccg", "year"]).reset_index(drop=True)
    long.to_parquet(PIPE / "contracts_ccg_year.parquet", index=False)
    print(f"contracts_ccg_year: {len(long):,} rows "
          f"({locked.N_COUNTIES:,} counties x {net.shape[1]} years, zero-filled)")

    ARR, NAT, cats, years = contracts.category_panel(
        units, locked.FIRST_YEAR, locked.LAST_YEAR, harmonize=True)
    assert len(cats) == locked.NCAT
    pd.DataFrame({"rank": range(1, len(cats) + 1), "cat": cats}).to_csv(
        PIPE / "cats.csv", index=False)

    sparse_parts = []
    nat_parts = []
    for rule, tag in (("net", "net"), ("positive", "pos")):
        a = ARR[(rule, 0.0)]
        i, j, t = np.nonzero(a)
        sparse_parts.append(pd.DataFrame(dict(
            rule=tag, ccg=units[i], cat=np.array(cats, dtype=object)[j],
            year=np.array(years)[t], dollars=a[i, j, t])))
        n = NAT[(rule, 0.0)]
        nat_parts.append(n.stack().rename("dollars").reset_index()
                          .rename(columns={"level_0": "cat", "level_1": "year"})
                          .assign(rule=tag))
    sp = pd.concat(sparse_parts, ignore_index=True)
    sp.to_parquet(PIPE / "contracts_category_ccg_year.parquet", index=False)
    nt = pd.concat(nat_parts, ignore_index=True)
    nt.columns = [c if c in ("cat", "year", "dollars", "rule") else c
                  for c in nt.columns]
    nt.to_parquet(PIPE / "national_category_year.parquet", index=False)
    print(f"category cells: {len(sp):,} nonzero (both rules); "
          f"national panel {len(nt):,} rows; {len(cats)} categories")
    print(f"done in {time.time() - t0:.1f}s -> {PIPE}")


if __name__ == "__main__":
    main()
