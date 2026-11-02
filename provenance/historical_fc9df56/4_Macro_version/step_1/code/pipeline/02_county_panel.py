"""Stage 2: the county panel with geography.

Writes <MACRO_DATA_ROOT>/temp/pipeline/county_panel_pipeline.parquet:
ccg, year, Y (real income), P (population), E (EFSY employment), state,
division, czone, msa, cea, ea. One row per county-year, 1969-2018, screened
universe (3,058 counties). msa is "M"+CBSA for 2003-metropolitan counties,
"R"+state otherwise (57's convention).
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
import panel  # noqa: E402
import dcads_common as dc  # noqa: E402

PIPE = dc.MACRO_DATA_ROOT / "temp" / "pipeline"


def main():
    t0 = time.time()
    PIPE.mkdir(parents=True, exist_ok=True)
    B = panel.base_panel()
    units = np.array(sorted(B["ccg"].unique()))
    lab = panel.rung_labels(units)
    stmap = B.drop_duplicates("ccg").set_index("ccg")["state"]
    lab["state"] = lab["ccg"].map(stmap)
    lab["msa"] = np.where(lab["msa_raw"].notna(),
                          "M" + lab["msa_raw"].astype(str),
                          "R" + lab["state"].astype(str))
    out = B.merge(lab[["ccg", "czone", "msa", "cea", "ea"]], on="ccg", how="left")
    assert out["ccg"].nunique() == locked.N_COUNTIES
    assert len(out) == len(B)
    out = out.sort_values(["ccg", "year"]).reset_index(drop=True)
    out.to_parquet(PIPE / "county_panel_pipeline.parquet", index=False)
    print(f"county_panel_pipeline: {len(out):,} rows, "
          f"{out['ccg'].nunique():,} counties, years "
          f"{out['year'].min()}-{out['year'].max()}")
    print(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
