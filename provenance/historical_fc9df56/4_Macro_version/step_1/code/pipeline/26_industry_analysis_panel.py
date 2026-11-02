"""WS1 step 2, part B (cont.): join the industry instrument onto the locked panel.

The locked `analysis_panel.parquet` is READ, never rewritten. This adds the four
industry-axis instrument columns to a copy and writes it beside the industry store,
so the sample, the outcome, the treatment and every label stay exactly the locked
ones and the ONLY thing that differs between an FSC-axis regression and an
industry-axis regression is which instrument column is used.

The row count is asserted against locked.N_COUNTY[h], which is what makes
"same sample, new instrument only" a checked claim rather than an intention.

Output: <MACRO_DATA_ROOT>/temp/pipeline_industry/analysis_panel_industry.parquet
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PIPELINE = Path(__file__).resolve().parent
sys.path.insert(0, str(PIPELINE.parent / "py"))
sys.path.insert(0, str(PIPELINE))
sys.path.insert(0, str(PIPELINE / "lib"))
import dcads_common as dc  # noqa: E402
import locked  # noqa: E402
import store  # noqa: E402

OUT = dc.MACRO_DATA_ROOT / "temp" / "pipeline_industry"
IND_COLS = ["L_ind", "z_ind", "z_ind_noloo", "z_ind_modal"]


def main():
    t0 = time.time()
    ap = store.load_analysis_panel()
    ind = pd.read_parquet(OUT / "industry_instrument_ccg_year.parquet")
    print(f"locked analysis panel: {len(ap):,} rows; "
          f"industry instrument: {len(ind):,} rows", flush=True)

    before = len(ap)
    m = ap.merge(ind, on=["ccg", "year", "h"], how="left", validate="one_to_one")
    assert len(m) == before, "the join changed the row count"

    for h in locked.HORIZONS:
        sub = m[m["h"] == h]
        assert len(sub) == locked.N_COUNTY[h], (h, len(sub), locked.N_COUNTY[h])
        assert sub["ccg"].nunique() == locked.N_COUNTIES, h
        fin = {c: int(np.isfinite(sub[c]).sum()) for c in IND_COLS}
        # The locked z is finite on every row of the locked sample by construction
        # (04's filter). The industry columns are built on the identical target
        # years and window rule, so they must be finite on exactly the same rows -
        # any shortfall is a window or merge error, not a data fact.
        assert all(v == len(sub) for v in fin.values()), (h, fin, len(sub))
        r = np.corrcoef(sub["z"], sub["z_ind"])[0, 1]
        r0 = np.corrcoef(sub["z"], sub["z_ind_noloo"])[0, 1]
        rm = np.corrcoef(sub["z_ind"], sub["z_ind_modal"])[0, 1]
        print(f"h={h}: {len(sub):,} rows (= locked.N_COUNTY), all industry columns "
              f"finite; corr(z, z_ind)={r:.4f} corr(z, z_ind_noloo)={r0:.4f} "
              f"corr(z_ind, z_ind_modal)={rm:.4f}", flush=True)

    m.to_parquet(OUT / "analysis_panel_industry.parquet", index=False)
    print(f"analysis_panel_industry: {len(m):,} rows, "
          f"{len(m.columns)} columns", flush=True)
    print(f"done in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
