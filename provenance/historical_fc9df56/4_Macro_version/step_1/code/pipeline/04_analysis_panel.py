"""Stage 4: THE final dataset.

Writes <MACRO_DATA_ROOT>/temp/pipeline/analysis_panel.parquet: LONG, one row
per (ccg, year, h), canonical sort (ccg, year, h). Columns:

  z            composition instrument (locked constants)
  z_int        intensity instrument (one exposure x one national series,
               same locked window and rule)
  x_pos, x_net treatment (G_w - G_b) / P_{t-h} under the positive (locked)
               and net contract rules
  y_mpgr, y_std, y_log   employment growth outcomes over [t-h, t]
  E_b, E_t     base and endpoint EFSY employment
  P_b          population at t-h
  popw         county population in POP_BASE_YEAR (the fixed weight input)
  state, division, czone, msa, cea, ea   rung labels

The row set per h reproduces 57's build_county at the locked cell exactly:
finite z, x_pos, E_b, E_t with E_b > 0 and E_t > 0, counties with any
missing rung label dropped whole. Sample sizes are asserted against the
frozen counts.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PIPELINE = Path(__file__).resolve().parent
sys.path.insert(0, str(PIPELINE))
sys.path.insert(0, str(PIPELINE / "lib"))
import locked  # noqa: E402
import panel as pnl_lib  # noqa: E402
import dcads_common as dc  # noqa: E402

PIPE = dc.MACRO_DATA_ROOT / "temp" / "pipeline"


def horizon_list(argv=None):
    """The horizons to write. Default is the lock and nothing else.

    A horizon outside locked.HORIZONS has no frozen county-year count, so its
    row count is measured and printed rather than asserted.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", type=int, nargs="+", default=None,
                    help="horizons to write; default locked.HORIZONS. "
                         f"Ungated horizons available: {locked.EXTRA_HORIZONS}")
    a = ap.parse_args(argv)
    if a.horizons is None:
        return tuple(locked.HORIZONS)
    allowed = set(locked.HORIZONS) | set(locked.EXTRA_HORIZONS)
    bad = sorted(set(a.horizons) - allowed)
    if bad:
        raise SystemExit(f"  FAIL: horizon(s) {bad} are neither locked "
                         f"{locked.HORIZONS} nor declared ungated "
                         f"{locked.EXTRA_HORIZONS}.")
    return tuple(sorted(set(a.horizons)))


def main(horizons=None):
    t0 = time.time()
    horizons = tuple(locked.HORIZONS) if horizons is None else tuple(horizons)
    pnl = pd.read_parquet(PIPE / "county_panel_pipeline.parquet")
    units = np.array(sorted(pnl["ccg"].unique()))
    piv = {c: pnl.pivot(index="ccg", columns="year", values=c).reindex(units)
           for c in ("P", "E")}
    labels = pnl.drop_duplicates("ccg").set_index("ccg")[
        ["state", "division", "czone", "msa", "cea", "ea"]]
    con = pd.read_parquet(PIPE / "contracts_ccg_year.parquet")
    C = {tag: con.pivot(index="ccg", columns="year", values=f"dollars_{tag}")
              .reindex(units) for tag in ("net", "pos")}
    inst = pd.read_parquet(PIPE / "composition_instrument_ccg_year.parquet")
    assert locked.POP_BASE_YEAR in piv["P"].columns
    P0 = piv["P"][locked.POP_BASE_YEAR].to_numpy(np.float64)

    frames = []
    for h in horizons:
        years = [t for t in range(max(1977, 1975 + h), locked.TARGET_LAST + 1)]
        Z = (inst[inst["h"] == h]
             .pivot(index="ccg", columns="year", values="z")
             .reindex(index=units, columns=years).to_numpy(np.float64))
        Z_INT = (inst[inst["h"] == h]
                 .pivot(index="ccg", columns="year", values="z_int")
                 .reindex(index=units, columns=years).to_numpy(np.float64))
        Xb = piv["P"].reindex(columns=[y - h for y in years]).to_numpy(np.float64)
        P_b = Xb
        E_b = piv["E"].reindex(columns=[y - h for y in years]).to_numpy(np.float64)
        E_t = piv["E"].reindex(columns=years).to_numpy(np.float64)
        xs = {}
        for tag in ("net", "pos"):
            lag = {k: C[tag].reindex(columns=[y - k for y in years])
                       .set_axis(years, axis=1) for k in range(h + 1)}
            G_b = lag[h].to_numpy(np.float64)
            G_w = (sum(lag[k] for k in range(h)) / h).to_numpy(np.float64)
            xs[tag] = (G_w - G_b) / Xb

        ok = (np.isfinite(Z) & np.isfinite(xs["pos"]) & np.isfinite(E_b)
              & np.isfinite(E_t) & (E_b > 0) & (E_t > 0))
        ri, ci = np.where(ok)
        d = pd.DataFrame(dict(ccg=units[ri], year=np.array(years)[ci], h=h,
                              z=Z[ok], z_int=Z_INT[ok],
                              x_pos=xs["pos"][ok], x_net=xs["net"][ok],
                              E_b=E_b[ok], E_t=E_t[ok], P_b=P_b[ok],
                              popw=np.broadcast_to(P0[:, None], P_b.shape)[ok]))
        y_log, y_std, y_mpgr = pnl_lib.outcomes(d["E_t"], d["E_b"])
        d["y_log"], d["y_std"], d["y_mpgr"] = y_log, y_std, y_mpgr
        for c in labels.columns:
            d[c] = d["ccg"].map(labels[c])
        d = d[d[["czone", "msa", "cea", "ea", "state", "division"]]
              .notna().all(axis=1)]
        if h in locked.N_COUNTY:
            assert len(d) == locked.N_COUNTY[h], \
                (h, len(d), locked.N_COUNTY[h])
            note = "matches frozen counts"
        else:
            # No frozen count exists at this horizon, so the row count is
            # reported and not asserted. Record it in locked.N_COUNTY_UNGATED
            # after the first build so a later drift is visible.
            want = locked.N_COUNTY_UNGATED.get(h)
            if want is None:
                note = "UNGATED, count measured not asserted"
            else:
                assert len(d) == want, (h, len(d), want)
                note = "matches the recorded ungated count"
        assert d["ccg"].nunique() == locked.N_COUNTIES, h
        frames.append(d)
        print(f"h={h}: {len(d):,} rows, {d['ccg'].nunique():,} counties "
              f"({note}); z_int NaN on "
              f"{int(d['z_int'].isna().sum()):,} rows")

    out = (pd.concat(frames, ignore_index=True)
             .sort_values(["ccg", "year", "h"]).reset_index(drop=True))
    out.to_parquet(PIPE / "analysis_panel.parquet", index=False)
    print(f"analysis_panel: {len(out):,} rows -> {PIPE / 'analysis_panel.parquet'}")
    print(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main(horizon_list())
