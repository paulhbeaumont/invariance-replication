"""Stage 3: the two instruments at the locked constants.

Reads stage 1/2 persisted outputs and writes to
<MACRO_DATA_ROOT>/temp/pipeline/:
  category_shocks_year.parquet          h, cat, year, g (the national shock
                                        g_kt, own-mean scaled, clipped)
  composition_instrument_ccg_year.parquet  ccg, year, h, with
      L      predicted dollars (composition numerator)
      z      the composition instrument, L / population at t-h
      z_int  the intensity instrument, (county exposure over the locked
             window) x (one national procurement series); the single-factor
             companion, needed by the shifter-count diagnostics and carried
             into the final dataset
  NaN wherever the exposure window falls outside the data.

Exposure shares C̄ are not materialized per (t, h): they are one call to
lib.instrument.window_sums on the stage-1 tensor (materializing every window
would duplicate the tensor ~40x). Ties the composition outputs to the golden
fixtures before writing.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PIPELINE = Path(__file__).resolve().parent
sys.path.insert(0, str(PIPELINE.parent / "py"))
sys.path.insert(0, str(PIPELINE))
sys.path.insert(0, str(PIPELINE / "lib"))
import locked  # noqa: E402
import instrument  # noqa: E402
import store  # noqa: E402
import dcads_common as dc  # noqa: E402

PIPE = store.PIPE
GOLD = PIPE / "_golden"


def horizon_list(argv=None):
    """The horizons to build. Default is the lock and nothing else.

    A horizon outside locked.HORIZONS has no golden gk/L fixture, so the
    tie-out assertions below are skipped for it and the run says so. That is
    the whole difference between a locked horizon and an ungated one.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", type=int, nargs="+", default=None,
                    help="horizons to build; default locked.HORIZONS. "
                         f"Ungated horizons available: {locked.EXTRA_HORIZONS}")
    a = ap.parse_args(argv)
    if a.horizons is None:
        return tuple(locked.HORIZONS)
    allowed = set(locked.HORIZONS) | set(locked.EXTRA_HORIZONS)
    bad = sorted(set(a.horizons) - allowed)
    if bad:
        raise SystemExit(f"  FAIL: horizon(s) {bad} are neither locked "
                         f"{locked.HORIZONS} nor declared ungated "
                         f"{locked.EXTRA_HORIZONS}. Add to EXTRA_HORIZONS "
                         f"deliberately rather than passing them here.")
    return tuple(sorted(set(a.horizons)))


def main(horizons=None):
    t0 = time.time()
    horizons = tuple(locked.HORIZONS) if horizons is None else tuple(horizons)
    units, cats, years, arr, nat = store.load_tensor(locked.RULE)
    pnl = store.load_county_panel()
    _, piv = store.load_pivots(pnl, cols=("P",))
    Ppiv = piv["P"]
    C = store.load_con(locked.RULE).reindex(units)

    shock_rows, inst_rows = [], []
    for h in horizons:
        gated = h in locked.HORIZONS
        ty = store.target_years(h)
        gk = instrument.category_rates(nat, h)

        if gated:
            ref_gk = (pd.read_parquet(GOLD / f"gk_h{h}.parquet")
                        .pivot(index="cat", columns="year", values="g")
                        .reindex(index=cats, columns=years).to_numpy())
            assert np.array_equal(np.isnan(ref_gk), np.isnan(gk)), \
                f"gk NaN h={h}"
            assert np.allclose(ref_gk, gk, rtol=locked.RTOL_VEC,
                               atol=locked.ATOL_VEC, equal_nan=True), \
                f"gk h={h}"

        L, _ = instrument.predicted_dollars(arr, gk, years, ty, h,
                                            "roll", locked.SPAN, locked.LAG,
                                            locked.DATA_FIRST)
        if gated:
            ref_L = (pd.read_parquet(GOLD / f"L_h{h}.parquet")
                       .pivot(index="ccg", columns="year", values="L")
                       .reindex(index=units, columns=ty).to_numpy())
            assert np.array_equal(np.isnan(ref_L), np.isnan(L)), f"L NaN h={h}"
            assert np.allclose(ref_L, L, rtol=locked.RTOL_VEC,
                               atol=locked.ATOL_VEC, equal_nan=True), f"L h={h}"

        Xb = Ppiv.reindex(columns=[y - h for y in ty]).to_numpy(np.float64)
        Z = L / Xb

        # The intensity instrument at the same constants: one predetermined
        # exposure times one national series (the 51/55/56 construction).
        lag = {k: C.reindex(columns=[y - k for y in ty]).set_axis(ty, axis=1)
               for k in range(h + 1)}
        G_b = lag[h].to_numpy(np.float64)
        G_w = (sum(lag[k] for k in range(h)) / h).to_numpy(np.float64)
        natshift = np.nansum(G_w - G_b, axis=0) / np.nansum(Xb, axis=0)
        expo = np.full_like(Z, np.nan)
        for j, t in enumerate(ty):
            hi = t - h - locked.LAG
            lo = hi - locked.SPAN + 1
            if lo < locked.DATA_FIRST:
                continue
            yy = list(range(lo, hi + 1))
            num = C.loc[:, yy].sum(axis=1).to_numpy(np.float64)
            dpop = (Ppiv.loc[:, yy].sum(axis=1, min_count=len(yy))
                    .to_numpy(np.float64))
            expo[:, j] = num / dpop
        Z_INT = expo * natshift[None, :]

        gdf = pd.DataFrame(gk, index=cats, columns=years).stack()
        shock_rows.append(gdf.rename("g").reset_index()
                          .set_axis(["cat", "year", "g"], axis=1).assign(h=h))
        idf = pd.DataFrame(dict(
            ccg=np.repeat(units, len(ty)),
            year=np.tile(ty, len(units)),
            h=h, L=L.ravel(), z=Z.ravel(), z_int=Z_INT.ravel()))
        inst_rows.append(idf)
        tie = "gk and L tie to golden" if gated else "UNGATED, no golden"
        print(f"h={h}: {tie}; z built "
              f"({np.isfinite(Z).sum():,} finite cells), z_int built "
              f"({np.isfinite(Z_INT).sum():,} finite cells)")

    pd.concat(shock_rows, ignore_index=True).to_parquet(
        PIPE / "category_shocks_year.parquet", index=False)
    out = (pd.concat(inst_rows, ignore_index=True)
             .sort_values(["ccg", "year", "h"]).reset_index(drop=True))
    out.to_parquet(PIPE / "composition_instrument_ccg_year.parquet", index=False)
    print(f"composition_instrument_ccg_year: {len(out):,} rows "
          f"(columns L, z, z_int)")
    print(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main(horizon_list())
