"""WS1 step 2, part B: the composition instrument on the industry axis.

WHAT THIS IS. `03_composition_instrument.py` builds the locked instrument on the
top-800 FSC/PSC category axis. This builds the same object on the 309-cell industry
axis producer 23 put the contract dollars on, at every locked constant except the one
WS1 step 2 changes on purpose: the national shock is LEAVE-ONE-OUT.

    locked      L_ct = sum_k  cbar_ck[t-h-9, t-h-5] . g_kt
                g_kt = clip( (N_kt - N_k,t-h) / mean_s|N_ks| , +/-CLIP )

    industry    L_ct = sum_j  cbar_cj[t-h-9, t-h-5] . g_jt^(-c)
                g_jt^(-c) = clip( (N_jt^(-c) - N_j,t-h^(-c)) / mean_s|N_js^(-c)| , +/-CLIP )
                N_j.^(-c) = N_j. - C_cj.

Everything else is locked.py: SPAN=5, LAG=5, CLIP, RULE="positive", DENOM=population,
HORIZONS, DATA_FIRST, N_COUNTIES. NOTHING LOCKED IS EDITED. This writes to a sibling
of the locked store, `<MACRO_DATA_ROOT>/temp/pipeline_industry/`, so the locked store
is not shadowed and `03`'s outputs are untouched.

NCAT IS NOT MIRRORED, and that is deliberate. locked.NCAT=800 cuts a ~5,000-code FSC
universe (lib/contracts.py:154). Producer 23 already runs on that cut - its cats_set()
asserts len(cats.csv)==locked.NCAT - so the axis cells are a RELABELING of exactly the
top-800 dollars, not a fresh universe. A second top-N rule would cut the same dollars
twice. All 309 populated cells are kept; --top-n runs the sensitivity.

THE ZERO-FILL RULE, which is the locked build's and is asymmetric on purpose:
  dollars are zero-filled, shocks are not.
An absent (county, industry, year) is genuinely zero dollars, so the tensor is dense
and zero-filled - which is what window_sums needs, since it takes a mean over a FIXED
five-year denominator and a NaN year would silently rescale the window. An absent
shock is UNDEFINED, so it is NaN and held out of the sum by the `live` mask, never
read as zero growth (lib/instrument.py:34-42 and :80).

THE LOO ADDS A CASE THE LOCKED BUILD DOES NOT HAVE. If county c is the only county
with dollars in industry j, then N_j^(-c) == 0 in every year, its scale is 0, and
g_j^(-c) is NaN for that county alone. That is the correct treatment - there is no
out-of-county national series to difference - and it makes the live shifter set
county-specific. Counted and reported to industry_instrument_loo_holes.csv.

The tensor reader and the sentinel list live in lib/industry.py, so producer 28 builds
the identical object from the identical code rather than a second copy of it.

Outputs (to <MACRO_DATA_ROOT>/temp/pipeline_industry/):
  industry_category_shocks_year.parquet   axis4, year, g, h   (the POOLED shock; the
                                          LOO shock is county-specific and is not
                                          persisted, only its contraction)
  industry_instrument_ccg_year.parquet    ccg, year, h, L_ind, z_ind,
                                          z_ind_noloo, z_ind_modal
  Results/pipeline_diagnostics/industry_instrument_loo_holes.csv
  Results/pipeline_diagnostics/industry_axis_concentration.csv
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
import dcads_common as dc  # noqa: E402
import locked  # noqa: E402
import instrument  # noqa: E402
import store  # noqa: E402
from industry import SENTINELS, load_axis_tensor  # noqa: E402

CELLS = PIPELINE.parents[2] / "data" / "temp" / "industry_axis"
OUT = dc.MACRO_DATA_ROOT / "temp" / "pipeline_industry"
DIAG = PIPELINE.parents[2] / "Results" / "pipeline_diagnostics"

# Conservation tolerances. See assert_conservation for why there are three: the
# county-year bound is floored by stage 1's float32 tensor storage (23's README
# records 1.8e-4 on one county in 2018); the pooled and per-year bounds are tight
# and are the ones that would catch an actual mapping error.
RTOL_CONSERVE = 5e-4      # worst single county-year
RTOL_YEAR = 1e-5          # worst single year, nationally
RTOL_POOLED = 1e-6        # pooled over everything
CHUNK = 256               # counties per chunk in the LOO scale pass


def log(msg):
    print(msg, flush=True)


def assert_conservation(units, years, arr, ref, tag):
    """County x year totals must reproduce the locked top-800 FSC tensor.

    THE REFERENCE IS THE TENSOR, NOT contracts_ccg_year.parquet. The latter is the
    ALL-category county total from lib/contracts.contract_matrices; the industry
    cells descend from category_panel, which is restricted to the locked top-800
    (lib/contracts.py:154), and producer 23 maps exactly that panel. Tying to the
    all-category total would be comparing two different universes - it fails at
    rel 1.0 on county-years whose only dollars are outside the top 800.

    Three tolerances, because one number cannot distinguish float32 noise from a
    merge error. The county-year bound is loose because stage 1 STORES the tensor
    in float32 (lib/contracts.py:158) and 23's README records the consequence: up
    to 1.8e-4 on one county in 2018, where recomputing from the raw records in
    float64 gives 23's number to the cent. The pooled and per-year bounds are
    tight, and those are the ones a real error would trip.
    """
    got = arr.sum(axis=1).astype(np.float64)              # (units, years)
    num = np.abs(got - ref)
    den = np.maximum(np.abs(ref), 1.0)
    rel = num / den
    worst = float(rel.max())
    where = np.unravel_index(np.argmax(rel), num.shape)
    pooled = float(num.sum() / np.abs(ref).sum())
    per_year = float((num.sum(axis=0) / np.maximum(np.abs(ref).sum(axis=0), 1.0)).max())

    assert pooled <= RTOL_POOLED, f"{tag}: pooled dollars off by rel {pooled:.3e}"
    assert per_year <= RTOL_YEAR, f"{tag}: a year's dollars off by rel {per_year:.3e}"
    assert worst <= RTOL_CONSERVE, (
        f"{tag}: county x year dollars do not tie to the locked tensor; worst rel "
        f"{worst:.3e} at ccg={units[where[0]]} year={years[where[1]]}")
    log(f"  {tag}: conservation against the locked top-800 tensor holds "
        f"(pooled {pooled:.2e}, worst year {per_year:.2e}, "
        f"worst county-year {worst:.2e})")
    return worst


def loo_scale(arr, nat):
    """mean_s |N_js - C_cjs| for every (county, cell): the LOO analogue of
    lib/instrument.py's `scale`. Chunked over counties to stay in memory."""
    n_u, n_c, n_y = arr.shape
    out = np.empty((n_u, n_c), np.float64)
    for a in range(0, n_u, CHUNK):
        b = min(a + CHUNK, n_u)
        d = nat[None, :, :] - arr[a:b].astype(np.float64)
        out[a:b] = np.abs(d).mean(axis=2)
    return out


def loo_rates(arr, nat, scale, years, t, h, clip=locked.CLIP):
    """g_jt^(-c) for one target year: (units, cells), NaN where undefined.

    Mirrors lib/instrument.py:33 category_rates exactly, one year at a time and with
    the own county removed from the national series. NaN where the h-year lag is
    outside the panel or where the LOO national series is identically zero (the
    sole-holder case); such cells are held out of the sum, never read as zero.
    """
    ti = years.index(t)
    if ti - h < 0:
        return np.full(arr.shape[:2], np.nan)
    cur = nat[:, ti][None, :] - arr[:, :, ti].astype(np.float64)
    lag = nat[:, ti - h][None, :] - arr[:, :, ti - h].astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        g = (cur - lag) / scale
    g[~np.isfinite(g)] = np.nan
    return np.clip(g, -clip, clip)


def predicted_loo(arr, nat, scale, years, target_years, h):
    """The LOO numerator. Same window rule as lib/instrument.predicted_dollars:
    the mix is measured over [t-h-LAG-SPAN+1, t-h-LAG], closing before the
    treatment window opens at t-h+1."""
    n_u = arr.shape[0]
    out = np.full((n_u, len(target_years)), np.nan, np.float64)
    live_counts = np.full((n_u, len(target_years)), np.nan)
    for c, t in enumerate(target_years):
        hi = t - h - locked.LAG
        lo = hi - locked.SPAN + 1
        if lo < locked.DATA_FIRST or lo < years[0] or hi > years[-1] or t not in years:
            continue
        g = loo_rates(arr, nat, scale, years, t, h)          # (units, cells)
        cbar = instrument.window_sums(arr, years, lo, hi).astype(np.float64)
        live = np.isfinite(g)
        out[:, c] = np.einsum("uc,uc->u", cbar, np.where(live, g, 0.0))
        live_counts[:, c] = live.sum(axis=1)
    return out, live_counts


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h", type=int, action="append", choices=list(locked.HORIZONS),
                    help="horizons to build (default: all of locked.HORIZONS)")
    ap.add_argument("--top-n", type=int, default=None,
                    help="sensitivity: keep only the top-N cells by pooled national "
                         "dollars. Default keeps all 309 - see the module docstring.")
    ap.add_argument("--dry-run", action="store_true", help="write nothing")
    args = ap.parse_args()
    horizons = tuple(args.h) if args.h else locked.HORIZONS

    t0 = time.time()
    log("loading the locked FSC tensor as the conservation reference")
    u0, _, y0, arr0, _ = store.load_tensor(locked.RULE)
    ref = arr0.sum(axis=1).astype(np.float64)
    del arr0

    log("loading the industry tensor (dollars_prob, headline)")
    units, cells, years, arr = load_axis_tensor("dollars_prob")
    log(f"  {len(units)} counties x {len(cells)} cells x {len(years)} years")
    assert np.array_equal(u0, units) and list(y0) == list(years), \
        "the industry tensor is not on the locked county x year grid"
    assert_conservation(units, years, arr, ref, "dollars_prob")

    log("loading the industry tensor (dollars_modal, robustness)")
    units_m, cells_m, _, arr_m = load_axis_tensor("dollars_modal")
    assert list(cells_m) == list(cells) and np.array_equal(units_m, units)
    assert_conservation(units_m, years, arr_m, ref, "dollars_modal")

    # --- the sentinels: kept for the assertion above, excluded from the sum
    keep = [k for k, c in enumerate(cells) if c not in SENTINELS]
    sent = [k for k, c in enumerate(cells) if c in SENTINELS]
    tot = float(arr.sum())
    sent_share = float(arr[:, sent, :].sum()) / tot if sent else 0.0
    log(f"  sentinels {[cells[k] for k in sent]}: {sent_share*100:.4f}% of dollars, "
        "excluded from the shift-share sum")

    real_cells = [cells[k] for k in keep]
    arr = np.ascontiguousarray(arr[:, keep, :])
    arr_m = np.ascontiguousarray(arr_m[:, keep, :])

    # --- concentration, reported so the no-cut decision is visible
    natpool = arr.sum(axis=(0, 2)).astype(np.float64)
    order = np.argsort(-natpool)
    cum = np.cumsum(natpool[order]) / natpool.sum()
    shares = natpool / natpool.sum()
    conc = pd.DataFrame(dict(
        rank=np.arange(1, len(real_cells) + 1),
        axis4=[real_cells[k] for k in order],
        dollars=natpool[order], share=shares[order], cum_share=cum))
    log(f"  {len(real_cells)} cells, {(natpool > 0).sum()} populated; "
        f"1/HHI = {1.0/np.square(shares).sum():.2f} effective shifters")

    if args.top_n:
        sel = sorted(order[:args.top_n])
        log(f"  --top-n {args.top_n}: keeping {len(sel)} cells, "
            f"{cum[args.top_n-1]*100:.4f}% of dollars")
        real_cells = [real_cells[k] for k in sel]
        arr = np.ascontiguousarray(arr[:, sel, :])
        arr_m = np.ascontiguousarray(arr_m[:, sel, :])

    nat = arr.sum(axis=0).astype(np.float64)                  # (cells, years)
    nat_m = arr_m.sum(axis=0).astype(np.float64)
    natdf = pd.DataFrame(nat, index=real_cells, columns=years)
    natdf_m = pd.DataFrame(nat_m, index=real_cells, columns=years)

    log("building the LOO scale (mean_s |N_js - C_cjs|)")
    scale = loo_scale(arr, nat)
    scale_m = loo_scale(arr_m, nat_m)
    holes = ~np.isfinite(scale) | (scale == 0)
    log(f"  LOO holes (sole-holder counties): {int(holes.sum()):,} of "
        f"{holes.size:,} (county, cell) pairs")

    pnl = store.load_county_panel()
    _, piv = store.load_pivots(pnl, cols=("P",))
    Ppiv = piv["P"].reindex(units)

    shock_rows, inst_rows, hole_rows = [], [], []
    for h in horizons:
        ty = store.target_years(h)
        Xb = Ppiv.reindex(columns=[y - h for y in ty]).to_numpy(np.float64)

        # headline: LOO, probabilistic allocation
        L, live = predicted_loo(arr, nat, scale, years, ty, h)
        Z = L / Xb

        # the same code path with the POOLED national shock: isolates the axis
        # change from the LOO change for the step-C comparison
        gk = instrument.category_rates(natdf, h)
        L0, _ = instrument.predicted_dollars(arr, gk, years, ty, h, "roll",
                                             locked.SPAN, locked.LAG,
                                             locked.DATA_FIRST)
        Z0 = L0 / Xb

        # modal allocation, LOO, so it differs from the headline in one thing
        Lm, _ = predicted_loo(arr_m, nat_m, scale_m, years, ty, h)
        Zm = Lm / Xb

        fin = np.isfinite(Z)
        lv = live[fin]
        hole_rows.append(dict(
            h=h, target_years=len(ty),
            loo_holes=int(holes.sum()),
            loo_hole_share_pairs=float(holes.mean()),
            loo_hole_dollar_share=float(
                arr[holes].sum() / arr.sum()) if holes.any() else 0.0,
            counties_with_any_hole=int(holes.any(axis=1).sum()),
            live_min=float(np.nanmin(lv)) if lv.size else np.nan,
            live_median=float(np.nanmedian(lv)) if lv.size else np.nan,
            live_max=float(np.nanmax(lv)) if lv.size else np.nan,
            n_cells=len(real_cells),
            corr_loo_vs_noloo=float(pd.Series(Z[fin]).corr(pd.Series(Z0[fin]))),
        ))

        shock_rows.append(pd.DataFrame(gk, index=real_cells, columns=years)
                          .stack().rename("g").reset_index()
                          .set_axis(["axis4", "year", "g"], axis=1).assign(h=h))
        inst_rows.append(pd.DataFrame(dict(
            ccg=np.repeat(units, len(ty)), year=np.tile(ty, len(units)), h=h,
            L_ind=L.ravel(), z_ind=Z.ravel(), z_ind_noloo=Z0.ravel(),
            z_ind_modal=Zm.ravel())))
        log(f"h={h}: z_ind {np.isfinite(Z).sum():,} finite cells; "
            f"live shifters per county-year "
            f"min {hole_rows[-1]['live_min']:.0f} / "
            f"med {hole_rows[-1]['live_median']:.0f} / "
            f"max {hole_rows[-1]['live_max']:.0f}; "
            f"corr(z_ind, z_ind_noloo) = {hole_rows[-1]['corr_loo_vs_noloo']:.4f}")

    if args.dry_run:
        log("\n--dry-run: nothing written")
        return

    OUT.mkdir(parents=True, exist_ok=True)
    DIAG.mkdir(parents=True, exist_ok=True)
    pd.concat(shock_rows, ignore_index=True).to_parquet(
        OUT / "industry_category_shocks_year.parquet", index=False)
    out = (pd.concat(inst_rows, ignore_index=True)
             .sort_values(["ccg", "year", "h"]).reset_index(drop=True))
    out.to_parquet(OUT / "industry_instrument_ccg_year.parquet", index=False)
    pd.DataFrame(hole_rows).to_csv(
        DIAG / "industry_instrument_loo_holes.csv", index=False)
    conc.to_csv(DIAG / "industry_axis_concentration.csv", index=False)
    log(f"\nindustry_instrument_ccg_year: {len(out):,} rows "
        "(L_ind, z_ind, z_ind_noloo, z_ind_modal)")
    log(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
