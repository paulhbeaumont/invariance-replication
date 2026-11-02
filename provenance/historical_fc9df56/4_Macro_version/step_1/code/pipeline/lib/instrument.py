"""The composition instrument: shares, shocks, and predicted dollars.

Three pure functions on the category tensors from contracts.category_panel:

- window_sums: the exposure C̄_ck, mean annual dollars per category over an
  inclusive year window;
- category_rates: the national shock g_kt, the h-year change in a category's
  national dollars scaled by the category's own mean size, clipped at +/-CLIP;
- predicted_dollars: the instrument's numerator, C̄ (window closing at
  t - h - LAG) times g_kt, summed over categories live at t.

The instrument is z_ct = predicted dollars / denominator at t-h; the division
happens in the runner (03), which holds the denominator panel.

Extracted from producer 54 (window_sums 54:459, category_rates 54:444,
predicted_dollars 54:466); operation order preserved so outputs are
bit-identical. Run this file directly to tie its output to the golden
fixtures (applies the NEW functions to the OLD tensors, isolating this
module).
"""
import sys
from pathlib import Path

import numpy as np

LIB = Path(__file__).resolve().parent
PIPELINE = LIB.parent
sys.path.insert(0, str(PIPELINE))
sys.path.insert(0, str(LIB))
import locked  # noqa: E402


def category_rates(nat, h, clip=locked.CLIP):
    """National change in each category over h years, scaled by its own mean
    size. Returns (n_cats, n_years) with NaN where the h-year lag is outside
    the panel; such categories are held out of the sum rather than read as
    zero growth."""
    v = nat.to_numpy(np.float64)
    out = np.full_like(v, np.nan)
    out[:, h:] = v[:, h:] - v[:, :-h]
    scale = np.nanmean(np.abs(v), axis=1, keepdims=True)
    scale[scale == 0] = np.nan
    return np.clip(out / scale, -clip, clip)


def window_sums(arr, years, lo, hi):
    """Mean annual dollars per category over the inclusive year window [lo, hi]."""
    a = years.index(lo)
    b = years.index(hi)
    return arr[:, :, a:b + 1].mean(axis=2)


def predicted_dollars(arr, gk, years, target_years, h, kind, spec, roll_lag,
                      first_window_year):
    """The instrument's numerator: predicted dollar change, county by year.

    For each target year t the county's category mix is measured over a
    window that closes at t - h - roll_lag, before the treatment window opens
    at t - h + 1, and is met with the national rate for that same t. Returns
    a (n_units, n_target_years) array with NaN in columns whose window falls
    outside the data, and the count of contributing categories. The rolling
    window stops at first_window_year (DATA_FIRST) so every cell sits on the
    same sample as the incumbent grid.
    """
    out = np.full((arr.shape[0], len(target_years)), np.nan, np.float64)
    ncontrib = np.zeros(len(target_years), np.int64)
    for c, t in enumerate(target_years):
        if kind == "fixed":
            lo, hi = spec
        else:
            hi = t - h - roll_lag
            lo = hi - spec + 1
            if lo < first_window_year:
                continue
        if lo < years[0] or hi > years[-1] or lo > hi:
            continue
        if t not in years:
            continue
        g = gk[:, years.index(t)]
        live = np.isfinite(g)
        if not live.any():
            continue
        cbar = window_sums(arr, years, lo, hi)
        out[:, c] = cbar[:, live].astype(np.float64) @ g[live]
        ncontrib[c] = int(live.sum())
    return out, ncontrib


if __name__ == "__main__":
    # Micro tie-out: NEW functions on OLD tensors vs the golden gk/L fixtures.
    import pandas as pd
    import oldpath as op
    GOLD = op.GOLD
    P = op.p57.panels()
    for h in locked.HORIZONS:
        years = [t for t in range(max(1977, 1975 + h), locked.TARGET_LAST + 1)]
        gk = category_rates(P["NAT"][(locked.RULE, locked.FLOOR)], h)
        ref_gk = (pd.read_parquet(GOLD / f"gk_h{h}.parquet")
                    .pivot(index="cat", columns="year", values="g")
                    .reindex(index=P["cats"], columns=P["cy"]).to_numpy())
        assert np.array_equal(np.isnan(ref_gk), np.isnan(gk)), h
        assert np.allclose(ref_gk, gk, rtol=locked.RTOL_VEC,
                           atol=locked.ATOL_VEC, equal_nan=True), h
        L, _ = predicted_dollars(P["ARR"][(locked.RULE, locked.FLOOR)], gk,
                                 P["cy"], years, h, "roll", locked.SPAN,
                                 locked.LAG, locked.DATA_FIRST)
        ref_L = (pd.read_parquet(GOLD / f"L_h{h}.parquet")
                   .pivot(index="ccg", columns="year", values="L")
                   .reindex(index=P["units"], columns=years).to_numpy())
        assert np.array_equal(np.isnan(ref_L), np.isnan(L)), h
        assert np.allclose(ref_L, L, rtol=locked.RTOL_VEC,
                           atol=locked.ATOL_VEC, equal_nan=True), h
        print(f"instrument tie-out h={h}: gk {gk.shape} and L {L.shape} match")
    print("instrument.py: PASS")
