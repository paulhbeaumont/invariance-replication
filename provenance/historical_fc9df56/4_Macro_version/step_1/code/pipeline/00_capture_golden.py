"""Golden capture: dump the OLD path's objects once so every tie-out compares
against fixtures instead of rerunning the old chain.

Writes to <MACRO_DATA_ROOT>/temp/pipeline/_golden/:
  base_panel.parquet                p51.base_panel() (screen applied)
  con_net.parquet, con_pos.parquet  county x year contract dollars, long
  nat_net.parquet, nat_pos.parquet  category x year national dollars, long
  cats.csv                          the top-NCAT category list, in rank order
  gk_h{h}.parquet                   category shocks at the locked constants
  L_h{h}.parquet                    predicted dollars (instrument numerator)
  county_h{h}.parquet               57's build_county frame at the lock
  category_actions_samples.parquet  raw + harmonized per-year action arrays
                                    for sample years (contracts/harmonize
                                    module tie-outs)

Also cross-checks locked.py against the old modules (fails loud on drift).
Read-only with respect to every old producer and its outputs.
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
import oldpath as op  # noqa: E402

GOLD = op.GOLD
SAMPLE_YEARS = (1975, 1979, 1990, 2007, 2015)


def wide_to_long(df, id_name, value_name):
    out = df.stack().rename(value_name).reset_index()
    out.columns = [id_name, "year", value_name]
    return out.sort_values([id_name, "year"]).reset_index(drop=True)


def main():
    t0 = time.time()
    GOLD.mkdir(parents=True, exist_ok=True)

    n = op.check_lock(locked)
    print(f"lock check: {n} constants match the old modules")

    B = op.p51.base_panel()
    B.to_parquet(GOLD / "base_panel.parquet", index=False)
    print(f"base_panel: {len(B):,} rows, {B['ccg'].nunique():,} counties")

    P = op.p57.panels()
    for rule, tag in (("net", "net"), ("positive", "pos")):
        con = P["CON"][(rule, 0.0)]
        wide_to_long(con, "ccg", "dollars").to_parquet(
            GOLD / f"con_{tag}.parquet", index=False)
        nat = P["NAT"][(rule, 0.0)]
        wide_to_long(nat, "cat", "dollars").to_parquet(
            GOLD / f"nat_{tag}.parquet", index=False)
    pd.Series(P["cats"], name="cat").to_csv(GOLD / "cats.csv", index=False)
    print(f"contract matrices and national panels written; "
          f"{len(P['cats'])} categories")

    units = P["units"]
    for h in locked.HORIZONS:
        years = [t for t in range(max(1977, 1975 + h), locked.TARGET_LAST + 1)]
        gk = op.p54.category_rates(P["NAT"][(locked.RULE, locked.FLOOR)], h)
        gk_df = pd.DataFrame(gk, index=P["cats"], columns=P["cy"])
        wide_to_long(gk_df, "cat", "g").to_parquet(
            GOLD / f"gk_h{h}.parquet", index=False)
        L, ncontrib = op.p54.predicted_dollars(
            P["ARR"][(locked.RULE, locked.FLOOR)], gk, P["cy"], years, h,
            "roll", locked.SPAN, locked.LAG, locked.DATA_FIRST)
        L_df = pd.DataFrame(L, index=units, columns=years)
        long = wide_to_long(L_df, "ccg", "L")
        long.to_parquet(GOLD / f"L_h{h}.parquet", index=False)
        d, dropped = op.p57.build_county(
            locked.INSTRUMENT, h=h, span=locked.SPAN, rule=locked.RULE,
            floor=locked.FLOOR, denom=locked.DENOM)
        d = d.sort_values(["ccg", "year"]).reset_index(drop=True)
        d.to_parquet(GOLD / f"county_h{h}.parquet", index=False)
        assert len(d) == locked.N_COUNTY[h], (h, len(d), locked.N_COUNTY[h])
        assert d["ccg"].nunique() == locked.N_COUNTIES
        print(f"h={h}: county frame {len(d):,} rows "
              f"({d['ccg'].nunique():,} counties, {dropped} dropped), "
              f"gk and L written")

    fips_to_ccg, _, _ = op.dc.load_ccg_map()
    frames = []
    for y in SAMPLE_YEARS:
        for harmonize, tag in ((False, "raw"), (True, "harmonized")):
            d, ccg, cat = op.p54.category_actions(
                y, fips_to_ccg, drop_construction=True, harmonize=harmonize)
            frames.append(pd.DataFrame(dict(
                year=y, variant=tag, ccg=ccg, cat=cat, dollars=d)))
    (pd.concat(frames, ignore_index=True)
       .to_parquet(GOLD / "category_actions_samples.parquet", index=False))
    print(f"category_actions samples written for years {SAMPLE_YEARS}")
    print(f"done in {time.time() - t0:.1f}s -> {GOLD}")


if __name__ == "__main__":
    main()
