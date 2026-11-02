"""Category-code harmonization across the three namespace breaks.

The contract category axis (4-character FSC/PSC codes) breaks three times:
FY1979 (service codes renumbered), FY1989 (R2xx to C2xx, R3xx to D3xx),
FY2012 (PSC revision). This module rewrites one year's (dollars, ccg, code)
arrays into the harmonized namespace:

- pre-1979 service records are expanded through the 1979 bridge
  (reference/service_code_bridge_1979.csv) or, for codes the bridge never
  saw, spread over the letter's modern codes by 1979-81 national shares;
- a single symmetric relabel map (1989 renames + 2012 renames and
  consolidations) applies to every year;
- orphan closure (2012 newborns with no predecessor) applies to years >= 2012.

Extracted from producer 54 (54:159-338); operation order preserved so outputs
are bit-identical. Run this file directly to tie its output to the golden
fixtures.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LIB = Path(__file__).resolve().parent
PIPELINE = LIB.parent
CODE = PIPELINE.parent
REF_DIR = CODE / "reference"
sys.path.insert(0, str(CODE / "py"))
sys.path.insert(0, str(PIPELINE))
import dcads_common as dc  # noqa: E402

_HARMONIZATION = None


def load_harmonization():
    """The seam repairs, loaded once.

    Returns (bridge, relabel, orphan_map, letter_spread). `bridge` maps a
    pre-1979 service code to a list of (modern code, weight) summing to 1;
    unmatched codes are absent and their records drop out (1.0% of pre-1979
    service dollars). `relabel` is applied symmetrically to every year.
    `orphan_map` applies to years >= 2012 only. `letter_spread` covers
    pre-1979 service codes the bridge never saw.
    """
    global _HARMONIZATION
    if _HARMONIZATION is not None:
        return _HARMONIZATION
    b = pd.read_csv(REF_DIR / "service_code_bridge_1979.csv", comment="#",
                    dtype={"old_code": str, "new_code": str})
    b = b[b["new_code"].notna() & (b["new_code"] != "") & (b["weight"] > 0)]
    bridge = {oc: list(zip(g["new_code"], g["weight"]))
              for oc, g in b.groupby("old_code")}

    def year_codes(year):
        if year < dc.SEAM_YEAR:
            df = pd.read_parquet(dc.OUT_DIR / f"fy{year}.parquet",
                                 columns=["dollars", "geo_class", "fsc"])
            df = df[df["geo_class"] == "county"]
            code = df["fsc"].astype(str).str.strip().str[:4]
            keep = df["fsc"].notna() & code.str.len().eq(4)
            return set(code[keep][df.loc[keep, "dollars"] > 0])
        df = pd.read_parquet(dc.USASPENDING_OUT_DIR / f"fy{year}.parquet",
                             columns=["federal_action_obligation",
                                      "award_or_idv_flag", "geo_class",
                                      "product_or_service_code"])
        df = df[df["award_or_idv_flag"].eq("AWARD")
                & df["geo_class"].eq("county")
                & (df["federal_action_obligation"] > 0)]
        code = df["product_or_service_code"].astype(str).str.strip().str[:4]
        return set(code[code.str.len().eq(4)])

    # FY1989: dying R-codes whose exact suffix reappears under C, D, or E.
    dead89 = year_codes(1988) - year_codes(1989)
    born89 = year_codes(1989) - year_codes(1988)
    relabel = {}
    for c in dead89:
        if c[0] == "R":
            for pre in ("C", "D", "E"):
                if pre + c[1:] in born89:
                    relabel[c] = pre + c[1:]
                    break

    # FY2012: unique-predecessor renames new-to-old, consolidations old-to-new.
    xw = pd.read_csv(REF_DIR / "psc2011_crosswalk.csv", comment="#",
                     dtype=str).fillna("")
    preds = {}
    for o, n in zip(xw.previous_psc, xw.current_psc):
        if n and n != o:
            preds.setdefault(n, []).append(o)
    for n, olds in preds.items():
        if len(olds) == 1:
            relabel[n] = olds[0]
        else:
            for o in olds:
                relabel[o] = n

    known11 = year_codes(2011)
    handled = set(relabel)
    orphan_map = {}
    fam_sets = {}
    for c in sorted(known11):
        fam_sets.setdefault(c[:2], []).append(c)
    for c in year_codes(2012) - known11 - handled:
        fam = c[:2]
        if fam + "99" in known11:
            orphan_map[c] = fam + "99"
        elif fam_sets.get(fam):
            orphan_map[c] = fam_sets[fam][0]

    # Resolve chains once so a single map application lands on the final label.
    for k in list(relabel):
        v, seen = relabel[k], {k}
        while v in relabel and v not in seen:
            seen.add(v)
            v = relabel[v]
        relabel[k] = v

    # Letter spreads for pre-1979 service codes the bridge never saw:
    # distribute over the letter's modern codes by 1979-81 national shares.
    nat = {}
    for y in (1979, 1980, 1981):
        df = pd.read_parquet(dc.OUT_DIR / f"fy{y}.parquet",
                             columns=["dollars", "dollars_unit", "geo_class",
                                      "fsc"])
        df = df[df["geo_class"] == "county"]
        dv = df["dollars"].to_numpy(np.float64)
        dv = np.where((df["dollars_unit"] == "thousands").to_numpy(),
                      dv * 1000.0, dv)
        code = df["fsc"].astype(str).str.strip().str[:4]
        keep = (df["fsc"].notna() & code.str.len().eq(4)
                & code.str[0].str.isalpha() & (dv > 0))
        s = pd.Series(dv[keep.to_numpy()]).groupby(
            code[keep].to_numpy()).sum()
        for k, v in s.items():
            nat[k] = nat.get(k, 0.0) + v
    nat = pd.Series(nat)
    letter_spread = {}
    for letter, g in nat.groupby(nat.index.str[0]):
        # top 30 codes per letter, renormalized
        g = g.nlargest(30)
        letter_spread[letter] = list(zip(g.index, (g / g.sum()).to_numpy()))

    _HARMONIZATION = (bridge, relabel, orphan_map, letter_spread)
    return _HARMONIZATION


def harmonize_arrays(year, d, ccg, cat):
    """Rewrite one year's (dollars, ccg, code) arrays into the harmonized
    namespace."""
    bridge, relabel, orphan_map, letter_spread = load_harmonization()
    if year <= 1978:
        s = pd.Series(cat)
        svc = s.str[0].str.isalpha().to_numpy()
        in_bridge = svc & s.isin(bridge.keys()).to_numpy()
        unb = svc & ~in_bridge
        goods = ~svc
        out_d = [d[goods]]
        out_c = [ccg[goods]]
        out_k = [cat[goods]]
        bt = pd.DataFrame([(oc, nc, w) for oc, pairs in bridge.items()
                           for nc, w in pairs],
                          columns=["code", "new", "w"])
        if in_bridge.any():
            m = pd.DataFrame({"d": d[in_bridge], "ccg": ccg[in_bridge],
                              "code": cat[in_bridge]})
            m = m.merge(bt, on="code", how="inner")
            out_d.append((m["d"] * m["w"]).to_numpy(np.float64))
            out_c.append(m["ccg"].to_numpy())
            out_k.append(m["new"].to_numpy(dtype=object))
        if unb.any():
            lt = pd.DataFrame([(letter, nc, w)
                               for letter, pairs in letter_spread.items()
                               for nc, w in pairs],
                              columns=["letter", "new", "w"])
            m = pd.DataFrame({"d": d[unb], "ccg": ccg[unb],
                              "letter": pd.Series(cat[unb]).str[0]})
            m = m.merge(lt, on="letter", how="inner")
            out_d.append((m["d"] * m["w"]).to_numpy(np.float64))
            out_c.append(m["ccg"].to_numpy())
            out_k.append(m["new"].to_numpy(dtype=object))
        d, ccg, cat = (np.concatenate(out_d), np.concatenate(out_c),
                       np.concatenate(out_k))
    s = pd.Series(cat)
    mapped = s.map(relabel)
    if year >= 2012:
        orph = s.map(orphan_map)
        mapped = mapped.fillna(orph)
    cat = mapped.fillna(s).to_numpy()
    return d, ccg, cat


if __name__ == "__main__":
    # Micro tie-out: harmonized sample years against the golden fixtures.
    # Requires contracts.py (raw extraction) to have passed its own tie-out.
    import locked  # noqa: E402
    import contracts  # noqa: E402
    GOLD = dc.MACRO_DATA_ROOT / "temp" / "pipeline" / "_golden"
    gold = pd.read_parquet(GOLD / "category_actions_samples.parquet")
    fips_to_ccg, _, _ = dc.load_ccg_map()
    for y in sorted(gold["year"].unique()):
        ref = gold[(gold["year"] == y) & (gold["variant"] == "harmonized")]
        d, ccg, cat = contracts.category_actions(
            y, fips_to_ccg, drop_construction=True, harmonize=True)
        assert len(ref) == len(d), (y, len(ref), len(d))
        assert np.array_equal(ref["ccg"].to_numpy(), ccg), y
        assert np.array_equal(ref["cat"].to_numpy(), cat), y
        assert np.allclose(ref["dollars"].to_numpy(), d,
                           rtol=locked.RTOL_VEC, atol=locked.ATOL_VEC), y
        print(f"harmonize tie-out {y}: {len(d):,} records match")
    print("harmonize.py: PASS")
