"""Concord the canonical Rajan-Zingales measure onto the EFSY NAICS4 universe.

WS2 step 4's second bin. 70_ingest_rajan_zingales.py transcribes the measure at
36 ISIC rev 2 industries and says in its own docstring that neither of its files
is concorded to NAICS4, and that "the canonical route needs ISIC rev 2 to SIC to
NAICS by hand, respecting the net-of-children semantics". This producer is that
route.

    ISIC rev 2  --hand table below-->  SIC 1987 (3-digit)
                --reference/sic87_naics97_census.csv-->  NAICS 1997 (5/6-digit)
                --leading 4 digits-->  NAICS4, the EFSY cell code

WHY THE HAND TABLE IS AT THREE-DIGIT SIC. RZ's industries are ISIC rev 2 three-
and four-digit; the Census concordance is keyed on four-digit SIC. Three-digit
SIC is the coarsest level at which the ISIC boundaries are clean, and matching
the Census file by three-digit PREFIX rather than by exact code means the table
does not have to enumerate 821 manufacturing SIC codes to stay complete.

THE NET-OF-PARENT SEMANTICS ARE ENFORCED STRUCTURALLY, NOT BY A LATER CHECK.
RZ report seven parents net of their separately-reported children (70's caveat
1). Here every parent's SIC3 list simply EXCLUDES its children's, so the 36
lists are pairwise disjoint by construction and no NAICS4 cell can ever receive
both a parent and its child. The disjointness is asserted, so an edit that
breaks it fails loud instead of silently double counting. Worked: ISIC 382
Machinery gets SIC 351-356, 358, 359 and NOT 357, which is ISIC 3825 Office and
computing's own row.

WHAT IS DELIBERATELY NOT COVERED. ISIC 3512 Fertilizers is absent from RZ's
Table 1, so SIC 287 has no ISIC assignment, and NAICS 3253 (pesticide and
fertilizer manufacturing) therefore carries no EFD value. That is 0.25-0.35% of
EFSY manufacturing employment over 1975-2016 - the whole span WS2 step 4 now
reports, both sides of the NAICS break - and it is a REPORTED gap, never a drop
(INVARIANCE_TEST_PROTOCOL.md A.3.1). SIC 266, 303 and 312 appear in the hand
table but not in SIC 1987; they contribute nothing and are listed as such.

POOLING, AND WHY THE WEIGHT IS THE CONCORDANCE MASS AND NOT EMPLOYMENT. 54 of
the 94 NAICS4 codes the route reaches are fed by more than one ISIC industry,
because a NAICS4 industry can draw on several SIC industries that RZ kept apart.
Employment cannot break the tie: all the competing ISIC values land on ONE
NAICS4 cell, whose employment is a single number, so an employment weight is
constant across the candidates and pools nothing. The identified weight is the
Census concordance mass `weight` (an equal split across a SIC code's NAICS
matches) accumulated per (NAICS4, ISIC), and the pooled value is the
mass-weighted mean of `extfin_all`. This departs from the plan document, which
proposed EFSY-1998 employment weights; that weight is not identified here and
the plan is wrong on this point.

Pooling is measurement error and the exhibit has to be able to say how much:
`n_isic` and `spread` (max minus min `extfin_all` among the contributors) are
carried on every row. Employment-weighted mean spread over manufacturing is
0.33 against an EFD range of -0.45 to 1.49; 31% of manufacturing employment
sits in single-ISIC cells, which is what makes a single-ISIC robustness panel
worth running.

NAICS 1997 TO NAICS 2012, WHICH EFSY ACTUALLY USES. Producer 11 encodes EFSY's
`naics12` column, so the cell universe is NAICS 2012 while the Census
concordance ends at NAICS 1997. Four-digit truncation is the bridge, and it is
ASSERTED rather than assumed: the gate is that the share of EFSY manufacturing
employment with no EFD value stays under MAX_UNCOVERED_MFG in every year of the
window. The threshold is stated here, before the number is read.

MANUFACTURING ONLY, AND THE CONCORDANCE SPILLOVER. Canonical RZ is US SIC
2000-3999. Eight NAICS4 codes outside NAICS 31-33 nevertheless receive a value,
because SIC 1987 manufacturing codes map partly into non-manufacturing NAICS
1997 (1133 logging, 2111, 2123, 4883, 5111, 5122, 5417, 8114). They are flagged
`mfg = 0` rather than deleted, and the consumer restricts to `mfg = 1` so that
"manufacturing only" means what it says. Keeping them would quietly widen the
bin by about one percentage point of total employment.

Writes step_1/code/reference/rz_naics4_extfin.csv and prints the coverage table
that the exhibit's note has to quote.

Cite Rajan and Zingales (1998).
"""

import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dcads_common as dc  # noqa: E402

REF_DIR = HERE.parent / "reference"

# WS2 step 4 reports BOTH sides of the 1998 NAICS break, so the coverage table
# and the truncation gate below have to be discharged on both. The concordance
# itself - rz_naics4_extfin.csv - does not depend on this at all: WINDOW drives
# only which years' EFSY employment the coverage arithmetic is weighted by.
WINDOW = (1975, 2016)
# Pre-1998 EFSY is SIC-bridged and a small residual survives producer 11's
# allocation on codes that could not be pushed to four digits. Producer 30 gives
# it a NAMED BIN (A.3.1) rather than dropping it, so this producer measures it
# and caps it rather than refusing to run. Post-1998 it is exactly zero.
MAX_ROOT_SHARE = 0.01
MFG_SECTORS = (31, 32, 33)     # NAICS 2012 manufacturing
MAX_UNCOVERED_MFG = 0.05       # pre-stated gate on the NAICS97 -> NAICS12 bridge
RESIDUAL3 = 100_000            # 11_build_employment_cells.py's root encoding
CODE_DIGITS = 4

# --- ISIC rev 2 -> SIC 1987, three-digit, PAIRWISE DISJOINT ----------------
# Every list is net of the lists of any ISIC child RZ reports separately, which
# is what makes the parent rows mean what RZ's footnote 10 says they mean. The
# comment on a parent row names the child it excludes.
ISIC_SIC3 = {
    "311":  ["201", "202", "203", "204", "205", "206", "207", "209"],
    "313":  ["208"],
    "314":  ["211", "212", "213", "214"],
    "3211": ["221", "222", "223", "224", "226", "228"],
    "321":  ["225", "227", "229"],                       # net of 3211 Spinning
    "322":  ["231", "232", "233", "234", "235", "236", "237", "238", "239"],
    "323":  ["311", "312", "315", "316", "317", "319"],  # leather, net of footwear
    "324":  ["313", "314"],
    "331":  ["241", "242", "243", "244", "245", "249"],
    "332":  ["251", "252", "253", "254", "259"],
    "3411": ["261", "262", "263", "266"],                # pulp and paper mills
    "341":  ["265", "267"],                              # net of 3411
    "342":  ["271", "272", "273", "274", "275", "276", "277", "278", "279"],
    "3511": ["281", "286"],                              # basic chem excl fert.
    "3513": ["282"],
    "3522": ["283"],                                     # drugs
    "352":  ["284", "285", "289"],                       # net of 3522
    "353":  ["291"],
    "354":  ["295", "299"],
    "355":  ["301", "302", "303", "305", "306"],
    "356":  ["308"],
    "361":  ["326"],
    "362":  ["321", "322", "323"],
    "369":  ["324", "325", "327", "328", "329"],
    "371":  ["331", "332", "339"],
    "372":  ["333", "334", "335", "336"],
    "381":  ["341", "342", "343", "344", "345", "346", "347", "348", "349"],
    "3825": ["357"],                                     # office and computing
    "382":  ["351", "352", "353", "354", "355", "356", "358", "359"],  # net 3825
    "3832": ["365", "366"],                              # radio and TV
    "383":  ["361", "362", "363", "364", "367", "369"],  # net of 3832
    "3841": ["373"],                                     # ship
    "3843": ["371"],                                     # motor vehicle
    "384":  ["372", "374", "375", "376", "379"],         # net of 3841 and 3843
    "385":  ["381", "382", "384", "385", "386", "387"],
    "390":  ["391", "393", "394", "395", "396", "399"],
}

# ISIC 3512 Fertilizers is not in RZ Table 1, so this SIC3 has no owner. Named
# rather than left as a silent hole in the coverage arithmetic.
SIC3_NO_ISIC = {"287": "ISIC 3512 Fertilizers, not reported by Rajan-Zingales"}


def load_rz():
    rz = pd.read_csv(REF_DIR / "rz1998_extfin_isic.csv", comment="#",
                     dtype={"isic_code": str})
    assert len(rz) == 36, len(rz)
    assert set(rz["isic_code"]) == set(ISIC_SIC3), (
        "the hand table and rz1998_extfin_isic.csv disagree on the ISIC set: "
        f"{sorted(set(rz['isic_code']) ^ set(ISIC_SIC3))}")
    assert rz["extfin_all"].notna().all()
    return rz


def check_disjoint():
    """The net-of-parent guarantee, asserted rather than trusted."""
    seen = {}
    for isic, sics in ISIC_SIC3.items():
        assert len(set(sics)) == len(sics), f"{isic} repeats a SIC3"
        for s in sics:
            if s in seen:
                raise SystemExit(
                    f"FAIL: SIC3 {s} is claimed by both ISIC {seen[s]} and "
                    f"{isic}. RZ's parent rows are net of their children, so a "
                    f"SIC3 must have exactly one owner; assigning both a parent "
                    f"and its child to one NAICS4 cell double counts.")
            seen[s] = isic
    return seen


def bridge(rz, sic3_owner):
    """NAICS4 -> pooled extfin_all, over the SIC87-to-NAICS97 concordance."""
    cw = pd.read_csv(REF_DIR / "sic87_naics97_census.csv", comment="#",
                     dtype={"sic": str, "naics97": str})
    cw["sic3"] = cw["sic"].str[:3]

    mfg_sic3 = set(cw.loc[cw["sic"].str[:2].astype(int).between(20, 39), "sic3"])
    unowned = sorted(mfg_sic3 - set(sic3_owner) - set(SIC3_NO_ISIC))
    if unowned:
        raise SystemExit(f"FAIL: manufacturing SIC3 codes with no ISIC owner "
                         f"and no documented exemption: {unowned}")
    phantom = sorted(set(sic3_owner) - set(cw["sic3"]))

    cw["isic"] = cw["sic3"].map(sic3_owner)
    m = cw.dropna(subset=["isic"]).copy()
    m["naics4"] = m["naics97"].str[:CODE_DIGITS].astype(int)

    efd = dict(zip(rz["isic_code"], rz["extfin_all"]))
    agg = (m.groupby(["naics4", "isic"], as_index=False)["weight"].sum()
             .assign(efd=lambda t: t["isic"].map(efd))
             .sort_values(["naics4", "isic"]))

    rows = []
    for n, g in agg.groupby("naics4", sort=True):
        w = g["weight"].to_numpy(np.float64)
        rows.append(dict(
            naics4=int(n),
            extfin_all=float(np.average(g["efd"].to_numpy(np.float64), weights=w)),
            n_isic=len(g),
            spread=float(g["efd"].max() - g["efd"].min()),
            isic_codes=";".join(g["isic"]),
            source_sic3=";".join(sorted({s for s in sic3_owner
                                         if sic3_owner[s] in set(g["isic"])
                                         and s in set(m.loc[m["naics4"] == n, "sic3"])})),
            weight_mass=float(w.sum()),
            mfg=int(n // 100 in MFG_SECTORS),
        ))
    out = pd.DataFrame(rows)
    assert out["naics4"].is_unique
    return out, phantom


def efsy_employment(years):
    """Total EFSY employment by four-digit code, per year, over the window.

    Unresolved roots are MEASURED AND CAPPED, not treated as a failure. They are
    exactly zero from 1998 on, and pre-1998 they are the SIC bridge's residue -
    0.21 to 0.29 per cent of employment - which producer 30 reports in a named
    bin. What would break the coverage arithmetic is a root large enough to
    matter, so that is what the gate is on.
    """
    per, worst, worst_y = {}, 0.0, None
    for y in years:
        d = pd.read_parquet(dc.EMP_OUT_DIR / "naics4_alloc" / f"y{y}.parquet",
                            columns=["code", "emp"])
        g = d.groupby("code")["emp"].sum()
        tot = float(np.abs(g).sum())
        root = float(np.abs(g[g.index >= RESIDUAL3]).sum())
        if tot > 0 and root / tot > worst:
            worst, worst_y = root / tot, y
        per[y] = g[g.index < RESIDUAL3]
    print(f"\n  [{'ok ' if worst <= MAX_ROOT_SHARE else 'FAIL'}] unresolved "
          f"NAICS roots: worst year {worst_y} at {100 * worst:.3f}% of "
          f"employment, against the pre-stated {100 * MAX_ROOT_SHARE:.0f}%. "
          f"Producer 30 reports this in a named bin; it is excluded from the "
          f"four-digit coverage arithmetic below, not from the exhibit.")
    if worst > MAX_ROOT_SHARE:
        raise SystemExit(
            f"FAIL: year {worst_y} carries {100 * worst:.3f}% of employment on "
            f"unresolved NAICS roots. The bins are defined on four-digit codes, "
            f"so at this size a root needs its own four-digit resolution and "
            f"this producer's coverage arithmetic is no longer complete.")
    return per


def coverage(out, per):
    """The table the exhibit's note quotes, and the NAICS97->NAICS12 gate."""
    have = set(out.loc[out["mfg"] == 1, "naics4"])
    have_all = set(out["naics4"])
    rows = []
    print(f"\n  {'year':>6}{'all emp':>14}{'RZ mfg %':>10}{'RZ incl':>9}"
          f"{'mfg %':>8}{'mfg unc %':>11}{'unc codes':>11}")
    worst = 0.0
    for y, g in per.items():
        tot = float(g.sum())
        mfg = g[[c // 100 in MFG_SECTORS for c in g.index]]
        mtot = float(mfg.sum())
        cov = float(g[g.index.isin(have)].sum())
        cov_all = float(g[g.index.isin(have_all)].sum())
        unc = mfg[[c not in have for c in mfg.index]]
        unc_sh = float(unc.sum()) / mtot
        worst = max(worst, unc_sh)
        rows.append(dict(year=y, emp_total=tot, emp_mfg=mtot,
                         rz_share_of_total=cov / tot,
                         rz_share_incl_spillover=cov_all / tot,
                         mfg_share_of_total=mtot / tot,
                         mfg_emp_uncovered_share=unc_sh,
                         n_mfg_codes_uncovered=int((unc > 0).sum())))
        print(f"  {y:>6}{tot:>14,.0f}{100 * cov / tot:>9.2f}%"
              f"{100 * cov_all / tot:>8.2f}%{100 * mtot / tot:>7.2f}%"
              f"{100 * unc_sh:>10.2f}%{int((unc > 0).sum()):>11}")

    ok = worst <= MAX_UNCOVERED_MFG
    print(f"\n  [{'ok ' if ok else 'FAIL'}] NAICS97 -> NAICS12 four-digit "
          f"truncation: worst-year uncovered manufacturing employment "
          f"{100 * worst:.2f}% against the pre-stated {100 * MAX_UNCOVERED_MFG:.0f}%")
    if not ok:
        raise SystemExit("FAIL: the vintage bridge loses too much manufacturing "
                         "employment; the concordance needs a real NAICS97 to "
                         "NAICS2012 crosswalk, not a truncation.")

    last = per[max(per)]
    unc = {int(c): float(last[c]) for c in last.index
           if c // 100 in MFG_SECTORS and c not in have and last[c] > 0}
    for c, e in sorted(unc.items(), key=lambda kv: -kv[1]):
        print(f"        uncovered with employment in {max(per)}: NAICS {c}, "
              f"{e:,.0f} employees")
    return pd.DataFrame(rows)


def report_pooling(out, per):
    o = out[out["mfg"] == 1].copy()
    g = per[min(per)]
    o["emp"] = [float(g.get(n, 0.0)) for n in o["naics4"]]
    tot = o["emp"].sum()
    print(f"\n  POOLING, {len(o)} manufacturing NAICS4 codes with a value, "
          f"{min(per)} employment weights")
    print(f"  {'contributors':>14}{'codes':>7}{'% of mfg emp':>14}{'mean spread':>13}")
    for lo, hi, lab in ((1, 1, "1"), (2, 2, "2"), (3, 3, "3"), (4, 99, ">=4")):
        s = o[o["n_isic"].between(lo, hi)]
        if not len(s):
            continue
        print(f"  {lab:>14}{len(s):>7}{100 * s['emp'].sum() / tot:>13.1f}%"
              f"{s['spread'].mean():>13.3f}")
    print(f"  employment-weighted mean spread "
          f"{np.average(o['spread'], weights=o['emp']):.3f} against an "
          f"extfin_all range of {out['extfin_all'].min():.3f} to "
          f"{out['extfin_all'].max():.3f}")
    print("  the single-contributor cells are the robustness panel WS2 step 4 runs")
    return o


def write_csv(out, phantom):
    p = REF_DIR / "rz_naics4_extfin.csv"
    with open(p, "w", newline="") as f:
        f.write("# Rajan-Zingales (1998) external finance dependence on the "
                "EFSY NAICS4 cell universe.\n")
        f.write("# Built by 71_concord_rz_to_naics4.py, whose docstring carries "
                "every caveat. Route: ISIC rev 2 -> SIC87 (hand table) -> "
                "NAICS97 (Census) -> leading 4 digits.\n")
        f.write("# extfin_all: Fisman and Love's rendering of RZ Table 1 column "
                "1, pooled across contributing ISIC industries by Census "
                "concordance mass.\n")
        f.write("# n_isic, spread: how many ISIC industries fed this cell and "
                "the max-minus-min of their extfin_all. spread > 0 IS "
                "measurement error; n_isic == 1 is the clean subset.\n")
        f.write("# mfg: 1 for NAICS 31-33. Canonical RZ is manufacturing only; "
                "the mfg = 0 rows are concordance spillover from SIC87 "
                "manufacturing into non-manufacturing NAICS97 and a consumer "
                "claiming 'manufacturing only' must restrict to mfg = 1.\n")
        f.write("# NAICS 3253 (fertilizers and pesticides) has NO value on "
                "purpose: RZ Table 1 omits ISIC 3512. It is a reported gap, "
                "never a drop (INVARIANCE_TEST_PROTOCOL.md A.3.1).\n")
        f.write(f"# NAICS97 -> NAICS2012 is four-digit truncation, gated at "
                f"{100 * MAX_UNCOVERED_MFG:.0f}% uncovered manufacturing "
                f"employment over {WINDOW[0]}-{WINDOW[1]}.\n")
        if phantom:
            f.write(f"# SIC3 codes in the hand table that SIC 1987 does not "
                    f"contain, contributing nothing: {', '.join(phantom)}\n")
        f.write("# Cite Rajan and Zingales (1998).\n")
        cols = ["naics4", "extfin_all", "n_isic", "spread", "isic_codes",
                "source_sic3", "weight_mass", "mfg"]
        w = csv.writer(f)
        w.writerow(cols)
        for r in out.sort_values("naics4").itertuples(index=False):
            w.writerow([r.naics4, f"{r.extfin_all:.6f}", r.n_isic,
                        f"{r.spread:.3f}", r.isic_codes, r.source_sic3,
                        f"{r.weight_mass:.6f}", r.mfg])
    return p


def main():
    print("=" * 80)
    print("  Rajan-Zingales external finance dependence, onto EFSY NAICS4")
    print("=" * 80)
    rz = load_rz()
    owner = check_disjoint()
    print(f"  [ok ] the 36 ISIC SIC3 lists are pairwise disjoint: "
          f"{len(owner)} SIC3 codes, one owner each")
    print(f"  [ok ] SIC3 with no ISIC owner, by documented exemption: "
          f"{', '.join(f'{k} ({v})' for k, v in SIC3_NO_ISIC.items())}")

    out, phantom = bridge(rz, owner)
    if phantom:
        print(f"  [ok ] hand-table SIC3 absent from SIC 1987, contributing "
              f"nothing: {', '.join(phantom)}")
    print(f"  NAICS4 codes reached: {len(out)}, of which "
          f"{int(out['mfg'].sum())} in manufacturing and "
          f"{int((out['mfg'] == 0).sum())} concordance spillover outside it")

    years = list(range(WINDOW[0], WINDOW[1] + 1))
    # efsy_employment prints and gates the root share itself. It used to be
    # asserted zero here, which was true on 1998-2016 and is NOT true on the
    # pre-1998 half the window now covers.
    per = efsy_employment(years)
    print(f"  the coverage arithmetic below runs on the four-digit codes of "
          f"{len(years)} years, {years[0]}-{years[-1]}")

    cov = coverage(out, per)
    report_pooling(out, per)

    p = write_csv(out, phantom)
    print(f"\n  wrote {p}")
    cov_out = REF_DIR / "rz_naics4_coverage.csv"
    cov.to_csv(cov_out, index=False, float_format="%.8g")
    print(f"  wrote {cov_out}")
    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
