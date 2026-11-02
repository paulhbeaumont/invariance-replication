"""WS1 step 6: the shift-share tests on the industry axis, clustered at FSC level.

WS1 step 6 reads: "Run the pipeline 20 shift-share tests on the industry axis,
with inference clustered at the FSC level per the framing above." The framing is
the one that governs everything here: the concordance RELABELS FSC shocks across
industries and creates no new ones, so the effective shock count remains the FSC
count, and industries loading on the same FSC carry mechanically correlated
shifters. A diagnostic that counted 309 independent industry shifters would
overstate the design's identifying variation.

HOW THE FSC CLUSTER IS REACHED, and why it is not a parent lookup. The industry
cells do not partition the FSC categories: under probabilistic allocation one
FSC spreads across many cells, so no cell has "the" FSC parent. Instead the
SHOCK is restated (lib/industry.restate_to_fsc):

    z_c = sum_j cbar_cj g_j = sum_k cbar_ck g~_k ,   g~_k = sum_j share_kj g_j

so the shock-level machinery runs on the LOCKED FSC exposure tensor with g~_k in
place of g_k. 800 well-defined clusters, no ties, no arbitrary parent -- and,
because the exposure side is then the locked one, the sample, the share sum, the
FSC groups rungs 3/4 are built on and every cluster level are LITERALLY the same
objects producer 22 uses. The only thing that moves between 22 and this file is
the shock. That is why this producer calls 22's ladder rather than restating it.

WHERE THE RESTATEMENT IS EXACT, MEASURED NOT ASSUMED. share_kj is a fixed
concordance share, so the identity is exact where producer 23's era rule uses
the concordance. Measured at h=4: EXACT to 3.3e-05 of mean |z| on the 103,972
cells whose exposure window closes by FY2006, and departing only on the three
target years whose window straddles the FY2007 native seam (2016-18, 9,174 cells,
8.1%). The pooled estimate barely notices -- beta 4.0532 on the industry
instrument itself against 4.0754 on its restatement, against an SE of 0.65 -- so
the restated representation is used for the shock-level machinery and the gap is
reported per target year in industry_fsc_restatement.csv rather than papered
over. NO SAMPLE IS RESTRICTED: dropping the seam years would be a sample move,
and it is not needed to make the point.

THE LADDER IS NOT THIS FILE'S TO INVENT. ws0_contingency.py names WS1's industry
axis in carry_forward_on_stop -- "the concordance relabels the same FSC shocks,
so the same controls apply" -- so this producer imports that frozen rule and runs
22's precommitment check against it. One pre-commitment for the whole
shift-share program, not two.

THREE COLUMNS, EVERY TABLE. No intensity column: pipeline 20 contrasts the K=1
intensity instrument against the composition one, and every column here is a
composition-type instrument.

    z             the locked FSC-axis instrument            (reference)
    z_ind_noloo   industry axis, probabilistic allocation   (pooled shock)
    z_ind_modal*  industry axis, modal allocation           (pooled shock)

* The persisted z_ind_modal is LEAVE-ONE-OUT (25 emits no pooled modal column),
  so the pooled modal instrument is rebuilt here from the modal tensor through
  lib/instrument, at the identical constants. It is rebuilt rather than read
  because a LOO column cannot enter the ladder: rung_fit's county/shock identity
  is the same identity the equivalence row rests on, and leave-one-out makes the
  shares county-specific and breaks it structurally. z_ind (LOO) is reported at
  the reference rung by the county route alone, with the missing assert named.

Outputs, all to Results/industry_axis/:
    Table_industry_many_shifters.tex     the GPSS side, three columns
    Table_industry_bhj_shock_level.tex   the BHJ side, three columns
    Table_industry_ladder.tex            the control ladder, 22's panel layout
    industry_*.csv                       every printed row, all three horizons
    industry_fsc_restatement.csv         where the restatement is exact
"""
import argparse
import importlib
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
import instrument as inst  # noqa: E402
import store  # noqa: E402
import industry as ind  # noqa: E402
import oldpath as op  # noqa: E402  (estimator only: t28/w22, as 20/21/22/27 load it)
from clusterstats import hhi_effective  # noqa: E402
from ws0_contingency import CONTINGENCY  # noqa: E402  (frozen, committed apart)

# 22 is the ladder. Its module name leads with a digit, so it is reached through
# importlib rather than a bare import; everything it exposes is a pure function
# of the (inp, sf) dicts, which is what makes this reuse rather than a fork.
p22 = importlib.import_module("22_bhj_assumption1")

IN = dc.MACRO_DATA_ROOT / "temp" / "pipeline_industry"
RESULTS = PIPELINE.parents[2] / "Results" / "industry_axis"

SCALE = locked.SCALE
HORIZONS = (4, 3, 2)          # 4 is the headline cell; 3 and 2 ride along
HEADLINE_H = 4

# The restatement is exact where producer 23's era rule uses the concordance,
# i.e. where the exposure window closes by CONC_LAST_YEAR. Beyond that the
# FSC->industry split is per record and the restatement approximates.
EXACT_RTOL = 1e-3             # of mean |z|, on the exact target years

COLUMNS = [
    ("z",           "FSC axis (locked)",          None),
    ("z_ind_prob",  "Industry axis, probabilistic", "prob"),
    ("z_ind_modal_pooled", "Industry axis, modal",  "modal"),
]


def log(m):
    print(m, flush=True)


# ------------------------------------------------------- the three columns --

def build_pooled_modal(h, units, years):
    """The pooled-shock modal instrument, rebuilt at the locked constants.

    25 persists z_ind_modal as LEAVE-ONE-OUT only, and a LOO column cannot enter
    the ladder (see the module docstring). This rebuilds the modal column the
    way 25 rebuilds its own z_ind_noloo -- lib/instrument, locked SPAN/LAG/CLIP,
    sentinels dropped -- so the modal and probabilistic columns differ in the
    ALLOCATION and in nothing else.
    """
    u, cells, cy, arr_m = ind.load_axis_tensor("dollars_modal")
    assert np.array_equal(u, units)
    cells, (arr_m,), _ = ind.drop_sentinels(cells, arr_m)
    nat = pd.DataFrame(arr_m.sum(axis=0).astype(np.float64), index=cells,
                       columns=cy)
    gk = inst.category_rates(nat, h)
    L, _ = inst.predicted_dollars(arr_m, gk, cy, years, h, "roll", locked.SPAN,
                                  locked.LAG, locked.DATA_FIRST)
    return L, gk, cells, cy


def industry_shock(h, allocation):
    """(g_ind on the industry axis, its cell list, the cy axis) for one allocation.

    The probabilistic side is READ from 25's persisted shocks, so this producer
    cannot drift from the instrument 27 already reports on. The modal side is
    rebuilt, because 25 persists no pooled modal shock.
    """
    if allocation == "prob":
        g = pd.read_parquet(IN / "industry_category_shocks_year.parquet")
        g = g[g["h"] == h].pivot(index="axis4", columns="year", values="g")
        cy = store.years_full()
        return g.reindex(columns=cy).to_numpy(np.float64), list(g.index), cy
    raise ValueError(allocation)


def columns_for(h, units, years, cats, cy):
    """Every instrument column at horizon h, with its restated FSC shock.

    Returns {key: dict(Z, gk_restated, g_ind, cells, label)}. Z is (units x
    years) and gk_restated is (cats x cy) -- exactly the two objects 22's
    build_inputs takes as overrides.
    """
    out = {}
    P_b = store.load_county_panel()
    _, piv = store.load_pivots(P_b, cols=("P",))
    Pb = piv["P"].reindex(index=units, columns=[y - h for y in years]).to_numpy(np.float64)

    units_t, cats_t, cy, arr, _ = store.load_tensor(locked.RULE)
    assert np.array_equal(units_t, units) and list(cats_t) == list(cats)

    out["z"] = dict(Z=store.load_instrument(h, units, "z"), Z_built=None,
                    gk_restated=None, g_ind=None, cells=None,
                    label="FSC axis (locked)")

    apn = pd.read_parquet(IN / "analysis_panel_industry.parquet")
    a = apn[apn["h"] == h]

    g_ind, cells, _ = industry_shock(h, "prob")
    gt = ind.restate_to_fsc(g_ind, ind.load_share_matrix(cats, cells, "prob"))
    out["z_ind_prob"] = dict(
        Z=rebuild_from_restated(h, units, years, cy, gt, arr, Pb),
        Z_built=(a.pivot(index="ccg", columns="year", values="z_ind_noloo")
                 .reindex(index=units, columns=years).to_numpy(np.float64)),
        gk_restated=gt, g_ind=g_ind, cells=cells,
        label="Industry axis, probabilistic")

    L, g_m, cells_m, _ = build_pooled_modal(h, units, years)
    gt_m = ind.restate_to_fsc(g_m, ind.load_share_matrix(cats, cells_m, "modal"))
    out["z_ind_modal_pooled"] = dict(
        Z=rebuild_from_restated(h, units, years, cy, gt_m, arr, Pb),
        Z_built=L / Pb, gk_restated=gt_m, g_ind=g_m, cells=cells_m,
        label="Industry axis, modal")

    # the LOO headline, county route only -- reported, never laddered
    out["z_ind"] = dict(
        Z=(a.pivot(index="ccg", columns="year", values="z_ind")
           .reindex(index=units, columns=years).to_numpy(np.float64)),
        Z_built=None, gk_restated=None, g_ind=None, cells=None,
        label="Industry axis, leave-one-out")
    return out


# ------------------------------------------- the restatement, measured ------

def rebuild_from_restated(h, units, years, cy, gt, arr, Pb):
    """z^r_c = sum_k cbar_ck g~_k / P_b: the industry instrument, restated.

    THIS IS THE OBJECT THE INDUSTRY COLUMNS ARE DIAGNOSED ON, and the reason is
    not convenience. Every shock-level statistic in this file -- the equivalence,
    the effective shock counts, the exposure-robust SE, the whole ladder --
    decomposes the instrument into (exposure, shock) pairs and then asserts that
    recombining them reproduces the county coefficient. That assert is the check
    that the aggregation is right. It can only hold for the instrument the
    decomposition actually describes, so the diagnosed object must be z^r and
    not the industry instrument it approximates.

    z^r IS an industry-axis shift-share instrument: the same national industry
    shocks, met with the same county contract mix, differing only in that the
    mix is carried on the FSC exposure the concordance is keyed on. Where
    producer 23's era rule sources the exposure window from the concordance the
    two are the SAME NUMBER; where the window reaches the FY2007 native seam
    they differ, and restatement_report measures by how much, per target year.
    """
    L = np.full((len(units), len(years)), np.nan)
    for j, t in enumerate(years):
        hi = t - h - locked.LAG
        lo = hi - locked.SPAN + 1
        if lo < locked.DATA_FIRST:
            continue
        a, b = cy.index(lo), cy.index(hi)
        cbar = arr[:, :, a:b + 1].mean(axis=2).astype(np.float64)
        g = gt[:, cy.index(t)]
        live = np.isfinite(g)
        L[:, j] = cbar[:, live] @ g[live]
    return L / Pb


def restatement_report(h, units, years, cats, cy, col, arr):
    """Per target year: does sum_k cbar_ck g~_k reproduce the industry z?

    This is the file that makes the FY2007 seam auditable. It compares the
    restated instrument the diagnostics run on against the instrument producer
    25 actually built on the industry axis, marking each target year by whether
    its exposure window is concordance-sourced.
    """
    gt = col["gk_restated"]
    Z0 = col["Z_built"]
    L = np.full((len(units), len(years)), np.nan)
    for j, t in enumerate(years):
        hi = t - h - locked.LAG
        lo = hi - locked.SPAN + 1
        if lo < locked.DATA_FIRST:
            continue
        a, b = cy.index(lo), cy.index(hi)
        cbar = arr[:, :, a:b + 1].mean(axis=2).astype(np.float64)
        g = gt[:, cy.index(t)]
        live = np.isfinite(g)
        L[:, j] = cbar[:, live] @ g[live]
    Pb = store.load_pivots(store.load_county_panel(), cols=("P",))[1]["P"] \
        .reindex(index=units, columns=[y - h for y in years]).to_numpy(np.float64)
    Zr = L / Pb
    ok = np.isfinite(Zr) & np.isfinite(Z0)
    sc = float(np.nanmean(np.abs(Z0[ok])))
    rel = np.abs(Zr - Z0) / sc

    rows, exact_mask = [], np.zeros(len(years), bool)
    for j, t in enumerate(years):
        m = ok[:, j]
        if not m.any():
            continue
        hi = t - h - locked.LAG
        lo = hi - locked.SPAN + 1
        era = ("concordance" if hi <= ind.CONC_LAST_YEAR
               else "native" if lo > ind.CONC_LAST_YEAR else "seam")
        exact_mask[j] = era == "concordance"
        rows.append(dict(h=h, target_year=t, window_lo=lo, window_hi=hi,
                         era=era, n=int(m.sum()),
                         corr=float(np.corrcoef(Zr[m, j], Z0[m, j])[0, 1]),
                         max_rel_gap=float(rel[m, j].max()),
                         mean_rel_gap=float(rel[m, j].mean())))
    me = ok & exact_mask[None, :]
    worst = float(rel[me].max()) if me.any() else np.nan
    assert worst < EXACT_RTOL, (
        f"h={h}: the FSC restatement is not exact on the concordance-era target "
        f"years (worst {worst:.2e} of mean |z|); the shock-level machinery is "
        "not running on the instrument it claims to")
    log(f"    restatement exact on {int(me.sum()):,} of {int(ok.sum()):,} cells "
        f"(worst {worst:.1e} of mean |z|); "
        f"{int((ok & ~exact_mask[None, :]).sum()):,} cells on seam/native years")
    return rows, Zr


# -------------------------------------------------- the GPSS-side table ----

def shifters_block(inp, sf, col, cats, level, arr, h):
    """Pipeline 20's Table_many_shifters rows, for one column.

    Rewritten against the shock the column actually carries: on the industry
    columns the shifter COUNT and the concentration are the industry cells'
    (that is what the axis change did), while the effective SHOCK count is
    reported on the restated FSC shock as well, because that is the number the
    WS1 framing predicts does not move.
    """
    cy, years = inp["cy"], inp["years"]
    is_ind = col["cells"] is not None
    cells = col["cells"] if is_ind else cats
    g = col["g_ind"] if is_ind else inp["gk"]

    # PC profile of the instrument matrix itself, 20:198
    Zm = col["Z"]
    okm = np.isfinite(Zm)
    c_ok = np.where(okm.any(axis=0))[0]
    r_ok = np.where(okm[:, c_ok].all(axis=1))[0]
    c_ok = c_ok[okm[np.ix_(r_ok, c_ok)].all(axis=0)]
    pc1, k90 = p22_pc_profile(Zm[np.ix_(r_ok, c_ok)])

    # common-cycle R2 over the column's own shifters, 20:201
    G = pd.DataFrame(g, index=cells, columns=cy).loc[:, years[0]:years[-1]]
    common = G.mean(axis=0)
    r2 = []
    for k in G.index:
        v = G.loc[k].dropna()
        if len(v) > 5 and v.std() > 0:
            r2.append(np.corrcoef(v, common.reindex(v.index))[0, 1] ** 2)
    r2 = np.array(r2)

    # national dollars by shifter -> 1/HHI, 20:210
    if is_ind:
        u, cl_all, _, ar = ind.load_axis_tensor(
            "dollars_modal" if "modal" in col["label"].lower() else "dollars_prob")
        cl_all, (ar,), _ = ind.drop_sentinels(cl_all, ar)
        dollars = ar.sum(axis=(0, 2)).astype(np.float64)
    else:
        dollars = inp["dollars_k"]
    eff_dollars = hhi_effective(dollars)

    # effective shocks on the RESTATED FSC axis: the WS1 prediction
    eff_restated = hhi_effective(pd.Series(np.abs(sf["sv"]))
                                 .groupby(sf["keys"][:, 0]).sum().to_numpy())

    # GPSS first-stage concentration, 20:239-255. alpha_k = sum_t sum_l w_l
    # s_lkt x~_l, which is exactly the shock-level aggregate of the demeaned
    # treatment summed over years -- so it comes free from the aggregation that
    # is already built rather than from a second loop over categories.
    nx = p22.aggregate(inp, sf, inp["xt"])
    alpha = pd.Series(nx).groupby(sf["keys"][:, 0]).sum().to_numpy()
    a_abs = np.abs(alpha)
    order = np.argsort(-a_abs)
    top1 = a_abs[order[0]] / a_abs.sum()
    top5 = a_abs[order[:5]].sum() / a_abs.sum()
    eff_rot = hhi_effective(a_abs)

    T = int(len(c_ok))
    return [
        ("Number of shifters $K$", f"{len(cells):,}"),
        ("Shock observations ($K \\times T$)", f"{len(cells)*T:,}"),
        ("Effective number of shifters (1/HHI)", f"{eff_dollars:.1f}"),
        ("Variance in first component", f"{pc1:.3f}"),
        ("Components for 90\\% of variance", f"{k90}"),
        ("Median $R^2$ of shifter on common cycle", f"{np.median(r2):.3f}"),
        ("Shifters with $R^2>0.5$ on common cycle",
         f"{int((r2>0.5).sum())} of {len(r2)}"),
        ("\\addlinespace Largest component's first-stage share", f"{top1:.3f}"),
        ("Five largest components' share", f"{top5:.3f}"),
        ("Effective components, first stage (1/HHI)", f"{eff_rot:.1f}"),
        ("\\addlinespace Effective shocks on the restated FSC axis",
         f"{eff_restated:.1f}"),
    ]


def p22_pc_profile(M):
    """20:65, unchanged."""
    ok = np.isfinite(M).all(axis=1)
    A = M[ok]
    A = A - np.nanmean(A, axis=1, keepdims=True)
    A = A - np.nanmean(A, axis=0, keepdims=True)
    s = np.linalg.svd(A, compute_uv=False)
    v = s ** 2
    v = v / v.sum()
    return float(v[0]), int(np.searchsorted(np.cumsum(v), 0.90) + 1)


# ---------------------------------------------------------------- writers --

def fill(note, **kw):
    """Placeholder substitution for the table notes.

    Plain replacement rather than str.format: the notes are LaTeX and are full
    of literal braces, and escaping every one of them to satisfy format() makes
    the prose unreadable for the sake of three substitutions.
    """
    out = note.replace("<H>", str(kw.get("h", "")))
    out = out.replace("<COMMIT>", str(kw.get("commit", "")))
    return out.replace("<DIRTY>", str(kw.get("dirty", "")))


def esc(s):
    return str(s).replace("_", r"\_").replace("'", r"$'$")


def write_tex(path, body, caption, label, notes):
    RESULTS.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\\begin{table}[!htbp]\n"
        f"\\caption{{{caption}}}\n\\label{{{label}}}\n"
        "{\\centering\n\\adjustbox{max width=\\linewidth}{%\n"
        f"{body}\n}}\n\\par}}\n\\medskip\n"
        "{\\scriptsize \\noindent\\textit{Notes:} " + notes + "\\par}\n"
        "\\end{table}\n")
    log(f"  wrote {path.name} -> {path.parent}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--allow-dirty", action="store_true",
                    help="proceed from an uncommitted ws0_contingency.py. "
                         "Development only; the recorded numbers are void")
    ap.add_argument("--h", type=int, action="append", choices=list(HORIZONS))
    args = ap.parse_args()
    horizons = tuple(args.h) if args.h else HORIZONS

    t0 = time.time()
    commit, dirty = p22.precommitment(strict=not args.allow_dirty)
    LEVEL = CONTINGENCY["cluster_level"]
    REF = CONTINGENCY["reference_rung"]

    import pyfixest as pf
    op.w22.DEMEANER = pf.MapDemeaner(fixef_tol=1e-10, fixef_maxiter=100_000)

    units, cats, cy, arr, _ = store.load_tensor(locked.RULE)
    shift_rows, ladder_rows, bhj_rows, restate_rows = [], [], [], []

    for h in horizons:
        log(f"== horizon h={h}")
        years = store.target_years(h)
        cols = columns_for(h, units, years, cats, cy)

        for key, label, alloc in COLUMNS:
            col = cols[key]
            log(f"  -- {label}")
            inp = p22.build_inputs(h, z_override=col["Z"],
                                   gk_override=col["gk_restated"])
            assert len(inp["d"]) == locked.N_COUNTY[h], (
                key, h, len(inp["d"]), locked.N_COUNTY[h])
            sf = p22.shock_frame(inp)
            eff, below = p22.check_effective_counts(
                sf["sv"], sf["keys"][:, 0], sf["keys"][:, 1], cats, h)

            if col["gk_restated"] is not None:
                rr, _ = restatement_report(h, units, years, cats, cy, col, arr)
                restate_rows += [dict(column=key, **r) for r in rr]
                # The instrument producer 25 actually built, estimated in the
                # same cell. It is not what the shock-level machinery can
                # decompose (see rebuild_from_restated), so it is reported here
                # rather than laddered -- and the gap between these two numbers
                # is the whole cost of the restatement, in the units that matter.
                rb = op.t28.fit2sls(
                    inp["d"]["y"].to_numpy(np.float64), inp["xw"],
                    pd.DataFrame(col["Z_built"], index=units, columns=years)
                      .stack().reindex(
                          pd.MultiIndex.from_arrays(
                              [inp["d"]["ccg"], inp["d"]["year"]])
                      ).to_numpy(np.float64),
                    inp["w"], inp["fe"], inp["nofe"], inp["cl_county"],
                    singleton=False)
                ladder_rows.append(dict(
                    h=h, column=key + "_built",
                    label=label + " (as built, county route only)",
                    rung="0", n_control=0, beta=rb["beta"] * SCALE,
                    se_exposure=np.nan, se_county=rb["se"] * SCALE,
                    F=rb["F"], joint_p=np.nan, G=np.nan, identity=np.nan))
                log(f"    as built (county route only): beta "
                    f"{rb['beta']*SCALE:.4f} ({rb['se']*SCALE:.4f}), "
                    f"F {rb['F']:.1f}")

            # ---- the shifters block
            for stat, v in shifters_block(inp, sf, col, cats, LEVEL, arr, h):
                shift_rows.append(dict(h=h, column=key, label=label,
                                       statistic=stat, value=v))

            # ---- the ladder, 22's, unchanged
            V, _dropped = p22.covariates(inp)
            f0 = p22.rung_fit(inp, sf, 0)
            house = op.t28.fit2sls(
                inp["d"]["y"].to_numpy(np.float64), inp["xw"],
                inp["d"]["z"].to_numpy(np.float64), inp["w"], inp["fe"],
                inp["nofe"], inp["cl_county"], singleton=False)
            gap = abs(f0["beta"] - house["beta"]) / abs(house["beta"])
            assert gap < locked.RTOL_COEF, (key, h, gap)
            se_county = house["se"]

            cov = p22.rung2_controls(inp, V)
            for rung in p22.LADDER:
                fit = p22.rung_fit(inp, sf, rung, cov)
                if fit["gap"] >= locked.RTOL_COEF:
                    raise SystemExit(
                        f"  FAIL: county/shock identity broken at rung {rung} "
                        f"on {key}, rel gap {fit['gap']:.2e}")
                se, G = p22.exposure_robust(fit, sf, cats, LEVEL)
                F = p22.first_stage_F(fit, sf, cats, LEVEL)
                bal, Fj, pj = p22.balance(inp, sf, V, cats, LEVEL, rung)
                ladder_rows.append(dict(
                    h=h, column=key, label=label, rung=str(rung),
                    n_control=fit["ncontrol"], beta=fit["beta"] * SCALE,
                    beta_shock=fit["beta_shock"] * SCALE,
                    # All (k,t) cells on the axis, which is what producer 20
                    # reports. shock_frame drops the s_kt == 0 cells -- they
                    # carry no county and so enter nothing that is weighted by
                    # s_kt -- but the reported count is the full axis, and the
                    # post-drop count rides along in the CSV.
                    n_shock=int(len(sf["live"])),
                    n_shock_live=int(len(sf["gv"])),
                    eff_kt=eff.get("(k,t)", np.nan),
                    eff_cat=eff.get("category", np.nan),
                    se_exposure=se * SCALE, se_county=se_county * SCALE,
                    F=F, joint_p=pj, G=G, identity=fit["gap"]))
                log(f"    rung {str(rung):<3} n_ctrl {fit['ncontrol']:>4}  "
                    f"beta {fit['beta']*SCALE:8.4f}  SE {se*SCALE:7.4f} "
                    f"(county {se_county*SCALE:.4f})  F {F:7.1f}  "
                    f"joint p {pj:.3f}  identity {fit['gap']:.1e}")
                if str(rung) == str(REF):
                    ref_fit, ref_bal = fit, bal

            # ---- Panel A at the reference rung, and Panel E
            for c, b_, s_, t_ in ref_bal:
                bhj_rows.append(dict(h=h, column=key, panel="A", item=c,
                                     coef=b_, se=s_, t=t_))
            pe = p22.pre_trend(inp, sf, ref_fit, cats, LEVEL)
            log(f"    Panel E pre-trend at rung {REF}: {pe['b']*SCALE:.4f} "
                f"({pe['se']*SCALE:.4f}), t {pe['b']/pe['se']:.2f}, "
                f"G {pe['G']}  [exposure-robust, {LEVEL}]")
            # Scaled like every other coefficient in these tables: pre_trend
            # returns the raw slope, and locked.SCALE is the reporting
            # convention 22 applies at print time rather than at storage.
            bhj_rows.append(dict(h=h, column=key, panel="E", item="pre-trend",
                                 coef=pe["b"] * SCALE, se=pe["se"] * SCALE,
                                 t=pe["b"] / pe["se"]))
            bhj_rows.append(dict(
                h=h, column=key, panel="E",
                item="pre-trend, county-clustered SE (audit)",
                coef=pe["b"] * SCALE, se=pe["se_county"] * SCALE,
                t=pe["b"] / pe["se_county"]))
            for lev in CONTINGENCY["cluster_robustness"]:
                se_l, G_l = p22.exposure_robust(ref_fit, sf, cats, lev)
                bhj_rows.append(dict(h=h, column=key, panel="D",
                                     item=f"SE, {lev}",
                                     coef=ref_fit["beta"] * SCALE,
                                     se=se_l * SCALE,
                                     t=eff.get(p22.norm_level(lev), np.nan)))
            for lab, v in eff.items():
                bhj_rows.append(dict(h=h, column=key, panel="D",
                                     item=f"effective clusters, {lab}",
                                     coef=np.nan, se=np.nan, t=v))

        # ---- the LOO column, county route only
        d_loo = cols["z_ind"]
        inp_l = p22.build_inputs(h, z_override=d_loo["Z"])
        r = op.t28.fit2sls(inp_l["d"]["y"].to_numpy(np.float64), inp_l["xw"],
                           inp_l["d"]["z"].to_numpy(np.float64), inp_l["w"],
                           inp_l["fe"], inp_l["nofe"], inp_l["cl_county"],
                           singleton=False)
        ladder_rows.append(dict(
            h=h, column="z_ind", label="Industry axis, leave-one-out",
            rung=str(REF), n_control=np.nan, beta=r["beta"] * SCALE,
            se_exposure=np.nan, se_county=r["se"] * SCALE, F=r["F"],
            joint_p=np.nan, G=np.nan, identity=np.nan))
        log(f"  -- LOO (county route only): beta {r['beta']*SCALE:.4f} "
            f"({r['se']*SCALE:.4f}), F {r['F']:.1f}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    # The row labels carry booktabs spacing macros for the table; the CSV is
    # read by people and by the next producer, so they come off here rather
    # than being maintained as two parallel label lists.
    sh = pd.DataFrame(shift_rows)
    sh["statistic"] = sh["statistic"].str.replace(r"\addlinespace ", "",
                                                  regex=False).str.strip()
    sh.to_csv(RESULTS / "industry_many_shifters.csv", index=False)
    pd.DataFrame(ladder_rows).to_csv(RESULTS / "industry_ladder.csv",
                                     index=False)
    pd.DataFrame(bhj_rows).to_csv(RESULTS / "industry_bhj_shock_level.csv",
                                  index=False)
    pd.DataFrame(restate_rows).to_csv(RESULTS / "industry_fsc_restatement.csv",
                                      index=False)
    write_tables(shift_rows, ladder_rows, bhj_rows, restate_rows, commit, dirty)
    log(f"done in {time.time()-t0:.1f}s")


def write_tables(shift_rows, ladder_rows, bhj_rows, restate_rows, commit, dirty):
    """The three tables, all three columns, at the headline horizon."""
    H = HEADLINE_H
    keys = [k for k, _, _ in COLUMNS]
    labs = {k: l for k, l, _ in COLUMNS}

    # ---- Table 1: the shifters table
    sf = pd.DataFrame(shift_rows)
    sf = sf[sf["h"] == H]
    stats = sf[sf["column"] == keys[0]]["statistic"].tolist()
    L = [r"\begin{tabular}{l" + "r" * len(keys) + "}", r"\toprule",
         " & " + " & ".join(esc(labs[k]) for k in keys) + r" \\", r"\midrule"]
    for s in stats:
        v = [sf[(sf.column == k) & (sf.statistic == s)]["value"].iloc[0]
             for k in keys]
        L.append(f"{s} & " + " & ".join(v) + r" \\")
    lad = pd.DataFrame(ladder_rows)
    lad = lad[(lad.h == H) & (lad.rung == str(CONTINGENCY["reference_rung"]))]
    L.append(r"\midrule")
    for nm, fmt in (("beta", "{:.4f}"), ("se_exposure", "({:.4f})"),
                    ("se_county", "[{:.4f}]"), ("F", "{:.1f}")):
        lbl = {"beta": r"2SLS coefficient ($\times 10^{5}$)",
               "se_exposure": r"\hspace{1.5em}exposure-robust SE, FSC cluster",
               "se_county": r"\hspace{1.5em}county-clustered SE",
               "F": r"\hspace{1.5em}exposure-robust first-stage $F$"}[nm]
        v = [fmt.format(lad[lad.column == k][nm].iloc[0]) for k in keys]
        L.append(f"{lbl} & " + " & ".join(v) + r" \\")
    L += [r"\midrule",
          "Observations & " + " & ".join(f"{locked.N_COUNTY[H]:,}"
                                         for _ in keys) + r" \\",
          r"\bottomrule", r"\end{tabular}"]
    write_tex(
        RESULTS / "Table_industry_many_shifters.tex", "\n".join(L),
        "How Many Shifters the Industry-Axis Instrument Has",
        "tab:industry_many_shifters",
        fill(NOTE_SHIFTERS, h=H))

    # ---- Table 2: the ladder, 22's panel layout
    lad = pd.DataFrame(ladder_rows)
    lad = lad[lad.h == H]
    B1 = ["0", "0'", "2'", "3'", "4'"]
    B2 = ["1", "2", "3", "4"]
    bh = pd.DataFrame(bhj_rows)
    bh = bh[bh.h == H]
    # TWO COLUMNS PER AXIS: the estimate on the left, its diagnostics on the
    # right, so a rung is two lines instead of four. Reading down a rung gives
    # beta over its standard error; reading across gives the first-stage F over
    # the movement in beta against the reference rung. 22 spends one column on
    # each of the four statistics because it carries one instrument; here the
    # axes take the columns and the pairing does the work its column headers did.
    # The joint p is NOT one of the four: it is the same seven-covariate test at
    # every rung, so it is reported once per axis, in
    # Table_industry_bhj_shock_level, rather than repeating down every rung.
    NCOL = 2 * len(keys)
    span = NCOL + 1
    L = [r"\begin{tabular}{l" + "rr" * len(keys) + "}", r"\toprule",
         " & " + " & ".join(r"\multicolumn{2}{c}{%s}" % esc(labs[k])
                            for k in keys) + r" \\",
         " ".join(r"\cmidrule(lr){%d-%d}" % (2 + 2 * i, 3 + 2 * i)
                  for i in range(len(keys))),
         " & " + " & ".join([r"$\hat\beta$ / (s.e.)", r"$F$"]
                            * len(keys)) + r" \\"]

    def panel_block(panel, title):
        """One BHJ-frame panel. Shared by A, D and E so the three cannot drift
        in layout; the value format is the only thing that varies.

        These panels carry ONE number per axis, not a pair, so each spans its
        axis's two columns rather than leaving a ragged empty cell under every
        heading.
        """
        sub = bh[bh.panel == panel]
        if sub.empty:
            return
        L.append(r"\midrule")
        L.append(r"\multicolumn{%d}{l}{%s} \\[0.4ex]" % (span, title))
        for it in sub[sub.column == keys[0]]["item"].tolist():
            v = []
            for k in keys:
                r_ = sub[(sub.column == k) & (sub.item == it)]
                if r_.empty:
                    cell_ = ""
                elif panel == "D" and it.startswith("effective"):
                    cell_ = f"{r_['t'].iloc[0]:.1f}"
                elif panel == "D":
                    cell_ = f"{r_['se'].iloc[0]:.4f}"
                else:
                    cell_ = f"{r_['coef'].iloc[0]:.4f} ({r_['se'].iloc[0]:.4f})"
                v.append(r"\multicolumn{2}{c}{%s}" % cell_)
            L.append(f"{esc(it)} & " + " & ".join(v) + r" \\")

    # 22's panel order: A, then the two ladder forks, then D and E.
    panel_block("A", r"\textbf{Panel A.} \textit{Shock-level balance at rung %s, "
                     r"the reference}" % esc(CONTINGENCY["reference_rung"]))
    for title, rungs in ((r"\textbf{Panel B1.} \textit{The ladder without the "
                          r"year $\times$ share-sum control}", B1),
                         (r"\textbf{Panel B2.} \textit{The same ladder with it "
                          r"(BHJ Section 4.3)}", B2)):
        L += [r"\midrule",
              r"\multicolumn{%d}{l}{%s} \\[0.4ex]" % (span, title)]
        for rg in rungs:
            sub = lad[lad.rung == rg]
            if sub.empty:
                continue
            nm = p22.RUNG_NAME.get(rg, rg)
            ref = r"~(reference)" if rg == str(CONTINGENCY["reference_rung"]) else ""

            def cell(k, fld, fmt):
                r_ = sub[sub.column == k]
                if r_.empty or not np.isfinite(r_[fld].iloc[0]):
                    return ""
                return fmt.format(r_[fld].iloc[0])

            top, bot = [], []
            for k in keys:
                top += [cell(k, "beta", "{:.4f}"), cell(k, "F", "{:.1f}")]
                bot += [cell(k, "se_exposure", "({:.4f})"), ""]
            L.append(f"{esc(rg)}~~{nm}{ref} & " + " & ".join(top) + r" \\")
            L.append(" & " + " & ".join(bot) + r" \\")
    panel_block("D", r"\textbf{Panel D.} \textit{Effective cluster counts and "
                     r"clustering robustness}")
    panel_block("E", r"\textbf{Panel E.} \textit{The pre-trend test, "
                     r"exposure-robust}")
    L += [r"\bottomrule", r"\end{tabular}"]
    write_tex(RESULTS / "Table_industry_ladder.tex", "\n".join(L),
              "The Shock-Control Ladder on the Industry Axis",
              "tab:industry_ladder",
              fill(NOTE_LADDER, h=H, commit=commit[:8],
                          dirty=" (DIRTY)" if dirty else ""))

    # ---- Table 3: the BHJ shock-level table, pipeline 20's rows
    lad = pd.DataFrame(ladder_rows)
    lad = lad[lad.h == H]

    def at(k, rg, fld):
        s = lad[(lad.column == k) & (lad.rung == rg)]
        return np.nan if s.empty else s[fld].iloc[0]

    R0, R1 = "0", str(CONTINGENCY["reference_rung"])
    L = [r"\begin{tabular}{l" + "r" * len(keys) + "}", r"\toprule",
         " & " + " & ".join(esc(labs[k]) for k in keys) + r" \\", r"\midrule"]
    spec = [
        ("Shock-level observations $(k,t)$",
         lambda k: f"{int(at(k, R0, 'n_shock')):,}"),
        ("Effective shocks, each $(k,t)$",
         lambda k: f"{at(k, R0, 'eff_kt'):.1f}"),
        ("Effective shocks, each category once",
         lambda k: f"{at(k, R0, 'eff_cat'):.1f}"),
        ("\\addlinespace County-level 2SLS coefficient",
         lambda k: f"{at(k, R0, 'beta'):.4f}"),
        ("Shock-level IV coefficient",
         lambda k: f"{at(k, R0, 'beta_shock'):.4f}"),
        ("Equivalence gap (relative)",
         lambda k: f"{at(k, R0, 'identity'):.1e}"),
        ("\\addlinespace Coefficient, no share-sum control",
         lambda k: f"{at(k, R0, 'beta'):.4f}"),
        ("\\hspace{1.5em}exposure-robust SE",
         lambda k: f"({at(k, R0, 'se_exposure'):.4f})"),
        ("Coefficient, share sum controlled",
         lambda k: f"{at(k, R1, 'beta'):.4f}"),
        ("\\hspace{1.5em}exposure-robust SE",
         lambda k: f"({at(k, R1, 'se_exposure'):.4f})"),
        ("\\addlinespace Balance: joint $F$ $p$-value at rung " + esc(R1),
         lambda k: f"{at(k, R1, 'joint_p'):.3f}"),
    ]
    for lbl, fn in spec:
        L.append(f"{lbl} & " + " & ".join(fn(k) for k in keys) + r" \\")
    L += [r"\midrule",
          "Observations, county level & "
          + " & ".join(f"{locked.N_COUNTY[H]:,}" for _ in keys) + r" \\",
          r"\bottomrule", r"\end{tabular}"]
    write_tex(RESULTS / "Table_industry_bhj_shock_level.tex", "\n".join(L),
              "Shock-Level Diagnostics on the Industry Axis",
              "tab:industry_bhj_shock_level", fill(NOTE_BHJ, h=H))

    # ---- Table 4: the restatement
    rs = pd.DataFrame(restate_rows)
    rs = rs[(rs.h == H) & (rs.column == "z_ind_prob")]
    agg = rs.groupby("era").agg(years=("target_year", "count"),
                                cells=("n", "sum"),
                                worst=("max_rel_gap", "max"),
                                min_corr=("corr", "min")).reset_index()
    L = [r"\begin{tabular}{lrrrr}", r"\toprule",
         r"Exposure window & Years & County-years & Worst gap & Min.\ corr. \\",
         r"\midrule"]
    for _, r_ in agg.iterrows():
        L.append(f"{r_.era} & {r_.years} & {r_.cells:,} & "
                 f"{r_.worst:.1e} & {r_.min_corr:.4f}" + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write_tex(RESULTS / "Table_industry_restatement.tex", "\n".join(L),
              "Where the FSC Restatement Is Exact",
              "tab:industry_restatement", fill(NOTE_RESTATE, h=H))


NOTE_SHIFTERS = (
    "The composition instrument at an $h=<H>$ horizon on the locked county-year "
    "sample, at the locked baseline constants, with division-by-year fixed "
    "effects, division-equal weights and the treatment winsorized within year. "
    "The estimation cell is identical across columns and identical to pipeline "
    "20's; the only thing that moves is which instrument enters. Column 1 is "
    "the locked FSC-axis instrument. Columns 2 and 3 relabel the same contract "
    "dollars onto 309 four-digit industry cells, spreading each contract "
    "category over its industries by the concordance shares (probabilistic) or "
    "placing it whole on the category's modal industry (modal); both carry a "
    "pooled national shock, so they differ from each other in the allocation "
    "and in nothing else. No intensity column is reported: every column here is "
    "a composition-type instrument. "
    "Two standard errors are given for the coefficient. The bracketed one is "
    "clustered by county, which is what pipeline 20 and producer 27 report and "
    "is included so the numbers stay comparable; the parenthesised one is the "
    "exposure-robust standard error from the equivalent shock-level regression, "
    "clustered at the Federal Supply Classification category, and it is the "
    "headline. Borusyak, Hull and Jaravel Section 5.1 is the reason: when the "
    "shocks are what is assigned, clustering by county is not the relevant "
    "level. "
    "The effective number of shifters is the inverse Herfindahl of national "
    "dollars by shifter, on each column's own axis. The last row reports the "
    "same statistic after the industry shock is restated onto the FSC axis it "
    "came from, and that is the number the design rests on: the concordance "
    "\\emph{relabels} FSC shocks across industries and creates no new ones, so "
    "the effective shock count is a property of the FSC axis however many "
    "industry cells the dollars are spread over. Industry cells loading on the "
    "same contract category carry mechanically correlated shifters. The "
    "industry axis is a robustness and interpretation exercise, not a fresh "
    "identification claim."
)

NOTE_LADDER = (
    "The shock-control ladder of Borusyak, Hull and Jaravel Sections 4.1 and "
    "4.3, run on the industry axis at $h=<H>$ and read off the frozen decision "
    "rule in \\texttt{ws0\\_contingency.py} (commit \\texttt{<COMMIT>}"
    "<DIRTY>), which names this axis explicitly: the concordance relabels the "
    "same contract-category shocks, so the same controls apply. This table does "
    "not define a second rule. "
    "Each rung's shock-level control enters the county regression as the "
    "exposure-weighted sum of that control over categories, two-way demeaned "
    "before partialling out. Panel B2's rungs stack their control on the "
    "period-interacted share sum their Section 4.3 prescribes under incomplete "
    "shares; Panel B1's primed rungs stack the same control on rung $0'$ "
    "instead, so the pair isolates what that interaction costs. Within each "
    "fork the named control is added to that fork's base rung -- $0'$ in B1, 1 "
    "in B2 -- so $2/2'$, $3/3'$ and $4/4'$ are matched pairs differing in one "
    "thing. The primed rungs are report-only and are not in the admissible set. "
    "Rungs 2 and $2'$ control for five predetermined county characteristics: "
    "income per head and log population, for the level of development and "
    "market size; the employment rate, for the local slack a fiscal multiplier "
    "is theoretically keyed to; and the professional and technical services and "
    "manufacturing shares, the procurement-facing and the cyclically sensitive "
    "sector. The other two of Panel A's seven are deliberately not controls: "
    "the non-tradable share is near-collinear with the other two shares once "
    "the levels are in, and pre-period growth is tested on the same "
    "exposure-robust basis in Panel E. That set is named in the producer before "
    "anything is estimated and is \\emph{not} a function of the realised balance "
    "$t$'s. It supersedes the frozen rule's rule (b), which selected one "
    "control per covariate crossing $|t|>2.5$; rungs 2 and $2'$ keep their "
    "labels and rung 2 keeps its place in the admissible set, and "
    "\\texttt{ws0\\_contingency.py} is unamended and stands in the record "
    "describing the cut it froze. "
    "Each axis takes two columns and each rung two lines. The left column of "
    "an axis carries the coefficient, $\\times 10^{5}$, over the exposure-robust "
    "standard error clustered at the contract category, in parentheses; the "
    "right column carries the exposure-robust first-stage $F$. What the ladder "
    "is for is the movement in $\\hat\\beta$ across the rungs, which is read "
    "down an axis's coefficient column against the standard error printed "
    "beneath each estimate. "
    "The balance battery is all seven covariates at every rung, so the joint "
    "$F$ is one comparable test throughout and is reported once per axis, in "
    "the shock-level diagnostics table, rather than in every rung row. It is "
    "the cluster-robust Wald test that the shocks are unrelated to "
    "\\emph{all} of Panel A's predetermined county characteristics at once, "
    "referred to $F(k, G-1)$ -- the operative statistic of the frozen rule, "
    "which fixes it as joint over Panel A only and excludes the shocks' own "
    "characteristics. It is joint rather than a scan of the individual $t$'s "
    "because one or two individual rejections among seven are the expected "
    "outcome under the null and are not evidence against the assumption. "
    "The number of control columns at each rung is in the accompanying CSV. "
    "Rungs 3 and 4 are built on the contract-category \\emph{group} of the "
    "restated shock, not on a NAICS sector. That is the point of the "
    "restatement: the control that absorbs correlated shifters on the FSC axis "
    "absorbs the same shifters here, because the correlation is inherited from "
    "the FSC shocks in the first place. "
    "Panel E's headline $t$ uses the exposure-robust standard error, the same "
    "inference basis as the estimate; the county-clustered version is carried "
    "in the accompanying CSV as an audit row only. Panel E and Panel A's "
    "pre-period-growth row test the same null on the same basis and are reverse "
    "regressions of one another. Balance failing to reject is evidence "
    "consistent with quasi-random assignment and is not proof of it. "
    "The leave-one-out instrument, which is WS1's headline, does not appear in "
    "the ladder and appears in the CSV at the reference rung by the county "
    "route alone: leave-one-out makes the exposure shares county-specific, "
    "which breaks the county/shock-level identity every rung above is validated "
    "against. That is a structural property of the estimator, not a failure of "
    "this design, and it is why the pooled-shock columns carry the ladder."
)

NOTE_BHJ = (
    "The composition instrument at $h=<H>$ on the locked county-year sample, "
    "with division-by-year fixed effects, division-equal weights and the "
    "treatment winsorized within year. Following Borusyak, Hull and Jaravel, "
    "each shift-share IV is rewritten as an IV at the level of the contract "
    "categories, using exposure-weighted aggregates of the outcome and the "
    "treatment and weighting by shock importance. The two coefficients agree to "
    "the reported gap in every column, so each design is a shock-level "
    "experiment in their sense; that agreement is asserted on every run at the "
    "locked coefficient tolerance and the table would not have been written "
    "otherwise. The industry columns reach the category level through the "
    "restatement described in the companion table, which is why the equivalence "
    "holds for them at all -- the leave-one-out instrument, WS1's headline, "
    "cannot be written this way, because leave-one-out makes the exposure "
    "shares county-specific; it is reported in the accompanying CSV by the "
    "county route alone. "
    "The shock-level count is every $(k,t)$ cell on the axis, the same count "
    "pipeline 20 reports for the FSC axis; cells whose exposure weight is zero "
    "carry no county and therefore contribute nothing to any $s_{kt}$-weighted "
    "statistic in the table. "
    "Effective shock counts are inverse Herfindahl indices of the importance "
    "weights, reported both treating each category-year as a shock and treating "
    "each category as one shock, the latter being conservative because a "
    "category's national series is serially correlated. They move very little "
    "across the columns, which is the point: relabelling dollars across 309 "
    "industry cells does not manufacture shocks. "
    "Their framework requires exposure shares summing to a constant; here they "
    "sum to total contract dollars per head, so the estimate is reported with "
    "and without that sum controlled. Standard errors are the exposure-robust "
    "ones, clustered at the contract category. Balance failing to reject is "
    "evidence consistent with quasi-random assignment and is not proof of it."
)

NOTE_RESTATE = (
    "The industry-axis shock is carried back onto the contract-category axis it "
    "came from, so that inference can be clustered at the Federal Supply "
    "Classification level as WS1 requires. Because concordance shares are "
    "fixed, that restatement is an exact identity wherever producer 23's era "
    "rule sources a county's exposure window from the concordance. Where the "
    "window reaches into the FY2007 seam, above which the industry code comes "
    "from the record itself and the split is per record rather than per "
    "category, it is an approximation. This table reports which, at $h=<H>$, "
    "and how large the departure is, measured as the worst county-year gap "
    "relative to mean $|z|$. No sample is restricted on this basis: the pooled "
    "estimate on the restated instrument and on the industry instrument itself "
    "differ by far less than a tenth of a standard error, and dropping the seam "
    "years would be a sample change made to flatter a diagnostic."
)


if __name__ == "__main__":
    main()
