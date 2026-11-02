"""Stage 8: the full BHJ Assumption-1 discharge on the locked instrument.

Borusyak, Hull and Jaravel (2022, REStud) Section 3.1 (quasi-random shock
assignment), Section 4.1/4.3 (shock-level controls under incomplete shares),
Section 5.1 (exposure-robust inference) and Section 5.2 (falsification and
relevance). Consumes ONLY the persisted pipeline stores. Writes to
Results/pipeline_diagnostics/:

  Table_bhj_balance.tex        six panel blocks:
                               A  exposure-weighted county covariates, the
                                  balance battery (BHJ 5.2, shock level)
                               B  the shock-control ladder in its two forks:
                                  B1 without the year x share-sum control,
                                  B2 with it; beta, exposure-robust SE, the
                                  first-stage F and the control count
                               C  the shocks' own characteristics, descriptive
                               D  clustering and exposure-robust relevance
                               E  the pre-trend test (BHJ 5.2, county level,
                                  exposure-robust SE since 2026-08-27)

  TWO LETTERINGS, DELIBERATELY. The letters above are the TABLE's. CONTINGENCY,
  bhj_balance_locked.csv's `panel` column and the print labels in this file keep
  the ORIGINAL codes, in which the descriptive block is B and the ladder is C.
  The frozen rule says "joint F over Panel A only; Panel B is excluded", and
  re-lettering it would mean editing the one file whose commit IS this table's
  pre-commitment provenance (see precommitment). A and E mean the same thing in
  both; read CONTINGENCY's Panel B as the table's Panel C.
  bhj_balance_locked.csv       every printed row, all three horizons
  ws0_verdict.csv              the two verdicts under CONTINGENCY, with the
                               pre-commitment provenance

WHAT THIS PRODUCER IS FOR. Pipeline 20 reports four balance rows and a joint F
of 1.90 on the locked instrument. That is a thin discharge of the assumption the
whole design rests on, and its standard errors are clustered by county, which
BHJ Section 5.1 shows is not valid when the shocks are what is assigned. 22
replaces both: a seven-covariate battery over shock-level period effects, and
the exposure-robust standard error from the equivalent shock-level regression.

WHAT THE LADDER REPORTS. The seven covariates are the seven TESTS at every rung
-- the battery does not shrink as controls enter, so the joint F is one
comparable statistic and is printed once, under Panel A, instead of repeating
down every rung row. What the rung rows carry is the coefficient and its SE, so
the movement in beta -- what a control ladder exists to show -- is what the eye
lands on going down a fork. Rung 2's controls are five
covariates named in RUNG2_COVARIATES on economic grounds; no rung's control set
is selected from the realised balance t-statistics.

THE ESTIMAND IS UNCHANGED. y_mpgr, x_pos (winsorized within year, the 55/56
convention preserved as locked.WINSOR_DIAG_GROUP) and z are exactly as the lock
defines them. Shock-level controls narrow the identifying variation; under
homogeneous effects they leave the coefficient's definition untouched. Under
heterogeneity BHJ footnote 16's convex weights are functions of s_kt and the
residualized shocks, so the probability limit is a different average of the same
underlying effects. That condition is stated in the table note; whether it is a
protocol gate is Adrien's call, not this file's.

THE PRE-COMMITMENT. The decision rule lives in ws0_contingency.py, NOT here, and
is committed before this file estimates anything. The two are separate on
purpose: the pre-commitment is evidenced by the commit that introduced the rule,
so if the rule shared a file with the code implementing the tests, every edit to
the implementation would move that commit and destroy the provenance. This file
checks that file's introducing commit and refuses to run from a dirty copy of
it. Following 46_build_post_anchor_sweep_panel.py:438 and
62_flow_construction_audit.py:210.
"""
import argparse
import subprocess
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
import store  # noqa: E402
import panel as pnl_lib  # noqa: E402
import oldpath as op  # noqa: E402  (estimator only: t28/w22, as 57 loads them)
from clusterstats import (cluster_sums, hhi_effective,  # noqa: E402
                          shock_ratio, wls_fe)
from sectors import SECTORS, naics2_shares  # noqa: E402
from ws0_contingency import CONTINGENCY  # noqa: E402  (frozen, committed apart)

TAB_DIR = PIPELINE.parents[2] / "Results" / "pipeline_diagnostics"
CONTINGENCY_FILE = PIPELINE / "ws0_contingency.py"

HORIZONS = (4, 3, 2)        # 4 is the verdict cell; 3 and 2 ride along in the CSV
VERDICT_H = 4
SCALE = locked.SCALE
Z95 = 1.959963984540054

# The ladder is run as TWO FORKS of the same control set (see rung_spec).
# Rungs 2, 3 and 4 stack their control on rung 1's year x share-sum
# interaction; 2', 3' and 4' stack the SAME control on rung 0' instead, so the
# fork isolates what that interaction costs. It costs the first stage: F falls
# 16.6 to 3.1 the moment rung 1 enters, and every rung above it inherits that,
# which is why 2/3/4 cannot be read as clean tests of their own control.
#
# REPORT-ONLY. The primed rungs are NOT in CONTINGENCY['admissible_rungs'], so
# adjudicate() skips them and neither verdict moves. Promoting them is a
# revision to the frozen rule, made in ws0_contingency.py with its reason, not
# a side effect of adding rows to a table.
LADDER = (0, "0'", "2'", "3'", "4'", 1, 2, 3, 4)

# WHAT RUNG 2 CONTROLS FOR, named here in advance. Five predetermined county
# characteristics, chosen on economic grounds rather than read off the realised
# balance t-statistics:
#
#   income per head                  the level of development
#   employment rate                  local slack -- the state variable fiscal
#                                    multipliers are theoretically keyed to
#   log population                   market size, in the form a per-head
#                                    outcome calls for
#   Prof./technical services share    NAICS 54, the procurement-facing sector
#   Manufacturing share               NAICS 31-33, procurement-heavy and the
#                                    cyclically sensitive sector
#
# Two of the seven are deliberately NOT here. The non-tradable share is
# near-collinear with the other two shares once the level controls are in (three
# shares off one simplex), and pre-period growth is tested separately, on the
# same exposure-robust inference basis, in Panel E.
#
# Names must match the SECTORS keys in lib/sectors.py, which capitalise; the
# lowercase spelling in ws0_contingency.py's flagged_covariate_map is
# documentation and is not what COVARIATES is keyed by.
#
# THIS SUPERSEDES THE FROZEN RULE'S rule (b) WITHOUT AMENDING IT. Rungs 2 and 2'
# keep their labels and rung 2 keeps its place in
# CONTINGENCY['admissible_rungs'], so that file validates unedited and stands in
# the record describing the |t| > 2.5 flagged set it froze. What changed is what
# the added control IS. Both table notes say so.
RUNG2_COVARIATES = ("income per head", "employment rate", "log population",
                    "Prof./technical services share", "Manufacturing share")

# What each rung's label MEANS, for the table's first column only. The label
# stays the key everywhere that matters -- CONTINGENCY['admissible_rungs'], the
# verdicts, the CSV -- and this decorates it so the table can be read without
# rung_spec in hand. The "+" is against the fork's BASE rung (0' in the primed
# fork, 1 in the other), not against the row above: each rung stacks ONE extra
# control on its base, so 3' carries the group exposures WITHOUT 2's covariate
# block. A rung with no entry here prints as the bare label.
# Keys are STRINGS throughout: tab_ladder is built with str(r) at 1043, so an
# int key here silently misses and the rung prints bare. That is how rungs
# 0-4 lost their names in the regenerated table on 2026-08-27 while the primed
# rungs, whose labels are natively strings, kept theirs.
RUNG_NAME = {
    "0":  "no shock controls",
    "0'": "share sum",
    "2'": r"$+$ predetermined covariates",
    "3'": r"$+$ FSC-group exposure",
    "4'": r"$+$ group $\times$ decade",
    "1":  r"share sum $\times$ year",
    "2":  r"$+$ predetermined covariates",
    "3":  r"$+$ FSC-group exposure",
    "4":  r"$+$ group $\times$ decade",
}


# ---------------------------------------------------------------- provenance --

def precommitment(strict=True):
    """The commit that introduced the frozen rule, and whether it is dirty.

    Checks ws0_contingency.py, not this file: the rule is what is frozen, and
    this file goes on changing as the tests are built. HEAD is not evidence
    either -- it moves with unrelated commits and says nothing about the
    working copy. The introducing commit for the RULE's path, plus a clean
    check on that path, is the pair that means something.
    """
    here = CONTINGENCY_FILE.resolve()
    root = PIPELINE.parents[3]

    def git(*a):
        return subprocess.run(("git", "-C", str(root)) + a,
                              capture_output=True, text=True).stdout.strip()

    commit = git("log", "-1", "--format=%H", "--", str(here))
    dirty = bool(git("status", "--porcelain", "--", str(here)))
    if not commit:
        msg = f"  FAIL: {here.name} has no commit; CONTINGENCY is not frozen"
        if strict:
            raise SystemExit(msg)
        print(msg)
    if dirty:
        msg = (f"  FAIL: {here.name} has uncommitted changes; a decision rule "
               "editable after the numbers land is not a decision rule")
        if strict:
            raise SystemExit(msg)
        print(msg)
    print(f"  pre-commitment: {commit[:12] or 'NONE'} dirty={dirty}")
    return commit, dirty


def show_contingency():
    """Print the frozen rule for the sign-off conversation."""
    import json
    print(json.dumps(CONTINGENCY, indent=2, default=str))


def fsc_group(code):
    """CONTINGENCY['fsc_group_rule'] as code.

    First character for letter-leading PSC codes, first two digits for numeric
    FSC codes. 412 of the 800 locked categories are alphanumeric PSC codes
    whose hierarchy is one letter plus three digits, so a flat first-two-
    characters rule would mix two namespaces.
    """
    c = str(code)
    return c[0] if c[0].isalpha() else c[:2]


def rung_spec(rung):
    """(share_sum, period, extra) for a rung label; the prime drops `period`.

    The primed fork 2'/3'/4' is rung 0' plus the same extra control that 2/3/4
    stack on rung 1 -- the same q_n, without BHJ 4.3's period interaction. That
    is what the prime already means here: 0' is the cross-sectional twin of
    rung 1's panel control, so the notation extends rather than invents.

    Reading the label rather than matching tuples is deliberate. control_block,
    balance and write_table each need to know what a rung is, and three
    hand-maintained membership tests over nine labels is three chances for them
    to disagree about one of them.
    """
    s = str(rung)
    primed = s.endswith("'")
    tier = int(s[:-1] if primed else s)
    if tier == 0:
        return primed, False, None            # 0 -> nothing; 0' -> share sum
    return True, not primed, {1: None, 2: "covariates", 3: "group",
                              4: "decade"}[tier]


def check_effective_counts(sv, kk, jj, cats, h):
    """1/HHI of cluster-summed |s_kt| at every candidate level, against the floor.

    Checked at EVERY level, not just the headline one. The floor is only
    meaningful where it binds, and here it binds in exactly one place: the
    category level clears at 44.1 and the FSC-group level does not, at 13.8.
    Checking the headline level alone would make the floor unfalsifiable.
    """
    grp = np.array([fsc_group(cats[k]) for k in kk])
    eff = {
        "(k,t)": hhi_effective(sv),
        "category": hhi_effective(pd.Series(sv).groupby(kk).sum().to_numpy()),
        "FSC group": hhi_effective(pd.Series(sv).groupby(grp).sum().to_numpy()),
        "year": hhi_effective(pd.Series(sv).groupby(jj).sum().to_numpy()),
    }
    floor = CONTINGENCY["effective_cluster_floor"]
    ref = CONTINGENCY["effective_counts_measured"].get(f"h={h}", {})
    below = []
    for lab, v in eff.items():
        want = ref.get(lab)
        drift = "" if want is None else f" (pre-run {want}"
        if want is not None:
            drift += " DRIFT" if abs(v - want) > 0.05 else ""
            drift += ")"
        flag = ""
        if lab != "(k,t)" and v < floor:
            flag = "  <- BELOW FLOOR"
            below.append(lab)
        print(f"    effective clusters, {lab:<10} {v:8.1f}{drift}{flag}")
    if below:
        print(f"    NOTE: {', '.join(below)} sits below the floor of {floor}. "
              "BHJ 5.3's calibration does not cover it; the SE at that level "
              "is reported with the stated limitation, not as covered.")
    return eff, below


# ------------------------------------------------------------ input build --

def build_inputs(H, z_override=None, gk_override=None):
    """The locked diagnostics cell at horizon H, from the persisted stores.

    Same sample, weights, winsorization and fixed effects as producer 20's
    shock-level table -- 22 keeps its own copy of this block rather than
    sharing 20's. 20's `--phase old` asserts its tables string-equal to the
    captured reruns of 55 and 56, and parameterizing that block over a third
    caller before 22's requirements are known would put a byte-exactness
    guarantee at risk for no gain. The extraction happens after 22 converges.

    THE TWO OVERRIDES ARE PRODUCER 28'S, and both default to None so that 22's
    own path is untouched -- its Table_bhj_balance.tex is byte-compared across
    this change. 28 runs the identical ladder on the industry axis by swapping
    exactly two objects and nothing else:

      z_override   (units x target_years) an industry-axis instrument column
      gk_override  (cats x cy) the industry shock RESTATED onto the FSC axis

    Everything else -- the exposure tensor, the category list, the share sum,
    the FSC groups rungs 3/4 are built on -- stays the locked FSC object,
    because that is what the restatement buys: see lib/industry.py. So the
    sample, the controls and the cluster levels are literally the same objects
    here and in 28, and the only thing that moves between them is the shock.
    """
    pnl = store.load_county_panel()
    units, piv = store.load_pivots(pnl, cols=("P", "E", "Y"))
    DIVI = pnl.drop_duplicates("ccg").set_index("ccg")["division"]
    units2, cats, cy, arr, nat = store.load_tensor(locked.RULE)
    assert np.array_equal(units, units2), "unit order drifted between stores"
    C = store.load_con(locked.RULE).reindex(units)

    years = [t for t in range(max(1977, 1975 + H), locked.TARGET_LAST + 1)]
    lg = {k: C.reindex(columns=[y - k for y in years]).set_axis(years, axis=1)
          for k in range(H + 1)}
    G_b = lg[H].to_numpy(np.float64)
    G_w = (sum(lg[k] for k in range(H)) / H).to_numpy(np.float64)

    def col(v, off):
        return piv[v].reindex(columns=[y - off for y in years]).to_numpy(np.float64)

    P_b, E_b, Y_b = col("P", H), col("E", H), col("Y", H)
    E_t = piv["E"].reindex(columns=years).to_numpy(np.float64)
    E_2b = col("E", 2 * H)

    X = (G_w - G_b) / P_b
    den = E_t + E_b
    Yv = np.where(den != 0, 2.0 * (E_t - E_b) / den, np.nan)
    dpre = E_b + E_2b
    PRE = np.where(dpre != 0, 2.0 * (E_b - E_2b) / dpre, np.nan)
    Z = (store.load_instrument(H, units, "z") if z_override is None
         else np.asarray(z_override, np.float64))
    gk = (store.load_gk(H, cats, cy) if gk_override is None
          else np.asarray(gk_override, np.float64))
    assert Z.shape == (len(units), len(years)), Z.shape
    assert gk.shape == (len(cats), len(cy)), gk.shape

    # C-bar: the predetermined exposure window, one matrix per target year.
    cbar = {}
    for j, t in enumerate(years):
        hi = t - H - locked.LAG
        lo = hi - locked.SPAN + 1
        if lo < locked.DATA_FIRST:
            continue
        a, b = cy.index(lo), cy.index(hi)
        cbar[j] = arr[:, :, a:b + 1].mean(axis=2).astype(np.float64)
    S0 = np.full_like(Z, np.nan)
    for j, m in cbar.items():
        S0[:, j] = m.sum(axis=1) / P_b[:, j]

    ok = (np.isfinite(X) & np.isfinite(Yv) & np.isfinite(Z) & np.isfinite(S0)
          & np.isfinite(E_b))
    ri, ci = np.where(ok)
    d = pd.DataFrame(dict(ccg=units[ri], year=np.array(years)[ci],
                          x=X[ok], y=Yv[ok], z=Z[ok], s0=S0[ok],
                          E_b=E_b[ok], P_b=P_b[ok], Y_b=Y_b[ok], pre=PRE[ok]))
    d["division"] = d["ccg"].map(DIVI)
    xw = pnl_lib.wins(d["x"].to_numpy(np.float64), d["year"].to_numpy())
    tot = d.groupby(["division", "year"])["E_b"].transform("sum")
    w = (d["E_b"] / tot).to_numpy(np.float64)
    w = w / w.mean()
    fe = pd.factorize(d["division"].astype(str) + "_"
                      + d["year"].astype(str))[0].astype(np.int64)
    nofe = np.zeros(len(d), np.int64)
    cl_county = pd.factorize(d["ccg"])[0].astype(np.int64)

    import pyfixest as pf
    op.w22.DEMEANER = pf.MapDemeaner(fixef_tol=1e-10, fixef_maxiter=100_000)
    M = op.w22.demean(np.column_stack(
        [d["y"].to_numpy(np.float64), xw, d["z"].to_numpy(np.float64),
         d["s0"].to_numpy(np.float64)]), fe, nofe, w)

    rowpos = {u: i for i, u in enumerate(units)}
    colpos = {y: j for j, y in enumerate(years)}
    return dict(H=H, d=d, units=units, cats=cats, cy=cy, years=years,
                dollars_k=arr.sum(axis=(0, 2)).astype(np.float64),
                cbar=cbar, gk=gk, w=w, fe=fe, nofe=nofe, xw=xw,
                cl_county=cl_county, yt=M[:, 0], xt=M[:, 1], zt=M[:, 2],
                st=M[:, 3],
                rr=d["ccg"].map(rowpos).to_numpy(),
                cc=d["year"].map(colpos).to_numpy(),
                pb=d["P_b"].to_numpy(np.float64))


# ------------------------------------------------- shock-level aggregation --

def shock_frame(inp):
    """One row per live (k,t): the exposure weight s_kt and the raw shock g_kt.

    s_kt = sum_l w_l s_lkt is BHJ's shock importance weight. (k,t) cells with
    s_kt == 0 carry no county and are dropped: the ratio y-bar = num_y / s_kt
    is undefined there, and producer 20 never hits this because it stays in
    numerator form throughout. 22 stays in numerator form too, so the drop
    changes no coefficient -- only the cluster and effective-shock counts,
    which is why it is logged rather than silent.
    """
    cats, cy, years, cbar, gk = (inp["cats"], inp["cy"], inp["years"],
                                 inp["cbar"], inp["gk"])
    rr, cc, pb, w = inp["rr"], inp["cc"], inp["pb"], inp["w"]
    keys, sv, gv, blocks = [], [], [], {}
    for j in sorted(cbar):
        m = cc == j
        if not m.any():
            continue
        Sk = cbar[j][rr[m], :] / pb[m][:, None]
        wsk = Sk * w[m][:, None]                 # (rows_in_year, ncat)
        blocks[j] = (m, wsk)
        denom = wsk.sum(axis=0)
        tcol = cy.index(years[j])
        for k in range(len(cats)):
            g = gk[k, tcol]
            if not np.isfinite(g):
                continue
            keys.append((k, j))
            sv.append(denom[k])
            gv.append(g)
    keys = np.array(keys)
    sv, gv = np.array(sv), np.array(gv)
    live = np.abs(sv) > 0
    ndrop = int((~live).sum())
    if ndrop:
        print(f"    dropped {ndrop:,} of {len(sv):,} (k,t) cells with s_kt == 0 "
              "(no county carries the category in its exposure window)")
    return dict(keys=keys[live], sv=sv[live], gv=gv[live], blocks=blocks,
                live=live)


def aggregate(inp, sf, v):
    """num_n = sum_l w_l s_lnt v_l, for a COUNTY-RESIDUALIZED v.

    This is the whole identity. Aggregate the residuals, pair them with the RAW
    g, and the shock-level IV reproduces the county 2SLS exactly at every rung,
    because sum_k g_k sum_l w_l s_lk v~_l = sum_l w_l z_l v~_l = sum_l w_l z~_l v~_l
    once v~ is orthogonal to the controls. Never residualize g.
    """
    per_year = {j: wsk.T @ v[m] for j, (m, wsk) in sf["blocks"].items()}
    return np.array([per_year[j][k] for k, j in sf["keys"]])


# ------------------------------------------------------ the control ladder --

def control_block(inp, sf, rung, cov=None):
    """County-level control matrix for a rung, ALREADY two-way demeaned.

    Each rung's shock-level control q_n enters the county regression as the
    exposure-weighted sum_k s_lkt q_kt (BHJ Prop 4). Built here on the county
    side, because that is the side the estimator runs on.
    """
    st, d = inp["st"], inp["d"]
    yr = d["year"].to_numpy()
    share, period, extra = rung_spec(rung)
    if not share:
        return np.zeros((len(st), 0))
    cols = [st.reshape(-1, 1)]
    if period:
        # q_n = period effects  ->  share sum x year
        ys = np.unique(yr)
        cols.append(st[:, None] * (yr[:, None] == ys[None, :]))
    if extra == "covariates" and cov:
        # One q_n per RUNG2_COVARIATES entry, in a fixed order so the control
        # block is reproducible; the caller built them with rung2_controls.
        cols.append(np.column_stack([cov[c] for c in RUNG2_COVARIATES]))
    if extra in ("group", "decade"):
        grp = np.array([fsc_group(c) for c in inp["cats"]])
        cols.append(exposure_weighted_group(inp, sf, grp,
                                            by_decade=(extra == "decade")))
    Cm = np.column_stack(cols)
    Cm = Cm[:, np.abs(Cm * inp["w"][:, None]).sum(0) > 1e-12]
    # EVERY control column is two-way demeaned before the FWL step. This is not
    # cosmetic: the county regression carries division-by-year effects, so the
    # residual must be orthogonal to them for
    #     sum_l w_l z_l v~_l == sum_l w_l z~_l v~_l
    # to hold, and that step is what makes the county and shock routes agree.
    # The share sum and its year interactions are already within-cell demeaned
    # by construction (st is demeaned and the year dummy is constant within a
    # division-year cell), which is why rungs 0'-2 held the identity without
    # this. The exposure-weighted group controls are raw exposures and are not,
    # so rung 3 broke the identity by 2.7e-01 until they were demeaned here.
    # Rank deficiency is left to fwl's SVD.
    return op.w22.demean(Cm, inp["fe"], inp["nofe"], inp["w"])


def exposure_weighted_group(inp, sf, grp, by_decade):
    """sum_k s_lkt 1{group(k) == G}, optionally x decade of the target year."""
    years, cbar, pb, rr, cc = (inp["years"], inp["cbar"], inp["pb"],
                               inp["rr"], inp["cc"])
    gs = sorted(set(grp.tolist()))
    gidx = {g: np.where(grp == g)[0] for g in gs}
    n = len(inp["d"])
    dec = np.array([(years[j] // 10) * 10 for j in cc])
    decs = sorted(set(dec.tolist())) if by_decade else [None]
    out = np.zeros((n, len(gs) * len(decs)))
    for j in sorted(cbar):
        m = cc == j
        if not m.any():
            continue
        blk = cbar[j][rr[m], :] / pb[m][:, None]
        for a, g in enumerate(gs):
            v = blk[:, gidx[g]].sum(axis=1)
            if by_decade:
                b = decs.index((years[j] // 10) * 10)
                out[m, a * len(decs) + b] = v
            else:
                out[m, a] = v
    return out


def fwl(Cm, w, *vs):
    """Residualize each v on the (already demeaned) control block, weighted.

    Projection via a thin SVD of the sqrt(w)-scaled control block, NOT the
    normal equations. The county/shock identity is algebraic but it needs the
    residual to be ORTHOGONAL to the controls to working precision:

        sum_k g_k sum_l w_l s_lk y~_l = sum_l w_l z_l y~_l = sum_l w_l z~_l y~_l

    where the last step is exactly the orthogonality. At rung 3 the block is 37
    period controls plus 93 exposure-weighted group controls, near-collinear by
    construction (the group controls sum to the share sum, which is already in
    there). Solving the normal equations there left the residual only
    approximately orthogonal and the two routes disagreed by 2.7e-01 -- a
    numerical artefact that would have read as a broken identity. The SVD also
    handles the rank deficiency, so no separate collinearity drop is needed.
    """
    if Cm.shape[1] == 0:
        return list(vs)
    sw = np.sqrt(w)
    A = Cm * sw[:, None]
    U, S, _ = np.linalg.svd(A, full_matrices=False)
    U = U[:, S > max(A.shape) * np.finfo(np.float64).eps * S.max()]
    out = []
    for v in vs:
        vs_ = v * sw
        out.append((vs_ - U @ (U.T @ vs_)) / sw)
    return out


def rung_fit(inp, sf, rung, cov=None):
    """beta at a rung, both routes, plus the exposure-robust SE and first stage.

    County route: the FWL ratio (z~'W y~)/(z~'W x~), validated against
    t28.fit2sls at rung 0 where fit2sls can actually run -- it takes one
    regressor, one instrument and two FE factors, and has no covariate slot, so
    rungs above 0 have no house estimator to call.

    Shock route: sum_n g_n num_y,n / sum_n g_n num_x,n, on the SAME residuals.
    The two agree by construction; the assert is a real check on the code, not
    on the mathematics.
    """
    w = inp["w"]
    Cm = control_block(inp, sf, rung, cov)
    yr, xr, zr = fwl(Cm, w, inp["yt"], inp["xt"], inp["zt"])

    b_county = float((w * zr) @ yr) / float((w * zr) @ xr)
    ny, nx, nz = (aggregate(inp, sf, yr), aggregate(inp, sf, xr),
                  aggregate(inp, sf, zr))
    g = sf["gv"]
    b_shock = float(g @ ny) / float(g @ nx)
    gap = abs(b_shock - b_county) / max(abs(b_county), 1e-300)

    return dict(rung=rung, ncontrol=Cm.shape[1], beta=b_county,
                beta_shock=b_shock, gap=gap, ny=ny, nx=nx, nz=nz,
                yr=yr, xr=xr, zr=zr, Cm=Cm)


# ----------------------------------------- tests 8-10: exposure-robust SEs --

def exposure_robust(fit, sf, cats, level):
    """BHJ Prop 5: the shock-level IV's own cluster-robust SE.

    beta = sum_n g_n ny_n / sum_n g_n nx_n, so the moment is g_n (ny_n - b nx_n)
    and the sandwich is (sum g nx)^-2 * sum_c (sum_{n in c} score_n)^2, with
    CRV1's G/(G-1) correction. This IS the exposure-robust standard error: the
    correlation across counties sharing exposure is what the shock-level
    aggregation absorbs.

    No inversion anywhere. BHJ's main text reports this SE and a conventional
    interval; their null-imposed LM interval is a supplementary-appendix
    robustness and WS0 does not run it (see CONTINGENCY['not_run']).
    """
    _, var, G = shock_ratio(sf["gv"], fit["ny"], fit["nx"],
                            cluster_index(sf, cats, level),
                            coef=fit["beta_shock"])
    return float(np.sqrt(var)), G


def first_stage_F(fit, sf, cats, level):
    """Exposure-robust first stage, the object CONTINGENCY['clears'] uses.

    pi = sum_n g_n nx_n / sum_n g_n nz_n with the same shock-level sandwich.
    BHJ 5.3 hand the weak-instrument job to the conventional F rule of thumb
    on exactly this constructed statistic, which is why WS0 needs no
    identification-robust set.
    """
    pi, var, _ = shock_ratio(sf["gv"], fit["nx"], fit["nz"],
                             cluster_index(sf, cats, level))
    if not np.isfinite(var):
        return np.nan
    return float(pi * pi / var) if var > 0 else np.nan


def norm_level(level):
    """Map CONTINGENCY's prose label onto the code's level names.

    The frozen rule says 'category k'; the code keys on 'category'. The rule is
    committed and is not edited to suit the implementation -- the consumer
    normalizes, which is the direction that keeps the pre-commitment intact.
    """
    s = str(level).strip().lower()
    if s.startswith("category"):
        return "category"
    if "group" in s:
        return "FSC group"
    if s.startswith("year"):
        return "year"
    raise SystemExit(f"  FAIL: unknown cluster level {level!r}")


def cluster_index(sf, cats, level):
    k = sf["keys"][:, 0]
    j = sf["keys"][:, 1]
    lev = norm_level(level)
    if lev == "category":
        return pd.factorize(k)[0].astype(np.int64)
    if lev == "FSC group":
        g = np.array([fsc_group(cats[i]) for i in k], dtype=object)
        return pd.factorize(g)[0].astype(np.int64)
    return pd.factorize(j)[0].astype(np.int64)


# ------------------------------------------------ test 11: the balance set --

COVARIATES = ["income per head", "employment rate", "log population",
              "pre-period growth"] + list(SECTORS)


def covariates(inp):
    """The seven predetermined county characteristics, county level, raw.

    Missing values are DROPPED and named, following 21:627-638, not zero-filled
    the way 20:349 and the 55/56 balance block do. At h = 4 the earliest target
    year's pre-period window reaches back before EFSY begins, so zero-filling
    would enter a fabricated zero growth rate into the covariate the whole
    falsification turns on.
    """
    d, H = inp["d"], inp["H"]
    pb = d["P_b"].to_numpy(np.float64)
    V = {
        "income per head": d["Y_b"].to_numpy(np.float64) / pb,
        "employment rate": d["E_b"].to_numpy(np.float64) / pb,
        "log population": np.log(pb),
        "pre-period growth": d["pre"].to_numpy(np.float64),
    }
    base = d["year"].to_numpy() - H
    sh = naics2_shares(sorted(set(base.tolist())), inp["units"])
    key = pd.DataFrame(dict(ccg=d["ccg"].to_numpy(), base_year=base))
    mg = key.merge(sh, on=["ccg", "base_year"], how="left", validate="m:1")
    for c in SECTORS:
        V[c] = mg[c].to_numpy(np.float64)
    dropped = {}
    for c, v in V.items():
        bad = ~np.isfinite(v)
        if bad.any():
            yrs = sorted(set(d["year"].to_numpy()[bad].tolist()))
            dropped[c] = (int(bad.sum()), yrs)
            print(f"    NOTE: '{c}' undefined for {int(bad.sum()):,} rows, "
                  f"years {yrs}")
    return V, dropped


def balance(inp, sf, V, cats, level, rung):
    """Panel A: g_n on exposure-weighted county covariates, over period effects.

    Each covariate is residualized on division-by-year at the COUNTY level
    first (20:349-351's operation order), then exposure-aggregated to the shock
    level, then standardized on s_kt weights. Shock-level period effects are
    the floor, not a rung: BHJ p.26 maintain them when evaluating balance, and
    procurement g_kt has a large common-year component that would otherwise
    show up as imbalance on anything trending.

    THE BATTERY IS ALL SEVEN COVARIATES AT EVERY RUNG. It did not used to be: a
    covariate that became a rung's control was dropped from the printed rows and
    from the joint F, so k ran from 7 down to 3 and the joint p was not
    comparable down a column -- the one thing a ladder's column is for. Every
    rung now reports the same seven tests against the same k, which is also why
    the joint F only needs printing once (write_table puts it under Panel A)
    rather than in every rung row.
    """
    w, fe, nofe = inp["w"], inp["fe"], inp["nofe"]
    g, s = sf["gv"], sf["sv"]
    cl = cluster_index(sf, cats, level)
    good_rows = np.ones(len(inp["d"]), bool)
    for v in V.values():
        good_rows &= np.isfinite(v)

    cols, names = [], []
    for c in COVARIATES:
        v = np.where(good_rows, V[c], 0.0)
        vt = op.w22.demean(v.reshape(-1, 1), fe, nofe, w)[:, 0]
        vt = np.where(good_rows, vt, 0.0)
        num = aggregate(inp, sf, vt)
        den = aggregate(inp, sf, good_rows.astype(np.float64))
        vb = np.where(np.abs(den) > 0, num / np.where(den == 0, 1, den), np.nan)
        m = np.average(vb[np.isfinite(vb)], weights=s[np.isfinite(vb)])
        sd = np.sqrt(np.average((vb[np.isfinite(vb)] - m) ** 2,
                                weights=s[np.isfinite(vb)]))
        cols.append(np.where(np.isfinite(vb), (vb - m) / (sd if sd > 0 else 1), 0.0))
        names.append(c)
    Xm = np.column_stack(cols)

    # Shock-level period effects are the FLOOR at every rung, not a rung.
    # BHJ p.26: "we maintain only the period fixed effects as controls when
    # evaluating balance". Procurement g_kt has a large common-year component,
    # so balance run without them would reject on anything trending.
    yrs = np.unique(sf["keys"][:, 1])
    D = [(sf["keys"][:, 1][:, None] == yrs[None, :]).astype(np.float64)]

    # The rung's own shock-level controls, so the battery is genuinely re-run
    # per rung -- but the seven covariates are always the seven TESTS, never
    # moved across into the control side.
    #
    # It re-runs but it does not always MOVE, and the coincidences are not bugs.
    # Only rungs 3/3' and 4/4' add anything here at all: rungs 0, 0', 1, 2 and 2'
    # give an identical Panel A, because rung 1's county-side control maps to
    # exactly the shock-level period effects that are already the floor above,
    # rungs 0/0' add nothing, and rung 2's covariate controls are the regressors
    # of this very regression rather than additions to it. For the same reason
    # each primed rung matches its unprimed twin -- 2' == 2, 3' == 3, 4' == 4 --
    # since the fork differs only in that period interaction, which this
    # regression already carries either way. So the joint statistic repeats down
    # the ladder, which is the second reason it is reported once.
    extra = rung_spec(rung)[2]
    if extra in ("group", "decade"):
        grp = np.array([fsc_group(cats[i]) for i in sf["keys"][:, 0]],
                       dtype=object)
        key = (grp if extra == "group"
               else np.array([f"{a}_{(inp['years'][b] // 10) * 10}"
                              for a, b in zip(grp, sf["keys"][:, 1])],
                             dtype=object))
        code = pd.factorize(key)[0]
        D.append((code[:, None] == np.arange(code.max() + 1)[None, :])
                 .astype(np.float64))
    A = np.column_stack([Xm] + D)
    b, se, V = wls_shock(g, A, s, cl)
    k = len(names)
    Fj, pj = joint_F(b[:k], V[:k, :k], k, int(np.unique(cl).size))
    rows = [(c, b[q], se[q], b[q] / se[q] if se[q] > 0 else np.nan)
            for q, c in enumerate(names)]
    return rows, Fj, pj


def panel_B(inp, sf, cats, level):
    """Panel B, the table's Panel C: the shocks' OWN characteristics.

    Descriptive, never a trigger. B is the code CONTINGENCY and the CSV use;
    the table prints it as Panel C, next to the cluster counts it speaks to.

    Two rows, both regressed on g_n over shock-level period effects, weighted
    by s_kt and clustered at `level`:

      g_{k,t-h}          the own lagged shock. This is evidence for the CLUSTER
                         choice, not for Assumption 1. BHJ fn.28: "Arbitrary
                         serial correlation of shocks can be allowed here via
                         Assumption 5, with n defining a cluster." A design
                         whose shocks are serially correlated is not thereby
                         non-random; it is a design that must be clustered at
                         the category level, which is what we do.
      log category size  the category's pooled national dollars. Close to
                         mechanical -- s_kt is itself increasing in category
                         size -- so it reports a weighting artefact more than a
                         confounder, and BHJ carry no analogue in their
                         industry covariate set.

    Both are EXCLUDED from the joint F by CONTINGENCY['operative_statistic'],
    which names Panel A only.
    """
    H, cy, years, gk = inp["H"], inp["cy"], inp["years"], inp["gk"]
    kk, jj = sf["keys"][:, 0], sf["keys"][:, 1]
    g, s = sf["gv"], sf["sv"]
    cl = cluster_index(sf, cats, level)

    lag = np.full(len(kk), np.nan)
    for q, (k, j) in enumerate(zip(kk, jj)):
        t = years[j] - H
        if t in cy:
            lag[q] = gk[k, cy.index(t)]
    size = np.log(np.maximum(inp["dollars_k"][kk], 1.0))

    yrs = np.unique(jj)
    D = (jj[:, None] == yrs[None, :]).astype(np.float64)
    rows = []
    for nm, v in (("own lagged shock $g_{k,t-h}$", lag),
                  ("log category size", size)):
        ok = np.isfinite(v) & np.isfinite(g)
        if ok.sum() < 50:
            rows.append((nm, np.nan, np.nan, np.nan, int(ok.sum())))
            continue
        vs = v[ok]
        m = np.average(vs, weights=s[ok])
        sd = np.sqrt(np.average((vs - m) ** 2, weights=s[ok]))
        z = (vs - m) / (sd if sd > 0 else 1.0)
        A = np.column_stack([z, D[ok]])
        b, se, _ = wls_shock(g[ok], A, s[ok], pd.factorize(cl[ok])[0])
        rows.append((nm, b[0], se[0], b[0] / se[0] if se[0] > 0 else np.nan,
                     int(ok.sum())))
    return rows


def wls_shock(y, A, w, cl):
    """Weighted LS with CRV1, dropping columns collinear with earlier ones.

    The rung 3/4 balance design stacks period dummies and group dummies, whose
    columns each sum to one, so A is rank deficient by construction. Solving the
    normal equations there produced negative variances on the diagonal (an
    `invalid value in sqrt`) and a joint F that printed 0.000 -- a number that
    looked like an emphatic rejection and was an artefact.

    QR processes columns in order, so a column dependent on earlier ones gets a
    negligible |R_ii|. Ordering the covariates of interest FIRST therefore keeps
    every one of them and drops only redundant dummies. Coefficients on dropped
    columns come back as nan rather than as zeros, so a dropped column can never
    be mistaken for an estimated null effect.
    """
    sw = np.sqrt(w)
    R = np.linalg.qr(A * sw[:, None], mode="r")
    dg = np.abs(np.diag(R))
    keep = dg > 1e-10 * max(1.0, dg.max())
    Ak = A[:, keep]

    XW = Ak * w[:, None]
    G = Ak.T @ XW
    bk = np.linalg.solve(G, XW.T @ y)
    u = y - Ak @ bk
    inv = np.linalg.inv(G)
    ncl = int(cl.max()) + 1
    S = np.column_stack([cluster_sums(XW[:, q] * u, cl, ncl)
                         for q in range(Ak.shape[1])])
    Gn = int(np.unique(cl).size)
    Vk = (Gn / max(Gn - 1, 1)) * (inv @ (S.T @ S) @ inv)

    b = np.full(A.shape[1], np.nan)
    se = np.full(A.shape[1], np.nan)
    V = np.full((A.shape[1], A.shape[1]), np.nan)
    b[keep] = bk
    se[keep] = np.sqrt(np.maximum(np.diag(Vk), 0.0))
    V[np.ix_(keep, keep)] = Vk
    return b, se, V


def covmat(y, A, w, cl):
    return wls_shock(y, A, w, cl)[2]


def joint_F(b, V, k, G):
    """Cluster-robust Wald/k against F(k, G-1), the frozen reference.

    Columns dropped as collinear by wls_shock arrive as nan and are excluded
    from the test rather than silently treated as zero restrictions.
    """
    from scipy import stats
    ok = np.isfinite(b) & np.isfinite(np.diag(V))
    if not ok.any():
        return np.nan, np.nan
    bb, VV = b[ok], V[np.ix_(ok, ok)]
    kk = int(ok.sum())
    F = float(bb @ np.linalg.solve(VV, bb) / kk)
    return F, float(stats.f.sf(F, kk, max(G - 1, 1)))


def pre_trend(inp, sf, fit, cats, level):
    """Panel E: the pre-trend test. County z on pre-period growth, BHJ 5.2.

    Deliberately OFF the rung ladder. A covariate imbalance says the shocks
    correlate with where counties were, which a shock-level control can absorb.
    A pre-trend says they correlate with where counties were already heading --
    the outcome path itself -- which no q_n fixes, because the confound sits in
    epsilon rather than in anything the control can span.

    INFERENCE. The headline SE is the EXPOSURE-ROBUST one, clustered at
    `level`, built from the same score aggregation exposure_robust() uses. BHJ
    5.2 is explicit that the falsification regressions carry the same exposure-
    clustering problem as the estimate: "To address the exposure clustering
    problem that may arise in these regressions, the shock-level regression of
    Proposition 5 can be used, yielding valid inference on these coefficients."
    Before 2026-08-27 this panel alone reported a county-clustered SE, which is
    the very SE this producer's opening paragraph rejects. The county SE is
    still returned, beside the headline, as the audit trail for that change.

    THE COEFFICIENT DOES NOT MOVE. Only the variance estimator changes; b is
    the same ratio on the same rows as before. Nothing about the outcome, the
    shock, the growth rate or the denominator is touched, so this is not an
    estimand change.

    THE MASKED YEAR, AND WHY THE PROJECTION IS REFIT ON IT. `pre` is undefined
    for the earliest target year, because its window reaches back before EFSY
    begins. The previous convention masked those rows to zero but kept the
    control projection fit on the FULL sample. That cannot carry an
    exposure-robust SE: the score is then not orthogonal to the control block
    in the inner product the shock-level aggregation reproduces, so the
    shock-route score sum does not vanish and the variance picks up a non-zero
    mean term. Measured at h = 3 on the first attempt it was 1.9e-01 against an
    SE of 1.2e-01 -- larger than the quantity it was meant to estimate.

    So the projection is refit on the rows where `pre` exists. This is safe
    here for a reason specific to this panel: the undefined rows are a COMPLETE
    set of division-by-year cells (every county in one year), so dropping them
    leaves the two-way demeaning of every retained cell untouched, and
    control_block already returns a demeaned block, so residualizing on it
    preserves that orthogonality. Both routes then agree exactly and
    `moment_gap` and `denom_gap` return to machine precision -- asserted, not
    merely printed.

    It does move the coefficient slightly, because the rung's control slope is
    now estimated on the retained rows. `b_prev` returns the previous
    convention's value so the size of that move is on the record rather than
    inferred.
    """
    w, fe, nofe = inp["w"], inp["fe"], inp["nofe"]
    pre = inp["d"]["pre"].to_numpy(np.float64)
    good = np.isfinite(pre)
    M0 = op.w22.demean(np.column_stack([np.where(good, pre, 0.0)]),
                       fe, nofe, w)[:, 0]
    M0 = np.where(good, M0, 0.0)

    # the previous convention, kept only to report the move
    zr_full = fit["zr"]
    zz_full = float((w * zr_full)[good] @ zr_full[good])
    b_prev = float((w * zr_full)[good] @ M0[good]) / zz_full

    # ---- refit the rung's projection on the rows where `pre` exists -------
    Cm, sw = fit["Cm"], np.sqrt(w) * good
    if Cm.shape[1]:
        A = Cm * sw[:, None]
        U, S, _ = np.linalg.svd(A, full_matrices=False)
        U = U[:, S > max(A.shape) * np.finfo(np.float64).eps * S.max()]

        def res(v):
            vs = v * sw
            return np.where(good, (vs - U @ (U.T @ vs)) / np.where(sw > 0, sw, 1.0),
                            0.0)
    else:
        def res(v):
            return np.where(good, v, 0.0)

    zr, Mr = res(inp["zt"]), res(M0)
    zz = float((w * zr) @ zr)
    b = float((w * zr) @ Mr) / zz
    u = Mr - b * zr

    # ---- headline: exposure-robust, the BHJ 5.2 / Prop 5 route ------------
    # Aggregate the score and the instrument to the shock level and pair them
    # with the RAW g, exactly as exposure_robust() does for the estimate.
    # Never residualize g.
    g = sf["gv"]
    D = float(g @ aggregate(inp, sf, zr))
    clk = cluster_index(sf, cats, level)
    sck = cluster_sums(g * aggregate(inp, sf, u), clk, int(clk.max()) + 1)
    Gk = int(np.unique(clk).size)
    se = np.sqrt((Gk / max(Gk - 1, 1)) * float(sck @ sck)) / abs(D)

    moment_gap = abs(float(sck.sum())) / abs(D)
    denom_gap = abs(D - zz) / abs(zz)
    if moment_gap > 1e-9 or denom_gap > 1e-9:
        raise SystemExit(
            f"  FAIL: Panel E county/shock identity broken -- score gap "
            f"{moment_gap:.2e}, denominator gap {denom_gap:.2e}. The "
            f"exposure-robust SE is not valid on a score that is not "
            f"orthogonal to the controls")

    # ---- audit trail: the county-clustered SE this replaces --------------
    cl = inp["cl_county"]
    sc = cluster_sums((w * zr * u)[good], cl[good], int(cl.max()) + 1)
    G = int(np.unique(cl[good]).size)
    se_county = np.sqrt((G / max(G - 1, 1)) * float(sc @ sc)) / abs(zz)

    return dict(b=b, se=se, se_county=se_county, n=int(good.sum()),
                G=Gk, G_county=G, b_prev=b_prev,
                moment_gap=moment_gap, denom_gap=denom_gap)


# ------------------------------------------------------------------- main --

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--show-contingency", action="store_true",
                    help="print the frozen decision rule and exit; this is the "
                         "artifact Adrien signs before any estimation runs")
    ap.add_argument("--allow-dirty", action="store_true",
                    help="proceed from an uncommitted working copy. For "
                         "development only; the recorded verdict is void")
    a = ap.parse_args()

    if a.show_contingency:
        show_contingency()
        return

    t0 = time.time()
    commit, dirty = precommitment(strict=not a.allow_dirty)

    LEVEL = CONTINGENCY["cluster_level"]
    REF = CONTINGENCY["reference_rung"]
    FLOOR = CONTINGENCY["effective_cluster_floor"]
    rows, verdict = [], {}
    for H in HORIZONS:
        print(f"== horizon h={H}")
        inp = build_inputs(H)
        sf = shock_frame(inp)
        cats = inp["cats"]
        eff, below = check_effective_counts(sf["sv"], sf["keys"][:, 0],
                                            sf["keys"][:, 1], cats, H)
        V, dropped = covariates(inp)

        # ---- rung 0: the one place the house estimator can run -----------
        f0 = rung_fit(inp, sf, 0)
        house = op.t28.fit2sls(inp["d"]["y"].to_numpy(np.float64), inp["xw"],
                               inp["d"]["z"].to_numpy(np.float64), inp["w"],
                               inp["fe"], inp["nofe"], inp["cl_county"],
                               singleton=False)
        gap = abs(f0["beta"] - house["beta"]) / abs(house["beta"])
        print(f"    rung 0 FWL ratio {f0['beta']*SCALE:.4f} vs t28.fit2sls "
              f"{house['beta']*SCALE:.4f}, rel gap {gap:.1e} "
              f"{'PASS' if gap < locked.RTOL_COEF else 'FAIL'}")
        if gap >= locked.RTOL_COEF:
            raise SystemExit("  FAIL: the FWL ratio does not reproduce the "
                             "house estimator; the substitute used at rungs "
                             "above 0 is not validated")
        se_county = house["se"]

        # ---- the ladder --------------------------------------------------
        # Rung 2/2's covariate controls are built ONCE, from a set named in
        # RUNG2_COVARIATES before anything is estimated, so no rung's control
        # set depends on any rung's realised t-statistics.
        cov = rung2_controls(inp, V)
        ladder = {}
        for rung in LADDER:
            fit = rung_fit(inp, sf, rung, cov)
            if fit["gap"] >= locked.RTOL_COEF:
                raise SystemExit(
                    f"  FAIL: county/shock identity broken at rung {rung}, "
                    f"rel gap {fit['gap']:.2e}; the aggregation is wrong")
            se, G = exposure_robust(fit, sf, cats, LEVEL)
            F = first_stage_F(fit, sf, cats, LEVEL)
            bal, Fj, pj = balance(inp, sf, V, cats, LEVEL, rung)
            ladder[rung] = dict(fit=fit, se=se, G=G, F=F, bal=bal, Fj=Fj, pj=pj)
            print(f"    rung {str(rung):<3} n_ctrl {fit['ncontrol']:>4}  "
                  f"beta {fit['beta']*SCALE:8.4f}  SE {se*SCALE:7.4f}  "
                  f"F {F:8.1f}  joint p {pj:.3f}  identity {fit['gap']:.1e}")
            for c, b, s_, t in bal:
                rows.append(dict(h=H, rung=str(rung), panel="A", item=c,
                                 coef=b, se=s_, t=t))
            rows.append(dict(h=H, rung=str(rung), panel="C", item="beta",
                             coef=fit["beta"] * SCALE, se=se * SCALE, t=F))

        # ---- Panel B, the table's Panel C: shocks' own characteristics ---
        for nm, b_, s_, t_, n_ in panel_B(inp, sf, cats, LEVEL):
            print(f"    Panel B {nm:<30} {b_: .4f} ({s_:.4f})  t {t_: 5.2f}"
                  f"  n {n_:,}")
            rows.append(dict(h=H, rung=str(REF), panel="B", item=nm,
                             coef=b_, se=s_, t=t_))

        # ---- Panel E, off the ladder -------------------------------------
        pe = pre_trend(inp, sf, ladder[REF]["fit"], cats, LEVEL)
        pb, pse = pe["b"], pe["se"]
        print(f"    Panel E pre-trend at rung {REF}: {pb*SCALE:8.4f} "
              f"({pse*SCALE:.4f}), t {pb/pse:5.2f}, n {pe['n']:,}, "
              f"G {pe['G']}  [exposure-robust, {LEVEL}]")
        print(f"            county-clustered SE {pe['se_county']*SCALE:.4f}, "
              f"t {pb/pe['se_county']:5.2f}, G {pe['G_county']:,} "
              f"-- audit only, not the headline")
        print(f"            projection refit on the {pe['n']:,} rows where "
              f"pre exists: beta {pb*SCALE:.4f} vs {pe['b_prev']*SCALE:.4f} "
              f"under the previous full-sample convention")
        print(f"            identity: score gap {pe['moment_gap']:.2e}, "
              f"denominator gap {pe['denom_gap']:.2e}")
        rows.append(dict(h=H, rung=str(REF), panel="E", item="pre-trend",
                         coef=pb, se=pse, t=pb / pse))
        rows.append(dict(h=H, rung=str(REF), panel="E",
                         item="pre-trend, county-clustered SE (audit)",
                         coef=pb, se=pe["se_county"],
                         t=pb / pe["se_county"]))

        # ---- clustering robustness (test 9) ------------------------------
        for lev in CONTINGENCY["cluster_robustness"]:
            se_l, G_l = exposure_robust(ladder[REF]["fit"], sf, cats, lev)
            note = "  BELOW FLOOR" if eff.get(norm_level(lev), np.inf) < FLOOR else ""
            print(f"    cluster {lev:<10} SE {se_l*SCALE:7.4f}  G {G_l:>4}  "
                  f"eff {eff.get(norm_level(lev), np.nan):6.1f}{note}")
            rows.append(dict(h=H, rung=str(REF), panel="D", item=f"SE, {lev}",
                             coef=ladder[REF]["fit"]["beta"] * SCALE,
                             se=se_l * SCALE, t=eff.get(norm_level(lev), np.nan)))

        if H == VERDICT_H:
            verdict = adjudicate(ladder, se_county, pb / pse, eff, below)
            tab_ladder = [(str(r), dict(beta=v["fit"]["beta"] * SCALE,
                                        se=v["se"] * SCALE, F=v["F"],
                                        p=v["pj"], n=v["fit"]["ncontrol"]))
                          for r, v in ladder.items()]
            tab_eff, tab_below = eff, below

    write_table(rows, verdict, tab_ladder, tab_eff, tab_below)
    write_outputs(rows, verdict, commit, dirty)
    print(f"done in {time.time()-t0:.1f}s")


def rung2_controls(inp, V):
    """The q_n that rungs 2 and 2' control for: RUNG2_COVARIATES, county level.

    One control per named covariate, in that tuple's order, each demeaned on the
    county regression's own fixed effects before it goes anywhere. Each enters
    the county regression as the exposure-weighted sum_k s_lkt v-bar_kt, which
    is the object the balance test itself regresses g_n on.

    THIS DOES NOT LOOK AT ANY ESTIMATE. It is the whole point of the change: the
    predecessor selected this control set from the realised balance t-statistics
    at rung 0' (|t| > 2.5, with a wider |t| > 1.96 twin at rungs 5/5'), which
    made the specification a function of the noise in the very test it was
    answering. RUNG2_COVARIATES names the set on economic grounds instead, and
    names it above, before anything runs.
    """
    w, fe, nofe = inp["w"], inp["fe"], inp["nofe"]
    missing = [c for c in RUNG2_COVARIATES if c not in V]
    assert not missing, f"RUNG2_COVARIATES not in the covariate battery: {missing}"
    out = {}
    for c in RUNG2_COVARIATES:
        v = np.where(np.isfinite(V[c]), V[c], 0.0)
        out[c] = op.w22.demean(v.reshape(-1, 1), fe, nofe, w)[:, 0]
    print(f"    rung 2/2' controls, named in advance: {list(RUNG2_COVARIATES)}")
    return out


def adjudicate(ladder, se_county, pre_t, eff, below):
    """The two verdicts, under the frozen ordered bands. No judgement here."""
    # The report-only rungs must stay report-only. Adding a row to the table is
    # not a revision to the frozen rule, and this is where that would show up:
    # a promoted rung would silently enter clears() and could move a verdict.
    # Rung 2 is NOT report-only: it keeps its label and its place in the frozen
    # admissible set. What changed is the content of its control (see
    # RUNG2_COVARIATES), not its standing, and ws0_contingency.py is unamended.
    report_only = {r for r in ladder if str(r).endswith("'")}
    report_only -= {"0'"}          # 0' IS admissible, by Revision 1
    assert not (report_only & set(CONTINGENCY["admissible_rungs"])), (
        "a report-only rung is in admissible_rungs; promoting one is a "
        "revision to ws0_contingency.py with its reason, not a side effect")
    adm = [r for r in CONTINGENCY["admissible_rungs"] if r in ladder]
    clears = [r for r in adm
              if ladder[r]["pj"] > 0.10 and np.isfinite(ladder[r]["F"])
              and ladder[r]["F"] >= 10]
    betas = [ladder[r]["fit"]["beta"] for r in clears]
    ref = ladder[CONTINGENCY["reference_rung"]]
    stable = bool(betas) and (
        len({np.sign(b) for b in betas}) == 1
        and (max(betas) - min(betas)) <= ref["se"])

    pre_rejects = abs(pre_t) > 1.96
    if ref["pj"] > 0.10:
        A = "PASS"
    elif not clears or (pre_rejects and ref["pj"] <= 0.10):
        A = "NO_RUNG_CLEARS"
    elif stable:
        A = "CONTROLS_BIND_STABLE"
    else:
        A = "CONTROLS_BIND_UNSTABLE"

    lo, hi = ref["fit"]["beta"] - Z95 * ref["se"], ref["fit"]["beta"] + Z95 * ref["se"]
    if lo * hi <= 0:
        B = "FAIL"
    elif ref["F"] < 10 or ref["se"] > 2 * se_county:
        B = "MARGINAL"
    else:
        B = "PASS"

    print(f"\n  VERDICT A {A}   VERDICT B {B}")
    print(f"    clearing rungs {clears or 'none'}; stable={stable}; "
          f"rung-{CONTINGENCY['reference_rung']} CI "
          f"({lo*SCALE:.3f}, {hi*SCALE:.3f})")
    if A in ("CONTROLS_BIND_UNSTABLE", "NO_RUNG_CLEARS"):
        print("\n  " + CONTINGENCY["unexplained_move"])
        print("\n  CARRY-FORWARD: " + CONTINGENCY["carry_forward_on_stop"])
    return dict(verdict_a=A, verdict_b=B,
                clearing_rung=(min(clears) if clears else None),
                recommended_rung=(min(clears) if clears else None),
                joint_F=ref["Fj"], joint_p=ref["pj"], n_clusters=ref["G"],
                eff_clusters=eff.get(norm_level(CONTINGENCY["cluster_level"])),
                eff_below_floor=";".join(below) or None,
                beta=ref["fit"]["beta"] * SCALE,
                se_exposure_robust=ref["se"] * SCALE,
                se_county=se_county * SCALE, first_stage_F=ref["F"],
                stable=stable, pre_trend_t=pre_t)


def write_table(rows, verdict, ladder_rows, eff, below):
    """Table_bhj_balance.tex: the six panel blocks, h = 4.

    The DISPLAY lettering is set here and only here (see the module docstring):
    the ladder prints as B1/B2 and the descriptive block as C, while `rows`,
    the CSV and CONTINGENCY keep the codes A/B/C/D/E throughout. `body()` is
    called with the CODE; the local names below say what the block IS, so
    neither lettering has to be carried in the reader's head.
    """
    TAB_DIR.mkdir(parents=True, exist_ok=True)

    def esc(s):
        # Underscores are the one that bites: NO_RUNG_CLEARS reaches the table
        # verbatim and an unescaped _ outside math mode is a compile error, not
        # a cosmetic issue. Item labels also carry raw '&' and '%'.
        return (str(s).replace("\\", r"\textbackslash{}").replace("&", r"\&")
                .replace("%", r"\%").replace("_", r"\_")
                .replace("#", r"\#"))

    REF = str(CONTINGENCY["reference_rung"])

    def body(panel, fmt):
        sel = [r for r in rows if r["h"] == VERDICT_H and r["panel"] == panel
               and r["rung"] == REF]
        return "\n".join(fmt(r) for r in sel)

    def stat_row(r):
        # Panel-B (printed C) labels are authored with math ($g_{k,t-h}$) and
        # are already LaTeX; everything else is raw text and must be escaped.
        lab = r["item"] if "$" in str(r["item"]) else esc(r["item"])
        return (rf"\hspace{{1em}}{lab} & {r['coef']:.4f} & "
                rf"({r['se']:.4f}) & {r['t']:.2f} & & \\")

    bal_body = body("A", stat_row)       # prints as Panel A
    desc_body = body("B", stat_row)      # code B, prints as Panel C

    # The joint statistic, ONCE. It is the same seven-covariate test at every
    # rung now (see balance), so a column of it down the ladder was seven-
    # eighths repetition sitting next to the coefficient the ladder is about.
    ref_se = f"{dict(ladder_rows)[REF]['se']:.4f}"
    joint_row = (rf"\addlinespace\hspace{{1em}}Joint $F$, all seven covariates & "
                 rf"{verdict['joint_F']:.2f} & [{verdict['joint_p']:.3f}] & & & \\")

    def rung_label(k):
        # Label first, name second: the label is what CONTINGENCY, the verdicts
        # and the CSV are keyed to, so it stays the thing the eye lands on.
        nm = RUNG_NAME.get(k)
        ref = " (reference)" if str(k) == REF else ""
        return esc(k) + (f": {nm}{ref}" if nm else ref)

    def ladder_body(rows_):
        # beta, its SE, the first stage and the control count. The movement in
        # beta down the ladder is read off the beta column itself; a derived
        # difference column was tried and removed, because its cell on the
        # continuity rung 0 invited being read as the baseline's own move.
        return "\n".join(
            rf"\hspace{{1em}}{rung_label(k)} & {v['beta']:.4f} & "
            rf"({v['se']:.4f}) & "
            rf"{v['F']:.1f} & {v['n']} & \\" for k, v in rows_)

    # The two forks, split on rung_spec's own `period` flag rather than on a
    # second hardcoded list of labels -- a list here could fall out of step with
    # LADDER and silently drop a rung from the table while it still ran.
    lad_noint_body = ladder_body([(k, v) for k, v in ladder_rows
                                  if not rung_spec(k)[1]])      # prints as B1
    lad_int_body = ladder_body([(k, v) for k, v in ladder_rows
                                if rung_spec(k)[1]])            # prints as B2

    eff_body = "\n".join(
        rf"\hspace{{1em}}{esc(k)} & {v:.1f} & "
        + (r"\textit{below floor}" if k in below else "--") + r" \\"
        for k, v in eff.items())

    tex = TAB_DIR / "Table_bhj_balance.tex"
    tex.write_text(rf"""\begin{{table}}[!htbp]
\caption{{Quasi-Random Shock Assignment for the Composition Instrument}}
\label{{tab:bhj_balance}}
{{\centering
\adjustbox{{max width=\linewidth}}{{%
\begin{{tabular}}{{lrrrrr}}
\toprule
\multicolumn{{6}}{{l}}{{\textbf{{Panel A.}}\hspace{{0.4em}}\textit{{Shock-level balance at rung {esc(REF)}, the reference}}}} \\[0.4ex]
 & Coef. & (s.e.) & $t$ & & \\
\midrule
{bal_body}
{joint_row}
\addlinespace[0.9em]
\multicolumn{{6}}{{l}}{{\textbf{{Panel B1.}}\hspace{{0.4em}}\textit{{The shock-control ladder, without the year $\times$ share-sum control}}}} \\[0.4ex]
 & $\hat\beta$ & (s.e.) & First-stage $F$ & Controls & \\
\midrule
{lad_noint_body}
\addlinespace[0.9em]
\multicolumn{{6}}{{l}}{{\textbf{{Panel B2.}}\hspace{{0.4em}}\textit{{The same ladder with it (BHJ Section 4.3)}}}} \\[0.4ex]
 & $\hat\beta$ & (s.e.) & First-stage $F$ & Controls & \\
\midrule
{lad_int_body}
\addlinespace[0.9em]
\multicolumn{{6}}{{l}}{{\textbf{{Panel C.}}\hspace{{0.4em}}\textit{{The shocks' own characteristics (descriptive; excluded from the joint $F$)}}}} \\[0.4ex]
 & Coef. & (s.e.) & $t$ & & \\
\midrule
{desc_body}
\addlinespace[0.9em]
\multicolumn{{6}}{{l}}{{\textbf{{Panel D.}}\hspace{{0.4em}}\textit{{Effective cluster counts}}}} \\[0.4ex]
 & 1/HHI & & & & \\
\midrule
{eff_body}
\addlinespace[0.9em]
\multicolumn{{6}}{{l}}{{\textbf{{Panel E.}}\hspace{{0.4em}}\textit{{The pre-trend test}}}} \\[0.4ex]
\hspace{{1em}}$t$ on pre-period growth & {verdict['pre_trend_t']:.2f} & & & & \\
\midrule
\multicolumn{{6}}{{l}}{{Verdict A: \textsc{{{esc(verdict['verdict_a'])}}} \quad
Verdict B: \textsc{{{esc(verdict['verdict_b'])}}}}} \\
\bottomrule
\end{{tabular}}}}
\par}}
\medskip
{{\scriptsize \noindent\textit{{Notes:}} The composition instrument at a four-year horizon at the
locked constants, division-by-year fixed effects, division-equal weights, the treatment
winsorized within year. Following Borusyak, Hull and Jaravel, the shift-share IV is rewritten as
an IV at the level of the Federal Supply Classification categories; standard errors are the
exposure-robust ones from that shock-level regression, clustered by category. Shock-level period
effects are maintained in every balance regression, which their Section 5.2 requires under
incomplete shares. The ladder adds shock-level controls, each entering the county regression as
the exposure-weighted $\sum_k s_{{\ell kt}} q_{{kt}}$; the county and shock-level routes are
asserted equal at every rung. The ladder is reported in two forks of the same control set: Panel
B2's rungs stack their control on the period-interacted share sum that their Section 4.3
prescribes under incomplete shares, and Panel B1's primed rungs stack the same control on the
share-sum level instead, so the pair isolates what that interaction costs the first stage. Within
each fork the named control is added to that fork's base rung -- 0' in Panel B1, 1 in Panel B2 --
and not to the rung printed above it, so rung 3 carries the group exposures without rung 2's
covariate block. Rungs 2 and $2'$ control for five predetermined county characteristics: income
per head and log population, for the level of development and market size; the employment rate,
for the local slack a fiscal multiplier is theoretically keyed to; and the professional and
technical services and manufacturing shares, the procurement-facing and the cyclically sensitive
sector. The remaining two of Panel A's seven are deliberately not controls: the non-tradable
share is near-collinear with the other two shares once the levels are in, and pre-period growth
is tested on the same exposure-robust basis in Panel E. That set is named in the producer before
anything is estimated and is \emph{{not}} a function of the realised balance $t$'s. It supersedes
the frozen rule's rule (b), which selected one control per covariate crossing $|t|>2.5$; rungs 2
and $2'$ keep their labels and rung 2 keeps its place in the admissible set, and
\texttt{{ws0\_contingency.py}} is unamended and stands in the record describing the cut it froze.
The
reference rung is 0', the share-sum level control, on record as an editorial choice of which
specification to lead with rather than a claim that Section 4.3's is wrong.
What the ladder is for is the movement in $\hat\beta$ across the rungs, which is read down the
coefficient column against the reference rung's own standard error of {ref_se}.
The balance battery is all seven covariates at every rung, so the joint $F$ is the same test
throughout and is reported once, beneath Panel A, rather than in every rung row: it is the
cluster-robust Wald test that the shocks are unrelated to all seven at once, referred to
$F(k,G-1)$, and it is the operative statistic of the frozen rule, which fixes it as joint over
Panel A only. One or two individual rejections among seven are the expected outcome under the
null and are not evidence against the assumption. Effective cluster
counts are inverse Herfindahls of shock importance; the FSC-group level sits below the 1/20 at
which their Section 5.3 calibrates the approximation, so the group-clustered standard error is
reported with that limitation rather than as covered. Panel C is descriptive and is excluded from the joint $F$: a serially correlated shock is an inference fact answered by the cluster level, not a violation of quasi-random assignment. It is placed beside the cluster counts because that is the question it speaks to. No identification-robust set is reported:
their main text uses the exposure-robust standard error and the conventional first-stage rule of
thumb, and the null-imposed interval is a supplementary-appendix robustness. Panel E's $t$ is
exposure-robust, as their Section 5.2 requires of a falsification regression and not merely of
the estimate; the county-clustered version is carried in the CSV as an audit row only. Panel E
and Panel A's pre-period-growth row test the same null on the same inference basis and are
reverse regressions of one another: Panel E is the county-level coefficient in their Section
5.2 form, Panel A's row is the shock-level form. Panel E's earliest target year is undefined,
because its pre-period window reaches back before the employment source begins, and the rung's
control projection is refit on the rows that remain so that the score stays orthogonal to the
controls; the county and shock-level routes are then asserted equal here as elsewhere. The estimand is
unchanged throughout; shock-level controls narrow the identifying variation, which leaves the
coefficient's definition intact under homogeneous effects.\par}}
\end{{table}}
""")
    print(f"  wrote {tex}")


def write_outputs(rows, verdict, commit, dirty):
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(TAB_DIR / "bhj_balance_locked.csv", index=False)
    verdict = dict(verdict)
    verdict["contingency_commit"] = commit
    verdict["tree_dirty"] = dirty
    pd.DataFrame([verdict]).to_csv(TAB_DIR / "ws0_verdict.csv", index=False)
    print(f"  wrote {TAB_DIR/'bhj_balance_locked.csv'}")
    print(f"  wrote {TAB_DIR/'ws0_verdict.csv'}")


if __name__ == "__main__":
    main()
