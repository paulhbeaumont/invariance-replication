"""Stage 50: the county lead-lag local projection on the locked instrument.

The county analogue of the state-level lead test in
Results/ns14_diagnostics/Table_NS_parallel_trends.tex, which fails (joint lead
p < 0.001). That test is the INTENSITY instrument on 51 states, biennial, in
Stata, with no lag augmentation, so its leads cannot separate a pre-trend from
the persistence of national procurement. This producer asks the question our own
design has to answer: does the locked COMPOSITION instrument show a pre-trend at
the county level?

Consumes ONLY the persisted pipeline stores. Writes to Results/lp/:

  Figure_lp_irf.pdf   the impulse response: leads, impact, posts, one series
  Figure_lp_irf.tex   the figure's fragment, so it enters a document as a table does
  lp_dynamics.csv     one row per printed number, every series
  lp_summary.csv      one wide row per grid cell: joint lead test, relevance,
                      cluster counts

and, under the robustness flags below, two more:

  Figure_lp_irf1.{pdf,tex}          the same IRF at rungs 0' and 0 (--rung)
  Figure_lp_irf5.{pdf,tex}          the same IRF at rungs 0' and 5' (--rung)
  lp_leads_sample_sensitivity.csv   the lead verdict at three windows
                                    (--leads-sensitivity)

THE ROBUSTNESS GRID (plans/2026-08-27_ws4_maximal_sample.md, 2026-08-27). The
default invocation is the headline and nothing else: the common sample at rung
0'. Three flags widen it, and every widening is a SAMPLE or a CONTROL move on a
fixed estimand -- y_mpgr, z1, x1, the windows, p, the fixed effects, the weights
and the clustering are untouched -- so none of them needs a protocol pass.

  --sample maximal|both      each horizon on its OWN longest window. The front
                             constraint t >= YEAR_LO is p = 3's and binds every
                             horizon including the leads; only the back
                             constraint is horizon-specific. NEVER plotted: the
                             post horizons then sit on different calendar
                             periods, so cross-horizon shape claims would mix
                             dynamics with sample composition.
  --rung 0|both              drop the share-sum level. Common sample only,
                             mirroring the headline. NOT an admissible rung
                             (ws0_contingency.py:34) -- a diagnostic.
  --rung 5'|all              ADD one control per predetermined county covariate
                             in 22's RUNG2_COVARIATES, on top of rung 0'. The set
                             is IMPORTED from producer 22, which names it before
                             estimating anything, so it is not selected from any
                             balance t and the two producers cannot drift. The
                             rung is report-only and carries no verdict.
                             Common sample only. `all` = 0', 0 and 5'.
  --leads-sensitivity        the lead verdict at the maximal, common and
                             pre-seam windows. The three leads share ONE maximal
                             sample, so unlike the posts their joint test is
                             well defined on identical rows at every window.

THE ESTIMAND IS NEW, AND WAS APPROVED BEFORE THIS FILE EXISTED. Two objects here
do not appear anywhere in the locked chain: z1, the composition instrument at
h = 1, and the forward/lead midpoint growth rates. Both are stated as formulas in
Results/lp/lp_formula_memo.md, which also answers the eleven pre-code questions
of INVARIANCE_TEST_PROTOCOL.md and records what the change invalidates (nothing:
locked.HORIZONS is (2,3,4) and no standing exhibit is computed at h = 1).

THE HORIZON GRID, and why it is not symmetric. z1_ct is the t-1 -> t national
change, so the shock year is the interval [t-1, t] and every window is dated
against it.

  LEAD   k = 2,3,4   g_{c,t-k -> t-1}   strictly before the shock year
  IMPACT h = 0       g_{c,t-1 -> t}     the shock year itself
  POST   h = 1..4    g_{c,t -> t+h}     after the shock year

There is no lead at k = 1: g_{c,t-1 -> t-1} is a zero-length window. The
plan-literal overlapping leads g_{c,t-k -> t} for k = 1..4 ride along in the CSV
and are NOT plotted -- they run through the shock year, which is the same defect
the state table's even/odd-year columns exist to break.

LAG AUGMENTATION, p = 3, AND WHY THE TWO SIDES DIFFER. Shock lags z1_{t-1..t-3}
enter every horizon: that is what makes each beta a response to the INNOVATION in
z1 rather than to its level, and it is the specific gap in the state test.
Outcome lags g_{c,t-j-1 -> t-j} enter the impact and post horizons ONLY. On the
lead side they cannot: at p = 3 the first outcome lag g_{c,t-2 -> t-1} IS the
k = 2 lead's dependent variable, and for every k >= 2 the outcome lags lie inside
the lead window. A lead regression is a falsification test, not a projection -- it
asks whether the shock predicts growth that already happened, and conditioning it
on that growth absorbs what the test exists to detect.

WHERE p = 3 COMES FROM. Not a convention, and not a measured order either. The
process that governs p is z1's, since the lags are there to stop persistence from
masquerading as a pre-trend. Measured with clustered standard errors, no z1
partial autocorrelation reaches |t| = 1.6 on 32 year clusters, yet the point
estimates run to 0.14 and do not decay: z1's order is NOT identifiable on 32
years of national variation. A 1/sqrt(N) yardstick on 100,000 county-years is
wrong here by a factor of 20-35, because z1 is built from 800 NATIONAL series
shared by all 3,058 counties. Given that ignorance the risks are asymmetric --
under-augmenting forfeits Montiel Olea and Plagborg-Moller's uniform-validity
guarantee, over-augmenting costs one year of sample per lag and little
efficiency -- so p = 3 is the conservative choice. The outcome, separately, is a
clean AR(1) (t = -8.90 at lag 1), so three outcome lags are two more than that
side needs; they are carried for symmetry with the shock lags.

RUNG 0', NOT RUNG 1. The incomplete-shares control is the share-sum LEVEL S0_ct,
one common slope. No share-sum x year interaction: that is rung 1, and
22_bhj_assumption1.py:88-90 records what it costs -- the first stage falls 16.6
to 3.1 the moment it enters. Rung 0' is also the reference rung and reported
baseline after ws0_contingency.py's Revision 1, so this exhibit sits on the same
rung as the baseline whose pre-trend it tests. --rung 0 drops the level as a
diagnostic; it is not admissible and it makes the pre-trend WORSE, not better
(lp_formula_memo.md 1.4). Rung 1 is not implemented here and control_names
rejects it rather than approximating it.

NO FIRST STAGE IS INVENTED. This is a reduced form; there is no endogenous
regressor, so "weak instrument" has no content here and the hollow-marker
convention of the pipeline's only other figure is deliberately NOT used. What IS
reported, once, is the exposure-robust first stage of the h = 1 treatment on z1,
as a relevance diagnostic with its own caveat (see first_stage).

INFERENCE. Exposure-robust at the shock level (BHJ Section 5.1), clustered on
category k per ws0_contingency.py's frozen cluster_level, with county-clustered
standard errors reported alongside in the CSV. County clusters alone are not
valid in a shift-share design. No HAC term: once the regressor is the innovation
the score is a martingale difference, which is the corollary of the augmentation,
not an omission.

Sample-build block: a FOURTH copy, on purpose. PIPELINE.md:181-188 defers the
extraction until the interface is designed against known callers rather than
guessed ones.
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
import locked  # noqa: E402
import store  # noqa: E402
import instrument  # noqa: E402
import panel as pnl_lib  # noqa: E402
from sectors import naics2_shares  # noqa: E402  (the rung-5' sector share)
import oldpath as op  # noqa: E402  (estimator only: t28/w22, as 57 loads them)
from clusterstats import cluster_sums, hhi_effective, wls_fe  # noqa: E402
from ws0_contingency import CONTINGENCY  # noqa: E402  (frozen, committed apart)

TAB_DIR = PIPELINE.parents[2] / "Results" / "lp"

H = 1                       # the shock's own differencing horizon
P_LAG = 3                   # lag augmentation depth (see the docstring)
# The joint lead test is a THREE-degree-of-freedom statement about the three
# reported leads, and the increment route is a re-parameterisation of that same
# statement on disjoint windows -- so it carries three increments whatever the
# lag depth is. This used to be spelled P_LAG, which is right at p = 3 and
# silently turns the increment test into a p-degree-of-freedom object the moment
# the depth moves. The two numbers are unrelated and are now named separately.
N_INCREMENT = 3
LEADS = (4, 3, 2)           # non-overlapping, window [t-k, t-1]
POSTS = (1, 2, 3, 4)        # window [t, t+h]
LEADS_LITERAL = (1, 2, 3, 4)  # plan-literal, window [t-k, t]; CSV only

# ---- the sample window, DERIVED from locked.TARGET_LAST ----------------------
# No year is written down in this block. Every one of them follows from
# locked.TARGET_LAST and the instrument's own window rule, so the end of the
# sample is moved in locked.py and this producer follows it.
#
# The indirection is not tidiness. SAMPLE_LAST and Z1_FINITE are measured
# against the stores in load_stores, so a stale value in either fails loud.
# YEAR_HI is not, and cannot be: county E is finite past locked.TARGET_LAST in
# the panel, so a YEAR_HI reaching beyond it would satisfy build_window's
# rectangle assert and the post horizons would quietly project onto the raw
# undesuppressed CBP years that TARGET_LAST exists to exclude (locked.py:61-76).
# YEAR_HI is the one window constant with no guard behind it, which is exactly
# why it is computed and never written down.
#
# Z1_FIRST is the instrument's front constraint. The exposure window
# lo = t - H - LAG - SPAN + 1 must clear DATA_FIRST, so z is first finite at
# DATA_FIRST + LAG + SPAN - 1 + H. load_stores measures that against the stores
# rather than trusting the arithmetic here.
Z1_FIRST = locked.DATA_FIRST + locked.LAG + locked.SPAN - 1 + H
SAMPLE_LAST = locked.TARGET_LAST
Z1_FINITE = locked.N_COUNTIES * (SAMPLE_LAST - Z1_FIRST + 1)

# The common window. Its front is the deepest shock lag z1_{c,t-p}; its back is
# the furthest-reaching post horizon, which needs E_{c,t+h}. Writing the back as
# SAMPLE_LAST - max(POSTS) is also what makes main()'s "post h = 4's maximal
# window IS the common window" identity true by construction rather than by
# coincidence.
YEAR_LO = Z1_FIRST + P_LAG
YEAR_HI = SAMPLE_LAST - max(POSTS)
N_YEARS = YEAR_HI - YEAR_LO + 1
N_EXPECT = locked.N_COUNTIES * N_YEARS
PRESEAM_HI = locked.SEAM_YEAR - 1   # the last wholly pre-USAspending year

SCALE = locked.SCALE
Z95 = 1.959963984540054
WEAK_F = 10.0

# ---- rung 5': rung 0' plus the predetermined covariates ---------------------
# 22_bhj_assumption1.py owns the ladder. Its LADDER carries a PRIMED fork that
# stacks each control on rung 0' instead of rung 1, and the covariate rung of
# that fork is 22's rung 2': rung 0' plus one control per covariate in 22's
# RUNG2_COVARIATES.
#
# THE SET IS IMPORTED, NEVER RESTATED OR RE-SELECTED. It used to be selected
# here, from the covariates whose balance t at rung 0' exceeded 1.96, read off
# 22's frozen CSV. That selection is gone: on 2026-08-31 22 retired both of its
# t-based cuts in favour of a set named in advance on economic grounds -- income
# per head and log population for development and market size, the employment
# rate for local slack, and the professional/technical services and
# manufacturing shares for sector composition. This producer takes that tuple
# straight from 22 rather than keeping a second copy, so the two cannot drift
# and no threshold survives anywhere in the ladder.
#
# THE LABEL 5' IS KEPT, DELIBERATELY. 22 has no rung 5' any more, and this rung
# is the LP frame's twin of 22's rung 2'. The label and the exhibit filename stay
# as they are because they are what the record and PIPELINE.md already name;
# what changed is the content of the control, and this comment is where that is
# stated. Rung 5' carries no verdict, here or in 22.
RUNG5 = "5'"
_p22 = importlib.import_module("22_bhj_assumption1")
COVARIATES = _p22.RUNG2_COVARIATES      # the single definition, imported
# One key per covariate in the demeaned frame. Prefixed so a covariate can never
# collide with z1, s0, x1, a lag or an outcome. Every name in COVARIATES must
# appear here and be constructed in build_inputs; build_cov asserts it.
COV_KEY = {"income per head": "cov_income_per_head",
           "employment rate": "cov_employment_rate",
           "log population": "cov_log_population",
           "Prof./technical services share": "cov_prof_technical_share",
           "Manufacturing share": "cov_manufacturing_share"}
# The share of rows a control may be zero-filled on before it stops being a
# control. Nothing is expected to bind: EFSY NAICS2 cells exist for every base
# year in [YEAR_LO - H, SAMPLE_LAST - H] this producer asks for, and Y, E and P
# are complete on the common window. The guard exists so a future store gap cannot pass as a control
# that is mostly zeros.
COV_ZEROFILL_MAX = 0.001

# ---- the robustness grid (2026-08-27_ws4_maximal_sample.md, plus rung 0) ----
# SAMPLE_LAST and PRESEAM_HI are defined once, in the derived window block near
# the top of the module. Every maximal window below is SAMPLE_LAST minus the
# horizon's own forward reach.

# ---- the pre-trend grid (--pretrend-grid) -----------------------------------
# Three specification choices that the grid above does NOT vary, each of which
# moves the lead verdict in the same direction. Rung 0' throughout; the headline
# exhibits are not rewritten by this flag.
#
#   WEIGHTS   the headline weight is E_{c,t-1} over its division-year total. For
#             the shock year and the posts that is a PREDETERMINED quantity. For
#             the leads it is the TERMINAL employment of the window
#             g_{c,t-k -> t-1}, so the regression weight is a monotone function
#             of the dependent variable's own numerator and a county that grew
#             fast over the pre-period is up-weighted in the test meant to find
#             exactly that. 57:116-118 fixes the locked baseline's weight at one
#             calendar year for this reason. Equal weighting is the comparison.
#   P_GRID    the k = 4 lead spans [t-4, t-1] and the posts say a z1 shock still
#             moves employment at h = 4, so growth inside the lead window
#             responds to z1_{t-4} and deeper, none of it controlled at p = 3.
#             z1 is persistent, so z1_t is correlated with those. The depth was a
#             module constant with no flag while the docstring conceded z1's
#             order is not identifiable; this varies it.
#   the df    joint_test spends G - 1 = 799 degrees of freedom on 800 categories
#             while effective_counts measures 43.6 effective shifters at that
#             same level. Reported side by side, never substituted.
P_GRID = (3, 4, 5, 6, 7, 8)
WEIGHTS = ("division-EW", "equal")
GRID_RUNG = "0'"

# The front constraint is p = 3's z1_{c,t-3} and is IDENTICAL for all eight
# horizons, so every maximal window starts at YEAR_LO. Only the back constraint
# is horizon-specific: post h needs E_{c,t+h}.
def maximal_hi(kind, hz):
    """The last target year at which one horizon is defined."""
    return SAMPLE_LAST - hz if kind == "post" else SAMPLE_LAST


# The four families, as main() enumerates them.
ALL_KINDS = (("lead", LEADS), ("impact", (0,)), ("post", POSTS),
             ("lead_literal", LEADS_LITERAL))
LEAD_KINDS = (("lead", LEADS), ("lead_literal", LEADS_LITERAL))


# ------------------------------------------------------------- shock groups --

def fsc_group(code):
    """CONTINGENCY['fsc_group_rule'] as code (22_bhj_assumption1.py:160).

    First character for letter-leading PSC codes, first two digits for numeric
    FSC codes. A flat first-two-characters rule would mix two namespaces.
    """
    c = str(code)
    return c[0] if c[0].isalpha() else c[:2]


def norm_level(level):
    """Map CONTINGENCY's prose label onto the code's level names.

    The frozen rule says 'category k'; the code keys on 'category'. The rule is
    committed and is not edited to suit the implementation.
    """
    s = str(level).strip().lower()
    if s.startswith("category"):
        return "category"
    if "group" in s:
        return "FSC group"
    if s.startswith("year"):
        return "year"
    raise SystemExit(f"  FAIL: unknown cluster level {level!r}")


def effective_counts(sf, cats):
    """1/HHI of cluster-summed |s_kt| at every candidate level.

    Measured the way 22_bhj_assumption1.py:194 measures it -- cluster-summed
    FIRST, then the inverse Herfindahl. The raw (k,t) figure is a different
    object and is far larger; comparing it to the frozen floor of 20 would be
    comparing two things that are not the same statistic. Checked at every
    level, not just the headline one, because a floor that is only ever read
    where it clears is unfalsifiable.
    """
    kk, jj, sv = sf["keys"][:, 0], sf["keys"][:, 1], sf["sv"]
    grp = np.array([fsc_group(cats[k]) for k in kk])
    eff = {
        "(k,t)": hhi_effective(sv),
        "category": hhi_effective(pd.Series(sv).groupby(kk).sum().to_numpy()),
        "FSC group": hhi_effective(pd.Series(sv).groupby(grp).sum().to_numpy()),
        "year": hhi_effective(pd.Series(sv).groupby(jj).sum().to_numpy()),
    }
    floor = CONTINGENCY["effective_cluster_floor"]
    below = []
    for lab, v in eff.items():
        flag = ""
        if lab != "(k,t)" and v < floor:
            flag = "  <- BELOW FLOOR"
            below.append(lab)
        print(f"    effective shifters, {lab:<10} {v:8.1f}{flag}")
    if below:
        print(f"    NOTE: {', '.join(below)} sits below the floor of {floor}; "
              "the SE at that level carries the stated limitation.")
    return eff, below


def cluster_index(sf, cats, level):
    k, j = sf["keys"][:, 0], sf["keys"][:, 1]
    lev = norm_level(level)
    if lev == "category":
        return pd.factorize(k)[0].astype(np.int64)
    if lev == "FSC group":
        g = np.array([fsc_group(cats[i]) for i in k], dtype=object)
        return pd.factorize(g)[0].astype(np.int64)
    return pd.factorize(j)[0].astype(np.int64)


# -------------------------------------------------------------- input build --

def flagged_names():
    """Rung 5's covariates: 22's RUNG2_COVARIATES, in a fixed order.

    Nothing is selected and nothing is read from an estimate. The set is
    imported from producer 22, which names it before it estimates anything, so
    there is one definition of what the covariate rung controls for and this
    producer cannot drift from it.

    Sorted, so the control block's column order is a property of the set rather
    than of the order 22 happens to list it in.
    """
    names = sorted(COVARIATES)
    missing = [c for c in names if c not in COV_KEY]
    if missing:
        raise SystemExit(
            f"  FAIL: no county-level constructor for covariate(s) {missing}. "
            "COV_KEY and build_inputs must both learn it before rung "
            f"{RUNG5} can control for it. 22's RUNG2_COVARIATES has moved and "
            "this producer has not followed it.")
    print(f"    rung {RUNG5} controls, imported from 22's RUNG2_COVARIATES: "
          f"{names}")
    return names


def load_stores():
    """Everything the LP frame needs that does NOT depend on the target window.

    Split out of build_inputs so a robustness grid over six windows pays the
    store and tensor cost ONCE. z1 is built here rather than read:
    store.load_instrument and store.load_gk are keyed on
    locked.HORIZONS = (2,3,4) and there is no persisted h = 1 object. Every
    locked constant is inherited -- SPAN, LAG, DATA_FIRST, RULE, DENOM, CLIP --
    so the only thing that changes relative to the locked instrument is h, which
    moves the window offset and the differencing horizon of the national shock
    and nothing else.
    """
    pnl = store.load_county_panel()
    # Y joins P and E for the rung-5' covariate "income per head". It is
    # loaded unconditionally -- the pivot is cheap, and a store set that depends
    # on a command-line flag would make the two rungs run off different frames.
    units, piv = store.load_pivots(pnl, cols=("P", "E", "Y"))
    DIVI = pnl.drop_duplicates("ccg").set_index("ccg")["division"]
    units2, cats, cy, arr, nat = store.load_tensor(locked.RULE)
    assert np.array_equal(units, units2), "unit order drifted between stores"

    # ---- z1 on its own full target-year grid, so lags can be taken -----------
    ty = store.target_years(H)
    gk = instrument.category_rates(nat, H)
    L1, _ = instrument.predicted_dollars(arr, gk, cy, ty, H, "roll",
                                        locked.SPAN, locked.LAG,
                                        locked.DATA_FIRST)
    P_lag1_full = piv["P"].reindex(columns=[t - H for t in ty]).to_numpy(np.float64)
    Z1_full = L1 / P_lag1_full
    zpos = {t: j for j, t in enumerate(ty)}

    fin = np.isfinite(Z1_full)
    fy = np.array(ty)[fin.any(0)]
    print(f"    z1 finite {int(fin.sum()):,} cells, years {fy.min()}-{fy.max()}, "
          f"counties {int(fin.any(1).sum())}")
    if int(fin.sum()) != Z1_FINITE:
        raise SystemExit(
            f"  FAIL: z1 finite count {int(fin.sum()):,} != {Z1_FINITE:,}. "
            f"The window rule lo = t-h-{locked.LAG + locked.SPAN - 1} >= "
            f"DATA_FIRST gives a first finite year of {Z1_FIRST}, so over "
            f"[{Z1_FIRST}, {SAMPLE_LAST}] the count is "
            f"{locked.N_COUNTIES:,} x {SAMPLE_LAST - Z1_FIRST + 1}; that "
            "formula reproduces every locked.N_COUNTY value at its own h. A "
            "mismatch means the constructor or the stores moved.")

    # SAMPLE_LAST drives every maximal window, so it is measured, not assumed.
    # The two sides are checked differently and deliberately so. z1 is BUILT on
    # store.target_years, so its last finite year must EQUAL SAMPLE_LAST and a
    # gap means the constructor and locked.TARGET_LAST have parted company. E is
    # a store this producer only reads, and the panel carries years past
    # locked.TARGET_LAST that the lock deliberately excludes (locked.py:61-76),
    # so the requirement on E is that it REACHES the end of the sample. E
    # running further is the expected state, not an error.
    ecols = piv["E"].columns
    e_last = int(max(c for c in ecols if piv["E"][c].notna().all()
                     and (piv["E"][c] > 0).all()))
    if int(fy.max()) != SAMPLE_LAST:
        raise SystemExit(
            f"  FAIL: SAMPLE_LAST is {SAMPLE_LAST} but z1's last finite target "
            f"year is {int(fy.max())}. Both derive from locked.TARGET_LAST = "
            f"{locked.TARGET_LAST}, so they cannot disagree unless the "
            "instrument constructor or the stores moved.")
    if e_last < SAMPLE_LAST:
        raise SystemExit(
            f"  FAIL: E is complete and strictly positive only to {e_last}, "
            f"short of SAMPLE_LAST = {SAMPLE_LAST}. Every maximal window is "
            "derived from SAMPLE_LAST; a post horizon would ask for an "
            "employment year the panel does not carry.")

    # The FRONT constraint, measured for the same reason SAMPLE_LAST is. At
    # depth p the deepest shock lag is z1_{c,t-p}, so the first target year a
    # window can start at is z1's first finite year plus p. At p = P_LAG that
    # reproduces YEAR_LO, and the assert below is what keeps the pre-trend
    # grid's windows tied to the headline's rather than to a second hard-coded
    # arithmetic that could drift away from it.
    z1_first = int(fy.min())
    if z1_first + P_LAG != YEAR_LO:
        raise SystemExit(
            f"  FAIL: z1's first finite target year is {z1_first}, so p = "
            f"{P_LAG} implies a front year of {z1_first + P_LAG}, but YEAR_LO "
            f"is {YEAR_LO}. The common window and the instrument have parted "
            "company; fix the constant before trusting any window.")

    # ---- the rung-5' sector share, read ONCE ---------------------------------
    # naics2_shares opens one parquet per base year and validates each, so it is
    # loaded here over every base year any window below can ask for rather than
    # inside build_window, which the pre-trend grid calls twelve times. The range
    # is the widest window's, [YEAR_LO, SAMPLE_LAST], at this producer's own base
    # offset t - H.
    sh = naics2_shares(range(YEAR_LO - H, SAMPLE_LAST - H + 1), units)

    return dict(units=units, piv=piv, DIVI=DIVI, cats=cats, cy=cy, arr=arr,
                gk=gk, Z1_full=Z1_full, zpos=zpos, z1_first=z1_first,
                sh=sh, C=store.load_con(locked.RULE).reindex(units))


def build_window(S, year_lo, year_hi, ykeys, p=P_LAG, weight="division-EW"):
    """The LP frame at h = 1 on one target window, for one set of horizons.

    ykeys is the (kind, hz) set whose outcomes are built AND whose finiteness
    the rectangle assert demands. It has to be a parameter: a window running to
    SAMPLE_LAST that built every outcome would ask for E past the end of the
    panel and fail, which is exactly the back constraint the maximal grid exists
    to respect. Every window used here reaches forward to SAMPLE_LAST at most,
    because each is its horizon's own SAMPLE_LAST - h. ("impact", 0) is
    always added -- it is defined on every window used here and validate_house
    needs it, so the per-window validation is never silently skipped.

    p and weight exist for --pretrend-grid and default to the headline's values,
    so every caller that predates them is unaffected. p sets how many lags of
    the shock and of the outcome are BUILT here; which of them a given horizon
    uses is control_names' business. weight is the county regression weight:

      division-EW  E_{c,t-1} over its division-year total, the headline
      equal        every county-year alike

    The weight is not merely a precision choice on the lead side. It enters the
    demeaning, the shock-level exposure weights s_kt = sum_l w_l s_lkt, and the
    FWL residualisation, so it is threaded from ONE place and never re-derived.
    """
    units, piv, DIVI = S["units"], S["piv"], S["DIVI"]
    cats, cy, arr, gk = S["cats"], S["cy"], S["arr"], S["gk"]
    Z1_full, zpos = S["Z1_full"], S["zpos"]
    sh = S["sh"]

    ykeys = sorted(set(ykeys) | {("impact", 0)})
    YEARS = list(range(year_lo, year_hi + 1))
    n_expect = locked.N_COUNTIES * len(YEARS)

    def zc(off):
        """z1 at year t - off, for t in YEARS."""
        return Z1_full[:, [zpos[t - off] for t in YEARS]]

    def ec(off):
        """E at year t - off (negative off reaches forward), for t in YEARS."""
        return piv["E"].reindex(columns=[t - off for t in YEARS]).to_numpy(np.float64)

    def pc(off):
        return piv["P"].reindex(columns=[t - off for t in YEARS]).to_numpy(np.float64)

    def yc(off):
        return piv["Y"].reindex(columns=[t - off for t in YEARS]).to_numpy(np.float64)

    def mpgr(E_t, E_b):
        """locked.OUTCOME, via the library, so the formula lives in one place."""
        return pnl_lib.outcomes(E_t, E_b)[2]

    Z1 = zc(0)
    P_b, E_b = pc(H), ec(H)

    # C-bar and the share-sum level: the predetermined exposure window.
    cbar = {}
    for j, t in enumerate(YEARS):
        hi = t - H - locked.LAG
        lo = hi - locked.SPAN + 1
        if lo < locked.DATA_FIRST:
            continue
        a, b = cy.index(lo), cy.index(hi)
        cbar[j] = arr[:, :, a:b + 1].mean(axis=2).astype(np.float64)
    S0 = np.full_like(Z1, np.nan)
    for j, m in cbar.items():
        S0[:, j] = m.sum(axis=1) / P_b[:, j]
    if len(cbar) != len(YEARS):
        raise SystemExit(f"  FAIL: exposure window missing for "
                         f"{len(YEARS) - len(cbar)} of {len(YEARS)} target years")

    # ---- outcomes, one per REQUESTED horizon -------------------------------
    def outcome(kind, hz):
        if kind == "lead":
            return mpgr(ec(1), ec(hz))                 # g_{c,t-k -> t-1}
        if kind == "impact":
            return mpgr(ec(0), ec(1))                  # g_{c,t-1 -> t}
        if kind == "post":
            return mpgr(ec(-hz), ec(0))                # g_{c,t -> t+h}
        if kind == "lead_literal":
            return mpgr(ec(0), ec(hz))                 # g_{c,t-k -> t}
        raise SystemExit(f"  FAIL: unknown horizon kind {kind!r}")

    Y = {key: outcome(*key) for key in ykeys}

    # The outcome lags run to max(p, N_INCREMENT): the increment route needs
    # gl1..gl3 as DEPENDENT variables whatever the control depth is, so at p < 3
    # they would otherwise be missing from the frame.
    n_gl = max(p, N_INCREMENT)
    ZL = {j: zc(j) for j in range(1, p + 1)}           # z1_{t-j}
    GL = {j: mpgr(ec(j), ec(j + 1)) for j in range(1, n_gl + 1)}  # g_{t-j-1 -> t-j}

    # The h = 1 treatment, for the relevance diagnostic ONLY (see first_stage).
    C = S["C"]
    Cc = C.reindex(columns=YEARS).to_numpy(np.float64)
    Cp = C.reindex(columns=[t - 1 for t in YEARS]).to_numpy(np.float64)
    X1 = (Cc - Cp) / P_b

    # ---- one finiteness mask, so every horizon runs on identical rows -------
    # S0 stays in the mask at BOTH rungs. Rung 0 does not use it as a control,
    # but it must run on identical rows or the rung comparison is confounded
    # with a sample change.
    ok = np.isfinite(Z1) & np.isfinite(S0) & np.isfinite(E_b) & np.isfinite(X1)
    for v in list(Y.values()) + list(ZL.values()) + list(GL.values()):
        ok &= np.isfinite(v)
    ri, ci = np.where(ok)
    n = len(ri)
    if n != n_expect:
        raise SystemExit(
            f"  FAIL: sample [{year_lo},{year_hi}] {n:,} != {n_expect:,} "
            f"({locked.N_COUNTIES} x {len(YEARS)}). Every window here must be "
            "the full rectangle; a shortfall means a county-year lost an "
            "outcome or a lag.")
    # ok.all() is the set-equality assert, not merely a count one: a full
    # rectangle over the FIXED `units` array is what makes the county set --
    # not just the county count -- identical in every year, which is the WS0
    # convention. The per-year count check is kept as the narrower diagnostic.
    if not ok.all():
        raise SystemExit(f"  FAIL: sample [{year_lo},{year_hi}] is not a full "
                         "rectangle; the county set differs across years")
    per_year = np.bincount(ci, minlength=len(YEARS))
    if not np.all(per_year == locked.N_COUNTIES):
        bad = [YEARS[j] for j in np.where(per_year != locked.N_COUNTIES)[0]]
        raise SystemExit(f"  FAIL: years off the {locked.N_COUNTIES}-county "
                         f"rectangle: {bad}")
    # The maximal lead window coincides with the locked h = 4 estimation panel.
    if (year_lo, year_hi) == (YEAR_LO, SAMPLE_LAST) and n != locked.N_COUNTY[4]:
        raise SystemExit(f"  FAIL: [{year_lo},{year_hi}] gives {n:,}, not "
                         f"locked.N_COUNTY[4] = {locked.N_COUNTY[4]:,}")

    d = pd.DataFrame(dict(ccg=units[ri], year=np.array(YEARS)[ci],
                          z1=Z1[ok], s0=S0[ok], E_b=E_b[ok], P_b=P_b[ok],
                          x1=X1[ok]))
    d["division"] = d["ccg"].map(DIVI)

    # ---- the rung-5' covariates, OFF the row mask ----------------------------
    # Built for every rung so the frame is identical at 0', 0 and 5' -- the same
    # reason S0 stays in `ok` above. They are deliberately NOT in `ok`: the
    # rectangle assert, the locked row counts and the byte-identical headline all
    # depend on that mask, and a covariate is not allowed to move the sample.
    #
    # Non-finite values are zero-filled BEFORE demeaning, which is what 22 does
    # to a control column. 22's covariates() drops instead, but only for the
    # BALANCE regression, where a fabricated zero would enter the test the
    # falsification turns on; here the covariate is a control and the convention
    # is 22's own.
    #
    # THE BASE YEAR IS THIS PRODUCER'S, t - H = t - 1. 22 measures the same five
    # covariates at its own t - 4. The definitions are identical and the base
    # follows each producer's horizon, like every other predetermined quantity
    # here (E_b, P_b, S0). The figure fragment says so.
    #
    # The five are 22's RUNG2_COVARIATES, and each is built the way 22's
    # covariates() builds it: Y_b/P_b, E_b/P_b, log(P_b) and the two EFSY NAICS2
    # shares at the base year. build_cov asserts the set matches, so adding a
    # name upstream cannot silently leave a control unbuilt here.
    Y_b = yc(H)
    base = d["year"].to_numpy() - H
    mg = pd.DataFrame(dict(ccg=d["ccg"].to_numpy(), base_year=base)).merge(
        sh, on=["ccg", "base_year"], how="left", validate="m:1")
    P_bv = d["P_b"].to_numpy(np.float64)
    cov_raw = {
        "income per head": Y_b[ok] / P_bv,
        "employment rate": d["E_b"].to_numpy(np.float64) / P_bv,
        "log population": np.log(P_bv),
        "Prof./technical services share":
            mg["Prof./technical services share"].to_numpy(np.float64),
        "Manufacturing share": mg["Manufacturing share"].to_numpy(np.float64),
    }
    unbuilt = [c for c in COVARIATES if c not in cov_raw]
    if unbuilt:
        raise SystemExit(
            f"  FAIL: rung {RUNG5} needs covariate(s) {unbuilt}, which this "
            "producer does not construct. 22's RUNG2_COVARIATES has moved; "
            "add the constructor here and the key to COV_KEY.")
    cov, cov_zero = {}, {}
    for c, v in cov_raw.items():
        bad = ~np.isfinite(v)
        cov_zero[c] = int(bad.sum())
        if bad.any():
            yrs = sorted(set(d["year"].to_numpy()[bad].tolist()))
            print(f"    NOTE: covariate '{c}' undefined for {int(bad.sum()):,} "
                  f"of {len(v):,} rows, years {yrs}; zero-filled as a control")
            if bad.mean() > COV_ZEROFILL_MAX:
                raise SystemExit(
                    f"  FAIL: covariate '{c}' is zero-filled on "
                    f"{bad.mean():.3%} of rows, over the "
                    f"{COV_ZEROFILL_MAX:.3%} tolerance. A control that is "
                    "mostly a constant is not a control; fix the store or drop "
                    f"the rung rather than reporting rung {RUNG5} on it.")
        cov[c] = np.where(bad, 0.0, v)

    if weight == "division-EW":
        tot = d.groupby(["division", "year"])["E_b"].transform("sum")
        w = (d["E_b"] / tot).to_numpy(np.float64)
    elif weight == "equal":
        w = np.ones(len(d), np.float64)
    else:
        raise SystemExit(f"  FAIL: unknown weight scheme {weight!r}; this "
                         f"producer implements {WEIGHTS}")
    w = w / w.mean()
    fe = pd.factorize(d["division"].astype(str) + "_"
                      + d["year"].astype(str))[0].astype(np.int64)
    nofe = np.zeros(len(d), np.int64)
    cl_county = pd.factorize(d["ccg"])[0].astype(np.int64)

    ncell = int(np.unique(fe).size)
    if ncell != 9 * len(YEARS):
        raise SystemExit(f"  FAIL: {ncell} division-year cells, expected "
                         f"{9 * len(YEARS)}")
    small = np.bincount(fe).min()
    print(f"    {ncell} division-year cells, smallest carries {small:,} counties")
    if small < 2:
        raise SystemExit("  FAIL: a degenerate division-year cell would be "
                         "absorbed entirely by the fixed effect")

    import pyfixest as pf
    op.w22.DEMEANER = pf.MapDemeaner(fixef_tol=1e-10, fixef_maxiter=100_000)

    # Everything the regressions touch is two-way demeaned ONCE, here.
    raw = {"z1": d["z1"].to_numpy(np.float64), "s0": d["s0"].to_numpy(np.float64),
           "x1": d["x1"].to_numpy(np.float64)}
    for j in ZL:
        raw[f"zl{j}"] = ZL[j][ok]
    for j in GL:
        raw[f"gl{j}"] = GL[j][ok]
    for key, v in Y.items():
        raw["y_" + key[0] + str(key[1])] = v[ok]
    for c, v in cov.items():
        raw[COV_KEY[c]] = v
    names = sorted(raw)
    M = op.w22.demean(np.column_stack([raw[k] for k in names]), fe, nofe, w)
    dm = {k: M[:, i] for i, k in enumerate(names)}

    rowpos = {u: i for i, u in enumerate(units)}
    colpos = {y: j for j, y in enumerate(YEARS)}
    return dict(d=d, units=units, cats=cats, cy=cy, years=YEARS, cbar=cbar,
                gk=gk, w=w, fe=fe, nofe=nofe, cl_county=cl_county,
                dm=dm, raw=raw, ykeys=list(Y), p=p, weight=weight,
                cov_zero=cov_zero,
                year_lo=year_lo, year_hi=year_hi, n=n,
                rr=d["ccg"].map(rowpos).to_numpy(),
                cc=d["year"].map(colpos).to_numpy(),
                pb=d["P_b"].to_numpy(np.float64))


def build_inputs():
    """The frame the headline runs on: the common window, every horizon."""
    return build_window(load_stores(), YEAR_LO, YEAR_HI,
                        [(k, h) for k, hzs in ALL_KINDS for h in hzs])


# ------------------------------------------------- shock-level aggregation --

def shock_frame(inp):
    """One row per live (k,t): the exposure weight s_kt and the raw shock g_kt.

    s_kt = sum_l w_l s_lkt is BHJ's shock importance weight. (k,t) cells with
    s_kt == 0 carry no county; they are dropped and the drop is logged, since it
    moves the cluster and effective-shock counts but no coefficient.
    """
    cats, cy, years = inp["cats"], inp["cy"], inp["years"]
    cbar, gk = inp["cbar"], inp["gk"]
    rr, cc, pb, w = inp["rr"], inp["cc"], inp["pb"], inp["w"]
    keys, sv, gv, blocks = [], [], [], {}
    for j in sorted(cbar):
        m = cc == j
        if not m.any():
            continue
        Sk = cbar[j][rr[m], :] / pb[m][:, None]
        wsk = Sk * w[m][:, None]
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
        print(f"    dropped {ndrop:,} of {len(sv):,} (k,t) cells with s_kt == 0")
    return dict(keys=keys[live], sv=sv[live], gv=gv[live], blocks=blocks)


def aggregate(sf, v):
    """num_n = sum_l w_l s_lnt v_l, for a COUNTY-RESIDUALIZED v.

    Aggregate the residuals and pair them with the RAW g. Never residualize g:
    sum_k g_k sum_l w_l s_lk v~_l = sum_l w_l z_l v~_l = sum_l w_l z~_l v~_l once
    v~ is orthogonal to the controls, and that last step is the whole identity.
    """
    per_year = {j: wsk.T @ v[m] for j, (m, wsk) in sf["blocks"].items()}
    return np.array([per_year[j][k] for k, j in sf["keys"]])


def fwl(Cm, w, *vs):
    """Residualize each v on the (already demeaned) control block, weighted.

    Thin SVD of the sqrt(w)-scaled block, NOT the normal equations. The identity
    needs the residual orthogonal to the controls to working precision, and at
    22's rung 3 the normal-equations route left it only approximately so and the
    two routes disagreed by 2.7e-01. The SVD also absorbs rank deficiency.
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


# ------------------------------------------------------------- the horizons --

def control_names(kind, rung="0'", p=P_LAG, flagged=None):
    """The control set for a horizon kind at one shock-control rung and depth p.

    Shock lags always; outcome lags not on the lead side, where they sit inside
    the dependent variable's window. The rung governs the share-sum term and, at
    rung 5', the covariate block, and nothing else: rung 0' adds the share-sum
    LEVEL s0, rung 0 has no share-sum term at all and is the previously reported
    specification (ws0_contingency.py:116-123). s0 keeps its slot, so at rung 0'
    the Route-A design matrix is column-for-column what it was before the rung
    became a parameter.

    Rung 5' is rung 0' plus one column per covariate in 22's RUNG2_COVARIATES,
    in the sorted order flagged_names() fixes. It sits on the PRIMED fork, so it
    never pays rung 1's first-stage collapse. `flagged` is that list; it is
    required at rung 5' and refused everywhere else, so a caller cannot smuggle
    covariates into rung 0' and have the output still say 0'.

    22's rung_spec is the general one. Only tiers 0, 0' and 5' are implemented
    here and every other label is a hard error, deliberately: rung_spec silently
    aliases "1'" to 0', which would mislabel an output, and rung 1 -- the
    share-sum x year interaction, the thing "share-sum FE" properly names --
    costs the first stage (module docstring, lines 62-67). Rung 3' (the
    exposure-weighted FSC group control) is not implemented. Note that 22's own
    rung 2' now carries exactly this covariate block: on the LP frame it is
    labelled 5', which is the label the record and PIPELINE.md already use.
    """
    if rung not in ("0'", "0", RUNG5):
        raise SystemExit(
            f"  FAIL: producer 50 implements rungs 0', 0 and {RUNG5} only, not "
            f"{rung!r}. Rung 1 interacts the share sum with year and "
            "22_bhj_assumption1.py records the cost: the first stage "
            f"falls 16.6 to 3.1. Rung {RUNG5} is rung 0' plus the "
            "predetermined covariates; rung 3' would be the FSC group control, "
            "which is not built here.")
    if (rung == RUNG5) != bool(flagged):
        raise SystemExit(
            f"  FAIL: rung {rung!r} was given flagged={flagged!r}. The "
            f"covariates belong to rung {RUNG5} and to no other rung; passing "
            "them at rung 0' would report a controlled specification under the "
            "baseline's label.")
    cols = [f"zl{j}" for j in range(1, p + 1)]
    if rung in ("0'", RUNG5):
        cols.append("s0")
    if rung == RUNG5:
        cols += [COV_KEY[c] for c in flagged]
    if kind in ("impact", "post"):
        cols += [f"gl{j}" for j in range(1, p + 1)]
    return cols


def horizon_fit(inp, sf, kind, hz, rung="0'", flagged=None):
    """beta at one horizon, by two independent routes, with both standard errors.

    Route A is the multi-regressor WLS: beta is the coefficient on z1 with every
    control estimated alongside. Route B partials the controls out by FWL and
    runs the just-identified reduced form on the residualized pair, which is what
    makes BHJ's shock-level equivalence apply and therefore what the
    exposure-robust standard error is computed on. They agree by construction;
    the assert is a check on this code, not on the algebra.
    """
    w, fe, nofe, dm = inp["w"], inp["fe"], inp["nofe"], inp["dm"]
    yk = "y_" + kind + str(hz)
    cn = control_names(kind, rung, inp["p"], flagged)

    # Route A: everything at once, county-clustered.
    XA = np.column_stack([dm["z1"]] + [dm[c] for c in cn])
    bA, seA, _ = wls_fe(dm[yk], XA, w, fe, nofe, inp["cl_county"])

    # Route B: FWL, then the just-identified reduced form.
    Cm = np.column_stack([dm[c] for c in cn])
    yr, zr = fwl(Cm, w, dm[yk], dm["z1"])
    zz = float((w * zr) @ zr)
    bB = float((w * zr) @ yr) / zz

    gap = abs(bB - bA[0]) / max(abs(bA[0]), 1e-300)
    if gap >= locked.RTOL_COEF:
        raise SystemExit(
            f"  FAIL: {kind} {hz}: the FWL route {bB * SCALE:.6f} does not "
            f"reproduce the multi-regressor coefficient {bA[0] * SCALE:.6f} "
            f"(rel gap {gap:.1e}). The exposure-robust standard error is "
            "computed on the FWL route, so it is not validated.")

    # The shock-level route on the SAME residuals, and its exposure-robust SE.
    ny, nz = aggregate(sf, yr), aggregate(sf, zr)
    g = sf["gv"]
    D = float(g @ nz)
    b_shock = float(g @ ny) / D
    sgap = abs(b_shock - bB) / max(abs(bB), 1e-300)
    if sgap >= locked.RTOL_COEF:
        raise SystemExit(
            f"  FAIL: {kind} {hz}: shock-level route {b_shock * SCALE:.6f} != "
            f"county route {bB * SCALE:.6f} (rel gap {sgap:.1e}); BHJ's "
            "equivalence is broken, so the exposure-robust SE is meaningless.")

    score = g * (ny - b_shock * nz)
    cl = cluster_index(sf, inp["cats"], CONTINGENCY["cluster_level"])
    ncl = int(cl.max()) + 1
    sc = cluster_sums(score, cl, ncl)
    G = int(np.unique(cl).size)
    se_exp = float(np.sqrt((G / max(G - 1, 1)) * float(sc @ sc)) / abs(D))

    return dict(kind=kind, hz=hz, beta=bB, se_exposure=se_exp,
                se_county=float(seA[0]), ncontrol=Cm.shape[1], n=len(yr),
                G_shock=G, equiv_gap=sgap, route_gap=gap,
                rung=rung, p=inp["p"], weight=inp["weight"],
                year_lo=inp["year_lo"], year_hi=inp["year_hi"],
                score=score, cl=cl, D=D, nz=nz, zr=zr)


def validate_house(inp):
    """The FWL route against the house estimator, at the one cell it can run.

    t28.fit2sls takes one regressor and has no covariate slot, so the only cell
    where both routes exist is the reduced form with NO controls. Passing z as x
    makes fit2sls the ordinary within estimator, which is exactly the LP at
    p = 0. If this disagrees, every controlled horizon above is suspect.
    """
    w, fe, nofe, dm = inp["w"], inp["fe"], inp["nofe"], inp["dm"]
    y, z = dm["y_impact0"], dm["z1"]
    b_fwl = float((w * z) @ y) / float((w * z) @ z)
    house = op.t28.fit2sls(inp["raw"]["y_impact0"], inp["raw"]["z1"],
                           inp["raw"]["z1"], w, fe, nofe, inp["cl_county"],
                           singleton=False)
    gap = abs(b_fwl - house["beta"]) / max(abs(house["beta"]), 1e-300)
    print(f"    p = 0 validation: FWL {b_fwl * SCALE:.4f} vs t28.fit2sls "
          f"{house['beta'] * SCALE:.4f}, rel gap {gap:.1e} "
          f"{'PASS' if gap < locked.RTOL_COEF else 'FAIL'}")
    if gap >= locked.RTOL_COEF:
        raise SystemExit("  FAIL: the reduced form does not reproduce the house "
                         "estimator at p = 0; the substitute is not validated")
    return b_fwl, float(house["se"])


def first_stage(inp, sf, rung="0'", flagged=None):
    """The exposure-robust first stage of the h = 1 treatment on z1.

    A RELEVANCE DIAGNOSTIC, NOT PART OF THE LP. The projections above are reduced
    forms, so there is no endogenous regressor and no first stage to be weak.
    This is reported because a one-year national dollar change is noisier than a
    2-4 year one and the +-1.0 clip binds differently at h = 1, so relevance at
    this horizon is an open question rather than something inherited.

    Read it with the caveat 13_build_county_panel.py:24-40 attaches to the h = 1
    treatment itself: the raw flow difference (C_t - C_{t-1})/P_{t-1} has
    second-order autocorrelation -0.39 against +0.64 for the cumulative object,
    and that critique is about this x, not about z1.
    """
    w, dm = inp["w"], inp["dm"]
    cn = [c for c in control_names("lead", rung, inp["p"], flagged)]
    Cm = np.column_stack([dm[c] for c in cn])
    xr, zr = fwl(Cm, w, dm["x1"], dm["z1"])
    nx, nz = aggregate(sf, xr), aggregate(sf, zr)
    g = sf["gv"]
    D = float(g @ nz)
    pi = float(g @ nx) / D
    cl = cluster_index(sf, inp["cats"], CONTINGENCY["cluster_level"])
    ncl = int(cl.max()) + 1
    sc = cluster_sums(g * (nx - pi * nz), cl, ncl)
    G = int(np.unique(cl).size)
    var = (G / max(G - 1, 1)) * float(sc @ sc) / (D * D)
    F = float(pi * pi / var) if var > 0 else np.nan
    return F, pi


def effective_df(F, k, G, G_eff):
    """The same Wald statistic read against the EFFECTIVE number of clusters.

    Two things in the test above are sized by G, the literal category count of
    800: the finite-cluster correction G/(G-1) applied to V, and the denominator
    degrees of freedom of the reference F. The same run measures 43.6 effective
    shifters at that same level, so both are spending independence the design
    does not have. This recomputes them at G_eff and returns the pair.

    G_eff IS NOT A CLUSTER COUNT. It is 1/HHI of the cluster-summed exposure
    weights, a continuous measure of concentration, and substituting it for a
    degrees-of-freedom parameter is an approximation with no exact justification.
    That is why nothing here overwrites F or p: the effective-df figures are
    reported ALONGSIDE the literal ones and the reader is told which is which.
    Treat the gap between them as the size of the question, not as a correction.

    Returns (F_eff, p_eff, df_eff). Rescaling V by c is equivalent to dividing
    the statistic by c, since F is linear in V^{-1}.
    """
    from scipy import stats
    if not np.isfinite(G_eff) or G_eff <= 1.0:
        return np.nan, np.nan, np.nan
    c = (G_eff / (G_eff - 1.0)) / (G / max(G - 1, 1))
    F_eff = F / c
    df_eff = max(int(round(G_eff)) - 1, 1)
    return float(F_eff), float(stats.f.sf(F_eff, k, df_eff)), df_eff


def joint_test(fits, kind, hzs, G_eff=np.nan):
    """Wald on one family of leads, exposure-robust and cross-horizon.

    Every horizon in a family shares one control set, hence one residualized z
    and one denominator D, so the shock-level scores live on the SAME (k,t)
    cells and the full covariance follows from stacking the cluster sums.
    Running the horizons separately and then ignoring the cross-horizon
    covariance would move the joint test arbitrarily; this does not.
    """
    from scipy import stats
    lead = [f for f in fits if f["kind"] == kind and f["hz"] in hzs]
    lead.sort(key=lambda f: f["hz"])
    b = np.array([f["beta"] for f in lead])
    cl, D = lead[0]["cl"], lead[0]["D"]
    for f in lead:
        if not np.array_equal(f["cl"], cl) or abs(f["D"] - D) > 0:
            # Also the guard against pooling two cells of the robustness grid:
            # a different rung or a different window gives a different D.
            raise SystemExit(
                f"  FAIL: the {kind}s do not share a residualized z; the "
                f"stacked covariance would be wrong. Offending fit: rung "
                f"{f.get('rung')}, window [{f.get('year_lo')},"
                f"{f.get('year_hi')}], horizon {f['hz']}")
    ncl = int(cl.max()) + 1
    S = np.column_stack([cluster_sums(f["score"], cl, ncl) for f in lead])
    G = int(np.unique(cl).size)
    V = (G / max(G - 1, 1)) * (S.T @ S) / (D * D)
    k = len(b)
    F = float(b @ np.linalg.solve(V, b) / k)
    p = float(stats.f.sf(F, k, max(G - 1, 1)))
    se = np.sqrt(np.maximum(np.diag(V), 0.0))
    ev = np.linalg.eigvalsh(V)
    F_eff, p_eff, df_eff = effective_df(F, k, G, G_eff)
    return dict(F=F, p=p, k=k, G=G, betas=b, ses=se, cl=cl, ncl=ncl,
                G_eff=G_eff, F_eff=F_eff, p_eff=p_eff, df_eff=df_eff,
                cond=float(ev.max() / max(ev.min(), 1e-300)),
                maxcorr=float(np.max(np.abs(
                    (V / np.outer(se, se))[np.triu_indices(k, 1)]))))


def increment_lead_test(inp, sf, cl, ncl, G, rung="0'", G_eff=np.nan,
                        flagged=None):
    """The lead test again on NON-NESTED one-year increments. THE CHECK THAT
    MATTERS for reading the joint rejection.

    The three reported leads are nested windows -- [t-2,t-1] inside [t-3,t-1]
    inside [t-4,t-1] -- so their estimates correlate up to 0.97 and a Wald test
    on a near-singular covariance can turn small differences into a large F.
    This re-runs the same test on the disjoint one-year increments
    g_{c,t-j-1 -> t-j}, j = 1,2,3, where the correlations fall to 0.58 and below.
    If the two agree, the rejection is a property of the data and not of the
    parameterization. They do agree, which is why both are reported.

    THE INCREMENT COUNT IS N_INCREMENT, NOT THE LAG DEPTH. This is a
    re-parameterisation of the same three-degree-of-freedom statement about the
    three reported leads, so it carries three increments at every depth. The
    control depth comes from the frame, exactly as horizon_fit takes it, so an
    increment can never condition on a lag the frame did not build.

    What it does NOT do, and cannot: the increments are windows inside the same
    pre-period, so they inherit whatever the shock-lag truncation leaves behind.
    Their agreement with the nested route rules out the near-singular-covariance
    artefact -- which is what it was built for -- and is silent about truncation,
    because both routes are truncated identically. That is what the p ladder in
    --pretrend-grid is for.
    """
    from scipy import stats
    w, dm = inp["w"], inp["dm"]
    Cm = np.column_stack([dm[c] for c in
                          control_names("lead", rung, inp["p"], flagged)])
    g = sf["gv"]
    b, cols, D0 = [], [], None
    for j in range(1, N_INCREMENT + 1):
        yr, zr = fwl(Cm, w, dm[f"gl{j}"], dm["z1"])
        ny, nz = aggregate(sf, yr), aggregate(sf, zr)
        D = float(g @ nz)
        D0 = D if D0 is None else D0
        bj = float(g @ ny) / D
        b.append(bj)
        cols.append(cluster_sums(g * (ny - bj * nz), cl, ncl))
    b = np.array(b)
    V = (G / max(G - 1, 1)) * (np.column_stack(cols).T @ np.column_stack(cols)) \
        / (D0 * D0)
    se = np.sqrt(np.maximum(np.diag(V), 0.0))
    k = len(b)
    F = float(b @ np.linalg.solve(V, b) / k)
    ev = np.linalg.eigvalsh(V)
    F_eff, p_eff, df_eff = effective_df(F, k, G, G_eff)
    return dict(F=F, p=float(stats.f.sf(F, k, max(G - 1, 1))), betas=b, ses=se,
                k=k, G=G, G_eff=G_eff, F_eff=F_eff, p_eff=p_eff, df_eff=df_eff,
                cond=float(ev.max() / max(ev.min(), 1e-300)),
                maxcorr=float(np.max(np.abs(
                    (V / np.outer(se, se))[np.triu_indices(k, 1)]))))


# ------------------------------------------------------- the robustness grid --

def run_cell(S, year_lo, year_hi, fitspec, rung, sample, label,
             p=P_LAG, weight="division-EW", flagged=None):
    """One (window, rung, horizon set) cell of the grid, with every check.

    Nothing here is a lighter version of the headline path: validate_house runs
    on the window, horizon_fit's FWL and BHJ shock-level identities run on every
    fit, and where the lead family is present the increment cross-check and the
    two-parameterisation agreement run too. A cell that cannot support a check
    SKIPS it explicitly rather than reporting a number that was never validated.
    That is what lets --pretrend-grid reuse this function unchanged: every cell
    of that grid is validated exactly as hard as the headline is.
    """
    kinds = [(k, tuple(hzs)) for k, hzs in fitspec]
    ykeys = [(k, h) for k, hzs in kinds for h in hzs]
    print(f"\n  -- {label}: t in [{year_lo}, {year_hi}], rung {rung}, "
          f"p = {p}, weight {weight}")
    inp = build_window(S, year_lo, year_hi, ykeys, p=p, weight=weight)
    print(f"    n = {inp['n']:,} county-years "
          f"({locked.N_COUNTIES:,} x {year_hi - year_lo + 1})")
    sf = shock_frame(inp)
    print(f"    {len(sf['gv']):,} live (k,t) shock cells")
    eff_all, eff_below = effective_counts(sf, inp["cats"])

    validate_house(inp)

    fsF, pi = first_stage(inp, sf, rung, flagged)
    print(f"    relevance (NOT part of the LP): exposure-robust first stage of "
          f"the h = 1 treatment on z1, F = {fsF:.2f}")
    if fsF < WEAK_F:
        print("      below 10; the projections remain reduced forms and are "
              "read as such, but note it in any 2SLS reading")

    fits = []
    for kind, hzs in kinds:
        for hz in hzs:
            f = horizon_fit(inp, sf, kind, hz, rung, flagged)
            fits.append(f)
            tag = "        " if kind == "lead_literal" else "    "
            print(f"{tag}{kind:13s} {hz}  beta {f['beta'] * SCALE:8.4f}  "
                  f"se_exp {f['se_exposure'] * SCALE:7.4f}  "
                  f"se_cty {f['se_county'] * SCALE:7.4f}  "
                  f"t {f['beta'] / f['se_exposure']:+6.2f}")

    have = {(f["kind"], f["hz"]) for f in fits}
    jl = jlit = il = None
    if {("lead", k) for k in (2, 3, 4)} <= have:
        jl = joint_test(fits, "lead", (2, 3, 4), eff_all["category"])
        print(f"    joint lead test (k = 2,3,4, full cross-horizon covariance): "
              f"F = {jl['F']:.3f}, p = {jl['p']:.4f} against F(3, {jl['G'] - 1})")
        print(f"      read against the EFFECTIVE {jl['G_eff']:.1f} shifters "
              f"instead of the literal {jl['G']}: F = {jl['F_eff']:.3f}, "
              f"p = {jl['p_eff']:.4f} against F(3, {jl['df_eff']})")
        print(f"      the three leads are NESTED windows: max |corr| "
              f"{jl['maxcorr']:.3f}, cond(V) {jl['cond']:.1f}")
        il = increment_lead_test(inp, sf, jl["cl"], jl["ncl"], jl["G"], rung,
                                 eff_all["category"], flagged)
        print(f"    same test on DISJOINT one-year increments: F = {il['F']:.3f}"
              f", p = {il['p']:.4f}, max |corr| {il['maxcorr']:.3f}")
        print("      increments x 1e5: "
              + "  ".join(f"{v * SCALE:+.4f}" for v in il["betas"])
              + "   t: " + "  ".join(f"{v:+.2f}"
                                     for v in il["betas"] / il["ses"]))
        # j = 1's window [t-2,t-1] IS lead k = 2's window, so the two routes
        # must agree. Free check on every cell, not just the headline.
        lead2 = next(f["beta"] for f in fits
                     if f["kind"] == "lead" and f["hz"] == 2)
        gap2 = abs(il["betas"][0] - lead2) / max(abs(lead2), 1e-300)
        print(f"      increment j = 1 vs lead k = 2 (the same window): rel gap "
              f"{gap2:.1e} {'PASS' if gap2 < locked.RTOL_COEF else 'FAIL'}")
        if gap2 >= locked.RTOL_COEF:
            raise SystemExit("  FAIL: the increment at j = 1 and the lead at "
                             "k = 2 run on the identical window [t-2,t-1] and "
                             "must agree; a lag is shifted somewhere")
        agree = abs(jl["p"] - il["p"]) < 0.02
        print(f"      the two parameterizations "
              f"{'AGREE' if agree else 'DISAGREE'}; "
              + ("the rejection is a property of the data, not of the nesting"
                 if agree else
                 "READ NEITHER until the discrepancy is explained"))
    if {("lead_literal", k) for k in (2, 3, 4)} <= have:
        jlit = joint_test(fits, "lead_literal", (2, 3, 4), eff_all["category"])
        print(f"    joint lead test, OVERLAPPING windows (plan-literal, "
              f"k = 2,3,4): F = {jlit['F']:.3f}, p = {jlit['p']:.4f}, "
              f"max |corr| {jlit['maxcorr']:.3f}")

    return dict(sample=sample, label=label, rung=rung, year_lo=year_lo,
                year_hi=year_hi, n=inp["n"], fits=fits, jl=jl, jlit=jlit,
                il=il, fsF=fsF, pi=pi, eff_all=eff_all, eff_below=eff_below,
                p=p, weight=weight, flagged=flagged,
                cov_zero=inp["cov_zero"], ncontrol=fits[0]["ncontrol"])


# ----------------------------------------------------------------- writers --

XPOS = {("lead", 4): -4.0, ("lead", 3): -3.0, ("lead", 2): -2.0,
        ("impact", 0): 0.0,
        ("post", 1): 1.0, ("post", 2): 2.0, ("post", 3): 3.0, ("post", 4): 4.0}

# The overlapping leads are plotted at the same DEPTH k, nudged right so the two
# series do not overplot. Their windows END at t rather than at t-1, so the
# position is the lead depth and not the window -- said in the note, because a
# reader would otherwise assume the x axis means the same thing for both series.
XPOS_LIT = {k: -float(k) + 0.12 for k in LEADS_LITERAL}


def _fig_open():
    """The shared figure frame: backend, rcParams, axes, shock strip, zero line.

    Extracted so the rung-0 companion cannot drift in style from the headline.
    Deliberately the ONLY thing shared -- each figure keeps its own series and
    its own note, so nothing that writes Figure_lp_irf can be changed by an
    edit aimed at the companion.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 8.5,
        "axes.linewidth": 0.6,
        "pdf.fonttype": 42,
        "savefig.bbox": "tight",
    })
    fig, ax = plt.subplots(1, 1, figsize=(5.2, 3.2))
    ax.axvspan(-1.0, 0.0, color="0.88", lw=0, zorder=0)
    ax.axhline(0.0, color="0.6", lw=0.5, zorder=2)
    return plt, fig, ax


def _fig_close(plt, fig, ax, pdf):
    """Ticks, labels, legend, spines, and the deterministic save."""
    ax.set_xticks([-4, -3, -2, -1, 0, 1, 2, 3, 4])
    ax.set_xticklabels(["$-4$", "$-3$", "$-2$", "$-1$", "0", "$+1$", "$+2$",
                        "$+3$", "$+4$"], fontsize=7.5)
    ax.legend(frameon=False, fontsize=6.8, loc="upper left")
    ax.set_xlabel("horizon, years relative to the shock year (shaded)",
                  fontsize=8.0)
    ax.set_ylabel(r"$\beta \times 10^{5}$, employment growth on $z1$")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=2.5, width=0.6)
    fig.tight_layout()
    fig.savefig(pdf, metadata={"CreationDate": None})
    plt.close(fig)
    print(f"  wrote {pdf}")


def write_figure(fits, jl, jlit, il, fsF, eff, out):
    """Figure_lp_irf.pdf plus its .tex fragment.

    Style follows the macro pipeline's only other figure, figure_a1 in
    step_3/code/py/58_export_aggregation_ladder.py:1762, which declares itself
    the convention-setter: Agg backend, no style sheet, serif to sit beside the
    tables, and a PDF plus a fragment so a figure enters a document exactly the
    way a table does.

    Two departures, both deliberate. The x axis carries leads and posts in one
    frame with the shock year shaded, and there is no -1 point because the
    non-overlapping lead at k = 1 is a zero-length window. And CreationDate is
    pinned to None: step_3/README.md:1882 records that matplotlib's timestamp
    makes figure bytes change every run so a digest gate is impossible. That fix
    changes no pixel but alters committed bytes, which is why it was a decision
    there; here there are no committed bytes to protect, so it is free.
    """
    pts = sorted(((XPOS[(f["kind"], f["hz"])], f) for f in fits
                  if (f["kind"], f["hz"]) in XPOS), key=lambda t: t[0])
    x = np.array([p for p, _ in pts])
    b = np.array([f["beta"] for _, f in pts]) * SCALE
    se = np.array([f["se_exposure"] for _, f in pts]) * SCALE

    plt, fig, ax = _fig_open()
    # Leads and posts are one series but not one continuous path: the gap at
    # x = -1 is the missing horizon, not a value, so the segments are drawn
    # separately rather than joined across it.
    for seg in ([p for p in x if p < 0], [p for p in x if p >= 0]):
        m = np.isin(x, seg)
        ax.plot(x[m], b[m], color="#1f3b73", ls="-", lw=1.1, zorder=3)
    ax.fill_between(x[x < 0], (b - Z95 * se)[x < 0], (b + Z95 * se)[x < 0],
                    color="#1f3b73", alpha=0.11, lw=0, zorder=1)
    ax.fill_between(x[x >= 0], (b - Z95 * se)[x >= 0], (b + Z95 * se)[x >= 0],
                    color="#1f3b73", alpha=0.11, lw=0, zorder=1)
    ax.scatter(x, b, s=17, color="#1f3b73", marker="o", zorder=4,
               label="non-overlapping leads (headline)")

    # The overlapping plan-literal leads, as a comparison series. Error bars
    # rather than a second filled band: these standard errors are up to three
    # times the headline's, so a band would swamp the one the reader is meant to
    # read, and band-on-band shading stops being legible in greyscale.
    lit = sorted((f for f in fits if f["kind"] == "lead_literal"),
                 key=lambda f: -f["hz"])
    xl = np.array([XPOS_LIT[f["hz"]] for f in lit])
    bl = np.array([f["beta"] for f in lit]) * SCALE
    sl = np.array([f["se_exposure"] for f in lit]) * SCALE
    ax.errorbar(xl, bl, yerr=Z95 * sl, fmt="none", ecolor="#b4472f",
                elinewidth=0.7, capsize=1.6, capthick=0.7, zorder=3)
    ax.plot(xl, bl, color="#b4472f", ls="--", lw=1.0, zorder=3)
    ax.scatter(xl, bl, s=15, color="#b4472f", marker="s", zorder=4,
               label="overlapping leads (run through the shock year)")

    pdf = out.with_suffix(".pdf")
    _fig_close(plt, fig, ax, pdf)

    pj = (r"$<0.001$" if jl["p"] < 0.001 else f"{jl['p']:.3f}")
    pi_txt = (r"$<0.001$" if il["p"] < 0.001 else f"{il['p']:.3f}")
    lead_t = [f["beta"] / f["se_exposure"] for f in fits if f["kind"] == "lead"]
    maxt = max(lead_t, key=abs)
    incr_txt = ", ".join(
        rf"${v * SCALE:+.3f}$ at $j={j}$"
        for j, v in zip(range(1, N_INCREMENT + 1), il["betas"]))
    plit = (r"$<0.001$" if jlit["p"] < 0.001 else f"{jlit['p']:.3f}")
    dlit = next(f["beta"] for f in fits
                if f["kind"] == "lead_literal" and f["hz"] == 1) * SCALE
    dimp = next(f["beta"] for f in fits if f["kind"] == "impact") * SCALE
    _bl = {f["hz"]: f["beta"] for f in fits if f["kind"] == "lead_literal"}
    _bn = {f["hz"]: f["beta"] for f in fits if f["kind"] == "lead"}
    _dif = [(_bl[k] - _bn[k]) * SCALE for k in (2, 3, 4)]
    dgap, dspread = float(np.mean(_dif)), float(max(_dif) - min(_dif))
    frag = rf"""% Produced by step_1/code/pipeline/50_lp_dynamics.py from
% Results/lp/lp_dynamics.csv. Do not edit.
\begin{{minipage}}{{\textwidth}}\centering
\textbf{{Figure. Lead and lag dynamics of the composition instrument}}\\[0.2em]
{{\small County midpoint employment growth on the one-year composition
instrument, by horizon}}\\[0.6em]
\includegraphics[width=\textwidth]{{{pdf.name}}}
\end{{minipage}}
\begin{{minipage}}{{\textwidth}}\vspace{{0.4em}}\footnotesize
\textbf{{What is plotted.}} Each point is $\beta$ from a separate reduced-form
regression of county midpoint employment growth over one window on the one-year
composition instrument $z1_{{ct}}$, scaled by $10^{{5}}$. The shaded band is the
95\% interval from the exposure-robust standard error, computed on the equivalent
shock-level regression (Borusyak, Hull and Jaravel 2022, Section 5.1) and
clustered on the FSC category. The horizontal line is zero, not a null of
interest. The shaded vertical strip is the shock year $[t-1,t]$: $z1_{{ct}}$ is
the one-year national change dated $t$.
\smallskip
\textbf{{The horizons are not symmetric, by construction.}} Negative horizons are
lead placebos on windows lying strictly before the shock year,
$g_{{c,t-k \to t-1}}$ for $k=2,3,4$. Horizon $0$ is the shock year itself,
$g_{{c,t-1 \to t}}$. Positive horizons are $g_{{c,t \to t+h}}$. There is no
horizon $-1$ in the headline series: the non-overlapping lead at $k=1$ is a
zero-length window.
\smallskip
\textbf{{The second series.}} The square markers are the overlapping leads
$g_{{c,t-k \to t}}$, $k=1,\dots,4$, the seeding plan's definition. Each is
plotted at its lead DEPTH $k$, nudged right to avoid overplotting; its window
ends at $t$, not at $t-1$, so for this series the $x$ position is the depth and
not the window. They are shown with 95\% error bars rather than a filled band,
because their standard errors run up to three times the headline's and a second
band would swamp the first. Their $k=1$ point has the same dependent variable as
the shock-year horizon and differs from it only in carrying no outcome lags
({dlit:+.4f} against {dimp:+.4f}), which is one way to see what those lags do.
\smallskip
\textbf{{Lead test, and what it rejects.}} The joint Wald test on the three lead
coefficients, using the full cross-horizon exposure-robust covariance, is
$F={jl['F']:.3f}$ with $p={pj}$ against $F(3,{jl['G'] - 1})$: the leads
\emph{{jointly reject}}. No individual lead does, the largest being
$t={maxt:+.2f}$. So the rejection is not a monotone pre-trend but a CONTRAST
across pre-shock years. On the disjoint one-year increments
$g_{{c,t-j-1 \to t-j}}$ the pattern is {incr_txt} --- alternating rather than
trending.
\smallskip
\textbf{{Both definitions of the lead, tested, and why they agree.}} On the
overlapping windows at the same depths $k=2,3,4$ the joint test gives
$F={jlit['F']:.3f}$ with $p={plit}$ --- all but identical to the headline's
${jl['F']:.3f}$. That is not a coincidence and not a redundancy. The overlapping
lead at each depth exceeds the non-overlapping one by a CONSTANT
{dgap:+.4f} (the spread across the three depths is {dspread:.5f}), and that
constant is the shock-year coefficient itself, {dlit:+.4f} on the same control
set: an overlapping window is a non-overlapping window plus the shock year, and
the shock year is common to all three depths. Since the rejection loads on the
CONTRAST across depths rather than on their common level, a common shift leaves
it untouched. So the finding survives either definition of the lead --- which is
a stronger statement than either test alone makes. The headline series remains
the one to read, because only its windows exclude the shock year and can be
called pre-trends at all.
\smallskip
\textbf{{The rejection is not an artefact of the nesting.}} The three reported
leads are nested windows, so their estimates correlate up to
${jl['maxcorr']:.2f}$, and a Wald test on a near-singular covariance can
manufacture a large $F$. It does not here: the condition number of the
covariance is ${jl['cond']:.0f}$, and the same test on the disjoint increments,
whose correlations reach only ${il['maxcorr']:.2f}$, gives $F={il['F']:.3f}$
with $p={pi_txt}$. Two parameterisations, one nested and one disjoint, agree.
\smallskip
\textbf{{Relation to the pre-trend already on record.}} Panel~E of the BHJ
balance table reports $t=0.12$ for the exposure-weighted pre-trend and does not
reject. That is one scalar at $h=4$ on an eight-year-back window, run as $z$ on
pre-period growth with county clusters; this is a three-horizon joint test at
$h=1$ with exposure-robust inference, and a joint test has power a scalar does
not. The two are not in contradiction, but they are not the same test either,
and this figure does not supersede that row.
\smallskip
\textbf{{Specification.}} Lag augmentation with $p=3$: three lags of the shock
enter every horizon, and three lags of the outcome enter the shock-year and
post-shock horizons only. On the lead side they cannot, because at $p=3$ the
first outcome lag \emph{{is}} the $k=2$ lead's dependent variable and for every
$k\ge2$ the outcome lags lie inside the lead window. Lag augmentation follows
Montiel Olea and Plagborg-M\o ller (2021); $p$ is set conservatively because
$z1$'s serial order is not identifiable on 32 years of national variation, not
because an order of 2 was measured. Note also that short-panel local projections
carry a small-sample bias (Herbst and Johannsen 2024). Division-by-year fixed
effects, division-year employment weights normalised to mean one, and the
share-sum level control of rung $0'$ throughout; no share-sum $\times$ year
interaction. Every horizon runs on the identical common sample of
{fits[0]['n']:,} county-years, $t \in [{YEAR_LO},{YEAR_HI}]$, all
{locked.N_COUNTIES:,} counties in every year.
\smallskip
\textbf{{Relevance.}} These are reduced forms, so no first stage can be weak and
no marker on this figure encodes one. As a separate diagnostic, the
exposure-robust first stage of the $h=1$ treatment on $z1$ is
$F={fsF:.2f}$, and the effective number of shifters is {eff:.1f} against the
frozen floor of {CONTINGENCY['effective_cluster_floor']}.
\end{{minipage}}
"""
    tex = out.with_suffix(".tex")
    tex.write_text(frag, encoding="utf-8")
    print(f"  wrote {tex}")


def write_figure_rungs(cp, c0, out):
    """Figure_lp_irf1.pdf plus its fragment: the same IRF at rungs 0' and 0.

    A SEPARATE figure, not a variant of the headline. Figure_lp_irf stays the
    exhibit; this one answers a single question -- does the share-sum level
    control drive the lead result -- and says in its own note that rung 0 is not
    an admissible rung.

    Only the headline non-overlapping series is plotted, at both rungs. The
    overlapping leads are dropped here on purpose: four series on nine horizons
    stops being readable, and the overlap comparison already has its own place
    in the headline figure and in lp_leads_sample_sensitivity.csv.
    """
    def series(cell):
        pts = sorted(((XPOS[(f["kind"], f["hz"])], f) for f in cell["fits"]
                      if (f["kind"], f["hz"]) in XPOS), key=lambda t: t[0])
        return (np.array([p for p, _ in pts]),
                np.array([f["beta"] for _, f in pts]) * SCALE,
                np.array([f["se_exposure"] for _, f in pts]) * SCALE)

    x, b, se = series(cp)
    x0, b0, se0 = series(c0)
    # Nudged right by the same convention the headline uses for its second
    # series, so the two rungs at one horizon do not overplot.
    x0 = x0 + 0.14

    plt, fig, ax = _fig_open()
    for seg in ([p for p in x if p < 0], [p for p in x if p >= 0]):
        m = np.isin(x, seg)
        ax.plot(x[m], b[m], color="#1f3b73", ls="-", lw=1.1, zorder=3)
    ax.fill_between(x[x < 0], (b - Z95 * se)[x < 0], (b + Z95 * se)[x < 0],
                    color="#1f3b73", alpha=0.11, lw=0, zorder=1)
    ax.fill_between(x[x >= 0], (b - Z95 * se)[x >= 0], (b + Z95 * se)[x >= 0],
                    color="#1f3b73", alpha=0.11, lw=0, zorder=1)
    ax.scatter(x, b, s=17, color="#1f3b73", marker="o", zorder=4,
               label=r"rung $0'$: share-sum level control (reported baseline)")

    # Rung 0 gets error bars rather than a second band, for the same reason the
    # headline's second series does: band-on-band shading is not legible.
    for seg in ([p for p in x0 if p < 0], [p for p in x0 if p >= 0]):
        m = np.isin(x0, seg)
        ax.plot(x0[m], b0[m], color="#7a4fa3", ls="--", lw=1.0, zorder=3)
    ax.errorbar(x0, b0, yerr=Z95 * se0, fmt="none", ecolor="#7a4fa3",
                elinewidth=0.7, capsize=1.6, capthick=0.7, zorder=3)
    ax.scatter(x0, b0, s=16, color="#7a4fa3", marker="^", zorder=4,
               label="rung 0: no share-sum control (not admissible; "
                     "the previously reported spec)")

    pdf = out.with_suffix(".pdf")
    _fig_close(plt, fig, ax, pdf)

    def ptxt(p):
        return r"$<0.001$" if p < 0.001 else f"{p:.4f}"

    dl = {f["hz"]: f for f in cp["fits"] if f["kind"] == "lead"}
    d0 = {f["hz"]: f for f in c0["fits"] if f["kind"] == "lead"}
    lead_tbl = "; ".join(
        rf"$k={k}$: ${dl[k]['beta'] * SCALE:+.4f}$ "
        rf"$\to$ ${d0[k]['beta'] * SCALE:+.4f}$" for k in (2, 3, 4))
    p1 = {f["hz"]: f for f in cp["fits"] if f["kind"] == "post"}
    p0 = {f["hz"]: f for f in c0["fits"] if f["kind"] == "post"}
    post_tbl = "; ".join(
        rf"$h={h}$: ${p1[h]['beta'] * SCALE:+.4f}$ "
        rf"$\to$ ${p0[h]['beta'] * SCALE:+.4f}$" for h in POSTS)
    frag = rf"""% Produced by step_1/code/pipeline/50_lp_dynamics.py from
% Results/lp/lp_dynamics.csv, rows sample = common. Do not edit.
\begin{{minipage}}{{\textwidth}}\centering
\textbf{{Figure. The lead and lag dynamics with and without the share-sum
control}}\\[0.2em]
{{\small The same local projection at shock-control rung $0'$ and at rung 0,
common sample $t \in [{cp['year_lo']},{cp['year_hi']}]$}}\\[0.6em]
\includegraphics[width=\textwidth]{{{pdf.name}}}
\end{{minipage}}
\begin{{minipage}}{{\textwidth}}\vspace{{0.4em}}\footnotesize
\textbf{{What is plotted, and what is NOT.}} Both series are the headline
non-overlapping horizons of the lead-lag figure: lead placebos
$g_{{c,t-k \to t-1}}$ at $k=2,3,4$, the shock year $g_{{c,t-1 \to t}}$ at
horizon $0$, and $g_{{c,t \to t+h}}$ at $h=1,\dots,4$, each $\beta$ scaled by
$10^{{5}}$. The circles and the shaded band are rung $0'$ --- identical to the
headline figure. The triangles, nudged right to avoid overplotting, are rung 0,
with 95\% error bars rather than a second band. The overlapping plan-literal
leads are omitted here; they are in the headline figure and in
\texttt{{lp\_leads\_sample\_sensitivity.csv}}.
\smallskip
\textbf{{The two rungs differ by exactly one regressor.}} Rung $0'$ conditions
on the share-sum LEVEL $S0_{{ct}}=\sum_k \bar s_{{ckt}}$, one common slope; rung
0 drops it. Everything else is held fixed: the same outcome, the same instrument
$z1_{{ct}}$, the same lag augmentation $p={P_LAG}$ (three shock lags on every
horizon, three outcome lags on the shock-year and post horizons only), the same
division-by-year fixed effects, the same division-year employment weights, the
same exposure-robust inference clustered on the FSC category, and the same
{cp['n']:,} county-years. This is a CONTROL change and not a change of estimand
(\texttt{{ws0\_contingency.py}}:88--95); no coefficient here answers a different
question from the one above it.
\smallskip
\textbf{{Rung 0 is not an admissible rung.}} The frozen contingency rule sets
\texttt{{admissible\_rungs}} to $(0',1,2,3,4)$ with $0'$ the reference
(\texttt{{ws0\_contingency.py}}:34--35, Revision~1). Rung 0 is on record as ``the
PREVIOUSLY reported spec'' and is shown here as a diagnostic only. Dropping
$S0_{{ct}}$ removes the control for the incomplete-shares problem that
Borusyak, Hull and Jaravel's Proposition~4 addresses, and the shock-level table
measures what that is worth at the locked baseline: $3.936$ at rung 0 against
$3.225$ at rung $0'$. Rung 0 is a LESS controlled specification, not a cleaner
one, and nothing in this figure licenses reporting it as the baseline.
\smallskip
\textbf{{What moves.}} The leads go {lead_tbl}. The joint Wald test on the three
leads, with the full cross-horizon exposure-robust covariance, is
$F={cp['jl']['F']:.3f}$ with $p={ptxt(cp['jl']['p'])}$ at rung $0'$ and
$F={c0['jl']['F']:.3f}$ with $p={ptxt(c0['jl']['p'])}$ at rung 0, both against
$F(3,{cp['jl']['G'] - 1})$. On the disjoint one-year increments the same test
gives $p={ptxt(cp['il']['p'])}$ and $p={ptxt(c0['il']['p'])}$. The post horizons
go {post_tbl}.
\smallskip
\textbf{{Relevance.}} These are reduced forms, so no first stage can be weak and
no marker encodes one. As a separate diagnostic, the exposure-robust first stage
of the $h=1$ treatment on $z1$ is $F={cp['fsF']:.2f}$ at rung $0'$ and
$F={c0['fsF']:.2f}$ at rung 0; the difference is what partialling the share-sum
level out of $z1$ costs. The effective number of shifters is
{cp['eff_all']['category']:.1f} against the frozen floor of
{CONTINGENCY['effective_cluster_floor']}, unchanged by the rung.
\end{{minipage}}
"""
    tex = out.with_suffix(".tex")
    tex.write_text(frag, encoding="utf-8")
    print(f"  wrote {tex}")


def write_figure_flagged(cp, c5, out):
    """Figure_lp_irf5.pdf plus its fragment: the same IRF at rungs 0' and 5'.

    A THIRD figure, on the pattern write_figure_rungs sets and for the same
    stated reason: each figure keeps its own series and its own note, so nothing
    that writes Figure_lp_irf or Figure_lp_irf1 can be changed by an edit aimed
    at this one. Only _fig_open and _fig_close are shared.

    The question is narrower than the rung-0 companion's. Rung 0 asks whether
    the share-sum control drives the lead result by REMOVING a control. This asks
    the opposite: does the pre-trend survive ADDING controls for the two county
    characteristics that actually fail the balance test?
    """
    def series(cell):
        pts = sorted(((XPOS[(f["kind"], f["hz"])], f) for f in cell["fits"]
                      if (f["kind"], f["hz"]) in XPOS), key=lambda t: t[0])
        return (np.array([p for p, _ in pts]),
                np.array([f["beta"] for _, f in pts]) * SCALE,
                np.array([f["se_exposure"] for _, f in pts]) * SCALE)

    x, b, se = series(cp)
    x5, b5, se5 = series(c5)
    x5 = x5 + 0.14

    plt, fig, ax = _fig_open()
    for seg in ([p for p in x if p < 0], [p for p in x if p >= 0]):
        m = np.isin(x, seg)
        ax.plot(x[m], b[m], color="#1f3b73", ls="-", lw=1.1, zorder=3)
    ax.fill_between(x[x < 0], (b - Z95 * se)[x < 0], (b + Z95 * se)[x < 0],
                    color="#1f3b73", alpha=0.11, lw=0, zorder=1)
    ax.fill_between(x[x >= 0], (b - Z95 * se)[x >= 0], (b + Z95 * se)[x >= 0],
                    color="#1f3b73", alpha=0.11, lw=0, zorder=1)
    ax.scatter(x, b, s=17, color="#1f3b73", marker="o", zorder=4,
               label=r"rung $0'$: share-sum level control (reported baseline)")

    for seg in ([p for p in x5 if p < 0], [p for p in x5 if p >= 0]):
        m = np.isin(x5, seg)
        ax.plot(x5[m], b5[m], color="#2f7a5a", ls="--", lw=1.0, zorder=3)
    ax.errorbar(x5, b5, yerr=Z95 * se5, fmt="none", ecolor="#2f7a5a",
                elinewidth=0.7, capsize=1.6, capthick=0.7, zorder=3)
    ax.scatter(x5, b5, s=16, color="#2f7a5a", marker="D", zorder=4,
               label=r"rung $5'$: $+$ the predetermined covariates")

    pdf = out.with_suffix(".pdf")
    _fig_close(plt, fig, ax, pdf)

    def ptxt(p):
        return r"$<0.001$" if p < 0.001 else f"{p:.4f}"

    def esc(s):
        return s.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")

    dl = {f["hz"]: f for f in cp["fits"] if f["kind"] == "lead"}
    d5 = {f["hz"]: f for f in c5["fits"] if f["kind"] == "lead"}
    lead_tbl = "; ".join(
        rf"$k={k}$: ${dl[k]['beta'] * SCALE:+.4f}$ "
        rf"$\to$ ${d5[k]['beta'] * SCALE:+.4f}$" for k in (2, 3, 4))
    p1 = {f["hz"]: f for f in cp["fits"] if f["kind"] == "post"}
    p5 = {f["hz"]: f for f in c5["fits"] if f["kind"] == "post"}
    post_tbl = "; ".join(
        rf"$h={h}$: ${p1[h]['beta'] * SCALE:+.4f}$ "
        rf"$\to$ ${p5[h]['beta'] * SCALE:+.4f}$" for h in POSTS)

    # The control set as the RUN used it, not as this function assumes it: the
    # note cannot claim a set the run did not carry. Printed in COVARIATES order
    # rather than the run's sorted one, so the list reads in the same order as
    # the sentence explaining it; the sort governs the design matrix, not prose.
    flag_tbl = ", ".join(esc(c[0].lower() + c[1:])
                         for c in COVARIATES if c in c5["flagged"])
    zf = "; ".join(f"{esc(c)} {n:,}" for c, n in c5["cov_zero"].items())

    frag = rf"""% Produced by step_1/code/pipeline/50_lp_dynamics.py from
% Results/lp/lp_dynamics.csv, rows sample = common. Do not edit.
\begin{{minipage}}{{\textwidth}}\centering
\textbf{{Figure. The lead and lag dynamics with and without the predetermined
covariates}}\\[0.2em]
{{\small The same local projection at shock-control rung $0'$ and at rung $5'$,
common sample $t \in [{cp['year_lo']},{cp['year_hi']}]$}}\\[0.6em]
\includegraphics[width=\textwidth]{{{pdf.name}}}
\end{{minipage}}
\begin{{minipage}}{{\textwidth}}\vspace{{0.4em}}\footnotesize
\textbf{{What is plotted.}} Both series are the headline non-overlapping
horizons: lead placebos $g_{{c,t-k \to t-1}}$ at $k=2,3,4$, the shock year
$g_{{c,t-1 \to t}}$ at horizon $0$, and $g_{{c,t \to t+h}}$ at $h=1,\dots,4$,
each $\beta$ scaled by $10^{{5}}$. The circles and the shaded band are rung $0'$
--- identical to the headline figure. The diamonds, nudged right to avoid
overplotting, are rung $5'$, with 95\% error bars rather than a second band. The
overlapping plan-literal leads are omitted here.
\smallskip
\textbf{{What rung $5'$ adds, and where the set comes from.}} Rung $5'$ is rung
$0'$ plus one control per predetermined county characteristic in producer 22's
named set: {flag_tbl}. Income per head and log population stand for the level of
development and market size, the employment rate for the local slack a fiscal
multiplier is theoretically keyed to, and the two sector shares for the
procurement-facing and the cyclically sensitive part of the local economy. The
set is not chosen here and is not selected from any estimate: it is imported
from \texttt{{22\_bhj\_assumption1.py}}'s \texttt{{RUNG2\_COVARIATES}}, which
names it before that producer estimates anything, so the two cannot drift. It
replaces an earlier rule that took whichever covariates failed the balance test
at $|t|>1.96$, a specification that was a function of the noise in the very test
it was answering. Rung $5'$ carries no verdict, here or in producer 22.
\smallskip
\textbf{{The two rungs differ by exactly five regressors.}} Everything else is
held fixed: the same outcome, the same instrument $z1_{{ct}}$, the same lag
augmentation $p={P_LAG}$, the same division-by-year fixed effects, the same
division-year employment weights, the same exposure-robust inference clustered on
the FSC category, and the same {cp['n']:,} county-years. The covariates are
built OFF the finiteness mask so the sample cannot move with the rung
({zf} rows zero-filled). This is a CONTROL change and not a change of estimand.
\smallskip
\textbf{{One definitional difference from producer 22.}} The covariates are
measured at this producer's own base year $t-{H}$, as $E_{{c,t-1}}$,
$P_{{c,t-1}}$ and $S0_{{ct}}$ are; producer 22 measures the same five at its own
$t-4$. The definitions are identical and the base follows each producer's
horizon.
\smallskip
\textbf{{What moves.}} The leads go {lead_tbl}. The joint Wald test on the three
leads, with the full cross-horizon exposure-robust covariance, is
$F={cp['jl']['F']:.3f}$ with $p={ptxt(cp['jl']['p'])}$ at rung $0'$ and
$F={c5['jl']['F']:.3f}$ with $p={ptxt(c5['jl']['p'])}$ at rung $5'$, both against
$F(3,{cp['jl']['G'] - 1})$. On the disjoint one-year increments the same test
gives $p={ptxt(cp['il']['p'])}$ and $p={ptxt(c5['il']['p'])}$. The post horizons
go {post_tbl}.
\smallskip
\textbf{{Relevance.}} These are reduced forms, so no first stage can be weak and
no marker encodes one. As a separate diagnostic, the exposure-robust first stage
of the $h=1$ treatment on $z1$ is $F={cp['fsF']:.2f}$ at rung $0'$ and
$F={c5['fsF']:.2f}$ at rung $5'$. The effective number of shifters is
{cp['eff_all']['category']:.1f} against the frozen floor of
{CONTINGENCY['effective_cluster_floor']}, unchanged by the rung.
\end{{minipage}}
"""
    tex = out.with_suffix(".tex")
    tex.write_text(frag, encoding="utf-8")
    print(f"  wrote {tex}")


def dynamics_rows(cell):
    """The lp_dynamics.csv rows for one cell: its fits, then its increments.

    sample / rung / year_lo / year_hi lead every row. year_hi is PER ROW, not
    per file, because under the maximal grid the post horizons sit on different
    windows and that is the one fact a reader must not have to look up.
    """
    key = dict(sample=cell["sample"], rung=cell["rung"])
    rows = []
    for f in cell["fits"]:
        rows.append(dict(
            **key, year_lo=f["year_lo"], year_hi=f["year_hi"],
            kind=f["kind"], horizon=f["hz"],
            x=(XPOS_LIT[f["hz"]] if f["kind"] == "lead_literal"
               else XPOS.get((f["kind"], f["hz"]), np.nan)),
            plotted=(f["kind"] == "lead_literal"
                     or (f["kind"], f["hz"]) in XPOS),
            coef=f["beta"] * SCALE,
            se_exposure=f["se_exposure"] * SCALE,
            se_county=f["se_county"] * SCALE,
            t_exposure=f["beta"] / f["se_exposure"],
            n=f["n"], n_controls=f["ncontrol"], n_shock_clusters=f["G_shock"],
            equiv_gap=f["equiv_gap"], route_gap=f["route_gap"]))
    # horizon here is j, the one-year window [t-j-1, t-j]. NOT the lead index:
    # j = 1 is the window [t-2,t-1], which is lead k = 2's window exactly, and
    # run_cell asserts the two coefficients agree.
    il, jl = cell["il"], cell["jl"]
    if il is not None:
        for j, (bj, sj) in enumerate(zip(il["betas"], il["ses"]), start=1):
            rows.append(dict(
                **key, year_lo=cell["year_lo"], year_hi=cell["year_hi"],
                kind="increment", horizon=j, x=np.nan, plotted=False,
                coef=bj * SCALE, se_exposure=sj * SCALE,
                se_county=np.nan, t_exposure=bj / sj,
                n=cell["n"], n_controls=np.nan,
                n_shock_clusters=jl["G"], equiv_gap=np.nan, route_gap=np.nan))
    return rows


def summary_row(cell, scope):
    """One lp_summary.csv row. Only cells carrying the full lead family get one:
    the joint tests are the point of this file and a cell without them would be
    a row of blanks."""
    fits, jl, jlit, il = cell["fits"], cell["jl"], cell["jlit"], cell["il"]
    eff_all = cell["eff_all"]
    n_lead = next(f["n"] for f in fits if f["kind"] == "lead")
    return dict(
        sample=cell["sample"],
        joint_lead_F=jl["F"], joint_lead_p=jl["p"], joint_lead_k=jl["k"],
        joint_lead_maxcorr=jl["maxcorr"], joint_lead_condV=jl["cond"],
        increment_lead_F=il["F"], increment_lead_p=il["p"],
        increment_lead_maxcorr=il["maxcorr"],
        overlapping_lead_F=jlit["F"], overlapping_lead_p=jlit["p"],
        overlapping_lead_maxcorr=jlit["maxcorr"],
        overlap_minus_nonoverlap_mean=float(np.mean(
            [(next(f["beta"] for f in fits
                   if f["kind"] == "lead_literal" and f["hz"] == k)
              - next(f["beta"] for f in fits
                     if f["kind"] == "lead" and f["hz"] == k)) * SCALE
             for k in (2, 3, 4)])),
        n_shock_clusters=jl["G"],
        eff_shifters_category=eff_all["category"],
        eff_shifters_group=eff_all["FSC group"],
        eff_shifters_year=eff_all["year"],
        eff_shifters_kt=eff_all["(k,t)"],
        eff_floor=CONTINGENCY["effective_cluster_floor"],
        first_stage_F_x_on_z1=cell["fsF"], first_stage_pi=cell["pi"],
        p_lag=cell["p"], rung=cell["rung"],
        year_lo=cell["year_lo"], year_hi=cell["year_hi"],
        # n_common keeps its original meaning -- the one sample every horizon
        # shares -- and is blank where no such sample exists. n_fit_lead is the
        # new column, always populated, so nothing silently redefines itself.
        n_common=(cell["n"] if cell["sample"] == "common" else np.nan),
        n_counties=locked.N_COUNTIES,
        cluster_level=CONTINGENCY["cluster_level"],
        lead_headline="non-overlapping g_{c,t-k -> t-1}, k = 2,3,4",
        n_fit_lead=n_lead, summary_scope=scope)


def write_outputs(cells, summaries):
    """lp_dynamics.csv and lp_summary.csv over the whole grid.

    The default invocation runs one cell, so both files hold exactly what they
    held before plus the four key columns.
    """
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    rows = [r for c in cells for r in dynamics_rows(c)]
    csv = TAB_DIR / "lp_dynamics.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    print(f"  wrote {csv}  ({len(rows)} rows)")

    csv2 = TAB_DIR / "lp_summary.csv"
    pd.DataFrame(summaries).to_csv(csv2, index=False)
    print(f"  wrote {csv2}  ({len(summaries)} rows)")


def write_leads_sensitivity(cells):
    """lp_leads_sample_sensitivity.csv: the lead verdict, sample by sample.

    Task 2 of 2026-08-27_ws4_maximal_sample.md. The three leads share ONE
    maximal sample -- none needs E_{c,t+h} -- so unlike the post horizons their
    joint test is well defined on identical rows at every window here, which is
    what makes this table readable as a precision comparison rather than a
    composition one.

    Three windows: maximal YEAR_LO-SAMPLE_LAST, the common sample, and the
    wholly pre-USAspending YEAR_LO-PRESEAM_HI. The maximal window's four extra
    years are ALL post locked.SEAM_YEAR, so it changes the source mix as well
    as the row count; the pre-seam row is what lets a reader tell those apart.
    """
    rows = []
    for c in cells:
        if c["jl"] is None:
            continue
        fits = {(f["kind"], f["hz"]): f for f in c["fits"]}
        lead_f = [fits[("lead", k)] for k in (2, 3, 4)]
        lit_f = [fits.get(("lead_literal", k)) for k in (2, 3, 4)]
        specs = [("cumulative", "k = 2,3,4", c["jl"], lead_f),
                 ("increment", "j = 1,2,3", c["il"], None),
                 ("literal", "k = 2,3,4", c["jlit"],
                  lit_f if all(lit_f) else None)]
        for param, coef_labels, t, fl in specs:
            if t is None:
                continue
            row = dict(sample=c["sample"], sample_label=c["label"],
                       year_lo=c["year_lo"], year_hi=c["year_hi"],
                       rung=c["rung"], param=param, coef_labels=coef_labels)
            for i in range(3):
                row[f"b{i + 1}"] = t["betas"][i] * SCALE
                row[f"se_exp{i + 1}"] = t["ses"][i] * SCALE
                # The increment route returns exposure-robust SEs only; a
                # county-clustered one is left blank rather than invented.
                row[f"se_cty{i + 1}"] = (fl[i]["se_county"] * SCALE
                                         if fl else np.nan)
            row.update(F=t["F"], p=t["p"], k=t["k"], maxcorr=t["maxcorr"],
                       condV=t["cond"], n=c["n"], n_shock_clusters=t["G"],
                       first_stage_F=c["fsF"])
            rows.append(row)
    csv = TAB_DIR / "lp_leads_sample_sensitivity.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    print(f"  wrote {csv}  ({len(rows)} rows)")
    return rows


# ------------------------------------------------------- the pre-trend grid --

def pretrend_grid(S):
    """The lead verdict over weight x lag depth x window, at rung 0'.

    THE THREE AXES, and why each is here rather than assumed away, are in the
    P_GRID block at the top of this file. What this function adds is the one
    thing that block cannot state: the two SAMPLE conventions, and why both are
    reported.

      fixed  every p on [z1_first + max(P_GRID), YEAR_HI]. Identical rows at
             every depth, so a movement in F across the ladder is the lag order
             and nothing else. This is the column to read.
      own    each p on its own longest window [z1_first + p, YEAR_HI]. Maximum
             precision per depth, and the p = 3 cell reproduces the headline
             exactly -- but F then moves with the sample and the depth together,
             which is the confound `fixed` exists to break.

    Only the lead families are built. The posts would force a shorter back
    constraint and are not part of a pre-trend statement; the shock year rides
    along because build_window adds it and validate_house needs it.

    THE HEADLINE IS AN ANCHOR, NOT A REFERENCE RUN. The (division-EW, p = 3,
    own) cell IS Figure_lp_irf's lead test, so it is checked against the frozen
    lp_summary.csv on disk rather than merely recomputed. If that check fails,
    the grid is measuring something other than what the figure reports and no
    row of it can be read.
    """
    lo_fixed = S["z1_first"] + max(P_GRID)
    cells = []
    for weight in WEIGHTS:
        for mode in ("fixed", "own"):
            for p in P_GRID:
                lo = lo_fixed if mode == "fixed" else S["z1_first"] + p
                cells.append(run_cell(
                    S, lo, YEAR_HI, LEAD_KINDS, GRID_RUNG, mode,
                    f"pre-trend grid, weight {weight}, p = {p}, {mode} window",
                    p=p, weight=weight))
    _anchor_check(cells)
    return cells


def _anchor_check(cells):
    """The (division-EW, p = P_LAG, own) cell against the frozen lp_summary.csv.

    Compared to the file on disk, not to a constant in this module: a constant
    would be this producer agreeing with itself, which is no evidence at all
    that the grid reproduces the exhibit. Absent or unreadable, the check is
    SKIPPED loudly rather than passed silently.
    """
    anchor = next((c for c in cells
                   if c["weight"] == "division-EW" and c["p"] == P_LAG
                   and c["sample"] == "own"), None)
    csv = TAB_DIR / "lp_summary.csv"
    if anchor is None or not csv.exists():
        print("\n  ANCHOR CHECK SKIPPED: no (division-EW, p = "
              f"{P_LAG}, own) cell or no {csv.name} on disk. The grid has not "
              "been tied to the published figure.")
        return
    s = pd.read_csv(csv)
    s = s[(s["sample"] == "common") & (s["rung"] == GRID_RUNG)]
    if s.empty:
        print(f"\n  ANCHOR CHECK SKIPPED: {csv.name} carries no common/rung "
              f"{GRID_RUNG} row.")
        return
    # The filter must name ONE row. It did when 0' and 0 were the only rungs; a
    # third rung makes "take the first" a silent choice, so it is now an error.
    if len(s) != 1:
        raise SystemExit(
            f"  FAIL: {csv.name} carries {len(s)} rows at sample = common, "
            f"rung {GRID_RUNG}; the anchor is not identified. The grid can only "
            "be tied to the published figure if exactly one row is the figure.")
    r = s.iloc[0]
    print(f"\n  anchor: (division-EW, p = {P_LAG}, own window "
          f"[{anchor['year_lo']},{anchor['year_hi']}]) vs the frozen "
          f"{csv.name}")
    ok = True
    for lab, got, want in (("joint lead F", anchor["jl"]["F"],
                            float(r["joint_lead_F"])),
                           ("increment lead F", anchor["il"]["F"],
                            float(r["increment_lead_F"])),
                           ("n", float(anchor["n"]), float(r["n_fit_lead"]))):
        gap = abs(got - want) / max(abs(want), 1e-300)
        good = gap < locked.RTOL_COEF
        ok &= good
        print(f"    {lab:<18} {got:.9f} vs {want:.9f}  rel gap {gap:.1e}  "
              f"{'PASS' if good else 'FAIL'}")
    if not ok:
        raise SystemExit(
            "  FAIL: the grid's headline cell does not reproduce the frozen "
            "lead test. Either the parameterisation changed the exhibit or the "
            "grid is not running the specification the figure reports; no row "
            "of this table is readable until they agree.")


GRID_PARAMS = (("cumulative", "jl", "leads k = 2,3,4, windows [t-k, t-1]"),
               ("increment", "il", "disjoint one-year windows [t-j-1, t-j]"),
               ("literal", "jlit", "overlapping leads, windows [t-k, t]"))


def pretrend_rows(cells):
    """One row per (weight, p, window mode, parameterisation)."""
    rows = []
    for c in cells:
        for param, key, note in GRID_PARAMS:
            t = c[key]
            if t is None:
                continue
            row = dict(weight=c["weight"], p=c["p"], sample=c["sample"],
                       rung=c["rung"], param=param, param_note=note,
                       year_lo=c["year_lo"], year_hi=c["year_hi"], n=c["n"],
                       k=t["k"], F=t["F"], p_value=t["p"],
                       G=t["G"], G_eff=t["G_eff"], df_eff=t["df_eff"],
                       F_eff=t["F_eff"], p_value_eff=t["p_eff"],
                       maxcorr=t["maxcorr"], condV=t["cond"],
                       first_stage_F=c["fsF"])
            for i in range(3):
                row[f"b{i + 1}"] = t["betas"][i] * SCALE
                row[f"se_exp{i + 1}"] = t["ses"][i] * SCALE
            n_ctrl = next((f["ncontrol"] for f in c["fits"]
                           if f["kind"] == "lead"), np.nan)
            row["n_controls"] = n_ctrl
            rows.append(row)
    return rows


def _grid_panel(df, mode, param, label):
    """One tabular block: rows p = 3..8, column groups per weight.

    Under the own-window mode the row label carries the front year, because the
    sample is then part of what the row is and a reader must not have to leave
    the table to find it.
    """
    ncol = 1 + 3 * len(WEIGHTS)
    out = [r"\multicolumn{%d}{l}{\textit{%s}} \\" % (ncol, label)]
    for p in P_GRID:
        cells = []
        lab = f"$p={p}$"
        for weight in WEIGHTS:
            r = df[(df["sample"] == mode) & (df["param"] == param)
                   & (df["weight"] == weight) & (df["p"] == p)]
            if r.empty:
                cells += ["---", "---", "---"]
                continue
            r = r.iloc[0]
            if mode == "own":
                lab = (f"$p={p}$, $t \\geq {int(r['year_lo'])}$ "
                       f"({int(r['n']):,})")
            cells += [f"{r['F']:.3f}", _p(r["p_value"]), _p(r["p_value_eff"])]
        out.append(lab + " & " + " & ".join(cells) + r" \\")
    return out


def _p(v):
    """A p-value, with the reject/not boundary legible at a glance."""
    if not np.isfinite(v):
        return "---"
    if v < 0.001:
        return r"$<$0.001"
    return f"{v:.3f}"


def write_pretrend_table(rows, out):
    """Table_lp_pretrend.tex: the joint lead F and p over the grid.

    No figure. Four panels -- {fixed, own} x {cumulative, increment} -- because
    those are the two windows and the two parameterisations a reader has to see
    together; the overlapping-lead rows stay in the CSV and are summarised in
    one line of the note.
    """
    df = pd.DataFrame(rows)
    ncol = 1 + 3 * len(WEIGHTS)
    lo_fixed = int(df[df["sample"] == "fixed"]["year_lo"].iloc[0])
    n_fixed = int(df[df["sample"] == "fixed"]["n"].iloc[0])
    body = []
    for mode, mlab in (("fixed", "Panel A. One window at every depth, "
                        f"$t \\in [{lo_fixed}, {YEAR_HI}]$, "
                        f"{n_fixed:,} county-years throughout"),
                       ("own", "Panel B. Each depth on its own longest window, "
                        f"ending {YEAR_HI}")):
        body.append(r"\addlinespace")
        body.append(r"\multicolumn{%d}{l}{\textbf{%s}} \\" % (ncol, mlab))
        for param, plab in (("cumulative", "nested leads $k=2,3,4$"),
                            ("increment", "disjoint increments $j=1,2,3$")):
            body.append(r"\addlinespace")
            body += _grid_panel(df, mode, param, plab)

    # The overlapping-lead verdict, compressed to one sentence: the largest gap
    # between its p and the nested one over cells that exist in both.
    key = ["weight", "p", "sample"]
    piv = df.pivot_table(index=key, columns="param", values="p_value")
    lit_gap = (float((piv["literal"] - piv["cumulative"]).abs().max())
               if {"literal", "cumulative"} <= set(piv.columns) else np.nan)

    hdr = " & ".join(
        [r"\multicolumn{3}{c}{%s}" % ("division-EW weights (headline)"
                                      if w == "division-EW"
                                      else "equal weights")
         for w in WEIGHTS])
    sub = " & ".join([r"$F$ & $p$ (df $=G-1$) & $p$ (df $\approx G_{\text{eff}}$)"
                      for _ in WEIGHTS])

    frag = "\n".join([
        r"% Produced by step_1/code/pipeline/50_lp_dynamics.py --pretrend-grid",
        r"% from Results/lp/lp_pretrend_grid.csv. Do not edit.",
        r"\begin{table}[htbp]",
        r"\caption{The county lead test, by weighting, lag depth and reference "
        r"distribution}",
        r"\label{tab:lp_pretrend}",
        r"{\centering",
        r"\adjustbox{max width=\linewidth}{",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r" & " + hdr + r" \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
        r" & " + sub + r" \\",
        r"\midrule",
        *body,
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        r"\par}",
        r"\medskip",
        r"{\scriptsize \textit{Notes:} " + _pretrend_note(df, lit_gap) + r" \par}",
        r"\end{table}",
        "",
    ])
    tex = out.with_suffix(".tex")
    tex.write_text(frag, encoding="utf-8")
    print(f"  wrote {tex}")


def _pretrend_note(df, lit_gap):
    base = df[(df["sample"] == "own") & (df["param"] == "cumulative")
              & (df["weight"] == "division-EW") & (df["p"] == P_LAG)]
    b = base.iloc[0] if not base.empty else None
    G = int(df["G"].iloc[0])
    Geff = df.groupby("weight")["G_eff"].first()
    geff_txt = "; ".join(f"{w} {v:.1f}" for w, v in Geff.items())

    # The one sentence a reader would otherwise get wrong: a lost rejection is
    # not a vanished pattern. Computed from the table rather than asserted, so
    # it cannot outlive the numbers it describes.
    def _incr(weight):
        r = df[(df["sample"] == "own") & (df["param"] == "increment")
               & (df["weight"] == weight) & (df["p"] == P_LAG)]
        if r.empty:
            return None
        r = r.iloc[0]
        return ([r[f"b{i}"] for i in (1, 2, 3)],
                [r[f"se_exp{i}"] for i in (1, 2, 3)])
    ba, bb = _incr("division-EW"), _incr("equal")
    if ba and bb:
        bigger = sum(abs(y) > abs(x) for x, y in zip(ba[0], bb[0]))
        wider = sum(y > x for x, y in zip(ba[1], bb[1]))
        precision = (
            r"\textbf{A lost rejection is not a vanished pattern.} At the "
            r"headline depth and window the move to equal weights leaves the "
            r"increments at "
            + ", ".join(f"${v:+.3f}$" for v in bb[0])
            + r" against "
            + ", ".join(f"${v:+.3f}$" for v in ba[0])
            + rf" --- {bigger} of three larger in absolute value, and "
            rf"{wider} of three carrying a wider standard error. Where the "
            r"joint test stops rejecting under equal weights it does so on "
            r"precision, not because the pre-shock pattern went away. ")
    else:
        precision = ""

    return (
        precision +
        r"Each cell is the joint Wald test on three lead coefficients from the "
        r"county reduced form of midpoint employment growth on the one-year "
        r"composition instrument $z1_{ct}$, at rung $0'$ (share-sum level "
        r"control, no share-sum $\times$ year interaction), division-by-year "
        r"fixed effects, exposure-robust and clustered on the FSC category. "
        r"The headline exhibit is the "
        + (rf"($p={P_LAG}$, division-EW, Panel B) cell, $F={b['F']:.3f}$, "
           rf"$p={b['p_value']:.4f}$. " if b is not None else "")
        + r"\textbf{Weighting.} The headline weight is $E_{c,t-1}$ over its "
        r"division-year total. For the shock year and the post horizons that is "
        r"predetermined; for a lead $g_{c,t-k \to t-1}$ it is the window's own "
        r"terminal employment, so the weight is a monotone function of the "
        r"dependent variable's numerator. Equal weighting removes that channel. "
        r"\textbf{Lag depth.} The $k=4$ lead spans $[t-4,t-1]$ and the post "
        r"horizons show a response still running at $h=4$, so growth inside the "
        r"lead window responds to $z1_{t-4}$ and deeper, uncontrolled at $p=3$. "
        r"Panel A holds the rows fixed so the ladder is the depth alone; Panel B "
        r"lets each depth keep its longest window and therefore moves the sample "
        r"with the depth. \textbf{Degrees of freedom.} The third column of each "
        rf"group reads the same statistic against the effective number of "
        rf"shifters ({geff_txt}) rather than the literal {G} categories, in both "
        r"the finite-cluster correction and the reference distribution. That "
        r"count is $1/HHI$ of the cluster-summed exposure weights, not a cluster "
        r"count, so it is a conservative sensitivity and not a correction: both "
        r"columns are reported and neither replaces the other. "
        + (rf"The overlapping-lead parameterisation is in "
           rf"\texttt{{lp\_pretrend\_grid.csv}}; its $p$ never departs from the "
           rf"nested one by more than {lit_gap:.3f}. "
           if np.isfinite(lit_gap) else "")
        + r"Point estimates and standard errors for every cell are in that same "
        r"file. Nothing here is winsorized and no county-year is dropped: every "
        r"window is the full "
        rf"{locked.N_COUNTIES:,}-county rectangle.")


def write_pretrend_outputs(cells):
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    rows = pretrend_rows(cells)
    csv = TAB_DIR / "lp_pretrend_grid.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    print(f"  wrote {csv}  ({len(rows)} rows)")
    write_pretrend_table(rows, TAB_DIR / "Table_lp_pretrend")
    return rows


# -------------------------------------------------------------------- main --

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample", choices=("common", "maximal", "both"),
                    default="common",
                    help="maximal re-runs each horizon on its OWN longest "
                         "window; the headline stays on the common sample and "
                         "the maximal grid is never plotted (the post horizons "
                         "then sit on different calendar periods)")
    ap.add_argument("--rung", choices=("0'", "0", RUNG5, "both", "all"),
                    default="0'",
                    help="0' is the reported baseline (share-sum level "
                         "control); 0 drops it and is NOT admissible under "
                         "ws0_contingency.py:34 -- a diagnostic only; 5' ADDS "
                         "22's named predetermined covariates to rung 0'. "
                         "both = 0' and 0, unchanged; all = 0', 0 and 5'. "
                         "Every companion rung runs on the common sample, "
                         "mirroring the headline")
    ap.add_argument("--leads-sensitivity", action="store_true",
                    help="also write lp_leads_sample_sensitivity.csv: the lead "
                         "verdict at the maximal, common and pre-seam windows")
    ap.add_argument("--pretrend-grid", action="store_true",
                    help="run ONLY the pre-trend grid -- the lead verdict over "
                         "weight (division-EW, equal) x lag depth (p = 3..8) x "
                         "window (fixed, own), at rung 0', with the effective- "
                         "cluster degrees of freedom beside the literal ones. "
                         "Writes lp_pretrend_grid.csv and Table_lp_pretrend.tex "
                         "and NOTHING else: the figure and the three headline "
                         "CSVs are left exactly as they are on disk")
    a = ap.parse_args()
    t0 = time.time()

    # The grid is a separate run, not an addition to the headline one. It shares
    # the stores and every check, and it writes only its own two files, so a
    # reader can never find the exhibit silently re-rendered under a weight or a
    # depth that is not the one the figure's note claims.
    if a.pretrend_grid:
        print(f"== stage 50: PRE-TREND GRID, h = {H}, rung {GRID_RUNG}")
        print(f"   weights {WEIGHTS}, p in {P_GRID}, windows fixed and own")
        print(f"   the headline exhibits are NOT rewritten by this flag")
        S = load_stores()
        cells = pretrend_grid(S)
        print()
        write_pretrend_outputs(cells)
        gaps = [f["equiv_gap"] for c in cells for f in c["fits"]]
        rgaps = [f["route_gap"] for c in cells for f in c["fits"]]
        print(f"  BHJ county-vs-shock-level identity over all {len(gaps)} "
              f"fits: max equiv_gap {max(gaps):.1e}, max route_gap "
              f"{max(rgaps):.1e} (tolerance {locked.RTOL_COEF:.0e})")
        print(f"  done in {time.time() - t0:.1f}s")
        return

    want_max = a.sample in ("maximal", "both")
    want_r0 = a.rung in ("0", "both", "all")
    want_r5 = a.rung in (RUNG5, "all")

    print(f"== stage 50: county lead-lag local projection, h = {H}, p = {P_LAG}")
    print(f"   common sample t in [{YEAR_LO}, {YEAR_HI}]; sample={a.sample}, "
          f"rung={a.rung}, leads_sensitivity={a.leads_sensitivity}")
    S = load_stores()

    all_kinds = ALL_KINDS
    cells, summaries, sens = [], [], []

    # ---- cell A: the headline. Always run; it is what the figure reports. ---
    cA = run_cell(S, YEAR_LO, YEAR_HI, all_kinds, "0'", "common",
                  "common sample, rung 0' (HEADLINE)")
    cells.append(cA)
    summaries.append(summary_row(cA, "all horizons on one common sample"))
    sens.append(cA)

    # ---- cell B: the rung-0 twin, same window, same horizons ---------------
    cB = None
    if want_r0:
        cB = run_cell(S, YEAR_LO, YEAR_HI, all_kinds, "0", "common",
                      "common sample, rung 0 (no share-sum control)")
        cells.append(cB)
        summaries.append(summary_row(cB, "all horizons on one common sample"))
        sens.append(cB)

    # ---- cell B2: the rung-5' twin, same window, same horizons -------------
    # The control set is resolved BEFORE the cell runs, so a set that has moved
    # upstream fails here rather than after a full estimation pass.
    cB5 = None
    if want_r5:
        flagged = flagged_names()
        cB5 = run_cell(S, YEAR_LO, YEAR_HI, all_kinds, RUNG5, "common",
                       f"common sample, rung {RUNG5} (0' + the predetermined "
                       "covariates)", flagged=flagged)
        cells.append(cB5)
        summaries.append(summary_row(cB5, "all horizons on one common sample"))
        sens.append(cB5)
        if cB5["n"] != cA["n"]:
            raise SystemExit(
                f"  FAIL: rung {RUNG5} ran on {cB5['n']:,} county-years against "
                f"the baseline's {cA['n']:,}. The covariates are built "
                "off the finiteness mask precisely so this cannot happen; a "
                "rung comparison on different rows is confounded with a sample "
                "change.")
        print(f"\n    rung {RUNG5} vs 0': {cB5['ncontrol']} controls against "
              f"{cA['ncontrol']}, on the identical {cA['n']:,} county-years")
        if cB5["ncontrol"] != cA["ncontrol"] + len(flagged):
            raise SystemExit(
                f"  FAIL: rung {RUNG5} carries {cB5['ncontrol']} controls, not "
                f"{cA['ncontrol']} + {len(flagged)}. The covariates "
                "were dropped as collinear or duplicated, so the rung is not "
                "what its label says.")

    # ---- cell C: the maximal grid, one window per horizon ------------------
    # Grouped by back constraint: everything that does not reach forward shares
    # YEAR_LO-SAMPLE_LAST, and each post horizon loses exactly h years.
    cmax_lead = None
    if want_max or a.leads_sensitivity:
        cmax_lead = run_cell(
            S, YEAR_LO, SAMPLE_LAST,
            (("lead", LEADS), ("impact", (0,)),
             ("lead_literal", LEADS_LITERAL)), "0'", "maximal",
            f"maximal sample {YEAR_LO}-{SAMPLE_LAST}, rung 0' "
            "(leads, shock year, overlapping leads)")
        sens.append(cmax_lead)
    if want_max:
        cells.append(cmax_lead)
        summaries.append(summary_row(
            cmax_lead,
            f"leads / shock year / overlapping leads on [{YEAR_LO},"
            f"{SAMPLE_LAST}]; the post horizons sit on SHORTER windows, one "
            "per horizon -- see year_hi in lp_dynamics.csv"))
        for h in POSTS:
            hi = maximal_hi("post", h)
            if hi == SAMPLE_LAST:
                raise SystemExit("  FAIL: a post horizon must lose years")
            cells.append(run_cell(
                S, YEAR_LO, hi, (("post", (h,)),), "0'", "maximal",
                f"maximal sample {YEAR_LO}-{hi}, rung 0' (post h = {h})"))

        # post h = 4's maximal window IS the common window, so the two fits are
        # the same regression and must agree exactly. Free identity check that
        # the per-horizon build reproduces the headline build.
        m4 = next(f for c in cells if c["sample"] == "maximal"
                  for f in c["fits"] if (f["kind"], f["hz"]) == ("post", 4))
        c4 = next(f for f in cA["fits"] if (f["kind"], f["hz"]) == ("post", 4))
        g4 = abs(m4["beta"] - c4["beta"]) / max(abs(c4["beta"]), 1e-300)
        print(f"\n    maximal post h = 4 vs common post h = 4 (the SAME "
              f"window {YEAR_LO}-{maximal_hi('post', 4)}): rel gap {g4:.1e} "
              f"{'PASS' if g4 == 0.0 else 'FAIL'}")
        if g4 != 0.0:
            raise SystemExit("  FAIL: post h = 4's maximal window is the common "
                             "window, so the two fits are one regression and "
                             "must be bit-identical; the per-horizon build "
                             "does not reproduce the headline build")

    # ---- cell D: the pre-seam window, leads only --------------------------
    if a.leads_sensitivity:
        cP = run_cell(S, YEAR_LO, PRESEAM_HI, LEAD_KINDS, "0'", "preseam",
                      f"pre-seam sample {YEAR_LO}-{PRESEAM_HI}, rung 0' "
                      "(wholly before locked.SEAM_YEAR)")
        cells.append(cP)
        summaries.append(summary_row(
            cP, f"leads and overlapping leads only, [{YEAR_LO},{PRESEAM_HI}]"))
        sens.append(cP)

    print()
    write_outputs(cells, summaries)
    write_figure(cA["fits"], cA["jl"], cA["jlit"], cA["il"], cA["fsF"],
                 cA["eff_all"]["category"], TAB_DIR / "Figure_lp_irf")
    if cB is not None:
        write_figure_rungs(cA, cB, TAB_DIR / "Figure_lp_irf1")
    if cB5 is not None:
        write_figure_flagged(cA, cB5, TAB_DIR / "Figure_lp_irf5")
    if a.leads_sensitivity:
        write_leads_sensitivity(sens)

    # The BHJ identity is the licence for every exposure-robust SE above, so it
    # is reported in the aggregate rather than only asserted per fit.
    gaps = [f["equiv_gap"] for c in cells for f in c["fits"]]
    rgaps = [f["route_gap"] for c in cells for f in c["fits"]]
    print(f"  BHJ county-vs-shock-level identity over all {len(gaps)} fits: "
          f"max equiv_gap {max(gaps):.1e}, max route_gap {max(rgaps):.1e} "
          f"(tolerance {locked.RTOL_COEF:.0e})")
    print(f"  done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
