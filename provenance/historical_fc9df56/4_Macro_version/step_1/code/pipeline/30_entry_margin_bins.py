"""Stage 30: which industries drive the entry margin of county employment growth.

WS2 step 4 of PLAN_ra_next_tasks.md, the no-NETS pass. Midpoint growth
reconstructs the county exactly from its cells,

    mpgr_ct = sum_m w_cmt mpgr_cmt ,   w_cmt = (E_cmt + E_cm,t-h)/(E_ct + E_c,t-h)

so the county outcome splits by what happened to the CELL and the pieces still
add to the county number. 31_margin_decomposition.py takes that split four ways
(continuers, entry, exit, always zero). This producer takes the ENTRY piece and
splits it again, by an ex-ante classification of the industry:

    Entry_ct(b) = sum_{m in b : E_cm,t-h <= 0, E_cmt > 0} 2 E_cmt / (E_ct + E_c,t-h)

and estimates one 2SLS coefficient per bin on the locked frame. Because the
regression is linear in the outcome the bin betas ADD to the entry beta, and the
entry beta plus the other three margins ADD to the county total. Both are
asserted, not hoped for.

THE FRAME IS THE LOCKED COMPOSITION FRAME, NOT 31's. 31 estimates on the
candidate Stata frame (z_income_fixed5, IQR5-by-year winsorization, ccg + year
FE, equal weights, pre-1998, nonoverlapping windows). This producer estimates on
the locked pipeline baseline: treatment x_pos, instrument z (the composition
instrument), division x year fixed effects, employment-share-within-division-year
weights mean-normalised, county clusters, NO winsorization (locked.py: "the
locked estimation in 57 never winsorizes"). So the entry total here is a NEW
number and does not reproduce 31's -1.308 at h = 2. 31 is reused as machinery -
its `partition` is imported rather than restated - and not as a tie-out target.

THREE WINDOWS ARE ESTIMATED; THE .tex PRINTS TWO OF THEM SIDE BY SIDE. Two are
kept strictly on one side of the NAICS break - pre is t - h >= 1975 and t <=
1997, post is t - h >= 1998 and t <= 2016 - so neither straddles 1998 in either
direction, which is 31's reason for its own pre-1998 restriction ("a cell is not
the same object on the two sides of it", w22.NAICS_BREAK). The third, `full`, is
the POOLED sample: the whole locked panel, no year restriction, n =
locked.N_COUNTY. It straddles the break on purpose, because the pooled estimate
over the whole time span is a thing worth decomposing - it is what the locked
baseline reports - and because printing it beside the clean-measurement window is
what lets a reader see what pooling does.

Table_entry_bins_efsy.tex carries TABLE_WINDOWS, pre-1998 and the pooled span, as
two column groups. entry_bins_coeff.csv carries all three, keyed by the `window`
column. What is presentational is only which two are printed: every gate, every
identity and every coefficient runs on all three windows on every run.

WHAT POOLING COSTS IS MEASURED, NOT WAVED AT. The pooled window contains
county-years with t - h < 1998 <= t, which neither one-sided window has. Their
midpoint growth compares a SIC-bridged cell at the base period to a native-NAICS
cell at the outcome year, so part of what the entry indicator counts as entry
there is the 1998 measurement change. A.3.1 forbids dropping them; they are sized
per horizon as diagnostic D1, reported in entry_bins_identities.csv and quoted in
the table note. D1 is not a gate: a straddle is intrinsic to a pooled window, so
a pass threshold for it would license no action.

THE POOLED WINDOW IS CHECKED TO BE THE LOCKED SAMPLE. Gate G6 asserts that its
row count is locked.N_COUNTY and that its county TOTAL y_mpgr reproduces the
coefficient window_context reads straight off the panel with no window filter -
the same estimand on the same rows by two different code paths. Its own row
accounting (pre + post + straddle + tail) is printed per horizon and asserted.

THE POOLED WINDOW SHARES PRE'S RZ CUT YEAR rather than taking one of its own, so
that a Panel C or D row is the same set of industries in both printed column
groups. A side-by-side whose rows mean different things left and right is worse
than no side-by-side.

The reason this producer estimated only the post window until now was the
classifications, not the cells: producer 69's EFSY coverage assertion for
Mian-Sufi was 1998-2016 and its docstring said 1975-97 "needs its own coverage
assertion". That assertion is now discharged - here, per window, as G3a and G3b.
It had to be, because THE COUNTY COEFFICIENT THIS TABLE DECOMPOSES CHANGES SIGN
ACROSS THE BREAK. Since only one side is printed, the note quotes the county
totals of BOTH, generated from the run rather than typed: a reader of the page
alone must not be able to mistake the printed window for the whole story.

PRE-1998 CARRIES MASS THAT POST-1998 DOES NOT, and A.3.1 forbids dropping it.
EFSY is SIC-bridged before 1998 and mixes 2-, 3-, 4- and 6-digit padded codes;
after producer 11's allocation a small residual stays on unresolved roots
(0.21-0.29% of employment), and 0.36-1.51% sits on four-digit codes outside
Mian-Sufi's 294. Both get a NAMED BIN - `Unresolved NAICS root` and `Not in the
Mian-Sufi 294` - which is why the bins still tile the universe and G2b still
holds. Both bins are exactly empty post-1998, so splitting them out cannot move
a single post number: every post coefficient reproduces the one-window table.

THE RZ QUARTILE CUTS ARE RECOMPUTED PER WINDOW, on that window's base year
(RZ_CUT_YEAR). They are employment-weighted and manufacturing is 25% of
employment in 1975 against 15.8% in 1998, so a shared cut would put most of the
pre window's manufacturing in the wrong quartile. The price is stated in the
note and at the panel: "EFD Q3" is NOT the same set of industries in the two
windows.

The COVID rule does not bite: the later window ends in 2016, before 2020-21.

THE ESTIMAND IS NEW AND THE ELEVEN PRE-CODE QUESTIONS ARE ANSWERED IN
Results/entry_margin/entry_formula_memo.md, written by this producer before any
coefficient is read. Nothing standing is invalidated: locked.py is untouched, and
the binned entry contributions and their windows are all new objects.

WHY 2SLS AND NOT THE REDUCED FORM. The plan document writes the estimand as
"regressed on z_ct". The headline is 2SLS on x_pos instrumented by z, so that the
bins decompose the same beta the locked baseline reports rather than a different
one. The reduced form on z is additive too and is the object the plan names, so
it is still estimated and still written to entry_bins_coeff.csv under
spec="rf" - it is no longer given a table panel, because the table spends its
width on the two printed windows instead.

Consumes ONLY persisted stores: analysis_panel.parquet from stage 04, the EFSY
cell parquets from producer 11, and the three reference classifications. Writes
to Results/entry_margin/:

  Table_entry_bins_efsy.tex   the payoff exhibit
  entry_bins_coeff.csv        one row per printed number
  entry_bins_identities.csv   the construction gates, auditable after the fact
  entry_formula_memo.md       the estimand and the protocol answers
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
import oldpath as op  # noqa: E402
import panel as pnl_lib  # noqa: E402
import dcads_common as dc  # noqa: E402

sys.path.insert(0, str(PIPELINE.parents[2] / "step_2" / "code" / "py"))
import out_paths  # noqa: E402

t28, w22 = op.t28, op.w22
m31 = op._load("margin_decomposition",
               PIPELINE.parents[2] / "step_2" / "code" / "py"
               / "31_margin_decomposition.py")
partition, MARGINS = m31.partition, m31.MARGINS

PIPE = dc.MACRO_DATA_ROOT / "temp" / "pipeline"
REF_DIR = PIPELINE.parents[1] / "code" / "reference"
TAB_DIR = PIPELINE.parents[2] / "Results" / "entry_margin"

# The three windows, (lo, hi) meaning `t - h >= lo` AND `t <= hi`. `pre` and
# `post` each sit strictly on one side of the 1998 NAICS break, in the outcome
# year and in the base period both. `full` is the POOLED sample and straddles it
# on purpose: it is the whole locked panel, and (locked.FIRST_YEAR,
# locked.LAST_YEAR) filters nothing off it. Keeping the (lo, hi) type uniform
# rather than a None sentinel is what lets load_frame, the memo and the console
# banner treat all three the same.
WINDOWS = {"pre":  (w22.EFSY_FIRST_YEAR, w22.NAICS_BREAK - 1),   # 1975-1997
           "post": (w22.NAICS_BREAK, 2016),                      # 1998-2016
           "full": (locked.FIRST_YEAR, locked.LAST_YEAR)}        # the whole panel
WINDOW_ORDER = ("pre", "post", "full")
WINDOW_LABEL = {"pre": "Pre-1998", "post": "1998--2016",
                "full": "Full span"}
# ALL THREE WINDOWS ARE ALWAYS ESTIMATED; THESE TWO ARE PRINTED IN THE .tex, as
# two column groups. `post` is in entry_bins_coeff.csv under the `window`
# column, and the table note quotes the county totals of all three so the sign
# flip is legible from the page. Running a single window explicitly
# (--window post) prints that one instead.
TABLE_WINDOWS = ("pre", "full")
# Employment-weighted RZ quartile cuts, recomputed on each window's own base
# year. Post keeps 1998 exactly, which is what makes the post column reproduce.
# THE POOLED WINDOW USES PRE'S CUT YEAR, not one of its own: the two printed
# column groups have to bin the same industry sets or a Panel C row means two
# different things left and right, which is the one thing a side-by-side must
# not do. The price is that post-1998 cells inside the pooled column are binned
# on 1975 cuts, said at the panel and in the note.
RZ_CUT_YEAR = {"pre": w22.EFSY_FIRST_YEAR, "post": w22.NAICS_BREAK,
               "full": w22.EFSY_FIRST_YEAR}
RUNG = "naics4_alloc"
RESIDUAL3 = 100_000              # 11_build_employment_cells.py's root encoding
TOL_IDENTITY = 1e-10             # margins against county midpoint growth
TOL_ADDITIVITY = 1e-8            # bin betas against the margin beta
TOL_LEVEL = 1e-9                 # cell employment sums against the locked panel
# Pre-stated ceilings on the two named residual bins that only pre-1998 fills.
# Realised: roots 0.29% and Mian-Sufi-unclassified 1.51% in the worst pre year,
# both exactly 0 post-1998. These are gates on the SIZE of a reported bin, never
# a licence to drop it (A.3.1).
MAX_ROOT_SHARE = 0.01
MAX_UNCLASS_SHARE = 0.05
NB_NONTRADABLE = "Non-tradable (strict)"
NB_OTHER = "All other industries"
NB_ROOT = "Unresolved NAICS root"
MS_UNCLASS = "Not in the Mian-Sufi 294"
RZ_UNCOVERED = "Manufacturing, no RZ value"
RZ_OUTSIDE = "Outside RZ coverage"
RZ_POOLED = "Pooled across several ISIC industries"


# ------------------------------------------------------------------ inputs --

def load_frame(h, win):
    """The locked analysis panel at one horizon, on one window of it."""
    lo, hi = WINDOWS[win]
    d = pd.read_parquet(PIPE / "analysis_panel.parquet")
    d = d[d["h"] == h]
    n_all = len(d)
    d = d[(d["year"] - h >= lo) & (d["year"] <= hi)]
    d = d.sort_values(["ccg", "year"]).reset_index(drop=True)
    print(f"    locked rows at h={h}: {n_all:,}; on the {win} window "
          f"(t-h >= {lo}, t <= {hi}): {len(d):,}, "
          f"{d['ccg'].nunique():,} counties, outcome years "
          f"{d['year'].min()}-{d['year'].max()}")
    # The pooled window is the WHOLE panel, so its bounds must bite on nothing.
    # If a future panel reaches outside them this is a silent sample change,
    # which is exactly the class of edit the locked pipeline refuses to make
    # quietly.
    if win == "full" and len(d) != n_all:
        raise SystemExit(f"FAIL: the full window dropped {n_all - len(d):,} of "
                         f"{n_all:,} rows at h={h}; it is meant to be the whole "
                         f"locked panel, so WINDOWS['full'] no longer covers it")
    # load_cells opens one EFSY file per year in [min(t-h), max(t)]; the cell
    # build starts at EFSY_FIRST_YEAR, so a base period before it would fail
    # deep inside load_rung as a missing file rather than here as a sample
    # statement.
    if int(d["year"].min()) - h < w22.EFSY_FIRST_YEAR:
        raise SystemExit(f"FAIL: the {win} window at h={h} needs base year "
                         f"{int(d['year'].min()) - h}, before the first EFSY "
                         f"cell file ({w22.EFSY_FIRST_YEAR})")
    return d


def load_cells(years):
    """Employment by cell and year on a support asserted identical, plus codes.

    w22.load_rung already asserts the support identity that makes the t-h lag an
    array shift rather than a join; the code vector comes from the same files.
    The support is identical over the whole 1975-2018 span, so this holds on
    either side of the NAICS break and the lag never becomes a join.

    UNRESOLVED ROOTS ARE NOT A FAILURE ANY MORE, THEY ARE A BIN. Post-1998 they
    carry exactly zero, but pre-1998 EFSY is SIC-bridged and a small residual
    survives producer 11's allocation. A.3.1 forbids dropping it, so bin_schemes
    sends it to NB_ROOT in every scheme and coverage_rows gates its size.
    """
    emp, ccg_arr = w22.load_rung(RUNG, years)
    _, code_arr = op._load("cell_checks",
                           PIPELINE.parents[2] / "step_2" / "code" / "py"
                           / "27_cell_construction_checks.py").rung_codes(
                               RUNG, years[0])
    return emp, ccg_arr, code_arr


def bin_schemes(code_arr, cut_year):
    """Ex-ante industry bins on the fixed cell-code vector.

    Every scheme TILES the code universe: a code that cannot be classified goes
    to a named bin, never to a drop (INVARIANCE_TEST_PROTOCOL.md A.3.1). There
    are two such bins that only the pre-1998 window ever fills - NB_ROOT for
    employment producer 11 could not push down to four digits, and MS_UNCLASS
    for four-digit codes outside Mian-Sufi's 294 - and both are empty in the
    post window, which is why the post coefficients are unaffected by them.

    THE BIN LISTS ARE WINDOW-INDEPENDENT and so are the labels; only `cut_year`
    differs, and it moves codes BETWEEN the RZ quartiles without changing which
    bins exist. That is what keeps the two column groups aligned row for row.
    """
    codes = np.unique(code_arr)
    genuine = codes[codes < RESIDUAL3]
    schemes = {}

    # --- Mian-Sufi ---------------------------------------------------------
    ms = pd.read_csv(REF_DIR / "miansufi_naics4_classification.csv", comment="#")
    strict = dict(zip(ms["naics4"].astype(int), ms["nontradable_strict"].astype(int)))
    indcat = dict(zip(ms["naics4"].astype(int), ms["indcat"].astype(int)))
    # THE 7225 PATCH, MANDATORY. EFSY uses NAICS4 7225 (restaurants and other
    # eating places), absent from Mian-Sufi's 294 rows, which absorbs their 7221
    # and 7222. Without the patch the single largest non-tradable bin - 8.6% of
    # 2016 EFSY employment - is unclassified. 69_ingest_miansufi_classification
    # states the mapping: 7225 is non-tradable, strict.
    strict[7225], indcat[7225] = 1, 2
    miss = sorted(int(c) for c in genuine if int(c) not in strict)
    print(f"    Mian-Sufi: {len(ms)} rows + the 7225 patch; "
          f"{len(miss)} of {len(genuine)} EFSY codes unclassified {miss}")

    # A code Mian-Sufi never classified is NOT quietly "all other industries".
    # Post-1998 the distinction is vacuous - the 294 plus the 7225 patch cover
    # every EFSY code - but pre-1998 it is up to 1.5% of employment, and folding
    # it into NB_OTHER would make that row mean two different things in the two
    # column groups.
    def _strict(c):
        c = int(c)
        if c >= RESIDUAL3:
            return NB_ROOT
        if c not in strict:
            return MS_UNCLASS
        return NB_NONTRADABLE if strict[c] == 1 else NB_OTHER

    lab_strict = np.array([_strict(c) for c in code_arr])
    schemes["ms_strict"] = ([NB_NONTRADABLE, NB_OTHER, MS_UNCLASS, NB_ROOT],
                            lab_strict,
                            "Mian-Sufi non-tradable, strict (20 industries)")

    IC = {0: "Other", 1: "Tradable", 2: "Non-tradable (indcat)",
          3: "Construction", -1: MS_UNCLASS}
    lab_ic = np.array([NB_ROOT if int(c) >= RESIDUAL3
                       else IC[indcat.get(int(c), -1)] for c in code_arr])
    schemes["ms_indcat"] = ([IC[k] for k in (2, 1, 3, 0, -1)] + [NB_ROOT],
                            lab_ic,
                            "Mian-Sufi indcat (non-tradable = 26 industries)")

    # --- Rajan-Zingales ---------------------------------------------------
    rz = pd.read_csv(REF_DIR / "rz_naics4_extfin.csv", comment="#")
    rzm = rz[rz["mfg"] == 1]
    efd = dict(zip(rzm["naics4"].astype(int), rzm["extfin_all"].astype(float)))
    nis = dict(zip(rzm["naics4"].astype(int), rzm["n_isic"].astype(int)))
    mfg = {int(c) for c in genuine if int(c) // 100 in (31, 32, 33)}

    # THE CUT YEAR IS THE WINDOW'S, NOT A SHARED CONSTANT. The cuts are
    # employment-weighted and manufacturing is 25% of employment in 1975 against
    # 15.8% in 1998, so sharing them would leave the pre window's quartiles badly
    # unbalanced. The cost is that a quartile is a different set of industries in
    # the two column groups, which the note and the panel line both say.
    base = pd.read_parquet(dc.EMP_OUT_DIR / RUNG / f"y{cut_year}.parquet",
                           columns=["code", "emp"]).groupby("code")["emp"].sum()
    have = sorted(efd)
    wq = np.array([float(base.get(c, 0.0)) for c in have])
    vq = np.array([efd[c] for c in have])
    o = np.argsort(vq)
    cdf = np.cumsum(wq[o]) / wq.sum()
    cuts = np.interp([0.25, 0.50, 0.75], cdf, vq[o])
    print(f"    Rajan-Zingales: {len(efd)} manufacturing NAICS4 with a value; "
          f"employment-weighted quartile cuts at {cut_year} "
          f"{np.array2string(cuts, precision=3)}")

    QL = ["EFD Q1 (least dependent)", "EFD Q2", "EFD Q3",
          "EFD Q4 (most dependent)"]

    def rz_bin(c, single_only=False):
        c = int(c)
        if c >= RESIDUAL3:
            return NB_ROOT
        if c not in mfg:
            return RZ_OUTSIDE
        if c not in efd:
            return RZ_UNCOVERED
        if single_only and nis[c] > 1:
            return RZ_POOLED
        return QL[int(np.searchsorted(cuts, efd[c], side="right"))]

    schemes["rz_quartile"] = (QL + [RZ_UNCOVERED, RZ_OUTSIDE, NB_ROOT],
                              np.array([rz_bin(c) for c in code_arr]),
                              "Rajan-Zingales external finance dependence, "
                              "manufacturing only")
    schemes["rz_single_isic"] = (QL + [RZ_POOLED, RZ_UNCOVERED, RZ_OUTSIDE,
                                       NB_ROOT],
                                 np.array([rz_bin(c, True) for c in code_arr]),
                                 "Rajan-Zingales, cells fed by a single ISIC "
                                 "industry")
    return schemes, cuts


# ------------------------------------------------------- the decomposition --

def _silent_check(ok, label, detail=""):
    """31's `check` prints a line per call; inside the year loop that is 19
    identical lines per horizon. This keeps the assertion and drops the line."""
    if not ok:
        raise SystemExit(f"FAIL: {label} - {detail}")
    return True


def decompose(frame, emp, ccg_arr, code_arr, schemes, h):
    """Margin pieces, and the entry piece split by bin, over the frame's rows.

    Accumulated a year at a time straight into county-by-bin bincounts, so the
    939,603-cell support is never concatenated across years into a long array.
    The weight is normalised by the COUNTY's midpoint denominator and the growth
    guard is `den != 0`, never `den > 0`, exactly as 28_table_d1.cell_outcomes -
    a `> 0` guard zeroes midpoint growth on EFSY's negative cells while leaving
    their weights intact and the reconstruction then fails at 7.4e-4.
    """
    n = len(frame)
    units = np.unique(ccg_arr)
    pos_of = {int(c): i for i, c in enumerate(units)}
    cid = np.searchsorted(units, ccg_arr)

    keys = list(schemes)
    bins = {k: schemes[k][0] for k in keys}
    bidx = {k: np.array([bins[k].index(b) for b in schemes[k][1]], np.int64)
            for k in keys}

    pieces = {m: np.zeros(n) for m in MARGINS}
    binned = {k: np.zeros((n, len(bins[k]))) for k in keys}
    cur_c = np.zeros(n)
    lag_c = np.zeros(n)
    n_cells = {m: 0 for m in MARGINS}
    n_entry_bin = {k: np.zeros(len(bins[k]), np.int64) for k in keys}

    first_year = int(frame["year"].min())
    for y, sub in frame.groupby("year", sort=True):
        cur, lag = emp[y], emp[y - h]
        cy_of_unit = np.full(len(units), -1, np.int64)
        cy_of_unit[[pos_of[int(c)] for c in sub["ccg"]]] = sub.index.to_numpy()
        cy = cy_of_unit[cid]
        ok = cy >= 0
        cy, cu, la = cy[ok], cur[ok], lag[ok]

        cur_c += np.bincount(cy, weights=cu, minlength=n)
        lag_c += np.bincount(cy, weights=la, minlength=n)
        den = cu + la
        denc = np.bincount(cy, weights=den, minlength=n)
        if not (denc[np.unique(cy)] != 0).all():
            raise SystemExit(f"FAIL: A.1.6 a county-year in {y} has a zero "
                             f"midpoint denominator at h={h}")
        # w * mpgr = 2 (E_cmt - E_cm,t-h) / (E_ct + E_c,t-h), with both the
        # growth guard and the weight guard collapsing to `den != 0`.
        contrib = np.where(den != 0.0, 2.0 * (cu - la) / denc[cy], 0.0)

        # 31's partition prints its exhaustiveness check on every call, so the
        # first year prints and the remaining eighteen only assert. Swapping
        # w22.check rather than skipping the call keeps the check itself on
        # every year, which is the part that matters.
        saved = m31.check
        if y != first_year:
            m31.check = _silent_check
        try:
            masks = partition(cu, la)
        finally:
            m31.check = saved

        for m in MARGINS:
            msk = masks[m]
            pieces[m] += np.bincount(cy[msk], weights=contrib[msk], minlength=n)
            n_cells[m] += int(msk.sum())

        e = masks["entry"]
        for k in keys:
            b = bidx[k][ok][e]
            nb = len(bins[k])
            flat = np.bincount(cy[e] * nb + b, weights=contrib[e],
                               minlength=n * nb)
            binned[k] += flat.reshape(n, nb)
            n_entry_bin[k] += np.bincount(b, minlength=nb)

    den_c = cur_c + lag_c
    mpgr_c = 2.0 * (cur_c - lag_c) / den_c
    return dict(pieces=pieces, binned=binned, bins=bins, mpgr_c=mpgr_c,
                cur_c=cur_c, lag_c=lag_c, n_cells=n_cells,
                n_entry_bin=n_entry_bin)


def coverage_rows(emp, code_arr, schemes, h, win, rows):
    """G3, the coverage assertion, per window.

    This is the object producer 69's docstring asks for when it says the 1975-97
    EFSY window "needs its own coverage assertion if WS2 reaches back before
    1998". It is NOT a licence to drop anything: every code already sits in a
    named bin by the time this runs (A.3.1), and what is gated here is how much
    employment those named residual bins carry, against a ceiling stated in the
    constants block before any coefficient is read.
    """
    lab = schemes["ms_strict"][1]
    is_root = code_arr >= RESIDUAL3
    is_unc = lab == MS_UNCLASS
    worst_root, worst_unc, y_root, y_unc = 0.0, 0.0, None, None
    for y, v in emp.items():
        tot = float(np.abs(v).sum())
        if tot == 0.0:
            continue
        r = float(np.abs(v[is_root]).sum()) / tot
        u = float(np.abs(v[is_unc]).sum()) / tot
        if r > worst_root:
            worst_root, y_root = r, y
        if u > worst_unc:
            worst_unc, y_unc = u, y

    out = True
    for name, share, tol, yr, what in (
            ("G3a unresolved-root employment share", worst_root,
             MAX_ROOT_SHARE, y_root, NB_ROOT),
            ("G3b Mian-Sufi-unclassified employment share", worst_unc,
             MAX_UNCLASS_SHARE, y_unc, MS_UNCLASS)):
        ok = bool(share <= tol)
        where = f"worst year {yr}: {100 * share:.3f}%" if yr is not None else \
            "0.000% in every year of the window"
        detail = (f"{where} of employment, reported in the named bin "
                  f"'{what}', ceiling {100 * tol:.1f}%")
        rows.append(dict(window=win, horizon=h, gate=name, value=share, tol=tol,
                         ok=int(ok), detail=detail))
        print(f"    [{'ok ' if ok else 'FAIL'}] {name}: {detail}")
        out = out and ok
    return out


def gates(frame, D, h, win, rows):
    """G1-G2. Every identity is checked BEFORE a single coefficient is fit."""
    def rec(name, value, tol, detail):
        ok = bool(value <= tol)
        rows.append(dict(window=win, horizon=h, gate=name, value=value, tol=tol,
                         ok=int(ok), detail=detail))
        print(f"    [{'ok ' if ok else 'FAIL'}] {name}: {detail}")
        return ok

    # G1 the cell sums ARE the locked panel's employment on the surviving rows.
    # Producer 11 aggregates the county rung FROM these cells, so this holds by
    # construction - but the locked panel then applies a zero-employment county
    # screen and stage 04 applies E_b > 0 & E_t > 0, so the identity has to be
    # re-checked on the rows that survived rather than assumed through them.
    e_t = frame["E_t"].to_numpy(np.float64)
    e_b = frame["E_b"].to_numpy(np.float64)
    g1 = max(float(np.abs(D["cur_c"] - e_t).max() / max(e_t.max(), 1.0)),
             float(np.abs(D["lag_c"] - e_b).max() / max(e_b.max(), 1.0)))
    ok1 = rec("G1 cells sum to the locked panel employment", g1, TOL_LEVEL,
              f"max relative gap {g1:.2e} over {len(frame):,} county-years, "
              f"both endpoints")

    # G2 the margins reconstruct county midpoint growth, and the bins
    # reconstruct the entry margin.
    y = frame["y_mpgr"].to_numpy(np.float64)
    g2a = float(np.abs(sum(D["pieces"].values()) - y).max())
    ok2 = rec("G2a the margins sum to county midpoint growth", g2a,
              TOL_IDENTITY,
              f"max |sum of margins - y_mpgr| = {g2a:.2e}")
    g2b = 0.0
    for k, B in D["binned"].items():
        g2b = max(g2b, float(np.abs(B.sum(axis=1) - D["pieces"]["entry"]).max()))
    ok2b = rec("G2b every bin scheme sums to the entry margin", g2b,
               TOL_IDENTITY,
               f"max |sum of bins - entry| = {g2b:.2e} over "
               f"{len(D['binned'])} schemes")

    return ok1 and ok2 and ok2b


# ------------------------------------------------------------- estimation --

def estimator(frame):
    """The locked frame's weights, fixed effects and clusters, once."""
    g = frame
    tot = g.groupby(["division", "year"])["E_b"].transform("sum")
    w = (g["E_b"] / tot).to_numpy(np.float64)
    w = w / w.mean()
    fe = pd.factorize(g["division"].astype(str) + "_"
                      + g["year"].astype(str))[0].astype(np.int64)
    nofe = np.zeros(len(g), np.int64)
    cl = pd.factorize(g["ccg"])[0].astype(np.int64)
    x = g["x_pos"].to_numpy(np.float64)      # NOT winsorized: locked estimation
    z = g["z"].to_numpy(np.float64)

    def fit(y, reduced=False):
        r = t28.fit2sls(y, z if reduced else x, z, w, fe, nofe, cl,
                        singleton=False)
        return dict(beta=r["beta"] * locked.SCALE, se=r["se"] * locked.SCALE,
                    F=r["F"], n=r["n"])
    return fit, w


def zero_fit(fit, y):
    """Fit, unless the outcome is identically zero on this window.

    A named residual bin can be genuinely empty on one side of the break -
    NB_ROOT and MS_UNCLASS both are, post-1998 - and 2SLS on an all-zero outcome
    returns a degenerate beta 0 with se 0, which `fmt` would then star. The bin
    still has to appear, because the two column groups are aligned row for row
    and because dropping it would break the tiling A.3.1 requires. So it is
    reported as an exact zero with no standard error, exactly like the
    `always zero` margin row above it.

    `n` is the row count, not zero: the county-years are all there and all enter
    the regression, it is the OUTCOME that is degenerate. Reporting n = 0 would
    read as a dropped sample, which is exactly what A.3.1 is about.
    """
    if float(np.abs(y).max()) == 0.0:
        return dict(beta=0.0, se=np.nan, F=np.nan, n=len(y))
    return fit(y)


def window_context(h, coeff):
    """The locked full-span county coefficient, computed independently.

    THIS IS NOW A TIE-OUT TARGET, NOT CONTEXT. It used to be the only full-span
    number in the file, because no window covered the whole panel. The `full`
    window does, so this regression and that window's `TOTAL y_mpgr` are the
    same estimand on the same rows reached by two different code paths - this
    one reads the panel straight and applies no window filter at all, that one
    goes through load_frame, the cell decomposition and the gates. Gate G6
    asserts they agree, which is the check that "full span" means the locked
    sample and not some other pooled thing.

    It is kept under its own `window` key, `lockedspan`, so the two never
    collide on the `window` column of entry_bins_coeff.csv.
    """
    full = pd.read_parquet(PIPE / "analysis_panel.parquet")
    full = full[full["h"] == h].sort_values(["ccg", "year"]).reset_index(drop=True)
    f2, _ = estimator(full)
    r = f2(full["y_mpgr"].to_numpy(np.float64))
    print(f"\n  tie-out target, locked full span read straight from the panel: "
          f"beta {r['beta']:8.4f} ({r['se']:.4f}), "
          f"F {r['F']:5.1f}, n {r['n']:,}")
    coeff.append(dict(window="lockedspan", horizon=h, scheme="window_context",
                      bin="full locked span (tie-out)", spec="2sls", **r,
                      n_cells=np.nan, share=np.nan))
    del full
    return r


def straddle_rows(frame, h):
    """The county-years whose base period is before the break and outcome after.

    Only the pooled window has any: `pre` stops at t <= 1997 and `post` starts
    at t - h >= 1998, so both exclude exactly these rows and neither has to
    think about them. In the pooled window they are t = 1998 .. 1997 + h, which
    on the balanced locked panel is exactly h x N_COUNTIES rows.
    """
    y = frame["year"].to_numpy(np.int64)
    return (y - h < w22.NAICS_BREAK) & (y >= w22.NAICS_BREAK)


def run_horizon(h, win, args, coeff, ident, target=None):
    lo, hi = WINDOWS[win]
    print(f"\n{'=' * 80}\n  h = {h}   window {win} ({lo}-{hi})\n{'=' * 80}")
    frame = load_frame(h, win)
    years = sorted(set(frame["year"]) | set(frame["year"] - h))
    print(f"    employment files opened: {years[0]}-{years[-1]} ({len(years)})")
    emp, ccg_arr, code_arr = load_cells(years)
    keep = np.isin(ccg_arr, frame["ccg"].unique())
    if not keep.all():
        emp = {y: v[keep] for y, v in emp.items()}
        ccg_arr, code_arr = ccg_arr[keep], code_arr[keep]
        print(f"    cells restricted to the frame's counties: "
              f"{int(keep.sum()):,} of {len(keep):,} rows per year")

    schemes, cuts = bin_schemes(code_arr, RZ_CUT_YEAR[win])
    D = decompose(frame, emp, ccg_arr, code_arr, schemes, h)
    print("\n  CONSTRUCTION GATES")
    ok = gates(frame, D, h, win, ident)
    ok = coverage_rows(emp, code_arr, schemes, h, win, ident) and ok
    if not ok and not args.force:
        raise SystemExit("FAIL: a construction gate failed; no coefficient is "
                         "reported. Rerun with --force only to inspect.")

    fit, w = estimator(frame)
    print("\n  ESTIMATION  2SLS, x_pos instrumented by z, division x year FE, "
          "employment-share weights, county clusters")

    tot = fit(frame["y_mpgr"].to_numpy(np.float64))
    print(f"    county y_mpgr total: beta {tot['beta']:.4f} "
          f"({tot['se']:.4f}), first-stage F {tot['F']:.1f}, n {tot['n']:,}")
    coeff.append(dict(window=win, horizon=h, scheme="county",
                      bin="TOTAL y_mpgr", spec="2sls", **tot,
                      n_cells=np.nan, share=1.0))

    # G6, the pooled window only: it has to BE the locked sample, and its county
    # coefficient has to reproduce the one window_context reads straight off the
    # panel. Two code paths, one estimand, one number. A real gate - if these
    # disagree, "full span" is not what the column header says it is.
    if win == "full" and target is not None:
        n_ok = len(frame) == locked.N_COUNTY[h]
        rel = abs(tot["beta"] - target["beta"]) / max(abs(target["beta"]), 1e-12)
        ok6 = bool(n_ok and rel <= locked.RTOL_COEF)
        det = (f"n {len(frame):,} vs locked.N_COUNTY {locked.N_COUNTY[h]:,}; "
               f"beta {tot['beta']:.6f} vs the panel-direct "
               f"{target['beta']:.6f}, relative gap {rel:.2e}, "
               f"tol {locked.RTOL_COEF:.0e}")
        ident.append(dict(window=win, horizon=h,
                          gate="G6 the pooled window is the locked full span",
                          value=rel, tol=locked.RTOL_COEF, ok=int(ok6),
                          detail=det))
        print(f"    [{'ok ' if ok6 else 'FAIL'}] G6 the pooled window is the "
              f"locked full span: {det}")
        if not ok6 and not args.force:
            raise SystemExit("FAIL: G6; the pooled window is not the locked "
                             "full span")

    marg = {}
    for m in MARGINS:
        r = zero_fit(fit, D["pieces"][m])
        marg[m] = r["beta"]
        coeff.append(dict(window=win, horizon=h, scheme="margin", bin=m,
                          spec="2sls", **r, n_cells=D["n_cells"][m],
                          share=r["beta"] / tot["beta"]))
        print(f"      {m:14s}{r['beta']:11.4f} ({r['se']:.4f})"
              f"{100 * r['beta'] / tot['beta']:8.1f}% of the total")
    gap = abs(sum(marg.values()) - tot["beta"])
    ident.append(dict(window=win, horizon=h,
                      gate="G4 margin betas add to the total",
                      value=gap, tol=TOL_ADDITIVITY, ok=int(gap < TOL_ADDITIVITY),
                      detail=f"|sum - total| = {gap:.2e}"))
    print(f"    [{'ok ' if gap < TOL_ADDITIVITY else 'FAIL'}] "
          f"G4 the margin betas add to the total: |gap| = {gap:.2e}")

    # D1, the pooled window only. REPORTED, NOT GATED. Pooling means some
    # county-years compare a SIC-bridged cell at t-h to a native-NAICS cell at
    # t, so part of what the entry indicator calls entry there is the 1998
    # measurement change rather than an establishment appearing. A.3.1 forbids
    # dropping that mass and the pooled estimate is the thing being asked for,
    # so the honest move is to size it and put the number in the audit file and
    # the note. There is no pass ceiling because there is no action a failure
    # would license: a straddle is intrinsic to a pooled window, and inventing
    # a threshold for it would be theatre.
    if win == "full":
        s = straddle_rows(frame, h)
        e = np.abs(D["pieces"]["entry"])
        share = float(e[s].sum()) / float(e.sum()) if e.sum() else 0.0
        det = (f"{int(s.sum()):,} of {len(frame):,} county-years have "
               f"t-h < {w22.NAICS_BREAK} <= t (outcome years "
               f"{w22.NAICS_BREAK}-{w22.NAICS_BREAK - 1 + h}); they carry "
               f"{100 * share:.2f}% of the window's entry-margin mass. In "
               f"neither one-sided window. Reported, not gated: the cells are "
               f"SIC-bridged at t-h and native NAICS at t.")
        ident.append(dict(window=win, horizon=h,
                          gate="D1 break-straddling mass (diagnostic)",
                          value=share, tol=np.nan, ok=1, detail=det))
        print(f"    [rep] D1 break-straddling mass: {det}")

    for k, B in D["binned"].items():
        print(f"\n    {schemes[k][2]}")
        bsum = 0.0
        # EVERY bin is recorded, including one that is empty on this side of the
        # break. The row has to exist in both column groups for them to line up,
        # and A.3.1's tiling is the reason it exists at all.
        for j, b in enumerate(D["bins"][k]):
            r = zero_fit(fit, B[:, j])
            rf = zero_fit(lambda y: fit(y, reduced=True), B[:, j])
            bsum += r["beta"]
            coeff.append(dict(window=win, horizon=h, scheme=k, bin=b,
                              spec="2sls", **r,
                              n_cells=int(D["n_entry_bin"][k][j]),
                              share=r["beta"] / marg["entry"]))
            coeff.append(dict(window=win, horizon=h, scheme=k, bin=b, spec="rf",
                              **rf, n_cells=int(D["n_entry_bin"][k][j]),
                              share=np.nan))
            print(f"      {b:28s}{r['beta']:11.4f} ({r['se']:.4f})"
                  f"{100 * r['beta'] / marg['entry']:8.1f}% of entry"
                  f"{int(D['n_entry_bin'][k][j]):>14,} entrant cells")
        g = abs(bsum - marg["entry"])
        ident.append(dict(window=win, horizon=h,
                          gate=f"G5 {k} bin betas add to entry",
                          value=g, tol=TOL_ADDITIVITY,
                          ok=int(g < TOL_ADDITIVITY),
                          detail=f"|sum - entry| = {g:.2e}, entry "
                                 f"{marg['entry']:.6f}"))
        print(f"      [{'ok ' if g < TOL_ADDITIVITY else 'FAIL'}] "
              f"the bin betas add to the entry beta: |gap| = {g:.2e}")

    # Q9's inadmissible comparison, run on purpose and labelled as the
    # benchmark: log and standard growth do not decompose by cell at all, so
    # they can only ever be the county number.
    for oc in ("y_log", "y_std"):
        r = fit(frame[oc].to_numpy(np.float64))
        coeff.append(dict(window=win, horizon=h, scheme="benchmark", bin=oc,
                          spec="2sls", **r, n_cells=np.nan, share=np.nan))
    return frame, cuts


# ---------------------------------------------------------------- exhibit --

def tex_escape(s):
    return s.replace("_", r"\_").replace("%", r"\%")


def fmt(b, se):
    if not np.isfinite(b):
        return "--", ""
    t = abs(b / se) if se and np.isfinite(se) and se > 0 else 0.0
    star = "***" if t > 2.576 else "**" if t > 1.96 else "*" if t > 1.645 else ""
    return f"{b:.3f}{star}", f"({se:.3f})" if np.isfinite(se) else ""


def write_table(coeff, horizons, windows):
    """One panel per bin scheme, one column group per window PRINTED.

    THE EXHIBIT PRINTS TABLE_WINDOWS, PRE-1998 AND THE POOLED FULL SPAN, SIDE BY
    SIDE. All three windows are estimated on every run and all three are in
    entry_bins_coeff.csv under the `window` column; what the .tex spends its
    width on is the one-sided pre window against the pooled one. The post-1998
    window stays in the CSV, and the note quotes the county totals of all three
    so the sign flip across the break is legible from the page alone.

    THE ROWS LINE UP ACROSS THE COLUMN GROUPS BECAUSE THE BIN LISTS ARE
    WINDOW-INDEPENDENT (see bin_schemes) and because the pooled window shares
    pre's RZ cut year, so a Panel C row is the same set of industries left and
    right. That is the whole reason RZ_CUT_YEAR["full"] is 1975 rather than a
    cut of its own.

    The layout is general in the number of windows: pass one and it drops the
    grouped header, pass three and it emits three groups. That is what
    `--window` controls.

    PANEL ORDER IS BY MEASURE, NOT BY HEADLINE-THEN-ROBUSTNESS. Each measure's
    headline definition and its robustness variant sit together - Mian-Sufi
    twice, then Rajan-Zingales twice - so a reader comparing the two
    non-tradable definitions, or the two RZ samples, reads adjacent panels
    instead of skipping one.

    THERE IS NO PANEL RF. The reduced form on z is still estimated and still
    written to entry_bins_coeff.csv under spec="rf".
    """
    d = pd.DataFrame(coeff)
    PANELS = [("ms_strict", "A", "Mian--Sufi non-tradables, strict definition"),
              ("ms_indcat", "B", "Mian--Sufi \\texttt{indcat}, the 26-industry "
                                 "non-tradable definition (robustness)"),
              ("rz_quartile", "C", "Rajan--Zingales external finance "
                                   "dependence, manufacturing only"),
              ("rz_single_isic", "D", "Rajan--Zingales, single-ISIC cells only "
                                      "(robustness)")]
    hs = list(horizons)
    ws = list(windows)
    ncol = len(hs) * len(ws)

    colspec = "l" + "@{\\hskip 1.5em}".join(["c" * len(hs)] * len(ws))
    # The two-group table runs 21pt over \linewidth at 1in margins, so it is
    # wrapped the way every other exhibit in this project is wrapped
    # (50_lp_dynamics, 22_bhj_assumption1, 43/44/45 in step_2). `max width` only
    # shrinks a table that is too wide, so a one-window run renders unchanged.
    # THE CONSUMING PREAMBLE MUST LOAD adjustbox; Results/preview_tables.tex and
    # the step_2 preview wrappers already do.
    L = [r"\adjustbox{max width=\linewidth}{",
         r"\begin{tabular}{" + colspec + "}", r"\toprule"]
    if len(ws) > 1:
        # Only label the column groups when there is more than one to tell
        # apart. With a single window the label would be a title over the whole
        # table, which is the note's job, not a header row's.
        head, rule = [], []
        for i, wname in enumerate(ws):
            a = 2 + i * len(hs)
            head.append(r"\multicolumn{%d}{c}{%s}"
                        % (len(hs), WINDOW_LABEL[wname]))
            rule.append(r"\cmidrule(lr){%d-%d}" % (a, a + len(hs) - 1))
        L += ["& " + " & ".join(head) + r" \\", "".join(rule)]
    L += ["& " + " & ".join(f"$h={h}$" for h in hs * len(ws)) + r" \\",
          r"\midrule"]

    def block(sub, order):
        out = []
        for b in order:
            s = sub[sub["bin"] == b]
            if s.empty:
                continue
            bt, st = [], []
            for wname in ws:
                for h in hs:
                    r = s[(s["window"] == wname) & (s["horizon"] == h)]
                    if r.empty:
                        bt.append("--")
                        st.append("")
                        continue
                    a, c = fmt(float(r["beta"].iloc[0]), float(r["se"].iloc[0]))
                    bt.append(a)
                    st.append(c)
            out.append(f"{tex_escape(b)} & " + " & ".join(bt) + r" \\")
            out.append(" & " + " & ".join(st) + r" \\")
        return out

    tot = d[(d["scheme"] == "county") & (d["spec"] == "2sls")]
    mar = d[(d["scheme"] == "margin") & (d["spec"] == "2sls")]
    L.append(r"\multicolumn{%d}{l}{\emph{The county total and its four "
             r"margins}} \\" % (ncol + 1))
    L += block(tot, ["TOTAL y_mpgr"])
    L += block(mar, list(MARGINS))
    for key, tag, title in PANELS:
        sub = d[(d["scheme"] == key) & (d["spec"] == "2sls")]
        if sub.empty:
            continue
        L.append(r"\midrule")
        L.append(r"\multicolumn{%d}{l}{\textbf{Panel %s.} %s} \\"
                 % (ncol + 1, tag, title))
        if key.startswith("rz"):
            # Flagged at the panel so a quartile row is never read as a fixed
            # set of industries; the note carries why. KEEP THIS SHORT - it is a
            # \multicolumn spanning a narrow table and cannot wrap, so a full
            # sentence here overflows the tabular by ~180pt.
            cy = {RZ_CUT_YEAR[w] for w in ws}
            if len(ws) == 1:
                said = (r"Quartile cuts on %d employment; the post-1998 cuts "
                        r"differ." % RZ_CUT_YEAR[ws[0]])
            elif len(cy) == 1:
                # The printed groups share a cut year, so a row IS the same set
                # of industries in each - which is what makes them comparable.
                said = (r"Quartile cuts on %d employment in every column."
                        % cy.pop())
            else:
                said = r"Quartile cuts are employment-weighted within each window."
            L.append(r"\multicolumn{%d}{l}{\footnotesize %s} \\"
                     % (ncol + 1, said))
        L += block(sub, list(dict.fromkeys(sub["bin"])))
    L += [r"\bottomrule", r"\end{tabular}}"]
    return "\n".join(L) + "\n"


NOTE_TEMPLATE = r"""% Notes for Table_entry_bins_efsy.tex, produced by
% step_1/code/pipeline/30_entry_margin_bins.py. Paste or \input as the table note.
%
% THE TABLE CARRIES TWO COLUMN GROUPS. The left group is the PRE-1998 window,
% t-h >= 1975 and t <= 1997, outcome years 1980-1997. The right group is the
% POOLED FULL SPAN: the whole locked panel, no year restriction at all, outcome
% years 1980-2018 at h = 2 and 1982-2018 at h = 4, n = 119,262 / 116,204 /
% 113,146 county-years, which is locked.N_COUNTY exactly. A third window,
% 1998-2016, is estimated by the same producer on the same run and is NOT
% printed: it is in entry_bins_coeff.csv, selected by window == "post". The
% county totals of all three are quoted below so the comparison can be made
% without opening the file.
%
% THE ROWS ARE COMPARABLE ACROSS THE TWO GROUPS. The bin lists are
% window-independent by construction, and the pooled window uses PRE'S
% Rajan-Zingales cut year (1975) rather than one of its own, so a Panel C or D
% row is the same set of industries on both sides of the table. The price is
% that post-1998 cells inside the right-hand group are binned on 1975 cuts.
%
% The entry margin of county employment growth, split by an ex-ante
% classification of the industry. The dependent variable in every bin row is
%
%     Entry_ct(b) = sum_{m in b : E_cm,t-h <= 0, E_cmt > 0}
%                       2 E_cmt / (E_ct + E_c,t-h),
%
% the contribution of industries in bin b to county midpoint employment growth
% through cells that were empty at t-h and populated at t. Bins tile the
% four-digit NAICS universe, so within each panel the bin coefficients sum to
% the entry coefficient, and entry plus exit plus continuers plus always-zero
% sum to the county total. Both identities are enforced numerically and are
% reported in entry_bins_identities.csv.
%
% SPECIFICATION. 2SLS of the row's outcome on x_pos, the locked contract
% treatment, instrumented by z, the locked composition instrument. Division x
% year fixed effects, employment-share-within-division-year weights normalised
% to mean one, standard errors clustered on the county, no winsorization.
% Coefficients are scaled by 1e5. The weighting is producer 20's, the companion
% of division x year fixed effects; the locked AGGREGATION baseline instead
% weights within its parent rung, so the county coefficient here is not that
% table's beta_b either. The reduced form on z is estimated for every bin and
% written to entry_bins_coeff.csv under spec "rf"; it is not given a panel.
%
% WHY THERE IS A ONE-SIDED WINDOW AT ALL. Before 1998 CBP imputes NAICS from
% SIC and a cell is not the same object across the break. The two one-sided
% windows are kept strictly on one side of it - pre is t-h >= 1975 and t <=
% 1997, post is t-h >= 1998 and t <= 2016 - so neither straddles 1998 in the
% outcome year or in the base period. The pooled window straddles it by
% construction, which is the point of having both on the page: the left group
% is the clean-measurement estimate and the right group is the whole sample.
%
% THE COUNTY COEFFICIENT CHANGES SIGN ACROSS THE BREAK. The same specification
% gives, on TOTAL y_mpgr:
%
@TOTALS@
%
% So the bins in the left group decompose a POSITIVE and significant county
% total, the same bins on the post window decompose a small negative
% insignificant one, and the pooled group decomposes the average of the two
% eras that the locked baseline actually reports. Whether the flip is the
% sample period, the post-1998 composition of procurement, or the break itself
% is NOT settled by this exhibit. What the exhibit adds is which industries
% carry each; the CSV carries the same split for post.
%
% WHAT POOLING COSTS, MEASURED RATHER THAN ASSERTED. In the pooled window the
% county-years with t-h < 1998 <= t compare a SIC-bridged cell at the base
% period to a native-NAICS cell at the outcome year, so some of what the entry
% indicator counts as entry there is the 1998 measurement change and not an
% establishment appearing. Those county-years are in NEITHER one-sided window.
% There are h x 3,058 of them and they carry the following share of the pooled
% window's entry-margin mass:
%
@STRADDLE@
%
% Protocol A.3.1 forbids dropping them, so they are sized and reported here and
% in entry_bins_identities.csv as diagnostic D1 rather than gated: a straddle is
% intrinsic to a pooled window, and a pass threshold for it would license no
% action. Read the right-hand group with this in mind, and the left-hand group
% as the estimate that does not carry it.
%
% THE PRE-1998 YEARS CARRY MASS THE POST WINDOW DOES NOT, AND IT IS REPORTED,
% NOT DROPPED. EFSY before 1998 is SIC-bridged and mixes 2-, 3-, 4- and 6-digit
% padded codes. After producer 11's allocation a small residual stays on codes
% that could not be pushed to four digits (0.21-0.29 per cent of employment),
% and 0.36-1.51 per cent sits on four-digit codes outside Mian and Sufi's 294.
% Protocol A.3.1 forbids dropping either, so both get a named bin - "Unresolved
% NAICS root" and "Not in the Mian-Sufi 294" - which is why the bins still tile
% the universe and the additivity identity still holds. Together they are about
% 0.6 per cent of the entry coefficient at h = 4. Both are exactly empty after
% 1998, which is why the post rows of entry_bins_coeff.csv carry them as zeros.
% Their sizes are gated, at 1 and 5 per cent respectively, and recorded per
% window in entry_bins_identities.csv as G3a and G3b: that is the coverage
% assertion producer 69 asks for before WS2 reaches back before 1998. Both bins
% appear in both printed groups, because the pooled window contains the
% pre-1998 years; they are exactly zero in the post rows of the CSV.
%
% EVERY BIN IN THE LEFT GROUP IS AN ASSIGNMENT ONTO AN IMPUTED NAICS CODE, not a
% native one, because that whole window predates the break. The residual above
% measures what could not be imputed at all; it does not certify that what was
% imputed was imputed correctly. The post-1998 rows in the CSV do not share this
% caveat, the pooled group carries it on its pre-1998 years only, and that
% asymmetry is itself a reason the windows are not a clean before-and-after.
%
% BINS. Panels A and B use Mian and Sufi (2014): A their strict non-tradable
% definition, 20 industries, B their broader indcat category, 26 industries.
% Both add NAICS 7225 - EFSY reports 7225, which absorbs Mian and Sufi's 7221
% and 7222, and without it 8.6 per cent of employment would be unclassified.
% Panels C and D use Rajan and Zingales's (1998) external finance dependence,
% concorded from ISIC rev 2 through 1987 SIC to four-digit NAICS by
% step_1/code/py/71_concord_rz_to_naics4.py, in employment-weighted quartiles.
% THE QUARTILE CUTS. Both printed groups use y1975; the post window in the CSV
% uses y1998. The cuts are employment-weighted and manufacturing is 25 per cent
% of employment in 1975 against 15.8 per cent in 1998, so giving the post window
% its own cut year is what keeps its quartiles balanced. The pooled window
% deliberately does NOT take a cut year of its own: it shares pre's, so that the
% two printed groups bin identical industry sets and a row can be read straight
% across. The price is that "EFD Q3" in this table and "EFD Q3" in the post rows
% of the CSV are NOT the same set of industries, and the panels say so. Compare
% those as "the third quartile of external finance dependence in that window",
% never as "these particular industries".
% Canonical Rajan-Zingales is manufacturing only: it reaches 27 per cent of EFSY
% employment in 1976 and 15.8 per cent by 1997, falling to 9.5 per cent by 2016,
% so the two named residual bins - manufacturing industries with no value, and
% everything outside manufacturing - are reported rather than dropped. 54 of the
% 94 four-digit codes the concordance reaches draw on more than one ISIC industry
% and their value is a concordance-mass-weighted mean; Panel D restricts to the
% codes fed by a single ISIC industry, which carry 31 per cent of manufacturing
% employment.
%
% Log and standard growth appear in entry_bins_coeff.csv under scheme
% "benchmark". They are the inadmissible comparison and are run on purpose:
% neither decomposes into cell contributions at all, so neither can produce this
% table.
"""


def note(coeff, ident, horizons):
    """The table note, with the run's own numbers filled into it.

    TWO BLOCKS ARE GENERATED, NOT TYPED. The county totals, because the
    post-1998 window is the one thing a reader of this table cannot see for
    themselves - it is CSV-only - so a stale figure here would be a claim about
    a window the reader has no way to check against the page. And the D1
    straddle shares, because they are the size of the caveat that governs how
    the pooled column reads, and a caveat quoted at the wrong magnitude is worse
    than none. Everything else in the note is a fact about the construction and
    is written by hand.
    """
    d = pd.DataFrame(coeff)
    d = d[(d["scheme"] == "county") & (d["spec"] == "2sls")]
    hs = list(horizons)
    head = f"%   {'':24s}" + "".join(f"{'h = ' + str(h):>12s}" for h in hs)

    lines = [head]
    for w, lab in (("pre", "pre-1998 (left group)"),
                   ("post", "1998-2016 (CSV only)"),
                   ("full", "full span (right group)")):
        s = d[d["window"] == w]
        if s.empty:
            continue
        cells = []
        for h in hs:
            r = s[s["horizon"] == h]
            cells.append(f"{float(r['beta'].iloc[0]):+.2f}" if len(r) else "--")
        lines.append(f"%   {lab:24s}" + "".join(f"{c:>12s}" for c in cells))

    g = pd.DataFrame(ident)
    st = [head]
    if len(g):
        g = g[g["gate"].str.startswith("D1")]
        for lab, col in (("county-years", "n"), ("share of entry mass", "s")):
            cells = []
            for h in hs:
                r = g[g["horizon"] == h]
                if r.empty:
                    cells.append("--")
                elif col == "n":
                    cells.append(f"{h * locked.N_COUNTIES:,}")
                else:
                    cells.append(f"{100 * float(r['value'].iloc[0]):.2f}%")
            st.append(f"%   {lab:24s}" + "".join(f"{c:>12s}" for c in cells))
    if len(st) == 1:
        st.append("%   (the pooled window was not estimated on this run)")

    return (NOTE_TEMPLATE.replace("@TOTALS@", "\n".join(lines))
            .replace("@STRADDLE@", "\n".join(st)))


def memo(horizons, wins, nrows, cuts):
    win_spec = "\n".join(
        f"  {w:4s}  `t - h >= {WINDOWS[w][0]}` and `t <= {WINDOWS[w][1]}`; "
        f"RZ quartile cuts on `y{RZ_CUT_YEAR[w]}` at "
        f"{np.array2string(cuts[w], precision=3)}; rows "
        + ", ".join(f"h={h}: {nrows[(w, h)]:,}" for h in horizons)
        for w in wins)
    return f"""# WS2 step 4: the estimand, and the eleven pre-code questions

Written by `step_1/code/pipeline/30_entry_margin_bins.py` before any
coefficient in `Table_entry_bins_efsy.tex` was read.
`INVARIANCE_TEST_PROTOCOL.md` requires this file to exist for a new estimand.

## The estimand

Baseline unit `i` = `ccg`, a county group. Cell `m` = (`ccg`, four-digit NAICS
2012 code), from producer 11's `{RUNG}` rung. For horizon `h` and a bin `b` of
the NAICS universe:

    Entry_ct(b) = sum_{{m in b : E_cm,t-h <= 0, E_cmt > 0}}
                      2 E_cmt / (E_ct + E_c,t-h)

    Entry_ct(b) = beta_b x_pos_ct + delta_{{division x year}} + eps_ct,
                  x_pos instrumented by z_ct

`h` in {list(horizons)}; weights = employment share within division-year,
mean-normalised; clusters = county; no winsorization; coefficients scaled by
`locked.SCALE`. The estimand is estimated **once per window**:

{win_spec}

`Table_entry_bins_efsy.tex` prints **{" and ".join(f"`{w}`" for w in TABLE_WINDOWS)}**
as two column groups. All three windows are estimated on every run and all three
are in `entry_bins_coeff.csv` under the `window` column; the table note quotes
the county totals of each so the comparison can be made without opening the CSV.

`pre` and `post` sit strictly on one side of the 1998 NAICS break, in both the
outcome year and the base period, because before 1998 CBP imputes NAICS from
SIC and a cell is not the same object across it. `full` is the whole locked
panel and straddles the break on purpose - it is the pooled estimate, and its
county total reproduces the locked full-span coefficient (gate G6). The
county-years it contains that neither one-sided window does, those with
`t - h < 1998 <= t`, compare a SIC-bridged base period to a native-NAICS outcome
year; they are sized per horizon as diagnostic **D1** and reported rather than
dropped (A.3.1).

## What changes and what it invalidates

Nothing standing. `locked.py` is not edited, the locked baseline and its
tie-out are untouched, and producer 31's pre-1998 margin decomposition is
untouched. The new objects are the binned entry contributions and the two
windows. The county total reported for each window is the locked estimand on a
shorter sample and is **not** the locked baseline number.

**The estimand did not change when the pre window was added, and it did not
change when the pooled window was added.** The formula, the right-hand side, the
fixed effects, the weights, the clusters, the no-winsorization rule and the
scale are all as they were. What was added each time is a SAMPLE. The proof that
nothing leaked, in both cases, is a reproduction: the pre-window edit left every
post-1998 coefficient identical because its two new residual bins are exactly
empty after 1998, and the pooled-window edit touches neither one-sided window's
rows, so **every `pre` and `post` coefficient reproduces the previous run digit
for digit**. If one moves, the edit did something it was not supposed to.

The pooled window has its own reproduction check, gate **G6**: its county
`TOTAL y_mpgr` is the same estimand on the same rows as `window_context`, which
reads the panel straight with no window filter, so the two must agree at
`locked.RTOL_COEF` and the row count must equal `locked.N_COUNTY`. That is what
makes "full span" a verified claim rather than a column header.

## The eleven questions

1. **Baseline and cell.** `i` = `ccg`; `m` = (`ccg`, NAICS4). Rows per window
   and horizon are listed with the windows above.
2. **Does `S_i = sum_m S_im` hold exactly?** Yes, and it is checked, not
   assumed. Producer 11 aggregates the county rung *from* these cells, so the
   identity holds by construction; the locked panel then applies a
   zero-employment county screen and stage 04 applies `E_b > 0 & E_t > 0`, so
   gate **G1** re-checks the cell sums against `E_b` and `E_t` on the rows that
   survived, at relative tolerance {TOL_LEVEL:.0e}. No cap and no
   transformation is applied at one level and not the other.
3. **Treatment and its level.** `x_pos_ct`, the locked contract treatment,
   varies at county x year. Subscript `ct`.
4. **Is the subscript identical in both regressions?** Yes. The right-hand side
   is byte-identical across the county regression and every bin regression;
   only the left-hand side is re-partitioned. This is a disaggregation of the
   outcome, not an aggregation test.
5. **Do the fixed effects operate where the treatment varies?** No. Fixed
   effects are division x year; the treatment varies at county x year, so it
   is not absorbed.
6. **Growth rate and weighting.** Midpoint growth with midpoint weights
   `w_cmt = (E_cmt + E_cm,t-h)/(E_ct + E_c,t-h)`, the A.1.5 admissible
   headline. The guard on the growth rate is `den != 0`, never `den > 0`: a
   `> 0` guard zeroes midpoint growth on EFSY's negative cells while leaving
   their weights intact and the reconstruction then fails at 7.4e-4.
7. **Is the baseline sample identical at every rung, and are exclusions
   admissible?** Yes, and there are no exclusions. Every county-year appears in
   every bin regression; a county with no entrants in bin `b` enters with
   `Entry_ct(b) = 0` rather than being dropped. Per A.3.1 an industry that
   cannot be classified goes to a *named* bin - `{RZ_UNCOVERED}`,
   `{RZ_OUTSIDE}`, `{RZ_POOLED}`, and, for the pre-1998 window, `{MS_UNCLASS}`
   and `{NB_ROOT}` - never to a drop, which is why the bins tile the universe
   and gate **G2b** can hold on both sides of the break and on the pooled
   window. The three windows are separate samples, not three rungs of one
   ladder; within each, every baseline unit appears in every bin regression.
   The pooled window is not the union of the other two: it also contains the
   county-years that straddle 1998 and the 2017-18 tail, which is why its row
   count is checked against `locked.N_COUNTY` rather than against a sum.
8. **Have the construction checks passed, and is the channel being measured
   before anything is estimated?** Gates G1, G2a, G2b and the G3 coverage
   assertion run to completion *per window* and the producer refuses to fit a
   single coefficient if any of them fails. G3a and G3b are the coverage
   assertion producer 69's docstring asks for before WS2 reaches back before
   1998: they cap the unresolved-root and Mian-Sufi-unclassified employment
   shares at {MAX_ROOT_SHARE:.0%} and {MAX_UNCLASS_SHARE:.0%}, both stated
   before the run. The additivity gates G4 and G5 then check the estimates
   themselves, again per window, and G6 checks that the pooled window is the
   locked full span. D1 is a reported diagnostic, not a gate, for the reason
   given above.
9. **Which inadmissible specifications are run on purpose?** Log growth
   (`y_log`) and standard growth (`y_std`) at the county level, recorded in
   `entry_bins_coeff.csv` under scheme `benchmark` and labelled in the table
   note as the benchmark rather than the answer. Neither decomposes into cell
   contributions at all, which is the point of showing them.
10. **The payoff exhibit.** `Table_entry_bins_efsy.tex`. The county regression
    reports one entry coefficient; the exhibit shows whether that coefficient
    is concentrated in non-tradable industries or in externally-finance-
    dependent ones. No county-level specification can see this.
11. **Does every rung of the ladder pass Part A on its own terms?** The exhibit
    is not a ladder from a published design. Its rows are one estimand at four
    bin schemes, in three windows. The two Rajan-Zingales panels differ only in
    whether concordance-pooled cells are included - a sample restriction on a
    fixed estimand, stated in the note - and the two Mian-Sufi panels only in
    which of the file's two non-tradable definitions is used, which producer 69
    requires a consumer to state.

## The finding that governs how this exhibit reads

The county coefficient this table decomposes **changes sign across the 1998
NAICS break**. The exhibit prints the pre-1998 side, where the total is positive
and significant, beside the pooled full span, which the pre-1998 years dominate;
the post-1998 side, where the total is small, negative and insignificant, is in
`entry_bins_coeff.csv`. The table note carries all three sets of county totals
for exactly this reason - a reader of the page alone would otherwise not know
the sign is window-dependent, and would read the pooled column as if it were a
statement about the whole period uniformly.

The reason the producer used to *estimate* only the post side was the
classifications, not the cells: producer 69's Mian-Sufi coverage assertion ran
1998-2016, and producer 31's reason for staying pre-1998 was the break itself.
Those two constraints were disjoint and the post side was chosen. Discharging
producer 69's assertion on 1975-97 (gates G3a and G3b here) removes the
constraint, so both sides are now estimated and both are in the CSV.

**What this still does not settle.** Having both sides separates "the two
windows disagree" from "the break broke the measurement", but it does not
adjudicate between them: the pre cells are SIC-bridged and the post cells are
native NAICS, so a difference between the windows is a difference in period
*and* in measurement at once. A clean adjudication needs a design that holds the
measurement fixed across the period, which this exhibit is not.

**The pooled window does not settle it either, and is not offered as doing so.**
It is the estimate on the whole locked sample, which is what the locked baseline
reports and therefore worth decomposing; it is not a reconciliation of the two
eras. It mixes SIC-bridged and native-NAICS cells inside one regression, and on
the county-years that straddle the break it mixes them inside a single
observation's own growth rate. D1 sizes that; nothing here removes it.

## Known limitations, stated here rather than discovered later

- Canonical Rajan-Zingales reaches 27% of EFSY employment in 1976, 15.8% in 1998
  and 9.5% in 2016. Panels C and D speak to a real share of the entry margin in
  the pre window and to under a tenth of it in the later years of the post
  window; the named residual bins carry the rest, in both.
- 54 of the 94 four-digit codes the RZ concordance reaches are fed by more than
  one ISIC industry; the employment-weighted mean spread of `extfin_all` among
  contributors is 0.33 against a range of -0.45 to 1.49. Panel D is the
  single-ISIC subset and is the honest read of Panel C's signal.
- The RZ quartile cuts are `y{w22.EFSY_FIRST_YEAR}` in both printed column
  groups and `y{w22.NAICS_BREAK}` for the post window, so `EFD Q3` in the table
  and `EFD Q3` in the CSV's post rows are not the same set of industries. Within
  the table a quartile row IS the same set of industries in both groups, which
  is why the pooled window was given pre's cut year rather than one of its own.
- The pooled window contains `h` x {locked.N_COUNTIES:,} county-years whose base
  period is before the 1998 break and whose outcome year is after it. Their
  growth rate compares a SIC-bridged cell to a native-NAICS cell, so part of
  what the entry indicator counts there is the measurement change. They are in
  neither one-sided window, they cannot be dropped (A.3.1), and their share of
  the pooled entry-margin mass is recorded per horizon as D1 in
  `entry_bins_identities.csv` and quoted in the table note.
- NAICS 1997 to NAICS 2012 is four-digit truncation, gated in producer 71 at 5%
  of manufacturing employment and realised at 0.35% in the worst year of
  1975-2016 (1975; it was 0.26% on the post window alone).
- The pre-1998 bins are assignments onto NAICS codes EFSY imputed from SIC. G3a
  measures what could not be imputed at all (0.21-0.29% of employment); it does
  not certify that what was imputed was imputed correctly.
"""


def row_accounting(h, wins, nrows):
    """Where the pooled window's rows come from, printed once per horizon.

    The pooled window is NOT the union of the two one-sided ones. It also holds
    the county-years that straddle the break, which both of those exclude by
    construction, and the 2017-18 tail, which `post` stops short of. On the
    balanced locked panel the arithmetic is exact, so printing it is the cheapest
    way to make "full span" auditable from the log rather than from the code.
    """
    if not {"pre", "post", "full"} <= set(wins):
        return
    n = locked.N_COUNTIES
    pre, post, full = (nrows[(w, h)] for w in ("pre", "post", "full"))
    straddle, tail = h * n, 2 * n
    parts = [("pre  (t <= 1997)", pre), ("post (t-h >= 1998)", post),
             (f"straddle ({w22.NAICS_BREAK}-{w22.NAICS_BREAK - 1 + h})", straddle),
             ("tail (2017-18)", tail)]
    print(f"\n  ROW ACCOUNTING, h = {h}   ({n:,} counties x years)")
    for lab, v in parts:
        print(f"    {lab:28s}{v:>10,}")
    got = sum(v for _, v in parts)
    ok = got == full
    print(f"    {'-' * 38}\n    {'sum':28s}{got:>10,}   "
          f"[{'ok ' if ok else 'FAIL'}] full window has {full:,}")
    if not ok:
        raise SystemExit(f"FAIL: row accounting at h={h}: the pooled window's "
                         f"{full:,} rows are not pre + post + straddle + tail "
                         f"({got:,}); the panel is not the balanced locked one")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--horizon", type=int, nargs="+", default=list(locked.HORIZONS))
    ap.add_argument("--window", nargs="+", choices=list(WINDOW_ORDER),
                    default=list(WINDOW_ORDER),
                    help="which window(s) to estimate: the two sides of the "
                         "1998 NAICS break and the pooled full locked span; "
                         "the default is all three, and the .tex prints "
                         + " and ".join(TABLE_WINDOWS) + " side by side")
    ap.add_argument("--force", action="store_true",
                    help="report coefficients even if a construction gate "
                         "fails; for inspection only, never for an exhibit")
    args = ap.parse_args()
    t0 = time.time()
    wins = [w for w in WINDOW_ORDER if w in args.window]
    # The .tex prints the TABLE_WINDOWS that were estimated, and whatever was
    # estimated otherwise - so `--window post` alone still produces a table.
    tab = [w for w in TABLE_WINDOWS if w in wins] or wins

    print("=" * 80)
    print("  The entry margin of county employment growth, by ex-ante "
          "industry bin")
    print("=" * 80)
    print(f"  Locked composition frame: x_pos instrumented by z, "
          f"{locked.FE} FE, employment-share weights, county clusters,")
    print(f"  no winsorization, coefficients x {locked.SCALE:.0e}.")
    for wname in wins:
        lo, hi = WINDOWS[wname]
        print(f"  Window {wname:4s}: t-h >= {lo}, t <= {hi}; "
              f"RZ quartile cuts on y{RZ_CUT_YEAR[wname]}; "
              f"{'PRINTED in the .tex' if wname in tab else 'CSV only'}.")
    print(f"  Cells: producer 11's {RUNG} rung. Margins: "
          f"31_margin_decomposition.partition, imported.")

    coeff, ident, nrows, cuts = [], [], {}, {}
    for h in args.horizon:
        target = window_context(h, coeff)
        for wname in wins:
            f, cu = run_horizon(h, wname, args, coeff, ident, target)
            nrows[(wname, h)] = len(f)
            cuts[wname] = cu
        row_accounting(h, wins, nrows)

    TAB_DIR.mkdir(parents=True, exist_ok=True)
    cf = pd.DataFrame(coeff)
    cf.to_csv(TAB_DIR / "entry_bins_coeff.csv", index=False, float_format="%.15g")
    idf = pd.DataFrame(ident)
    idf.to_csv(TAB_DIR / "entry_bins_identities.csv", index=False,
               float_format="%.15g")
    (TAB_DIR / "entry_formula_memo.md").write_text(
        memo(args.horizon, wins, nrows, cuts))
    (TAB_DIR / "Table_entry_bins_efsy_note.tex").write_text(
        note(coeff, ident, args.horizon))
    # write_exhibit mkdirs only the Results root, so the Dropbox mirror needs
    # its subdirectory made here or the copy fails on a machine that has the
    # share mounted.
    drop = out_paths.dropbox_results()
    if drop is not None:
        (drop / TAB_DIR.name).mkdir(parents=True, exist_ok=True)
    out_paths.write_exhibit(f"{TAB_DIR.name}/Table_entry_bins_efsy.tex",
                            write_table(coeff, args.horizon, tab))

    nf = int((idf["ok"] == 0).sum())
    print(f"\n{'=' * 80}")
    print(f"  wrote entry_bins_coeff.csv ({len(cf):,} rows), "
          f"entry_bins_identities.csv ({len(idf)} gates), "
          f"entry_formula_memo.md, Table_entry_bins_efsy_note.tex")
    print(f"  {nf} gates failed, {time.time() - t0:.1f}s")
    print("=" * 80)
    return 1 if nf else 0


if __name__ == "__main__":
    raise SystemExit(main())
