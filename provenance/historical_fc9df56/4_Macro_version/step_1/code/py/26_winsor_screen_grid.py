#!/usr/bin/env python3
"""Table VI's grid, OLS only, WINSORIZING the outlying counties instead of dropping them.

WHAT IS BEING WINSORIZED, AND IT IS NOT THE CHARACTERISTIC

23_sample_screen_grid.py takes three county characteristics and DELETES the
counties in their tails. This file takes the same idea and caps instead:

    1  each characteristic supplies a DISTRIBUTION
    2  winsor4 varlist, method(iq) level(L) - outside p50 +/- L*IQR - reads off
       that distribution which counties are in the tail
    3  THOSE COUNTIES' ESTIMATION OBSERVATIONS are capped, at winsor4's bound on
       the ESTIMATION VARIABLE - p50 +/- L*IQR of x, and of z, over the era
    4  and the same flag set is also run as a DROP, beside the cap

THE SAME RULE ON BOTH SIDES, AND THE VERSION THAT WAS TRIED FIRST

One L does two jobs: it picks the counties out of the characteristic's
distribution, and it sets the cap on the regression variable's own distribution.

The first version of this file capped the flagged counties at the most extreme x
observed among the counties that were NOT flagged. It is the more literal reading
of "replace the tail with the value at the cutoff", and it is VACUOUS BY
CONSTRUCTION here - measured, not suspected: 0 county-years moved on the
population flag and 0 on employment growth, with every capped F equal to the
baseline to two decimals.

The reason is structural. x's extreme values sit in small counties that landed one
large contract, and those are exactly the counties these flags do not catch - the
flags were chosen to be exogenous to the instrument, which guarantees they miss x's
tails. So the unflagged set already spanned a wider range than anything the flagged
set held, and the clip never bound. A design that cannot bite whenever its own
selection criterion is satisfied is not a design, so the cap moved onto the
variable being capped, where winsor4 puts it.

The characteristic itself is never the argument of the cap. That is what makes all
three usable: the flagging variable does not have to appear in the regression at
all, which is why population and rurality - neither of which the estimator ever
touches - can each drive a panel.

Cap against drop against neither, on ONE flag set, is the point of the file. 23
can say "deleting these counties changes the first stage"; it cannot separate that
from "these counties matter", because deletion changes the sample and the moments
at once. Capping holds the sample fixed - same n_units, same n_frame, asserted as
check 9 - and moves only the tail's leverage.

y IS NEVER WINSORIZED                                          protocol A.3.2

The literature standard here would be to cap the outcome growth rate at p1/p99.
A.3.2 governs growth rates, step_2's 22_winsor_construction_error.py says so in its
own docstring, and README 1.7 records that this project does not winsorize the
shock either. So only x and z are capped, and emp_growth is asserted IDENTICAL to
the baseline in every panel, max dev 0.0 - check 10.

That assertion is what distinguishes Table XIII's flag from its target. The
employment panel RANKS ON a growth rate and does not WINSORIZE one, and a reader
who sees "growth" in the column head has no way to tell those apart except by the
check.

THE THREE FLAGS, AND WHERE EACH IS MEASURED

Every measurement window IS THE ERA'S FIXED-EXPOSURE WINDOW - 1969-74 on pre,
1993-97 on post, taken from g20.WINDOWS through ERA_WINDOW and asserted equal to
the exposure's own window rather than assumed (check 8b). Two consequences, and
neither is a coincidence: the flag describes the same pre-period the design loads
off, and it sits strictly outside every estimated year, so no flag can select on
the estimated treatment or the estimated outcome.

    XI    population          window mean of population_ccg_year.parquet
    XII   rural               2003 OMB CBSA membership - a binary, so L selects
                              nobody here and only sets the cap
    XIII  employment growth   across the window, from bea_employment_ccg_year
                              .parquet. EFSY starts in 1975 and cannot supply a
                              1969-74 growth rate; 25_ingest_bea_employment.py is
                              why this window rule holds on both eras. The flag is
                              therefore BEA employment growth while the outcome is
                              EFSY employment growth - stated in the table note.

Ordered as 23 orders its rules: the flag furthest from the instrument first.

THERE IS NO CONTRACT-DOLLAR PANEL, AND BOTH TAILS ARE ALREADY ANSWERED

The zero end is 23's rule (a), the `nonzero` screen, at h=3 and h=4 in Table IX.
The upper tail is 24_exposure_screen_grid.py's Table X, which screens on s_c -
contracts over income - at p99.5 and p99, and those are precisely the counties an
IQR bound on contract dollars would catch. README 1.8.9 reading 3 is the reason
not to run it a third time: z_ct = s_c x buildup_t, so with a year effect absorbing
buildup_t the cross-sectional spread of z IS the spread of s_c, and a screen that
ranks on the instrument's own loading cannot be read as evidence about it.

Leaving it out is what makes the three panels here readable: population comes from
BEA counts, rurality from the OMB delineation, employment growth from BEA jobs, and
NONE of the three appears anywhere in z. So 1.8.9 reading 3's warning applies to no
panel in this file.

What it does NOT buy: capping x and z is still not a sample restriction. It is a
change to the treatment and the instrument for the flagged counties. The flag being
exogenous to the instrument is what is established; the operation is not a screen.

Output:
    dc.LOG_DIR/winsor_screen_grid.csv
    dc.LOG_DIR/26_winsor_screen_grid.log
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dcads_common as dc  # noqa: E402

CODE = Path(__file__).resolve().parent


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, CODE / fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# 23 owns the pre/post design, the build and the screens; it holds 20, which holds
# 14 and 19. All four come from that ONE import and none is loaded again, for the
# module-identity reason 23's own import block records: a second _load would give
# this file a private copy of 14 whose PERIODS dict nobody registers into, and
# bi.contaminated_years would raise on the first design row.
g23 = _load("sample_screen_grid", "23_sample_screen_grid.py")
g20 = g23.g20
bi = g23.bi
sym = g23.sym

# fixed->pre, fixed_9397->post, rolling_t5->both. Taken from 23 by filtering, for
# the reason 23 takes its own from 20: a change to the pairing must not leave two
# files reporting different designs under the same era names.
#
# PINNED, 2026-08-13. 23 gained the two `fixed` x post blocks that day - 1999-2018
# and its 2016 truncation - and inheriting them here would put 6 more pairs x 3
# rungs x 3 horizons x 8 flags into winsor_screen_grid.csv, and 6 new rows into
# each of Tables XI, XII, XIII, XVII, XVIII and XIX, none of which were asked for.
# The winsorization exhibit is about what the tails do to the ORIGINAL four pairs
# and is left exactly as published. Excluded by era, so a later pair on pre or post
# still arrives here on its own.
SKIP_ERAS = ("post", "post_pre2017")
DESIGN = tuple((e, x) for e, x in g23.DESIGN
               if not (e == "fixed" and x in SKIP_ERAS))
ERA_LABEL = g23.ERA_LABEL
RUNGS = g23.RUNGS
WINDOWS = g23.WINDOWS
EXPOSURES = g23.EXPOSURES
ROLL_NAMES = g23.ROLL_NAMES
OMEGA_COL = g23.OMEGA_COL
DEFAULT_OMEGA = g23.DEFAULT_OMEGA

RTOL_FIT = g23.RTOL_FIT
ATOL_FIT = g23.ATOL_FIT
TOL_TABLE_F = g23.TOL_TABLE_F
TOL_BUILD = 1e-12

HORIZONS = (2, 3, 4)

# Which fixed window describes each era. This is the measurement-window rule's
# whole content, and it is also the DESIGN pairing restated in the one form the
# flags need. 24 defines the same dict for the same reason.
ERA_WINDOW = {"pre": "fixed", "post": "fixed_9397"}

# winsor4 method(iq) levels. "4 or 5 times the interquartile range", both reported.
LEVELS = (4, 5)

# Check 7's level. No county can be 1000 IQRs from the median of anything here, so
# the flag set must come out empty and the capped frame must be the baseline frame.
NULL_LEVEL = 1000

FLAGS = ("population", "rural", "emp_growth")
FLAG_TABLE = {"population": "XI", "rural": "XII", "emp_growth": "XIII"}
FLAG_LABEL = {
    "population": "county population, window mean",
    "rural": "outside a 2003 CBSA",
    "emp_growth": "BEA total employment growth across the window",
}

# Rurality is a binary, so its two "levels" are which CBSA definition is used
# rather than a multiple of an IQR. Named so the column heads read as themselves.
RURAL_VARIANTS = ("any", "metro")

DEMEANER = None

check = g23.check
eras_for = g23.eras_for
build = g23.build


def fit(frame, fe, weights=None):
    """bi.fit_rung with the demeaner pinned, exactly as 20, 23 and 24 do it."""
    return bi.fit_rung(frame, fe, weights=weights, demeaner=DEMEANER)


def windows_for(name, h):
    """g23.windows_for on this file's design, which is 23's."""
    return g23.windows_for(name, h, design=DESIGN)


# --------------------------------------------------------------------------------
# The flags
# --------------------------------------------------------------------------------

def flag_values():
    """{(flag, era): Series ccg -> the one number per county}.

    Every window is the era's fixed-exposure window, read out of WINDOWS rather
    than typed, so check 8b's assertion is against the same object the exposure is
    built from and cannot drift from it.
    """
    pop = pd.read_parquet(dc.PANEL_OUT_DIR / "population_ccg_year.parquet")
    bea_p = dc.PANEL_OUT_DIR / "bea_employment_ccg_year.parquet"
    if not bea_p.exists():
        raise FileNotFoundError(
            f"{bea_p} not found - run 25_ingest_bea_employment.py. EFSY starts in "
            f"1975 and cannot supply a 1969-74 growth rate, which is why this file "
            f"does not fall back to it silently")
    bea = pd.read_parquet(bea_p)

    out, windows = {}, {}
    for era, wname in ERA_WINDOW.items():
        w0, w1 = WINDOWS[wname]
        windows[era] = (w0, w1)

        p = pop[pop["year"].between(w0, w1)]
        out[("population", era)] = p.groupby("ccg")["population"].mean()

        a = bea[bea["year"] == w0].set_index("ccg")["bea_emp"]
        b = bea[bea["year"] == w1].set_index("ccg")["bea_emp"]
        common = a.index.intersection(b.index)
        out[("emp_growth", era)] = ((b[common] - a[common]) / a[common])
    return out, windows


def rural_flagged(universe):
    """{variant: frozenset of CCGs OUTSIDE the relevant CBSA class}.

    A CCG is a GROUP of counties, so "in a CBSA" means ANY member is - 23's rule,
    reused through its loader so the two files cannot disagree about it. The
    FLAGGED set is the complement, because rurality flags the rural counties and
    the drop column then has to reproduce 23's cbsa_any / cbsa_metro screens,
    which KEEP the urban ones.
    """
    any_cbsa, metro_cbsa = g23.load_cbsa()
    fips_to_ccg, _, _ = dc.load_ccg_map()
    uni = set(int(c) for c in universe)
    in_any = {int(fips_to_ccg[f]) for f in any_cbsa if f in fips_to_ccg}
    in_metro = {int(fips_to_ccg[f]) for f in metro_cbsa if f in fips_to_ccg}
    return {"any": frozenset(uni - in_any), "metro": frozenset(uni - in_metro)}


def iq_bounds(s, level):
    """winsor4's own arithmetic: p50 -/+ level * (p75 - p25).

    Stata's `summarize, detail` then two `replace`s. Written out rather than
    delegated to a quantile helper because it is the definition the section is
    named after and it should be readable beside the README sentence that quotes
    it.
    """
    q25, q50, q75 = s.quantile(0.25), s.median(), s.quantile(0.75)
    iqr = q75 - q25
    return q50 - level * iqr, q50 + level * iqr


def build_flags(universe, levels=LEVELS):
    """{(flag, variant, era): (flagged frozenset, label, lo, hi)}.

    One frozen set per key. The continuous flags are two-sided by construction -
    a county below p50 - L*IQR is as flagged as one above it - which is what an
    IQR rule is for and what a percentile screen on a skewed level is not.
    """
    vals, windows = flag_values()
    rural = rural_flagged(universe)
    uni = set(int(c) for c in universe)

    out = {}
    for era in ERA_WINDOW:
        w0, w1 = windows[era]
        for flag in ("population", "emp_growth"):
            s = vals[(flag, era)].dropna()
            for L in levels:
                lo, hi = iq_bounds(s, L)
                bad = frozenset(int(c) for c in s.index[(s < lo) | (s > hi)]) & uni
                out[(flag, str(L), era)] = (
                    bad, f"{FLAG_LABEL[flag]} {w0}-{w1}, outside p50 +/- {L}*IQR",
                    float(lo), float(hi))
        for v in RURAL_VARIANTS:
            out[("rural", v, era)] = (
                rural[v], f"outside a 2003 {'metropolitan ' if v == 'metro' else ''}"
                          f"CBSA", np.nan, np.nan)
    return out, windows


def treatments(flags, levels=LEVELS):
    """The column list, in the order the README carries it.

    ("all", None, None) first, then per flag: cap at each level, then drop at each
    level, so a reader scanning a row sees the two treatments of one flag set side
    by side rather than in separate blocks.
    """
    cols = [("all", "", "")]
    for flag in FLAGS:
        if flag == "rural":
            # The flag is a binary, so L does not select anyone here - but the cap
            # still needs a bound, so every CBSA variant is capped at both levels
            # and dropped once. That asymmetry is the binary's, not a choice.
            for v in RURAL_VARIANTS:
                for L in levels:
                    cols.append(("cap", flag, f"{v}{L}"))
            for v in RURAL_VARIANTS:
                cols.append(("drop", flag, v))
        else:
            for how in ("cap", "drop"):
                for L in levels:
                    cols.append((how, flag, str(L)))
    return cols


def col_name(how, flag, variant):
    return "all" if how == "all" else f"{how}_{flag}_{variant}"


def split_variant(flag, variant):
    """(flag key variant, cap level) for a column's variant string.

    rural's cap columns carry both the CBSA definition and the level in one token
    - "any4" - because the flag key is indexed on the definition alone.
    """
    if flag == "rural":
        for L in LEVELS:
            if variant.endswith(str(L)):
                return variant[:-len(str(L))], L
        return variant, LEVELS[0]
    return variant, int(variant)


# --------------------------------------------------------------------------------
# The winsorization
# --------------------------------------------------------------------------------

def winsorize(b, flagged, level, cols=("x", "z")):
    """Cap the flagged counties' x and z at winsor4's bound on x and on z.

    (frame, n_moved, removed_share, lo, hi). The bound is p50 +/- level*IQR of the
    column itself, taken over the WHOLE era frame - the same arithmetic iq_bounds
    applies to the characteristic, applied here to the variable being capped. The
    flag decides WHO is exposed to that bound; counties the flag did not catch keep
    their tails whatever they are.

    Taking the bound on the whole era rather than within the flagged group is what
    makes the columns comparable: the cutoff is then the same number in every
    column of a row, and only the set exposed to it changes.

    y IS NOT IN cols AND MUST NOT BE. See the docstring - A.3.2 - and check 10,
    which asserts emp_growth came through untouched.

    Rows are not removed and counties are not removed, so the frame this returns
    has exactly the shape it was given. That is what makes the cap column
    comparable to the baseline column cell for cell, and check 9 asserts it.
    """
    d = b.copy()
    mask = d["ccg"].isin(flagged)
    n_moved, removed, lo_x, hi_x = 0, 0.0, np.nan, np.nan
    if mask.any():
        for c in cols:
            lo, hi = iq_bounds(d[c].dropna(), level)
            before = d.loc[mask, c]
            after = before.clip(lower=lo, upper=hi)
            if c == "x":
                lo_x, hi_x = float(lo), float(hi)
                n_moved = int((before != after).sum())
                tot = float(before.abs().sum())
                removed = (float((before - after).abs().sum()) / tot
                           if tot > 0 else 0.0)
            d.loc[mask, c] = after
    return d, n_moved, removed, lo_x, hi_x


def dollar_share(levels_frame, flagged, h, era):
    """% of the era's real contract dollars sitting in the flagged counties."""
    eras, _ = eras_for(h)
    y0, y1 = eras[era]
    d = levels_frame[levels_frame["year"].between(y0, y1)]
    tot = float(d["dollars_real"].sum())
    hit = float(d.loc[d["ccg"].isin(flagged), "dollars_real"].sum())
    return 100.0 * hit / tot if tot else np.nan


def emp_share(levels_frame, flagged, h, era):
    """% of the era's covered employment sitting in the flagged counties."""
    eras, _ = eras_for(h)
    y0, y1 = eras[era]
    d = levels_frame[levels_frame["year"].between(y0, y1)]
    tot = float(d["emp"].sum())
    hit = float(d.loc[d["ccg"].isin(flagged), "emp"].sum())
    return 100.0 * hit / tot if tot else np.nan


# --------------------------------------------------------------------------------
# The fits
# --------------------------------------------------------------------------------

def fit_grid(county, rungs, pre, horizons, rung_names, omega, flags, levels_frame,
             levels=LEVELS):
    """One row per treatment x exposure x era x h x rung. OLS throughout.

    The finiteness screen sits OUTSIDE the era loop, where 20 and 23 put theirs,
    so an era differs from another only in which years it contains. The cap and
    the drop are applied INSIDE it, on the era's own frame, because both the
    cutoff and the flag set are era-specific.
    """
    wanted = [(r, c) for r, c in RUNGS if r in set(rung_names)]
    omega_col = OMEGA_COL[omega]
    cols = treatments(flags, levels)
    rows, repeat_devs, y_devs = [], [], []

    for h in horizons:
        for name in EXPOSURES:
            base_full = county[county["h"] == h].copy()
            base_full["z"] = base_full[f"z_{name}"]
            if name in pre:
                base_full["pre_D"] = base_full["ccg"].map(pre[name]["pre_D"])
                base_full["pre_Y"] = base_full["ccg"].map(pre[name]["pre_Y"])
            else:
                base_full["pre_D"] = np.nan
                base_full["pre_Y"] = np.nan
            base_full = base_full.merge(rungs, on="ccg", how="left")
            keep = np.isfinite(base_full[["shock", "emp_growth", "z",
                                          "income_real_lag", "emp_lag"]]
                               .to_numpy()).all(axis=1)
            base_full = base_full[keep & (base_full["emp_lag"] > 0)
                                  & (base_full["income_real_lag"] > 0)]

            for role, era, pkey, y0, y1 in windows_for(name, h):
                b0 = base_full[base_full["year"].between(y0, y1)].copy()
                b0 = b0.rename(columns={"shock": "x"})
                n_con = len(bi.contaminated_years(pkey, name, h))

                for how, flag, variant in cols:
                    # The anchor exists to reproduce a published number, so it
                    # runs unwinsorized only.
                    if role == "anchor" and how != "all":
                        continue
                    if how == "all":
                        b, flagged, moved, removed = b0, frozenset(), 0, 0.0
                        lo = hi = cap_lo = cap_hi = np.nan
                    else:
                        fvar, L = split_variant(flag, variant)
                        key = (flag, fvar, era)
                        if key not in flags:
                            continue
                        flagged, _label, lo, hi = flags[key]
                        cap_lo = cap_hi = np.nan
                        if how == "cap":
                            b, moved, removed, cap_lo, cap_hi = winsorize(
                                b0, flagged, L)
                        else:
                            b = b0[~b0["ccg"].isin(flagged)]
                            moved, removed = 0, np.nan

                    bb = b.rename(columns={"x": "shock"})
                    for rung, unit_col in wanted:
                        d = g20.unit_frame(bb, rung, unit_col, omega_col)
                        r = fit(d, "unit + year")
                        if (role == "variant" and how == "all"
                                and h == min(horizons)):
                            again = fit(d, "unit + year")
                            repeat_devs.append(
                                (f"{name} x {era} x {rung} h={h}",
                                 max(abs(again[k] - r[k])
                                     for k in ("fs_F", "beta", "beta_se"))))
                        # Check 10's evidence, collected where the frame exists:
                        # the outcome must be the baseline's in every treatment.
                        if how == "cap" and rung == "county":
                            m = b0[["ccg", "year", "emp_growth"]].merge(
                                b[["ccg", "year", "emp_growth"]],
                                on=["ccg", "year"], suffixes=("_base", "_cap"))
                            y_devs.append(float(
                                (m["emp_growth_base"] - m["emp_growth_cap"])
                                .abs().max()))
                        rows.append(dict(
                            role=role, treatment=how, flag=flag, variant=variant,
                            column=col_name(how, flag, variant),
                            flag_label=(FLAG_LABEL.get(flag, "") if flag else
                                        "no winsorization"),
                            n_ccg_flagged=len(flagged),
                            flag_lo=lo, flag_hi=hi,
                            cap_lo=cap_lo, cap_hi=cap_hi,
                            n_countyyears_moved=moved,
                            pct_x_removed=100.0 * removed if how == "cap" else np.nan,
                            pct_dollars_flagged=(
                                np.nan if role == "anchor" or not flagged
                                else dollar_share(levels_frame, flagged, h, era)),
                            pct_emp_flagged=(
                                np.nan if role == "anchor" or not flagged
                                else emp_share(levels_frame, flagged, h, era)),
                            era=era, era_label=ERA_LABEL.get(era, era),
                            era_span=f"{y0}-{y1}", exposure=name,
                            exposure_window=g20.window_label(name), rung=rung,
                            n_units=int(d["unit"].nunique()), omega=omega,
                            reg_weights="unweighted", h=h,
                            year_first=y0, year_last=y1, fe="unit+year",
                            contaminated_years=n_con, n_frame=len(d), **r))
    return pd.DataFrame(rows), repeat_devs, y_devs


# --------------------------------------------------------------------------------
# The checks
# --------------------------------------------------------------------------------

def check_baseline(failures, grid, horizons, rung_names):
    """Check 5. The baseline column IS 23's `all` column, at every horizon.

    THE CHECK THIS FILE RESTS ON. Every capped and every dropped number is read as
    a movement away from this column, so a baseline that is not 23's makes all of
    them unreadable - and it costs nothing but the assertion, the cells being
    design rows here anyway.
    """
    p = dc.LOG_DIR / "sample_screen_grid.csv"
    if not p.exists():
        return check(failures, False, 5, "the baseline column IS 23's `all` column",
                     f"{p.name} not found - run 23 first")
    o = pd.read_csv(p)
    o = o[(o["role"] == "variant") & (o["screen"] == "all")
          & o["h"].isin(horizons)]
    d = grid[(grid["role"] == "variant") & (grid["treatment"] == "all")]
    m = o.merge(d, on=["exposure", "era", "rung", "h"], suffixes=("_frozen", "_new"))
    want = len(DESIGN) * len(rung_names) * len(horizons)
    return g20._compare(m, ("fs_F", "beta", "beta_se", "n_obs"), want, 5,
                        "the baseline column IS 23's `all` column", failures)


def check_rural_drop(failures, grid, horizons, rung_names):
    """Check 6. The rural DROP columns ARE Table VIII's CBSA screens.

    Free, and load-bearing. `drop_rural_any` keeps exactly the counties 23's
    `cbsa_any` keeps and `drop_rural_metro` exactly `cbsa_metro`'s, so if the two
    disagree then either the flag's complement or 23's screen is wrong, and
    nothing else in either run would say which.
    """
    p = dc.LOG_DIR / "sample_screen_grid.csv"
    if not p.exists():
        return check(failures, False, 6, "the rural drop columns ARE Table VIII",
                     f"{p.name} not found - run 23 first")
    o = pd.read_csv(p)
    o = o[(o["role"] == "variant") & o["screen"].isin(("cbsa_any", "cbsa_metro"))
          & o["h"].isin(horizons)]
    o["variant"] = o["screen"].str.replace("cbsa_", "", regex=False)
    d = grid[(grid["role"] == "variant") & (grid["treatment"] == "drop")
             & (grid["flag"] == "rural")]
    m = o.merge(d, on=["exposure", "era", "rung", "h", "variant"],
                suffixes=("_frozen", "_new"))
    want = len(DESIGN) * len(rung_names) * len(horizons) * len(RURAL_VARIANTS)
    return g20._compare(m, ("fs_F", "beta", "beta_se", "n_obs"), want, 6,
                        "the rural drop columns ARE Table VIII's CBSA screens",
                        failures)


def check_confined(failures, county, universe, horizons):
    """Check 7. The cap touches the flagged counties and nobody else.

    WHAT THIS REPLACED, AND WHY IT COULD NOT BE TESTED AS POSED.

    The intended check was "at an absurd L nothing is flagged and the cap is a
    no-op". It is not implementable on this variable: the shock's IQR at h=2 is
    0.0092 and its maximum is 14.07, which is 1,529 IQRs above the median, so
    p50 +/- 1000*IQR STILL BINDS. There is no finite multiple of x's interquartile
    range that leaves x alone. That is a fact about the shock's tail rather than
    about the cap, it is worth knowing, and it is recorded in README 1.8.10 as one
    of the reasons an IQR rule is aggressive on this regressor.

    So the plumbing is proved in the two directions that ARE testable:

      a  an EMPTY flag set is a no-op. Nothing outside the mask may move, so with
         nothing in the mask nothing may move at all.
      b  under a REAL flag set at a REAL level, the UNFLAGGED rows are bitwise
         unchanged. A mask that leaked would cap the whole cross-section and read
         as "winsorizing costs the first stage a lot", which is the strongest
         result this table could produce and the one most worth not faking.

    The opposite direction - that the cap DID bite - is check 9, and it has to be
    read beside this one.
    """
    d = county[county["h"] == min(horizons)].rename(columns={"shock": "x"})
    d = d.assign(z=d["z_rolling_t5"])

    def dev(a, b):
        g = a.notna() & b.notna()
        return max(float((a[g] - b[g]).abs().max()) if g.any() else 0.0,
                   0.0 if a.isna().equals(b.isna()) else np.inf)

    empty, moved_e, _rm, _lo, _hi = winsorize(d, frozenset(), LEVELS[0])
    worst_empty = max(dev(d[c], empty[c]) for c in ("x", "z")) + float(moved_e)

    flags, _ = build_flags(universe)
    worst_leak, n_keys = 0.0, 0
    for (flag, variant, era), (flagged, *_rest) in flags.items():
        if era != "pre":
            continue
        capped, _m, _r, _l, _h = winsorize(d, flagged, LEVELS[0])
        out = ~d["ccg"].isin(flagged)
        n_keys += 1
        worst_leak = max(worst_leak,
                         max(dev(d.loc[out, c], capped.loc[out, c])
                             for c in ("x", "z")))
    ok = worst_empty == 0.0 and worst_leak == 0.0 and n_keys > 0
    return check(failures, ok, 7,
                 "the cap is confined to the flagged counties",
                 f"empty flag set dev {worst_empty:.1e}; "
                 f"{n_keys} real flag sets, unflagged dev {worst_leak:.1e}")


def check_flags(failures, flags, universe, windows, horizons):
    """Check 8, 8b and 8c. The flag sets, their windows and their support."""
    uni = set(int(c) for c in universe)
    bad = []
    for (flag, variant, era), (flagged, _lab, _lo, _hi) in flags.items():
        if not flagged <= uni:
            bad.append(f"{flag}/{variant}/{era} is not a subset of the universe")
        if not flagged:
            bad.append(f"{flag}/{variant}/{era} flags nobody - it cannot bite")
        if len(flagged) >= len(uni):
            bad.append(f"{flag}/{variant}/{era} flags everybody")
    check(failures, not bad, 8, "every flag set is a subset that bites",
          f"{len(flags)} keys" if not bad else f"{len(bad)} deviation(s)")
    for b in bad:
        print(f"          {b}")

    # 8b. THE MEASUREMENT WINDOW IS THE EXPOSURE'S WINDOW, and it precedes the era.
    wbad = []
    for era, wname in ERA_WINDOW.items():
        if windows[era] != WINDOWS[wname]:
            wbad.append(f"{era}: flag window {windows[era]} != "
                        f"{wname} {WINDOWS[wname]}")
        for h in horizons:
            eras, _ = eras_for(h)
            y0 = eras[era][0]
            if windows[era][1] >= y0:
                wbad.append(f"{era} h={h}: window ends {windows[era][1]}, "
                            f"era starts {y0}")
    check(failures, not wbad, "8b",
          "every flag window IS the era's exposure window, and precedes it",
          ", ".join(f"{e} {windows[e][0]}-{windows[e][1]}" for e in ERA_WINDOW)
          if not wbad else "; ".join(wbad))

    # 8c. The employment flag's support is the estimation support.
    bea = pd.read_parquet(dc.PANEL_OUT_DIR / "bea_employment_ccg_year.parquet")
    have = set(int(c) for c in bea["ccg"].unique())
    covered = {}
    for era, (w0, w1) in windows.items():
        n = bea[bea["year"].isin((w0, w1))].groupby("ccg").size()
        covered[era] = int((n == 2).sum())
    ok = all(v == len(have) for v in covered.values()) and len(have) >= len(uni) - 1
    check(failures, ok, "8c", "the BEA employment flag covers the estimation support",
          f"{len(have):,} CCGs, both window years complete for "
          + ", ".join(f"{e} {v:,}" for e, v in covered.items()))


def check_winsorization(failures, grid, y_devs, rung_names):
    """Check 9 and 10. The cap is a winsorization, and y came through untouched."""
    base = grid[(grid["role"] == "variant") & (grid["treatment"] == "all")]
    cap = grid[(grid["role"] == "variant") & (grid["treatment"] == "cap")]
    k = ["exposure", "era", "rung", "h"]
    m = cap.merge(base[k + ["n_units", "n_frame", "n_obs"]], on=k,
                  suffixes=("_cap", "_base"))
    same = bool((m["n_units_cap"] == m["n_units_base"]).all()
                and (m["n_frame_cap"] == m["n_frame_base"]).all()
                and (m["n_obs_cap"] == m["n_obs_base"]).all())
    # A cap that never moved anything anywhere would satisfy the shape test and
    # mean nothing, so the fact that it BIT somewhere is asserted beside it.
    bit = int(cap["n_countyyears_moved"].sum()) > 0
    check(failures, same and bit and len(m) > 0, 9,
          "capping keeps every county and every row, and bites",
          f"{len(m)} capped cells, shape identical, "
          f"{int(cap['n_countyyears_moved'].sum()):,} county-years moved")

    worst = max(y_devs) if y_devs else np.inf
    check(failures, bool(y_devs) and worst == 0.0, 10,
          "y is never winsorized - emp_growth identical to the baseline",
          f"{len(y_devs)} capped frames, max dev {worst:.1e}")


def check_rest(failures, grid, repeat_devs, horizons, rung_names, flags,
               canonical):
    """Checks 11 and 12, plus the contamination assertion."""
    d = grid[grid["role"] == "variant"]
    check(failures, int(d["contaminated_years"].max()) == 0, "11a",
          "zero contaminated years on every design row",
          f"max {int(d['contaminated_years'].max())} over {len(d)} rows")

    n_cols = len(treatments(flags))
    w_var = len(DESIGN) * len(horizons) * len(rung_names) * n_cols
    w_anc = len(sym.ANCHOR_EXPOSURES) * len(horizons) * len(rung_names)
    n_var = int((grid["role"] == "variant").sum())
    n_anc = int((grid["role"] == "anchor").sum())
    check(failures, n_var == w_var and n_anc == w_anc, 11,
          "row count is the announced grid" if canonical
          else "row count is the requested subset",
          f"{n_var} of {w_var} design ({n_cols} columns), {n_anc} of {w_anc} anchor")

    worst = max((v for _, v in repeat_devs), default=np.inf)
    where = max(repeat_devs, key=lambda t: t[1])[0] if repeat_devs else "none"
    check(failures, bool(repeat_devs) and worst == 0.0, 12,
          "a refit is bit-identical, the demeaner being pinned",
          f"{len(repeat_devs)} cells refitted, max dev {worst:.1e} ({where})")


# --------------------------------------------------------------------------------

def main():
    global DEMEANER
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--horizon", type=int, nargs="+", default=list(HORIZONS))
    ap.add_argument("--rung", nargs="+", default=[r for r, _ in RUNGS],
                    choices=[r for r, _ in RUNGS])
    ap.add_argument("--flag", nargs="+", default=list(FLAGS), choices=list(FLAGS))
    ap.add_argument("--omega", choices=sorted(OMEGA_COL), default=DEFAULT_OMEGA,
                    help="the AGGREGATION weight above the county. There is no "
                         "regression weight: this table is OLS")
    args = ap.parse_args()

    horizons = sorted(set(args.horizon))
    wanted_flags = [f for f in FLAGS if f in set(args.flag)]
    canonical = (horizons == sorted(HORIZONS)
                 and set(args.rung) == {r for r, _ in RUNGS}
                 and set(wanted_flags) == set(FLAGS)
                 and args.omega == DEFAULT_OMEGA)
    sfx = "" if canonical else ("__h" + "-".join(map(str, horizons)) + "_"
                               + "-".join(sorted(args.rung)).replace(" ", "")
                               + "_" + "-".join(wanted_flags))

    tee = sym.Tee(dc.LOG_DIR / f"26_winsor_screen_grid{sfx}.log")
    try:
        import pyfixest as pf
        DEMEANER = pf.MapDemeaner(fixef_tol=1e-12, fixef_maxiter=100_000)
        g23.DEMEANER = DEMEANER      # 23's helpers fit through their own global

        failures = []
        print("=" * 78)
        print("Table VI's grid, OLS only, WINSORIZING the outlying counties")
        print("=" * 78)
        print("  design     " + ", ".join(f"{e} -> {ERA_LABEL[x]}"
                                          for e, x in DESIGN))
        print(f"  h          {horizons}")
        for h in horizons:
            eras, straddle = eras_for(h)
            print(f"    h={h}   " + "   ".join(
                f"{ERA_LABEL[k]} {v[0]}-{v[1]}" for k, v in eras.items())
                + f"   straddle {straddle[0]}-{straddle[1]}, estimated nowhere")
        print(f"  rungs      {', '.join(args.rung)}")
        print(f"  flags      " + ", ".join(
            f"{FLAG_TABLE[f]} {f}" for f in wanted_flags))
        print(f"  levels     winsor4 method(iq), L = "
              f"{', '.join(map(str, LEVELS))}; rural is a binary and has none")
        print(f"  capped     x and z. NOT y - protocol A.3.2, asserted as check 10")
        print(f"  windows    the era's own fixed-exposure window: "
              + ", ".join(f"{e} {WINDOWS[w][0]}-{WINDOWS[w][1]}"
                          for e, w in ERA_WINDOW.items()))
        print(f"  demeaner   PINNED, fixef_tol=1e-12, fixef_maxiter=100,000")

        print("\n[1] the eras, and the registration")
        g23.check_eras(failures, horizons)
        before = g20.register_variants(horizons)
        ok, detail = g23.only_added(before)
        check(failures, ok, 2, "registering the variants only ADDED keys", detail)

        print("\n[2] the instrument, every exposure, in memory")
        county, levels_frame, universe, pre = build(horizons)
        rungs = bi.load_rungs(universe)

        print("\n[3] the build, against production and against source")
        g23.check_build(failures, county, horizons)
        sym.check_rolling_pair(failures, county, horizons)

        print("\n[4] the flags")
        flags, windows = build_flags(universe)
        flags = {k: v for k, v in flags.items() if k[0] in set(wanted_flags)}
        for (flag, variant, era), (flagged, label, lo, hi) in flags.items():
            cut = ("" if not np.isfinite(lo)
                   else f"   outside [{lo:+.4f}, {hi:+.4f}]")
            print(f"    {FLAG_TABLE[flag]:<4} {flag:<11} {variant:<6} {era:<5} "
                  f"flags {len(flagged):>5,}{cut}")

        print("\n[5] the fits")
        grid, repeat_devs, y_devs = fit_grid(
            county, rungs, pre, horizons, args.rung, args.omega, flags,
            levels_frame)
        print(f"    {len(grid)} fits: "
              f"{int((grid['role'] == 'variant').sum())} design, "
              f"{int((grid['role'] == 'anchor').sum())} anchor")

        print("\n[6] the checks")
        g23.check_anchor(failures, grid, horizons, args.rung, args.omega)
        check_baseline(failures, grid, horizons, args.rung)
        if "rural" in wanted_flags:
            check_rural_drop(failures, grid, horizons, args.rung)
        check_confined(failures, county, universe, horizons)
        check_flags(failures, flags, universe, windows, horizons)
        check_winsorization(failures, grid, y_devs, args.rung)
        check_rest(failures, grid, repeat_devs, horizons, args.rung, flags,
                   canonical)

        if failures:
            print(f"\n{len(failures)} CHECK(S) FAILED, nothing written")
            for num, msg in failures:
                print(f"    [{num}] {msg}")
            report(grid, horizons, args.rung, wanted_flags)
            return 1

        if not canonical:
            print(f"\n  PARTIAL RUN. The canonical outputs are left untouched; "
                  f"this run writes to *{sfx}.* instead.")

        dc.LOG_DIR.mkdir(parents=True, exist_ok=True)
        out = dc.LOG_DIR / f"winsor_screen_grid{sfx}.csv"
        grid.to_csv(out, index=False)
        print(f"\nall checks passed   {len(grid)} fits -> {out}")
        report(grid, horizons, args.rung, wanted_flags)
        return 0
    finally:
        tee.close()


def report(grid, horizons, rung_names, wanted_flags):
    """Tables XI-XIII, in the shape the README carries them.

    Printed even on failure: a run that failed check 5 is exactly the run whose
    numbers need reading.
    """
    d = grid[grid["role"] == "variant"]
    short = {"fixed": "fixed 1969-74", "fixed_9397": "fixed 1993-97",
             "rolling_t5": f"rolling K={bi.ROLL_K} t5"}
    all_cols = treatments(dict(), LEVELS)
    for flag in wanted_flags:
        cols = [("all", "", "")] + [c for c in all_cols if c[1] == flag]
        for h in horizons:
            s = d[d["h"] == h]
            if s.empty:
                continue
            print(f"\n  Table {FLAG_TABLE[flag]}. Winsorizing on {FLAG_LABEL[flag]}"
                  f" - first-stage F, h={h}")
            head = f"    {'era':<11}{'exposure':<15}{'rung':<17}"
            for how, fl, v in cols:
                head += f"{col_name(how, fl, v):>18}"
            print(head)
            for label, col, fmt in (("counties flagged", "n_ccg_flagged", ",d"),
                                    ("county-yrs moved", "n_countyyears_moved", ",d"),
                                    ("% pre-era $", "pct_dollars_flagged", ".2f"),
                                    ("% of |x| removed", "pct_x_removed", ".2f")):
                line = f"    {'':<11}{'':<15}{label:<17}"
                for how, fl, v in cols:
                    q = s[(s["column"] == col_name(how, fl, v))
                          & ((s["era"] == "pre") | (col != "pct_dollars_flagged"))]
                    val = q[col].iloc[0] if len(q) else np.nan
                    line += ("".rjust(18) if not np.isfinite(np.float64(val))
                             else f"{val:>18{fmt}}")
                print(line)
            for exposure, era in DESIGN:
                first = True
                for rung, _ in RUNGS:
                    if rung not in set(rung_names):
                        continue
                    r = s[(s["exposure"] == exposure) & (s["era"] == era)
                          & (s["rung"] == rung)]
                    if r.empty:
                        continue
                    lab = r["era_span"].iloc[0] if first else ""
                    exp = short.get(exposure, exposure) if first else ""
                    first = False
                    line = f"    {lab:<11}{exp:<15}{rung:<17}"
                    for how, fl, v in cols:
                        q = r[r["column"] == col_name(how, fl, v)]
                        line += (f"{float(q['fs_F'].iloc[0]):>18.2f}"
                                 if len(q) else "".rjust(18))
                    print(line)

    print(f"\n  2SLS beta (se), the same fits")
    for h in horizons:
        for exposure, era in DESIGN:
            s = d[(d["exposure"] == exposure) & (d["era"] == era) & (d["h"] == h)]
            if s.empty:
                continue
            print(f"    h={h}  {short.get(exposure, exposure)} x "
                  f"{ERA_LABEL[era]} ({s['era_span'].iloc[0]})")
            for rung, _ in RUNGS:
                r = s[s["rung"] == rung]
                if r.empty:
                    continue
                parts = []
                for col in ["all"] + [col_name(*c) for c in treatments(dict())
                                      if c[1] in set(wanted_flags)]:
                    q = r[r["column"] == col]
                    if len(q):
                        parts.append(f"{col} {float(q['beta'].iloc[0]):+.3f} "
                                     f"({float(q['beta_se'].iloc[0]):.3f})")
                print(f"      {rung:<17}" + "  ".join(parts))


if __name__ == "__main__":
    raise SystemExit(main())
