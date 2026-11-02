#!/usr/bin/env python3
"""Table VI's grid, OLS only, screening on the EXPOSURE's own right tail.

WHAT THIS ASKS

23_sample_screen_grid.py tried three restrictions of the county set - population,
CBSA membership, nonzero contracts - and none of them rescues the post-1998 county
cell: the fixed_9397 x post county F stays between 3.38 and 7.20 across all fifteen
screen x horizon combinations and never clears 10. All three rank counties on
something OTHER than the instrument.

This file ranks them on the instrument's own cross-sectional loading, s_c, and
drops the right tail. s_c is a contracts-over-income ratio with a mass at exactly
zero and a max above 1.0 - counties whose window defense contracts exceed their
entire personal income. With unit and year effects those are high-leverage points
in the first stage, and removing them asks directly whether the ladder's strength
is a handful of base towns or the whole cross-section.

THE CUT IS p99.5 AND p99, AND THAT IS NOT AN ARBITRARY PAIR

s_c is a RATIO, so its rank order is not "how much defense" but "how much defense
per dollar of income". Ranks below about p98 are therefore not small base towns;
they are large defense-industrial metros whose big income denominator dilutes an
enormous numerator. Pricing the ladder on the pre era shows where that starts:

    cut       counties   % of pre-era defense $   % of pre-era income
    p99.5           16                    10.08                  0.69
    p99             31                    14.40                  1.40
    p98             62                    20.80                  3.12
    p97.5           78                    26.76                  4.75
    p95            155                    55.72                 17.40

p95 was priced and rejected: it deletes over half the defense obligations and 17%
of the pre-era economy, so its first stage answers a different question rather than
a trimmed version of the same one. p99.5 and p99 keep at least 86% of the treatment
and 98.5% of the economy in either era.

THIS IS THE FIRST SCREEN IN STEP 1 THAT SELECTS ON THE INSTRUMENT

23's ordering argument is that rule (b) is the clean one because it is the only one
that does not select on the treatment, and rule (a) is the compromised one because
it keeps a county only if its treatment is nonzero. This rule sits between them: it
conditions on z's cross-sectional loading, not on the realised regressor, so it
cannot manufacture a first stage out of the outcome - but it is not the clean null
either, and the positives here should be read as "the tail was carrying it" or "the
tail was not", never as evidence for the design.

TWO FAMILIES, AND WHY ONLY ONE OF THEM APPLIES TO THE FIXED EXPOSURES

    own   rank on the design row's OWN exposure. s_c over 1969-74 for fixed x pre,
          over 1993-97 for fixed_9397 x post, and the county MEAN of s_ct over the
          years the era estimates for the two rolling cells.
    fix   rank on the ERA's fixed window for every row - 1969-74 on pre, 1993-97 on
          post - so the screened sample is identical to the paired fixed-exposure
          cell's. A difference between a rolling row and its fixed counterpart is
          then attributable to the exposure alone and not to the two screens having
          cut different counties.

On fixed x pre and fixed_9397 x post the fix rule IS the own rule, so running both
would be 36 duplicate fits under two labels. screens_for() returns three screens for
the fixed-exposure cells and five for the rolling ones, and the equality is asserted
at set level (check 9) rather than paid for in fits.

THE SCREENS ARE COUNTY SETS, BUT NO LONGER ONE SET PER SCREEN

23 could key its screens on the horizon alone because population and CBSA membership
are properties of a county. s_c is a property of a county AND a window, so the key
here is (exposure, era, h) and the screen loop sits INSIDE the era loop rather than
outside it. Each key still resolves to one frozen set of CCGs applied before the
finiteness screen, so within a design row the two eras still differ only in which
years they contain.

Every screen reports what it drops in three currencies - defense dollars, personal
income and employment - because a ratio screen's whole hazard is being cheap in
counties and ruinous in treatment, and a count of counties hides exactly that.

Output:
    dc.LOG_DIR/exposure_screen_grid.csv
    dc.LOG_DIR/24_exposure_screen_grid.log
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


# 23 owns the pre/post design, the build and five of the twelve checks; it holds 20,
# which holds 14 and 19. All four are taken from that one import and none is loaded
# again, for the module-identity reason 23's own import block records: a second
# _load would give this file a private copy of 14 whose PERIODS dict nobody
# registers into, and bi.contaminated_years would raise on the first design row.
g23 = _load("sample_screen_grid", "23_sample_screen_grid.py")
g20 = g23.g20
bi = g23.bi
sym = g23.sym

# THIS FILE IS FROZEN ON THE ORIGINAL `rolling`, AND DECLARES ITS OWN DESIGN.
#
# 23 moved to rolling_t5 - the trailing-5 window ending t-5 - on 2026-08-13, and
# taking g23.DESIGN would have silently carried this table with it. README 1.8.9's
# numbers are the record of a screen that ranks on the instrument's own loading;
# re-cutting them on a different exposure would not update that record, it would
# replace it with a different question under the same table number. So the pairing
# is written out here, and windows_for below is passed it explicitly.
#
# It is what g23.DESIGN was before that date. Anything else is a change to a frozen
# table, and the check is byte-equality of exposure_screen_grid.csv against the
# pre-2026-08-13 copy.
DESIGN = (("fixed", "pre"), ("fixed_9397", "post"),
          ("rolling", "pre"), ("rolling", "post"))
ERA_LABEL = g23.ERA_LABEL
RUNGS = g23.RUNGS
WINDOWS = g23.WINDOWS
OMEGA_COL = g23.OMEGA_COL
DEFAULT_OMEGA = g23.DEFAULT_OMEGA

RTOL_FIT = g23.RTOL_FIT
ATOL_FIT = g23.ATOL_FIT
TOL_TABLE_F = g23.TOL_TABLE_F

# h=4 is not here and h=2 is. Every horizon in this run has a block in
# instrument_ccg_year.parquet (bi.HORIZONS = 1,2,3), so check 3 is the full
# production comparison and 23's recompute-from-source branch never fires; and
# first_stage_table.csv covers all three, so check 6 anchors 27 baseline cells
# instead of 23's nine.
HORIZONS = (1, 2, 3)

# The two tail cuts, as (label, upper-tail mass). Keep county c iff s_c is at or
# below the (1-p) quantile, so a tie AT the cutoff is kept and the drop count is
# approximately, not exactly, p x N. In this tail there are no ties.
SCREEN_CUTS = (("p995", 0.005), ("p99", 0.01))
FAMILIES = ("own", "fix")

# Which fixed window describes each era. This is the `fix` family's whole content
# and it is also the DESIGN pairing, restated in the one form the screen needs.
ERA_WINDOW = {"pre": "fixed", "post": "fixed_9397"}

ALL_SCREENS = ("all",) + tuple(f"sc_{f}_{t}" for f in FAMILIES
                               for t, _ in SCREEN_CUTS)

SCREEN_LABEL = {
    "all": "no screen",
    "sc_own_p995": "drop the top 0.5% of s_c, measured on the row's own exposure",
    "sc_own_p99": "drop the top 1% of s_c, measured on the row's own exposure",
    "sc_fix_p995": "drop the top 0.5% of s_c, measured on the era's fixed window",
    "sc_fix_p99": "drop the top 1% of s_c, measured on the era's fixed window",
}

# Check 7's target. Unlike 23's, these were known before the first run: the tail is
# continuous, so every one of the 48 keys drops exactly the nominal count. A key
# that starts dropping 15 or 17 is a key whose ranking variable acquired ties or
# NaNs, and nothing else in the run would notice.
EXPECT_DROPPED = {"all": 0, "sc_own_p995": 16, "sc_own_p99": 31,
                  "sc_fix_p995": 16, "sc_fix_p99": 31}

# ------------------------------------------------------------------------------
# THIS TABLE'S BASELINES ARE FROZEN CONSTANTS, NOT LIVE JOINS.        2026-08-13
#
# Checks 5 and 6 used to join first_stage_grid.csv and first_stage_table.csv - the
# outputs of 20 and 19 - to assert that this file's unscreened column IS Table VI
# and reproduces Table V. On 2026-08-13 both of those tables moved: Table VI went
# to h=2 and Tables VII-IX with it, and both replaced `rolling` with `rolling_t5`.
# The cells this file needs therefore no longer exist in either file, and a join
# would return an empty merge - which g20._compare correctly reports as a FAILURE.
#
# A frozen table cannot have a moving baseline. So the twelve Table VI cells and
# the twenty-seven Table V cells are copied in here, from the files as they stood
# before that date, and checks 5 and 6 assert against these. The assertion is
# exactly as strong as it was; what changed is that it no longer depends on two
# files that are allowed to move underneath it.
#
# If this table is ever re-cut, these go and the joins come back.
# ------------------------------------------------------------------------------

# Table VI's OLS column at h=1, README 1.8.7. (exposure, era, rung) -> (units,
# n_frame, F_ols).
TABLEVI_H1_OLS = {
    ("fixed", "pre", "county"):               (3094, 68068, 77.07),
    ("fixed", "pre", "commuting zone"):       (738, 16236, 37.58),
    ("fixed", "pre", "state"):                (51, 1122, 45.24),
    ("fixed_9397", "post", "county"):         (3094, 61847, 4.85),
    ("fixed_9397", "post", "commuting zone"): (738, 14758, 6.85),
    ("fixed_9397", "post", "state"):          (51, 1020, 61.51),
    ("rolling", "pre", "county"):             (3094, 68068, 0.30),
    ("rolling", "pre", "commuting zone"):     (738, 16236, 69.33),
    ("rolling", "pre", "state"):              (51, 1122, 134.55),
    ("rolling", "post", "county"):            (3094, 61847, 27.86),
    ("rolling", "post", "commuting zone"):    (738, 14758, 83.97),
    ("rolling", "post", "state"):             (51, 1020, 124.54),
}

# Table V, README 1.8.6, the nine cells 19 published at each of h=1,2,3.
# (exposure, era, rung, h) -> (fs_F, beta, beta_se, n_obs).
#
# STALE, AND KNOWN TO BE. Two independent changes moved this block and it has been
# re-cut for only one of them. The values below are the 2026-08-14 re-freeze for the
# 3,094-CCG positive-employment universe (dc.load_analysis_ccg_map), which is why
# every n_obs here is smaller than the 3,101-universe figures it replaced - 68,184
# -> 68,068 at the pre county cell, and so on down the block.
#
# What they do NOT carry is the EFSY negative-cell deletion of 2026-08-17, which
# moves county employment in 826 county-years over 1981-1997 and therefore moves
# every fit with employment growth on the left, by 4.1e-05 relative at worst against
# this file's 1e-6 tolerance. That re-freeze was measured on the 3,101 universe, so
# its values cannot simply be pasted here either: the two perturbations have to be
# applied together and the block re-cut from a run that has both.
#
# So this check WILL fire on the next run, and firing is correct. Do not silence it
# by copying one side; re-cut it. check_tablev prints the current values at full
# precision on failure for exactly that purpose. 20_first_stage_grid.py's REQUESTED
# is stale for the same reason and was not re-cut at all. README section 4 carries
# both as open.
TABLEV_FROZEN = {
    ("rolling", "pre", "county", 1): (0.302905536844, 0.201781830979, 0.347214047111, 68068),
    ("rolling", "pre", "county", 2): (1.37540077277e-08, 1169.05640929, 9967984.44109, 64974),
    ("rolling", "pre", "county", 3): (0.319270450852, -0.185780519081, 0.422703045762, 61880),
    ("rolling", "pre", "commuting zone", 1): (69.3272815494, 0.212209806034, 0.0840699467854, 16236),
    ("rolling", "pre", "commuting zone", 2): (46.4045362618, 0.314372945841, 0.125648337331, 15498),
    ("rolling", "pre", "commuting zone", 3): (25.4857513319, 0.396140464719, 0.181718336536, 14760),
    ("rolling", "pre", "state", 1): (134.549360853, 0.558960931092, 0.190756074531, 1122),
    ("rolling", "pre", "state", 2): (117.201568866, 0.596753308535, 0.215405439051, 1071),
    ("rolling", "pre", "state", 3): (92.3289751882, 0.609354464953, 0.241667367341, 1020),
    ("rolling", "post", "county", 1): (27.8618003854, -0.0460433739553, 0.0513815486057, 61847),
    ("rolling", "post", "county", 2): (14.5194714174, -0.0986878465613, 0.0647243855547, 55692),
    ("rolling", "post", "county", 3): (5.24290230613, -0.105757621893, 0.0956737838901, 49504),
    ("rolling", "post", "commuting zone", 1): (83.9713201659, -0.0616776953832, 0.269155594323, 14758),
    ("rolling", "post", "commuting zone", 2): (73.3817534839, -0.32756246791, 0.166445901784, 13284),
    ("rolling", "post", "commuting zone", 3): (33.4561498517, -0.241936403719, 0.173275202711, 11808),
    ("rolling", "post", "state", 1): (124.541197811, 0.292295533908, 0.210783880229, 1020),
    ("rolling", "post", "state", 2): (113.218811844, 0.384240196441, 0.272880345883, 918),
    ("rolling", "post", "state", 3): (54.495863103, 0.36393632512, 0.28872659452, 816),
    ("fixed_9397", "post", "county", 1): (4.85068791724, -0.0762817066942, 0.0809381906409, 61847),
    ("fixed_9397", "post", "county", 2): (6.17700705689, -0.0583054468595, 0.0556135088085, 55692),
    ("fixed_9397", "post", "county", 3): (6.43402377778, -0.10861071921, 0.066211563752, 49504),
    ("fixed_9397", "post", "commuting zone", 1): (6.85319346089, -0.250827219936, 0.593692317594, 14758),
    ("fixed_9397", "post", "commuting zone", 2): (5.69323391001, -0.292756929755, 0.52057125794, 13284),
    ("fixed_9397", "post", "commuting zone", 3): (6.50823761606, -0.414664667729, 0.44772305444, 11808),
    ("fixed_9397", "post", "state", 1): (61.5061275063, -0.239669565647, 0.231386812793, 1020),
    ("fixed_9397", "post", "state", 2): (40.8629326931, -0.175960878243, 0.237070835192, 918),
    ("fixed_9397", "post", "state", 3): (31.0461988039, -0.123874643262, 0.245750358414, 816),
}

DEMEANER = None

check = g23.check
eras_for = g23.eras_for
build = g23.build


def windows_for(name, h):
    """g23.windows_for on THIS file's design, not on 23's.

    23's version closes over its own module-level DESIGN by default, which is
    rolling_t5. Passing ours is what keeps this table on `rolling`.
    """
    return g23.windows_for(name, h, design=DESIGN)


def fit(frame, fe, weights=None):
    """bi.fit_rung with the demeaner pinned, exactly as 20 and 23 do it."""
    return bi.fit_rung(frame, fe, weights=weights, demeaner=DEMEANER)


def screens_for(exposure):
    """The screens this exposure runs. Three for a fixed window, five for rolling.

    The fix family is dropped where it is redundant rather than fitted and then
    labelled as a separate column, which is what check 9 licenses.
    """
    if exposure == "rolling":
        return ALL_SCREENS
    return ("all",) + tuple(f"sc_own_{t}" for t, _ in SCREEN_CUTS)


def family_of(screen):
    return screen.split("_")[1] if screen != "all" else ""


def cut_of(screen):
    if screen == "all":
        return np.nan
    tag = screen.rsplit("_", 1)[1]
    return 1.0 - dict(SCREEN_CUTS)[tag]


# --------------------------------------------------------------------------------
# The screens
# --------------------------------------------------------------------------------

def fixed_sc(pre):
    """{window name -> s_c}, from the window sums 23's build already returns.

    Identical to bi.fixed_exposure by construction - it is that function's body -
    but taken from `pre` so the screen is ranking the same two components the
    instrument was built from rather than a second pass over the levels.
    """
    out = {}
    for name in WINDOWS:
        p = pre[name]
        with np.errstate(divide="ignore", invalid="ignore"):
            out[name] = (p["pre_D"] / p["pre_Y"]).rename(f"exposure_{name}")
    return out


def ranking(county, sc, exposure, era, h, family):
    """The county-level s_c this (design row, family) ranks on, NaNs dropped.

    For the rolling exposure `own` is the county mean of s_ct over the years the
    era actually estimates - the collapse that turns a county-year loading into the
    county set the screen has to be. It is read off the frame the fits use, whose
    exposure_rolling column build_horizon does NOT seam-mask (only the z's are), so
    the mean is over the era's full span either side of the splice.
    """
    if family == "fix":
        return sc[ERA_WINDOW[era]].dropna()
    if exposure != "rolling":
        return sc[exposure].dropna()
    y0, y1 = eras_for(h)[0][era]
    d = county[(county["h"] == h) & county["year"].between(y0, y1)]
    return d.groupby("ccg")["exposure_rolling"].mean().dropna()


def build_screens(universe, county, pre, horizons):
    """({screen: {(exposure, era, h): frozenset}}, {key: cutoff}).

    `all` also carries the anchor key, so the anchor row looks the set up the same
    way every other row does instead of special-casing it at the call site.
    """
    uni = frozenset(int(c) for c in universe)
    sc = fixed_sc(pre)
    screens = {n: {} for n in ALL_SCREENS}
    cutoffs = {}
    for h in horizons:
        for exposure, era in DESIGN:
            key = (exposure, era, h)
            screens["all"][key] = uni
            for fam in FAMILIES:
                for tag, p in SCREEN_CUTS:
                    name = f"sc_{fam}_{tag}"
                    if name not in screens_for(exposure):
                        continue
                    s = ranking(county, sc, exposure, era, h, fam)
                    cut = float(s.quantile(1.0 - p))
                    drop = {int(c) for c in s.index[s > cut]}
                    screens[name][key] = uni - drop
                    cutoffs[(name,) + key] = cut
        screens["all"][("rolling", sym.ANCHOR_PERIOD, h)] = uni
    return screens, cutoffs


def cost(levels, kept, y0, y1):
    """What a screen drops over one era, in three currencies.

    Defense dollars alone is what 23 reports and it is not enough here. A screen on
    a contracts-over-income RATIO is by construction correlated with the numerator,
    so it can be cheap in counties and ruinous in treatment; income and employment
    are what say whether the sample that remains is still the country.
    """
    d = levels[levels["year"].between(y0, y1)]
    drop = ~d["ccg"].isin(kept)
    out = {}
    for col, tag in (("dollars_real", "dollars"), ("income_real", "income"),
                     ("emp", "emp")):
        tot = float(d[col].sum())
        out[tag] = 100.0 * float(d.loc[drop, col].sum()) / tot if tot else np.nan
    return out


# --------------------------------------------------------------------------------
# The fits
# --------------------------------------------------------------------------------

def fit_grid(county, rungs, pre, horizons, exposures, rung_names, omega,
             screens, cutoffs, levels):
    """One row per exposure x era x screen x h x rung. OLS throughout.

    The screen loop sits INSIDE the era loop, which is where 23 differs: there a
    screen was one set per horizon and could be applied before the era split, here
    it is one set per (exposure, era, h) and cannot. The finiteness screen still
    goes second and still sits outside the era loop, so the ordering 23 states -
    which counties exist, then which of those have a usable row - is unchanged.
    """
    wanted = [(r, c) for r, c in RUNGS if r in set(rung_names)]
    omega_col = OMEGA_COL[omega]
    n_uni = len(screens["all"][("rolling", sym.ANCHOR_PERIOD, horizons[0])])
    rows, repeat_devs, cost_cache = [], [], {}
    for h in horizons:
        for name in exposures:
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
                n_con = len(bi.contaminated_years(pkey, name, h))
                # The anchor is the frozen 1975-1998 convention and exists to
                # reproduce a published number, so it runs unscreened only.
                names = ("all",) if role == "anchor" else screens_for(name)
                for screen in names:
                    key = (name, era, h)
                    kept = screens[screen][key]
                    b = base_full[base_full["ccg"].isin(kept)]
                    b = b[b["year"].between(y0, y1)]
                    if role == "anchor":
                        c = {"dollars": np.nan, "income": np.nan, "emp": np.nan}
                    else:
                        ck = (screen,) + key
                        if ck not in cost_cache:
                            cost_cache[ck] = cost(levels, kept, y0, y1)
                        c = cost_cache[ck]
                    for rung, unit_col in wanted:
                        d = g20.unit_frame(b, rung, unit_col, omega_col)
                        r = fit(d, "unit + year")
                        if role == "variant" and screen == "all" and h == 1:
                            again = fit(d, "unit + year")
                            worst = max(abs(again[k] - r[k])
                                        for k in ("fs_F", "beta", "beta_se"))
                            repeat_devs.append(
                                (f"{name} x {era} x {rung} h={h}", worst))
                        rows.append(dict(
                            role=role, screen=screen,
                            screen_label=SCREEN_LABEL[screen],
                            screen_family=family_of(screen),
                            screen_cut=cut_of(screen),
                            screen_cutoff=cutoffs.get((screen,) + key, np.nan),
                            n_ccg_screen=len(kept),
                            n_ccg_dropped=n_uni - len(kept),
                            pct_dollars_dropped=c["dollars"],
                            pct_income_dropped=c["income"],
                            pct_emp_dropped=c["emp"],
                            era=era, era_label=ERA_LABEL.get(era, era),
                            era_span=f"{y0}-{y1}", exposure=name,
                            exposure_window=g20.window_label(name), rung=rung,
                            n_units=int(d["unit"].nunique()), omega=omega,
                            reg_weights="unweighted", h=h,
                            year_first=y0, year_last=y1, fe="unit+year",
                            contaminated_years=n_con, n_frame=len(d), **r))
    return pd.DataFrame(rows), repeat_devs


# --------------------------------------------------------------------------------
# The checks
# --------------------------------------------------------------------------------

def check_tablevi(failures, grid, rung_names):
    """Check 5. The unscreened h=1 column IS Table VI's OLS column.

    Against TABLEVI_H1_OLS above rather than against 20's live output, for the
    reason that block records: Table VI moved to h=2 on rolling_t5 and this table
    is frozen at h=1 on rolling. F to 0.01, units and n_frame exact.
    """
    d = grid[(grid["role"] == "variant") & (grid["screen"] == "all")
             & (grid["h"] == 1)]
    if d.empty:
        return check(failures, False, 5, "the unscreened h=1 grid is Table VI",
                     "no h=1 rows in this run")
    n = d.set_index(["exposure", "era", "rung"])
    bad, checked = [], 0
    for key, (units, obs, f_ols) in TABLEVI_H1_OLS.items():
        if key[1] not in ERA_LABEL or key[2] not in set(rung_names):
            continue
        checked += 1
        if key not in n.index:
            bad.append(f"{key} MISSING")
            continue
        row = n.loc[key]
        if abs(float(row["fs_F"]) - f_ols) > TOL_TABLE_F:
            bad.append(f"{key} F {float(row['fs_F']):.2f} want {f_ols:.2f}")
        if int(row["n_units"]) != units:
            bad.append(f"{key} units {int(row['n_units'])} want {units}")
        if int(row["n_frame"]) != obs:
            bad.append(f"{key} n_frame {int(row['n_frame'])} want {obs}")
    ok = not bad and checked == len(DESIGN) * len(rung_names)
    check(failures, ok, 5, "the unscreened h=1 grid IS Table VI's OLS column",
          f"{checked} cells" if ok else f"{len(bad)} deviation(s) of {checked}")
    for b in bad:
        print(f"          {b}")
    return ok


def check_tablev(failures, grid, horizons, rung_names):
    """Check 6. The unscreened grid reproduces Table V, at EVERY horizon.

    23's version only ran at h=3 because that was the one horizon it shared with
    19. This run shares all three, so rolling x pre, rolling x post and
    fixed_9397 x post give 9 published cells per horizon - 27 in all - and the
    baseline every screened number is read against is anchored throughout rather
    than at one horizon out of three.

    Against TABLEV_FROZEN rather than against first_stage_table.csv, for the reason
    that block records: 19 now reports h=2,3,4 on rolling_t5, so the `rolling`
    variant rows this check needs are no longer in the file and the merge would be
    empty. The values are that file's own, as of 2026-08-12.
    """
    d = grid[(grid["role"] == "variant") & (grid["screen"] == "all")]
    n = d.set_index(["exposure", "era", "rung", "h"])
    worst_rel, worst_abs, where, checked, bad = 0.0, 0.0, "", 0, []
    for key, (f_f, f_b, f_se, f_n) in TABLEV_FROZEN.items():
        if key[2] not in set(rung_names) or key[3] not in set(horizons):
            continue
        checked += 1
        if key not in n.index:
            bad.append(f"{key} MISSING")
            continue
        row = n.loc[key]
        if int(row["n_obs"]) != f_n:
            bad.append(f"{key} n_obs {int(row['n_obs'])} want {f_n}")
        for col, frozen in (("fs_F", f_f), ("beta", f_b), ("beta_se", f_se)):
            dev = abs(float(row[col]) - frozen)
            rel = dev / (ATOL_FIT + abs(frozen))
            if rel > worst_rel:
                worst_rel, where = rel, f"{col} at {key[2]} h={key[3]}"
            worst_abs = max(worst_abs, dev)
    shared = sorted(set(horizons) & {k[3] for k in TABLEV_FROZEN})
    ok = not bad and checked > 0 and worst_rel < RTOL_FIT
    check(failures, ok, 6, f"Table V's cells reproduce 19 at h={shared}",
          f"{checked} cells, rel {worst_rel:.1e} abs {worst_abs:.1e}"
          + (f" ({where})" if where else "")
          + (f", {len(bad)} bad" if bad else ""))
    for b in bad:
        print(f"          {b}")
    if not ok:
        # A stale anchor and a broken fit look identical in a single number, and the
        # difference is whether the panel was MEANT to move. So print what this run
        # actually produced, at full precision: a deliberate re-freeze becomes a
        # copy-paste that can be diffed against the block above, and an accidental
        # break becomes obvious because the values are nowhere near the frozen ones.
        # Printing is not accepting - the check still fails and nothing is written.
        print("          current values, for a DELIBERATE re-freeze of TABLEV_FROZEN:")
        for key in TABLEV_FROZEN:
            if key[2] not in set(rung_names) or key[3] not in set(horizons):
                continue
            if key not in n.index:
                continue
            r = n.loc[key]
            print(f"    {key!r}: ({float(r['fs_F'])!r}, {float(r['beta'])!r}, "
                  f"{float(r['beta_se'])!r}, {int(r['n_obs'])}),")
    return ok


def check_screens(failures, screens, horizons):
    """Check 7. Every screen is a declared subset that bites and does not empty."""
    uni = screens["all"][("rolling", sym.ANCHOR_PERIOD, horizons[0])]
    bad, n_keys = [], 0
    for name in ALL_SCREENS:
        for key, kept in screens[name].items():
            n_keys += 1
            if not kept <= uni:
                bad.append(f"{name} {key} is not a subset of the universe")
            dropped = len(uni) - len(kept)
            if dropped != EXPECT_DROPPED[name]:
                bad.append(f"{name} {key} drops {dropped}, "
                           f"declared {EXPECT_DROPPED[name]}")
            if not kept:
                bad.append(f"{name} {key} is empty")
    ok = not bad
    check(failures, ok, 7, "every screen is the declared subset of the universe",
          f"{n_keys} keys" if ok else f"{len(bad)} deviation(s)")
    for b in bad:
        print(f"          {b}")
    return ok


def check_applied(failures, grid, screens):
    """Check 8. The county rung's unit count tracks the screen, key by key.

    A screen that silently failed to apply would produce a full-sample fit under a
    screened label, which reads as "the tail changes nothing" - and that is a
    result this table could plausibly produce for real, which is exactly why the
    bug has to be excluded by assertion rather than by the number looking wrong.
    """
    d = grid[(grid["role"] == "variant") & (grid["rung"] == "county")]
    base = {k: int(v["n_units"].iloc[0]) for k, v in
            d[d["screen"] == "all"].groupby(["exposure", "era", "h"])}
    bad = []
    for (screen, exposure, era, h), block in d.groupby(
            ["screen", "exposure", "era", "h"]):
        n = int(block["n_units"].max())
        cap = len(screens[screen][(exposure, era, h)])
        if n > cap:
            bad.append(f"{screen} {exposure} x {era} h={h}: {n} units > {cap}")
        if screen != "all" and n >= base[(exposure, era, h)]:
            bad.append(f"{screen} {exposure} x {era} h={h}: the screen did not bite")
    ok = not bad
    check(failures, ok, 8, "the county unit count tracks each screen",
          f"{len(base)} baselines" if ok else f"{len(bad)} deviation(s)")
    for b in bad:
        print(f"          {b}")
    return ok


def check_families(failures, county, pre, screens, horizons):
    """Check 9. The two families coincide where they must, at set level.

    Two statements, and the file's economy rests on both:

      A  on fixed x pre and fixed_9397 x post the fix rule IS the own rule, which
         is why those cells run three screens instead of five. Asserted by building
         the fix set on the fly and comparing.
      B  a rolling cell's fix set equals its paired fixed cell's own set, which is
         the whole point of the family - it holds the sample identical while the
         exposure changes.

    Without these the removed columns would be an unverified assumption and a
    mis-keyed family would be invisible: it would produce a plausible table.
    """
    sc = fixed_sc(pre)
    bad = []
    for h in horizons:
        for exposure, era in DESIGN:
            for tag, p in SCREEN_CUTS:
                s = ranking(county, sc, exposure, era, h, "fix")
                cut = float(s.quantile(1.0 - p))
                want = (screens["all"][(exposure, era, h)]
                        - {int(c) for c in s.index[s > cut]})
                if exposure == "rolling":
                    got = screens[f"sc_fix_{tag}"][(exposure, era, h)]
                    pair = screens[f"sc_own_{tag}"][(ERA_WINDOW[era], era, h)]
                    if got != want:
                        bad.append(f"B {exposure} x {era} h={h} {tag}: "
                                   f"fix set is not the fix rule")
                    if got != pair:
                        bad.append(f"B {exposure} x {era} h={h} {tag}: "
                                   f"fix set != {ERA_WINDOW[era]}'s own set")
                else:
                    got = screens[f"sc_own_{tag}"][(exposure, era, h)]
                    if got != want:
                        bad.append(f"A {exposure} x {era} h={h} {tag}: "
                                   f"own set != fix set on a fixed exposure")
    ok = not bad
    check(failures, ok, 9, "the own and fix families coincide where they must",
          f"{len(DESIGN) * len(horizons) * len(SCREEN_CUTS)} pairings"
          if ok else f"{len(bad)} deviation(s)")
    for b in bad:
        print(f"          {b}")
    return ok


def check_rest(failures, grid, repeat_devs, horizons, rung_names, canonical):
    """Checks 10, 11 and 12."""
    d = grid[grid["role"] == "variant"]
    check(failures, int(d["contaminated_years"].max()) == 0, 10,
          "zero contaminated years on every design row",
          f"max {int(d['contaminated_years'].max())} over {len(d)} rows")

    per_h = sum(len(screens_for(e)) for e, _ in DESIGN) * len(rung_names)
    w_var = per_h * len(horizons)
    w_anc = len(horizons) * len(rung_names)
    n_var = int((grid["role"] == "variant").sum())
    n_anc = int((grid["role"] == "anchor").sum())
    label = ("row count is the announced grid" if canonical
             else "row count is the requested subset")
    check(failures, n_var == w_var and n_anc == w_anc, 11, label,
          f"{n_var} of {w_var} design, {n_anc} of {w_anc} anchor")

    worst = max((v for _, v in repeat_devs), default=np.inf)
    where = max(repeat_devs, key=lambda t: t[1])[0] if repeat_devs else "none"
    check(failures, bool(repeat_devs) and worst == 0.0, 12,
          "a refit is bit-identical, the demeaner being pinned",
          f"{len(repeat_devs)} cells refitted, max dev {worst:.1e} ({where})")


# --------------------------------------------------------------------------------
# The report
# --------------------------------------------------------------------------------

# Table X's rows. The own/fix distinction lives here, in the ROWS, and not in the
# columns - which is what lets the fixed exposures carry no fix column at all
# instead of two duplicates of their own.
ROW_GROUPS = (("pre", "fixed", "own"),
              ("post", "fixed_9397", "own"),
              ("pre", "rolling", "own"),
              ("pre", "rolling", "fix"),
              ("post", "rolling", "own"),
              ("post", "rolling", "fix"))

SHORT = {"fixed": "fixed 1969-74", "fixed_9397": "fixed 1993-97",
         "rolling": "rolling K=5"}


def group_label(era, exposure, family):
    if exposure == "rolling":
        span = ("own span" if family == "own"
                else SHORT[ERA_WINDOW[era]].replace("fixed ", "fix "))
        return f"{SHORT[exposure]}, {span}"
    return SHORT[exposure]


def report(grid, horizons, rung_names):
    """Table X, then the screen table, then the betas.

    Printed even on failure: a run that failed check 5 is exactly the run whose
    numbers need reading.
    """
    d = grid[grid["role"] == "variant"]
    cols = [("all", "all")] + [(f"sc_{{}}_{t}", t) for t, _ in SCREEN_CUTS]

    print("\n  Table X. Screening on the exposure - first-stage F")
    head = f"    {'era':<6}{'exposure':<26}{'rung':<17}"
    for h in horizons:
        for _, tag in cols:
            head += f"{f'h={h} {tag}':>13}"
    print(head)
    for era, exposure, fam in ROW_GROUPS:
        first = True
        for rung, _ in RUNGS:
            if rung not in set(rung_names):
                continue
            line = (f"    {(era if first else ''):<6}"
                    f"{(group_label(era, exposure, fam) if first else ''):<26}"
                    f"{rung:<17}")
            first = False
            for h in horizons:
                for pat, _ in cols:
                    screen = pat.format(fam) if "{}" in pat else pat
                    r = d[(d["h"] == h) & (d["era"] == era)
                          & (d["exposure"] == exposure) & (d["rung"] == rung)
                          & (d["screen"] == screen)]
                    line += (f"{float(r['fs_F'].iloc[0]):>13.2f}" if len(r)
                             else f"{'':>13}")
            print(line)

    print("\n  The screens and what they cost")
    print(f"    {'era':<6}{'exposure':<26}{'screen':<8}{'cutoff s_c':>28}"
          f"{'drop':>6}{'% defense $':>26}{'% inc':>8}{'% emp':>8}")
    for era, exposure, fam in ROW_GROUPS:
        first = True
        for tag, _ in SCREEN_CUTS:
            screen = f"sc_{fam}_{tag}"
            cut, dol, n = [], [], None
            inc = emp = np.nan
            for h in horizons:
                r = d[(d["h"] == h) & (d["era"] == era)
                      & (d["exposure"] == exposure) & (d["screen"] == screen)]
                if not len(r):
                    continue
                cut.append(f"{float(r['screen_cutoff'].iloc[0]):.4f}")
                dol.append(f"{float(r['pct_dollars_dropped'].iloc[0]):.2f}")
                n = int(r["n_ccg_dropped"].iloc[0])
                if h == horizons[0]:
                    inc = float(r["pct_income_dropped"].iloc[0])
                    emp = float(r["pct_emp_dropped"].iloc[0])
            if n is None:
                continue
            cut = "/".join(dict.fromkeys(cut))
            print(f"    {(era if first else ''):<6}"
                  f"{(group_label(era, exposure, fam) if first else ''):<26}"
                  f"{tag:<8}{cut:>28}{n:>6}{'/'.join(dol):>26}"
                  f"{inc:>8.2f}{emp:>8.2f}")
            first = False
    hs = "/".join(map(str, horizons))
    print(f"    cutoffs and % defense $ are h={hs}; "
          f"% inc and % emp are h={horizons[0]}")

    print("\n  2SLS beta (se), the same fits")
    for h in horizons:
        for era, exposure, fam in ROW_GROUPS:
            s = d[(d["exposure"] == exposure) & (d["era"] == era)
                  & (d["h"] == h)]
            if s.empty:
                continue
            print(f"    h={h}  {group_label(era, exposure, fam)} x {era} "
                  f"({s['era_span'].iloc[0]})")
            for rung, _ in RUNGS:
                r = s[s["rung"] == rung]
                if r.empty:
                    continue
                parts = []
                for pat, tag in cols:
                    screen = pat.format(fam) if "{}" in pat else pat
                    v = r[r["screen"] == screen]
                    if len(v):
                        parts.append(f"{tag} {float(v['beta'].iloc[0]):+.3f} "
                                     f"({float(v['beta_se'].iloc[0]):.3f})")
                print(f"      {rung:<17}" + "   ".join(parts))


# --------------------------------------------------------------------------------

def main():
    global DEMEANER
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--horizon", type=int, nargs="+", default=list(HORIZONS),
                    help="horizons to report; each sets its own era bounds")
    ap.add_argument("--rung", nargs="+", default=[r for r, _ in RUNGS],
                    choices=[r for r, _ in RUNGS])
    ap.add_argument("--omega", choices=sorted(OMEGA_COL), default=DEFAULT_OMEGA,
                    help="the AGGREGATION weight above the county. There is no "
                         "regression weight: this table is OLS")
    args = ap.parse_args()

    horizons = sorted(set(args.horizon))
    canonical = (horizons == sorted(HORIZONS)
                 and set(args.rung) == {r for r, _ in RUNGS}
                 and args.omega == DEFAULT_OMEGA)
    sfx = "" if canonical else ("__h" + "-".join(map(str, horizons)) + "_"
                                + "-".join(sorted(args.rung)).replace(" ", "")
                                + "_" + args.omega)

    tee = sym.Tee(dc.LOG_DIR / f"24_exposure_screen_grid{sfx}.log")
    try:
        import pyfixest as pf
        DEMEANER = pf.MapDemeaner(fixef_tol=1e-12, fixef_maxiter=100_000)
        g23.DEMEANER = DEMEANER      # 23's check_build and check_anchor use its fit

        failures = []
        print("=" * 78)
        print("Table VI's grid, OLS only, screening on the exposure's right tail")
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
        print(f"  screens    fixed cells  {', '.join(screens_for('fixed'))}")
        print(f"             rolling      {', '.join(screens_for('rolling'))}")
        print(f"  omega      {args.omega} ({OMEGA_COL[args.omega]}) above the "
              f"county, omega = 1 at the county")
        print(f"  regression OLS only - section 1.8.7 is the record of the "
              f"weighted variant")
        print(f"  demeaner   PINNED, fixef_tol=1e-12, fixef_maxiter=100,000")

        print("\n[1] the eras, and the registration")
        g23.check_eras(failures, horizons)
        before = g20.register_variants(horizons)
        ok, detail = g23.only_added(before)
        check(failures, ok, 2, "registering the variants only ADDED keys", detail)

        print("\n[2] the instrument, every exposure, in memory")
        exposures = list(dict.fromkeys(e for e, _ in DESIGN))
        county, levels, universe, pre = build(horizons)
        rungs = bi.load_rungs(universe)

        print("\n[3] the build, against production")
        g23.check_build(failures, county, horizons)

        print("\n[4] the screens")
        screens, cutoffs = build_screens(universe, county, pre, horizons)
        for h in horizons:
            for exposure, era in DESIGN:
                for screen in screens_for(exposure):
                    if screen == "all":
                        continue
                    key = (exposure, era, h)
                    kept = screens[screen][key]
                    y0, y1 = eras_for(h)[0][era]
                    c = cost(levels, kept, y0, y1)
                    print(f"    h={h} {exposure:<11}{era:<5}{screen:<12}"
                          f"cutoff {cutoffs[(screen,) + key]:.4f}   "
                          f"keep {len(kept):>5,}   "
                          f"drop {len(screens['all'][key]) - len(kept):>3,}   "
                          f"drops {c['dollars']:5.2f}% of defense $, "
                          f"{c['income']:4.2f}% of income, "
                          f"{c['emp']:4.2f}% of employment")

        print("\n[5] the fits")
        grid, repeat_devs = fit_grid(county, rungs, pre, horizons, exposures,
                                     args.rung, args.omega, screens, cutoffs,
                                     levels)
        print(f"    {len(grid)} fits: "
              f"{int((grid['role'] == 'variant').sum())} design, "
              f"{int((grid['role'] == 'anchor').sum())} anchor")

        print("\n[6] the checks")
        g23.check_anchor(failures, grid, horizons, args.rung, args.omega)
        check_tablevi(failures, grid, args.rung)
        check_tablev(failures, grid, horizons, args.rung)
        check_screens(failures, screens, horizons)
        check_applied(failures, grid, screens)
        check_families(failures, county, pre, screens, horizons)
        check_rest(failures, grid, repeat_devs, horizons, args.rung, canonical)

        if failures:
            print(f"\n{len(failures)} CHECK(S) FAILED, nothing written")
            for num, msg in failures:
                print(f"    [{num}] {msg}")
            report(grid, horizons, args.rung)
            return 1

        if not canonical:
            print(f"\n  PARTIAL RUN. The canonical outputs are left untouched; "
                  f"this run writes to *{sfx}.* instead.")

        dc.LOG_DIR.mkdir(parents=True, exist_ok=True)
        out = dc.LOG_DIR / f"exposure_screen_grid{sfx}.csv"
        grid.to_csv(out, index=False)
        print(f"\nall checks passed   {len(grid)} fits -> {out}")
        report(grid, horizons, args.rung)
        return 0
    finally:
        tee.close()


if __name__ == "__main__":
    raise SystemExit(main())
