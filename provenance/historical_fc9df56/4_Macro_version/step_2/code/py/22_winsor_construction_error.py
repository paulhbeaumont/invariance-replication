#!/usr/bin/env python3
"""Plan section 2.2: the winsorization construction error, going DOWN.

INVARIANCE_TEST_PROTOCOL.md A.3.3, mirrored. A paper reports a county result and a
county-by-industry result and winsorizes both, each on its own distribution. The
two caps come from different distributions, so different observations bind and by
different amounts, the cells stop summing to the county, and the two coefficients
are no longer estimated on the same variable. Burchardi, Chaney, Hassan, Tarquinio
and Terry (AER 2026) is the worked case going UP and cost 59% of their state
coefficient; this is the same trap going down.

WE DO NOT WINSORIZE OUR OWN GROWTH RATES. A.3.2 forbids it and nothing here does
it. What is built is the pair of outcomes a REPLICATED design would produce, and
the wedge between them. Standard growth throughout, because that is the only one
of the three anybody winsorizes: midpoint is bounded on [-2, 2] and log compresses
its own tail.

REBUILT ON THE CANDIDATE INSTRUMENT, 2026-08-18. Two vintages were discarded to get
here and neither is revived:

    withdrawn   h=2, z_fixed_7074 from 14_build_instrument.py, 1977-1997 overlapping
                windows, the employment build that carried EFSY's negative cells,
                p1/p99 by year. That is what this file used to be.
    withdrawn   h=2, zfixed4 on 41's symmetric nonoverlapping panel. Its instrument
                divides by the FOCAL COUNTY'S OWN income; 29_candidate_iv_mpgr_check.py
                names that as the flaw the candidate package corrects. The TableW*.tex
                that were on disk before this run came from it.

    current     y_income  = sum_{t+1}^{t+h} D_ct / PI_ct                 the regressor
                z_income_fixed5 = s_c^{5yr} * (sum_{t+1}^{t+h} D^nat) / PI^nat
                                                                         the instrument
                h = 4 headline, h = 2 and 3 estimated and reported but not typeset
                nonoverlapping start years, pre AND post, neither window spanning 1998
                winsor4 method(iq) level(5) within start year, Stata's own percentile

This is an ESTIMAND CHANGE and it was the requested one. The regressor went from a
backward flow ratio to a forward cumulative spending-to-income ratio and the horizon
from 2 to 4, so every number this exhibit ever reported is superseded.

THE LEFT-HAND SIDES. Forward-dated: the frame row is keyed on the OUTCOME year
t + h, the base is E_c,t and the current value is E_c,t+h. Four are typeset:

    g                (E_c,t+h - E_c,t) / E_c,t              uncapped county
    gA               cap(g) on the COUNTY distribution      the county cap
    gB_cells_entry   gB_cells + entry                       cells capped on their own
    gD               gA - gB_cells_entry                    THE CONSTRUCTION ERROR

and three are computed, estimated, printed and written to the CSV but not typeset:

    gB_cells         Sum w cap(g_cell)                      cell-capped, entrants absent
    entry            Sum_{E_m,t=0} E_m,t+h / E_c,t          entrant mass, never capped
    gD_noentry       gA - gB_cells = gD + entry             the entry-excluded wedge

THE CAP IS NOT SPLIT BY LEVEL. `cap_cty = gA - g` and `cap_cell = Sum w g - gB_cells`
are not constructed and `beta_D = beta_cty + beta_cell` is not run, on instruction:
which level's cap moved the coefficient is not the question this exhibit asks, and
g_D is the answer either way.

WHY g~^B CARRIES + entry. Standard growth decomposes as

    Delta^s S_ct = Sum_{m: E_m,t > 0} w_m Delta^s S_cmt + E_ct / S_c,t

- the cell sum runs only over cells with a POSITIVE BASE, so entrants sit outside it
and are recovered by the last term. Drop it and Sum w g_cell fails to reconstruct g
EVEN WITH NOTHING CAPPED, and g_D would be dominated by entrants, which have no
growth rate, are in neither cutoff distribution, and are identical in both
constructions. Carrying it is what makes the null gate bite: no cap anywhere means
g~^B == g, so g_D == 0 and every non-zero g_D is the cap and nothing else.

AND gA NEEDS NO ENTRY TERM, because it already carries entrants. g is the county's
own rate and E_c,t+h is a TOTAL, so a cell that entered is already inside it; entry
becomes a separate term only when the county is REBUILT from cell growth rates,
where a zero-base cell has no rate to contribute. Both sides therefore carry the
same entrant mass - which is exactly what the null gate pins. Treat entrants
differently on the two sides and that check fails at the SIZE OF THE ENTRY TERM
rather than at machine noise.

entry IS CELL ENTRY INSIDE A SURVIVING COUNTY, not county entry netted out. Three
disjoint cases and only the third is in the term:

    E_c,t = 0, E_c,t+h > 0    a COUNTY entrant   dropped: no county growth rate
    E_c,t = 0, E_c,t+h = 0    county absent      dropped, same screen
    E_c,t > 0, E_m,t = 0 < E_m,t+h               THIS is entry

County entrants leave the estimation sample entirely and are counted and named
below as an A.3.1 declared deviation. E_c,t+h = 0 with E_c,t > 0 is NOT dropped:
that is total county exit, g = -1, and it belongs in the cutoff distribution.

The one asymmetry left is that entry is added UNCAPPED while cell entrants inside g
are exposed to the county cap. That is the replicated practice and not our choice: a
paper caps its county growth rate as it finds it, and a cell with no base never
enters the cell cutoff distribution. 36_table_standard.py does cap E_c for its own
constr_err; this file does not, because entry is a level ratio and not a growth rate,
so A.3.2's question does not arise for it.

WHERE THE DATA COMES FROM, AND WHY THAT IS THE PROVENANCE GATE. Nothing is rebuilt
here. The frame and the cells are read from the cache mpgr_disagg.py writes and
Tables 1-5 estimate on, and the manifest's source fingerprint is re-validated on the
way in. This exhibit was built once on 2026-08-07 and withdrawn on 2026-08-10 because
the regressor was wrong, and not one identity in this file can detect that: every one
of them is internal to the constructed outcome or a linearity property of the
estimator, and all of them passed verbatim on the withdrawn panel. Reading the same
bytes as the headline tables is the strongest available answer to that, and it is
stronger than the old cross-check against a single step 1 coefficient.

22 CANNOT IMPORT mpgr_disagg. 27_cell_construction_checks.py imports THIS file, and
28, 29 and mpgr_disagg all reach it through 27, so the import would be circular. The
cache is therefore read as a FILE FORMAT rather than through its writer, and the
handful of names those three still need are kept in the LEGACY block below.

Output: dc.LOG_DIR/winsor_construction_error.csv
        dc.LOG_DIR/winsor_cutoffs.csv
        dc.LOG_DIR/22_winsor_construction_error.log
        dc.PANEL_OUT_DIR.parent/macro_step2/winsor_county_year.parquet
        step_2/result/TableW1..W6, coeff_winsor.tex, _preview_winsor.tex
"""

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
STEP2_CODE = HERE.parent
STEP1_CODE = HERE.parents[3] / "step_1" / "code" / "py"
REPO_ROOT = HERE.parents[4]
RESULT_DIR = HERE.parents[2] / "result"
CANDIDATE = REPO_ROOT / "Candidate for first stage"

sys.path.insert(0, str(STEP1_CODE))
sys.path.insert(0, str(STEP2_CODE))
import dcads_common as dc  # noqa: E402
from out_paths import SCRATCH_ENV, scratch_root  # noqa: E402


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Step 2 writes beside step 1's panel rather than inside it, and the constant is
# defined HERE: step 1's shared module is not edited from step 2.
STEP2_OUT = dc.PANEL_OUT_DIR.parent / "macro_step2"

failures = []


def check(ok, label, detail=""):
    """One line per check, and the run exits non-zero if any of them is False."""
    print(f"    [{'ok ' if ok else 'FAIL'}] {label}" + (f" - {detail}" if detail else ""))
    if not ok:
        failures.append((label, detail))
    return ok


class Tee:
    """The Verification block invokes the script with no shell redirection, so the
    script tees its own stdout and stderr into the log it declares."""

    def __init__(self, path):
        self.f = open(path, "w")
        self.out, self.err = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = self

    def write(self, s):
        self.out.write(s)
        self.f.write(s)
        # Flushed per write, not per buffer. Python buffers stdout whenever it is
        # not a terminal, so an unattended run would leave both the log and the
        # redirect empty until the process exited - and a run that hung would look
        # exactly like a run that had not started.
        self.f.flush()
        self.out.flush()

    def flush(self):
        self.out.flush()
        self.f.flush()

    def close(self):
        sys.stdout, sys.stderr = self.out, self.err
        self.f.close()


def sha256(path, cap=64 << 20):
    """First 64 MB is enough to identify a vintage and keeps the gate cheap."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(cap))
    return h.hexdigest()[:16]


def demean(X, f1, f2, wt, tol=1e-12, maxit=2000):
    """Weighted alternating projections on two fixed effects.

    The demeaner behind the bootstrap, where 200 replicates x 7 left-hand sides
    through pyfixest would cost hours and this costs seconds. It is validated
    against pyfixest on the real sample before any bootstrap number is reported,
    which is what TOL_DEMEAN is for.

    THE TOLERANCE IS RELATIVE TO THE COLUMN'S OWN SCALE, which is what reghdfe
    and lfe do and what makes the stopping rule reachable. Demeaning is linear,
    so demean(y / s) == demean(y) / s exactly, and a criterion stated in units of
    y is therefore the same criterion at every scale. An ABSOLUTE criterion is
    not: standard growth `(E_t+h - E_t) / E_t` on cells with a near-zero base
    reaches 1.5e8 here, where a double's own resolution is 1.5e8 * 2.2e-16 =
    3.3e-8, so an absolute 1e-12 asks for four orders more precision than the
    arithmetic can express and cannot be met at any iteration count.

    Exhausting `maxit` RAISES. A demeaner that returns whatever it happens to
    hold when the loop runs out hands back an unconverged coefficient with
    nothing in the output to say so.
    """
    X = np.array(X, dtype=np.float64, copy=True)
    n1, n2 = f1.max() + 1, f2.max() + 1
    s1 = np.bincount(f1, weights=wt, minlength=n1)
    s2 = np.bincount(f2, weights=wt, minlength=n2)
    s1[s1 == 0], s2[s2 == 0] = 1.0, 1.0
    scale = float(np.abs(X[:, 0]).max())
    thresh = tol * scale if scale > 0.0 else tol
    for it in range(1, maxit + 1):
        prev = X[:, 0].copy()
        for f, s, n in ((f1, s1, n1), (f2, s2, n2)):
            for j in range(X.shape[1]):
                m = np.bincount(f, weights=wt * X[:, j], minlength=n) / s
                X[:, j] -= m[f]
        moved = float(np.abs(X[:, 0] - prev).max())
        if moved < thresh:
            return X
    raise RuntimeError(
        f"demean did not converge in {maxit} sweeps on {len(X):,} rows: last "
        f"sweep moved {moved:.3e} against a threshold of {thresh:.3e} "
        f"(tol {tol:g} relative to a column scale of {scale:.3e}). The "
        f"coefficient this would produce is unconverged and is not returned.")


def _stars(b, se):
    if not (se > 0) or not np.isfinite(b):
        return ""
    t = abs(b / se)
    return r"\sym{***}" if t > 2.576 else r"\sym{**}" if t > 1.96 else \
           r"\sym{*}" if t > 1.645 else ""


def load_rung(rung, years):
    """One float array of employment per year, on a support asserted identical.

    11_build_employment_cells.py writes one file per year over a support that is
    the same in every year and in the same row order, so the t-h lag is an array
    shift. That is asserted rather than assumed, because the whole panel
    misaligns silently if it ever stops being true.

    NOTHING IN THIS FILE'S LIVE PATH CALLS THIS ANY MORE - the cells come from
    the cache. It is kept because mpgr_disagg.py and 28_table_d1.py call it.
    """
    emp, ccg0, code0 = {}, None, None
    for y in years:
        f = pd.read_parquet(dc.EMP_OUT_DIR / rung / f"y{y}.parquet",
                            columns=["ccg", "code", "emp"])
        c, k = f["ccg"].to_numpy(np.int64), f["code"].to_numpy(np.int64)
        if ccg0 is None:
            ccg0, code0 = c, k
        elif not (np.array_equal(c, ccg0) and np.array_equal(k, code0)):
            check(False, f"cell support identical in every year ({rung})",
                  f"year {y} differs from {years[0]}")
            raise SystemExit(1)
        emp[y] = f["emp"].to_numpy(np.float64)
    check(True, f"cell support identical in every year, {years[0]}-{years[-1]} ({rung})",
          f"{len(ccg0):,} rows/year, {len(np.unique(ccg0)):,} CCGs, "
          f"{len(np.unique(code0)):,} codes")
    return emp, ccg0


# ================================================================================
# LEGACY. The fixed_7074 machinery, kept because 27_cell_construction_checks.py
# and 28_table_d1.py bind these names at module level and both still import this
# file. NOTHING BELOW THIS LINE RUNS IN THIS EXHIBIT ANY MORE. Both of those
# scripts are stale on the current instrument and the current employment build -
# step_2/README.md section 4 records it - and deleting these names would turn that
# staleness into an ImportError at the top of every step 2 script instead.
# ================================================================================

# 14 owns the instrument build. This is the legacy path's own copy of it, imported
# only so build_exposure_panel below still resolves. 14's writers live behind its
# main(), which exec_module does not run, so this import writes nothing.
bi = _load("build_instrument", STEP1_CODE / "14_build_instrument.py")

FLOOR0_ACTIONS = 10_859_971
FLOOR0_DOLLARS_BN = 5968.5168
STEP1_BETA0 = 0.145009
STEP1_SE0 = 0.044408
STEP1_F0 = 61.4177
STEP1_N0 = 65_085
EXPOSURE_DEFAULT = "fixed_7074"
NEW_WINDOWS = {"fixed_7074": (1970, 1974)}
TOL_PROV = 1e-12
TOL_FLOW = 1.0
TOL_STEP1 = 1e-3
TOL_SHOCK = 0.0

EFSY_FIRST_YEAR = 1975
CONTRACT_LAST_YEAR = 2006
NAICS_BREAK = 1998      # CBP goes native NAICS; before it EFSY imputes from SIC

RUNGS_ALL = ("naics4_alloc", "naics2")


def periods_for(h):
    """LEGACY. The overlapping-window era split the fixed_7074 build used."""
    return {"pre": (EFSY_FIRST_YEAR + h, NAICS_BREAK - 1),
            "post": (NAICS_BREAK + h, CONTRACT_LAST_YEAR)}, \
           (NAICS_BREAK, NAICS_BREAK + h - 1)


def provenance_gate(h):
    """LEGACY. The fixed_7074 build's regressor gate."""
    print("\n[-1] provenance of the regressor and the vintage of both inputs")
    ivp = dc.PANEL_OUT_DIR / "instrument_ccg_year.parquet"
    cpp = dc.PANEL_OUT_DIR / "county_panel.parquet"
    for p in (ivp, cpp):
        st = p.stat()
        print(f"    {p.name}: {st.st_size:,} bytes, mtime "
              f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(st.st_mtime))}, "
              f"sha256[:16] {sha256(p)}")
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=dc.PROJECT_ROOT,
                             capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        sha = "unknown"
    import pyfixest as pf
    print(f"    git {sha}, pyfixest {pf.__version__}, pandas {pd.__version__}, "
          f"numpy {np.__version__}")

    iv = pd.read_parquet(ivp)
    cp = pd.read_parquet(cpp)
    check(set(iv["h"].unique()) >= {h}, f"the instrument file carries h={h}",
          f"horizons present {sorted(iv['h'].unique())}")

    m = cp.merge(iv[iv["h"] == 1][["ccg", "year", "shock"]], on=["ccg", "year"],
                 suffixes=("_cp", "_iv"))
    ok = m["shock_iv"].notna()
    cum = m["dollars_real"] / m["income_real_lag"]
    flow = (m["dollars_real"] - m["dollars_real_lag"]) / m["income_real_lag"]
    d_cum = float((m.loc[ok, "shock_iv"] - cum[ok]).abs().max())
    d_flow = float((m.loc[ok, "shock_iv"] - flow[ok]).abs().max())
    check(d_cum < TOL_PROV, "the h=1 shock is the CUMULATIVE dG / lagged income",
          f"max dev {d_cum:.3e} < {TOL_PROV:.0e}")
    check(d_flow > TOL_FLOW, "and is NOT the flow difference D_t - D_t-h",
          f"max dev {d_flow:.3e} > {TOL_FLOW:.0e}; rerun 13 and 14 without "
          f"--delta flow if this fails")

    era = cp[cp["year"] <= CONTRACT_LAST_YEAR]
    n_act = int(era["n_actions"].sum())
    dol_bn = float(era["dollars_real"].sum() / 1e9)
    check(n_act == FLOOR0_ACTIONS and abs(dol_bn - FLOOR0_DOLLARS_BN) < 0.01,
          f"no dollar floor anywhere (FY<={CONTRACT_LAST_YEAR})",
          f"{n_act:,} actions, ${dol_bn:,.1f}bn against step 1 section 1.7.2's "
          f"{FLOOR0_ACTIONS:,} / ${FLOOR0_DOLLARS_BN:,.1f}bn")
    check(bool(cp.loc[cp["year"] > CONTRACT_LAST_YEAR, "n_actions"].isna().all()),
          f"post-FY{CONTRACT_LAST_YEAR} carries no action count, as designed",
          "n_actions is <NA> there; its source has none, and 0 would be summed")
    return iv, cp


def only_added(before):
    """LEGACY. Every pre-existing key of bi still maps to its original value."""
    fw, pe, ad = before
    bad = []
    for k, v in fw.items():
        if bi.FIXED_WINDOWS.get(k) != v:
            bad.append(f"FIXED_WINDOWS[{k}]")
    for k, v in pe.items():
        if bi.PERIODS.get(k) != v:
            bad.append(f"PERIODS[{k}]")
    for k, v in ad.items():
        if set(bi.ADMISSIBLE.get(k, ())) != v:
            bad.append(f"ADMISSIBLE[{k}]")
    return not bad, ("nothing redefined" if not bad else ", ".join(bad))


def build_exposure_panel(h, exposure, iv_disk, era):
    """LEGACY. 14's production build, in memory, with the requested exposure."""
    print(f"\n[-0] the {exposure} exposure, built in memory")
    if f"z_{exposure}" in iv_disk.columns:
        print(f"    z_{exposure} is already in the parquet; using it unchanged")
        return iv_disk
    if exposure not in NEW_WINDOWS:
        check(False, f"-0  {exposure} is neither on disk nor a declared window",
              f"declared: {sorted(NEW_WINDOWS)}")
        raise SystemExit(1)
    window = NEW_WINDOWS[exposure]

    before = (dict(bi.FIXED_WINDOWS), dict(bi.PERIODS),
              {k: set(v) for k, v in bi.ADMISSIBLE.items()})
    bi.FIXED_WINDOWS.update({exposure: window})
    ok, detail = only_added(before)
    check(ok, "-0a registering the exposure only ADDED to 14's dicts", detail)

    levels, universe, _ = bi.build_levels(bi.DEFAULT_FLOOR, True,
                                          bi.EXPOSURE_FIRST, bi.LAST_YEAR)
    emp = pd.read_parquet(dc.EMP_OUT_DIR / "efsy_ccg_year.parquet",
                          columns=["ccg", "year", "emp"])
    levels = levels.merge(emp, on=["ccg", "year"], how="left")
    levels = levels.sort_values(["ccg", "year"]).reset_index(drop=True)

    s = bi.fixed_exposure(levels, window, exposure)
    print(f"    {exposure} {window[0]}-{window[1]}   mean {s.mean():.5f}  "
          f"sd {s.std():.5f}  {int((s == 0).sum()):,} counties at zero, "
          f"{int(s.isna().sum())} with no income denominator")

    p = bi.build_horizon(levels, {exposure: s}, h, bi.FIRST_YEAR, bi.LAST_YEAR,
                         True, delta=bi.DEFAULT_DELTA)

    a = p[["ccg", "year", "shock"]].rename(columns={"shock": "mem"})
    b = iv_disk[iv_disk["h"] == h][["ccg", "year", "shock"]].rename(
        columns={"shock": "disk"})
    m = a.merge(b, on=["ccg", "year"], how="inner")
    both = m["mem"].notna() & m["disk"].notna()
    worst = float((m.loc[both, "mem"] - m.loc[both, "disk"]).abs().max())
    same_na = bool((m["mem"].isna() == m["disk"].isna()).all())
    check(worst <= TOL_SHOCK and same_na,
          "-0b the in-memory shock IS the parquet's, exactly",
          f"max dev {worst:.3e} over {int(both.sum()):,} rows, "
          f"missingness identical: {same_na}")

    key = f"_step2_h{h}"
    bi.PERIODS[key] = era
    contam = bi.contaminated_years(key, exposure, h)
    del bi.PERIODS[key]
    check(not contam, f"-0c no estimation year reaches into the {exposure} window",
          f"window ends {window[1]}, era {era[0]}-{era[1]}, contaminated {contam}")
    return p


def build_frame(iv, h, exposure, est_years):
    """LEGACY. The fixed_7074 estimation frame, keyed on the outcome year t."""
    print("\n[0] the frozen estimation frame")
    d = iv[iv["h"] == h].copy()
    d["z"] = d[f"z_{exposure}"]
    n_all = len(d)

    cols = ["shock", "emp_growth", "z", "income_real_lag", "emp_lag"]
    keep = np.isfinite(d[cols].to_numpy()).all(axis=1)
    keep &= (d["emp_lag"] > 0).to_numpy() & (d["income_real_lag"] > 0).to_numpy()
    dropped = d[~keep]
    d = d[keep].copy()

    print(f"    {n_all:,} rows at h={h} -> {len(d):,} after step 1's screen")
    outside = dropped[~dropped["year"].isin(est_years)]
    inside = dropped[dropped["year"].isin(est_years)]
    print(f"    {len(outside):,} of the drops are outside every estimation window "
          f"(years {sorted(outside['year'].unique())}) and are arithmetic, not a screen")
    print(f"    A.3.1 deviation, declared: {len(inside):,} county-years inside the "
          f"estimation windows, {inside['ccg'].nunique()} CCGs, dropped because lagged "
          f"employment or income is zero. Standard growth is undefined there.")
    for ccg, grp in inside.groupby("ccg"):
        yrs = sorted(grp["year"])
        print(f"      ccg {ccg}: {len(yrs)} year(s) " + ", ".join(str(y) for y in yrs))

    d = d.sort_values(["ccg", "year"]).reset_index(drop=True)
    d["g"] = d["emp_growth"]
    key = np.ascontiguousarray(d[["ccg", "year"]].to_numpy(np.int64))
    d.attrs["hash"] = hashlib.sha256(key.tobytes()).hexdigest()[:16]
    print(f"    {len(d):,} rows, {d['ccg'].nunique():,} CCGs, "
          f"{d['year'].min()}-{d['year'].max()}, index hash {d.attrs['hash']}")
    check(d["ccg"].nunique() == 3100, "the CCG set is step 1's frozen 3,100 support",
          f"{d['ccg'].nunique():,} CCGs; the universe is 3,101 and Kalawao HI has "
          f"no BEA income denominator by declaration")
    return d


def step1_crosscheck(frame, era):
    """LEGACY. beta_0 against step 1 section 1.8.6 Table V on the fixed_7074 build."""
    print("\n[9] the step 1 cross-check, on the estimation frame itself")
    b, F = fit_block(frame, ["g"], None, True)
    beta, se, n = b["g"]["beta"], b["g"]["se"], b["g"]["n_fit"]
    print(f"    {era[0]}-{era[1]}, unweighted 2SLS: beta_0 = {beta:+.4f} ({se:.4f}), "
          f"F = {F:.1f}, N = {n:,}")
    print(f"    step 1 Table V:                     {STEP1_BETA0:+.4f} "
          f"({STEP1_SE0:.4f}), F = {STEP1_F0:.1f}, N = {STEP1_N0:,}")
    check(abs(beta - STEP1_BETA0) < TOL_STEP1 and abs(F - STEP1_F0) < 0.5
          and n == STEP1_N0,
          "9  beta_0 reproduces step 1 section 1.8.6 Table V exactly",
          f"{beta:+.4f} vs {STEP1_BETA0:+.4f}, F {F:.1f} vs {STEP1_F0:.1f}, "
          f"N {n:,} vs {STEP1_N0:,}")
    return beta, se, F


# ================================================================================
# END LEGACY. Everything below is the current exhibit.
# ================================================================================

# --------------------------------------------------------------------------------
# Declared tolerances. Every one is a measured quantity plus headroom, not a guess,
# and none of them is "machine precision" - that is not implementable.
# --------------------------------------------------------------------------------
TOL_RECON = 1e-11       # g = Sum w g_cell + entry, absolute; 36's own bar
TOL_WEIGHT = 1e-12      # Sum w - 1 over the positive-base cells
TOL_ADD_REL = 1e-10     # beta identities, relative to max(|beta_A|, 1e-6)
TOL_NULL = 1e-12        # nocap: max|gD| and |beta_D|
TOL_DEMEAN = 1e-9       # own demeaner against pyfixest, for the bootstrap machinery

# The absorption is PINNED rather than inherited. pyfixest's default is 1e-6, and
# the additivity identities are read at 1e-10 relative, so the projection has to be
# converged well below the tolerance the identities are judged at.
DEMEANER = None         # set in main(), once pyfixest is imported

# The cache mpgr_disagg.py writes. Read as a FILE FORMAT, because importing its
# writer from here would be circular - see the docstring. The three constants are
# restated rather than imported for the same reason, and the manifest's own
# `version` field is asserted against CACHE_VERSION so a format change fails here
# instead of being misread.
CACHE_VERSION = 1
CELL_KEYS = ("cur", "lag", "f_cell", "f_ky", "f_cy")
_SCRATCH = scratch_root()
CACHE_DIR = ((_SCRATCH / "cells_panel") if _SCRATCH is not None
             else dc.CURRENT_PAPER_DATA_ROOT / "temp" / "cells_panel")

# The candidate package's design, restated here so a change to it fails a check
# rather than silently redefining this exhibit. Start years, not outcome years.
PERIODS_ALL = ("pre", "post")
SHARE_DEFAULT = "fixed5"
LEVEL_DEFAULT = 5.0
HORIZONS_ALL = (2, 3, 4)
HEADLINE_H = 4
PLACEMENTS_ALL = ("year", "pooled")

# The nine columns of the fit table. `g`, `gA`, `gB_cells_entry` and `gD` are
# typeset; the rest are estimated and written but not.
ROWS = ("g", "gA", "gB_cells_entry", "gD", "gB_cells", "entry", "gD_noentry")
ROWS_TEX = ("g", "gA", "gB_cells_entry", "gD")


# --------------------------------------------------------------------------------
# Stata's percentile and winsor4. Copied VERBATIM from
# 29_candidate_iv_mpgr_check.py:103-128, which cannot be imported from here
# (29 -> 28 -> 27 -> 22). The candidate cutoffs are Stata's and a numpy quantile is
# a different estimator, so this has to be the same arithmetic and not merely a
# similar one. If 29's copy ever changes, this one has to change with it.
# --------------------------------------------------------------------------------

def stata_pctile(xs, n, p):
    """`summarize, detail`: j = n p / 100; integer j gives the midpoint of x(j) and
    x(j+1), otherwise x(ceil(j)). Integer arithmetic, so the integer case is exact."""
    num = n * p
    if num % 100 == 0:
        j = num // 100
        return 0.5 * (xs[j - 1] + xs[j]) if j < n else xs[n - 1]
    return xs[-(-num // 100) - 1]


def iq_cutoffs(v, level):
    """`winsor4 varname, method(iq) level(#)`: p50 -/+ level * IQR."""
    xs = np.sort(v)
    n = len(xs)
    p25 = stata_pctile(xs, n, 25)
    p50 = stata_pctile(xs, n, 50)
    p75 = stata_pctile(xs, n, 75)
    iqr = p75 - p25
    return p50 - level * iqr, p50 + level * iqr


def cap(v, grp, level, placement, tag, nocap=False):
    """Clip v, and record what the cutoff was and how many observations it bound.

    `placement` is the only thing that varies between the two canonical arms:
    `year` computes one cutoff per start year, `pooled` computes one over the whole
    era. The METHOD is IQR5 in both, which is the pipeline's rule - the same cap
    the candidate first stage puts on the shock and the instrument, and the same
    one mpgr_winsor.ado applies in Stata.

    `bind` means the clip CHANGED the value: x < lo or x > hi, strictly. With the
    exit atom at g = -1 spanning a large share of cells, `x <= lo` and `x < lo`
    differ by an order of magnitude, so the strict form is the only honest one.
    """
    out = v.astype(np.float64).copy()
    rows = []
    if nocap:
        return out, rows
    groups = [None] if placement == "pooled" else list(np.unique(grp))
    for gval in groups:
        sel = np.ones(len(v), bool) if gval is None else (grp == gval)
        x = v[sel]
        if not len(x):
            continue
        lo, hi = iq_cutoffs(x, level)
        out[sel] = np.clip(x, lo, hi)
        rows.append(dict(**tag, year=(-1 if gval is None else int(gval)),
                         cut_lo=float(lo), cut_hi=float(hi), n=int(len(x)),
                         bind_lo=int((x < lo).sum()), bind_hi=int((x > hi).sum())))
    return out, rows


# --------------------------------------------------------------------------------
# The provenance gate. Nothing is rebuilt; the frame and the cells are the bytes
# Tables 1-5 estimated on, and the manifest's source fingerprint is re-validated.
# --------------------------------------------------------------------------------

def cache_dir(period, h, rung, share, level):
    return CACHE_DIR / f"{period}_{rung}_h{h}_{share}_iqr{level:g}"


def _fingerprint(period, h, rung, years):
    """Size and mtime of every file the panel is derived from. mpgr_disagg.py's
    own function, restated - see the docstring for why it is not imported."""
    paths = [CANDIDATE / "data" / "temp" / f"{period}_h{h}.dta",
             dc.EMP_OUT_DIR / "efsy_ccg_year.parquet"]
    paths += [dc.EMP_OUT_DIR / rung / f"y{y}.parquet" for y in years]
    return {str(p): [p.stat().st_size, p.stat().st_mtime_ns] for p in paths}


def read_cache(period, h, rung, share, level):
    """The frame, the cells, and a hard failure if the cache is not current.

    A STALE CACHE IS NEVER REBUILT HERE. mpgr_disagg.py owns the build and runs the
    construction checks that have to re-read the source parquets; rebuilding it from
    this file would put the exhibit on cells no check had passed and would defeat the
    only reason to read the cache at all.
    """
    path = cache_dir(period, h, rung, share, level)
    man_path = path / "manifest.json"
    if not man_path.exists():
        check(False, f"cache present for {path.name}",
              f"{man_path} does not exist. Run 40_build_cells_cache.py "
              f"--period {period} --share {share} --horizon {h} first.")
        raise SystemExit(1)
    man = json.loads(man_path.read_text())

    want = dict(version=CACHE_VERSION, period=period, h=h, rung=rung, share=share,
                level=level)
    bad = {k: (man.get(k), v) for k, v in want.items() if man.get(k) != v}
    if not check(not bad, f"-1a the cache is the one asked for ({path.name})",
                 "matches version/period/h/rung/share/level" if not bad
                 else f"mismatched {bad}"):
        raise SystemExit(1)

    try:
        fresh = _fingerprint(period, h, rung, man["years"])
    except OSError as e:
        check(False, f"-1b the cache's sources are all present ({path.name})", str(e))
        raise SystemExit(1)
    same = fresh == man["fingerprint"]
    if not check(same, f"-1b the cache is CURRENT against its sources ({path.name})",
                 f"{len(fresh)} source files fingerprinted on size and mtime"
                 if same else
                 "the fingerprint has moved. This is expected the first time the "
                 "cache is read on a machine that did not build it, because the "
                 "manifest stores absolute paths. Rerun 40_build_cells_cache.py "
                 f"--period {period} --share {share} --horizon {h}"):
        raise SystemExit(1)

    frame = pd.read_parquet(path / "frame.parquet")
    z = np.load(path / "cells.npz")
    d = {k: z[k] for k in CELL_KEYS}
    d.update(den_c=z["den_c"], lag_c=z["lag_c"], cur_c=z["cur_c"])
    print(f"    {path.name}: {len(frame):,} county-years, {len(d['cur']):,} cells, "
          f"built {time.strftime('%Y-%m-%d %H:%M', time.localtime(man_path.stat().st_mtime))}")
    return frame, d, man


def assert_frame(frame, period, h, tag):
    """The frame IS the candidate panel, forward-dated, on non-overlapping windows."""
    need = {"ccg", "year", "start_year", "shock", "z"}
    check(need <= set(frame.columns), f"-1c the frame carries the candidate columns ({tag})",
          f"missing {sorted(need - set(frame.columns))}" if not need <= set(frame.columns)
          else f"{sorted(need)}")
    y, s = frame["year"].to_numpy(), frame["start_year"].to_numpy()
    check(bool((y == s + h).all()),
          f"-1d the frame is forward-dated, year = start_year + h ({tag})",
          f"h={h}, {int((y != s + h).sum())} rows disagree")
    starts = sorted(pd.unique(s))
    gaps = {starts[i + 1] - starts[i] for i in range(len(starts) - 1)}
    check(all(gp >= h for gp in gaps) if gaps else True,
          f"-1e the start years are non-overlapping at h={h} ({tag})",
          f"start years {starts}, gaps {sorted(gaps)}")
    spans = [(a, a + h) for a in starts if a < NAICS_BREAK < a + h]
    check(not spans, f"-1f no window spans the {NAICS_BREAK} NAICS break ({tag})",
          f"outcome years {int(y.min())}-{int(y.max())}"
          if not spans else f"straddling windows {spans}")
    key = np.ascontiguousarray(frame[["ccg", "year"]].to_numpy(np.int64))
    return hashlib.sha256(key.tobytes()).hexdigest()[:16]


# --------------------------------------------------------------------------------
# The outcomes. Everything is a bincount over the cache's own county-year index.
# --------------------------------------------------------------------------------

def screen(frame, d, tag):
    """A.3.1 DECLARED DEVIATION: county-years with no base employment.

    E_c,t = 0 has no standard growth rate AT EITHER LEVEL, so the county-year cannot
    enter this exhibit. The two sub-cases are counted and named SEPARATELY, because
    only the first is a lost observation with a real outcome:

        E_c,t = 0, E_c,t+h > 0    a county entrant
        E_c,t = 0, E_c,t+h = 0    a county absent in both periods

    E_c,t+h = 0 with E_c,t > 0 is NOT dropped. That is total county exit, g = -1,
    and it belongs in the cutoff distribution.
    """
    lag_c, cur_c = d["lag_c"], d["cur_c"]
    ok = lag_c > 0
    ent = (~ok) & (cur_c > 0)
    absent = (~ok) & (cur_c <= 0)
    print(f"\n  A.3.1 deviation, declared ({tag}): {int((~ok).sum())} of {len(ok):,} "
          f"county-years have E_c,t = 0 and are dropped.")
    print(f"    {int(ent.sum())} are COUNTY ENTRANTS (E_c,t+h > 0) - a real outcome "
          f"this exhibit cannot measure, because no growth rate exists at either level")
    print(f"    {int(absent.sum())} are absent in both periods")
    for nm, msk in (("entrant", ent), ("absent", absent)):
        if not msk.any():
            continue
        sub = frame.loc[msk, ["ccg", "year"]]
        for ccg, grp in sub.groupby("ccg"):
            print(f"      {nm} ccg {int(ccg)}: "
                  + ", ".join(str(int(v)) for v in sorted(grp['year'])))
    return ok, int(ent.sum()), int(absent.sum())


def build_outcomes(frame, d, ok, level, placement, rung, period, h, nocap=False):
    """The seven left-hand sides on the estimable county-years, plus the cutoffs.

    Forward-dated throughout: `lag` is E_m,t and `cur` is E_m,t+h, `lag_c` is E_c,t
    and is BOTH the omega denominator and the county denominator. The frame row is
    keyed on the outcome year t + h.
    """
    cur, lag, cy = d["cur"], d["lag"], d["f_cy"]
    lag_c, cur_c = d["lag_c"], d["cur_c"]
    n = len(frame)
    start = frame["start_year"].to_numpy()

    # The county rate. Entrant cells are already inside cur_c, which is what makes
    # gA carry them without an entry term of its own.
    g_all = np.where(ok, (cur_c - lag_c) / np.where(ok, lag_c, 1.0), np.nan)

    # M^- : cells with a positive base, in a county-year this exhibit can estimate.
    m = (lag > 0) & ok[cy]
    g_cell = (cur[m] - lag[m]) / lag[m]
    w = lag[m] / lag_c[cy[m]]

    # Entrants: no base, positive current. Negative cells were deleted upstream by
    # 11_build_employment_cells.py and pruned by mpgr_disagg.prune, so `lag <= 0`
    # here is `lag == 0` and the old negative-base term has no rows left to carry.
    ent = (lag <= 0) & (cur > 0)
    entry_all = np.bincount(cy[ent], weights=cur[ent], minlength=n) \
        / np.where(ok, lag_c, 1.0)

    tag_c = dict(level="county", rung=rung, period=period, h=h, placement=placement)
    tag_m = dict(level="cell", rung=rung, period=period, h=h, placement=placement)
    gA_ok, cut_c = cap(g_all[ok], start[ok], level, placement, tag_c, nocap)
    g_cell_cap, cut_m = cap(g_cell, start[cy[m]], level, placement, tag_m, nocap)

    gB_cells_all = np.bincount(cy[m], weights=w * g_cell_cap, minlength=n)
    sum_wg_all = np.bincount(cy[m], weights=w * g_cell, minlength=n)
    w_sum_all = np.bincount(cy[m], weights=w, minlength=n)
    n_pos_all = np.bincount(cy[m], minlength=n)

    out = pd.DataFrame({
        "ccg": frame["ccg"].to_numpy()[ok],
        "year": frame["year"].to_numpy()[ok],
        "start_year": start[ok],
        "shock": frame["shock"].to_numpy()[ok],
        "z": frame["z"].to_numpy()[ok],
        "emp_base": lag_c[ok],
        "g": g_all[ok],
        "gA": gA_ok,
        "gB_cells": gB_cells_all[ok],
        "entry": entry_all[ok],
        "sum_wg": sum_wg_all[ok],
        "w_sum": w_sum_all[ok],
        "n_pos": n_pos_all[ok],
    })
    out["gB_cells_entry"] = out["gB_cells"] + out["entry"]
    out["gD"] = out["gA"] - out["gB_cells_entry"]
    out["gD_noentry"] = out["gA"] - out["gB_cells"]

    # Cell-side diagnostics, reported not asserted.
    diag = dict(n_cells_pos=int(m.sum()), n_entry_cells=int(ent.sum()),
                n_exit_cells=int((m & (cur == 0)).sum()))
    return out, cut_c + cut_m, diag


def assert_construction(o, diag, tag):
    """The construction checks. None of them is a decomposition."""
    print(f"\n[1-4] construction, {tag}")
    check(bool((o["n_pos"] > 0).all()),
          "1  every estimable county-year has a positive-base cell",
          f"min {int(o['n_pos'].min())} cells with E_m,t > 0; this is what makes "
          f"Sum w = 1 and the reconstruction exact")
    dev = float(np.abs(o["w_sum"].to_numpy() - 1.0).max())
    check(dev < TOL_WEIGHT, "2  omega sums to one over the positive-base cells",
          f"max |Sum w - 1| {dev:.3e} < {TOL_WEIGHT:.0e}")

    # THE LOAD-BEARING CHECK. eq standard_entry_decomp: it catches essentially any
    # error in the base, the merge or the weights, and it is the reason g~^B carries
    # + entry - without the term this does not hold even with nothing capped.
    recon = (o["sum_wg"] + o["entry"] - o["g"]).to_numpy()
    check(float(np.abs(recon).max()) < TOL_RECON,
          "3  uncapped reconstruction g = Sum w g_cell + entry",
          f"max |dev| {float(np.abs(recon).max()):.3e} < {TOL_RECON:.0e}")

    nan = {k: int(o[k].isna().sum()) for k in ROWS}
    bad = {k: v for k, v in nan.items() if v}
    check(not bad, "4  no NaN in any left-hand side",
          f"{len(o):,} rows, {diag['n_cells_pos']:,} positive-base cells, "
          f"{diag['n_entry_cells']:,} entrant cells, {diag['n_exit_cells']:,} exiting"
          + ("" if not bad else f"; offending {bad}"))


# --------------------------------------------------------------------------------
# Estimation.
# --------------------------------------------------------------------------------

def fit_block(d, lhs, weights, iv_mode, cluster="ccg"):
    """One coefficient per left-hand side, all on the identical frame.

    fixef_rm IS "none", AND THAT IS NOT THE DEFAULT THIS FILE ONCE USED. The
    candidate frame already had reghdfe's singleton removal applied on
    (unit, state_year) inside mpgr_disagg's build, and every fit behind Tables 1-5
    passes singleton=False. Dropping singletons again here would put this exhibit on
    a different sample from the cache it has just asserted it is reading.
    """
    import pyfixest as pf
    kw = dict(vcov={"CRV1": cluster}, fixef_rm="none", demeaner=DEMEANER)
    if weights is not None:
        kw["weights"] = weights
    out = {}
    for k in lhs:
        f = (f"{k} ~ 1 | ccg + year | shock ~ z" if iv_mode
             else f"{k} ~ shock | ccg + year")
        m = pf.feols(f, d, **kw)
        out[k] = dict(beta=float(m.coef()["shock"]), se=float(m.se()["shock"]),
                      n_fit=int(m._N))
    fs = pf.feols("shock ~ z | ccg + year", d, **kw)
    b, se = float(fs.coef()["z"]), float(fs.se()["z"])
    return out, ((b / se) ** 2 if se > 0 else np.nan)


def identities(b, tag):
    """The three that survive. Relative tolerance, because beta can be tiny.

    None of these is a decomposition of the cap by level - that was dropped on
    instruction. The first says the fits ran on byte-identical rows; the other two
    say what excluding entrants would have cost, which is the one question a reader
    of this exhibit asks next.
    """
    scale = max(abs(b["gA"]["beta"]), 1e-6) * TOL_ADD_REL
    r = {
        "resid_triple": b["gA"]["beta"] - b["gB_cells_entry"]["beta"] - b["gD"]["beta"],
        "resid_entry": (b["gD_noentry"]["beta"] - b["gD"]["beta"]
                        - b["entry"]["beta"]),
        "resid_cells": (b["gA"]["beta"] - b["gB_cells"]["beta"]
                        - b["gD_noentry"]["beta"]),
    }
    worst = max(abs(v) for v in r.values())
    check(worst < scale, f"5  the three identities add, {tag}",
          f"max |resid| {worst:.3e} < {scale:.3e}  ("
          + ", ".join(f"{k.split('_')[1]} {v:+.2e}" for k, v in r.items()) + ")")
    ns = {v["n_fit"] for v in b.values()}
    check(len(ns) == 1, f"6  every fit ran on the same n_obs, {tag}", f"n_fit {ns}")
    return r


def validate_demeaner(d, weights, betas_ref):
    """Assert the bootstrap machinery reproduces pyfixest before it is trusted."""
    keys = list(betas_ref)
    wt = d[weights].to_numpy() if weights else np.ones(len(d))
    f1 = pd.factorize(d["ccg"].to_numpy())[0].astype(np.int64)
    f2 = pd.factorize(d["year"].to_numpy())[0].astype(np.int64)
    M = demean(np.column_stack([d["shock"], d["z"]] + [d[k] for k in keys]), f1, f2, wt)
    zx = float(M[:, 1] @ (wt * M[:, 0]))
    worst, where = 0.0, ""
    for j, k in enumerate(keys):
        b = float(M[:, 1] @ (wt * M[:, 2 + j])) / zx
        dev = abs(b - betas_ref[k]["beta"])
        if dev > worst:
            worst, where = dev, k
    check(worst < TOL_DEMEAN, "own demeaner reproduces pyfixest 2SLS",
          f"max dev {worst:.3e} on {where}, over {len(keys)} left-hand sides")


def bootstrap_ratio(d, weights, nrep, seed, betas_ref):
    """Paired cluster bootstrap over CCGs, for g_D / g^A.

    The ratio of two coefficients estimated on one sample has no clustered SE worth
    quoting. Resample whole CCGs, demean inside the replicate because the fixed
    effects move with the resample, and read every beta off the same projection as
    (z'y_k) / (z'x) - which is exactly why the betas add up.
    """
    rng = np.random.default_rng(seed)
    ccgs = d["ccg"].unique()
    by = {c: np.flatnonzero(d["ccg"].to_numpy() == c) for c in ccgs}
    keys = list(betas_ref)
    acc = {k: [] for k in keys}
    acc["ratio_D_A"] = []
    for _ in range(nrep):
        pick = rng.choice(ccgs, size=len(ccgs), replace=True)
        ix = np.concatenate([by[c] for c in pick])
        s = d.iloc[ix]
        # The resampled CCG ids repeat, so the unit effect must be the DRAW and not
        # the CCG, or a county picked twice would be treated as one unit.
        rep = np.concatenate([np.full(len(by[c]), i) for i, c in enumerate(pick)])
        f2 = pd.factorize(s["year"].to_numpy())[0]
        wt = (s[weights].to_numpy() if weights else np.ones(len(s)))
        M = demean(np.column_stack([s["shock"], s["z"]] + [s[k] for k in keys]),
                   rep.astype(np.int64), f2.astype(np.int64), wt)
        zx = float(M[:, 1] @ (wt * M[:, 0]))
        if abs(zx) < 1e-300:
            continue
        b = {k: float(M[:, 1] @ (wt * M[:, 2 + j])) / zx for j, k in enumerate(keys)}
        for k in keys:
            acc[k].append(b[k])
        if abs(b["gA"]) > 1e-12:
            acc["ratio_D_A"].append(b["gD"] / b["gA"])

    def q(v):
        v = np.asarray(v)
        return (float(np.std(v, ddof=1)), float(np.quantile(v, 0.025)),
                float(np.quantile(v, 0.975))) if len(v) > 10 else (np.nan,) * 3
    return {k: q(v) for k, v in acc.items()}


def null_gate(frame, d, ok, rung, period, h, tag):
    """With no cap anywhere the reconstruction IS g, so the wedge is zero.

    g~^B is RECOMPUTED from the reconstruction rather than aliased to g, which is
    the whole point: the test exercises the base, the weights, the entry term and
    the sample in one line. It is also the guard against the two sides treating
    entrants differently - that failure shows up here at the size of the entry
    term, not at machine noise.
    """
    print(f"\n[0] the null, {tag} - no cap anywhere")
    o, _, _ = build_outcomes(frame, d, ok, 0.0, "year", rung, period, h, nocap=True)
    mx = float(np.abs(o["gD"]).max())
    check(mx < TOL_NULL, f"0a max |g_D| is zero with no cap ({tag})",
          f"{mx:.3e} < {TOL_NULL:.0e}; this is the reconstruction residual, "
          f"not an identity failure")
    with np.errstate(divide="ignore", invalid="ignore"):
        b, _ = fit_block(o, ["gD"], None, True)
    check(abs(b["gD"]["beta"]) < TOL_NULL, f"0b beta_D is zero with no cap ({tag})",
          f"{b['gD']['beta']:+.3e} < {TOL_NULL:.0e}")


# --------------------------------------------------------------------------------
# The printed report.
# --------------------------------------------------------------------------------

LABELS = {
    "g": "(0) g       uncapped county",
    "gA": "(1) g^A     county cap",
    "gB_cells_entry": "(2) g~^B    cells capped + entry",
    "gD": "(3) g_D     = (1)-(2)  CAP GAP",
    "gB_cells": "    g^B     cells capped, no entry",
    "entry": "    entry   entrant mass",
    "gD_noentry": "    g_D^ne  = (1)-g^B  entry excluded",
}
ORDER = ["g", "gA", "gB_cells_entry", "gD", "gB_cells", "entry", "gD_noentry"]


def _cell(r):
    return "        --" if r is None or not len(r) else f"{r['beta'].iloc[0]:+8.4f}"


def _se(r):
    if r is None or not len(r):
        return "          "
    x = r.iloc[0]
    s = x["boot_se"] if np.isfinite(x.get("boot_se", np.nan)) else x["se"]
    star = "*" if abs(x["beta"]) > 1.96 * s and s > 0 else " "
    return f"({s:.4f}){star}"


def panel(fit, rung, placement, period, h, blocks, title):
    f = fit[(fit["rung"] == rung) & (fit["placement"] == placement)
            & (fit["period"] == period) & (fit["h"] == h)]
    if not len(f):
        return
    print(f"\n  {title}")
    print("  " + " " * 36 + "".join(f"{w + ' ' + e:>22s}" for w, e in blocks))
    for k in ORDER:
        line = f"  {LABELS[k]:<36s}"
        for w, e in blocks:
            r = f[(f["row"] == k) & (f["weighting"] == w) & (f["estimator"] == e)]
            line += f"{_cell(r):>12s}{_se(r):>10s}"
        print(line)
        if k == "gD":
            print("  " + "-" * 34)
    line = f"  {'g_D / (1)':<36s}"
    for w, e in blocks:
        r = f[(f["row"] == "gD") & (f["weighting"] == w) & (f["estimator"] == e)]
        v = r["ratio_D_A"].iloc[0] * 100 if len(r) else np.nan
        lo = r["ratio_D_A_lo"].iloc[0] if len(r) else np.nan
        hi = r["ratio_D_A_hi"].iloc[0] if len(r) else np.nan
        ci = (f"[{lo * 100:+.0f},{hi * 100:+.0f}]" if np.isfinite(lo) else "")
        line += f"{v:>11.1f}%{ci:>10s}"
    print(line)
    for nm, fn in (("first-stage F", lambda r: f"{r['fs_F'].iloc[0]:.1f}"),
                   ("N", lambda r: f"{r['n_fit'].iloc[0]:,}"),
                   ("max |identity resid|", lambda r: "%.1e" % max(
                       abs(r[c].iloc[0]) for c in ("resid_triple", "resid_entry",
                                                   "resid_cells")))):
        line = f"  {nm:<36s}"
        for w, e in blocks:
            r = f[(f["row"] == "gD") & (f["weighting"] == w) & (f["estimator"] == e)]
            line += f"{(fn(r) if len(r) else '--'):>22s}"
        print(line)


def report(fit, cut, frames, args, screens):
    blocks4 = [("unweighted", "2SLS"), ("unweighted", "OLS"),
               ("employment", "2SLS"), ("employment", "OLS")]
    for period in args.periods:
        print("\n" + "=" * 84)
        if period == "pre":
            print(f"  HEADLINE - PRE-{NAICS_BREAK}, h = {HEADLINE_H}, "
                  f"z_income_{args.share}, IQR{args.level:g}")
        else:
            print(f"  SIDE TABLE - POST-{NAICS_BREAK}, h = {HEADLINE_H}. Reported "
                  f"separately, never stacked with the pre block: a four-digit cell")
            print("  is a different object either side of the NAICS break, so pooling")
            print("  the eras or differencing their coefficients would be comparing")
            print("  two constructions rather than two samples.")
        print("=" * 84)
        for placement in args.placement:
            rule = ("cutoffs within start year" if placement == "year"
                    else "ONE cutoff pooled over the era")
            print(f"\n  --- cutoff placement {placement}: {rule}, at BOTH levels ---")
            for i, rung in enumerate(args.rung):
                panel(fit, rung, placement, period, HEADLINE_H, blocks4,
                      f"Panel {chr(65 + i)}. {rung}, {period}, h={HEADLINE_H}, "
                      f"z_income_{args.share}, IQR{args.level:g} {placement}")

    print("\n" + "=" * 84)
    print("  THE HORIZON SWEEP. h = 4 is the headline and the only one typeset;")
    print("  h = 2 and h = 3 are estimated and reported here and in the CSV.")
    print("=" * 84)
    s = fit[(fit["row"] == "gD") & (fit["estimator"] == "2SLS")
            & (fit["weighting"] == "unweighted")]
    print(f"\n  {'period':7s}{'rung':14s}{'place':8s}" +
          "".join(f"{'h=' + str(h):>22s}" for h in args.horizon))
    for period in args.periods:
        for rung in args.rung:
            for placement in args.placement:
                line = f"  {period:7s}{rung:14s}{placement:8s}"
                for h in args.horizon:
                    r = s[(s["period"] == period) & (s["rung"] == rung)
                          & (s["placement"] == placement) & (s["h"] == h)]
                    line += (f"{r['beta'].iloc[0]:>13.4f}"
                             f"{'(%.4f)' % r['se'].iloc[0]:>9s}") if len(r) else \
                            f"{'--':>22s}"
                print(line)

    print("\n" + "=" * 84)
    print("  REPORTED, NOT ASSERTED")
    print("=" * 84)

    print("\n  Cutoffs by start year, at the headline horizon. The lower CELL cutoff")
    print("  sits near -1 because the exit atom at g = -1 spans a large share of")
    print("  cells, NOT because -1 is a floor.")
    c = cut[(cut["placement"] == "year") & (cut["h"] == HEADLINE_H)]
    for period in args.periods:
        for rung in args.rung:
            for lev in ("county", "cell"):
                sub = c[(c["period"] == period) & (c["rung"] == rung)
                        & (c["level"] == lev)].sort_values("year")
                if not len(sub):
                    continue
                print(f"\n  {period} / {rung} / {lev}")
                print("    start   cut_lo    cut_hi         n  bind_lo  bind_hi")
                for _, r in sub.iterrows():
                    print(f"    {int(r['year'])}  {r['cut_lo']:+8.4f}  "
                          f"{r['cut_hi']:+8.2f}  {int(r['n']):9,d}  "
                          f"{int(r['bind_lo']):7,d}  {int(r['bind_hi']):7,d}")

    print("\n  THE ENTRY BLOCK. Counties are lumpy and the entrant mass is not a")
    print("  rounding term, so it is measured rather than asserted away. g_D cancels")
    print("  it by construction - the null gate is what proves that - and these")
    print("  numbers say how much was cancelled.")
    for (period, rung, placement, h), o in sorted(frames.items()):
        if placement != "year" or h != HEADLINE_H:
            continue
        g, e = o["g"].to_numpy(), o["entry"].to_numpy()
        gD = o["gD"].to_numpy()
        b = fit[(fit["period"] == period) & (fit["rung"] == rung)
                & (fit["placement"] == placement) & (fit["h"] == h)
                & (fit["row"] == "entry") & (fit["estimator"] == "2SLS")
                & (fit["weighting"] == "unweighted")]
        print(f"\n  {period} / {rung}: n = {len(o):,}")
        print(f"    entry    mean {e.mean():+.5f}  median {np.median(e):+.5f}  "
              f"median(entry/|g|) {np.median(e / np.maximum(np.abs(g), 1e-12)):.3f}")
        print(f"             county-years with an entrant cell "
              f"{100 * (e != 0).mean():.1f}%  "
              f"beta_entry {b['beta'].iloc[0]:+.4f} ({b['se'].iloc[0]:.4f})"
              if len(b) else "")
        print(f"    g_D      mean {gD.mean():+.5f}  sd {gD.std(ddof=1):.5f}  "
              f"sd(g_D)/sd(g) {gD.std(ddof=1) / g.std(ddof=1):.4f}  "
              f"exactly zero {100 * (gD == 0).mean():.1f}%")
        n_ent, n_abs = screens[(period, rung, h)]
        print(f"    dropped  {n_ent} county entrants, {n_abs} county-years absent "
              f"in both periods (A.3.1, named above)")

    print("\n  CAVEATS THE WRITE-UP MUST CARRY.")
    print("  - The cutoffs are estimated from the same sample and then treated as")
    print("    data, so every standard error here understates by an amount this")
    print("    design does not correct.")
    print("  - Shift-share inference: Adao, Kolesar and Morrison (2019) applies to")
    print("    this instrument and clustering on the CCG does not address it.")
    print("  - Protocol A.1.8: the paper is written with a binary treatment and this")
    print("    regressor is continuous. To be confirmed with Chenzi before any draft.")
    print("  - No fiscal-multiplier paper has yet been named that winsorizes county")
    print("    and cell employment growth. IQR5 is the candidate package's own rule,")
    print("    carried across so the exhibit's cap matches the shock's.")


# --------------------------------------------------------------------------------
# LaTeX. Emitted by the run so the tables can never drift from the CSV.
# --------------------------------------------------------------------------------

TEX_LABEL = {
    "g": r"Uncapped county growth $g$",
    "gA": r"County cap $g^{A}$",
    "gB_cells_entry": r"Cells capped on their own distribution $\tilde{g}^{B}$",
    "gD": r"\textbf{Construction error} $g_{\Delta}=g^{A}-\tilde{g}^{B}$",
}
PRETTY = {"naics4_alloc": r"County $\times$ NAICS4", "naics2": r"County $\times$ NAICS2"}

# The notes every coefficient table carries. Written once, because the four
# regression tables differ only in era, weighting and cutoff placement, and a note
# that drifted between them would describe one table while sitting under another.
# Kept SHORT on instruction: what the four rows are, what is and is not capped, and
# how to read the standard errors. Everything else lives in step_2/README.md §2.
NOTE_COMMON = (
    r"Standard growth of county employment over $t\rightarrow t+h$, $h=%(h)d$. $g$ "
    r"is the county's own rate; $g^{A}$ caps it on the distribution of county "
    r"growth; $\tilde{g}^{B}=\sum_{m}\omega_{m}g^{B}_{m}+\mathrm{entry}$ rebuilds "
    r"the county from cells capped on the distribution of \emph{cell} growth, with "
    r"$\omega_{m}=E_{m,t}/E_{c,t}$ and $\mathrm{entry}=\sum_{m:E_{m,t}=0}"
    r"E_{m,t+h}/E_{c,t}$ restoring the cells that have no base to cap. "
    r"$g_{\Delta}$ is protocol A.3.3's construction error: what a paper reports at "
    r"the county level minus what it would report having capped its cells "
    r"separately. With no cap anywhere $\tilde{g}^{B}=g$, so $g_{\Delta}$ is the cap "
    r"alone. $g$ and $g^{A}$ do not depend on the split and repeat across columns. "
    r"\textbf{No growth rate of ours is winsorized} (A.3.2) and nothing is dropped; "
    r"the caps are \texttt{winsor4 \dots method(iq) level(%(lev)g)}, %(rule)s, at "
    r"both levels. %(sample)s 2SLS on the income-based national instrument, county "
    r"and year fixed effects, standard errors clustered on the county in "
    r"parentheses; the interval is a 200-replicate paired cluster bootstrap over "
    r"counties. $^{*}$, $^{**}$, $^{***}$: 10, 5, 1 percent.")


def tex_table(fit, period, weighting, placement, h, cols, title, label, extra, path):
    """One column per (rung, estimator); rows are the four typeset left-hand sides.

    House style, matching 43_tables_tex.py: a `table` float with a caption, a label
    the other tables cross-reference, and a Notes paragraph defining every row. The
    fragments used to carry their caption as a LaTeX COMMENT, which meant a reader of
    the typeset output was handed four Greek letters and no way to tell what they
    were.

    ONLY 2SLS IS TYPESET, on instruction. The OLS block is still estimated, still
    passes the same identities, and is still in winsor_construction_error.csv, but it
    is not a column here: the ratio, the bootstrap interval and the first-stage F
    have no OLS counterpart, so half of every OLS column was blank. What the two
    projections say about each other belongs in prose - see step_2/README.md §2.3.
    """
    f = fit[(fit["period"] == period) & (fit["weighting"] == weighting)
            & (fit["placement"] == placement) & (fit["h"] == h)]
    rule = ("computed within each start year" if placement == "year"
            else "one cutoff pooled over the era")
    era = ("County-years before 1998" if period == "pre"
           else "County-years after 1998")
    wt = ("" if weighting == "unweighted"
          else " Weighted by base county employment $E_{c,t}$.")
    sample = (f"{era}, non-overlapping windows, positive base employment.{wt}")
    notes = NOTE_COMMON % dict(h=h, lev=5.0, rule=rule, sample=sample)
    if extra:
        notes = extra + " " + notes

    L = [f"% {title}",
         "% Rendered by 22_winsor_construction_error.py from "
         "winsor_construction_error.csv. Do not edit.",
         r"\begin{table}[htbp]",
         r"\def\sym#1{\ifmmode^{#1}\else\(^{#1}\)\fi}",
         r"\caption{%s}" % title,
         r"\label{%s}" % label,
         r"{\centering",
         r"\adjustbox{max width=\linewidth}{",
         r"\begin{tabular}{l*{%d}{c}}" % len(cols),
         r"\toprule",
         r"\textit{Dependent variable} & \multicolumn{%d}{c}{Standard growth of "
         r"county employment, $h=%d$} \\" % (len(cols), h),
         r"\cmidrule(r){2-%d}" % (1 + len(cols))]
    head, mid, i = r"\textit{Cells the county is split into}", "", 2
    for rung in dict.fromkeys(r for r, _ in cols):
        n = sum(1 for r, _ in cols if r == rung)
        head += r" & \multicolumn{%d}{c}{%s}" % (n, PRETTY.get(rung, rung))
        mid += r"\cmidrule(lr){%d-%d}" % (i, i + n - 1)
        i += n
    L += [head + r" \\"]
    # The estimator row is only printed when the columns actually differ in it.
    # With 2SLS alone it would be a row of one repeated word.
    if len({e for _, e in cols}) > 1:
        L += [mid, r"\textit{Estimator}"
              + "".join(" & %s" % e for _, e in cols) + r" \\"]
    L += ["".join(" & (%d)" % (j + 1) for j in range(len(cols))) + r" \\",
          r"\midrule"]
    for k in ROWS_TEX:
        r1, r2 = TEX_LABEL[k], ""
        for rung, est in cols:
            m = f[(f["row"] == k) & (f["rung"] == rung) & (f["estimator"] == est)]
            if len(m):
                b, se = float(m["beta"].iloc[0]), float(m["se"].iloc[0])
                r1 += " & %.4f%s" % (b, _stars(b, se))
                r2 += " & (%.4f)" % se
            else:
                r1 += " & "
                r2 += " & "
        L += [r1 + r" \\", r2 + r" \\"]
        if k == "gB_cells_entry":
            L.append(r"\addlinespace")
    L.append(r"\\[-1.8ex]\hline\addlinespace")
    row = r"$g_{\Delta}$ \textit{as a share of} $g^{A}$"
    row2 = r"\hspace{1.5em}\textit{bootstrap 95\% interval}"
    for rung, est in cols:
        m = f[(f["row"] == "gD") & (f["rung"] == rung) & (f["estimator"] == est)]
        if not len(m) or est != "2SLS":
            row += " & "
            row2 += " & "
            continue
        row += r" & %.1f\%%" % (100 * float(m["ratio_D_A"].iloc[0]))
        lo, hi = float(m["ratio_D_A_lo"].iloc[0]), float(m["ratio_D_A_hi"].iloc[0])
        row2 += ((" & [%.0f, %.0f]" % (100 * lo, 100 * hi))
                 if np.isfinite(lo) else " & ")
    L += [row + r" \\", row2 + r" \\", r"\addlinespace"]
    row = r"First-stage $F$"
    for rung, est in cols:
        m = f[(f["row"] == "gD") & (f["rung"] == rung) & (f["estimator"] == est)]
        row += (" & %.1f" % float(m["fs_F"].iloc[0])) if len(m) and est == "2SLS" \
            else " & "
    L.append(row + r" \\")
    row = "Observations"
    for rung, est in cols:
        m = f[(f["row"] == "gD") & (f["rung"] == rung) & (f["estimator"] == est)]
        row += " & %s" % (f"{int(m['n_fit'].iloc[0]):,}" if len(m) else "")
    L += [row + r" \\", r"\bottomrule", r"\end{tabular}", "}", r"\par}",
          r"\medskip",
          r"{\scriptsize \textit{Notes:} " + notes + r" \par}",
          r"\end{table}"]
    path.write_text("\n".join(L) + "\n")
    return path


def tex_diagnostics(fit, cut, frames, screens, args, path):
    """The construction side: cutoffs, what binds, the wedge, and the entry block.

    One panel per era, columns rung x placement, at the headline horizon. Nothing
    here is hardcoded: the era and the horizon come from the arguments.
    """
    title = "Construction diagnostics: the two cutoffs, what they bind, and entry"
    keys = [(r, p) for r in args.rung for p in args.placement]
    vlab = {"year": "by start year", "pooled": "pooled"}
    L = [f"% {title}",
         "% Rendered by 22_winsor_construction_error.py from winsor_cutoffs.csv "
         "and the county-year frame. Do not edit.",
         r"\begin{table}[htbp]",
         r"\caption{%s}" % title,
         r"\label{tab:winsor_construction}",
         r"{\centering",
         r"\adjustbox{max width=\linewidth}{",
         r"\begin{tabular}{l%s}" % ("c" * len(keys)),
         r"\toprule",
         r"\textit{Cells the county is split into}" + "".join(
             r" & \multicolumn{%d}{c}{%s}" % (len(args.placement), PRETTY[r])
             for r in args.rung) + r" \\"
         + " ".join(r"\cmidrule(lr){%d-%d}" % (2 + i * len(args.placement),
                                               1 + (i + 1) * len(args.placement))
                    for i in range(len(args.rung))),
         r"\textit{Where the cutoff is computed}"
         + "".join(" & %s" % vlab[p] for _, p in keys) + r" \\"]

    def row(label, fn):
        return label + "".join(" & %s" % fn(r, p) for r, p in keys) + r" \\"

    def sub(period, rung, level, placement):
        return cut[(cut["period"] == period) & (cut["rung"] == rung)
                   & (cut["level"] == level) & (cut["placement"] == placement)
                   & (cut["h"] == HEADLINE_H)]

    for period in args.periods:
        L += [r"\midrule",
              r"\multicolumn{%d}{l}{\textit{Panel %s. %s-%d, $h=%d$}} \\" % (
                  1 + len(keys), "AB"[args.periods.index(period)],
                  period.capitalize(), NAICS_BREAK, HEADLINE_H),
              r"\addlinespace"]

        def cs(level, col, agg, fmt, _p=period):
            def f(rung, placement):
                s = sub(_p, rung, level, placement)
                return fmt % getattr(s[col], agg)() if len(s) else ""
            return f

        L.append(row(r"Cell cutoff, upper: max", cs("cell", "cut_hi", "max", "%.1f")))
        L.append(row(r"Cell cutoff, upper: median",
                     cs("cell", "cut_hi", "median", "%.2f")))
        L.append(row(r"Cell cutoff, lower: median",
                     cs("cell", "cut_lo", "median", "%.3f")))
        L.append(row(r"County cutoff, upper: median",
                     cs("county", "cut_hi", "median", "%.3f")))
        L.append(row(r"County cutoff, lower: median",
                     cs("county", "cut_lo", "median", "%.3f")))
        L.append(r"\addlinespace")

        def bind(level, _p=period):
            def f(rung, placement):
                s = sub(_p, rung, level, placement)
                return ("%.2f" % (100 * (s["bind_lo"] + s["bind_hi"]).sum()
                                  / max(s["n"].sum(), 1))) if len(s) else ""
            return f

        L.append(row(r"Cells bound (\%)", bind("cell")))
        L.append(row(r"County-years bound (\%)", bind("county")))
        L.append(r"\addlinespace")

        def fr(fn, _p=period):
            def f(rung, placement):
                o = frames.get((_p, rung, placement, HEADLINE_H))
                return fn(o) if o is not None else ""
            return f

        L.append(row(r"$\sigma(g_{\Delta})/\sigma(g)$", fr(
            lambda o: "%.3f" % (o["gD"].std(ddof=1) / o["g"].std(ddof=1)))))
        L.append(row(r"$g_{\Delta}$ exactly zero (\%)", fr(
            lambda o: "%.1f" % (100 * (o["gD"] == 0).mean()))))
        L.append(r"\addlinespace")
        L.append(row(r"Mean entry term", fr(lambda o: "%.4f" % o["entry"].mean())))
        L.append(row(r"Median entry term", fr(
            lambda o: "%.4f" % o["entry"].median())))
        L.append(row(r"Median entry\,/\,$|g|$", fr(lambda o: "%.3f" % np.median(
            o["entry"].to_numpy() / np.maximum(np.abs(o["g"].to_numpy()), 1e-12)))))
        L.append(row(r"County-years with an entrant (\%)", fr(
            lambda o: "%.1f" % (100 * (o["entry"] != 0).mean()))))
        L.append(r"\addlinespace")
        L.append(row(r"Dropped: county entrants (A.3.1)", lambda r, p, _p=period:
                     "%d" % screens[(_p, r, HEADLINE_H)][0]))
        L.append(row(r"Dropped: absent both periods", lambda r, p, _p=period:
                     "%d" % screens[(_p, r, HEADLINE_H)][1]))
        L.append(row(r"Observations", fr(lambda o: "%s" % f"{len(o):,}")))

    L += [r"\bottomrule", r"\end{tabular}", "}", r"\par}", r"\medskip",
          r"{\scriptsize \textit{Notes:} "
          + (r"The construction side of Tables~\ref{tab:winsor_headline} to "
             r"\ref{tab:winsor_pooled_weighted}. Both caps are \texttt{winsor4 \dots "
             r"method(iq) level(%(lev)g)}: the \emph{county} cutoff over county "
             r"standard growth in the estimation frame, the \emph{cell} cutoff over "
             r"cell standard growth on all cells with positive base, exiting cells at "
             r"$-1$ included. Bound means the clip strictly changed the value; the "
             r"lower cell cap binds on nothing because cell growth cannot fall below "
             r"$-1$ while the cutoff sits near $-5$. The entry block measures what "
             r"$g_{\Delta}$ cancels by construction rather than assuming it is small. "
             r"The last two rows are the declared A.3.1 deviation, county-years with "
             r"$E_{c,t}=0$, which have no growth rate at either level; total county "
             r"exit ($E_{c,t+h}=0$, $E_{c,t}>0$) is \emph{not} removed."
             % dict(lev=args.level))
          + r" \par}",
          r"\end{table}"]
    path.write_text("\n".join(L) + "\n")
    return path


def tex_coeff(fit, path, args):
    r"""\newcommand macros, so prose in a draft cites the run and not a typist."""
    def g(rung, period, row, placement="year", est="2SLS", w="unweighted",
          col="beta", h=HEADLINE_H):
        m = fit[(fit["placement"] == placement) & (fit["rung"] == rung)
                & (fit["period"] == period) & (fit["row"] == row)
                & (fit["h"] == h) & (fit["estimator"] == est)
                & (fit["weighting"] == w)]
        return float(m[col].iloc[0]) if len(m) else np.nan

    out = [r"% Auto-generated by 22_winsor_construction_error.py. Do not edit.",
           r"% Every number a draft quotes from plan section 2.2 comes from here.",
           r"% Suffixes: Pooled = pooled cutoff, Post = post-1998.",
           r"%% h = %d, z_income_%s, winsor4 iq level(%g) at BOTH levels."
           % (HEADLINE_H, args.share, args.level)]
    for period, psfx in (("pre", ""), ("post", "Post")):
        if period not in args.periods:
            continue
        for placement, vsfx in (("year", ""), ("pooled", "Pooled")):
            if placement not in args.placement:
                continue
            for tag, rung in (("Four", "naics4_alloc"), ("Two", "naics2")):
                if rung not in args.rung:
                    continue
                for row, nm in (("g", "Raw"), ("gA", "CapA"),
                                ("gB_cells_entry", "CapB"), ("gD", "Wedge"),
                                ("entry", "Entry")):
                    out.append(r"\newcommand{\Winsor%s%s%s%s}{%.4f}"
                               % (tag, nm, vsfx, psfx,
                                  g(rung, period, row, placement)))
                    out.append(r"\newcommand{\Winsor%s%s%s%sSE}{%.4f}"
                               % (tag, nm, vsfx, psfx,
                                  g(rung, period, row, placement, col="se")))
                out.append(r"\newcommand{\Winsor%sWedgeShare%s%s}{%.1f}"
                           % (tag, vsfx, psfx,
                              100 * g(rung, period, "gD", placement,
                                      col="ratio_D_A")))
                for nm, col in (("WedgeLo", "ratio_D_A_lo"),
                                ("WedgeHi", "ratio_D_A_hi")):
                    out.append(r"\newcommand{\Winsor%s%s%s%s}{%.1f}"
                               % (tag, nm, vsfx, psfx,
                                  100 * g(rung, period, "gD", placement, col=col)))
            out.append(r"\newcommand{\WinsorF%s%s}{%.1f}"
                       % (vsfx, psfx, g("naics4_alloc", period, "gD", placement,
                                        col="fs_F")))
            out.append(r"\newcommand{\WinsorN%s%s}{%s}"
                       % (vsfx, psfx,
                          f"{int(g('naics4_alloc', period, 'gD', placement, col='n_fit')):,}"))
    path.write_text("\n".join(out) + "\n")
    return path


PREVIEW_HEAD = r"""%% Preview wrapper for the winsorization exhibit's fragments.
%% WRITTEN BY 22_winsor_construction_error.py FROM THE LIST OF TABLES IT JUST
%% WROTE, so it cannot name a fragment the run did not produce. Do not edit.
%% Not part of the paper build: the TableW*.tex are `table` floats meant to be
%% \input into a draft and have no preamble of their own.
%%
%% BUILD IT TWICE. The tables cross-reference each other by \label, so a single
%% xelatex pass leaves "??" where the references should be.
%%     xelatex _preview_winsor.tex && xelatex _preview_winsor.tex
\documentclass[11pt]{article}

\usepackage[margin=1in]{geometry}
\usepackage{fontspec}
\setmainfont{Palatino}
\usepackage{amsmath,amssymb}
\usepackage{booktabs}
\usepackage{adjustbox}

%% caption/microtype are deliberately not loaded: this TinyTeX install does not
%% carry them, and nothing in the fragments needs them.
\setlength{\parindent}{0pt}
\pagestyle{empty}

%% The macro file, inputted here so the build fails if it does not compile.
\input{coeff_winsor}

\begin{document}

\begin{center}
{\large\bfseries The winsorization construction error}\\[0.3em]
{\normalsize Plan \S2.2, protocol A.3.3, going down}\\[0.6em]
{\small %(subtitle)s}
\end{center}
\vspace{1.2em}

{\small
\textbf{What this exhibit measures.} An applied paper that reports a county result
and a county-by-industry result, and winsorizes both, computes its two cutoffs on
two different distributions. Different observations bind, and by different
amounts, so the cells stop summing to the county and the two coefficients are no
longer estimated on the same variable. Every table below puts the difference
between the two constructions on the left-hand side and regresses it on the shock.
That coefficient, $g_{\Delta}$, is the construction error. It is the mirror going
down of the trap that cost 59 percent of the state coefficient in Burchardi,
Chaney, Hassan, Tarquinio and Terry (2026).

\medskip
\textbf{We winsorize nothing of our own.} Protocol A.3.2 forbids capping a growth
rate, and none of ours is capped. What is built is the pair of outcomes a
\emph{replicated} design would produce, and the wedge between them.

\medskip
\textbf{The tables.}
\begin{itemize}\setlength{\itemsep}{0.35em}
%(contents)s
\end{itemize}
\par}

"""


def tex_preview(entries, path, subtitle):
    """The wrapper, with a contents page that says what each table is for.

    `entries` is (filename, short title, one-sentence purpose), built from the same
    list the tables were written from, so the preview cannot describe a fragment
    that does not exist or omit one that does.
    """
    contents = "\n".join(
        r"  \item \textbf{%s.} %s" % (nm.split("_")[0].replace("TableW", "Table W"),
                                      blurb)
        for nm, _, blurb in entries)
    body = "\n\\clearpage\n".join(r"\input{%s}" % nm for nm, _, _ in entries)
    path.write_text((PREVIEW_HEAD % dict(subtitle=subtitle, contents=contents))
                    + "\\clearpage\n" + body + "\n\n\\end{document}\n")
    return path


# --------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rung", nargs="+", default=list(RUNGS_ALL), choices=list(RUNGS_ALL))
    ap.add_argument("--periods", nargs="+", default=list(PERIODS_ALL),
                    choices=list(PERIODS_ALL))
    ap.add_argument("--placement", nargs="+", default=list(PLACEMENTS_ALL),
                    choices=list(PLACEMENTS_ALL))
    ap.add_argument("--horizon", type=int, nargs="+", default=list(HORIZONS_ALL))
    ap.add_argument("--share", default=SHARE_DEFAULT)
    ap.add_argument("--level", type=float, default=LEVEL_DEFAULT)
    ap.add_argument("--boot", type=int, default=200,
                    help="cluster bootstrap replicates, headline horizon only")
    ap.add_argument("--seed", type=int, default=20260818)
    args = ap.parse_args()

    dc.LOG_DIR.mkdir(parents=True, exist_ok=True)
    STEP2_OUT.mkdir(parents=True, exist_ok=True)

    # A PARTIAL RUN NEVER OVERWRITES THE CANONICAL OUTPUT, and that includes the
    # log. A debugging run on one rung would otherwise silently replace the headline
    # numbers with a subset of themselves, and the file on disk would look complete.
    canonical = (set(args.rung) == set(RUNGS_ALL)
                 and set(args.periods) == set(PERIODS_ALL)
                 and set(args.placement) == set(PLACEMENTS_ALL)
                 and set(args.horizon) == set(HORIZONS_ALL)
                 and args.share == SHARE_DEFAULT and args.level == LEVEL_DEFAULT)
    sfx = "" if canonical else (
        "__" + "-".join(sorted(args.rung)) + "_" + "-".join(sorted(args.periods))
        + "_" + "-".join(sorted(args.placement))
        + "_h" + "-".join(str(h) for h in sorted(args.horizon))
        + f"_{args.share}_iqr{args.level:g}")

    tee = Tee(dc.LOG_DIR / f"22_winsor_construction_error{sfx}.log")
    global DEMEANER
    import pyfixest as pf
    DEMEANER = pf.MapDemeaner(fixef_tol=1e-12, fixef_maxiter=100_000)
    t0 = time.time()

    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             cwd=dc.PROJECT_ROOT, capture_output=True, text=True,
                             timeout=10).stdout.strip()
    except Exception:
        sha = "unknown"

    print("=" * 84)
    print("Plan section 2.2 - the winsorization construction error, going DOWN")
    print("=" * 84)
    print(f"  regressor   y_income, cumulative contracts over [t+1, t+h] / county income")
    print(f"  instrument  z_income_{args.share}, national numerator and denominator")
    print(f"  horizons    {args.horizon}   (h = {HEADLINE_H} is the headline and the "
          f"only one typeset)")
    print(f"  rungs       {args.rung}")
    print(f"  periods     {args.periods}   pre is the headline, post is a separate "
          f"side table")
    print(f"  placement   {args.placement}   IQR{args.level:g} in both; only WHERE the "
          f"cutoff is computed changes")
    print(f"  git {sha}, pyfixest {pf.__version__}, pandas {pd.__version__}, "
          f"numpy {np.__version__}")
    if _SCRATCH is not None:
        print(f"  {SCRATCH_ENV} is set: the cache is read from {CACHE_DIR}")
    print("\n  DECLARED DECISIONS, recorded before any coefficient is read.")
    print("  - standard growth, per A.3.2; nothing of OURS is winsorized. The caps")
    print("    are the ones a replicating paper would apply, at both levels.")
    print("  - the cap is winsor4 method(iq) level(%g), Stata's own percentile rule,"
          % args.level)
    print("    which is what the candidate first stage puts on the shock and the")
    print("    instrument and what mpgr_winsor.ado applies in Stata. p1/p99 is retired.")
    print("  - BOTH placements are canonical: `year` computes the cutoff within start")
    print("    year, `pooled` computes one over the era, and each moves BOTH caps.")
    print("  - bind means the clip changed the value: x < lo or x > hi, strictly")
    print("  - exiting cells (base>0, E_t+h=0, g=-1) ARE in the cutoff distribution")
    print("  - entrants are in NEITHER cutoff distribution and are never capped")
    print("  - the cap is NOT split by level: cap_cty and cap_cell are not built")
    print("  - counties with E_c,t = 0 drop from everything: an A.3.1 deviation,")
    print("    declared and named below, because standard growth is undefined there")

    fits, cuts, frames, screens = [], [], {}, {}
    for period in args.periods:
        for h in args.horizon:
            hashes = {}
            for rung in args.rung:
                print(f"\n{'=' * 84}\n  {period}, h = {h}, rung {rung}\n{'=' * 84}")
                print("\n[-1] provenance: the cache Tables 1-5 estimate on")
                frame, d, man = read_cache(period, h, rung, args.share, args.level)
                hashes[rung] = assert_frame(frame, period, h, f"{period}/h{h}/{rung}")
                ok, n_ent, n_abs = screen(frame, d, f"{period}/h{h}/{rung}")
                screens[(period, rung, h)] = (n_ent, n_abs)

                null_gate(frame, d, ok, rung, period, h, f"{period}/h{h}/{rung}")

                for placement in args.placement:
                    o, cut_rows, diag = build_outcomes(
                        frame, d, ok, args.level, placement, rung, period, h)
                    assert_construction(o, diag, f"{period}/h{h}/{rung}/{placement}")
                    cuts.extend(cut_rows)
                    frames[(period, rung, placement, h)] = o

                    for wname, wcol in (("unweighted", None),
                                        ("employment", "emp_base")):
                        for est, ivm in (("2SLS", True), ("OLS", False)):
                            b, F = fit_block(o, ROWS, wcol, ivm)
                            r = identities(
                                b, f"{period}/h{h}/{rung}/{placement}/{wname}/{est}")
                            boot = {}
                            # The bootstrap runs at the HEADLINE horizon only. h=2 and
                            # h=3 are reported from the clustered SE; nothing typeset
                            # reads their interval.
                            if est == "2SLS" and args.boot and h == HEADLINE_H:
                                validate_demeaner(o, wcol, b)
                                boot = bootstrap_ratio(o, wcol, args.boot,
                                                       args.seed, b)
                            rat = b["gD"]["beta"] / b["gA"]["beta"] \
                                if b["gA"]["beta"] else np.nan
                            bs, lo, hi = boot.get("ratio_D_A", (np.nan,) * 3)
                            for k in ROWS:
                                kbs, klo, khi = boot.get(k, (np.nan,) * 3)
                                fits.append(dict(
                                    period=period, h=h, rung=rung,
                                    placement=placement, estimator=est,
                                    weighting=wname, row=k, beta=b[k]["beta"],
                                    se=b[k]["se"], boot_se=kbs, boot_lo=klo,
                                    boot_hi=khi, fs_F=F, n_frame=len(o),
                                    n_fit=b[k]["n_fit"], ratio_D_A=rat,
                                    ratio_D_A_se=bs, ratio_D_A_lo=lo,
                                    ratio_D_A_hi=hi, index_hash=hashes[rung],
                                    share=args.share, level=args.level, **r))
                del d
            # One frame per (period, h): the rungs differ in how the outcome is BUILT,
            # never in which county-years it is built for. If these ever diverged, a
            # cross-rung comparison would be comparing samples.
            check(len(set(hashes.values())) == 1,
                  f"7  one estimation frame across every rung, {period}/h{h}",
                  f"hash {set(hashes.values())}")

    fit = pd.DataFrame(fits)
    cut = pd.DataFrame(cuts)
    n_blocks = len(fit) // len(ROWS)
    check(len(fit) == n_blocks * len(ROWS),
          "8  every block wrote all seven rows",
          f"{len(fit):,} rows = {n_blocks} blocks x {len(ROWS)}")

    if not canonical:
        print(f"\n  PARTIAL RUN. The canonical outputs are left untouched; this run "
              f"writes to *{sfx}.* instead.")
        print("  Rerun with no arguments to rebuild the canonical set.")

    fit.to_csv(dc.LOG_DIR / f"winsor_construction_error{sfx}.csv", index=False)
    cut.to_csv(dc.LOG_DIR / f"winsor_cutoffs{sfx}.csv", index=False)
    keep = ["ccg", "year", "start_year", "shock", "z", "emp_base", "g", "gA",
            "gB_cells", "gB_cells_entry", "gD", "gD_noentry", "entry", "sum_wg"]
    pq = pd.concat([v.assign(period=k[0], rung=k[1], placement=k[2], h=k[3])[
        keep + ["period", "rung", "placement", "h"]] for k, v in frames.items()],
        ignore_index=True)
    pq.to_parquet(STEP2_OUT / f"winsor_county_year{sfx}.parquet", index=False)
    print(f"\n  {len(fit):,} fits -> {dc.LOG_DIR / f'winsor_construction_error{sfx}.csv'}")
    print(f"  {len(cut):,} cutoff rows -> {dc.LOG_DIR / f'winsor_cutoffs{sfx}.csv'}")
    print(f"  {len(pq):,} county-years -> {STEP2_OUT / f'winsor_county_year{sfx}.parquet'}")

    if canonical:
        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        # 2SLS ONLY, on instruction. The OLS block still runs, still passes the same
        # identities and is still in the CSV; it is not typeset because the ratio,
        # the bootstrap interval and the first-stage F have no OLS counterpart and
        # half of every OLS column was blank.
        cols = [("naics4_alloc", "2SLS"), ("naics2", "2SLS")]

        # (file, caption, label, extra note sentence, preview blurb). The caption
        # states the finding, in the house style 43_tables_tex.py uses; the blurb is
        # what the preview's contents page says the table is for. Both are written
        # from this one list, so a table cannot be described twice differently.
        # Captions are deliberately plain: the era, and the one setting that differs
        # from the other tables. What the numbers mean is the Notes' job and the
        # README's; a caption that argues the finding would have to be rewritten
        # every time a coefficient moved.
        SPEC = [
            ("TableW1_winsor.tex",
             "The winsorization construction error, pre-1998",
             "tab:winsor_headline", "",
             "The headline: pre-1998, unweighted, cutoffs within each start year. "
             "The other tables each change one setting against this one."),
            ("TableW2_winsor_weighted.tex",
             "The construction error, pre-1998, employment-weighted",
             "tab:winsor_weighted",
             "Table~\\ref{tab:winsor_headline} weighted by base county employment.",
             "Changes the weighting. Employment weights move mass onto large "
             "counties, whose cells are individually less extreme."),
            ("TableW3_winsor_post.tex",
             "The construction error, post-1998",
             "tab:winsor_post",
             "Reported separately from Table~\\ref{tab:winsor_headline} and never "
             "stacked with it: a four-digit cell is a different object either side "
             "of the 1998 NAICS break.",
             "Changes the era. Post-1998, otherwise identical to Table~W1."),
            ("TableW4_winsor_construction.tex", None, None, None,
             "Construction diagnostics: where the two cutoffs fall, how much they "
             "bind, the entrant mass, and the declared A.3.1 screen counts."),
            ("TableW5_winsor_pooled.tex",
             "The construction error, pre-1998, cutoff pooled over the era",
             "tab:winsor_pooled",
             "One cutoff over the whole era rather than one per start year, which is "
             "what a paper capping its whole sample at once would do.",
             "Changes where the cutoff is computed: one for the whole era instead of "
             "one per start year."),
            ("TableW6_winsor_pooled_weighted.tex",
             "The construction error, pre-1998, employment-weighted, cutoff pooled "
             "over the era",
             "tab:winsor_pooled_weighted",
             "The pooled cutoff of Table~\\ref{tab:winsor_pooled} under the weights "
             "of Table~\\ref{tab:winsor_weighted}.",
             "Changes both at once: pooled cutoff and employment weights."),
        ]
        arg_by_file = {
            "TableW1_winsor.tex": ("pre", "unweighted", "year"),
            "TableW2_winsor_weighted.tex": ("pre", "employment", "year"),
            "TableW3_winsor_post.tex": ("post", "unweighted", "year"),
            "TableW5_winsor_pooled.tex": ("pre", "unweighted", "pooled"),
            "TableW6_winsor_pooled_weighted.tex": ("pre", "employment", "pooled"),
        }
        made, entries = [], []
        for name, title, label, extra, blurb in SPEC:
            if name in arg_by_file:
                per, wt, pl = arg_by_file[name]
                made.append(tex_table(fit, per, wt, pl, HEADLINE_H, cols, title,
                                      label, extra, RESULT_DIR / name))
            else:
                made.append(tex_diagnostics(fit, cut, frames, screens, args,
                                            RESULT_DIR / name))
            entries.append((Path(name).stem, title, blurb))
        made.append(tex_coeff(fit, RESULT_DIR / "coeff_winsor.tex", args))
        made.append(tex_preview(
            entries, RESULT_DIR / "_preview_winsor.tex",
            f"$h={HEADLINE_H}$, instrument \\texttt{{z\\_income\\_{args.share}}}, "
            f"caps at \\texttt{{winsor4 iq level({args.level:g})}} on both levels. "
            f"Compiled from the saved fragments in "
            f"\\texttt{{4\\_Macro\\_version/step\\_2/result/}}; not a paper section."))

        # Superseded tables are REMOVED by the run. TableW3 was the exposure-5
        # robustness of the withdrawn symmetric-nonoverlap build; the exposure IS
        # fixed5 now, so the table has no referent. A stale table on disk beside
        # current ones is worse than no table.
        for stale_name in ("TableW3_winsor_exposure5.tex",
                           "TableW1_winsor_pre1998.tex",
                           "TableW2_winsor_pre1998_weighted.tex",
                           "TableW3_winsor_post1998.tex"):
            stale = RESULT_DIR / stale_name
            if stale.exists():
                stale.unlink()
                print(f"\n  removed {stale.name}: superseded by this build")
        print("\n  tables ->")
        for m in made:
            print(f"    {m}")

    report(fit, cut, frames, args, screens)
    print(f"\n  elapsed {time.time() - t0:.0f}s")
    if failures:
        print(f"\nFAIL: {len(failures)} check(s) failed", file=sys.stderr)
        for n, w in failures:
            print(f"  {n}: {w}", file=sys.stderr)
        tee.close()
        return 1
    print("\nall checks passed")
    tee.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
