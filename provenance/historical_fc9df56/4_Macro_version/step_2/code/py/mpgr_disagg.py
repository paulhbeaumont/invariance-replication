#!/usr/bin/env python3
"""Shared machinery for the macro disaggregation exhibits.

One frame, one cell panel, one estimator, one set of assertions, consumed by
every table producer so that no two exhibits can silently differ in their
sample or their construction.

THE COLUMN LADDER. Every table has the same five columns:

    (1) county                                    ccg + year
    (2) county x NAICS2                           ccg + year
    (3) county x NAICS2  + industry x year        ccg + industry x year
    (4) county x NAICS4                           ccg + year
    (5) county x NAICS4  + industry x year        ccg + industry x year

`ccg x industry` appears in NO growth-rate column. Under Part 0's mapping the
cell m IS the industry, so `industry x year` is `m x t` and admissible by A.1.3,
while `ccg x industry` is `i x m`, which is in neither A.1.3's admissible list
nor A.1.4's inadmissible one, and which removes each cell's own mean growth and
so destroys the (c, t)-function condition Proposition 1 needs. The one exception
is Poisson, where A.1.5 prescribes cell FE at the disaggregated level by name.

THE WEIGHTS, and which is which. Four vectors, and the difference between the
first two is the whole reason this file exists:

    w_naive   E_im,t-h / E_c,t-h        the paper's w^-. Denominator is the
                                        county's FULL lagged employment. On the
                                        survivor cells a log regression can use,
                                        these sum to LESS than one, and a county
                                        that lost employment to exits therefore
                                        carries less total weight. This is what
                                        an applied person computes from CBP,
                                        where lagged zeros are rarely visible.
    w_hat_c   E_im,t-h / E^c_c,t-h      eq continuer_objects. Denominator is
                                        lagged employment of SURVIVORS only, so
                                        these sum to one. Needed for the Jensen
                                        identity to close, and used only inside
                                        the decomposition.
    w_mpgr    (E_imt + E_im,t-h)        the midpoint weight, A.1.5. Sums to one
              / (E_ct + E_c,t-h)        over ALL cells including entrants and
                                        exiters, which is what makes weighted
                                        cell midpoint growth reconstruct the
                                        county exactly.
    w_equal   1                          equal per cell: beta^d_OLS, what a
                                        replicating paper runs.

NOTHING HERE WINSORIZES A GROWTH RATE (A.3.2). The shock and the instrument
carry the candidate table's IQR5-by-year rule and nothing else does.

THE EMPLOYMENT SOURCE IS A PARAMETER (WS2 step 3). `source` selects which
employment series the cells are built from:

    efsy          the desuppressed CBP cells, 11_build_employment_cells.py
    nets_actual   NETS, actual-employment universe   <- the WS2 headline
    nets_all      NETS, full panel                   <- its robustness

It changes the OUTCOME and nothing else: the frame, the instrument, the
treatment, the fixed effects, the weights and every assertion are the same
objects. Under the WS2 protocol gate this is a single bundled ladder row
(source + universe + support intersection + A.3.4 exclusions), and the exhibit
notes have to enumerate the bundle rather than call it "the NETS column".

Two things the NETS sources do to the sample, both reported and neither silent:
NETS begins in 1990, so a window [t-h, t] is estimable only from t = 1990 + h;
and estimation runs on the INTERSECTION of the EFSY and NETS ccg supports, with
any ccg reporting zero NETS employment inside the window excluded whole at every
rung (A.3.4). See step_1/code/py/76_nets_rung_loader.py.
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
STEP2_CODE = HERE.parent
STEP1_CODE = HERE.parents[3] / "step_1" / "code" / "py"
REPO_ROOT = HERE.parents[4]
RESULT_DIR = HERE.parents[2] / "result"

sys.path.insert(0, str(STEP1_CODE))
sys.path.insert(0, str(STEP2_CODE))
import dcads_common as dc  # noqa: E402
from out_paths import SCRATCH_ENV, scratch_root  # noqa: E402


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


t29 = _load("candidate_mpgr", STEP2_CODE / "29_candidate_iv_mpgr_check.py")
t28, c27, w22 = t29.t28, t29.c27, t29.w22
check, Tee = w22.check, w22.Tee

RUNGS = ("naics2", "naics4_alloc")
PRETTY = {"naics2": r"NAICS2", "naics4_alloc": r"NAICS4"}

# The cache normally lives in the shared Dropbox tree so that co-authors share it.
# Under INVARIANCE_SCRATCH_ROOT it moves to a local directory instead - see
# out_paths.py for why. The manifest fingerprints the SOURCE files either way, so a
# cache built under one root and a cache built under the other are validated against
# the same inputs and neither can be silently estimated on after a source rebuild.
_SCRATCH = scratch_root()
CACHE_DIR = ((_SCRATCH / "cells_panel") if _SCRATCH is not None
             else dc.CURRENT_PAPER_DATA_ROOT / "temp" / "cells_panel")
# Bumped from 1 when `source` entered the cache key: a v1 directory carries no
# record of which employment series built it, so it can never be trusted again.
CACHE_VERSION = 2
CELL_KEYS = ("cur", "lag", "f_cell", "f_ky", "f_cy")

SOURCES = ("efsy", "nets_actual", "nets_all")
_N76 = None


def _n76():
    """step_1's NETS rung adapter, loaded once and only if a NETS source asks."""
    global _N76
    if _N76 is None:
        _N76 = _load("nets_rung", STEP1_CODE / "76_nets_rung_loader.py")
    return _N76


def _load_cells(rung, years, source, frame_ccgs, quiet=False):
    """(emp, ccg_arr, code_arr) for either employment source, on one support.

    The EFSY path asks two files for the support (`w22.load_rung` for the
    employment, `c27.rung_codes` for the codes) because there the support is a
    property of what is on disk. The NETS path constructs the support, so its
    adapter hands back all three together -- see 76's docstring on why the
    zeros have to be materialised rather than joined in later.
    """
    if source == "efsy":
        emp, ccg_arr = w22.load_rung(rung, years)
        _, code_arr = c27.rung_codes(rung, years[0])
    else:
        emp, ccg_arr, code_arr = _n76().load_rung(
            rung, years, _n76().universe_of(source), quiet=quiet)
    keep = np.isin(ccg_arr, frame_ccgs)
    if not keep.all():
        emp = {y: v[keep] for y, v in emp.items()}
        ccg_arr, code_arr = ccg_arr[keep], code_arr[keep]
    return emp, ccg_arr, code_arr


def _county_reference(source):
    """The independent county series A.1.2 checks the cell sums against."""
    if source == "efsy":
        return pd.read_parquet(dc.EMP_OUT_DIR / "efsy_ccg_year.parquet",
                               columns=["ccg", "year", "emp"])
    return _n76().county_reference(_n76().universe_of(source))


def nets_frame_rule(frame, h, source, quiet=False):
    """Apply the two NETS sample rules to a frame, and say what they cost.

    1. the window has to lie inside the NETS panel: t >= 1990 + h;
    2. the ccg support is EFSY-intersect-NETS, less the A.3.4 degenerates
       (zero NETS employment in any year of the span), dropped WHOLE.

    Returns (frame, info). `info` is what the exhibit note prints; nothing here
    is allowed to happen without being counted.
    """
    if source == "efsy":
        return frame, dict(source=source)
    n76 = _n76()
    uni = n76.universe_of(source)
    n0, ccg0 = len(frame), frame["ccg"].nunique()
    floor = n76.year_floor(h)
    frame = frame[frame["year"] >= floor]
    n_after_year = len(frame)

    years = sorted(set(frame["year"]) | set(frame["year"] - h))
    have = set(n76.ccg_support(uni).tolist())
    degen = set(n76.degenerate_ccgs(uni, years).tolist())
    ccgs = frame["ccg"].to_numpy()
    n_no_nets = int(len(set(ccgs.tolist()) - have))
    n_degen = int(len(set(ccgs.tolist()) & degen))
    frame = frame[np.isin(ccgs, sorted(have - degen))].reset_index(drop=True)

    info = dict(source=source, universe=uni, year_floor=int(floor),
                n_dropped_pre_nets=int(n0 - n_after_year),
                n_ccg_absent_from_nets=n_no_nets,
                n_ccg_degenerate=n_degen,
                n_county_years=len(frame), n_counties=int(frame["ccg"].nunique()),
                n_county_years_before=int(n0), n_counties_before=int(ccg0))
    if not quiet:
        print(f"    [nets] sample: {n0:,} county-years / {ccg0:,} CCGs -> "
              f"{len(frame):,} / {frame['ccg'].nunique():,}; "
              f"{n0 - n_after_year:,} dropped for year < {floor} (NETS starts "
              f"1990), {n_no_nets:,} CCGs absent from NETS, {n_degen:,} "
              f"degenerate under A.3.4")
    return frame, info

TOL_SUM = 1e-9        # cells against EFSY's county employment, relative
TOL_RECON = 1e-12     # sum_m w mpgr = mpgr_c, absolute
TOL_WEIGHT = 1e-12    # weight vectors summing to one
MPGR_BOUND = 2.0


def prune(d, n_frame, rung, h):
    """Reduce the cells to the estimable support, then rebuild the county totals
    from what is left.

    Two rules, applied in this order.

    A cell carrying NEGATIVE employment in either period is excluded outright.
    Employment is a headcount; a negative one is an allocation artefact and not a
    firm leaving, and it has no admissible growth rate at all.

    A cell that is EXACTLY zero in both periods is excluded because no estimator
    can use it: the log sample needs both periods positive, the standard-growth
    sample needs a positive base, and its midpoint weight is
    `(0 + 0) / (E_ct + E_c,t-h)`, which is zero. It contributes nothing to any
    coefficient and costs a row in every array and every bincount.

    `S_i = sum_m S_im` is then TRUE OF THE CELLS THAT SURVIVE, because the three
    county aggregates are recomputed here rather than carried over from the rows
    that were read. That is what keeps A.1.2 an honest check: the county totals
    are compared against EFSY's independent series afterwards, so if the negative
    rule ever drops anything the cells stop matching EFSY and the check fails
    loudly instead of the difference being absorbed silently.
    """
    cur, lag = d["cur"], d["lag"]
    neg = (cur < 0) | (lag < 0)
    dead = ~neg & (cur == 0) & (lag == 0)
    keep = ~(neg | dead)
    n0 = len(cur)
    for k in CELL_KEYS:
        v = d[k][keep]
        d[k] = v.astype(np.int32) if k != "cur" and k != "lag" else v
    d["den_c"] = np.bincount(d["f_cy"], weights=d["cur"] + d["lag"],
                             minlength=n_frame)
    d["lag_c"] = np.bincount(d["f_cy"], weights=d["lag"], minlength=n_frame)
    d["cur_c"] = np.bincount(d["f_cy"], weights=d["cur"], minlength=n_frame)
    d["n_built"], d["n_neg"], d["n_dead"] = n0, int(neg.sum()), int(dead.sum())
    check(True, f"cells pruned to the estimable support (h={h}, {PRETTY[rung]})",
          f"{int(keep.sum()):,} kept of {n0:,}; dropped {d['n_neg']:,} negative, "
          f"{d['n_dead']:,} zero in both periods ({100 * dead.mean():.1f}%)")
    return d


_P57 = None


def _p57():
    """step_1's 57, loaded once; its internal panels() cache makes the second
    and later build_county calls cheap."""
    global _P57
    if _P57 is None:
        _P57 = _load("p57", STEP1_CODE / "57_aggregation_invariance.py")
    return _P57


BASELINE_SPEC = dict(span=5, rule="positive", floor=0.0, denom="population")

_S0_CACHE = {}


def share_sum(h, ccg, year):
    """S0_ct, the county share-sum LEVEL: rung 0' minus rung 0.

    The cross-sectional incomplete-shares control BHJ Assumption 1 asks for, and
    the reference rung `ws0_contingency.py` records Adrien choosing (revision 1,
    superseding rung 1). Built with the same three lines as
    `22_bhj_assumption1.py:335-341` -- the predetermined exposure window summed
    over categories, divided by base-period population -- off the same tensor, so
    the object is the producer-22 one and not a second construction of it.

    Returned aligned to (ccg, year); the caller decides how it enters. Note that
    22 evaluates it on ITS frame (employment-share weights within division-year,
    winsorized treatment); on the locked baseline's frame and weights this is a
    new object and every exhibit that uses it has to say so.
    """
    key = int(h)
    if key not in _S0_CACHE:
        sys.path.insert(0, str(STEP1_CODE.parents[0] / "pipeline"))
        sys.path.insert(0, str(STEP1_CODE.parents[0] / "pipeline" / "lib"))
        import locked as lk           # noqa: E402
        import store                  # noqa: E402
        pnl = store.load_county_panel()
        units, piv = store.load_pivots(pnl, cols=("P",))
        units2, cats, cy, arr, _ = store.load_tensor(lk.RULE)
        assert np.array_equal(units, units2), "unit order drifted between stores"
        years = store.target_years(h)
        P_b = piv["P"].reindex(columns=[y - h for y in years]).to_numpy(np.float64)
        S0 = np.full((len(units), len(years)), np.nan)
        for j, t in enumerate(years):
            hi = t - h - lk.LAG
            lo = hi - lk.SPAN + 1
            if lo < lk.DATA_FIRST:
                continue
            a, b = cy.index(lo), cy.index(hi)
            m = arr[:, :, a:b + 1].mean(axis=2).astype(np.float64)
            S0[:, j] = m.sum(axis=1) / P_b[:, j]
        _S0_CACHE[key] = pd.DataFrame(S0, index=units, columns=years).stack()
    s = _S0_CACHE[key]
    idx = pd.MultiIndex.from_arrays([np.asarray(ccg), np.asarray(year)])
    return s.reindex(idx).to_numpy(np.float64)


def baseline_frame(h, source="efsy", quiet=False, year_floor=None):
    """The locked baseline's county frame: 57's build_county at the locked
    specification (harmonized top-800 composition instrument, span 5, positive
    rule, population denominator), 1975+h to 2018, no IQR winsor -- the locked
    baseline carries none -- with the NETS sample rules applied on top.

    Shared with 37_table_poisson.py, which needs the same frame and a different
    cell construction, so the frame cannot live inside the cell builder.
    """
    d0, _ = _p57().build_county("composition", h=h, **BASELINE_SPEC)
    d0 = d0[np.isfinite(d0["y_mpgr"]) & np.isfinite(d0["z"])]
    d0 = d0[(d0["year"] >= 1975 + h) & (d0["year"] <= 2018)]
    frame = (d0[["ccg", "year", "x", "z", "division", "P0", "y_mpgr"]]
             .rename(columns={"x": "shock"}).reset_index(drop=True))
    # `year_floor` exists for ONE purpose: an EFSY run on the window NETS can
    # actually reach. NETS begins in 1990, so a NETS column starts at 1990 + h
    # and an unrestricted EFSY column starts at 1982 -- comparing them confounds
    # the data source with the era, which is the one thing this exhibit is meant
    # to separate. It is not a general sample knob and no NETS run passes it:
    # there the floor is a property of the data, applied by nets_frame_rule.
    if year_floor is not None:
        n0 = len(frame)
        frame = frame[frame["year"] >= int(year_floor)].reset_index(drop=True)
        if not quiet:
            print(f"    [window] matched to the NETS floor {int(year_floor)}: "
                  f"{n0:,} county-years -> {len(frame):,}")
    return nets_frame_rule(frame, h, source, quiet)


def _build_baseline(h, rung, quiet, source="efsy", year_floor=None):
    """The baseline frame plus the same cell construction and pruning as the
    candidate frame."""
    frame, info = baseline_frame(h, source, quiet, year_floor)
    years = sorted(set(frame["year"]) | set(frame["year"] - h))
    emp, ccg_arr, code_arr = _load_cells(rung, years, source,
                                         frame["ccg"].unique(), quiet)
    d = prune(t28.build_long(rung, emp, ccg_arr, code_arr, frame, h),
              len(frame), rung, h)
    return frame, d, (ccg_arr, code_arr, years), info


def _build(period, h, rung, share, level, quiet, source="efsy"):
    """The frame, the pruned cells, and the unpruned support the checks need."""
    frame = t29.build_frame(period, h, share, level) if not quiet else \
        _quiet(lambda: t29.build_frame(period, h, share, level))
    frame, info = nets_frame_rule(frame, h, source, quiet)
    years = sorted(set(frame["year"]) | set(frame["year"] - h))
    emp, ccg_arr, code_arr = _load_cells(rung, years, source,
                                         frame["ccg"].unique(), quiet)
    d = prune(t28.build_long(rung, emp, ccg_arr, code_arr, frame, h),
              len(frame), rung, h)
    return frame, d, (ccg_arr, code_arr, years), info


def cache_dir(period, h, rung, share, level, source="efsy"):
    return CACHE_DIR / f"{period}_{source}_{rung}_h{h}_{share}_iqr{level:g}"


def _fingerprint(period, h, rung, years, source="efsy"):
    """Size and mtime of every file the panel is derived from, so a rebuilt
    source invalidates the cache instead of being silently estimated on.

    The employment files listed are the ones the panel was ACTUALLY built from.
    A cache keyed on the rung alone would let a NETS panel pass a freshness
    check against EFSY parquets it never read, which is why `source` is in the
    directory name as well as here."""
    paths = [t29.CANDIDATE / "data" / "temp" / f"{period}_h{h}.dta"]
    if source == "efsy":
        paths.append(dc.EMP_OUT_DIR / "efsy_ccg_year.parquet")
        paths += [dc.EMP_OUT_DIR / rung / f"y{y}.parquet" for y in years]
    else:
        paths += _n76().fingerprint_paths()
    return {str(p): [p.stat().st_size, p.stat().st_mtime_ns] for p in paths}


def _cache_read(period, h, rung, share, level, source="efsy"):
    path = cache_dir(period, h, rung, share, level, source)
    if not (path / "manifest.json").exists():
        return None
    man = json.loads((path / "manifest.json").read_text())
    if man.get("version") != CACHE_VERSION:
        return None
    try:
        fresh = _fingerprint(period, h, rung, man["years"], source)
    except OSError:
        return None
    if fresh != man["fingerprint"]:
        print(f"    [cache] {path.name} is stale against its sources, rebuilding")
        return None
    frame = pd.read_parquet(path / "frame.parquet")
    z = np.load(path / "cells.npz")
    d = {k: z[k] for k in CELL_KEYS}
    d.update(den_c=z["den_c"], lag_c=z["lag_c"], cur_c=z["cur_c"])
    d["n_code"], d["n_year"], d["n_cell"] = (int(v) for v in z["meta"])
    d["n_built"], d["n_neg"], d["n_dead"] = (int(man[k]) for k in
                                             ("n_built", "n_neg", "n_dead"))
    print(f"    [cache] {path.name}: {len(d['cur']):,} cells, "
          f"{len(frame):,} county-years")
    return frame, d, man


def _cache_write(path, frame, d, man):
    path.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path / "frame.parquet", index=False)
    np.savez(path / "cells.npz", **{k: d[k] for k in CELL_KEYS},
             den_c=d["den_c"], lag_c=d["lag_c"], cur_c=d["cur_c"],
             meta=np.array([d["n_code"], d["n_year"], d["n_cell"]], np.int64))
    (path / "manifest.json").write_text(json.dumps(man, indent=1))
    mb = sum(f.stat().st_size for f in path.iterdir()) / 1e6
    print(f"    [cache] wrote {path.name} ({mb:.0f} MB)")


class Panel:
    """One (h, rung) bundle: the frame, the cells, the weights, and its checks.

    Construction is verified before the Panel exists, either here or at the time
    the cache it was loaded from was written. The two checks that have to re-read
    the source parquets are replayed from the cache manifest; every other
    assertion is a bincount over arrays already in memory and runs on every
    construction, cached or not, so a cache corrupted after it was written still
    fails. The manifest is keyed to the size and mtime of every source file, so a
    rebuilt source invalidates the cache rather than being estimated on.
    """

    def __init__(self, h, rung, share="fixed5", level=5.0, period="pre",
                 quiet=False, cache=True, rebuild=False, frame_src="candidate",
                 source="efsy", year_floor=None):
        if source not in SOURCES:
            raise ValueError(f"unknown employment source {source!r} "
                             f"(expected one of {SOURCES})")
        self.h, self.rung, self.share, self.level = h, rung, share, level
        self.period = period
        self.frame_src = frame_src
        self.source = source
        self.checks = []

        if frame_src == "baseline":
            # no cache: the frame depends on step_1 state the fingerprint does
            # not cover, and the build is about two minutes
            self.frame, self.d, support, self.sample = _build_baseline(
                h, rung, quiet, source, year_floor)
            self._bind()
            self._assert(quiet, support=support)
            return

        hit = None if rebuild or not cache else \
            _cache_read(period, h, rung, share, level, source)
        if hit is None:
            self.frame, self.d, support, self.sample = _build(
                period, h, rung, share, level, quiet, source)
            self._bind()
            self._assert(quiet, support=support)
            if cache:
                years = support[2]
                _cache_write(
                    cache_dir(period, h, rung, share, level, source),
                    self.frame, self.d,
                    dict(version=CACHE_VERSION, period=period, h=h, rung=rung,
                         share=share, level=level, source=source, years=years,
                         fingerprint=_fingerprint(period, h, rung, years, source),
                         sample=self.sample,
                         checks=self.checks, n_built=self.d["n_built"],
                         n_neg=self.d["n_neg"], n_dead=self.d["n_dead"]))
        else:
            self.frame, self.d, man = hit
            self.sample = man.get("sample", dict(source=source))
            self._bind()
            self._assert(quiet, cached=man["checks"])

    def _bind(self):
        frame, d = self.frame, self.d
        self.n = len(frame)
        self.shock = frame["shock"].to_numpy()
        self.z = frame["z"].to_numpy()
        self.ccg = frame["ccg"].to_numpy()
        self.yr = frame["year"].to_numpy()
        self.cur, self.lag, self.cy = d["cur"], d["lag"], d["f_cy"]
        self.cur_c, self.lag_c = d["cur_c"], d["lag_c"]
        self.mpgr, self.w_mpgr, self.log, self.pos = t28.cell_outcomes(d)

        # Survivor aggregates. `pos` is the set a log regression can use.
        self.cc = np.bincount(self.cy[self.pos], weights=self.cur[self.pos],
                              minlength=self.n)
        self.lc = np.bincount(self.cy[self.pos], weights=self.lag[self.pos],
                              minlength=self.n)
        self.n_c = np.bincount(self.cy[self.pos], minlength=self.n).astype(float)

        # `ok` is not a cleaning rule. It is the set on which the LOG objects
        # exist at all; midpoint and Poisson are defined everywhere. The
        # county-years outside it are counties where every industry turned over,
        # which is the extensive margin at 100%, and they are COUNTED and
        # reported rather than quietly dropped (A.3.1).
        self.ok = (self.cc > 0) & (self.lc > 0) & (self.cur_c > 0) \
            & (self.lag_c > 0)
        self.m = self.pos & self.ok[self.cy]
        self.n_log_blind = int((~self.ok).sum())

        self._weights()

    def _rec(self, ok, label, detail):
        self.checks.append(dict(ok=bool(ok), label=label, detail=detail))
        return check(ok, label, detail)

    # ---------------------------------------------------------------------
    def _weights(self):
        m, cy, lag = self.m, self.cy, self.lag
        self.w_naive = lag[m] / self.lag_c[cy[m]]
        self.w_hat_c = lag[m] / self.lc[cy[m]]
        self.w_equal = np.ones(int(m.sum()))
        self.w_1n = 1.0 / self.n_c[cy[m]]
        self.mass_naive = np.bincount(cy[m], weights=self.w_naive,
                                      minlength=self.n)

    def _assert(self, quiet, support=None, cached=None):
        h, rung, d, ok = self.h, self.rung, self.d, self.ok
        tag = f"h={h}, {PRETTY[rung]}"

        # The two checks that re-read the source parquets, replayed from the
        # manifest on a cache hit. A recorded FAIL is re-raised into the failure
        # list rather than being printed and forgotten.
        if cached is not None:
            for c in cached:
                print(f"    [{'ok ' if c['ok'] else 'FAIL'}|cached] {c['label']}"
                      + (f" - {c['detail']}" if c["detail"] else ""))
                if not c["ok"]:
                    w22.failures.append((c["label"], c["detail"]))
        else:
            # The county series is INDEPENDENT of the cells at every source: for
            # EFSY it is 11's own county file, for NETS producer 73's own county
            # GROUP BY. Comparing against the matching one keeps A.1.2 a real
            # check; comparing a NETS panel against EFSY would fail everywhere
            # and say nothing about the construction.
            ref = _county_reference(self.source)
            e = {y: g.set_index("ccg")["emp"] for y, g in ref.groupby("year")}
            worst = 0.0
            for y, sub in self.frame.groupby("year", sort=True):
                i = sub.index.to_numpy()
                for series, arr in ((e[y], self.cur_c), (e[y - h], self.lag_c)):
                    ref = series.reindex(sub["ccg"].to_numpy()).to_numpy(np.float64)
                    worst = max(worst, float(np.nanmax(
                        np.abs(arr[i] - ref) / np.maximum(np.abs(ref), 1.0))))
            self._rec(worst < TOL_SUM,
                      f"A.1.2  cells sum to county employment ({tag})",
                      f"max relative gap {worst:.2e}")

        den_c = self.cur_c + self.lag_c
        nz = den_c != 0.0
        mpgr_c = np.where(nz, 2.0 * (self.cur_c - self.lag_c)
                          / np.where(nz, den_c, 1.0), np.nan)
        self.mpgr_c = mpgr_c
        agg = np.bincount(self.cy, weights=self.w_mpgr * self.mpgr,
                          minlength=self.n)
        worst = float(np.nanmax(np.abs(agg - mpgr_c)[nz]))
        check(worst < TOL_RECON,
              f"A.1.2  weighted cell midpoint growth reconstructs the county ({tag})",
              f"max |sum_m w mpgr - mpgr_c| = {worst:.2e} over {int(nz.sum()):,}")

        check(bool(np.all(np.abs(self.mpgr) <= MPGR_BOUND + 1e-12)),
              f"A.3.1  cell midpoint growth is inside [-2, 2] ({tag})",
              f"max |mpgr| = {float(np.max(np.abs(self.mpgr))):.6f}")

        s_mp = np.bincount(self.cy, weights=self.w_mpgr, minlength=self.n)
        check(float(np.abs(s_mp[nz] - 1.0).max()) < TOL_WEIGHT,
              f"the MPGR weights sum to one over ALL cells ({tag})",
              f"max |sum w - 1| = {float(np.abs(s_mp[nz] - 1.0).max()):.2e}")
        s_hc = np.bincount(self.cy[self.m], weights=self.w_hat_c,
                           minlength=self.n)
        check(float(np.abs(s_hc[ok] - 1.0).max()) < TOL_WEIGHT,
              f"w-hat^c sums to one over the survivors ({tag})",
              f"max |sum w - 1| = {float(np.abs(s_hc[ok] - 1.0).max()):.2e}")
        check(bool((self.mass_naive[ok] <= 1.0 + 1e-12).all()),
              f"w^- sums to at most one on the survivors ({tag})",
              f"mean retained {self.mass_naive[ok].mean():.4f}, "
              f"min {self.mass_naive[ok].min():.4f}")

        if cached is None:
            r = c27.support_completeness(rung, support[0], support[1])
            self.checks.append(dict(
                ok=r["sup_short"] == 0,
                label=f"2e the support IS the cross product ({rung})",
                detail=f"{r['sup_pairs']:,} pairs = {r['sup_n_ccg']:,} x "
                       f"{r['sup_n_code']:,}, short by {r['sup_short']:,}"))

    # ---------------------------------------------------------------------
    def fit_county(self, y, mask, w=None, correction_n=None):
        """One row per county-year. `mask` is explicit because midpoint is
        defined on every county-year and log is not, and the two samples are
        deliberately NOT harmonised: the county-years log cannot use are the
        extensive margin at 100%, and are reported rather than removed.
        """
        w = np.ones(int(mask.sum())) if w is None else w
        return t28.fit2sls(y[mask], self.shock[mask], self.z[mask], w,
                           self.ccg[mask], self.yr[mask], self.ccg[mask],
                           singleton=False, correction_n=correction_n)

    def fit_cell(self, y, w, mask, indyear=False, correction_n=None):
        f2 = self.d["f_ky"][mask] if indyear else self.yr[self.cy][mask]
        return t28.fit2sls(y[mask], self.shock[self.cy][mask],
                           self.z[self.cy][mask], w, self.ccg[self.cy][mask],
                           f2, self.ccg[self.cy][mask], singleton=False,
                           correction_n=correction_n)

    def _divyear(self):
        return pd.factorize(self.frame["division"].astype(str) + "_"
                            + self.frame["year"].astype(str))[0].astype(np.int64)

    def _s0(self):
        """The rung-0' control on this frame, cached per Panel."""
        if not hasattr(self, "_s0_vec"):
            self._s0_vec = share_sum(self.h, self.ccg, self.yr)
            bad = int(np.isnan(self._s0_vec).sum())
            if bad:
                raise SystemExit(
                    f"the share sum is missing on {bad:,} of {self.n:,} "
                    f"county-years at h={self.h}. Rung 0' cannot be estimated "
                    f"on a frame the exposure window does not cover; fix the "
                    f"frame rather than dropping the rows, which would make "
                    f"rung 0 and rung 0' two different samples.")
        return self._s0_vec

    @staticmethod
    def _fwl(c, w, *vs):
        """Residualise on a single already-demeaned control column, weighted.

        One column, so a scalar projection rather than 22's SVD -- but the same
        step, and it is done AFTER the two-way demeaning for the same reason 22
        gives: the residual has to be orthogonal to the control to working
        precision or the identity the invariance check tests is not the one the
        algebra describes.
        """
        den = float(c @ (w * c))
        if den <= 0:
            return list(vs)
        return [v - c * (float(c @ (w * v)) / den) for v in vs]

    def fit_county_base(self, y, mask, w=None, rung="0", correction_n=None):
        """The locked baseline's county regression: reduced form on z,
        division-by-year effects, county clusters, weights passed in.

        `rung` is the BHJ shock-control rung: "0" is no control, the spec these
        tables have always run; "0'" adds the county share-sum level and is the
        reference rung `ws0_contingency.py` records.
        """
        w = np.ones(int(mask.sum())) if w is None else w
        dy = self._divyear()
        s0 = self._s0()[mask] if rung != "0" else None
        return self._rung_fit_vec(y[mask], self.z[mask], w, dy[mask],
                                  np.zeros(int(mask.sum()), np.int64),
                                  self.ccg[mask], s0, correction_n=correction_n)

    def fit_cell_base(self, y, w, mask, indyear=False, rung="0",
                      correction_n=None):
        dy = self._divyear()[self.cy]
        f2 = self.d["f_ky"][mask] if indyear \
            else np.zeros(int(mask.sum()), np.int64)
        s0 = self._s0()[self.cy][mask] if rung != "0" else None
        return self._rung_fit_vec(y[mask], self.z[self.cy][mask], w, dy[mask],
                                  f2, self.ccg[self.cy][mask], s0,
                                  correction_n=correction_n)

    def _rung_fit_vec(self, y, z, w, f1, f2, cl, s0, correction_n=None):
        """Reduced form, with the rung-0' control partialled out when given.

        The control is removed by FWL because t28.fit2sls has no covariate slot,
        and it is demeaned inside the regression's own fixed effects and weights
        first, for the reason 22_bhj_assumption1.py's `fwl` records: the residual
        has to be orthogonal to the control to working precision, not merely
        close to it. At the cell rung the control is county-constant and repeats
        across cells exactly as z does, so the weighting identity behind
        Proposition 1 is untouched and the invariance assertion stays live.
        """
        w = w / w.mean()
        if s0 is not None:
            c = w22.demean(np.column_stack([s0]), f1, f2, w)[:, 0]
            y, z = self._fwl(c, w, y, z)
        return t28.fit2sls(y, z, z, w, f1, f2, cl, singleton=False,
                           correction_n=correction_n)

    # ---------------------------------------------------------------------
    def diagnostics(self):
        m, ok = self.m, self.ok
        return dict(
            source=self.source,
            n_county_years=int(ok.sum()),
            n_log_blind=self.n_log_blind,
            n_cells=int((self.cy >= 0).sum()),
            n_cells_log=int(m.sum()),
            n_cells_built=self.d["n_built"],
            n_dropped_negative=self.d["n_neg"],
            pct_zero_both=100.0 * self.d["n_dead"] / self.d["n_built"],
            share_emp_log=100.0 * float(self.cur[m].sum())
            / float(self.cur_c[ok].sum()),
            mass_naive=float(self.mass_naive[ok].mean()),
            n_counties=int(pd.unique(self.ccg[ok]).size))


def _quiet(fn):
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        return fn()
