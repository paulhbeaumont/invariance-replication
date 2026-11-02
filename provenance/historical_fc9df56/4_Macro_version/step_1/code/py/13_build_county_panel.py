#!/usr/bin/env python3
"""The county panel and the contract shock. Part 1.4.

Contracts collapsed to CCG x year, joined to BEA county personal income from
12_build_county_income.py and to county employment from
11_build_employment_cells.py. The geographic crosswalk remains complete, while
the analysis universe is fixed to CCGs with strictly positive employment in
every year from 1975 through 2006. The output is the regressor every
specification in Parts 2 and 3 runs on.

THE SHOCK

    G_ct     = sum over s <= t of D_cs                    the cumulative stock
    Shock_ct = ( G_ct - G_c,t-h ) / Y_c,t-h  =  ( sum of D over t-h+1..t ) / Y_c,t-h

D is real defense contract dollars, Y is real county personal income, both in
CPI_BASE dollars, and Y is LAGGED so the regressor is predetermined. The plan's
section 3.3 defines the shock as a ratio of an additive numerator to an additive
denominator because that is what makes the aggregate version - unit contract flow
over unit income - the income-weighted average of its counties' shocks, which is
the arithmetic Part 3 rests on. Accumulating D does not touch that property: a
window sum of sums is a sum of window sums.

WHY G IS CUMULATED                                                     --delta

DCADS records contract OBLIGATIONS, not spending flows. An obligation is a lumpy
one-time award covering years of work, so D_ct - D_c,t-h measures award TIMING
rather than the defense buildup. A county receiving one $1bn award in 1985 and
nothing else enters the h=2 flow difference as +$1bn in 1985 and again as
-$1bn in 1987: the same award, once as a positive shock and once as a spurious
negative one. On the panel that shows up as a second-order autocorrelation of
-0.39 for the flow difference against +0.64 for the cumulative one, and a
between-county standard deviation of 0.0051 against a within-county 0.0708 -
essentially no cross-sectional signal at all. Cumulating turns the same award
into a clean pulse of width h.

The cumulation's START YEAR CANCELS in the difference, which is why this is
implemented as an h-year rolling sum and asserted equal to the cumulative
difference (check 4b) and to itself under a truncated first year (check 4c).

    --delta flow    recovers D_ct - D_c,t-h, the default until 2026-08-10.
                    It is kept so that README 1.8.4's record of the abandoned
                    design stays reproducible, and for no other reason.

Negative D is fine here and is not an error: deobligations, a contract action
backed out. Under `cumulative` a negative SHOCK needs net deobligations across
the whole window and is rarer than the raw negative-D count; both are counted in
[4]. They sit in a numerator over an income denominator and never in a
denominator, which is exactly why the income scaling is worth having and why
scaling by lagged contracts would not be.

THREE CONSTRUCTION DECISIONS, each with a default and a flag

  dollar floor        NONE, as of 2026-08-10. Every contract action counts,
                      whatever its size, and nothing downstream reimposes one.
                      The default was $25,000 nominal until then, and it was
                      dropped because the floor was a guard against a reporting
                      threshold rather than a feature of the design, and one
                      convention across every step beats two.
                      What the floor was guarding against has NOT gone away and
                      is now a live confound rather than a closed decision. The
                      source's own reporting threshold moves - $10,000 FY1966-82,
                      $25,000 FY1983-99, about $1 FY2000-06 - so the unfloored
                      series carries breaks at FY1983 and FY2000 that are DoD
                      reporting dates and not defense-policy dates, and 423 of
                      FY2005's 2,638 counties exist only because of actions under
                      $25,000, so the county support now moves with the threshold
                      too. README 1.6.6 and 1.7.2 carry the accounting. Pass
                      --floor 25000 to recover the old default.       --floor
  TQ76                folded into FY1976. July-September 1976 is a one-off
                      bridging quarter created when the fiscal year moved from
                      July-June to October-September, and folding it into FY1976
                      is the convention Pallante's published series uses, which
                      is what 09_pallante_test.py verified FY1976 against.
                                                                    --drop-tq76
  fiscal -> calendar  FY t is calendar year t. FY t runs October t-1 to
                      September t from FY1977, and CBP's March-12 reference date
                      for year t sits inside it.                    --fy-offset

The fiscal calendar itself changed at FY1977, so 1975 and 1976 are aligned on a
different convention from every later year. --first-year 1978 drops them.

THE SEAM                                                    dc.SEAM_YEAR = 2007

The DCADS archive ends at FY2006. From FY2007 the contract dollars come from the
parsed USAspending prime-contract actions. dc.build_contract_panel performs the
splice so that this file and 14_build_instrument.py cannot disagree about it.

dG is an h-year WINDOW SUM, so it mixes the two sources only in the years whose
window straddles the seam - exactly SEAM_YEAR .. SEAM_YEAR + h - 2:

    h = 1   window {t}              nothing straddles; FY2007 is wholly post-seam
    h = 2   window {t-1, t}         FY2007
    h = 3   window {t-2, t-1, t}    FY2007, FY2008

Those years are MASKED, not deleted: dG and shock become NaN and the row stays.
Deleting them would break checks 1, 1b and 1c, which assert an exact rectangular
spine, and the panel's rectangularity is what Part 3's aggregation arithmetic
rests on. The masked count is asserted to be n_ccg * (h - 1).

Nothing else is contaminated. Checks 4b and 4c already prove the window sum does
not depend on where the cumulation starts, so `dollars_cum` crossing the seam
cannot leak into any other year's dG. dollars_cum is reported and never used.

Both sources carry individual actions, so n_actions and action-level dollar
floors are defined on either side of the seam.

Output:
    dc.PANEL_OUT_DIR/contracts_ccg_year.parquet
    dc.PANEL_OUT_DIR/county_panel.parquet
    dc.LOG_DIR/county_panel_report.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dcads_common as dc  # noqa: E402

FIRST_YEAR = 1975          # EFSY's NAICS panel starts here, so the county panel does
LAST_YEAR = 2018           # EFSY employment ends here, so the outcome sets the bound
DEFAULT_FLOOR = 0          # no floor as of 2026-08-10; see the docstring
DEFAULT_DELTA = "cumulative"   # G is a cumulative stock as of 2026-08-10; see the docstring
DELTAS = ("cumulative", "flow")

# 12_build_county_income.py declares Kalawao HI as having no income in any year,
# because BEA has never separated it from Maui. It draws $0 of contracts in every
# year of the panel, which that script asserts. Repeated here rather than
# imported so that this file states its own universe.
INCOME_GAP_CCGS = (15005,)


def check(failures, ok, num, label, detail=""):
    """Print one numbered PASS/FAIL line and record the failure."""
    print(f"    [{num}] {label:<52s} {detail:<34s} {'PASS' if ok else 'FAIL'}")
    if not ok:
        failures.append((label, detail))
    return ok


def reference_year_totals(years, floor, fold_tq76, universe):
    """Contract dollars per fiscal year, computed WITHOUT the collapse.

    The point of this check is the collapse and the rectangularisation, not the
    filter: both sides apply the same geo_class and floor rules, so a field read
    from the wrong offset would agree with itself. What it does catch is a county
    dropped by the groupby, a spine that does not cover the universe, and a
    reindex that filled a real value with zero - which are the failure modes a
    rectangular panel actually has.

    The reference splits at the seam because the two raw transaction schemas
    have separate annual parquets. The caller asserts that every requested year
    is present on exactly one side.
    """
    fips_to_ccg, unplaceable, _ = dc.load_ccg_map()
    eligible = set(map(int, universe))
    totals = {}
    pre = [y for y in years if y < dc.SEAM_YEAR]
    sources = [(y, f"fy{y}.parquet") for y in pre]
    if fold_tq76 and 1976 in pre:
        sources.append((1976, "fy1976tq.parquet"))
    for year, fname in sources:
        path = dc.OUT_DIR / fname
        if not path.exists():
            continue
        df = pd.read_parquet(
            path, columns=["dollars", "dollars_unit", "geo_class", "county_fips5"])
        df = df[df["geo_class"] == "county"]
        d = df["dollars"].to_numpy(dtype=np.float64)
        d = np.where((df["dollars_unit"] == "thousands").to_numpy(), d * 1000.0, d)
        mapped = df["county_fips5"].map(fips_to_ccg)
        keep = mapped.isin(eligible).to_numpy() & (np.abs(d) >= floor)
        totals[year] = totals.get(year, 0.0) + float(d[keep].sum())

    post = [y for y in years if y >= dc.SEAM_YEAR]
    if post:
        r = dc.collapse_usaspending_contracts(post, floor=floor)
        r = (r[r["ccg"].isin(eligible)]
             .groupby("year", sort=True)["dollars_nominal"].sum())
        for y in post:
            if y in r.index:
                totals[y] = float(r[y])
    return totals


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--first-year", type=int, default=FIRST_YEAR)
    ap.add_argument("--last-year", type=int, default=LAST_YEAR)
    ap.add_argument("--floor", type=int, default=DEFAULT_FLOOR,
                    help="nominal dollar floor on |action|; 0 means none")
    ap.add_argument("--drop-tq76", action="store_true",
                    help="exclude the FY1976 transition quarter instead of folding it in")
    ap.add_argument("--fy-offset", type=int, default=0,
                    help="calendar year = fiscal year + offset")
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--delta", choices=DELTAS, default=DEFAULT_DELTA,
                    help="cumulative: G is a stock, dG is the h-year window sum. "
                         "flow: dG = D_t - D_t-h, the pre-2026-08-10 default")
    args = ap.parse_args()

    failures = []
    years = list(range(args.first_year, args.last_year + 1))
    fold = not args.drop_tq76
    floor_label = "none" if args.floor == 0 else f"${args.floor:,}"
    align = "t" if args.fy_offset == 0 else f"t{args.fy_offset:+d}"
    print(f"\ncounty panel and contract shock   {years[0]}-{years[-1]}, "
          f"h={args.horizon}, delta {args.delta}, floor {floor_label}, "
          f"TQ76 {'folded into FY1976' if fold else 'dropped'}, "
          f"FY t -> calendar {align}")

    _, _, universe = dc.load_analysis_ccg_map()
    profile = dc.positive_employment_profile()
    n_ccg, n_year = len(universe), len(years)
    expected_rows = n_ccg * n_year
    print(f"    {n_ccg:,} CCGs x {n_year} years = {expected_rows:,} rows")

    # --- contracts -------------------------------------------------------------------
    print("\n[0] contracts collapsed to CCG x year")
    contracts = dc.build_contract_panel(universe, years, floor=args.floor, fold_tq76=fold)
    if args.fy_offset:
        contracts["year"] = contracts["year"] + args.fy_offset
        contracts = contracts[contracts["year"].between(years[0], years[-1])]
    neg = contracts["dollars_nominal"] < 0
    print(f"    {int(neg.sum()):,} negative CCG-years, "
          f"${contracts.loc[neg, 'dollars_real'].sum() / 1e9:,.2f}bn - deobligations, kept")

    # --- income ----------------------------------------------------------------------
    print("\n[1] BEA county personal income")
    inc_path = dc.PANEL_OUT_DIR / "income_ccg_year.parquet"
    if not inc_path.exists():
        raise FileNotFoundError(f"{inc_path} is missing; run 12_build_county_income.py")
    income = pd.read_parquet(inc_path)
    income = income[income["year"].between(years[0], years[-1])]
    print(f"    {len(income):,} CCG-years, "
          f"{income['ccg'].nunique():,} CCGs, "
          f"{income['year'].min()}-{income['year'].max()}")

    # --- employment ------------------------------------------------------------------
    print("\n[2] county employment")
    emp = pd.read_parquet(dc.EMP_OUT_DIR / "efsy_ccg_year.parquet",
                          columns=["ccg", "year", "emp"])
    emp = emp[emp["year"].between(years[0], years[-1]) & emp["ccg"].isin(universe)]
    print(f"    {len(emp):,} CCG-years, {emp['ccg'].nunique():,} CCGs")

    # --- the panel -------------------------------------------------------------------
    print("\n[3] the join, on one spine")
    spine = pd.MultiIndex.from_product(
        [np.sort(universe), years], names=["ccg", "year"]).to_frame(index=False)
    panel = spine.copy()
    n0 = len(panel)
    for name, df, cols in (
            ("contracts", contracts, ["dollars_nominal", "dollars_real", "n_actions"]),
            ("income", income, ["income_nominal", "income_real"]),
            ("employment", emp, ["emp"])):
        before = len(panel)
        panel = panel.merge(df[["ccg", "year"] + cols], on=["ccg", "year"], how="left")
        check(failures, len(panel) == before == n0, f"3.{name}",
              f"joining {name} leaves the row count fixed",
              f"{before:,} -> {len(panel):,}")

    panel = panel.sort_values(["ccg", "year"]).reset_index(drop=True)

    # --- the shock -------------------------------------------------------------------
    print("\n[4] the shock")
    h = args.horizon
    g = panel.groupby("ccg", sort=False)
    panel["dollars_real_lag"] = g["dollars_real"].shift(h)
    panel["income_real_lag"] = g["income_real"].shift(h)
    panel["emp_lag"] = g["emp"].shift(h)
    # Guard the shift: it is only a lag of h years if the years within a CCG are
    # contiguous and ascending, which the spine guarantees and which is asserted
    # rather than assumed.
    panel["year_lag"] = g["year"].shift(h)
    contiguous = ((panel["year"] - panel["year_lag"]).dropna() == h).all()
    check(failures, bool(contiguous), "4a", "the h-year lag is a true h-year lag",
          f"h={h}")

    # The numerator. Under `cumulative` it is the h-year window sum, computed as a
    # rolling sum rather than as a difference of cumulative sums: the two are equal
    # wherever both are defined (check 4b) and the rolling form is additionally
    # defined in the h-1 years where the cumulative one has no lag to subtract.
    panel["dollars_cum"] = g["dollars_real"].cumsum()
    panel["dollars_cum_lag"] = panel.groupby("ccg", sort=False)["dollars_cum"].shift(h)
    window = (g["dollars_real"].rolling(h).sum()
              .reset_index(level=0, drop=True).sort_index())
    if args.delta == "cumulative":
        panel["dG"] = window
    else:
        panel["dG"] = panel["dollars_real"] - panel["dollars_real_lag"]

    # 4b. The rolling sum IS the cumulative difference. A shift-by-one here is
    #     invisible in the output and fatal to the design, so it is asserted.
    #     Not to the cent: a CCG's cumulative stock reaches ~$1e11 by 2006 and the
    #     difference of two such float64 sums loses about 1e-5 dollars to rounding,
    #     while the rolling sum never accumulates past the window. The tolerance is
    #     therefore relative to the stock, and $1 absolute is the reporting unit.
    both = window.notna() & panel["dollars_cum_lag"].notna()
    cum_dev = float(np.abs(window[both]
                           - (panel.loc[both, "dollars_cum"]
                              - panel.loc[both, "dollars_cum_lag"])).max())
    stock = float(panel["dollars_cum"].abs().max())
    check(failures, cum_dev / max(stock, 1.0) < 1e-12, "4b",
          "the h-year window sum equals G_t - G_t-h",
          f"max dev ${cum_dev:.6f} of ${stock/1e9:,.0f}bn, {int(both.sum()):,} rows")

    # 4c. And it does not depend on where the cumulation starts, which is the whole
    #     reason a stock with no natural origin is usable here. Recomputed on a probe
    #     from a later first year; the rows the truncation cannot reach are excluded.
    #     The tolerance is RELATIVE, for the same reason 4b's is. pandas' rolling
    #     sum slides an accumulator rather than re-adding each window, so its
    #     rounding depends on how many earlier elements it processed - and the
    #     probe processes five fewer years than the panel does. On the 1975-2006
    #     panel the two agreed exactly; adding FY2007-2018 raises the largest
    #     window to about $55bn and one float64 ulp becomes visible, at 1.9e-06
    #     dollars, or 3e-17 relative, on 7,332 of 117,838 rows. That is the
    #     accumulator, not the design: what this check exists to catch is a
    #     window whose VALUE depends on the cumulation origin, which would show
    #     up at the scale of the dollars and not at the scale of the ulp.
    probe_first = args.first_year + 5
    sub = panel[panel["year"] >= probe_first].copy()
    sub_roll = (sub.groupby("ccg", sort=False)["dollars_real"].rolling(h).sum()
                .reset_index(level=0, drop=True).sort_index())
    comp = window.reindex(sub.index)
    ok_rows = sub_roll.notna() & comp.notna()
    start_dev = float(np.abs(sub_roll[ok_rows] - comp[ok_rows]).max())
    scale = float(np.abs(window).max())
    check(failures, start_dev / max(scale, 1.0) < 1e-12, "4c",
          "the window sum is invariant to the cumulation start year",
          f"max dev ${start_dev:.6f} of ${scale/1e9:,.0f}bn, probe from {probe_first}")

    with np.errstate(divide="ignore", invalid="ignore"):
        panel["shock"] = panel["dG"] / panel["income_real_lag"]
    panel = panel.drop(columns=["year_lag"])

    # 4d. The seam. dG for year t sums D over {t-h+1..t}, so it mixes DCADS with
    #     the post-seam block exactly when that window straddles FY2007 - and
    #     nowhere else, which is what 4b and 4c above establish. Those years are
    #     masked rather than dropped so the spine stays rectangular.
    seam_years = [y for y in range(dc.SEAM_YEAR, dc.SEAM_YEAR + h - 1) if y in years]
    n_masked = 0
    if seam_years:
        mask = panel["year"].isin(seam_years)
        n_masked = int(mask.sum())
        panel.loc[mask, ["dG", "shock"]] = np.nan
        print(f"    seam FY{dc.SEAM_YEAR}: masked {n_masked:,} rows in "
              f"{', '.join(f'FY{y}' for y in seam_years)} - their h={h} window "
              f"straddles the source change")
    else:
        print(f"    seam FY{dc.SEAM_YEAR}: no year masked at h={h}; the window "
              f"never straddles the source change")
    check(failures, n_masked == len(seam_years) * n_ccg, "4d",
          "the seam mask covers exactly the straddling years",
          f"{n_masked:,} rows, expected {len(seam_years) * n_ccg:,}")

    est = panel["shock"].notna()
    v = panel.loc[est, "shock"].to_numpy()
    q = np.percentile(v, [1, 50, 99])
    neg_shock = int((v < 0).sum())
    print(f"    {int(est.sum()):,} CCG-years carry a shock "
          f"({n_ccg} CCGs x {n_year - h - len(seam_years)} years minus the "
          f"declared income gaps)")
    print(f"    sd {v.std():.5f}   p1 {q[0]:+.5f}   median {q[1]:+.5f}   "
          f"p99 {q[2]:+.5f}   min {v.min():+.4f}   max {v.max():+.4f}")
    print(f"    {neg_shock:,} negative shocks "
          f"({100 * neg_shock / len(v):.2f}% of the estimation sample)"
          + ("  - net deobligations across the whole window"
             if args.delta == "cumulative" else "  - flow differences, mostly award timing"))
    print("    Not winsorized. A.3.2 is about growth rates; the tails here are small "
          "counties\n    landing one large contract, and the answer to a heavy-tailed "
          "regressor is the\n    specification rather than a cap we exist to criticise.")

    # --- the checks ------------------------------------------------------------------
    print("\n[5] the checks plan section 1.5 requires")

    # 1. the row count and the row order
    ok = len(panel) == expected_rows
    check(failures, ok, 1, "row count is analysis CCGs x n_years after every join",
          f"{len(panel):,} of {expected_rows:,}")
    ccg_seen = panel["ccg"].to_numpy()
    expect_ccg = np.repeat(np.sort(universe), n_year)
    expect_year = np.tile(np.array(years), n_ccg)
    check(failures, np.array_equal(ccg_seen, expect_ccg)
          and np.array_equal(panel["year"].to_numpy(), expect_year),
          "1b", "row ORDER is (ccg, year) ascending and complete",
          "byte-identical to the spine")
    check(failures, len(contracts) == expected_rows, "1c",
          "the contract file sits on the same spine",
          f"{len(contracts):,} of {expected_rows:,}")

    # 2. dollar conservation, against a total computed without the collapse
    ref = reference_year_totals(years, args.floor, fold, universe)
    # Every year must have a reference. Before the era split this check iterated
    # over whatever reference_year_totals happened to return, so a year with no
    # source file was silently not compared and the check still read PASS.
    check(failures, set(ref) == set(years), "2a",
          "every year has an independent reference total",
          f"{len(ref)} of {len(years)} years")
    ours = panel.groupby("year")["dollars_nominal"].sum()
    devs = {y: abs(float(ours.get(y, 0.0)) - ref[y]) for y in ref}
    worst_y = max(devs, key=devs.get)
    check(failures, max(devs.values()) == 0.0, 2,
          "contract dollars conserved by the collapse, every year",
          f"max dev ${devs[worst_y]:,.2f} (FY{worst_y})")

    # 3. identical CCG-YEAR sets, not pooled sets. README 1.6.7 is the record of
    #    what pooling hides: DC passed a pooled test while missing 28 of its years.
    key = lambda d: set(map(tuple, d[["ccg", "year"]].to_numpy()))  # noqa: E731
    all_keys = key(spine)
    check(failures, key(contracts) == all_keys, "3a",
          "contract CCG-years == the universe", f"{len(key(contracts)):,}")
    check(failures, key(emp) == all_keys, "3b",
          "employment CCG-years == the universe", f"{len(key(emp)):,}")
    missing_income = all_keys - key(income)
    gap_keys = {(c, y) for c in INCOME_GAP_CCGS if c in set(universe) for y in years}
    check(failures, missing_income == gap_keys, "3c",
          "income covers everything but the declared gaps",
          f"{len(missing_income):,} missing, {len(gap_keys):,} declared")

    # 4. cells sum to the county, both rungs, every county-year
    for rung in ("naics2", "naics4_alloc"):
        worst = 0.0
        for y in years:
            cells = pd.read_parquet(dc.EMP_OUT_DIR / rung / f"y{y}.parquet",
                                    columns=["ccg", "emp"])
            cells = cells[cells["ccg"].isin(universe)]
            s = cells.groupby("ccg", sort=True)["emp"].sum()
            t = panel[panel["year"] == y].set_index("ccg")["emp"].reindex(s.index)
            worst = max(worst, float(np.abs(s.to_numpy() - t.to_numpy()).max()))
        check(failures, worst < 1e-6, f"4.{rung}",
              f"{rung} cells sum to the county total", f"max dev {worst:.3e}")

    # 5. finiteness on the estimation sample
    check(failures, bool(np.isfinite(v).all()), 5,
          "the shock is finite wherever it is defined",
          f"{int(est.sum()):,} CCG-years")

    # --- write -----------------------------------------------------------------------
    # NOTHING is written until every check has passed. This used to write first and
    # evaluate `failures` afterwards, so a FAILING run replaced the good panel with
    # the bad one and there was no copy left to compare against.
    if failures:
        print(f"\nFAIL: {len(failures)} check(s) failed - nothing written",
              file=sys.stderr)
        for name, why in failures:
            print(f"  {name}: {why}", file=sys.stderr)
        return 1

    print("\n[6] writing")
    dc.PANEL_OUT_DIR.mkdir(parents=True, exist_ok=True)
    contracts.to_parquet(dc.PANEL_OUT_DIR / "contracts_ccg_year.parquet", index=False)
    keep = ["ccg", "year", "emp", "emp_lag", "dollars_nominal", "dollars_real",
            "dollars_real_lag", "dollars_cum", "dollars_cum_lag", "dG", "n_actions",
            "income_nominal", "income_real", "income_real_lag", "shock"]
    panel[keep].to_parquet(dc.PANEL_OUT_DIR / "county_panel.parquet", index=False)
    print(f"    {len(panel):,} rows -> {dc.PANEL_OUT_DIR / 'county_panel.parquet'}")

    rep = panel.groupby("year").agg(
        n_ccg=("ccg", "size"),
        contracts_bn=("dollars_real", lambda s: s.sum() / 1e9),
        income_bn=("income_real", lambda s: s.sum() / 1e9),
        emp_m=("emp", lambda s: s.sum() / 1e6),
        n_shock=("shock", "count"),
        shock_sd=("shock", "std")).reset_index()
    rep["source"] = np.where(rep["year"] < dc.SEAM_YEAR, "DCADS", "USAspending")
    dc.LOG_DIR.mkdir(parents=True, exist_ok=True)
    rep.to_csv(dc.LOG_DIR / "county_panel_report.csv", index=False)
    profile.to_csv(dc.LOG_DIR / "positive_employment_universe.csv", index=False)
    print(f"    per-year report -> {dc.LOG_DIR / 'county_panel_report.csv'}")
    print(f"    universe audit -> {dc.LOG_DIR / 'positive_employment_universe.csv'}")

    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
