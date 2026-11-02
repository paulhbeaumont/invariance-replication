#!/usr/bin/env python3
"""Build one county-year contract source from raw DCADS and USAspending actions.

DCADS supplies FY1966-2006 and USAspending supplies FY2007-2018. Both branches
use signed action obligations, domestic mapped places of performance, the same
constant-county crosswalk, and the same CPI deflator. The final panel is
rectangular on 3,101 constant-county groups for all 53 fiscal years.

Outputs:
    dc.PANEL_OUT_DIR/contracts_dcads_ccg_year.parquet
    dc.PANEL_OUT_DIR/contracts_usaspending_ccg_year.parquet
    dc.PANEL_OUT_DIR/contracts_harmonized_ccg_year.parquet
    dc.PANEL_OUT_DIR/contracts_harmonized_report.csv
"""

import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dcads_common as dc  # noqa: E402

FIRST_YEAR = 1966
LAST_DCADS_YEAR = dc.SEAM_YEAR - 1
LAST_YEAR = 2018
OUTPUT_COLUMNS = (
    "ccg",
    "year",
    "source",
    "dollars_nominal",
    "dollars_real",
    "n_actions",
)
REPORT_FIELDS = (
    "fiscal_year",
    "source",
    "ccgs",
    "nonzero_ccgs",
    "actions",
    "negative_ccg_years",
    "dollars_nominal",
    "dollars_real",
)


def rectangularize(frame, universe, years, source):
    spine = pd.MultiIndex.from_product(
        [np.sort(np.asarray(universe, dtype=np.int64)), list(years)], names=["ccg", "year"]
    )
    out = frame.set_index(["ccg", "year"]).reindex(spine).reset_index()
    out["dollars_nominal"] = out["dollars_nominal"].fillna(0.0)
    out["n_actions"] = out["n_actions"].fillna(0).astype("Int64")
    out["dollars_real"] = out["dollars_nominal"] * out["year"].map(dc.deflate)
    out.insert(2, "source", source)
    return out[list(OUTPUT_COLUMNS)]


def verification_totals():
    path = dc.USASPENDING_VERIFICATION_REPORT
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; run 32_verify_usaspending.py")
    frame = pd.read_csv(path)
    failed = frame[frame["status"] != "PASS"]
    if not failed.empty:
        raise AssertionError(f"USAspending verification has failed years {failed.fiscal_year.tolist()}")
    return frame.set_index("fiscal_year")["county_dollars"].to_dict()


def write_report(panel):
    report = panel.groupby(["year", "source"], as_index=False).agg(
        ccgs=("ccg", "size"),
        nonzero_ccgs=("dollars_nominal", lambda values: int((values != 0).sum())),
        actions=("n_actions", "sum"),
        negative_ccg_years=("dollars_nominal", lambda values: int((values < 0).sum())),
        dollars_nominal=("dollars_nominal", "sum"),
        dollars_real=("dollars_real", "sum"),
    )
    report = report.rename(columns={"year": "fiscal_year"})
    path = dc.HARMONIZED_REPORT
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REPORT_FIELDS)
        writer.writeheader()
        writer.writerows(report[list(REPORT_FIELDS)].to_dict("records"))
    return path, report


def check(failures, condition, label, detail):
    print(f"    {label:<48s} {detail:<36s} {'PASS' if condition else 'FAIL'}")
    if not condition:
        failures.append((label, detail))


def main():
    failures = []
    _, _, universe = dc.load_ccg_map()
    dcads_years = list(range(FIRST_YEAR, LAST_DCADS_YEAR + 1))
    usa_years = list(range(dc.SEAM_YEAR, LAST_YEAR + 1))
    expected_years = list(range(FIRST_YEAR, LAST_YEAR + 1))

    print(f"Harmonized raw contract actions, FY{FIRST_YEAR}-{LAST_YEAR}")
    print("  [1] DCADS")
    dcads = dc.build_contract_panel(
        universe, dcads_years, floor=0, fold_tq76=True, verbose=True
    )
    dcads.insert(2, "source", "DCADS")
    dcads = dcads[list(OUTPUT_COLUMNS)]

    print("  [2] USAspending")
    usa_sparse = dc.collapse_usaspending_contracts(usa_years, floor=0)
    usa = rectangularize(usa_sparse, universe, usa_years, "USAspending")
    print(
        f"    {int(usa['n_actions'].sum()):,} award actions, "
        f"${usa['dollars_nominal'].sum() / 1e9:,.3f}bn nominal"
    )

    print("  [3] source checks")
    verified = verification_totals()
    usa_totals = usa.groupby("year")["dollars_nominal"].sum()
    missing_verified = sorted(set(usa_years) - set(verified))
    check(failures, not missing_verified, "verification coverage", f"missing {missing_verified}")
    deviations = {
        year: abs(float(usa_totals[year]) - float(verified[year]))
        for year in usa_years
        if year in verified
    }
    max_dev = max(deviations.values(), default=float("inf"))
    check(
        failures,
        max_dev <= 0.01,
        "USAspending collapse identity",
        f"maximum annual deviation ${max_dev:.6f}",
    )
    check(
        failures,
        not dcads.duplicated(["ccg", "year"]).any(),
        "DCADS keys",
        f"{len(dcads):,} unique CCG-years",
    )
    check(
        failures,
        not usa.duplicated(["ccg", "year"]).any(),
        "USAspending keys",
        f"{len(usa):,} unique CCG-years",
    )

    print("  [4] harmonized panel")
    panel = pd.concat([dcads, usa], ignore_index=True)
    expected_rows = len(universe) * len(expected_years)
    check(
        failures,
        len(panel) == expected_rows,
        "rectangular row count",
        f"{len(universe):,} CCGs x {len(expected_years)} years = {expected_rows:,}",
    )
    check(
        failures,
        sorted(panel["year"].unique()) == expected_years,
        "fiscal-year coverage",
        f"FY{panel.year.min()}-{panel.year.max()}",
    )
    check(
        failures,
        not panel.duplicated(["ccg", "year"]).any(),
        "harmonized keys",
        "one row per CCG-year",
    )
    check(
        failures,
        not panel[list(OUTPUT_COLUMNS)].isna().any().any(),
        "complete values",
        "zero missing output cells",
    )
    source_by_year = panel.groupby("year")["source"].nunique()
    check(
        failures,
        bool(source_by_year.eq(1).all()),
        "one source per fiscal year",
        f"{int(source_by_year.max())} maximum",
    )

    if failures:
        print(f"FAIL: {len(failures)} check(s) failed; nothing written", file=sys.stderr)
        for label, detail in failures:
            print(f"  {label}: {detail}", file=sys.stderr)
        return 1

    dc.PANEL_OUT_DIR.mkdir(parents=True, exist_ok=True)
    dcads_path = dc.PANEL_OUT_DIR / "contracts_dcads_ccg_year.parquet"
    usa_path = dc.PANEL_OUT_DIR / "contracts_usaspending_ccg_year.parquet"
    panel_path = dc.PANEL_OUT_DIR / "contracts_harmonized_ccg_year.parquet"
    dcads.to_parquet(dcads_path, index=False)
    usa.to_parquet(usa_path, index=False)
    panel.to_parquet(panel_path, index=False)
    report_path, _ = write_report(panel)
    print(f"    {len(dcads):,} rows -> {dcads_path}")
    print(f"    {len(usa):,} rows -> {usa_path}")
    print(f"    {len(panel):,} rows -> {panel_path}")
    print(f"    53 rows -> {report_path}")
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
