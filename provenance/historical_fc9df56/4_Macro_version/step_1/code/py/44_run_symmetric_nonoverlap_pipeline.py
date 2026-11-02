#!/usr/bin/env python3
"""Run the symmetric county-contract first-stage pipeline.

The default certified-cache mode verifies all raw-derived inputs and reruns only
the symmetric panel, Stata regressions, and independent verification. The
--from-raw mode executes the authoritative DCADS, USAspending, county,
employment, income, and source-harmonization producers before those steps.

Output:
    batch_logs/macro_version/44_run_symmetric_nonoverlap_pipeline.log
    batch_logs/macro_version/symmetric_nonoverlap_run_receipt.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
CODE = ROOT / "4_Macro_version" / "step_1" / "code"
PY = CODE / "py"
STATA = CODE / "stata"
sys.path.insert(0, str(PY))
import dcads_common as dc  # noqa: E402

LOG_DIR = ROOT / "batch_logs" / "macro_version"
LOG = LOG_DIR / "44_run_symmetric_nonoverlap_pipeline.log"
RECEIPT = LOG_DIR / "symmetric_nonoverlap_run_receipt.json"
DCADS_OUT = dc.OUT_DIR
PANEL_OUT = dc.PANEL_OUT_DIR
USA_OUT = dc.USASPENDING_OUT_DIR

RAW_PRODUCERS = (
    "01_build_dcads_manifest.py",
    "02_download_dcads.py",
    "03_verify_dcads.py",
    "04_parse_dcads.py",
    "05_parse_dcads_delimited.py",
    "06_verify_dcads_parse.py",
    "07_add_county_fips.py",
    "08_verify_county_fips.py",
    "09_pallante_test.py",
    "10_build_county_universe.py",
    "11_build_employment_cells.py",
    "12_build_county_income.py",
    "13_build_county_panel.py",
    "27_build_cz_state_assignment.py",
    "29_build_usaspending_manifest.py",
    "30_download_usaspending.py",
    "31_parse_usaspending.py",
    "32_verify_usaspending.py",
    "33_harmonize_contract_sources.py",
)


class Tee:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = path.open("w", encoding="utf-8")
        self.stdout, self.stderr = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = self

    def write(self, value):
        self.stdout.write(value)
        self.file.write(value)
        self.stdout.flush()
        self.file.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        sys.stdout, sys.stderr = self.stdout, self.stderr
        self.file.close()


def utc_now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def run(command: list[str], label: str, receipt_rows: list[dict]):
    started = utc_now()
    print(f"\n[{label}] {' '.join(command)}")
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    for line in process.stdout:
        print(line, end="")
    return_code = process.wait()
    receipt_rows.append(
        {
            "label": label,
            "command": command,
            "started_utc": started,
            "finished_utc": utc_now(),
            "return_code": return_code,
        }
    )
    if return_code != 0:
        raise RuntimeError(f"{label} failed with return code {return_code}")


def certified_cache_check():
    required = [
        DCADS_OUT / "parse_verification_report.csv",
        DCADS_OUT / "county_fips_verification_report.csv",
        PANEL_OUT / "county_panel.parquet",
        PANEL_OUT / "contracts_harmonized_ccg_year.parquet",
        dc.USASPENDING_VERIFICATION_REPORT,
        dc.HARMONIZED_REPORT,
        dc.POST_GRID_LABELED,
        CODE / "reference" / "county_ccg_crosswalk.csv",
        CODE / "reference" / "cz_state_1969_population.csv",
    ]
    required.extend(DCADS_OUT / f"fy{year}.parquet" for year in range(1966, 2007))
    required.extend(USA_OUT / f"fy{year}.parquet" for year in range(2007, 2019))
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("certified-cache inputs missing: " + ", ".join(missing))

    parse = pd.read_csv(DCADS_OUT / "parse_verification_report.csv")
    if len(parse) != 42 or not parse["status"].eq("ok").all() or not parse["fails"].isna().all():
        raise AssertionError("DCADS parse verification cache is not certified")
    fips = pd.read_csv(DCADS_OUT / "county_fips_verification_report.csv")
    if len(fips) != 42 or int(
        fips[["domain_violations", "unresolved", "wrong_state"]].sum().sum()
    ) != 0:
        raise AssertionError("DCADS county verification cache is not certified")
    usa = pd.read_csv(dc.USASPENDING_VERIFICATION_REPORT)
    if len(usa) != 12 or not usa["status"].eq("PASS").all():
        raise AssertionError("USAspending verification cache is not certified")
    harmonized = pd.read_csv(dc.HARMONIZED_REPORT)
    if len(harmonized) != 53 or not harmonized["ccgs"].eq(3_101).all():
        raise AssertionError("harmonized county contract cache is not certified")
    print(
        "Certified-cache gate passed: 42 DCADS files, 12 USAspending years, "
        "53 harmonized county-year source blocks"
    )


def parser():
    argument_parser = argparse.ArgumentParser(description=__doc__)
    mode = argument_parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--from-raw",
        action="store_true",
        help="execute authoritative raw-data producers before the symmetric pipeline",
    )
    mode.add_argument(
        "--certified-cache",
        action="store_true",
        help="verify and reuse certified raw-derived caches; this is the default",
    )
    return argument_parser


def main():
    args = parser().parse_args()
    mode = "from_raw" if args.from_raw else "certified_cache"
    uvx = shutil.which("uvx")
    if uvx is None:
        raise FileNotFoundError("uvx is required for the Stata-MCP runner")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    tee = Tee(LOG)
    receipt_rows = []
    receipt = {
        "pipeline": "symmetric_nonoverlap_first_stage_v2",
        "mode": mode,
        "started_utc": utc_now(),
        "project_root": str(ROOT),
        "dropbox_project_root": str(dc.DROPBOX_PROJECT_ROOT),
        "steps": receipt_rows,
    }
    try:
        print(f"Pipeline mode: {mode}")
        if args.from_raw:
            for producer in RAW_PRODUCERS:
                run(
                    [sys.executable, str(PY / producer)],
                    f"raw:{producer.removesuffix('.py')}",
                    receipt_rows,
                )
        certified_cache_check()
        run(
            [sys.executable, str(PY / "41_build_symmetric_nonoverlap_panel.py")],
            "build:symmetric_panel",
            receipt_rows,
        )
        run(
            [
                uvx,
                "--with",
                "mcp<2",
                "stata-mcp==1.21.3",
                "tool",
                "do",
                str(STATA / "42_symmetric_nonoverlap_first_stages.do"),
                "--log-file-name",
                "symmetric_nonoverlap_pipeline_mcp",
                "--is-read-log",
                "true",
                "--is-replace-log",
                "true",
                "--enable-smcl",
                "false",
                "--timeout",
                "600",
            ],
            "estimate:stata_mcp",
            receipt_rows,
        )
        run(
            [sys.executable, str(PY / "43_verify_symmetric_nonoverlap_pipeline.py")],
            "verify:symmetric_pipeline",
            receipt_rows,
        )
        receipt["status"] = "PASS"
        return 0
    except Exception as error:
        receipt["status"] = "FAIL"
        receipt["error"] = str(error)
        raise
    finally:
        receipt["finished_utc"] = utc_now()
        RECEIPT.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        print(f"Run receipt -> {RECEIPT}")
        tee.close()


if __name__ == "__main__":
    raise SystemExit(main())
