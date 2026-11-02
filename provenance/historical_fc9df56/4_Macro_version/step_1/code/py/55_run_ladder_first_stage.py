#!/usr/bin/env python3
"""Run the aggregation-ladder first stages, end to end.

The default certified-cache mode verifies the raw-derived inputs and reruns the
ladder panel, the Stata first stages, the verification and the export. The
--from-raw mode executes the authoritative DCADS, USAspending, county,
employment, income and source-harmonization producers first. Both reuse 44's
gate and step runner rather than restating them, as 50 does.

44's and 50's pipelines are untouched by this one: 41 through 50 and the frozen
reference CSV are read, never written. Rerunning either after this must pass.

ONE GATE THIS PIPELINE ADDS

The county rung of the ladder IS the candidate first stage, so 53 pins it to the
certified 252-row candidate result. That target lives under a gitignored
data/temp, which means a fresh checkout has the certified CSV but not the panel
behind it. Rather than fail three steps later inside a pin, this file rebuilds
the candidate panel first whenever its Stata inputs are absent.

Output:
    batch_logs/macro_version/55_run_ladder_first_stage.log
    batch_logs/macro_version/ladder_first_stage_run_receipt.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
CODE = ROOT / "4_Macro_version" / "step_1" / "code"
PY = CODE / "py"
STATA = CODE / "stata"
sys.path.insert(0, str(PY))
import dcads_common as dc  # noqa: E402

LOG_DIR = ROOT / "batch_logs" / "macro_version"
LOG = LOG_DIR / "55_run_ladder_first_stage.log"
RECEIPT = LOG_DIR / "ladder_first_stage_run_receipt.json"
SYMMETRIC_RUNNER = PY / "44_run_symmetric_nonoverlap_pipeline.py"
SYMMETRIC_PANEL = dc.SYMMETRIC_FIRST_STAGE_DIR / "panel.parquet"
CANDIDATE = ROOT / "Candidate for first stage"
CANDIDATE_BUILDER = CANDIDATE / "code" / "01_build_candidate_panel.py"
CANDIDATE_INPUT = CANDIDATE / "data" / "temp" / "pre_h4.dta"
CANDIDATE_RESULTS = CANDIDATE / "output" / "candidate_first_stage_all.csv"


def load_numbered(path: Path):
    spec = importlib.util.spec_from_file_location("symmetric_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parser():
    argument_parser = argparse.ArgumentParser(description=__doc__)
    mode = argument_parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--from-raw",
        action="store_true",
        help="execute authoritative raw-data producers before the ladder",
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

    runner = load_numbered(SYMMETRIC_RUNNER)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    tee = runner.Tee(LOG)
    receipt_rows = []
    receipt = {
        "pipeline": "ladder_first_stage_v1",
        "mode": mode,
        "started_utc": runner.utc_now(),
        "project_root": str(ROOT),
        "dropbox_project_root": str(dc.DROPBOX_PROJECT_ROOT),
        "steps": receipt_rows,
    }
    try:
        print(f"Pipeline mode: {mode}")
        if args.from_raw:
            for producer in runner.RAW_PRODUCERS:
                runner.run(
                    [sys.executable, str(PY / producer)],
                    f"raw:{producer.removesuffix('.py')}",
                    receipt_rows,
                )
        runner.certified_cache_check()

        if not SYMMETRIC_PANEL.exists():
            raise FileNotFoundError(
                f"41's symmetric panel is missing: {SYMMETRIC_PANEL}\n"
                "Run 44_run_symmetric_nonoverlap_pipeline.py first."
            )
        if not CANDIDATE_RESULTS.exists():
            raise FileNotFoundError(
                f"certified candidate result is missing: {CANDIDATE_RESULTS}"
            )
        # The pin target is git-tracked; the panel behind it is not. Rebuild it
        # rather than let 51 fail inside its own pin with a less obvious message.
        if not CANDIDATE_INPUT.exists():
            print(f"Candidate Stata inputs absent; rebuilding from {CANDIDATE_BUILDER}")
            runner.run(
                [sys.executable, str(CANDIDATE_BUILDER)],
                "build:candidate_panel",
                receipt_rows,
            )
        print(f"Certified candidate result present for the pin: {CANDIDATE_RESULTS.name}")

        runner.run(
            [sys.executable, str(PY / "51_build_ladder_first_stage_panel.py")],
            "build:ladder_panel",
            receipt_rows,
        )
        runner.run(
            [
                uvx,
                "--with",
                "mcp<2",
                "stata-mcp==1.21.3",
                "tool",
                "do",
                str(STATA / "52_ladder_first_stages.do"),
                "--log-file-name",
                "ladder_first_stage_mcp",
                "--is-read-log",
                "true",
                "--is-replace-log",
                "true",
                "--enable-smcl",
                "false",
                "--timeout",
                "1800",
            ],
            "estimate:stata_mcp",
            receipt_rows,
        )
        runner.run(
            [sys.executable, str(PY / "53_verify_ladder_first_stage.py")],
            "verify:ladder",
            receipt_rows,
        )
        runner.run(
            [sys.executable, str(PY / "54_export_ladder_first_stage.py")],
            "export:ladder",
            receipt_rows,
        )
        receipt["status"] = "PASS"
        return 0
    except Exception as error:
        receipt["status"] = "FAIL"
        receipt["error"] = str(error)
        raise
    finally:
        receipt["finished_utc"] = runner.utc_now()
        RECEIPT.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        print(f"Run receipt -> {RECEIPT}")
        tee.close()


if __name__ == "__main__":
    raise SystemExit(main())
