#!/usr/bin/env python3
"""Run the post-1998 anchor-year sweep.

The default certified-cache mode verifies the raw-derived inputs and reruns
only the sweep panel, the Stata first stages, the independent verification and
the export. The --from-raw mode executes the authoritative DCADS, USAspending,
county, employment, income and source-harmonization producers first. Both reuse
44's gate and step runner rather than restating them.

44's own pipeline is untouched by this one: 41 through 45 and their frozen
reference CSV are read, never written. Rerunning 44 after this must still pass.

Output:
    batch_logs/macro_version/50_run_post_anchor_sweep.log
    batch_logs/macro_version/post_anchor_sweep_run_receipt.json
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
LOG = LOG_DIR / "50_run_post_anchor_sweep.log"
RECEIPT = LOG_DIR / "post_anchor_sweep_run_receipt.json"
SYMMETRIC_RUNNER = PY / "44_run_symmetric_nonoverlap_pipeline.py"
# 48's pin reads this. It is git-tracked, unlike the live CSV 42 rewrites on
# every 44 run, which is why the pin targets it.
V1_BASELINE = CODE / "reference" / "symmetric_nonoverlap_first_stage_v1.csv"


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
        help="execute authoritative raw-data producers before the sweep",
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
        "pipeline": "post_anchor_sweep_v1",
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
        # 44's gate does not cover the frozen reference, because 44 only needs
        # it inside 43. Here it gates the run: without it 48 cannot pin.
        if not V1_BASELINE.exists():
            raise FileNotFoundError(f"frozen v1 reference missing: {V1_BASELINE}")
        print(f"Frozen v1 reference present for the anchor-1998 pin: {V1_BASELINE.name}")

        runner.run(
            [sys.executable, str(PY / "46_build_post_anchor_sweep_panel.py")],
            "build:anchor_sweep_panel",
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
                str(STATA / "47_post_anchor_sweep_first_stages.do"),
                "--log-file-name",
                "post_anchor_sweep_mcp",
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
            [sys.executable, str(PY / "48_verify_post_anchor_sweep.py")],
            "verify:anchor_sweep",
            receipt_rows,
        )
        runner.run(
            [sys.executable, str(PY / "49_export_post_anchor_sweep.py")],
            "export:anchor_sweep",
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
