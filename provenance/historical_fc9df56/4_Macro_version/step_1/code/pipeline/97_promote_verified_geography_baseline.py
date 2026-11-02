"""Promote verified contract geography to the locked active baseline.

This deliberate recapture runs the registered golden constructor, rebuilds
stages 3 and 4, estimates the 30-cell aggregation baseline from the persisted
analysis panel, refreshes the committed baseline hash, and runs stage 5.
The reported-source archive must pass its SHA256 integrity gate first.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


PIPELINE = Path(__file__).resolve().parent
ROOT = PIPELINE.parents[2]
sys.path.insert(0, str(PIPELINE.parent / "py"))
import dcads_common as dc  # noqa: E402


PYTHON = ROOT / ".venv" / "bin" / "python"
PIPE = dc.MACRO_DATA_ROOT / "temp" / "pipeline"
GOLD = PIPE / "_golden"
FREEZE = PIPE / "freezes" / "reported_source_geography_v1"
REPO_GOLD = PIPELINE / "golden"
BASELINE_COLUMNS = [
    "horizon", "level", "weight", "n_units", "n_county", "n_agg",
    "straddle", "beta_b", "delta", "beta_a_aligned", "beta_a_own",
    "jensen", "construction_gap", "t_agg_own", "t_county", "t_agg",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_freeze() -> None:
    sums = FREEZE / "SHA256SUMS"
    if not sums.exists():
        raise SystemExit(f"FAIL: reported-source archive is absent: {FREEZE}")
    for line in sums.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        path = FREEZE / relative
        if not path.exists() or sha256(path) != expected:
            raise SystemExit(f"FAIL: reported-source archive failed at {relative}")
    print(f"reported-source archive PASS -> {FREEZE}")


def run(script: str) -> None:
    subprocess.run([str(PYTHON), str(PIPELINE / script)], cwd=ROOT, check=True)


def load_tieout_module():
    path = PIPELINE / "05_baseline_tieout.py"
    spec = importlib.util.spec_from_file_location("baseline_tieout_promote", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def refresh_repo_hashes() -> None:
    assets = sorted(path for path in REPO_GOLD.iterdir()
                    if path.is_file() and path.name != "SHA256SUMS")
    lines = [f"{sha256(path)}  {path.name}" for path in assets]
    (REPO_GOLD / "SHA256SUMS").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")


def write_active_manifest() -> None:
    relative_assets = [
        "contracts_category_ccg_year.parquet",
        "contracts_ccg_year.parquet",
        "category_shocks_year.parquet",
        "composition_instrument_ccg_year.parquet",
        "analysis_panel.parquet",
        "_golden/county_h2.parquet",
        "_golden/county_h3.parquet",
        "_golden/county_h4.parquet",
        "_golden/frozen_aggregation_baseline.csv",
    ]
    assets = {relative: sha256(PIPE / relative) for relative in relative_assets}
    manifest = {
        "baseline_id": "verified_production_geography_v1",
        "geography": "verified production geography",
        "reported_source_archive": (
            "freezes/reported_source_geography_v1"
        ),
        "assets_sha256": assets,
        "promotion_producer": "97_promote_verified_geography_baseline.py",
        "standing_gate": "05_baseline_tieout.py",
    }
    payload = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    (GOLD / "active_baseline.json").write_text(payload, encoding="utf-8")
    (REPO_GOLD / "active_baseline.json").write_text(payload, encoding="utf-8")


def main() -> None:
    check_freeze()
    run("00_capture_golden.py")
    run("03_composition_instrument.py")
    run("04_analysis_panel.py")

    tieout = load_tieout_module()
    panel = pd.read_parquet(PIPE / "analysis_panel.parquet")
    baseline = tieout.estimate_baseline(panel)[BASELINE_COLUMNS]
    if len(baseline) != 30:
        raise SystemExit(f"FAIL: aggregation baseline has {len(baseline)} rows")
    for destination in (
        GOLD / "frozen_aggregation_baseline.csv",
        REPO_GOLD / "frozen_aggregation_baseline.csv",
    ):
        baseline.to_csv(destination, index=False, float_format="%.15g")
    write_active_manifest()
    refresh_repo_hashes()

    run("05_baseline_tieout.py")
    print("verified production geography is the active locked baseline")


if __name__ == "__main__":
    main()
