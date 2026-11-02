"""Freeze the reported-source geography design before baseline promotion.

The archive contains the exact persisted stage-3 and stage-4 inputs used by
the reported-source geography design, the five registered key-table results,
the matching aggregation baseline, the reconstruction producer, and the two
guarded routing tables.  SHA256SUMS is the integrity gate for every archived
asset.  An existing archive is verified and never overwritten.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


PIPELINE = Path(__file__).resolve().parent
ROOT = PIPELINE.parents[2]
sys.path.insert(0, str(PIPELINE.parent / "py"))
import dcads_common as dc  # noqa: E402


PIPE = dc.MACRO_DATA_ROOT / "temp" / "pipeline"
FREEZE = PIPE / "freezes" / "reported_source_geography_v1"
KEY_RESULTS = (
    dc.MACRO_RESULTS_ROOT / "instrument_normalization"
    / "old_design_geography_key_tables"
)
SCRATCH = ROOT / "data" / "temp" / "old_design_geography_key_tables"


ASSETS = {
    "pipeline_outputs/category_shocks_year.parquet":
        PIPE / "category_shocks_year.parquet",
    "pipeline_outputs/composition_instrument_ccg_year.parquet":
        PIPE / "composition_instrument_ccg_year.parquet",
    "pipeline_outputs/analysis_panel.parquet": PIPE / "analysis_panel.parquet",
    "key_table_results/estimates_long.csv": KEY_RESULTS / "estimates_long.csv",
    "key_table_results/comparison_wide.csv": KEY_RESULTS / "comparison_wide.csv",
    "key_table_results/table_summary.csv": KEY_RESULTS / "table_summary.csv",
    "key_table_results/aggregation_baseline.csv":
        SCRATCH / "reported_source_geography" / "aggregation_baseline.csv",
    "reconstruction/95_old_design_geography_key_tables.py":
        PIPELINE / "95_old_design_geography_key_tables.py",
    "reconstruction/dcads_city_county_overrides.csv":
        PIPELINE.parent / "reference" / "dcads_city_county_overrides.csv",
    "reconstruction/dcads_verified_city_county_overrides.csv":
        PIPELINE.parent / "reference" / "dcads_verified_city_county_overrides.csv",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_sums(path: Path) -> dict[str, str]:
    sums = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        value, relative = line.split("  ", 1)
        sums[relative] = value
    return sums


def verify_archive() -> None:
    sums_path = FREEZE / "SHA256SUMS"
    manifest_path = FREEZE / "manifest.json"
    if not sums_path.exists() or not manifest_path.exists():
        raise SystemExit(f"FAIL: incomplete existing archive at {FREEZE}")
    sums = read_sums(sums_path)
    expected = set(ASSETS)
    if set(sums) != expected:
        raise SystemExit("FAIL: archive asset list differs from the registered set")
    for relative, expected_hash in sums.items():
        path = FREEZE / relative
        if not path.exists() or sha256(path) != expected_hash:
            raise SystemExit(f"FAIL: archive integrity check failed for {relative}")
    print(f"archive PASS: {len(sums)} assets -> {FREEZE}")


def main() -> None:
    if FREEZE.exists():
        verify_archive()
        return

    missing = [str(path) for path in ASSETS.values() if not path.exists()]
    if missing:
        raise SystemExit(f"FAIL: registered source assets are missing: {missing}")

    stage = FREEZE.with_name(FREEZE.name + ".building")
    if stage.exists():
        raise SystemExit(f"FAIL: incomplete staging directory exists: {stage}")
    stage.mkdir(parents=True)
    try:
        sums = {}
        for relative, source in ASSETS.items():
            destination = stage / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            sums[relative] = sha256(destination)

        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
            capture_output=True, text=True).stdout.strip()
        manifest = {
            "archive_id": "reported_source_geography_v1",
            "design": "locked old composition instrument",
            "geography": "reported source geography",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": git_commit,
            "asset_count": len(sums),
            "integrity_command": (
                "cd " + str(FREEZE) + " && shasum -a 256 -c SHA256SUMS"
            ),
            "reconstruction_command": (
                "4_Macro_version/.venv/bin/python "
                "4_Macro_version/step_1/code/pipeline/"
                "95_old_design_geography_key_tables.py"
            ),
            "scope": [
                "persisted category shocks",
                "persisted composition instrument",
                "persisted analysis panel",
                "five registered key-table comparisons",
                "reported-source aggregation baseline",
                "geography reconstruction code and routing tables",
            ],
        }
        (stage / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        lines = [f"{value}  {relative}" for relative, value in sorted(sums.items())]
        (stage / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
        stage.rename(FREEZE)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise

    verify_archive()


if __name__ == "__main__":
    main()
