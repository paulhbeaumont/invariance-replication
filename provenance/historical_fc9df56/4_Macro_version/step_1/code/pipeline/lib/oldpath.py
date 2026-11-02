"""Importlib harness for the OLD producers, read-only.

Loads the numbered files exactly the way 57 does, so tie-outs compare the new
pipeline against the old call surface, not a reconstruction of it. Used only
by 00_capture_golden.py and 05_baseline_tieout.py; the pipeline runners never
import this module.
"""
import importlib.util
import sys
from pathlib import Path

LIB = Path(__file__).resolve().parent           # .../step_1/code/pipeline/lib
PIPELINE = LIB.parent                            # .../step_1/code/pipeline
CODE = PIPELINE.parent                           # .../step_1/code
PY = CODE / "py"
STEP2 = CODE.parents[1] / "step_2" / "code" / "py"
sys.path.insert(0, str(PY))
import dcads_common as dc  # noqa: E402


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


p54 = _load("p54", PY / "54_composition_instrument_grid.py")
p51 = p54.p51                       # 54 loads 51 at module scope; reuse its copy
p57 = _load("p57", PY / "57_aggregation_invariance.py")
t28 = _load("t28", STEP2 / "28_table_d1.py")
w22 = _load("w22", STEP2 / "22_winsor_construction_error.py")

GOLD = dc.MACRO_DATA_ROOT / "temp" / "pipeline" / "_golden"


def check_lock(locked):
    """Assert locked.py against the values the old modules actually hold.

    Raises AssertionError on any mismatch; returns the number of checks run.
    """
    B = p57.BASELINE
    checks = [
        ("INSTRUMENT", locked.INSTRUMENT, B["instrument"]),
        ("SPAN", locked.SPAN, B["span"]),
        ("RULE", locked.RULE, B["rule"]),
        ("FLOOR", locked.FLOOR, B["floor"]),
        ("DENOM", locked.DENOM, B["denom"]),
        ("FE", locked.FE, B["fe"]),
        ("OUTCOME", locked.OUTCOME, B["outcome"]),
        ("HORIZONS", tuple(locked.HORIZONS), tuple(B["horizons"])),
        ("LEVELS", tuple(locked.LEVELS), tuple(B["levels"])),
        ("SCALARS", tuple(locked.SCALARS), tuple(p57.SCALARS)),
        ("POP_BASE_YEAR", locked.POP_BASE_YEAR, p57.POP_BASE_YEAR),
        ("SCALE", locked.SCALE, p57.SCALE),
        ("LAG", locked.LAG, p57.LAG),
        ("NDIG", locked.NDIG, p54.NDIG),
        ("NCAT", locked.NCAT, p54.NCAT),
        ("CLIP", locked.CLIP, p54.CLIP),
        ("LAG(51)", locked.LAG, p51.ROLL_LAG),
        ("DATA_FIRST", locked.DATA_FIRST, p51.DATA_FIRST),
        ("SEAM_YEAR", locked.SEAM_YEAR, dc.SEAM_YEAR),
        # The end of the sample. Added 2026-09-02, when it turned out this was
        # declared in locked.py, read by nobody, spelled as a literal at seven
        # call sites, and asserted here not at all. It is now one constant on
        # each side and they are checked against each other.
        ("TARGET_LAST", locked.TARGET_LAST, p57.TARGET_END_EXCL - 1),
    ]
    bad = [(n, ours, theirs) for n, ours, theirs in checks if ours != theirs]
    if bad:
        raise AssertionError(f"locked.py drifted from the old modules: {bad}")
    return len(checks)
