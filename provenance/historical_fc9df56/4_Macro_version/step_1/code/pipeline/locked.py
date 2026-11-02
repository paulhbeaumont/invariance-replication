"""The locked baseline of the macro fiscal-multiplier application.

Single source of truth for every constant the pipeline uses. Precedence when
sources disagree: the values producer 57 actually executes > producer 54's
defaults as executed by 57 > anything in producers 55/56. A constant that
cannot be traced to what 57 executed stops the work: report, do not guess.

00_capture_golden.py and 05_baseline_tieout.py assert this file against the
old modules at runtime (lib/oldpath.check_lock), so drift in either copy
fails loud. Provenance is quoted per constant as old-file:line.
"""

# ---- contract category axis (54:103, 54:104, 54:111) -----------------------
NDIG = 4            # 4-character FSC/PSC category codes
NCAT = 800          # top-800 categories by pooled national real dollars
CLIP = 1.0          # symmetric clip on scaled category shocks

# ---- instrument constructor (57:539-542 BASELINE; 57:137 LAG; 51:143) ------
INSTRUMENT = "composition"
SPAN = 5            # exposure window length, years
LAG = 5             # exposure window lag, years
RULE = "positive"   # contract aggregation rule for the locked cell
FLOOR = 0.0         # dollar floor
DENOM = "population"

# ---- estimation (57:539-542, 57:556 SCALARS, 57:140 SCALE, 57:126) ---------
FE = "divisionyear"
HORIZONS = (2, 3, 4)
LEVELS = ("czone", "msa", "cea", "ea", "state")
SCALARS = ("EW", "P_j")     # the legal parent-level weight family

# The COUNTY regression's weight and cluster, ruled by Adrien 2026-09-02.
# Distinct from SCALARS above: that is the parent-level c_j family used when
# climbing the ladder, this is the single cell the county regression itself
# runs. Traceable to 34_tables_payoff.py:193, `wc = None if not popw`, which is
# c_j = 1 evaluated at cell = county; the population arm on the same line
# (`wc = P0`) is a variation, not a co-equal baseline.
#
# NOT ASSERTED BY oldpath.check_lock, and that is a real gap rather than an
# omission: the old path expresses both of these as a code branch and a
# hard-coded factorize, not as a module constant, so there is nothing for
# check_lock to compare against without inventing one. A producer running a
# different cell or cluster is therefore a recorded variation, visible here,
# and not a failing assertion.
WEIGHT_CELL = "county"      # the group om_c is normalized within
WEIGHT_CJ = "EW"            # c_j = 1; every unit of WEIGHT_CELL counts once
CLUSTER = "ccg"             # county clusters (mpgr_disagg.py:643)
OUTCOME = "y_mpgr"          # y_std and y_log ride along as companion columns
SCALE = 1e5
POP_BASE_YEAR = 1975

# ---- sample (51:145 DATA_FIRST; dcads_common SEAM_YEAR; panel span) --------
FIRST_YEAR, LAST_YEAR = 1966, 2018
# Until 2026-09-02 the end of the sample was a literal `range(..., 2019)` at
# seven call sites, so no constant could move it and check_lock asserted none of
# them. store.target_years is now the single definition and all seven call it.
# FIRST_YEAR/LAST_YEAR are the DATA span: contract dollars run to FY2018 and are
# not in question. TARGET_LAST is the last year a regression is estimated ON,
# and it is the one that moves.
#
# 2016, not 2018, since 2026-09-02, because 2016 is where EFSY's imputation
# ends. Eckert, Fort, Schott and Yang (NBER WP 26632) build a 1975-2016 panel
# and impute only years prior to 2017; the 2017 and 2018 rows shipped in
# efsy_panel_naics.csv are raw CBP, the published unsuppressed cells with no
# desuppression applied, and they carry a `v1` index column that is empty in
# every earlier year.
#
# Measured on the source CSV, not inferred: 784,474 rows in 2016 against 350,097
# in 2017 and 349,203 in 2018, on the same 3,156 counties and the same ~960
# NAICS codes. County employment here is RECONSTRUCTED by summing cells, since
# the panel carries no county-total row and its readme says the sum of all
# observations IS the national total. Summing unimputed cells therefore gives
# 112.5M for 2017 against CBP's published figure near 130M, the reconstruction
# falls 11.2%, and the cross-sectional sd of y_log goes from 0.084 in 2016 to
# 0.595 in 2017. Leaving those years in put 24% to 49% of every headline
# coefficient on cells that were never desuppressed.
TARGET_LAST = 2016
DATA_FIRST = 1969
SEAM_YEAR = 2007

# ---- diagnostics convention (55:236, 56:182 vs 57: none) -------------------
# The shifter-count and BHJ shock-level tables winsorize x within year; the
# locked estimation in 57 never winsorizes. Recorded as an inconsistency in
# PIPELINE.md; the 06/07 ports preserve the 55/56 convention row-for-row.
WINSOR_DIAG_GROUP = "year"

# ---- REF cell (54:113-119): the assertion cell, NOT the locked baseline ----
REF = dict(rule="net", horizon=2, exposure="roll5", denom="population")
# per-horizon county-year counts from the frozen aggregation_baseline.csv
# (golden/frozen_aggregation_baseline.csv, column n_county); equal across
# rules by zero-fill invariance.
N_COUNTY = {2: 113_146, 3: 110_088, 4: 107_030}
N_COUNTIES = 3_058

# ---- horizons outside the lock --------------------------------------------
# HORIZONS above is the lock: the horizons producer 57 actually executed, and
# oldpath.check_lock asserts it field for field. EXTRA_HORIZONS is a separate
# constant BECAUSE it is not part of that lock. 57 never ran h = 1, so there is
# no old-path value to assert against, no golden gk/L fixture, and no frozen
# coefficient row. Stages 03 and 04 accept it only through an explicit
# --horizons flag; the default invocation of both is HORIZONS and nothing else,
# which is what keeps the stage-05 gate meaningful.
#
# The consequence, stated rather than buried: rows at a horizon in this tuple
# are ungated. Their only guarantee is that adding them leaves every HORIZONS
# row byte-identical.
EXTRA_HORIZONS = (1,)

# Measured on the first h = 1 build, not frozen from an old-path source, so it
# is kept apart from N_COUNTY. A mismatch here means the construction moved.
N_COUNTY_UNGATED = {1: 116_204}   # h=1, target years 1979-2016, measured 2026-09-02

# ---- tie-out tolerances ----------------------------------------------------
# Vector comparisons are exactness-by-construction: the extraction preserves
# operation order, so any loosening is a report-to-Adrien event, not an
# iteration step. Coefficients tie to the frozen CSV at RTOL_COEF.
RTOL_VEC = 1e-12
ATOL_VEC = 0.0
RTOL_COEF = 1e-6
