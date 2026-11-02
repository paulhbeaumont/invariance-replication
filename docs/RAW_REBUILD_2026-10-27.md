# Raw rebuild - October 27, 2026

## Source recovery completed

GitHub authentication now works. An isolated sparse checkout was detached at
`fc9df562037324ecfeb1cae1d34de8b13d408ce5`, not the current branch tip.
All eight historical source-snapshot hashes match. The package preserves 297
unmodified scripts, layouts and reference files under
`provenance/historical_fc9df56/`, with per-file SHA-256 and Git object IDs in
`provenance/historical_code_manifest_fc9df56.json`.

No missing-code request to the co-authors remains at this point. This does not
yet certify a complete raw-to-results reconstruction.

## Current checkpoint: paper inputs reproduce; Kalawao exclusion documented

- All 42 DCADS fiscal files and all 12 USAspending production years are parsed
  and geographically classified using the pinned historical code.
- Employment rebuilt under pandas 2.3.3 exactly matches all 136,444 original
  county/year records, including keys, flags, values and types. The initial
  pandas 1.4.4 outputs are preserved under `employment_pandas144/` within the
  isolated build. Their one-ULP differences have not been substituted or hidden.
- The complete source51 procurement matrix (3,058 counties x 53 years) matches
  the independent historical target exactly in every value and dataframe type.
  The 152,900-row annual county panel also matches every value, missing-value
  location, key and column. Its only type difference is Windows `int32` versus
  `int64` storage for state codes. The strict comparison and this separately
  verified lossless storage distinction are both retained.
- From those reconstructed annual sources, the two h=3 county regressions
  reproduce the saved log and midpoint coefficients, SEs and F-statistics.
  All six reported aggregation specifications also reproduce, with 150
  additional numerical-field comparisons against independent saved references.
  All compared coefficients and first-stage F-statistics are numerically equal;
  the largest SE difference is 3.39e-21. No qualitative change was found in
  these checkpoint results.

Full raw-to-results certification remains false because the remaining
disaggregation/decomposition and balance/
heterogeneity inputs have not yet been rebuilt and re-estimated from these new
sources. No missing raw-derived input was replaced with an old prepared input
in the checkpoint.

### The historical Kalawao audit

The unchanged `12_build_county_income.py` (SHA-256
`10603714f93d829f4c87a08173235a1558de10ef0155cc949e1a3639ea20f023`)
states that Kalawao, CCG 15005, has no separate income/population denominator
and receives zero procurement. The rebuilt data violate the latter assertion:
the sum of absolute annual net procurement in its audit is $34,759,561 in
2006-dollar units, over six nonzero years. The BEA state-total, mapping and
population checks pass; the producer writes its data but returns failure on
that single assertion.

Kalawao is absent from both the original and rebuilt 3,058-county paper sample.
The retained-sample annual inputs match the historical targets, and the
checkpoint regressions match as described above. This failure does not change
those paper estimates, but cannot be relabeled a passing original audit.
`population_attempt2_stage.json` remains `FAILED`.

Paul approved keeping and documenting the existing exclusion on October 27.
**Kalawao is excluded because BEA does not report a separate income/population
denominator for it**, not because procurement is zero. No denominator is
imputed and its separately identified procurement is not moved to Maui.

Implemented in Paul's wrapper only: the original producer still runs every
check and emits its original failure. The wrapper accepts only the single
Kalawao zero-procurement assertion, and only after rebuilding the retained
county identifiers and checking exact agreement with the independently saved
3,058-county reference, with Kalawao absent from both. An additional failure,
another county, changed county support, an exception, or a different return
code remains fatal. The warning is bound to the SHA-256 hashes of the actual
income, population, employment and independent county-reference inputs; changed
inputs invalidate it.

The fresh `population_attempt3_stage.json` therefore records application status
`READY_WITH_EXCLUDED_UNIT_WARNING` while separately retaining
`original_audit_status: FAILED` and return code 1. This explicit application
readiness satisfies the population prerequisite; it is not an original-audit
PASS or full raw-to-results certification. No outcome, shock, denominator,
weight, sample, fixed effect, cluster or numerical reference changed.

The two completed attempt-2 Parquets were moved, recoverably, into
`data/raw_rebuild/built/historical_fc9df56/population_attempt2_preserved/` before
the fresh run. Both newly generated files match those preserved originals
exactly, including their SHA-256 hashes. Earlier logs and failed certificates
remain untouched. The 297 pinned source files still verify, all 184 frozen
package checks pass, and all 10 current host/audit-policy tests pass.

### Checkpoint evidence

Reports under project-root `batch_logs/ns_replication_Paul/raw_rebuild_20261027/`:

- `employment_comparison_pandas233.json`: exact employment reproduction.
- `source51_diagnostic_comparison.json`: the initial strict type comparison.
- `source51_exact_value_comparison.json`: exact values, with the sole integer
  storage-width distinction explicitly recorded.
- `population_attempt2.log` and `population_attempt2_stage.json`: unchanged
  original failure, not suppressed or reported as passed.
- `population_attempt3.log` and `population_attempt3_stage.json`: fresh audit,
  original failure plus the approved application-specific warning.
- `population_exclusion_policy_check.json`: identical old/new income and
  population bytes, preserved original failure, and accepted scoped warning.
- `kalawao_policy_guard_tests_final.log`: ten software guard tests, including
  rejection of other/additional failures, retained-sample changes and changed
  warning-evidence bytes.
- `verify_after_kalawao_policy.log`: all 184 frozen package checks still pass.
- `checkpoint_results.json`, `checkpoint_results_attempt5.log` and
  `checkpoint_all_aggregation_fields.json`: independent numerical comparisons.
- `checkpoint_attempt5/results/aggregation_summary_tables/selected_results.csv`:
  newly estimated aggregation checkpoint results.

From the package directory, re-run the diagnostic value comparison with the
raw-build environment:

```text
python -B code/diagnose_historical_inputs.py --compare-existing
```

The initial diagnostic script constructs annual inputs in a separate directory
and records the unresolved original audit. The checkpoint uses
`code/checkpoint_historical_results.py --attempt N` with the existing prepared-
data Python environment. New attempt numbers preserve old failed diagnostics;
neither script changes `replicate.py`'s input boundary.

Before the exclusion-policy update, all 15 source-recovery/host-I/O guard tests
passed. The ten current host/audit-policy tests also pass; the 297
preserved historical files still verify against their manifest and all eight
independent source anchors; all 184 original package input/reference/function
checks still pass. These checks do not erase the failed original population
audit or certify the not-yet-reconstructed application stages.

## Execution boundary and specification

The eight pre-code protocol answers remain those in `SPECIFICATION.md` and
`RAW_TO_RESULTS.md`: county spending and its historical IV are repeated across
industry cells; midpoint disaggregation uses midpoint shares; standard-growth
aggregation uses lag employment shares and the augmented county IV equation;
the fixed county support and reported log/weight benchmarks are unchanged.
There is no population regression multiplier and no new outcome cap, exposure
window, fixed effect, cluster, or scaling denominator.

`code/historical_rebuild.py` verifies every preserved source before execution.
The historical common module's supported path override points at an isolated
workspace under `data/raw_rebuild/built/historical_fc9df56/`. Output/scratch paths
are checked to stay there; logs stay at the enclosing project's single root
`batch_logs/ns_replication_Paul/raw_rebuild_20261027/`. No original shared inputs,
prepared package data, manuscript or numerical references are overwritten.

## Runs and diagnostics

- Employment: the original producer 11 completed for 1975-2018, including its
  own allocation, conservation and extensive-margin accounting. The explicit
  historical options are national industry support and `--negative-cells drop`,
  matching the preserved producer's defaults, not a newly selected cleaning
  rule. The original's declared 134.44 employee-year adjustment is retained.
- Fixed-width procurement: the original parser handled 39 declared fiscal files
  through FY2003, including the transition quarter. Early attempts encountered
  Windows sharing locks on output moves. Explicit attempts 2/3 preserve those
  failures and rerun only missing files; attempt 3 completed.
- Delimited procurement: FY2004-2006 parsing completed with the original
  declared layout/anomaly/month-key rules and original row counts.
- Geography and the FY2007-2018 USAspending production block: completed. The
  latter reads the exact August archive bytes verified against independent receipts.
- Population/income: constructed, with the single historical assertion failure
  described above. Attempt 2 retained the full original audit and returned
  failure on Kalawao. Attempt 3 retains the same original failure and records
  Paul's application-specific warning after verifying unchanged retained county
  support. The earlier missing-dependency diagnostic is also retained.

The first independent employment comparison checked the original source file's
saved SHA-256 before comparing all 136,444 rows. County/year keys, flags and
dtypes match. Employment is bitwise identical in 133,499 rows; 2,945 values differ
by exactly one ULP (one representable floating-point step), at most
1.1641532182693481e-10 workers and 2.220446044704377e-16 relatively. The strict
comparison failed and remains recorded, rather than being relabeled exact.
These were numerical differences, not a change to the construction formula.
The subsequent pandas 2.3.3 rebuild is exactly equal to the original; its
separate comparison and regression checkpoint are recorded above.

Reports: `employment_independent_comparison.log` preserves the strict failure;
`employment_value_comparison.json` records the full size/count/precision check.
The original comparison data and all 184 frozen package inputs/references remain
untouched. The county and aggregation checkpoint has now been re-estimated;
the other application stages remain pending.

## Host compatibility

The existing Anaconda environment (pandas 1.4.4, NumPy 1.21.5) cannot run the
original USAspending parser: nullable boolean arrays convert to object arrays
and fail as NumPy indices. A separate `.venv_raw` uses the pinned versions in
`requirements_raw.txt`; pandas 2.3.3 produces the expected boolean arrays. No
mask, geography classification or raw parsing rule was rewritten to fix this.
The older prepared-data launcher/environment is unchanged.

Windows/Dropbox can briefly lock newly closed output files. A process-local
I/O shim retries the same file move on Windows sharing errors only when both
paths belong to the isolated build. It covers the legacy Python pathlib
accessor as well as `os.rename`/`os.replace`; it changes neither bytes nor
destinations. The original geography writer also retains a completed Parquet
reader while replacing its source, which Windows prohibits. The shim tracks
workspace readers and releases matching completed readers when a file move is
blocked. This is a host-I/O adaptation, not a geography or dollar correction.
Host-I/O guard tests cover both the legacy pathlib accessor and actual Parquet
reader/output replacement. These software tests are not independent evidence
about the scientific results.

## Reproduction commands

From the research-project root, choose the raw-build interpreter (on this
machine `4_Macro_version/NS_replication_Paul/.venv_raw/Scripts/python.exe`).
The following use `python` as shorthand for that interpreter:

```text
python -B 4_Macro_version/NS_replication_Paul/raw_rebuild.py freeze-historical-code --checkout <exact-commit-checkout>
python -B 4_Macro_version/NS_replication_Paul/raw_rebuild.py historical-stage --stage employment
python -B 4_Macro_version/NS_replication_Paul/raw_rebuild.py historical-stage --stage dcads-fixed
python -B 4_Macro_version/NS_replication_Paul/raw_rebuild.py historical-stage --stage dcads-delimited
python -B 4_Macro_version/NS_replication_Paul/raw_rebuild.py historical-stage --stage dcads-geography
python -B 4_Macro_version/NS_replication_Paul/raw_rebuild.py verify-usaspending
python -B 4_Macro_version/NS_replication_Paul/raw_rebuild.py historical-stage --stage usaspending --years 2007-2018
python -B 4_Macro_version/NS_replication_Paul/raw_rebuild.py historical-stage --stage population
python -B 4_Macro_version/NS_replication_Paul/raw_rebuild.py historical-stage --stage county-inputs
```

These are construction stages, not a certified replacement for `replicate.py all`.
Existing stage certificates and outputs are protected. A failed attempt needs
an explicit new `--attempt N`; fixed-width parsing alone supports
`--resume-missing` to retain completed fiscal files. All failed diagnostics are
kept. The original production URLs' availability problem remains documented in
`UPSTREAM_RECOVERY_2026-10-23.md`; a fresh future download needs a verified host
for the retained vintage, not a silent current-vintage substitution. Following
Paul's approval, exact archives are now bundled locally and the default parser
source is the package production directory. An explicit checksum-verified
`--usaspending-root` remains supported. See
[USASPENDING_ARCHIVES.md](USASPENDING_ARCHIVES.md) for the local bundle and
separate public-deposit boundary; no upload has taken place.

## Local USAspending bundle validation

The package now contains independent copies of all 18 receipt-pinned ZIPs
(9,359,636,428 bytes): 12 production files and six diagnostic overlaps, plus
nine original receipts/request files. No original archive, frozen prepared
input, numerical reference, or historical parser source was changed.
`provenance/usaspending_bundle_manifest.json` records the exact local paths,
roles, vintages and historical hashes; `data/raw/USAspending/SHA256SUMS` gives
the 27 archive/metadata checksums.

The copy report is under root
`batch_logs/ns_replication_Paul/usaspending_bundle/`. The first attempt stopped
after four archives on a Windows sharing violation during a verified partial
file's rename. Its `FAILED` report remains intact. The next attempt reused the
four identical copies, verified and completed the saved partial, and copied
the remaining inputs. The bounded retry applies only to the same file move
inside the bundle and does not alter or overwrite file contents.

All 13 bundle I/O guards pass, including corruption/overwrite rejection,
diagnostic-versus-production selection, completion of a verified partial and
operation without the original project. All 20 existing source-recovery and
historical host/audit guards also pass. The 184-file frozen-input/function
verification passes. These software checks do not replace the independent
original receipt hashes or certify remaining raw-to-results stages.

The actual default production resolver has additionally selected and verified
the local FY2007/FY2018 endpoints and rejected a diagnostic year. This is a
file-resolution check, not a rerun of completed annual parsers or regressions.
Reports and guard logs are kept in the same root-log directory. No public
deposition has been performed.
