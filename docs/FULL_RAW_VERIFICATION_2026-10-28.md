# Full raw-to-results verification - October 28, 2026

Status: **PASS - complete staged raw-to-results verification**, using the
source boundary below and the approved Kalawao exclusion. Paul approved this
clean-run plan on October 28. Previous prepared-data runs and checkpoints are
not substituted for this run.

## Outcome

- All annual sources rebuilt inside an initially empty derived-data run.
- 94 application input files constructed; 96 independent input comparisons
  pass, with floating-point differences explicitly recorded.
- All ten disaggregation columns, six aggregation specifications, four separate
  control specifications, twelve balance tests and the decomposition reproduce
  their saved estimates, clustered SEs and ordinary first-stage F-statistics.
- **422 final numerical comparisons pass.** Significance categories at 1%, 5%
  and 10% are unchanged. No qualitative result changed.
- The 184 original reference/source-file and numerical-function checks still
  pass. No benchmark was regenerated or replaced; the manuscript was not edited.
- Both Stata figures and the five-page recap were regenerated from fresh
  estimates. Every page was rendered and visually inspected. Missing beta
  glyphs and crowded heterogeneity notes were corrected without changing data.

Easy-to-find PDF: `output/pdf/NS_results_recap_raw_verified_2026-10-28.pdf`.
It is a checksum-identical copy of the certified run PDF, SHA-256
`f2e995c9331f8cd70d238a0166fa1ea1731bb9358409e201d04d7844f40999b6`.

The final certificate is at project-root
`batch_logs/ns_replication_Paul/full_raw_20261028_clean_01/full_verification.json`.
The field-by-field audit is `all_reported_numeric_fields.json` there. Successful
application attempt 9 is selected explicitly. Earlier failed or interrupted
attempts remain documented below. This is a staged clean run with verified
same-run recovery, not a claim that the final one-command recipe was separately
rerun from beginning to end without interruption.

## What is and is not exact

Raw archives and historical source code retain their exact hashes. The retained
annual county/procurement values match exactly; only state-code storage differs
(Windows int32 versus historical int64). NAICS6 endpoint summation roundoff is
at most 1.46e-11. In tiny positive cells it amplifies to individual MPGR/log
differences up to 4.15e-7/4.60e-7. Cell identities and missing-log support agree.
The largest normalized weighted difference in the explicit nonlinear audits is
1.64e-11. No value is clipped, discarded or replaced to obtain a match. Original
formulas are checked against both old and new endpoints. The largest final
result difference is 1.01e-11, in percentage deviation; displayed numbers and
significance decisions are unchanged.

Kalawao remains outside the unchanged 3,058-county sample. Its historical
population audit still records FAILED; the false zero-procurement assertion is
not relabeled PASS. Only that approved excluded-unit condition is accepted as
an explicit warning. All other audit failures remain fatal.

The harmonized EFSY panel is an external published source, not a reconstruction
of its authors' Census imputation. USAspending uses the preserved historical
vintage, not today's revised download. The local chain is complete; public
hosting/licensing review and GitHub release remain separate. No public upload
was performed.

The protocol's eight answers are unchanged from `SPECIFICATION.md`: county
baseline, local industry cells and state/CEA aggregates; additive employment;
county-year spending and historical IV repeated across disaggregated cells;
lag-employment-weighted exposure/instrument at the regional level; county/year
effects and the reported industry-year augmentation, never county-year effects;
midpoint/midpoint disaggregation and standard/lag-share aggregation; identical
3,058-county support, Kalawao excluded and natural log losses retained; checks
against independently saved inputs and coefficients/SEs/F values, not newly
regenerated benchmarks. The existing log/weight comparisons are deliberate
benchmarks, not a specification search. No estimand changes are authorized.

## Clean-run boundary

Run ID: `20261028_clean_01`. Its directory under
`data/raw_rebuild/runs/` was created empty and contains no copied prepared
analysis inputs. The run uses checksum-verified external raw archives already
acquired for this package, including the exact bundled USAspending vintage.
It freshly converts the public EFSY CSV and runs the pinned historical annual
producers before constructing any regression inputs. This does not claim a
rerun of the EFSY authors' imputation from original Census files; their public
harmonized panel is an explicitly declared external source.

Original producer source bytes, data definitions, sample restrictions, old
references and old failed audits remain unchanged. Independent original data
may be read only for verification, never to seed the fresh construction.
Logs belong only under project-root
`batch_logs/ns_replication_Paul/full_raw_20261028_clean_01/`.

## Software

Raw producers use `.venv_raw` and `requirements_raw.txt`. Estimators retain the
existing Python 3.9 analysis environment (NumPy 1.21.5 / pandas 1.4.4), preserving
the numerical routines and their historical software. DuckDB 1.4.1 is available
only in `.venv_raw`; an identical, isolated copy of its CPython 3.9 package and
extension is staged under `.analysis_extra/`, without replacing analysis NumPy
or pandas. The binary hashes are checked before use. A fresh installation can
use `pip install --target .analysis_extra --no-deps -r requirements_analysis_extra.txt`.
The current machine reused the installed bytes because network access failed.

## Required completion checks

1. All annual sources rebuilt inside this initially empty run, with the approved
   Kalawao warning explicit and every other audit failure fatal.
2. All regression inputs rebuilt there, including NAICS6 reconciliation,
   industry-year-effect cells, the decomposition and characteristic interactions.
3. Ten disaggregation columns, six aggregation cases, four separate-control
   cases and twelve balance diagnostics re-estimated against unchanged references.
4. Tables/figures regenerated from fresh estimates, then final source/reference
   integrity and all discrepancies documented. Full certification remains false
   until the whole chain has passed.

## Re-run from external source archives

Keep two environments: `requirements_raw.txt` for original annual producers,
and `requirements.txt` plus the isolated `requirements_analysis_extra.txt`
DuckDB installation for application construction/estimation. On Windows the
analysis interpreter's `Library/bin` must be on its process PATH. Stata 17 and
pdfLaTeX are required only for figure/PDF rendering.

From this package folder, with a **new** run identifier:

```powershell
& .\.venv_raw\Scripts\python.exe -B .\raw_to_results.py all --run-id YOUR_NEW_RUN --analysis-python 'C:/ProgramData/Anaconda3/python.exe'
```

The controller verifies the external raw archives, starts with an empty derived
tree, rebuilds annual and regression inputs, estimates the locked specifications,
renders the recap and performs the final comparison. It will not overwrite a
previous run. An interrupted annual build can resume completed stages within
its own run, never from another run's intermediates. Failed application builds
require an explicit higher `--attempt`; earlier failed artifacts are retained.

`replicate.py all` deliberately remains the separate prepared-data workflow.
Neither entry point updates the independent `reference/` targets. A final
certificate is valid only when every required stage in that specific run passes;
source checks or a prepared-data PASS alone do not suffice.

## Execution notes

- The public EFSY CSV was freshly converted: all 44,262,620 rows, with no
  filtering, allocation or recoding in conversion. The pinned employment
  producer then rebuilt all 44 years and passed its own checks.
- The original declared `--negative-cells drop` rule remains unchanged: its
  134.44 employee-year adjustment is a legacy construction rule, not a new
  cleaning decision. No cells or counties are additionally removed.
- All 39 declared fixed-width procurement files and the three later delimited
  files rebuild successfully. Geography harmonization has also completed for
  all 42 fiscal files. Original year/city/contract override assertions pass.
- All 12 production USAspending archives, FY2007-2018, have now been parsed
  afresh inside this run using the unchanged historical parser. The separate
  source stage verified all 220 catalogued source records and all 18 bundled
  USAspending archives; the six overlap archives remain diagnostic only.
- The annual build is complete. The fresh 152,900-row county panel and the
  3,058-county annual net-procurement matrix match every independent original
  data value exactly. Only the county panel's `state` storage width differs
  (Windows int32, original int64); values are identical and no reference or
  saved reconstructed file is cast to disguise the difference. The approved
  Kalawao warning is retained with the original audit failure visible.
- Ten historical host/audit guards and six new clean-run boundary guards pass;
  the 184 original input/function checks pass. These guards are software checks,
  not substitutes for independent old input/result comparisons.
- The old source catalog still pins the original full specification's hash.
  The downloader now accepts only the byte-exact, separately hashed approved
  Kalawao note: removing that note must recover the original document hash.
  All other changes still fail. Both the original catalog and its hash are
  unchanged; six downloader guards, including this narrow case, pass.
- `.gitattributes` disables automatic Git line-ending conversion so a later
  checkout does not invalidate the byte-pinned source/reference manifests.
- Application attempt 1 failed before constructing any regression input: a
  nested provenance-copy path exceeded Windows' legacy path-length limit.
  Its artifacts and failed certificate are preserved. Source copying now uses
  Windows' extended-path form, without shortening/omitting provenance files or
  changing construction arithmetic. Seven clean-run software guards pass,
  including the new path-copy check. Attempt 2 starts separately from the same
  successfully rebuilt annual sources.
- Attempt 2 rebuilt and independently compared all 39 annual NAICS6 files
  successfully, then stopped when Windows refused to resolve a literal
  `y*.parquet` wildcard in the query boundary guard. The guard now resolves
  the containing directory and retains the original query glob; parent-path
  wildcards and paths escaping the clean run remain forbidden. Eight guards
  pass. Failed attempt 2 is preserved; attempt 3 independently reconstructs
  its application inputs again, without seeding from attempt 2's files.
- A read-only horizon preflight then exposed growth-rate amplification of
  tiny endpoint roundoff: in 1981 the largest MPGR/log-growth differences were
  approximately 1.14e-7/1.23e-7, while the largest weighted MPGR contribution
  difference was 8.44e-17. Endpoints still differ by at most 1.46e-11 and cell
  identifiers/support agree. These are not declared identical individual
  growth inputs. The audit verifies the original growth formula on both sets
  of independent endpoints and checks weighted first/second outcome moments;
  all endpoint/weight/key checks retain their
  strict tolerances. No observation, growth value, weight, formula, reference
  or regression tolerance is changed. The final saved-result comparisons
  remain mandatory. Ten clean-run software guards pass, including rejection
  of a changed growth formula even with a negligible weight.
- Attempt 3 was deliberately interrupted during its repeated annual stage;
  its RUNNING record and partial artifacts remain with a separate interruption
  note. Attempt 4 explicitly resumes only the 39 fresh annual outputs of
  failed attempt 2, inside this same initially empty run. The annual function
  must be byte-identical to its saved attempt-2 code, every file must already
  have an independent PASS, its reference anchor is checked again, copies are
  checksum-verified and re-compared. `same_run_annual_resume.json` records the
  files. No original prepared analysis file is used as a construction seed.
- Attempt 4's extra diagnostic failed because it mistakenly used the
  intermediate file's unused survivor-midpoint weight for log-growth moments,
  and an initially over-strict 1e-14 tolerance. Neither was the estimator's
  actual weight or its established input tolerance. The diagnostic now uses
  full-county lag shares, after independently checking identical county
  baseline employment, and the existing input policy (rtol 1e-12, atol 1e-10).
  The 1981 log first/second weighted-moment differences are 1.10e-12/2.59e-11;
  MPGR differences are 8.44e-17/1.60e-16. The failed diagnostic is preserved;
  it is not represented as a numerical regression failure. Attempt 5 uses the
  corrected audit and the same verified same-run annual files. No builder
  growth formula, regression array, old target or result tolerance changed.
- The all-year read-only diagnostic confirms identical horizon-cell and
  missing-log support in every year. Its largest log `w*y` difference is
  1.52e-11. Attempt 5's extra second-moment check used `w*y^2`, not the square
  `(w*y)^2` relevant to weighted IV scores, and stopped at 1984. Its record
  remains. The diagnostic is corrected to compare `w*y` and `(w*y)^2` using
  the existing input tolerances, with formula, endpoint and final result
  checks still mandatory. Attempt 6 resumes the same verified annual files.
  This changes only the newly added audit, not any construction/estimation
  output or independently saved benchmark.
- Attempt 6 passes all 36 horizon partitions and the combined 22,506,999-row
  log file. Its final decomposition audit flags the unnormalized `lag_log`
  numerator (35 values; maximum 7.50e-7 employment-log units). The actual
  outcome `mean_log=lag_log/E_b_c` already passes the original input tolerance,
  with maximum difference 1.64e-11. The audit now verifies that same original
  identity on both saved/new data and compares its normalized contribution;
  neither the numerator nor denominator nor outcome is changed. All other
  decomposition columns already pass, including counts and entry/exit terms.
  Attempt 7 resumes the same verified annual files with this explicit audit.
- The remaining component preflight reproduces all eight aggregation frames.
  Its first characteristic-source comparison incorrectly compared the filtered
  3,058-county frame to an archived **full 3,101-CCG** snapshot. The comparison
  now occurs before the unchanged sample filter; both 939,603-row 1978/2000
  snapshots pass. A second diagnostic identified a 5.11e-15 income-standardization
  difference amplified in its IV interaction: the original 1978 sensitivity
  builder uses NumPy ndarray mean/std, whereas the 2000 builder uses pandas.
  The reconstruction now follows each original reduction exactly, with the
  same mathematical standardized characteristic, not a new scale/definition.
  All characteristic and interaction frames then pass, without adding an
  interaction-tolerance exception. Attempt 8 incorporates these two faithful
  comparison/arithmetic corrections; older attempts and diagnostics remain.
- Attempt 8 passes all 96 independent input comparisons and writes the fresh
  94-file manifest. Only the final selection-record write fails: the root
  project's log directory is outside the package and cannot be expressed by
  `relative_to(package_root)`. The pointer now stores its resolved absolute
  log-record path, while application data remain package-relative. Attempt 9
  repeats the already-passing build with this metadata-only correction. The
  failed attempt-8 certificate and complete inputs are retained.
