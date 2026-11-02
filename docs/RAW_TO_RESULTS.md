# Raw-to-results reconstruction

Current status, October 28, 2026: **verified complete** for the current paper
specifications, using preserved external source vintages and the published
EFSY panel. The staged clean run rebuilt annual and regression inputs, passes
all reported estimates/SEs/F's and 422 final numerical checks, and produced a
visually checked PDF. No qualitative result changed. See
[the verification report](FULL_RAW_VERIFICATION_2026-10-28.md) for commands,
source boundaries, numerical differences and the approved Kalawao warning.
The dated notes below are historical records, not current blockers.

Paul approved this work on October 22, 2026. The target is the existing newer
Nakamura-Steinsson-style instrument, no population regression multiplier,
principal horizon h = 3, with the already reported robustness specifications.
The co-authors' application and immutable numerical references must not change.

## Protocol and specification, fixed before implementation

1. Baseline units are consistent county groups; disaggregation cells are local
   industries at NAICS2/4/6; aggregates are states and component economic areas.
2. County employment must equal the sum of its prepared industry cells. Raw
   employment allocation/reconciliation must be checked against independent old
   inputs, not certified merely because two newly constructed totals agree.
3. Spending x_ct and the excluded instrument z_ct vary at county-year level.
4. Disaggregation repeats both unchanged; aggregation uses lag-employment-share
   averages X_st and Z_st. There is no product-composition instrument.
5. Retain county/year effects and the reported industry-year augmentation;
   aggregation retains county/year or region/year effects. No county-year FE.
6. Midpoint growth with midpoint industry weights is the invariant disaggregation
   specification. Log uses the preserved full-county lag shares. Aggregation
   uses standard growth with lag employment shares and equal aggregate weights.
   Population remains a spending denominator, not a regression multiplier.
7. The historical support is 3,058 consistent counties selected through 2018;
   h = 3 has 110,088 county-years through 2016. Preserve natural log-sample losses
   and zero-weight cells exactly as documented in SPECIFICATION.md.
8. Existing input-hash verification passes for all 184 frozen files and extracted
   numerical functions. Raw reconstruction is NOT yet certified. Checks proceed
   from sources and conservation to coefficients, SEs, and first-stage F values;
   frozen reference cases are never regenerated to accommodate a new builder.
9. The existing log/weight comparisons remain deliberate benchmarks, not new
   preferred specifications or specification searches.
10. The deliverable is the same paper tables, decomposition and balance exhibits
    generated from fresh raw downloads, without relying on bundled derived caches.

Spending remains (mean annual real net procurement over the h outcome years
minus baseline-year procurement) divided by baseline population, with the same
pooled 1/99 winsorization of spending only. The instrument remains historical
five-year procurement/population intensity, ending five years before baseline,
times the national spending change. No outcome transformation, deflator, source
seam, sample screen, exposure window, or winsorization rule is being changed.

## Initial findings

- The existing launcher is a prepared-data reproduction. Its successful `all`
  run is not evidence of a raw-to-results rebuild. The `build` stage rebuilds the
  county frame but reads already prepared annual employment/procurement inputs;
  several estimation stages read already prepared industry/regional frames.
- Historical source download receipts survive for DCADS/MPCF, BEA CAINC1,
  Census NAICS concordances and the BEA 2004 area delineation. They record URLs,
  byte counts and SHA-256 checksums and must be retained as original evidence.
- Original employment, population and geography producer scripts survive in the
  September county-input snapshot. They can be preserved for source recovery,
  but must not be executed against their historical paths in the parent project.
- Historical producer 51 and its shared `dcads_common.py` are not in this package
  or the macro code folders searched so far. Additional raw-to-source dependencies
  must be recovered before procurement reconstruction can be certified.
- The EFSY authors publish the harmonized 1975-2018 panel and imputation code:
  https://www.fpeckert.me/cbp/ . Downloading their source data is different from
  claiming to have rerun their confidential-cell imputation from Census files.
- Annual raw procurement includes historical NARA actions and the later
  USAspending block. The older product-composition project's audit and correction
  choices do not authorize changing this package's spending definition.

## Completion criteria

1. Every required external input has a source, exact vintage or retrieval record,
   local filename, checksum, download instructions and explicit dependencies.
2. All prepared inputs used by the executable analysis have raw-source builders.
3. A clean run starts without any derived data and completes without importing
   another project folder or silently using old prepared inputs.
4. Compare the rebuilt data, support, estimates, SEs and F-statistics with the
   immutable originals. Report differences and revised-source vintages explicitly.
5. Until these checks pass, keep the README's prepared-data boundary and do not
   claim that raw-to-results replication is complete.

Logs and diagnostics belong only under the enclosing project's root
`batch_logs/ns_replication_Paul/raw_rebuild_20261022/`. Progress and unresolved
dependencies will be recorded here as work proceeds.

## October 22 execution record

Completed:

- Preserved the historical download receipts, full NARA catalog, unmodified
  employment/population/geography producers, the early parser/layout, county
  crosswalk and original prepared-input manifest under
  `provenance/raw_source_evidence_20261022/`.
- Added `raw_rebuild.py` and `code/raw_sources.py`. Acquisition is independent
  of the old prepared data, uses SHA-256 checks, records retrievals, quarantines
  unexpected sources and refuses changed existing inputs/specifications. The
  downloader never disables TLS verification or logs temporary redirect tokens.
- Retrieved every catalogued asset: 220 source records representing 218 distinct
  files and 3,802,277,820 bytes. These include 211 historical NARA catalog records,
  EFSY source/readme, BEA CAINC1, five Census concordances and the BEA 2004 area
  delineation. This list does NOT include the missing USAspending source chain.
- All receipt-pinned files reproduce their original checksums. For 53 NARA
  objects absent from successful historical receipts, the old catalog sizes
  differ from actual files (e.g. some ZIPs differ by 160 bytes). Fresh files
  match the independently saved old raw files byte-for-byte. Only those
  independently verified hashes/sizes were added to the active source catalog.
  The old catalog, original failed checks, quarantined files and reconciliation
  report remain preserved. No economic-data revision was detected in this step.
- Converted the EFSY authors' public 1975-2018 CSV to Parquet without filtering,
  recoding or imputation. Compared every record to the pre-existing Parquet:
  all 44,262,620 rows, keys and employment values match exactly, with zero
  employment differences in every year. Serialization bytes can differ.
- Ran the unchanged historical 272-byte procurement parser on all 19 files in
  its declared applicability list (FY1966-1983 and the 1976 transition quarter).
  All 4,557,244 reconstructed records match the independently saved old files.
  The first strict-storage comparison caught Arrow `string` versus
  `large_string` offset widths. Its report is retained. The subsequent value
  comparison widens strings losslessly and still compares every character and
  numeric value exactly; both the storage differences and value comparisons are
  recorded. Neither parser nor source data was changed to force a match.
- Five source-acquisition guard tests pass: escaping paths, corrupt downloads,
  altered cached sources, changed specifications and unreceipted first
  retrievals. These are software tests, not independent scientific evidence.

Current limitations / next dependency:

- Need the original `4_Macro_version/step_1/` code, especially
  `code/py/51_county_multiplier_grid.py`, `code/py/dcads_common.py`, the later
  record-format parsers/layouts and USAspending acquisition/harmonization code.
  Paul does not have another checkout/archive and will ask the co-authors.
  [FILES_TO_REQUEST_FROM_COAUTHORS.md](FILES_TO_REQUEST_FROM_COAUTHORS.md)
  contains a forwardable request, prioritized dependencies and historical
  version/hash identifiers. Full raw reconstruction awaits those dependencies.
- The saved employment and population producers import `dcads_common`, which
  owns shared geography and deflation rules. Do not replace those rules with
  guessed defaults or silently borrow the newer product-instrument corrections.
- Rebuild later annual procurement, allocated/reconciled employment, population,
  income and geography, then all industry, regional and characteristic frames.
  Finally run the analysis on a clean derived-data tree and compare every paper
  estimate, clustered SE and first-stage F statistic to the frozen references.
- No raw-derived frame has been substituted into the existing analysis, and no
  paper coefficient or first-stage statistic has been changed or re-estimated
  during this source-recovery stage. Full raw-to-results certification is false.

### Reproduce source comparisons (developer audit)

On Paul's machine use the Anaconda/PATH setup in the main README:

```text
python -B raw_rebuild.py compare-efsy --reference <original_raw_efsy.parquet>
python -B raw_rebuild.py compare-early --reference <original_dcads_parsed/dcads_272>
```

Those explicit reference arguments are for independent historical comparisons,
not runtime inputs to the new source conversion/parser. The one-time
`inventory`, `complete-catalog` and `reconcile-catalog` commands preserve evidence
from the parent project; normal future downloads use the package-owned catalog
and do not consult the parent tree. Do not rerun the evidence-import commands to
overwrite their original snapshots.

Key reports under root `batch_logs/ns_replication_Paul/raw_rebuild_20261022/`:
`raw_rebuild_status.json`, `efsy_reference_comparison.json`,
`early_reference_comparison.json`, `early_reference_comparison_strict_storage.json`,
and `guard_tests.log`. The independently evidenced catalog-size reconciliation
is `provenance/raw_source_evidence_20261022/catalog_size_reconciliation.json`.

## October 23: co-author response and recovered evidence

Adrien identified the exact source51-producing commit and warned against the
current branch's changed common module. The original replay manifest and
USAspending archives, manifests, receipts and overlap requests are now present
locally. Code retrieval currently needs authenticated GitHub access or an
exact-commit archive. See [UPSTREAM_RECOVERY_2026-10-23.md](UPSTREAM_RECOVERY_2026-10-23.md)
for the evidence audit, historical URL availability problem, the separately
identified older replay mismatch and executable recovery commands. Neither the
old source catalog nor numerical benchmarks have been replaced.

## October 27: pinned code recovered; retained-data checkpoint reproduced

GitHub access is resolved. All eight historical source hashes match and 297
files are preserved under `provenance/historical_fc9df56/`. Raw employment,
DCADS/USAspending parsing and geography are rebuilt. Every retained annual
source51 input value matches the independent historical targets; a Windows
integer-width distinction is recorded separately. The two county regressions
and all six reported aggregation specifications reproduce their saved
coefficients, SEs and F-statistics.

Kalawao (CCG 15005) is excluded because BEA does not report a separate
income/population denominator for it. It was already excluded from the original
paper sample. Paul approved keeping and documenting this exclusion on October
27. The original population audit's false zero-procurement assertion remains
visible as a failure. The wrapper now records only that specific assertion as
an application-scoped warning after checking the unchanged 3,058-county support
against the independent historical reference. No data or specification changes;
all other audit failures remain fatal. A fresh population build produces files
with exactly the same SHA-256 hashes as the preserved failed-attempt outputs.
See [RAW_REBUILD_2026-10-27.md](RAW_REBUILD_2026-10-27.md) for exact evidence,
reproduction commands and the remaining application stages. Full raw-to-results
certification remains false.

## October 27: preserve the exact USAspending release inside the package

Paul approved a local bundle of the existing 12 FY2007-2018 production ZIPs and
six FY2001-2006 diagnostic overlap ZIPs, with original receipts and API payloads.
The August production vintage cannot be recovered by requesting the current
monthly archive. Copies now belong under `data/raw/USAspending/`, separate from
raw-build scratch and from frozen prepared inputs. The diagnostic years are
never production replacements.

`raw_rebuild.py bundle-usaspending --project <original-project-root>` copies
only receipt-matching bytes, refuses changed sources/targets, and creates a
separate bundle manifest and checksum list. `raw_rebuild.py verify-usaspending`
checks the complete local bundle without consulting the original project or
the network. The historical USAspending parser now defaults to the verified
local production directory; its numerical parsing rules are unchanged.

No public upload or data license is implied. The destination and distribution
review remain pending. See [USASPENDING_ARCHIVES.md](USASPENDING_ARCHIVES.md).
This resolves the local raw-archive packaging dependency, not the remaining
full raw-to-results application rebuild.
