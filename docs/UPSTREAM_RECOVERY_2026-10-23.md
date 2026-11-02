# Upstream recovery - October 23, 2026

This is Paul's separate replication package. No paper source, prepared input,
estimation function, sample definition or frozen result reference is changed by
this recovery stage.

October 27 update: authentication is resolved and the pinned code has been
retrieved and independently hash-verified. The October 23 account below is
retained as historical provenance; current work is in
[RAW_REBUILD_2026-10-27.md](RAW_REBUILD_2026-10-27.md).

## Historical code: identified, retrieval awaiting authentication

Adrien identified the source51-producing version as commit
`fc9df562037324ecfeb1cae1d34de8b13d408ce5` of
`thetfplab/aggregation-disaggregation-invariance` (branch `workstream/adrien`).
He confirmed the three essential source hashes and warned that
`dcads_common.py` has changed on the branch since then. Recover the pinned
commit, not the current branch version.

On this machine, the current Dropbox workspace is not a Git checkout. An
anonymous GitHub API request returns 404 and a non-interactive Git request has
no usable saved credentials. The commit has NOT been downloaded and its source
files have NOT yet been verified locally. A GitHub integration was suggested,
but no authenticated connection is confirmed. Do not change credentials or
account permissions automatically.

To continue, either enable GitHub access for an account authorized to read that
repository or save its exact-commit ZIP under this project. The archive URL is:

`https://github.com/thetfplab/aggregation-disaggregation-invariance/archive/fc9df562037324ecfeb1cae1d34de8b13d408ce5.zip`

Code recovery must remain separate from the current workspace and from frozen
provenance. Check recovered bytes against the recorded hashes before executing
them; their old absolute paths must not overwrite shared project inputs.

## Dropbox evidence recovered

The following historical material is now available locally and preserved under
`provenance/upstream_recovery_20261023/`:

- The source51 replay manifest and the original saved-reference/replay CSVs.
- USAspending production manifest and download receipts for FY2007-2018
  (archive vintage August 6, 2026).
- The six FY2001-2006 overlap requests and download receipts (August 14, 2026).
  These are diagnostics only, not replacement production inputs.

The replay's two input-cache hashes exactly match the package's frozen
`source51_contracts_net.parquet` and `source51_county_panel.parquet` targets.
Its recorded source51/common-module hashes agree with the separately saved
source snapshot. Agreement between historical manifests is not a check of code
bytes that have not yet been retrieved.

`provenance/usaspending_source_catalog_20261023.json` records the original URLs,
expected SHA-256 hashes, sizes, source roles and evidence locations. The original
October 22 acquisition catalog and scientific references remain unchanged.

All 18 original ZIPs (12 production and six diagnostic; 9,359,636,428 bytes)
match their independently saved SHA-256 hashes, byte counts and ZIP member
counts. The completed audit is recorded at the project root in
`batch_logs/ns_replication_Paul/raw_rebuild_20261023/upstream_evidence_check.json`.
An exact local archive match is not a fresh web download or a parsed-value check.

## Historical download availability

GET requests with a one-byte Range header checked the recorded URLs without
reading archive bodies or accepting another vintage. The initial availability
report records 11 production URLs returning HTTP 404 and one timing out. Five
overlap URLs responded HTTP 206 and one timed out. A successful response does
not establish a checksum match or a permanent hosting guarantee.

The complete per-URL record is
`batch_logs/ns_replication_Paul/raw_rebuild_20261023/historical_url_availability.json`.
Do not replace missing August production archives with current releases and
call them identical. For future raw-to-results replication, the package needs
checksummed distribution of the retained raw archives or another verified host
for exactly the same bytes. The host and redistribution terms still need review;
no archive has been uploaded or published.

## The old replay mismatch is not a current-paper result

The supplied original manifest itself reports `status: reference_mismatch`.
Its comparison is for `h = 2`, exposure `roll10`, through 2018, and explicitly
sets `reference_is_paper_result` to false. The saved reference versus replay
reports scaled coefficients 23.914984 versus 22.927134 and first-stage F
statistics 23.725879 versus 25.402861; observation and county counts match.

That is not the current paper's `h = 3`, `roll5_gap5` specification. Preserve
the discrepancy, inspect its cause after the original code is retrieved, and
do not rewrite any benchmark to conceal it. No conclusion about a change in the
current paper's estimates follows from this older comparison alone.

## Re-run the recovery checks

From the research-project root, using the package's usual Python environment:

```text
python -B 4_Macro_version/NS_replication_Paul/raw_rebuild.py recover-upstream-evidence --project .
python -B 4_Macro_version/NS_replication_Paul/raw_rebuild.py probe-upstream-downloads
python -B 4_Macro_version/NS_replication_Paul/code/test_upstream_recovery.py
python -B 4_Macro_version/NS_replication_Paul/code/test_raw_sources.py
```

Evidence preservation is idempotent only for identical files; changed originals
or catalogs are rejected. Verification does not copy the large ZIPs into the
package or substitute them into an estimation dataset. All logs/reports stay
under the single project-root `batch_logs/ns_replication_Paul/`.

The five new I/O guard tests pass, as do the five existing acquisition guards.
An initial test cleanup failed because Dropbox locked a temporary directory;
its log is retained. Test fixtures now use the operating system's temporary
directory, while diagnostic logs remain at the project root. These software
tests are not independent scientific evidence.

`replicate.py verify` still passes for all 184 frozen data/source/reference
files and source-extracted numerical functions. Full raw-to-results replication
remains incomplete: pinned code retrieval, portable historical builders, a clean
raw-to-derived build and independent data/result comparisons are still needed.
