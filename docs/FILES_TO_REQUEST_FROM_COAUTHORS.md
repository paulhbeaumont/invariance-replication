# Files to request from the co-authors

Current status, October 28, 2026: **no additional co-author files are needed
for local replication**. The complete staged raw-to-results run passes all
reported models and 422 final numerical checks, with unchanged significance
categories. See [the verification report](FULL_RAW_VERIFICATION_2026-10-28.md).
Public hosting of the raw archive bundle remains a release task. The earlier
request and recovery history below are preserved, not an outstanding request.

Paul's Nakamura-Steinsson replication package; source-recovery status as of
October 22, 2026. This request concerns the newer historical county spending
intensity x national spending shift instrument, not the product-composition IV.

Update, October 23: Adrien identified the historical commit and supplied the
Dropbox locations. The replay manifest and USAspending archives/receipts are
now available locally. GitHub code retrieval awaits authenticated access, not
another search by the co-authors. See
[the recovery record](UPSTREAM_RECOVERY_2026-10-23.md) for current status; the
original forwardable request below is retained for provenance.

Update, October 27: GitHub access is working; the pinned commit was retrieved,
all eight historical source hashes match, and 297 source/layout/reference files
are preserved in the package. The code-recovery request is resolved. See
[the raw rebuild record](RAW_REBUILD_2026-10-27.md) for construction progress and
comparisons; full raw-to-results verification is still in progress.

The retained annual inputs and the county/aggregation checkpoint estimates now
reproduce. One original out-of-sample audit assumption (zero procurement in
excluded Kalawao) fails. Paul approved keeping the existing exclusion and
documenting that assertion as a specific application warning on October 27.
No further co-author file request or validation decision is pending for this
issue. The raw-build record gives the details; the full application rebuild
remains in progress.

The exact USAspending archives, receipts and overlap API payloads are now
bundled locally in the replication package. See
[USASPENDING_ARCHIVES.md](USASPENDING_ARCHIVES.md). Public hosting of this
historical vintage is a separate pending release step, not another missing-file
request to the co-authors.

## Short message to forward

Could you please send an archive of the original `4_Macro_version/step_1/`
folder, including its Python scripts, helper modules, record layouts, reference
files, configuration and run instructions? We have the prepared inputs and can
reproduce the estimates, but some upstream code is missing from Paul's copy, so
we cannot yet reconstruct all inputs from raw downloads.

The most important missing files are `code/py/51_county_multiplier_grid.py`,
`code/py/dcads_common.py` and `code/pipeline/lib/contracts.py`, together with the
later DCADS parsers and the USAspending download/parse/harmonization scripts
(the 29-34 series). Please include their dependencies rather than only these
individual scripts.

Ideally, send the version that generated `source51_contracts_net.parquet` and
`source51_county_panel.parquet`. Our saved source snapshot records commit
`fc9df562037324ecfeb1cae1d34de8b13d408ce5`, which may help locate it. The original
replay manifest at
`batch_logs/macro_paper_application/source51_replay_20160914/manifest.json`
would also be valuable, along with the run command/log if available.

Please also include the USAspending download manifests, original requests,
retrieval dates and checksums, which apparently lived in the original local
checkout's ignored `_data/` tree. We need these to distinguish reconstruction
differences from later revisions to the public data. No credentials are needed.
If the historical files are unavailable, please identify the version you can
provide so that we keep it separate from the historical specification.

## Detailed checklist

All paths below are relative to the original project root.

| Priority | Missing material | Why it is needed |
| --- | --- | --- |
| Essential | `4_Macro_version/step_1/code/py/51_county_multiplier_grid.py` | Recover the full county baseline builder, historical exposure construction and sample selection. Only selected arithmetic functions survive locally. |
| Essential | `4_Macro_version/step_1/code/py/dcads_common.py` | Shared source loaders, geography, deflation and contract-panel routines imported by the surviving employment/population scripts. |
| Essential | `4_Macro_version/step_1/code/pipeline/lib/contracts.py` and imported helpers | Recover contract processing without guessing its rules. |
| Essential | Later DCADS parsers and associated layout files | The early 272-byte parser survives, but later record formats needed through FY2006 are not recovered. Please include the entire parser/layout set; individual later filenames are not established here. |
| Essential | USAspending scripts 29-34 and their configuration/helpers | Recover acquisition, parsing and source harmonization. Confirmed names include `29_build_usaspending_manifest.py`, `31_parse_usaspending.py`, `33_harmonize_contract_sources.py` and `34_download_usaspending_overlap.py`. |
| Essential dependencies | Remaining annual county builders, including `code/py/13_build_county_panel.py`; `code/reference/`; configuration and run instructions | Reproduce the original annual inputs, mappings and price adjustments. References named in surviving code include `va_bea_combinations.csv` and `zark_fips_supplement.csv`. Include the actual files read by the recovered code. |
| Strongly preferred | `batch_logs/macro_paper_application/source51_replay_20160914/manifest.json`, run command and log | Identify the precise source files and input hashes of the historical replay. The directory spelling/date above is copied literally from the saved snapshot script. |
| Essential source provenance | USAspending original download manifests, request payloads, retrieval receipts and archive checksums | Reacquire the same source vintage where possible and detect revisions independently of the new builder. |
| Helpful, not a current blocker | `code/py/52_journey_ladder.py`, `code/py/53_ns14_exact_replication.py`, environment/requirements files | Confirm surrounding provenance and invocation. These are not substitutes for the source51 builder. |

The documented production source rule is DCADS through FY2006 and USAspending
for FY2007-2018. FY2001-2006 USAspending files are overlap diagnostics, not a
replacement for DCADS. If the production USAspending vintage can no longer be
downloaded, we may need its original archives; please send their inventory first
rather than a large undifferentiated raw-data dump. The Muratori benchmark file
is not a missing production input.

## Already available: no need to resend

- The prepared county/industry/regional inputs, extracted estimation routines
  and independent numerical benchmarks used by Paul's package.
- Full saved scripts `11_build_employment_cells.py`,
  `12_build_county_income.py` and `14_build_instrument.py`, plus the consistent
  county crosswalk. Their missing imports/reference files still need recovery.
- The early DCADS parser and 272-byte layout.
- Fresh copies of the catalogued NARA files, public EFSY panel, BEA CAINC1,
  Census NAICS concordances and historical BEA area definitions.

Every one of the 44,262,620 EFSY source records and 4,557,244 parsed early
procurement records matches the independent historical comparison data. This
does not yet establish a complete raw-to-results replication.

## Technical version identifiers

The saved snapshot describes inspected source files, not a certified execution
of the entire pipeline at that commit. These SHA-256 values identify the
specific historical files we are trying to recover:

```text
4_Macro_version/step_1/code/py/51_county_multiplier_grid.py
9d1b2d2e814478b68254e59f6c4e61ece2342aac81c6e9c01a099b973066620c

4_Macro_version/step_1/code/py/dcads_common.py
2244d364afc768a9c040fa48a74ff9236746b6f0576e2db727cad50b7457d9c3

4_Macro_version/step_1/code/pipeline/lib/contracts.py
ab244dcbfd0435f64fd559421af90293017bb6a5177b2e29216a779f2b3abe75

source51_contracts_net.parquet (already available comparison target)
212258ca316215ceccb567e4bbe8dfbfcec5443785e00e099bfd10186d69b678

source51_county_panel.parquet (already available comparison target)
2cea0645682623b0d7c355411c8dadd806db81e6563e8ec30eff9152cfe93c21
```

Local evidence: `provenance/original_4a/config/source_snapshot.json`,
`provenance/original_4a/code/build/snapshot_baseline.py`,
`reference/upstream_source_manifest.json`, and
`provenance/raw_source_evidence_20261022/county_input_snapshot_manifest.json`.
The production source chain is documented in the parent project's
`4_Macro_version/data_sources.md`.

No preparation rules, paper estimates, first-stage statistics or frozen
benchmarks have been changed during this recovery. Missing code will not be
replaced with inferred defaults to claim a complete replication.
