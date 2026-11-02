# Preserved USAspending archives

**Public distribution, November 2, 2026:** the unchanged archives accompany
[release v1.0.0](https://github.com/paulhbeaumont/invariance-replication/releases/tag/v1.0.0).
Run `python download_data.py --group all` before the complete raw-to-results
workflow, whose source audit verifies production and diagnostic archives.
`--group production` and `--group diagnostics` also retrieve them separately.
The diagnostics do not enter the regressions. Current instructions are in
[PUBLIC_RELEASE.md](PUBLIC_RELEASE.md); former local-only status below is dated
development history. Source manifests and historical hashes are not rewritten.

This package contains independent local copies of the exact raw ZIPs used by
the historical pipeline, not newly downloaded substitutes. No archive is
recompressed, renamed internally, edited or linked to its original file. The
copying command does not upload or publish anything.

## Production and diagnostic inputs

| Bundle directory | Contents | Role | ZIP bytes |
| --- | --- | --- | ---: |
| `data/raw/USAspending/production/` | 12 `FY20xx_097_Contracts_Full_20260806.zip` files, FY2007-2018 | Paper production inputs; Department of Defense Award Data Archive, August 6, 2026 vintage | 7,741,628,072 |
| `data/raw/USAspending/diagnostics/fy2001/` through `fy2006/` | Six bulk-download ZIPs and the corresponding `overlap_request.json` payloads/responses | Overlap diagnostics only; August 14, 2026 vintage | 1,618,008,356 |

Total: **18 ZIPs, 9,359,636,428 bytes** (about 9.36 GB, or 8.72 GiB).
The production directory also contains its original `manifest.csv` and
`download_receipt.csv`. The diagnostics directory contains its original
`download_receipt.json`. These nine metadata files are copied byte-for-byte
from the separately preserved historical evidence.

FY2001-2006 USAspending files are not production replacements for the earlier
DCADS source. The production loader accepts only FY2007-2018; diagnostics are
stored separately to prevent accidental substitution.

## Why the archives must accompany the package

Adrien confirmed that the production files came from USAspending's Award Data
Archive. USAspending regenerates these archives monthly and removes older
versions; the recorded August 6 URLs no longer provide that historical vintage.
The overlap files came from the bulk-download API; their saved request payloads
document the original queries.

A new download from USAspending is a **newer data vintage**, not an exact
replacement for these files. Repeating an old API request does not freeze its
underlying records. A revised vintage may produce different data or estimates
and must be documented and checked separately. The replication loader never
downloads or silently accepts a newer archive in place of the preserved one.

## Verification and use

From the package directory:

```text
python -B raw_rebuild.py verify-usaspending
```

This checks all 18 ZIPs against independently saved historical receipt hashes,
file sizes and ZIP member counts. It also verifies the copied receipts/request
payloads, the immutable source catalog, the bundle manifest, and
`data/raw/USAspending/SHA256SUMS`. It does not extract data, rerun regressions or
claim full raw-to-results certification. Logs go to the enclosing project's
single root `batch_logs/ns_replication_Paul/usaspending_bundle/`.

The separate manifest is `provenance/usaspending_bundle_manifest.json`. Its
expected archive hashes come from the existing historical catalog/receipts,
not from the files being copied. The old catalog, original failed checks,
prepared-data input manifest, and numerical references remain unchanged.

For a fresh raw-build workspace, use the raw environment in
`requirements_raw.txt` and run:

```text
python -B raw_rebuild.py historical-stage --stage usaspending --years 2007-2018
```

The default source is now the verified package `production/` directory. Missing
or altered local files are fatal: there is no fallback to another project or a
web download. An explicit `--usaspending-root <directory>` remains available for
an independently checksum-verified copy of the same production archives; it
does not authorize another vintage. Existing annual build outputs are protected
and are not overwritten by this command.

The one-time preparation command, for the authors who possess the original
project, is `python -B raw_rebuild.py bundle-usaspending --project <project-root>`.
It copies and verifies the raw archives and metadata. Identical completed copies
can be reused; changed originals or targets are rejected without overwriting.
Recipients of the complete bundle do not need the original Dropbox project.

## Public-deposit boundary

This is a local bundle only. No Zenodo/openICPSR record, public upload, DOI, or
data license has been created. Choose the destination, review distribution
terms and attribution, and approve the public deposit separately.

The large ZIPs are excluded from ordinary Git tracking, while the code,
receipts, checksums and manifests can be tracked. A `git archive` alone is
therefore **not** a complete raw-data distribution: deposit the actual
`data/raw/USAspending/` directory as a companion data asset, preserving its
relative paths, and link the permanent deposit in the public README. Do not
publish a code-only archive and describe it as containing these ZIPs.

Bundling this source removes the need for obsolete USAspending URLs. Archive
checks alone do not certify a complete raw-to-results run. That separate
staged verification was completed on October 28, including disaggregation,
decomposition, aggregation, balance and heterogeneity; see
[the verification report](FULL_RAW_VERIFICATION_2026-10-28.md). Public hosting
of the archives remains a separate release step.
