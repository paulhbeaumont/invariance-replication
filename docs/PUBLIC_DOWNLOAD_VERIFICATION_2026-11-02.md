# Public release download verification - November 2, 2026

Status: **PASS**.

Repository: https://github.com/paulhbeaumont/invariance-replication

Release: https://github.com/paulhbeaumont/invariance-replication/releases/tag/v1.0.0

Release source commit: `7cd2aae0e0d8b1c25048019fe002ee7aeedc55d8`.

## Checks performed

1. A fresh public checkout was cloned with Git's credential helper disabled.
   It contains 479 ordinary repository files and no virtual environment,
   large data blobs, reconstruction cache or previous generated outputs.
2. All 297 pinned historical source-code hashes match in that checkout.
   The five downloader I/O/overwrite/path-safety tests pass.
3. All 22 release attachments match GitHub's server-side SHA-256 digests and
   sizes: 19 data attachments, the attachment manifest, checksum list and
   the previously verified recap PDF. The 19 data attachments total
   11,266,241,859 bytes; all attachments total 11,266,654,769 bytes.
4. Running the public `python download_data.py --group all` in the initially
   data-empty checkout downloaded every dataset from the published release,
   without an authorization header or sign-in. The downloader verified the
   attachment hashes and installed the 100 prepared files against their
   independently recorded original hashes. No local data were copied or
   hard-linked into this checkout.
5. `python -B replicate.py verify` passes: all 184 frozen data/source/reference
   hashes match and the extracted numerical functions are unchanged.
6. `python -B raw_rebuild.py verify-usaspending` passes: every production and
   diagnostic archive, receipt, request payload and manifest matches the
   independent historical source evidence.

The twelve production archives enter the raw construction. The six overlap
archives do not enter regressions, but the unchanged complete raw-source audit
verifies them; therefore the complete raw-chain command requires `--group all`.

## What this check does not claim

This is a fresh **public transport, installation and input-integrity check**,
not a new November 2 rerun of the multi-stage econometric pipeline. The separate
October 28 staged raw-to-results verification remains the numerical evidence:
all 422 reported comparisons pass, with the documented source boundary,
roundoff and Kalawao exclusion. Its benchmarks were not regenerated.

The publication changed only packaging, data-download support, documentation
and the manuscript repository-link footnote. No outcome, instrument, growth
rate, denominator, weight, sample restriction or estimator changed. The
coauthors' upstream repository was not modified.

Author-side transport, build and verification logs are retained in the parent
project's single `batch_logs/ns_replication_Paul/github_publication_20261102/`
directory. These machine-specific logs are not necessary to run the public
package. A publication certificate is also retained under `provenance/`.
