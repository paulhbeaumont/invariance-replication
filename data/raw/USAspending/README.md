# Exact USAspending source vintage

`production/` contains the 12 FY2007-2018 Department of Defense Award Data
Archive ZIPs dated August 6, 2026, plus the original manifest and receipt.
`diagnostics/` contains the six FY2001-2006 overlap ZIPs dated August 14, 2026,
the original bulk-download API requests/responses, and their receipt.
The diagnostic files do not enter the paper's production source chain.

These are unchanged local copies, not fresh downloads. USAspending refreshes
its archives; a new download supplies a newer vintage and is not a substitute
for these exact files. `SHA256SUMS` lists the original receipt hashes for all
18 ZIPs and nine metadata files.

From the package root, run `python -B raw_rebuild.py verify-usaspending`.
The main manifest is `provenance/usaspending_bundle_manifest.json`; construction
defaults to this directory's production archives and checks their hashes.
See `docs/USASPENDING_ARCHIVES.md` for complete provenance and commands.

Local ZIPs total 9,359,636,428 bytes. They are ignored by ordinary Git tracking
but must accompany the eventual research-data deposit. No public upload has
been performed and full raw-to-results replication is still being verified.
