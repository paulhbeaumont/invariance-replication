# Public distribution - November 2, 2026

Owner: Paul Beaumont. Repository:
https://github.com/paulhbeaumont/invariance-replication

This release covers only the local fiscal-multiplier empirical application.
It does not purport to replicate the other applications or all simulations
in the paper. It preserves the approved specification, raw-source boundary,
184 independent file checks and the October 28 raw-to-results verification.
No estimator, outcome, shock, growth rate, weight, denominator or benchmark
was changed for publication.

## Distribution

- Ordinary Git: code, configuration, independent references, pinned historical
  code, original source receipts, checksums and documentation.
- Release `v1.0.0`: one prepared-data ZIP (100 original inputs), twelve exact
  FY2007-2018 production USAspending ZIPs, six optional FY2001-2006 diagnostic
  ZIPs, and the verified five-page recap PDF. The diagnostics do not enter
  production. Every asset is smaller than 2 GiB.
- Download `prepared` for the prepared-data workflow; add `production` for
  the complete raw-chain workflow. Other public raw sources are downloaded
  from the recorded original/publisher endpoints by the raw-source tools.
- `config/release_assets.json` pins data attachment hashes. Existing source
  manifests, saved estimates and independent expected hashes are unchanged.
  Historical `public_deposit_url: null` fields remain as dated evidence, not
  current hosting instructions.
- Virtual environments, installed binary dependencies, the original project,
  manuscript drafts and reconstruction scratch are deliberately not uploaded.

Use `python download_data.py`, then the requirements and commands in README.
The independent October 28 certificates are preserved byte-for-byte under
`provenance/verification_20261028/`; absolute paths inside those dated
certificates describe the authors' completed run, not runtime dependencies.
The historical report is not relabeled as a new, uninterrupted public run.

## Sources, attribution and terms

The package supplies numerical research data and source code; it does not
claim ownership over upstream data or apply one blanket license to everything.
Do not infer that a GitHub publication assigns a new license to author-owned
code, a published imputation, a government source or a third-party program.
No additional author-code license has been selected by this release.

| Component | Attribution and source terms |
| --- | --- |
| Military procurement | U.S. Department of Defense, Defense Contract Action Data System through the National Archives; later years through U.S. Treasury USAspending. The exact public archive vintages and receipts are retained. Federal-agency materials are generally public domain; see [NARA's notice](https://www.archives.gov/global-pages/privacy.html). The release adds no data license or agency endorsement. |
| Employment | U.S. Census Bureau County Business Patterns, using the published imputed/harmonized panel by Fabian Eckert, Teresa C. Fort, Peter K. Schott and Natalie J. Yang, *Imputing Missing Values in the US Census Bureau's County Business Patterns* (2021). [Publisher data page](https://www.fpeckert.me/cbp/), [Census citation policy](https://www.census.gov/about/policies/citation.html). The original EFSY download/readme are fetched from the authors' site rather than relicensed here; the prepared attachment contains the application's constructed numerical inputs. Credit EFSY as well as Census. Their page/readme do not specify a separate data license, so no CC0 or other license is imputed to their work. |
| Population, income and geography | Source: U.S. Bureau of Economic Analysis. BEA permits reproduction of its public-domain information unless stated otherwise and requests source credit: [BEA policy](https://www.bea.gov/help/faq/147). Constructed geographic crosswalks and variables are documented in the source catalog. |
| NAICS concordances | U.S. Census Bureau; original source URLs and hashes are recorded in `provenance/raw_source_catalog.json`. |
| Historical preparation and estimation code | Preserved author code from commit `fc9df562037324ecfeb1cae1d34de8b13d408ce5` of `thetfplab/aggregation-disaggregation-invariance`, plus the original application/sandbox routines. Headers and immutable source hashes are retained. Historical machine paths in archived code are provenance, not credentials or runtime dependencies. |
| `code/stata_dependencies/coefplot.ado` | Ben Jann, coefplot 1.8.6 (22 February 2023), unchanged. Distributed under GPL v3; see the adjacent `COPYING` and [SSC/RePEc entry](https://ideas.repec.org/c/boc/bocode/s457686.html). No Stata executable or other proprietary software is supplied. |

The package contains no restricted French credit-register, Compustat, Panjiva
or other licensed microdata from the wider project. Public government award
records are preserved as published; derived datasets describe county/industry
aggregates. This attribution inventory is not a legal warranty about all
possible downstream uses or jurisdictions.

## Verification boundary

The raw-to-results verification remains the staged clean run documented in
`FULL_RAW_VERIFICATION_2026-10-28.md`: 422 reported numerical comparisons pass,
with unchanged significance categories and explicitly documented floating-point
differences. Kalawao remains excluded, and the narrow original audit exception
remains visible. Publication adds download-integrity and relocation checks;
these do not replace the independent numerical benchmarks.
