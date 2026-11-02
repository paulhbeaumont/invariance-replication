# Invariance replication: local fiscal multipliers

Replication code and data for the **local fiscal-multiplier application** in
*Zooming In, Zooming Out: Reconciling Treatment Effects across Aggregation Levels*.
This package covers the Nakamura–Steinsson-style empirical exercise, not the
paper's simulations or its other empirical examples. The maintained manuscript
is `1_current_paper/paper/_paper/1_MidpointDraft.tex` in the authors' project.

Repository: https://github.com/paulhbeaumont/invariance-replication

## Start here: obtain the data

Code, source receipts, independent benchmarks and documentation are ordinary
Git files. **Cloning this repository or downloading its source ZIP does not
download the datasets.** The pinned datasets are separate attachments to
[release v1.0.0](https://github.com/paulhbeaumont/invariance-replication/releases/tag/v1.0.0).
No GitHub sign-in or Git LFS is needed to retrieve them.

From the repository folder, use Python 3.9 (the downloader itself needs only
the standard library):

```text
python download_data.py
```

This retrieves the prepared-data attachment (1.91 GB) and installs exactly
100 inputs, checking their original, independently recorded SHA-256 hashes.
It refuses to overwrite different existing inputs. Then install the analysis
requirements and run `python -B replicate.py verify` or `replicate.py all`
as described below.

For the raw-to-results workflow, also retrieve the exact production archives:

```text
python download_data.py --group production
```

The 12 production ZIPs total 7.74 GB. The six diagnostic-only ZIPs (1.62 GB)
are optional: `python download_data.py --group diagnostics`. Use `--group all`
to retrieve all 11.27 GB. Other public raw sources are fetched from their
recorded original/publisher URLs by the raw workflow; no previous Dropbox
checkout is required. The published EFSY panel is the external employment
source, not a rerun of its authors' original Census imputation.

`config/release_assets.json` and `SHA256SUMS.release` pin attachment sizes and
hashes. `python download_data.py --group all --verify-only` checks installed
inputs without network access or writes. Downloads never substitute newer
USAspending vintages. Interrupted `.partial` files are retained for diagnosis;
move them aside before retrying. See [release and attribution notes](docs/PUBLIC_RELEASE.md).

The package reproduces the **current draft's no-population-multiplier specification**, not every historical result in the parent project. The instrument is the newer county historical-intensity × national-spending-shift instrument, not the older product-composition instrument. The principal horizon is **h = 3**.

## Full raw-to-results pipeline - October 28, 2026

The staged clean run is **verified complete**: ten disaggregation columns,
six aggregation specifications, four control specifications, twelve balance
tests and the decomposition reproduce. All 422 final numerical comparisons
pass, with unchanged significance categories. See
[the current verification report](docs/FULL_RAW_VERIFICATION_2026-10-28.md).
The regenerated, visually checked recap is
`output/pdf/NS_results_recap_raw_verified_2026-10-28.pdf`.
The dated development notes below record earlier checkpoints, not additional
missing-source requests.

Use separate Python 3.9 environments: `requirements_raw.txt` for raw producers;
`requirements.txt` for analysis, plus DuckDB installed into an isolated target:

```text
ANALYSIS_PYTHON -m pip install --target .analysis_extra --no-deps -r requirements_analysis_extra.txt
```

With the verified raw archive bundle present, run from this folder with a
**new** identifier (never reuse the identifier of an existing completed run):

```powershell
& .\.venv_raw\Scripts\python.exe -B .\raw_to_results.py all --run-id MY_NEW_RUN --analysis-python 'C:/ProgramData/Anaconda3/python.exe'
```

`all` checks sources, starts an empty derived-data tree, reconstructs annual
and regression inputs, re-estimates the locked specifications, renders the
recap and checks independent saved results. Raw sources may be cached after
checksum verification; old prepared regression inputs are comparison targets,
never construction seeds. The published harmonized EFSY panel is an external
source; this is not a rerun of its authors' Census imputation. Exact historical
USAspending ZIPs are required: today's download is a different vintage.

New data go under `data/raw_rebuild/runs/MY_NEW_RUN/`; logs and certificates go
under the enclosing project's single `batch_logs/ns_replication_Paul/`.
Stata 17 and pdfLaTeX are required for the final exhibits. Kalawao remains
excluded as approved; its known historical audit failure is retained as an
explicit warning, not relabeled as an audit PASS.

## Run from prepared inputs

Use Python 3.9 with the versions in `requirements.txt`. From this folder:

```text
python -B replicate.py verify
python -B replicate.py all
```

`all` verifies inputs, rebuilds the county spending/outcome/IV frame from the included annual working sources, re-estimates the results, compares them to independent pre-existing references, and builds a short results PDF. It needs Stata 17 and pdfLaTeX for figures and PDF rendering; neither Perl nor latexmk is used. Set `STATA_BIN` or pass `--stata "path/to/Stata"` if Stata is not detected. The Python-only stages do not require Stata or LaTeX.

The industry-year regressions use approximately 22.5 million log observations and 36.5 million midpoint observations. Run on a machine with substantial available RAM; 64 GB total RAM is recommended. These regressions can take several minutes. They run sequentially without altering the estimator to reduce memory use.

Separate stages are available:

```text
python -B replicate.py build
python -B replicate.py small
python -B replicate.py cells
python -B replicate.py aggregation
python -B replicate.py heterogeneity
python -B replicate.py balance
python -B replicate.py check
python -B replicate.py render
```

Results are written to `output/`; the recap is `output/pdf/NS_results_recap.pdf`. Generated tables and plot data are under `source/results_recap_v3/` and `output/results_recap_v3/`. Logs go to the enclosing project's single root `batch_logs/ns_replication_Paul/`. After this folder becomes a standalone repository, logs go to its own root `batch_logs/ns_replication_Paul/`.

On Paul's Windows machine, the tested interpreter is `C:/ProgramData/Anaconda3/python.exe`. Its `Library/bin` must be on that process's PATH for the existing NumPy installation. This process-local PowerShell command does not change Windows settings:

```powershell
$env:PATH='C:/ProgramData/Anaconda3;C:/ProgramData/Anaconda3/Library/bin;C:/ProgramData/Anaconda3/Scripts;'+$env:PATH
& 'C:/ProgramData/Anaconda3/python.exe' -B .\replicate.py all
```

## What is reproduced

- Ten-column disaggregation comparison: log growth with full-county lagged employment shares versus midpoint growth with midpoint employment shares, with no population multiplier; h = 3; county, NAICS2, NAICS4 and NAICS6, including industry-year fixed effects.
- The Jensen/weights/entry/exit decomposition. The tiny sample remainder is retained in the numerical accounting but omitted from the waterfall, as in the paper.
- State aggregation: recommended standard-growth weighted specification, unweighted-county and log-growth comparisons, h = 2 and h = 4, and h = 3 component economic areas.
- Separate employment, income-per-resident and manufacturing treatment-heterogeneity adjustments; balance diagnostics for the fixed 1978 characteristics and the 2000 sensitivity. The paper's balance figure displays 1978 characteristics and excludes population.
- Coefficients, clustered standard errors and ordinary first-stage F-statistics. No Sanderson–Windmeijer statistics are added.

`docs/SPECIFICATION.md` describes outcomes, weights, IVs, fixed effects and the eight protocol answers. `docs/DATA_AND_RELEASE.md` documents data and the public-release boundary.

## Replication boundary

### Exact USAspending vintage included locally

`data/raw/USAspending/production/` contains the 12 FY2007-2018 DoD archives
dated August 6, 2026. The six FY2001-2006 overlap archives and their API payloads
are kept separately in `data/raw/USAspending/diagnostics/`; these are diagnostic
only, not production replacements. The 18 unchanged ZIPs total about 9.36 GB,
in addition to the prepared-data bundle. Original receipts, a separate bundle
manifest, and `SHA256SUMS` preserve the independently recorded hashes.

USAspending regenerates its archives and removes older vintages. **A new
download supplies a newer vintage, not the exact files used in the paper.**
The raw-build loader now defaults to these verified package copies and never
silently falls back to another project or a newer download.

```text
python -B raw_rebuild.py verify-usaspending
```

See [the archive instructions](docs/USASPENDING_ARCHIVES.md) for provenance
and construction commands. Large ZIPs are excluded from ordinary Git
tracking and supplied through the versioned release attachments. Their inclusion does
not by itself certify a raw-to-results pipeline; the October 28 report above
documents the separately completed full verification.

### Raw reconstruction extension - October 22, 2026

Paul approved extending this package to a fresh-download, raw-to-results build.
That extension was **completed and verified on October 28**. The existing
`replicate.py all` remains the prepared-data replication described below; it does
not silently switch to a newly reconstructed dataset.

The new, separate entry point is `raw_rebuild.py`. Its checked-in source catalog
records official/publisher URLs, exact checksums and preserved historical
evidence. From this folder:

```text
python -B raw_rebuild.py download --source efsy --source bea_cainc1 --source dcads --source naics_concordances --source bea_areas_2004
python -B raw_rebuild.py build-efsy
python -B raw_rebuild.py parse-early
python -B raw_rebuild.py status
```

Downloads are written only under `data/raw_rebuild/`, which is gitignored. They
never replace the prepared data or references. The EFSY converter and early
parser refuse to overwrite an existing build. Run the numerical stages with the
same tested Python/Anaconda environment as the existing package. The guard tests
use only the standard library: `python -B code/test_raw_sources.py`.

The initial source acquisition has been tested: 218 distinct files, approximately
3.8 GB, covering the catalogued NARA material, the published EFSY panel/readme,
BEA CAINC1, five Census concordances and the historical BEA area definition.
The fresh EFSY panel matches every one of 44,262,620 historical source records,
including employment values exactly. Nineteen early procurement files rebuild
4,557,244 records with unchanged values. Container compression and Arrow
string-offset storage can differ without changing a value.

The old NARA catalog's sizes were inaccurate for 53 objects absent from its
successful-download receipts. Their fresh contents were accepted only after
matching independently saved old raw files byte-for-byte. The original catalog,
failed checks and reconciliation evidence remain preserved; no expected data
values or regression benchmarks were redefined.

At the October 22 checkpoint, later procurement formats, USAspending, shared
historical preparation code and the complete build still needed integration.
Those dependencies are now resolved. See
`docs/RAW_TO_RESULTS.md` for exact progress, developer-only comparison commands
and [the co-author file checklist](docs/FILES_TO_REQUEST_FROM_COAUTHORS.md) for
the historical code request. A source-acquisition PASS alone is not raw-to-results
certification.

October 23 update: the original replay and USAspending source evidence have been
located. Retrieval then required GitHub access, which has since been resolved.
[The recovery record](docs/UPSTREAM_RECOVERY_2026-10-23.md) documents the archive
checks and why the original production download URLs cannot currently be
assumed to support a fresh rebuild. No newer source vintage is substituted.

October 27 update: the pinned historical code is now recovered (all eight
snapshot hashes match), and isolated raw construction is running. See
[the raw-build record](docs/RAW_REBUILD_2026-10-27.md) and `requirements_raw.txt`.
This does not change the prepared-data launcher's inputs or certify the full
raw-to-results chain yet.

The October 27 checkpoint now reproduces every retained annual source51 input
value, both county regressions and all six reported aggregation specifications.
Kalawao, Hawaii (CCG 15005), is excluded because BEA does not report a separate
income/population denominator for it. It was already absent from the original
3,058-county sample. Paul approved retaining this exclusion on October 27.
The historical audit's incorrect zero-procurement assertion remains recorded as
failed; the wrapper now accepts only that specific excluded-unit failure as an
explicit warning, without changing data or specifications. Other audit failures
remain fatal. The remaining disaggregation/balance/heterogeneity rebuild was
completed in the October 28 run above. No paper or frozen reference has
been changed.

`replicate.py` remains the **analysis-ready-data** entry point. The separately
verified `raw_to_results.py` reconstructs its inputs from external source
archives before estimation. Both retain independent benchmarks. All required
inputs are present locally; a Git-only export is not a complete data release.
No numerical stage imports another project's analysis inputs.

The prepared-data entry point does not read raw archives. The original bundle
did not contain them; USAspending archives are now preserved separately under
`data/raw/USAspending/`, while other fresh downloads live in the gitignored
raw-rebuild area. Historical producer51 code has been recovered
at the pinned commit; the full application rebuild is now verified separately.
Constructed snapshots remain the explicit starting point for
`replicate.py`. Archived preparation scripts retain their historical paths and
are provenance, not alternate portable launchers. Do not run them against the
parent project. The archived documentation can describe specifications other
than this package; `docs/SPECIFICATION.md` is authoritative here.

The prepared-data bundle is approximately 1.8 GiB, excluding the 9.36 GB
USAspending bundle, other downloaded raw files and reconstruction scratch
outputs. The public distribution uses checksummed release assets rather than
ordinary Git blobs. Environments, working directories, downloaded rebuild
caches and intermediate outputs are not part of the upload. Historical dated
notes below and in `docs/` record the local development history; the public
release instructions at the top supersede their former hosting status.

## Verification policy

`reference/` contains pre-existing estimates and paper exhibits copied before the new launcher was written. The launcher must match these; it never regenerates them. `provenance/input_manifest.json` records original relative paths, bytes and SHA-256 hashes. Numerical functions in `code/vendor/` are source-extracted and checked against their original function hashes. Changes to the new orchestration cannot redefine the saved benchmark.

The all-stage report is `output/replication_check.json`. Test results, relocation checks and manuscript QA are recorded in the paper sandbox's `CONSISTENCY_REPLICATION_2026-10-20.md`.

The portable fresh-folder test can also be run directly with `python -B code/test_relocation.py`. It creates a new temporary package without copying previous outputs, runs `all`, and tests rejection of a changed copied manifest. It retains the temporary folder and records `output/relocation_check.json`; immutable data can be hard-linked to avoid a second 1.8 GiB copy. This is a portability test, not an independently implemented numerical estimator.
