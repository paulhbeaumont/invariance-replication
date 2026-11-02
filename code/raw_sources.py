"""Checksummed public-source acquisition; never treats prepared caches as raw.

The inventory command imports historical receipts as evidence exactly once.
Downloads thereafter use only the package-owned catalog. A different public
vintage is quarantined, not substituted for a checksum-pinned historical file.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile

from package_io import ROOT, logs, read, require, sha, write

CATALOG = ROOT / 'provenance/raw_source_catalog.json'
STORAGE = ROOT / 'data/raw_rebuild'
RECEIPTS = STORAGE / 'download_receipts.json'
USER_AGENT = 'NS-academic-replication/1.0 (public-data provenance audit)'


def checked_specification(expected):
    path=ROOT/'docs/SPECIFICATION.md'
    if sha(path)==expected:
        return 'EXACT_ORIGINAL_SPECIFICATION'
    # Only the already approved, byte-pinned Kalawao documentation addition
    # may differ. Removing it must recover the ORIGINAL catalog's exact hash.
    raw=path.read_bytes()
    start=raw.find(b'## Sample exclusion: Kalawao\n')
    end=raw.find(b'## County spending and instrument\n',start)
    approved_note='297145954caf75f76f0940239b35661ce136b1ec122e75353beb773704052a0f'
    require(start>=0 and end>start
            and hashlib.sha256(raw[start:end]).hexdigest()==approved_note
            and hashlib.sha256(raw[:start]+raw[end:]).hexdigest()==expected,
            'Locked specification changed')
    return 'ORIGINAL_SPECIFICATION_PLUS_EXACT_APPROVED_KALAWAO_NOTE'


def utc():
    return datetime.now(timezone.utc).isoformat()


def safe_relative(value):
    value = value.replace('\\', '/')
    path = PurePosixPath(value)
    require(not path.is_absolute() and '..' not in path.parts and ':' not in value,
            'Unsafe relative data path: ' + value)
    return path


def raw_path(asset):
    path = STORAGE / 'raw' / asset['source'] / safe_relative(asset['relative_path'])
    require(STORAGE.resolve() in path.resolve().parents, 'Data path escapes raw storage')
    return path


def inventory(project):
    """Freeze historical receipts and source-code hashes, without importing code."""
    require(not CATALOG.exists(), 'Existing source catalog is immutable; use a new version for revisions')
    project = project.resolve()
    spec = ROOT / 'docs/SPECIFICATION.md'
    assets = []
    evidence = []
    receipts = [('dcads', '1_current_paper/_data/source/DCADS/download_receipt.csv'),
                ('bea_cainc1', '1_current_paper/_data/source/BEA/CAINC1/download_receipt.csv'),
                ('naics_concordances', '1_current_paper/_data/source/CensusNAICS/download_receipt.csv'),
                ('bea_areas_2004', '1_current_paper/_data/source/BEA_AREAS/download_receipt.csv')]
    archive = ROOT / 'provenance/raw_source_evidence_20261022'
    archive.mkdir(parents=True, exist_ok=False)
    for source, relative in receipts:
        original = project / relative
        require(original.is_file(), 'Missing historical receipt: ' + relative)
        copy = archive / (source + '_download_receipt.csv')
        shutil.copy2(original, copy)
        require(sha(copy) == sha(original), 'Historical receipt changed during preservation')
        evidence.append(dict(source=source, original=relative, copy=str(copy.relative_to(ROOT)), sha256=sha(copy)))
        with copy.open(encoding='utf-8-sig', newline='') as stream:
            for index, row in enumerate(csv.DictReader(stream), 1):
                digest = row.get('sha256', '')
                require(len(digest) == 64 and all(c in '0123456789abcdef' for c in digest),
                        'Historical receipt lacks a usable checksum: ' + source + ' row ' + str(index))
                url = row['url']
                filename = row.get('path') or row.get('file') or Path(urllib.parse.urlsplit(url).path).name
                asset = dict(id=source + '_' + str(index), source=source, relative_path=str(safe_relative(filename)),
                             url=url, expected_bytes=int(row.get('actual_bytes') or row['bytes']),
                             expected_sha256=digest, checksum_authority='historical_download_receipt',
                             evidence=str(copy.relative_to(ROOT)))
                if source == 'dcads':
                    asset.update(fiscal_year=row['fiscal_year'], kind=row['kind'], series=row['series_label'])
                assets.append(asset)
    # Published release metadata retrieved from the authors' GitHub release API
    # on 2026-10-22. This pins the public source, not a successful value comparison.
    for filename, size, digest in [
        ('efsy_panel_naics.csv.zip', 338501296, 'a5206da0e10ae369bdd688979d12ccf350d00bb3bb9d47ef096d1f31fc4d1957'),
        ('efsy_readme_panel.txt', 881, 'b31721aff481d0baa6a32d8cc8c531382d1f6ec5e0afe8ff4e182cf3178592f8')]:
        assets.append(dict(id='efsy_' + filename, source='efsy', relative_path=filename,
                           url='https://github.com/fpeckert/website/releases/download/cbp-data/' + filename,
                           expected_bytes=size, expected_sha256=digest,
                           checksum_authority='publisher_release_digest',
                           evidence='https://api.github.com/repos/fpeckert/website/releases/tags/cbp-data'))
    snapshot = project / '4_Macro_version/4a_PaperApplication/data/intermediate/county_inputs_20260913/snapshot'
    snapshot_manifest = json.loads((snapshot / 'manifest.json').read_text())
    recovered = []
    for name in ['employment_producer.py', 'population_producer.py', 'geography_producer.py', 'crosswalk.csv']:
        original = snapshot / name
        item = next(r for r in snapshot_manifest['files'] if r['relative_path'] == name)
        require(sha(original) == item['sha256'], 'Recovered producer differs from historical snapshot: ' + name)
        destination = archive / name
        shutil.copy2(original, destination)
        recovered.append(dict(file=str(destination.relative_to(ROOT)), sha256=sha(destination),
                              original=item['source'], role='unmodified_provenance_not_executed'))
    shutil.copy2(snapshot / 'manifest.json', archive / 'county_input_snapshot_manifest.json')
    reference_manifest = read(ROOT / 'provenance/input_manifest.json')
    write(archive / 'prepared_input_manifest.json', reference_manifest)
    write(CATALOG, dict(created_utc=utc(), status='acquisition_catalog_not_raw_to_results_certification',
                        specification_sha256=sha(spec), receipt_evidence=evidence,
                        recovered_producers=recovered, assets=assets,
                        unresolved=['historical producer51 and dcads_common dependency',
                                    'USAspending acquisition/version manifest and parser',
                                    'complete application-owned raw-to-annual source builders',
                                    'raw-to-derived industry/regional/characteristic builders and clean-room test']))
    print('INVENTORY PASS:', len(assets), 'pinned public assets;', len(recovered), 'preserved source/code files', flush=True)
    print('Expected download bytes by source:', flush=True)
    for source in sorted({a['source'] for a in assets}):
        group = [a for a in assets if a['source'] == source]
        print(source, len(group), sum(a['expected_bytes'] for a in group), flush=True)


def download(sources, redownload=False, certificate=None):
    catalog = read(CATALOG)
    specification_validation=checked_specification(catalog['specification_sha256'])
    chosen = [a for a in catalog['assets'] if a['source'] in sources]
    require(chosen and set(sources) <= {a['source'] for a in catalog['assets']}, 'Unknown source selection')
    missing_bytes = sum(a['expected_bytes'] for a in chosen if redownload or not raw_path(a).exists())
    STORAGE.mkdir(parents=True, exist_ok=True)
    require(shutil.disk_usage(STORAGE).free > 2 * missing_bytes + 1024**3, 'Insufficient free disk for safe download')
    records = read(RECEIPTS) if RECEIPTS.exists() else []
    failures = []
    for asset in chosen:
        target = raw_path(asset)
        target.parent.mkdir(parents=True, exist_ok=True)
        record = dict(asset_id=asset['id'], source=asset['source'], started_utc=utc(),
                      original_url=asset['url'], expected_sha256=asset['expected_sha256'],
                      path=str(target.relative_to(ROOT)))
        try:
            if target.exists() and not redownload:
                actual_sha = sha(target)
                digest_expected = asset['expected_sha256']
                prior = next((r for r in reversed(records) if r['asset_id'] == asset['id'] and
                              r.get('status') != 'FAILED'), None)
                require(digest_expected is not None or prior is not None,
                        'Unpinned file has no prior retrieval receipt; do not silently adopt it')
                digest_expected = digest_expected or prior['actual_sha256']
                require(target.stat().st_size == asset['expected_bytes'] and actual_sha == digest_expected,
                        'Existing downloaded input differs; do not overwrite it')
                record.update(status='already_present_verified', actual_bytes=target.stat().st_size,
                              actual_sha256=actual_sha)
            else:
                # Avoid stale cached signed GitHub redirects. Never store the
                # redirect's ephemeral token, and never disable TLS verification.
                original = asset['url']
                candidates = [original]
                if original.startswith('http://'):
                    candidates = ['https://' + original[len('http://'):], original]
                last_error = None
                for attempt in range(3):
                    base = candidates[min(attempt, len(candidates)-1)]
                    requested = base
                    if 'github.com/' in requested:
                        requested += ('&' if '?' in requested else '?') + 'replication_download=' + str(time.time_ns())
                    temporary = target.with_name(target.name + '.partial')
                    try:
                        request = urllib.request.Request(requested, headers={'User-Agent': USER_AGENT, 'Cache-Control': 'no-cache'})
                        print('DOWNLOAD', asset['source'], asset['relative_path'], asset['expected_bytes'], flush=True)
                        digest = hashlib.sha256()
                        total = 0
                        with urllib.request.urlopen(request, timeout=30) as response, temporary.open('wb') as stream:
                            response_url = urllib.parse.urlsplit(response.geturl())
                            record.update(http_status=response.status, last_modified=response.headers.get('Last-Modified'),
                                          etag=response.headers.get('ETag'),
                                          final_url_without_token=urllib.parse.urlunsplit((response_url.scheme,response_url.netloc,response_url.path,'','')))
                            while True:
                                block = response.read(4 * 1024 * 1024)
                                if not block:
                                    break
                                stream.write(block)
                                digest.update(block)
                                previous = total
                                total += len(block)
                                require(total <= asset['expected_bytes'] + 1024**2, 'Response exceeds expected source size')
                                if total // (64*1024**2) != previous // (64*1024**2):
                                    print('  bytes', total, '/', asset['expected_bytes'], flush=True)
                        actual_sha = digest.hexdigest()
                        record.update(actual_bytes=total, actual_sha256=actual_sha)
                        if total != asset['expected_bytes'] or (asset['expected_sha256'] is not None and actual_sha != asset['expected_sha256']):
                            quarantine = temporary.with_name(target.name + '.different_' + actual_sha[:16])
                            if quarantine.exists():
                                quarantine = quarantine.with_name(quarantine.name + '_' + str(time.time_ns()))
                            temporary.rename(quarantine)
                            record['quarantine'] = str(quarantine.relative_to(ROOT))
                            raise ValueError('PUBLIC_SOURCE_VINTAGE_MISMATCH: retained separately; not installed as approved input')
                        temporary.replace(target)
                        record['status'] = ('downloaded_exact_checksum' if asset['expected_sha256'] is not None
                                            else 'first_retrieval_size_verified_no_historical_hash')
                        last_error = None
                        break
                    except ValueError:
                        raise
                    except (urllib.error.URLError, TimeoutError, OSError) as exc:
                        last_error = exc
                        print('  attempt failed:', type(exc).__name__, str(exc), flush=True)
                if last_error is not None:
                    raise last_error
            print('SOURCE PASS', asset['id'], flush=True)
        except Exception as exc:
            record.update(status='FAILED', error=type(exc).__name__ + ': ' + str(exc))
            failures.append(asset['id'])
            print('SOURCE FAILED', asset['id'], str(exc), flush=True)
        record['finished_utc'] = utc()
        records.append(record)
        write(RECEIPTS, records)
    report = dict(status='PASS' if not failures else 'INCOMPLETE', sources=sources, assets=len(chosen),
                  failed=failures, raw_to_results_certified=False, receipt_file=str(RECEIPTS.relative_to(ROOT)),
                  specification_validation=specification_validation)
    write(certificate if certificate is not None else logs() / 'raw_rebuild_20261022/download_check.json', report)
    print(json.dumps(report, indent=2), flush=True)
    require(not failures, 'Some sources failed; do not proceed with unavailable or changed inputs')


def complete_catalog(project):
    """Preserve the full old NARA inventory, including never-receipted archives.

    Original receipts cover only part of the original catalog. The other objects
    have published byte counts but no independently saved content hashes. A new
    download records its own checksum without claiming historical equivalence.
    """
    catalog = read(CATALOG)
    require('full_nara_manifest' not in catalog, 'Full NARA inventory is already preserved')
    archive = ROOT / 'provenance/raw_source_evidence_20261022'
    before = archive / 'receipt_only_catalog.json'
    require(not before.exists(), 'Do not overwrite the receipt-only inventory')
    shutil.copy2(CATALOG, before)
    original = project.resolve() / '1_current_paper/_data/source/DCADS/manifest.csv'
    frozen = archive / 'dcads_full_manifest.csv'
    shutil.copy2(original, frozen)
    require(sha(frozen) == sha(original), 'Full source inventory changed during preservation')
    known = {a['url'] for a in catalog['assets']}
    new = []
    with frozen.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    for index, row in enumerate(rows, 1):
        if row['url'] in known:
            continue
        require(row['url'] and int(row['bytes']) > 0, 'Unusable NARA catalog object: ' + row['filename'])
        new.append(dict(id='dcads_catalog_' + str(index), source='dcads',
                        relative_path=str(safe_relative('objects/' + row['na_id'] + '/' + row['filename'])),
                        url=row['url'], expected_bytes=int(row['bytes']), expected_sha256=None,
                        checksum_authority='historical_catalog_size_only_no_saved_content_hash',
                        evidence=str(frozen.relative_to(ROOT)), fiscal_year=row['fiscal_year'],
                        kind=row['kind'], series=row['series_label']))
    predecessor = project.resolve() / '4_Macro_version/_archive/2026-09_predecessor_acquisition_code'
    for relative in ['py/04_parse_dcads.py','layouts/dcads_272.json']:
        source = predecessor / relative
        dest = archive / Path(relative).name
        require(not dest.exists(), 'Do not overwrite preserved parser/layout evidence')
        shutil.copy2(source, dest)
        catalog['recovered_producers'].append(dict(file=str(dest.relative_to(ROOT)),
                   sha256=sha(dest), original=str(source.relative_to(project.resolve())),
                   role='unmodified_provenance_not_executed'))
    catalog['assets'].extend(new)
    catalog['full_nara_manifest'] = dict(path=str(frozen.relative_to(ROOT)), sha256=sha(frozen),
                        rows=len(rows), additional_unreceipted_objects=len(new),
                        earlier_receipt_only_catalog_sha256=sha(before))
    write(CATALOG, catalog)
    print('FULL CATALOG:',len(rows),'original NARA records;',len(new),'additional objects without historical hashes',flush=True)


def reconcile_catalog(project):
    """Resolve stale catalog sizes using independent PRE-EXISTING raw files.

    A new download is never made its own reference. Only byte-identical matches
    to the old, separately stored raw source can supply the missing historical
    hash. The old catalog/receipts and all failed attempts remain preserved.
    """
    catalog = read(CATALOG)
    candidates = [a for a in catalog['assets'] if a['expected_sha256'] is None]
    require(candidates, 'No unreceipted source records remain to reconcile')
    archive = ROOT / 'provenance/raw_source_evidence_20261022'
    before = archive / 'catalog_before_size_reconciliation.json'
    require(not before.exists(), 'Do not overwrite the pre-reconciliation source catalog')
    shutil.copy2(CATALOG, before)
    receipts = read(RECEIPTS)
    latest = {r['asset_id']:r for r in receipts}
    records = []
    for asset in candidates:
        last = latest.get(asset['id'],{})
        year = asset['fiscal_year'] or 'series'
        original_relative = '1_current_paper/_data/source/DCADS/' + asset['series'] + '/' + year + '/' + Path(asset['relative_path']).name
        original = project.resolve() / original_relative
        fresh = ROOT / safe_relative(last.get('quarantine','__missing__'))
        row = dict(id=asset['id'], historical_catalog_bytes=asset['expected_bytes'],
                   original_raw_file=original_relative, independently_verified=False)
        if original.is_file() and fresh.is_file():
            before_stat = original.stat()
            original_digest = sha(original)
            fresh_digest = sha(fresh)
            row.update(original_raw_bytes=before_stat.st_size,original_raw_sha256=original_digest,
                       original_raw_modified_utc=datetime.fromtimestamp(before_stat.st_mtime,timezone.utc).isoformat(),
                       fresh_bytes=fresh.stat().st_size,fresh_sha256=fresh_digest)
            require(sha(original) == original_digest, 'Independent original changed during comparison')
            if before_stat.st_size == fresh.stat().st_size and original_digest == fresh_digest:
                row['independently_verified'] = True
                asset['historical_catalog_bytes'] = asset['expected_bytes']
                asset['expected_bytes'] = before_stat.st_size
                asset['expected_sha256'] = original_digest
                asset['checksum_authority'] = 'independent_preexisting_raw_file_stale_catalog_size'
                asset['size_reconciliation'] = str((archive/'catalog_size_reconciliation.json').relative_to(ROOT))
                target = raw_path(asset)
                require(not target.exists(), 'Do not overwrite an installed raw input during size reconciliation')
                shutil.copy2(fresh,target)
                require(sha(target) == original_digest, 'Verified source copy differs')
                receipt = dict(last)
                receipt.update(status='reconciled_exact_independent_preexisting_raw_file',
                               expected_sha256=original_digest,finished_utc=utc(),
                               path=str(target.relative_to(ROOT)),error=None,
                               original_catalog_bytes=asset['historical_catalog_bytes'])
                receipts.append(receipt)
        else:
            row['unresolved'] = 'Missing independent original or quarantined fresh download'
        records.append(row)
    report = dict(status='PASS' if all(r['independently_verified'] for r in records) else 'INCOMPLETE',
                  compared=len(records),matching=sum(r['independently_verified'] for r in records),
                  old_catalog_sha256=sha(before),records=records,
                  interpretation='stale catalog byte counts, not revised data, for independently matching files')
    write(archive/'catalog_size_reconciliation.json',report)
    write(CATALOG,catalog)
    write(RECEIPTS,receipts)
    print('CATALOG RECONCILIATION:',report['matching'],'/',report['compared'],'files match independent pre-existing raw bytes.',flush=True)
    require(report['status'] == 'PASS', 'Unresolved source differences remain; do not substitute')


def verified_asset(source, filename):
    asset = next(a for a in read(CATALOG)['assets']
                 if a['source'] == source and a['relative_path'] == filename)
    path = raw_path(asset)
    require(path.is_file() and path.stat().st_size == asset['expected_bytes']
            and sha(path) == asset['expected_sha256'], 'Raw source absent or checksum differs: ' + filename)
    return path


def status():
    """Certify source acquisition only; explicitly retain downstream blockers."""
    catalog = read(CATALOG)
    checked_specification(catalog['specification_sha256'])
    receipts = read(RECEIPTS) if RECEIPTS.exists() else []
    latest = {r['asset_id']:r for r in receipts}
    results = []
    hash_cache = {}
    for asset in catalog['assets']:
        path = raw_path(asset)
        receipt = latest.get(asset['id'])
        digest = None
        if path.is_file():
            if path not in hash_cache:
                hash_cache[path] = sha(path)
            digest = hash_cache[path]
        expected = asset['expected_sha256'] or (receipt or {}).get('actual_sha256')
        good = path.is_file() and path.stat().st_size == asset['expected_bytes'] and digest == expected and expected is not None
        good = good and receipt is not None and receipt.get('status') != 'FAILED'
        results.append(dict(id=asset['id'], source=asset['source'], verified=good,
                            historical_checksum=asset['expected_sha256'] is not None))
    report = dict(status='CATALOG_ACQUISITION_COMPLETE' if all(r['verified'] for r in results) else 'CATALOG_ACQUISITION_INCOMPLETE',
                  source_records=len(results), verified_records=sum(r['verified'] for r in results),
                  records_with_independent_expected_hash=sum(r['historical_checksum'] for r in results),
                  newly_retrieved_without_historical_hash=sum(not r['historical_checksum'] for r in results),
                  distinct_local_files=len(hash_cache), distinct_local_bytes=sum(p.stat().st_size for p in hash_cache),
                  failed_or_missing=[r['id'] for r in results if not r['verified']],
                  raw_to_results_certified=False, remaining_dependencies=catalog['unresolved'])
    comparison = logs() / 'raw_rebuild_20261022/efsy_reference_comparison.json'
    if comparison.exists():
        checked = read(comparison)
        report['efsy_comparison'] = {k:checked[k] for k in ['status','rows_compared',
            'employment_values_not_bitwise_equal','employment_max_abs_difference']}
    write(logs() / 'raw_rebuild_20261022/raw_rebuild_status.json', report)
    print(json.dumps(report, indent=2), flush=True)
    require(not report['failed_or_missing'], 'Raw source acquisition has missing or changed assets')


def build_efsy(output=None, certificate=None):
    """Convert the authors' public CSV without allocation, filtering or recoding.

    This is the source-file conversion, not a rerun of the EFSY imputation and
    not a claim that the application's employment-allocation pipeline is complete.
    """
    import pyarrow as pa
    import pyarrow.csv as pcsv
    import pyarrow.parquet as pq
    source = verified_asset('efsy', 'efsy_panel_naics.csv.zip')
    out = Path(output) if output is not None else STORAGE / 'built/efsy_panel_naics.parquet'
    require(ROOT.resolve() in out.resolve().parents, 'Source conversion output escapes the package')
    require(not out.exists(), 'Do not overwrite a rebuilt source; use the existing checked build')
    out.parent.mkdir(parents=True, exist_ok=True)
    schema = pa.schema([('fipstate', pa.int64()), ('fipscty', pa.int64()),
                        ('naics12', pa.string()), ('emp', pa.float64()),
                        ('year', pa.int64()), ('v1', pa.int64())])
    count = 0
    temporary = out.with_name(out.name + '.partial')
    with zipfile.ZipFile(source) as archive:
        members = [n for n in archive.namelist() if n.endswith('.csv') and not n.startswith('__MACOSX/')]
        require(members == ['efsy_panel_naics.csv'], 'Unexpected EFSY archive contents')
        with archive.open(members[0]) as stream, pq.ParquetWriter(temporary, schema, compression='zstd') as writer:
            reader = pcsv.open_csv(stream, read_options=pcsv.ReadOptions(block_size=8*1024**2),
                                  convert_options=pcsv.ConvertOptions(column_types=schema))
            require(reader.schema.equals(schema), 'Unexpected EFSY source schema')
            for batch in reader:
                writer.write_batch(batch)
                previous = count
                count += batch.num_rows
                if count // 5000000 != previous // 5000000:
                    print('EFSY source conversion:', count, 'rows', flush=True)
    require(count == 44262620, 'Published EFSY source row count differs; not accepted as the historical source')
    temporary.rename(out)
    report = dict(status='SOURCE_CONVERSION_PASS', rows=count, source_sha256=sha(source),
                  output=str(out.relative_to(ROOT)), output_sha256=sha(out),
                  filtering=False, recoding=False, allocation=False, application_raw_to_results_certified=False)
    write(certificate if certificate is not None else logs() / 'raw_rebuild_20261022/efsy_conversion.json', report)
    print(json.dumps(report, indent=2), flush=True)


def compare_efsy(reference):
    """Compare every record to an immutable pre-existing source, in bounded RAM."""
    import numpy as np
    import pyarrow.parquet as pq
    original_manifest = read(ROOT / 'provenance/raw_source_evidence_20261022/county_input_snapshot_manifest.json')
    expected = next(a['sha256'] for a in original_manifest['files'] if a['relative_path'] == 'raw_efsy.parquet')
    require(sha(reference) == expected, 'The comparison reference is not the immutable historical EFSY source')
    rebuilt = STORAGE / 'built/efsy_panel_naics.parquet'
    report_conversion = read(logs() / 'raw_rebuild_20261022/efsy_conversion.json')
    require(sha(rebuilt) == report_conversion['output_sha256'], 'Rebuilt source changed after conversion')
    old = pq.ParquetFile(reference)
    new = pq.ParquetFile(rebuilt)
    require(old.metadata.num_rows == new.metadata.num_rows == 44262620, 'Different EFSY source row counts')
    require(old.schema_arrow.equals(new.schema_arrow, check_metadata=False), 'Different EFSY source columns/types')
    new_batches = iter(new.iter_batches(batch_size=250000))
    current = next(new_batches)
    new_offset = 0
    compared = 0
    exact_emp_differences = 0
    key_differences = 0
    nonclose_emp_differences = 0
    max_abs = 0.
    annual_old = {}
    annual_new = {}
    for old_batch in old.iter_batches(batch_size=250000):
        old_offset = 0
        while old_offset < old_batch.num_rows:
            if new_offset == current.num_rows:
                current = next(new_batches)
                new_offset = 0
            length = min(old_batch.num_rows - old_offset, current.num_rows - new_offset)
            left = old_batch.slice(old_offset, length)
            right = current.slice(new_offset, length)
            for col in ['fipstate','fipscty','naics12','year','v1']:
                if not left.column(col).equals(right.column(col)):
                    key_differences += 1
            a = left.column('emp').to_numpy(zero_copy_only=False)
            b = right.column('emp').to_numpy(zero_copy_only=False)
            exact_emp_differences += int((~((a == b) | (np.isnan(a) & np.isnan(b)))).sum())
            nonclose_emp_differences += int((~np.isclose(a,b,atol=1e-12,rtol=1e-12,equal_nan=True)).sum())
            if np.isfinite(a-b).any():
                max_abs = max(max_abs, float(np.nanmax(np.abs(a-b))))
            years = left.column('year').to_numpy(zero_copy_only=False)
            for year in np.unique(years):
                mask = years == year
                annual_old[int(year)] = annual_old.get(int(year),0.) + float(np.nansum(a[mask]))
                annual_new[int(year)] = annual_new.get(int(year),0.) + float(np.nansum(b[mask]))
            old_offset += length
            new_offset += length
            previous = compared
            compared += length
            if compared // 5000000 != previous // 5000000:
                print('EFSY immutable-reference comparison:', compared, 'rows', flush=True)
    require(compared == old.metadata.num_rows, 'Incomplete EFSY comparison')
    good = key_differences == nonclose_emp_differences == 0
    report = dict(status='PASS' if good else 'DIFFERENT', rows_compared=compared,
                  reference_sha256=expected, rebuilt_sha256=sha(rebuilt),
                  metadata_and_compression_may_differ=True,
                  key_or_order_mismatching_segments=key_differences,
                  employment_values_not_bitwise_equal=exact_emp_differences,
                  employment_values_outside_1e12_tolerance=nonclose_emp_differences,
                  employment_max_abs_difference=max_abs,
                  annual_employment_differences={str(y):annual_new[y]-annual_old[y] for y in sorted(annual_old)},
                  application_raw_to_results_certified=False)
    write(logs() / 'raw_rebuild_20261022/efsy_reference_comparison.json', report)
    print(json.dumps(report, indent=2), flush=True)
    require(good, 'Re-downloaded EFSY source does not reproduce the historical reference; do not publish or substitute')


def parse_early():
    """Run the preserved 272-byte parser on exactly its declared source files.

    No record format is inferred for later years; geography, harmonization,
    construction exclusion and deflation remain separate missing stages.
    """
    catalog = read(CATALOG)
    folder = ROOT / 'provenance/raw_source_evidence_20261022'
    parser_path = folder / '04_parse_dcads.py'
    layout_path = folder / 'dcads_272.json'
    for path in [parser_path, layout_path]:
        item = next(x for x in catalog['recovered_producers']
                    if safe_relative(x['file']) == PurePosixPath(path.relative_to(ROOT).as_posix()))
        require(sha(path) == item['sha256'], 'Preserved parser/layout changed: ' + path.name)
    spec = read(layout_path)
    assets = {Path(a['relative_path']).name:a for a in catalog['assets']
              if a['source'] == 'dcads' and Path(a['relative_path']).name in spec['applies_to']}
    require(set(assets) == set(spec['applies_to']), 'The early-layout source list is incomplete')
    certificate = logs() / 'raw_rebuild_20261022/early_parse.json'
    destination = STORAGE / 'built/dcads_272'
    require(not destination.exists(), 'An early-parser output already exists; do not overwrite it')
    destination.mkdir(parents=True)
    module_spec = importlib.util.spec_from_file_location('preserved_dcads_272_parser', parser_path)
    parser = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(parser)
    records = []
    write(certificate,dict(status='INCOMPLETE', expected_files=len(assets), files=records))
    for name in spec['applies_to']:
        asset = assets[name]
        source = verified_asset('dcads',asset['relative_path'])
        frame = parser.parse_file(source, spec)
        require(len(frame) * spec['record_length'] == source.stat().st_size, 'Raw-record count mismatch')
        target = destination / (name + '.parquet')
        frame.to_parquet(target,index=False)
        amounts = frame.dollars_thousands
        row = dict(source=name, raw_sha256=sha(source), rows=len(frame),
                   dollars_thousands_sum=int(amounts.sum()), missing_amounts=int(amounts.isna().sum()),
                   negative_amounts=int((amounts < 0).sum()),
                   parsed_file=str(target.relative_to(ROOT)), parsed_sha256=sha(target))
        records.append(row)
        print('EARLY PARSE',name,len(frame),'records',flush=True)
        write(certificate,dict(status='INCOMPLETE',expected_files=len(assets),files=records))
    write(certificate,dict(status='EARLY_SOURCE_PARSE_COMPLETE',expected_files=len(assets),
                           parser_sha256=sha(parser_path),layout_sha256=sha(layout_path),
                           rows=sum(x['rows'] for x in records),files=records,
                           application_raw_to_results_certified=False))
    print('EARLY PARSE COMPLETE:',len(records),'source files; later formats and application construction are not certified.',flush=True)


def compare_early(reference):
    """Compare complete parsed records, not just newly generated aggregates."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    certificate = read(logs() / 'raw_rebuild_20261022/early_parse.json')
    require(certificate['status'] == 'EARLY_SOURCE_PARSE_COMPLETE', 'Incomplete early-source parse')
    checked = []
    for item in certificate['files']:
        path = ROOT / safe_relative(item['parsed_file'])
        require(sha(path) == item['parsed_sha256'], 'Rebuilt early file changed')
        original = reference / path.name
        require(original.is_file(), 'Missing independent historical parsed reference: ' + path.name)
        digest = sha(original)
        left = pq.read_table(original)
        right = pq.read_table(path)
        same_rows = left.num_rows == right.num_rows
        same_names = left.column_names == right.column_names
        different = []
        storage_types = []
        for name in left.column_names:
            if name not in right.column_names:
                different.append(name)
                continue
            a, b = left.column(name), right.column(name)
            if a.type != b.type:
                storage_types.append(dict(column=name,old=str(a.type),new=str(b.type)))
            # Arrow string vs large_string differs only in offset-storage width.
            # Compare every character after lossless widening; never coerce a
            # numeric, date, missing value or geography code to make it match.
            string = lambda t:pa.types.is_string(t) or pa.types.is_large_string(t)
            if string(a.type) and string(b.type):
                a,b = a.cast(pa.large_string()), b.cast(pa.large_string())
            if not a.equals(b):
                different.append(name)
        checked.append(dict(source=item['source'],rows=right.num_rows,
                            reference_sha256=digest,matching_rows=same_rows,
                            matching_column_order=same_names,different_columns=different,
                            storage_type_differences=storage_types))
        require(sha(original) == digest, 'Historical reference changed during comparison')
        print('EARLY COMPARE',item['source'],'MATCH' if same_rows and same_names and not different else 'DIFFERENT',flush=True)
    good = all(x['matching_rows'] and x['matching_column_order'] and not x['different_columns'] for x in checked)
    report = dict(status='PASS' if good else 'DIFFERENT', files=len(checked),
                  rows=sum(x['rows'] for x in checked),complete_record_comparisons=checked,
                  application_raw_to_results_certified=False)
    write(logs() / 'raw_rebuild_20261022/early_reference_comparison.json',report)
    print(json.dumps({k:v for k,v in report.items() if k != 'complete_record_comparisons'},indent=2),flush=True)
    require(good, 'The early-source parser does not reproduce the immutable original records')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='stage', required=True)
    inv = sub.add_parser('inventory', help='Preserve the historical receipts and source-code evidence once')
    inv.add_argument('--project', required=True, type=Path)
    full = sub.add_parser('complete-catalog', help='Add source-manifest objects absent from the historical download receipts')
    full.add_argument('--project', required=True, type=Path)
    rec = sub.add_parser('reconcile-catalog', help='Resolve catalog-size discrepancies only against independently saved original raw files')
    rec.add_argument('--project', required=True, type=Path)
    dl = sub.add_parser('download', help='Fetch and verify explicitly selected public sources')
    dl.add_argument('--source', action='append', required=True,
                    choices=['efsy','bea_cainc1','dcads','naics_concordances','bea_areas_2004'])
    dl.add_argument('--redownload', action='store_true')
    sub.add_parser('build-efsy', help='Convert the downloaded public EFSY CSV into a source Parquet; no cell allocation')
    cmp = sub.add_parser('compare-efsy', help='Compare the entire converted source with the original immutable source file')
    cmp.add_argument('--reference', required=True, type=Path)
    sub.add_parser('status', help='Verify all acquired sources and report the still-incomplete raw-to-results boundary')
    sub.add_parser('parse-early', help='Run the unchanged historical 272-byte parser; no guessing for later layouts')
    early = sub.add_parser('compare-early', help='Compare all early parsed records to the independently saved original files')
    early.add_argument('--reference', required=True, type=Path)
    recover = sub.add_parser('recover-upstream-evidence', help='Preserve the recovered source51/USAspending evidence and verify historical archives; no data construction')
    recover.add_argument('--project', required=True, type=Path)
    sub.add_parser('probe-upstream-downloads', help='Check whether recorded historical USAspending URLs respond; do not download or replace data')
    bund = sub.add_parser('bundle-usaspending', help='Copy exact historical production/diagnostic USAspending archives into the local package; no public upload')
    bund.add_argument('--project',required=True,type=Path)
    sub.add_parser('verify-usaspending', help='Verify bundled USAspending archives, receipts and request payloads against historical hashes')
    freeze = sub.add_parser('freeze-historical-code', help='Preserve exact pinned historical Git sources and verify independent hashes')
    freeze.add_argument('--checkout', required=True, type=Path)
    historical = sub.add_parser('historical-stage', help='Run unchanged historical construction in isolated raw-rebuild storage')
    historical.add_argument('--stage',dest='historical_stage',required=True,choices=['population','employment','dcads-fixed','dcads-delimited','dcads-geography','usaspending','county-inputs'])
    historical.add_argument('--years')
    historical.add_argument('--usaspending-root',type=Path,help='Optional explicit archive directory; defaults to the verified package production bundle')
    historical.add_argument('--attempt',type=int,default=1,help='Explicit new attempt number; earlier diagnostics are retained')
    historical.add_argument('--resume-missing',action='store_true',help='Explicitly reuse completed fixed-width fiscal files and parse only missing files')
    args = parser.parse_args()
    if args.stage == 'inventory':
        inventory(args.project)
    elif args.stage == 'complete-catalog':
        complete_catalog(args.project)
    elif args.stage == 'reconcile-catalog':
        reconcile_catalog(args.project)
    elif args.stage == 'download':
        download(args.source, args.redownload)
    elif args.stage == 'build-efsy':
        build_efsy()
    elif args.stage == 'status':
        status()
    elif args.stage == 'parse-early':
        parse_early()
    elif args.stage == 'compare-early':
        compare_early(args.reference)
    elif args.stage == 'recover-upstream-evidence':
        from recover_upstream_evidence import recover as recover_evidence
        recover_evidence(args.project)
    elif args.stage == 'probe-upstream-downloads':
        from recover_upstream_evidence import probe_downloads
        probe_downloads()
    elif args.stage == 'bundle-usaspending':
        from usaspending_bundle import bundle
        bundle(args.project)
    elif args.stage == 'verify-usaspending':
        from usaspending_bundle import verify as verify_bundle
        verify_bundle()
    elif args.stage == 'freeze-historical-code':
        from historical_rebuild import freeze as freeze_code
        freeze_code(args.checkout)
    elif args.stage == 'historical-stage':
        from historical_rebuild import stage as historical_stage
        historical_stage(args.historical_stage,args.years,args.usaspending_root,args.attempt,args.resume_missing)
    else:
        compare_efsy(args.reference)


if __name__ == '__main__':
    main()
