"""Preserve recovered historical provenance and audit the original raw archives.

This stage reads explicitly supplied historical sources, never writes to them,
and does not build or substitute any regression input. Its checksums come from
independent original receipts, not from the archive being checked.
"""
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

from package_io import ROOT, logs, read, require, sha, write

EVIDENCE = ROOT / 'provenance/upstream_recovery_20261023'
CATALOG = ROOT / 'provenance/usaspending_source_catalog_20261023.json'
REPORT = 'raw_rebuild_20261023/upstream_evidence_check.json'
REPLAY = 'batch_logs/macro_paper_application/source51_replay_20160914'
PINNED_COMMIT = 'fc9df562037324ecfeb1cae1d34de8b13d408ce5'


def relative_path(value):
    """Reject absolute paths, traversal and Windows drive/ADS names."""
    value = str(value).replace('\\', '/')
    parts = value.split('/')
    require(value and not value.startswith('/') and ':' not in value and
            all(p not in ('', '.', '..') for p in parts), 'Unsafe evidence path')
    return Path(*parts)


def preserve(project, relative, destination):
    original = project / relative_path(relative)
    target = EVIDENCE / relative_path(destination)
    require(original.is_file(), 'Missing historical evidence: ' + relative)
    digest = sha(original)
    if target.exists():
        require(sha(target) == digest, 'Historical evidence changed; refusing overwrite: ' + destination)
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, target)
    require(sha(target) == digest and sha(original) == digest,
            'Evidence changed while being preserved: ' + relative)
    return dict(original=relative, preserved=target.relative_to(ROOT).as_posix(), sha256=digest)


def verify_archive(project, asset):
    path = project / relative_path(asset['original_relative_path'])
    require(path.is_file(), 'Missing historical archive: ' + asset['original_relative_path'])
    actual_bytes = path.stat().st_size
    require(actual_bytes == asset['expected_bytes'], 'Historical archive size differs: ' + path.name)
    actual_sha = sha(path)
    require(actual_sha == asset['expected_sha256'], 'Historical archive checksum differs: ' + path.name)
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        require(len(members) == asset['expected_zip_members'], 'Archive member count differs: ' + path.name)
        member_names = [m.filename for m in members]
    # SHA-256 covers the whole compressed file. This is not a fresh download,
    # a record-by-record parser audit, or a full decompression/CRC test.
    return dict(asset_id=asset['id'], status='EXACT_HISTORICAL_ARCHIVE',
                bytes=actual_bytes, sha256=actual_sha, members=member_names,
                role=asset['role'], reacquired_from_web=False)


def probe_downloads():
    """Check historical URLs without downloading or accepting another vintage."""
    catalog = read(CATALOG)
    require(sha(ROOT / 'docs/SPECIFICATION.md') == catalog['specification_sha256'],
            'Locked specification changed')
    for record in catalog['evidence']:
        require(sha(ROOT / relative_path(record['preserved'])) == record['sha256'],
                'Preserved historical source evidence changed')

    def probe(asset):
        request = urllib.request.Request(asset['url'], headers={
            'User-Agent': 'NS-academic-replication/1.0 (historical-source availability audit)',
            'Range': 'bytes=0-0', 'Cache-Control': 'no-cache'})
        result = dict(asset_id=asset['id'], original_url=asset['url'],
                      role=asset['role'], method='GET', range='bytes=0-0',
                      content_downloaded_or_checksum_verified=False)
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                final = urllib.parse.urlsplit(response.geturl())
                result.update(http_status=response.status,
                              final_url_without_query=urllib.parse.urlunsplit((final.scheme,final.netloc,final.path,'','')),
                              status='URL_RESPONDED_NOT_CONTENT_VERIFIED')
                # Close without reading an entire archive if Range was ignored.
        except urllib.error.HTTPError as exc:
            result.update(http_status=exc.code, status='HTTP_ERROR')
            exc.close()
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            result.update(status='CONNECTION_ERROR', error_type=type(exc).__name__)
        return result

    rows = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(probe, asset) for asset in catalog['assets']]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print('HISTORICAL URL', row['asset_id'], row.get('http_status', row['status']), flush=True)
    report = dict(checked_utc=datetime.now(timezone.utc).isoformat(), catalog_sha256=sha(CATALOG),
                  purpose='historical URL availability only; no new data accepted',
                  raw_to_results_certified=False, files=sorted(rows,key=lambda r:r['asset_id']))
    write(logs() / 'raw_rebuild_20261023/historical_url_availability.json', report)
    print('Historical URLs checked:', len(rows), '; unavailable/error:',
          sum(r['status'] != 'URL_RESPONDED_NOT_CONTENT_VERIFIED' for r in rows), flush=True)


def recover(project):
    project = project.resolve()
    require((project / 'INVARIANCE_TEST_PROTOCOL.md').is_file(), 'Not the declared research-project root')
    evidence = []
    declarations = [
        (REPLAY + '/manifest.json', 'source51_replay/manifest.json'),
        (REPLAY + '/replay_2016.csv', 'source51_replay/replay_2016.csv'),
        (REPLAY + '/replay_2018.csv', 'source51_replay/replay_2018.csv'),
        (REPLAY + '/saved_reference.csv', 'source51_replay/saved_reference.csv'),
        ('_data/source/USAspending/manifest.csv', 'usaspending/manifest.csv'),
        ('_data/source/USAspending/download_receipt.csv', 'usaspending/download_receipt.csv'),
        ('_data/source/USAspending/overlap/download_receipt.json', 'usaspending/overlap/download_receipt.json')]
    declarations += [('_data/source/USAspending/overlap/fy%d/overlap_request.json' % y,
                      'usaspending/overlap/fy%d/overlap_request.json' % y) for y in range(2001, 2007)]
    for original, target in declarations:
        evidence.append(preserve(project, original, target))

    snapshot = read(ROOT / 'provenance/original_4a/config/source_snapshot.json')
    require(snapshot['git_commit'] == PINNED_COMMIT, 'Pinned historical source commit changed')
    replay = read(EVIDENCE / 'source51_replay/manifest.json')
    baseline = read(ROOT / 'reference/upstream_source_manifest.json')
    require(replay['input_cache_sha256'] == baseline['inputs'], 'Replay input hashes differ from frozen inputs')
    source_checks = []
    for relative, digest in snapshot['files'].items():
        suffix = '/4_Macro_version/' + relative
        matches = [(name, value) for name, value in replay['source_sha256'].items() if name.endswith(suffix)]
        if matches:
            require(len(matches) == 1 and matches[0][1] == digest, 'Historical source anchors disagree: ' + relative)
            source_checks.append(dict(file='4_Macro_version/' + relative, sha256=digest,
                                      status='INDEPENDENT_MANIFESTS_AGREE_NOT_CODE_FETCHED'))
    require(any(x['file'].endswith('/51_county_multiplier_grid.py') for x in source_checks),
            'Replay lacks the source51 builder anchor')
    require(any(x['file'].endswith('/dcads_common.py') for x in source_checks),
            'Replay lacks the common-module anchor')

    with (EVIDENCE / 'usaspending/manifest.csv').open(encoding='utf-8-sig', newline='') as stream:
        manifest = list(csv.DictReader(stream))
    with (EVIDENCE / 'usaspending/download_receipt.csv').open(encoding='utf-8-sig', newline='') as stream:
        receipts = list(csv.DictReader(stream))
    require(len(manifest) == len(receipts) == 12, 'Unexpected production archive count')
    require({int(r['fiscal_year']) for r in receipts} == set(range(2007, 2019)),
            'Production archive years differ')
    assets = []
    for receipt in receipts:
        year = int(receipt['fiscal_year'])
        rows = [r for r in manifest if int(r['fiscal_year']) == year]
        require(len(rows) == 1, 'Ambiguous production archive declaration')
        row = rows[0]
        require(receipt['file_name'] == row['file_name'] and receipt['url'] == row['url'],
                'Production receipt/manifest disagree')
        require(row['updated_date'] == '2026-08-06', 'Unexpected production archive vintage')
        name = relative_path(receipt['file_name']).as_posix()
        require('/' not in name, 'Production filename is not a basename')
        assets.append(dict(id='usaspending_fy%d' % year, fiscal_year=year,
                           role='production_fy2007_2018', vintage='2026-08-06', url=receipt['url'],
                           original_relative_path='_data/source/USAspending/' + name,
                           expected_bytes=int(receipt['bytes']), expected_sha256=receipt['sha256'],
                           expected_zip_members=int(receipt['zip_members']),
                           checksum_authority='preserved_historical_download_receipt'))
    overlaps = read(EVIDENCE / 'usaspending/overlap/download_receipt.json')
    require(len(overlaps) == 6 and {int(r['fiscal_year']) for r in overlaps} == set(range(2001, 2007)),
            'Unexpected diagnostic overlap years')
    for receipt in overlaps:
        year = int(receipt['fiscal_year'])
        request = read(EVIDENCE / ('usaspending/overlap/fy%d/overlap_request.json' % year))
        require(request['payload'] == receipt['request'], 'Historical overlap requests disagree')
        require(request['response']['file_name'] == receipt['file_name'], 'Overlap filenames disagree')
        name = relative_path(receipt['file_name']).as_posix()
        require('/' not in name, 'Overlap filename is not a basename')
        assets.append(dict(id='usaspending_overlap_fy%d' % year, fiscal_year=year,
                           role='diagnostic_only_not_production', vintage='2026-08-14',
                           url=request['response']['file_url'],
                           original_relative_path='_data/source/USAspending/overlap/fy%d/%s' % (year, name),
                           expected_bytes=int(receipt['bytes']), expected_sha256=receipt['sha256'],
                           expected_zip_members=int(receipt['zip_members']),
                           checksum_authority='preserved_historical_overlap_receipt'))
    catalog = dict(pinned_code_commit=PINNED_COMMIT,
                   specification_sha256=sha(ROOT / 'docs/SPECIFICATION.md'),
                   boundary='historical_source_recovery_not_raw_to_results',
                   assets=assets, evidence=evidence)
    if CATALOG.exists():
        require(read(CATALOG) == catalog, 'Existing recovery catalog differs; refusing overwrite')
    else:
        write(CATALOG, catalog)

    report = dict(status='INCOMPLETE', checked_utc=datetime.now(timezone.utc).isoformat(),
                  catalog_sha256=sha(CATALOG), pinned_code_commit=PINNED_COMMIT,
                  code_retrieved=False, raw_to_results_certified=False,
                  replay_input_hashes_match_frozen_inputs=True, source_anchor_checks=source_checks,
                  original_replay_status=replay['status'], original_replay_reference_checks=replay['reference_checks'],
                  historical_replay_comparison='h=2, roll10; not the current h=3, roll5_gap5 paper specification',
                  original_replay_reference_is_paper_result=replay['reference_is_paper_result'],
                  files=[])
    certificate = logs() / REPORT
    write(certificate, report)
    for asset in assets:
        print('VERIFY HISTORICAL ARCHIVE', asset['id'], asset['expected_bytes'], flush=True)
        try:
            result = verify_archive(project, asset)
        except Exception as exc:
            result = dict(asset_id=asset['id'], status='FAILED', error=str(exc))
        report['files'].append(result)
        write(certificate, report)
        print(' ', result['status'], flush=True)
    good = all(r['status'] == 'EXACT_HISTORICAL_ARCHIVE' for r in report['files'])
    report.update(status='HISTORICAL_ARCHIVES_VERIFIED_CODE_PENDING' if good else 'ARCHIVE_DIFFERENCE',
                  archives=len(assets), bytes=sum(a['expected_bytes'] for a in assets),
                  specification_unchanged=sha(ROOT / 'docs/SPECIFICATION.md') == catalog['specification_sha256'])
    write(certificate, report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('files','source_anchor_checks')}, indent=2), flush=True)
    require(good, 'At least one historical archive differs; do not substitute it into the analysis')
