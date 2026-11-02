"""Preserve the exact historical USAspending vintage inside the local package.

No network calls, publication, extraction, data construction or source updates.
Expected hashes come from the independently preserved historical receipts, not
from the files being copied. The older source catalog remains immutable.
"""
from datetime import datetime, timezone
from pathlib import Path
import shutil
import time
import uuid

from package_io import ROOT, logs, read, require, sha, write
from recover_upstream_evidence import PINNED_COMMIT, relative_path, verify_archive

CATALOG = ROOT/'provenance/usaspending_source_catalog_20261023.json'
CATALOG_SHA256 = '0fe55cb79dc4d5d9a435fdbbdbf335e1a55f0832d792f8706e1b56c9249bed7f'
BUNDLE = ROOT/'data/raw/USAspending'
MANIFEST = ROOT/'provenance/usaspending_bundle_manifest.json'
PRODUCTION = 'production_fy2007_2018'
DIAGNOSTIC = 'diagnostic_only_not_production'


def checked_catalog():
    require(sha(CATALOG)==CATALOG_SHA256, 'Historical USAspending catalog changed')
    catalog = read(CATALOG)
    require(catalog['pinned_code_commit']==PINNED_COMMIT, 'Historical source commit changed')
    assets = catalog['assets']
    require(len(assets)==18 and len({a['id'] for a in assets})==18, 'Unexpected archive inventory')
    for role, years in [(PRODUCTION,set(range(2007,2019))), (DIAGNOSTIC,set(range(2001,2007)))]:
        selected = [a for a in assets if a['role']==role]
        require(len(selected)==len(years) and {a['fiscal_year'] for a in selected}==years,
                'Archive roles or years differ from the historical receipts')
    for item in catalog['evidence']:
        require(sha(ROOT/relative_path(item['preserved']))==item['sha256'],
                'Preserved historical receipt/evidence changed')
    return catalog


def archive_relative(asset):
    original = relative_path(asset['original_relative_path'])
    if asset['role']==PRODUCTION:
        require(asset['fiscal_year'] in range(2007,2019) and asset['vintage']=='2026-08-06',
                'Not the production USAspending vintage')
        require(original.name=='FY%d_097_Contracts_Full_20260806.zip'%asset['fiscal_year'],
                'Unexpected production archive filename')
        return Path('production')/original.name
    require(asset['role']==DIAGNOSTIC and asset['fiscal_year'] in range(2001,2007),
            'Unknown archive role; diagnostic data cannot enter production')
    return Path('diagnostics')/('fy%d'%asset['fiscal_year'])/original.name


def metadata(catalog):
    rows = []
    prefix = 'provenance/upstream_recovery_20261023/usaspending/'
    for item in catalog['evidence']:
        if not item['preserved'].startswith(prefix):
            continue
        tail = item['preserved'][len(prefix):]
        relative = Path('diagnostics')/relative_path(tail[len('overlap/'):]) if tail.startswith('overlap/') else Path('production')/relative_path(tail)
        source = ROOT/relative_path(item['preserved'])
        rows.append(dict(path=relative.as_posix(),preserved_evidence=item['preserved'],
                         sha256=item['sha256'],bytes=source.stat().st_size))
    require(len(rows)==9, 'Expected two production receipts, one overlap receipt and six request payloads')
    return rows


def expected_manifest(catalog):
    return dict(version=1,source_catalog=CATALOG.relative_to(ROOT).as_posix(),
                source_catalog_sha256=CATALOG_SHA256,pinned_code_commit=PINNED_COMMIT,
                raw_to_results_certified=False,public_deposit_url=None,
                archives=[dict(id=a['id'],fiscal_year=a['fiscal_year'],role=a['role'],
                    vintage=a['vintage'],path=archive_relative(a).as_posix(),
                    bytes=a['expected_bytes'],sha256=a['expected_sha256'],
                    zip_members=a['expected_zip_members'],original_url=a['url']) for a in catalog['assets']],
                metadata=metadata(catalog))


def bundle_path(relative):
    path = BUNDLE/relative_path(relative)
    require(BUNDLE.resolve() in path.resolve().parents, 'Bundle target escapes its dedicated directory')
    return path


def checksum_text(manifest):
    rows = manifest['archives']+manifest['metadata']
    return ''.join(row['sha256']+'  '+row['path']+'\n' for row in sorted(rows,key=lambda r:r['path']))


def check_metadata(catalog):
    expected = expected_manifest(catalog)
    require(MANIFEST.is_file() and read(MANIFEST)==expected, 'Missing or changed local bundle manifest; run bundle-usaspending')
    for row in expected['metadata']:
        path = bundle_path(row['path'])
        require(path.is_file() and path.stat().st_size==row['bytes'] and sha(path)==row['sha256'],
                'Missing or changed bundled receipt/request: '+row['path'])
    checksums = bundle_path('SHA256SUMS')
    require(checksums.is_file() and checksums.read_text(encoding='utf-8')==checksum_text(expected),
            'Missing or changed bundled SHA256SUMS')
    return expected


def verify_file(path, asset):
    # Reuse the prior verifier, with only its explicitly selected file path
    # redirected. It checks original receipt size, SHA-256 and ZIP member count.
    return verify_archive(path.parent,dict(asset,original_relative_path=path.name))


def production_archives(years, override=None):
    """Resolve production years only; no external-directory or web fallback."""
    years = set(years)
    require(years and years<=set(range(2007,2019)), 'Only FY2007-2018 are production USAspending inputs')
    catalog = checked_catalog()
    if override is None:
        check_metadata(catalog)
    folder = Path(override).resolve() if override is not None else bundle_path('production').resolve()
    output = []
    for asset in catalog['assets']:
        if asset['role']!=PRODUCTION or asset['fiscal_year'] not in years:
            continue
        path = folder/Path(asset['original_relative_path']).name
        require(folder in path.resolve().parents, 'Production archive escapes the selected folder')
        verify_file(path,asset)
        output.append((asset,path))
    require(len(output)==len(years), 'Incomplete production archive selection')
    return output


def publish_partial(partial, target):
    require(all(BUNDLE.resolve() in p.resolve().parents for p in (partial,target)),
            'Bundle file move escapes the dedicated directory')
    for attempt in range(81):
        require(not target.exists(), 'Bundle target appeared during copying; partial file retained')
        try:
            partial.rename(target)
            return
        except PermissionError as exc:
            if getattr(exc,'winerror',None) not in (5,32,33) or attempt==80:
                raise
            if attempt==0:
                print('  Transient Windows file lock; retrying the same verified file move.',flush=True)
            time.sleep(0.25)


def copy_checked(source, target, expected_sha, expected_bytes):
    require(source.is_file() and source.stat().st_size==expected_bytes and sha(source)==expected_sha,
            'Original source differs from historical receipt: '+source.name)
    if target.exists():
        require(target.stat().st_size==expected_bytes and sha(target)==expected_sha,
                'Existing bundled file differs; refusing overwrite: '+target.name)
        return 'REUSED_IDENTICAL_COPY'
    target.parent.mkdir(parents=True,exist_ok=True)
    existing = list(target.parent.glob(target.name+'.*.partial'))
    require(len(existing)<=1, 'Multiple prior partial copies; preserve them and investigate')
    partial = existing[0] if existing else target.with_name(target.name+'.'+uuid.uuid4().hex+'.partial')
    if not existing:
        shutil.copy2(source,partial)
    require(partial.stat().st_size==expected_bytes and sha(partial)==expected_sha,
            'Copied bytes differ; partial file retained for diagnosis')
    require(sha(source)==expected_sha, 'Original source changed during copying')
    publish_partial(partial,target)
    return 'COMPLETED_HASH_VERIFIED_PARTIAL' if existing else 'COPIED_AND_HASH_VERIFIED'


def report_path(action):
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    return logs()/'usaspending_bundle'/(action+'_'+stamp+'.json')


def bundle(project):
    project = project.resolve()
    require((project/'INVARIANCE_TEST_PROTOCOL.md').is_file(), 'Provide the explicit original research-project root')
    catalog = checked_catalog()
    expected = expected_manifest(catalog)
    if MANIFEST.exists():
        require(read(MANIFEST)==expected, 'Existing bundle manifest changed; refusing overwrite')
    missing_bytes = sum(a['expected_bytes'] for a in catalog['assets'] if not bundle_path(archive_relative(a).as_posix()).exists())
    require(shutil.disk_usage(ROOT).free>=missing_bytes+64*1024*1024, 'Insufficient free space for independent archive copies')
    certificate = report_path('copy')
    record = dict(status='RUNNING',public_upload_performed=False,raw_to_results_certified=False,
                  source_catalog_sha256=CATALOG_SHA256,files=[])
    write(certificate,record)
    try:
        for asset in catalog['assets']:
            source = project/relative_path(asset['original_relative_path'])
            require(project in source.resolve().parents, 'Original archive escapes declared project')
            target = bundle_path(archive_relative(asset).as_posix())
            print('BUNDLE',asset['id'],asset['expected_bytes'],'bytes',flush=True)
            status = copy_checked(source,target,asset['expected_sha256'],asset['expected_bytes'])
            verified = verify_file(target,asset)
            record['files'].append(dict(verified,copy_status=status,path=target.relative_to(ROOT).as_posix()))
            write(certificate,record)
            print(' ',status,flush=True)
        for row in expected['metadata']:
            copy_checked(ROOT/relative_path(row['preserved_evidence']),bundle_path(row['path']),row['sha256'],row['bytes'])
        if not MANIFEST.exists():
            write(MANIFEST,expected)
        sums = bundle_path('SHA256SUMS')
        text = checksum_text(expected)
        if sums.exists():
            require(sums.read_text(encoding='utf-8')==text, 'Existing SHA256SUMS changed; refusing overwrite')
        else:
            with sums.open('w',encoding='utf-8',newline='\n') as stream:
                stream.write(text)
        check_metadata(catalog)
        record.update(status='LOCAL_BUNDLE_VERIFIED',archives=len(catalog['assets']),
                      archive_bytes=sum(a['expected_bytes'] for a in catalog['assets']),
                      production_archives=12,diagnostic_archives=6,metadata_files=9,
                      manifest_sha256=sha(MANIFEST),original_archives_not_modified=True)
    except Exception as exc:
        record.update(status='FAILED',error_type=type(exc).__name__,error=str(exc))
        raise
    finally:
        write(certificate,record)
    print('LOCAL BUNDLE PASS: 18 archives, 9 metadata files; no public upload or numerical changes.',flush=True)


def verify():
    catalog = checked_catalog()
    check_metadata(catalog)
    certificate = report_path('verify')
    record = dict(status='RUNNING',public_upload_performed=False,raw_to_results_certified=False,
                  manifest_sha256=sha(MANIFEST),files=[])
    try:
        for asset in catalog['assets']:
            result = verify_file(bundle_path(archive_relative(asset).as_posix()),asset)
            record['files'].append(result)
            print('VERIFY BUNDLE',asset['id'],result['status'],flush=True)
        record.update(status='LOCAL_BUNDLE_VERIFIED',archives=18,metadata_files=9,
                      archive_bytes=sum(a['expected_bytes'] for a in catalog['assets']))
    except Exception as exc:
        record.update(status='FAILED',error_type=type(exc).__name__,error=str(exc))
        raise
    finally:
        write(certificate,record)
    print('BUNDLE VERIFY PASS: every archive matches its independent historical receipt.',flush=True)
