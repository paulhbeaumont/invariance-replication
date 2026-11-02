"""Download the versioned data attachments, checking bytes before installation.

Standard library only. No account, token, original Dropbox, or Git LFS needed.
Never overwrites a different existing input; never downloads a newer vintage.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import sys
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / 'config/release_assets.json'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def checked_path(root, relative):
    rel = PurePosixPath(relative)
    if rel.is_absolute() or not rel.parts or '..' in rel.parts or '\\' in relative or ':' in relative:
        raise ValueError('Unsafe attachment path: ' + relative)
    target = Path(root).joinpath(*rel.parts)
    if Path(root).resolve() not in target.resolve().parents:
        raise ValueError('Attachment path escapes package: ' + relative)
    return target


def valid(path, item):
    return path.is_file() and path.stat().st_size == item['bytes'] and digest(path) == item['sha256']


def obtain(item, cache, base_url, verify_only=False):
    target = checked_path(cache, item['name'])
    if target.exists():
        if valid(target, item):
            print('Cached and verified:', item['name'], flush=True)
            return target
        raise RuntimeError('Different cached file; move it aside before retrying: ' + str(target))
    if verify_only:
        raise RuntimeError('Missing cached attachment: ' + str(target))
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + '.partial')
    if partial.exists():
        raise RuntimeError('Interrupted download retained; remove or move it before retrying: ' + str(partial))
    url = base_url.rstrip('/') + '/' + item['name']
    request = urllib.request.Request(url, headers={'User-Agent': 'invariance-replication-data',
                                                  'Cache-Control': 'no-cache'})
    print('Downloading:', item['name'], f"({item['bytes'] / 1e9:.2f} GB)", flush=True)
    h = hashlib.sha256()
    received = 0
    next_report = 256 * 1024 * 1024
    with urllib.request.urlopen(request, timeout=120) as response, partial.open('xb') as output:
        for block in iter(lambda: response.read(8 * 1024 * 1024), b''):
            received += len(block)
            if received > item['bytes']:
                raise RuntimeError('Download exceeds pinned size: ' + item['name'])
            h.update(block)
            output.write(block)
            if received >= next_report:
                print(f"  {item['name']}: {received / 1e9:.2f} GB", flush=True)
                next_report += 256 * 1024 * 1024
    if received != item['bytes'] or h.hexdigest() != item['sha256']:
        raise RuntimeError('Downloaded size/SHA-256 mismatch; partial retained: ' + str(partial))
    partial.rename(target)
    return target


def install_prepared(archive, root, expected):
    with zipfile.ZipFile(archive) as bundle:
        members = bundle.infolist()
        names = [item.filename for item in members]
        if len(names) != len(set(names)) or set(names) != set(expected):
            raise RuntimeError('Prepared attachment does not have exactly the pinned input files')
        # Check every member and every existing destination BEFORE writing any input.
        for item in members:
            target = checked_path(root, item.filename)
            spec = expected[item.filename]
            if item.is_dir() or ((item.external_attr >> 16) & 0o170000) == 0o120000:
                raise RuntimeError('Directory/symlink in prepared attachment')
            if item.file_size != spec['bytes']:
                raise RuntimeError('Prepared member size differs: ' + item.filename)
            h = hashlib.sha256()
            with bundle.open(item) as source:
                for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
                    h.update(block)
            if h.hexdigest() != spec['sha256']:
                raise RuntimeError('Prepared member hash differs: ' + item.filename)
            if target.exists() and not valid(target, spec):
                raise RuntimeError('Refusing to overwrite different input: ' + str(target))
        for item in members:
            target = checked_path(root, item.filename)
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_name(target.name + '.partial')
            with bundle.open(item) as source, partial.open('xb') as output:
                shutil.copyfileobj(source, output, 8 * 1024 * 1024)
            if not valid(partial, expected[item.filename]):
                raise RuntimeError('Installed prepared input failed verification: ' + str(partial))
            partial.rename(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group', choices=['prepared', 'production', 'diagnostics', 'all'], default='prepared',
                        help='Default: prepared inputs (1.91 GB). Production adds exact FY2007-2018 ZIPs; '
                             'diagnostics are not needed for the production raw build.')
    parser.add_argument('--verify-only', action='store_true', help='Verify installed inputs; no network or writes.')
    parser.add_argument('--cache', type=Path, default=ROOT / '.downloads')
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    selected = [item for item in manifest['assets'] if args.group in ('all', item['group'])]
    expected = {item['path']: item for item in json.loads(
        (ROOT / 'provenance/input_manifest.json').read_text(encoding='utf-8')) if item['path'].startswith('data/')}
    for item in selected:
        if item['group'] == 'prepared':
            installed = all(valid(checked_path(ROOT, name), spec) for name, spec in expected.items())
            if installed:
                print('Prepared inputs: all 100 independent hashes match.', flush=True)
                continue
            if args.verify_only:
                raise RuntimeError('Prepared inputs are missing or differ from the independent manifest')
            archive = obtain(item, args.cache, manifest['base_url'])
            install_prepared(archive, ROOT, expected)
        else:
            target = checked_path(ROOT, item['path'])
            if target.exists():
                if not valid(target, item):
                    raise RuntimeError('Refusing to overwrite different raw archive: ' + str(target))
                print('Installed and verified:', item['name'], flush=True)
                continue
            if args.verify_only:
                raise RuntimeError('Missing raw archive: ' + str(target))
            # Download beside its final destination, so a rename needs no second 9.36 GB copy.
            raw = obtain(item, target.parent, manifest['base_url'])
            if raw.resolve() != target.resolve():
                raise RuntimeError('Raw filename/path mismatch')
    print('DATA PASS:', args.group, flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print('DATA FAILED:', str(error), file=sys.stderr)
        sys.exit(1)
