"""Bundle I/O guards using small synthetic ZIPs, not scientific benchmarks."""
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile
import sys

import usaspending_bundle as bundle
from package_io import read, sha, write


class BundleGuards(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='ns_bundle_guard_')
        self.root = Path(self.temp.name)
        self.project = self.root/'original_project'
        self.project.mkdir()
        (self.project/'INVARIANCE_TEST_PROTOCOL.md').write_bytes(b'test project marker')
        self.catalog = self.root/'provenance/catalog.json'
        self.destination = self.root/'data/raw/USAspending'
        self.manifest = self.root/'provenance/bundle.json'
        self.assets = []
        for year in range(2001,2019):
            production = year>=2007
            name = 'FY%d_097_Contracts_Full_20260806.zip'%year if production else 'overlap_%d.zip'%year
            relative = '_data/source/USAspending/'+('' if production else 'overlap/fy%d/'%year)+name
            path = self.project/relative
            path.parent.mkdir(parents=True,exist_ok=True)
            with zipfile.ZipFile(path,'w') as archive:
                archive.writestr('data.csv','year,value\n%d,1\n'%year)
            self.assets.append(dict(id='fy%d'%year,fiscal_year=year,
                role=bundle.PRODUCTION if production else bundle.DIAGNOSTIC,
                vintage='2026-08-06' if production else '2026-08-14',
                original_relative_path=relative,url='https://example.invalid/'+name,
                expected_bytes=path.stat().st_size,expected_sha256=sha(path),expected_zip_members=1))
        evidence = []
        names = ['manifest.csv','download_receipt.csv','overlap/download_receipt.json']
        names += ['overlap/fy%d/overlap_request.json'%year for year in range(2001,2007)]
        for name in names:
            relative = 'provenance/upstream_recovery_20261023/usaspending/'+name
            path = self.root/relative
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(('original receipt: '+name).encode())
            evidence.append(dict(preserved=relative,sha256=sha(path)))
        write(self.catalog,dict(pinned_code_commit=bundle.PINNED_COMMIT,assets=self.assets,evidence=evidence))
        self.patches = [patch.object(bundle,'ROOT',self.root),patch.object(bundle,'BUNDLE',self.destination),
                        patch.object(bundle,'MANIFEST',self.manifest),patch.object(bundle,'CATALOG',self.catalog),
                        patch.object(bundle,'CATALOG_SHA256',sha(self.catalog)),
                        patch.object(bundle,'logs',return_value=self.root/'test_reports')]
        for item in self.patches:
            item.start()

    def tearDown(self):
        for item in reversed(self.patches):
            item.stop()
        self.temp.cleanup()

    def copy(self):
        with redirect_stdout(io.StringIO()):
            bundle.bundle(self.project)

    def target(self,year):
        asset = next(a for a in self.assets if a['fiscal_year']==year)
        return self.destination/bundle.archive_relative(asset)

    def test_independent_copies_and_complete_receipt_inventory(self):
        self.copy()
        manifest = read(self.manifest)
        self.assertEqual(len(manifest['archives']),18)
        self.assertEqual(len(manifest['metadata']),9)
        for asset in self.assets:
            target = self.target(asset['fiscal_year'])
            self.assertEqual(sha(target),asset['expected_sha256'])
            self.assertFalse(target.samefile(self.project/asset['original_relative_path']))
        self.assertIsNone(manifest['public_deposit_url'])
        self.assertFalse(manifest['raw_to_results_certified'])

    def test_identical_second_copy_does_not_rewrite_files(self):
        self.copy()
        before = {p:p.stat().st_mtime_ns for p in self.destination.rglob('*') if p.is_file()}
        digest = sha(self.manifest)
        self.copy()
        self.assertEqual(before,{p:p.stat().st_mtime_ns for p in before})
        self.assertEqual(sha(self.manifest),digest)

    def test_corrupted_bundle_is_rejected_not_overwritten(self):
        self.copy()
        target = self.target(2007)
        value = bytearray(target.read_bytes());value[40]^=1;target.write_bytes(value)
        with self.assertRaisesRegex(RuntimeError,'checksum differs'):
            bundle.production_archives({2007})
        with self.assertRaisesRegex(RuntimeError,'refusing overwrite'):
            self.copy()
        self.assertEqual(target.read_bytes(),value)

    def test_changed_original_is_not_silently_accepted(self):
        self.copy()
        asset = next(a for a in self.assets if a['fiscal_year']==2007)
        source = self.project/asset['original_relative_path']
        value = bytearray(source.read_bytes());value[40]^=1;source.write_bytes(value)
        with self.assertRaisesRegex(RuntimeError,'Original source differs'):
            self.copy()
        self.assertEqual(sha(self.target(2007)),asset['expected_sha256'])

    def test_missing_local_file_does_not_fall_back_to_original_project(self):
        self.copy()
        self.target(2007).unlink()
        with self.assertRaisesRegex(RuntimeError,'Missing historical archive'):
            bundle.production_archives({2007})
        override = self.project/'_data/source/USAspending'
        self.assertEqual(len(bundle.production_archives({2007},override=override)),1)

    def test_diagnostic_years_cannot_enter_the_production_loader(self):
        self.copy()
        with self.assertRaisesRegex(RuntimeError,'Only FY2007-2018'):
            bundle.production_archives({2001})
        with self.assertRaisesRegex(RuntimeError,'Only FY2007-2018'):
            bundle.production_archives({2001,2007})

    def test_changed_receipt_and_changed_checksum_list_are_rejected(self):
        self.copy()
        path = self.destination/'production/manifest.csv'
        original = path.read_bytes();path.write_bytes(b'changed receipt')
        with self.assertRaisesRegex(RuntimeError,'bundled receipt/request'):
            bundle.production_archives({2007})
        path.write_bytes(original)
        (self.destination/'SHA256SUMS').write_bytes(b'changed checksums')
        with self.assertRaisesRegex(RuntimeError,'SHA256SUMS'):
            bundle.production_archives({2007})

    def test_changed_catalog_and_manifest_are_rejected(self):
        self.copy()
        original = self.catalog.read_bytes();self.catalog.write_bytes(b'{}')
        with self.assertRaisesRegex(RuntimeError,'catalog changed'):
            bundle.production_archives({2007})
        self.catalog.write_bytes(original)
        manifest = read(self.manifest);manifest['archives'][0]['role']=bundle.PRODUCTION
        write(self.manifest,manifest)
        with self.assertRaisesRegex(RuntimeError,'bundle manifest'):
            bundle.production_archives({2007})

    def test_bundle_remains_usable_without_the_original_project(self):
        self.copy()
        self.project.rename(self.root/'original_project_not_available')
        paths = bundle.production_archives(set(range(2007,2019)))
        self.assertEqual(len(paths),12)
        self.assertTrue(all(self.destination in p.resolve().parents for _,p in paths))
        with redirect_stdout(io.StringIO()):
            bundle.verify()

    def test_paths_cannot_escape_the_bundle(self):
        for value in ['../outside','C:/outside','/outside','a/../outside']:
            with self.assertRaises(RuntimeError):
                bundle.bundle_path(value)

    def test_verified_partial_can_complete_without_recopying(self):
        asset = self.assets[0]
        target = self.target(asset['fiscal_year'])
        partial = target.with_name(target.name+'.fixture.partial')
        partial.parent.mkdir(parents=True,exist_ok=True)
        source = self.project/asset['original_relative_path']
        partial.write_bytes(source.read_bytes())
        with patch.object(bundle.shutil,'copy2',side_effect=AssertionError('Must not recopy')):
            status = bundle.copy_checked(source,target,asset['expected_sha256'],asset['expected_bytes'])
        self.assertEqual(status,'COMPLETED_HASH_VERIFIED_PARTIAL')
        self.assertEqual(sha(target),asset['expected_sha256'])

    def test_windows_lock_retries_the_same_bundle_move_only(self):
        target = self.destination/'target.zip'
        partial = self.destination/'target.zip.fixture.partial'
        error = PermissionError('test Windows lock');error.winerror=32
        with patch.object(Path,'rename',autospec=True,side_effect=[error,None]) as moved:
            with patch.object(bundle.time,'sleep'),redirect_stdout(io.StringIO()):
                bundle.publish_partial(partial,target)
            self.assertEqual(moved.call_count,2)
            self.assertEqual(moved.call_args_list[0],moved.call_args_list[1])
        with self.assertRaisesRegex(RuntimeError,'escapes'):
            bundle.publish_partial(self.root/'outside.partial',target)

    def test_unknown_file_move_error_remains_fatal(self):
        with patch.object(Path,'rename',autospec=True,side_effect=PermissionError('not a Windows sharing error')) as moved:
            with self.assertRaises(PermissionError):
                bundle.publish_partial(self.destination/'source.partial',self.destination/'target.zip')
            self.assertEqual(moved.call_count,1)


if __name__=='__main__':
    unittest.main(testRunner=unittest.TextTestRunner(stream=sys.stdout,verbosity=2))
