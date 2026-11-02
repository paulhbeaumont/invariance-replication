"""Recovery I/O guard tests, not independent evidence of scientific correctness."""
import hashlib
from pathlib import Path
import tempfile
import unittest
import sys
from unittest.mock import patch
import zipfile

import recover_upstream_evidence as recovery


class RecoveryGuards(unittest.TestCase):
    def setUp(self):
        # Fixtures are scratch data, not logs. Keep them out of Dropbox so its
        # file watcher does not lock a temporary directory during cleanup.
        self.temp = tempfile.TemporaryDirectory(prefix='ns_upstream_guard_')
        self.folder = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_paths_cannot_escape(self):
        for value in ['../outside', '/absolute', 'C:/outside', 'a\\..\\outside', 'a//b', 'name:stream']:
            with self.assertRaises(RuntimeError):
                recovery.relative_path(value)

    def test_preserved_evidence_cannot_be_replaced(self):
        original = self.folder / 'original'
        original.mkdir()
        (original / 'receipt.json').write_bytes(b'original historical receipt')
        with patch.object(recovery, 'ROOT', self.folder), patch.object(recovery, 'EVIDENCE', self.folder / 'evidence'):
            first = recovery.preserve(original, 'receipt.json', 'receipt.json')
            self.assertEqual(first, recovery.preserve(original, 'receipt.json', 'receipt.json'))
            (original / 'receipt.json').write_bytes(b'changed source receipt')
            with self.assertRaises(RuntimeError):
                recovery.preserve(original, 'receipt.json', 'receipt.json')
            self.assertEqual((self.folder / 'evidence/receipt.json').read_bytes(), b'original historical receipt')

    def fixture(self):
        path = self.folder / 'archive.zip'
        with zipfile.ZipFile(path, 'w') as archive:
            archive.writestr('data.csv', 'id,value\n1,2\n')
        return dict(id='test', original_relative_path=path.name, expected_bytes=path.stat().st_size,
                    expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    expected_zip_members=1, role='test_only')

    def test_exact_archive_is_not_called_a_redownload(self):
        result = recovery.verify_archive(self.folder, self.fixture())
        self.assertEqual(result['status'], 'EXACT_HISTORICAL_ARCHIVE')
        self.assertFalse(result['reacquired_from_web'])

    def test_equal_size_corruption_is_rejected(self):
        asset = self.fixture()
        path = self.folder / asset['original_relative_path']
        data = bytearray(path.read_bytes())
        data[40] ^= 1
        path.write_bytes(data)
        with self.assertRaisesRegex(RuntimeError, 'checksum differs'):
            recovery.verify_archive(self.folder, asset)

    def test_member_count_mismatch_is_rejected(self):
        asset = self.fixture()
        asset['expected_zip_members'] = 2
        with self.assertRaisesRegex(RuntimeError, 'member count differs'):
            recovery.verify_archive(self.folder, asset)


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(stream=sys.stdout, verbosity=2))
