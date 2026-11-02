"""Publication I/O tests; no estimator or scientific benchmark is changed."""
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest
import zipfile

spec = importlib.util.spec_from_file_location('download_data', Path(__file__).resolve().parents[1] / 'download_data.py')
download = importlib.util.module_from_spec(spec)
spec.loader.exec_module(download)


class DownloadGuards(unittest.TestCase):
    def test_safe_paths(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for relative in ('../bad', '/absolute', 'C:/bad', 'data\\bad', 'data/../../bad'):
                with self.subTest(relative=relative), self.assertRaises(ValueError):
                    download.checked_path(root, relative)
            self.assertEqual(download.checked_path(root, 'data/test.txt'), root / 'data/test.txt')

    def fixture(self, root, content=b'original'):
        path = root / 'prepared.zip'
        with zipfile.ZipFile(path, 'w') as archive:
            archive.writestr('data/a.txt', content)
        expected = {'data/a.txt': {'bytes': 8, 'sha256': hashlib.sha256(b'original').hexdigest()}}
        return path, expected

    def test_install_and_repeat(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            archive, expected = self.fixture(root)
            download.install_prepared(archive, root, expected)
            download.install_prepared(archive, root, expected)
            self.assertEqual((root / 'data/a.txt').read_bytes(), b'original')

    def test_reject_changed_member_before_write(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            archive, expected = self.fixture(root, b'changed!')
            with self.assertRaisesRegex(RuntimeError, 'hash differs'):
                download.install_prepared(archive, root, expected)
            self.assertFalse((root / 'data').exists())

    def test_reject_existing_different_input(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            archive, expected = self.fixture(root)
            (root / 'data').mkdir()
            (root / 'data/a.txt').write_bytes(b'different')
            with self.assertRaisesRegex(RuntimeError, 'overwrite different'):
                download.install_prepared(archive, root, expected)
            self.assertEqual((root / 'data/a.txt').read_bytes(), b'different')

    def test_reject_unexpected_member(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            archive, expected = self.fixture(root)
            with zipfile.ZipFile(archive, 'a') as bundle:
                bundle.writestr('../bad', b'x')
            with self.assertRaisesRegex(RuntimeError, 'exactly the pinned'):
                download.install_prepared(archive, root, expected)


if __name__ == '__main__':
    unittest.main()
