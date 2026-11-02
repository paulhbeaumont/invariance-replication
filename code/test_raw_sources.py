"""I/O guard tests, not evidence that the scientific estimand is correct.

Scientific evidence comes from original source receipts, the immutable EFSY
reference and unchanged prepared-data/result reference files.
"""
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import raw_sources as raw

_approved_specification=(raw.ROOT/'docs/SPECIFICATION.md').read_bytes()


class Response(io.BytesIO):
    status = 200
    headers = {}

    def geturl(self):
        return 'https://example.invalid/public.csv?temporary_token=not_for_logs'


class SourceGuards(unittest.TestCase):
    def setUp(self):
        parent = raw.logs() / 'raw_rebuild_20261022/tmp'
        parent.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix='source_guard_', dir=str(parent))
        self.root = Path(self.temp.name).resolve()
        assert parent.resolve() in self.root.parents
        (self.root/'docs').mkdir()
        (self.root/'docs/SPECIFICATION.md').write_text('test-fixture-only', encoding='utf-8')
        self.patches = [patch.object(raw,'ROOT',self.root),
                        patch.object(raw,'CATALOG',self.root/'catalog.json'),
                        patch.object(raw,'STORAGE',self.root/'scratch'),
                        patch.object(raw,'RECEIPTS',self.root/'scratch/receipts.json'),
                        patch.object(raw,'logs',lambda:self.root/'test_logs')]
        for item in self.patches:
            item.start()

    def tearDown(self):
        for item in reversed(self.patches):
            item.stop()
        self.temp.cleanup()

    def catalog(self, digest):
        raw.write(raw.CATALOG,dict(specification_sha256=raw.sha(self.root/'docs/SPECIFICATION.md'),
                  assets=[dict(id='case',source='efsy',relative_path='case.csv',
                     url='https://example.invalid/public.csv',expected_bytes=4,expected_sha256=digest)]))

    def test_paths_cannot_escape(self):
        for value in ['../outside','/absolute','C:/outside','a/../../outside','a\\..\\outside']:
            with self.assertRaises(RuntimeError):
                raw.safe_relative(value)

    def test_exact_download_and_cached_file(self):
        self.catalog(hashlib.sha256(b'data').hexdigest())
        with patch.object(raw.urllib.request,'urlopen',return_value=Response(b'data')):
            raw.download(['efsy'])
        raw.download(['efsy'])
        receipts=raw.read(raw.RECEIPTS)
        self.assertEqual(receipts[0]['status'],'downloaded_exact_checksum')
        self.assertEqual(receipts[1]['status'],'already_present_verified')
        self.assertNotIn('temporary_token',json.dumps(receipts))

    def test_mismatched_source_is_quarantined(self):
        self.catalog(hashlib.sha256(b'data').hexdigest())
        with patch.object(raw.urllib.request,'urlopen',return_value=Response(b'xxxx')):
            with self.assertRaises(RuntimeError):
                raw.download(['efsy'])
        folder=raw.STORAGE/'raw/efsy'
        self.assertFalse((folder/'case.csv').exists())
        self.assertEqual(len(list(folder.glob('case.csv.different_*'))),1)
        self.assertEqual(raw.read(raw.RECEIPTS)[0]['status'],'FAILED')

    def test_first_retrieval_never_claims_historical_match(self):
        self.catalog(None)
        with patch.object(raw.urllib.request,'urlopen',return_value=Response(b'data')):
            raw.download(['efsy'])
        self.assertEqual(raw.read(raw.RECEIPTS)[0]['status'],
                         'first_retrieval_size_verified_no_historical_hash')
        (raw.STORAGE/'raw/efsy/case.csv').write_bytes(b'xxxx')
        with self.assertRaises(RuntimeError):
            raw.download(['efsy'])

    def test_changed_specification_is_not_accepted(self):
        self.catalog(hashlib.sha256(b'data').hexdigest())
        (self.root/'docs/SPECIFICATION.md').write_text('changed-fixture',encoding='utf-8')
        with self.assertRaises(RuntimeError):
            raw.download(['efsy'])

    def test_only_the_exact_approved_note_can_extend_the_original_lock(self):
        path=self.root/'docs/SPECIFICATION.md'
        path.write_bytes(_approved_specification)
        original='9b171135afb9c200f9118a793637c9c74abd76564b23f188ed536c74e7bf9b2c'
        self.assertEqual(raw.checked_specification(original),'ORIGINAL_SPECIFICATION_PLUS_EXACT_APPROVED_KALAWAO_NOTE')
        path.write_bytes(_approved_specification.replace(b'county-year level',b'county-month level',1))
        with self.assertRaises(RuntimeError):
            raw.checked_specification(original)
        path.write_bytes(_approved_specification.replace(b'Kalawao, Hawaii',b'Kalawao, Nevada',1))
        with self.assertRaises(RuntimeError):
            raw.checked_specification(original)


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(stream=sys.stdout, verbosity=2))
