"""Host-I/O guard tests; scientific evidence remains the historical targets."""
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import historical_rebuild as build


class HistoricalGuards(unittest.TestCase):
    failure = ('\nFAIL: 1 check(s) failed\n'
               '  income gap: CCG 15005 has no income denominator but $34,759,561 of contracts\n')

    def test_only_known_excluded_unit_failure_is_a_warning(self):
        counties = list(range(10000,13058))
        warning = build.excluded_population_failure(1,self.failure,counties,counties)
        self.assertEqual(warning['code'],build.EXCLUDED_UNIT_WARNING)
        self.assertEqual(warning['original_audit_status'],'FAILED')
        self.assertFalse(warning['data_or_specification_changed'])
        self.assertTrue(warning['excluded_from_both_samples'])

    def test_other_or_additional_audit_failures_remain_fatal(self):
        counties = list(range(10000,13058))
        for result, stderr in [
            (2,self.failure), (0,self.failure), (1,''),
            (1,self.failure.replace('15005','15009')),
            (1,self.failure.replace('1 check(s)','2 check(s)')+'  state total: failed\n'),
            (1,self.failure+'  population state total: failed\n'),
            (1,self.failure.replace('$34,759,561','$0')),
            (1,self.failure+'Traceback: other error\n'),
        ]:
            with self.subTest(result=result,stderr=stderr):
                with self.assertRaises(RuntimeError):
                    build.excluded_population_failure(result,stderr,counties,counties)

    def test_warning_cannot_change_or_include_the_retained_sample(self):
        counties = list(range(10000,13058))
        for rebuilt, reference in [
            (counties[:-1],counties),
            (counties[:-1]+[14000],counties),
            (counties[:-1]+[15005],counties[:-1]+[15005]),
            (counties[:-1]+[counties[0]],counties),
        ]:
            with self.subTest(rebuilt=rebuilt[-2:]):
                with self.assertRaises(RuntimeError):
                    build.excluded_population_failure(1,self.failure,rebuilt,reference)

    def test_warning_does_not_unlock_other_construction_stages(self):
        with tempfile.TemporaryDirectory(prefix='ns_audit_guard_') as folder:
            root = Path(folder)
            record = dict(stage='employment',commit=build.COMMIT,
                          status='READY_WITH_EXCLUDED_UNIT_WARNING',original_audit_status='FAILED',
                          warning={'code':build.EXCLUDED_UNIT_WARNING})
            build.write(root/build.RUNLOG/'employment_stage.json',record)
            with patch.object(build,'logs',return_value=root):
                self.assertFalse(build.passed_stage('employment'))

    def test_population_warning_is_bound_to_unchanged_input_bytes(self):
        with tempfile.TemporaryDirectory(prefix='ns_audit_guard_') as folder:
            root = Path(folder)
            workspace = root/'built'
            counties = list(range(10000,13058))
            warning = build.excluded_population_failure(1,self.failure,counties,counties)
            paths = [workspace/'1_current_paper/_data/temp/county_panel/income_ccg_year.parquet',
                     workspace/'1_current_paper/_data/temp/county_panel/population_ccg_year.parquet',
                     workspace/'1_current_paper/_data/temp/employment/efsy_ccg_year.parquet',
                     root/'data/upstream/baseline_roll5_gap5/source51_county_panel.parquet']
            for path in paths:
                build.write(path,{'fixture':1})
            warning['evidence_files'] = [dict(path=p.relative_to(root).as_posix(),sha256=build.sha(p)) for p in paths]
            record = dict(stage='population',commit=build.COMMIT,
                          status='READY_WITH_EXCLUDED_UNIT_WARNING',original_audit_status='FAILED',warning=warning)
            build.write(root/build.RUNLOG/'population_stage.json',record)
            with patch.object(build,'ROOT',root), patch.object(build,'WORKSPACE',workspace), patch.object(build,'logs',return_value=root):
                self.assertTrue(build.passed_stage('population'))
                build.write(paths[0],{'fixture':2})
                with self.assertRaisesRegex(RuntimeError,'evidence changed'):
                    build.passed_stage('population')

    def test_sharing_violation_retries_only_the_same_output_move(self):
        with tempfile.TemporaryDirectory(prefix='ns_host_guard_') as folder:
            source, target = Path(folder)/'source', Path(folder)/'target'
            error = PermissionError('test Windows sharing lock')
            error.winerror = 32
            with patch.object(build,'WORKSPACE',Path(folder)), patch.object(os,'replace',side_effect=[error,None]) as moved:
                with patch.object(build.time,'sleep'):
                    with build.output_io_retries():
                        os.replace(source,target)
                self.assertEqual(moved.call_count,2)
                self.assertEqual(moved.call_args_list[0],moved.call_args_list[1])

    def test_external_output_does_not_get_a_retry(self):
        error = PermissionError('test external move')
        error.winerror = 32
        with patch.object(os,'replace',side_effect=error) as moved:
            with build.output_io_retries():
                with self.assertRaises(PermissionError):
                    os.replace(Path('external_source'),Path('external_target'))
            self.assertEqual(moved.call_count,1)

    def test_pathlib_move_retries_on_legacy_and_modern_python(self):
        with tempfile.TemporaryDirectory(prefix='ns_host_guard_') as folder:
            source, target = Path(folder)/'source', Path(folder)/'target'
            error = PermissionError('test pathlib sharing lock')
            error.winerror = 32
            with patch.object(build,'WORKSPACE',Path(folder)), patch.object(Path,'replace',side_effect=[error,None]) as moved:
                with patch.object(build.time,'sleep'):
                    with build.output_io_retries():
                        source.replace(target)
                self.assertEqual(moved.call_count,2)

    def test_original_arguments_restored_after_failure(self):
        previous = sys.argv
        def main():
            self.assertEqual(sys.argv,['fixture.py','--unchanged-default'])
            return 1
        module = SimpleNamespace(__file__='fixture.py',main=main)
        with self.assertRaises(RuntimeError):
            build.original_main(module,['--unchanged-default'])
        self.assertIs(sys.argv,previous)

    def test_completed_parquet_reader_does_not_block_output_replacement(self):
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            self.skipTest('PyArrow is installed in the raw-build environment')
        with tempfile.TemporaryDirectory(prefix='ns_host_guard_') as folder:
            source, staged = Path(folder)/'data.parquet',Path(folder)/'data.parquet.tmp'
            pq.write_table(pa.table({'x':[1,2]}),source)
            pq.write_table(pa.table({'x':[3,4]}),staged)
            with patch.object(build,'WORKSPACE',Path(folder)):
                with build.output_io_retries():
                    reader=pq.ParquetFile(source)
                    self.assertEqual(sum(b.num_rows for b in reader.iter_batches()),2)
                    os.replace(staged,source)
                    reader.close(force=True)
            self.assertEqual(pq.read_table(source).column('x').to_pylist(),[3,4])


if __name__=='__main__':
    unittest.main(testRunner=unittest.TextTestRunner(stream=sys.stdout,verbosity=2))
