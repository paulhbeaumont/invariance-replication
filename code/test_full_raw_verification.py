"""I/O/boundary guards. Scientific evidence is the immutable original targets."""
from pathlib import Path
import tempfile
import sys
import unittest
from unittest.mock import Mock,patch

import numpy as np
import pandas as pd
import raw_application as application
import full_raw_verification as driver
from package_io import sha


class FullRawGuards(unittest.TestCase):
    def test_growth_roundoff_requires_unchanged_formula_and_tiny_weighted_moments(self):
        old=pd.DataFrame({'cur':[1e-7],'lag':[1e-7],'y_mpgr':[0.],'w_mpgr':[1e-12]})
        new=old.copy();new['cur']+=1e-14
        new['y_mpgr']=2*(new.cur-new.lag)/(new.cur+new.lag)
        result=application.growth_roundoff(new,old,'y_mpgr','cells/mpgr_iy_h3/y1981.parquet')
        self.assertTrue(result['growth_formula_unchanged'])
        expected=float(((new.w_mpgr*new.y_mpgr)**2-(old.w_mpgr*old.y_mpgr)**2).abs().max())
        self.assertEqual(result['max_squared_weighted_outcome_difference'],expected)
        bad=new.copy();bad['y_mpgr']=.1
        with self.assertRaisesRegex(RuntimeError,'Growth formula differs'):
            application.growth_roundoff(bad,old,'y_mpgr','cells/mpgr_iy_h3/y1981.parquet')
        new['w_mpgr']=1.
        with self.assertRaisesRegex(RuntimeError,'Growth weight differs'):
            application.growth_roundoff(new,old,'y_mpgr','cells/mpgr_iy_h3/y1981.parquet')

    def test_combined_growth_roundoff_needs_checked_source_years(self):
        frame=pd.DataFrame({'y':[1.],'w_lag':[1.]})
        with patch.object(application,'COMPARISONS',[]):
            with self.assertRaisesRegex(RuntimeError,'all independently checked annual'):
                application.growth_roundoff(frame,frame,'y','cells/log_iy_h3.parquet')

    @unittest.skipUnless(application.os.name=='nt','Windows-specific long-path I/O')
    def test_windows_source_copy_uses_extended_paths(self):
        with patch.object(application.shutil,'copytree') as copy:
            application.copy_source_tree(Path.cwd()/'source',Path.cwd()/'destination')
        args=copy.call_args.args
        self.assertTrue(all(value.startswith('\\\\?\\') for value in args))
        self.assertEqual(copy.call_count,1)

    def test_unsafe_run_identifier_is_rejected_before_creation(self):
        for name in ('../old','/root','C:/old','a/b','','x'*65):
            with self.subTest(name=name),self.assertRaises(RuntimeError):
                driver.configure(name)

    def test_prepared_data_cannot_seed_construction(self):
        with tempfile.TemporaryDirectory(prefix='ns_clean_guard_') as folder:
            root=Path(folder)
            with patch.object(application,'ROOT',root),patch.object(application,'_reference_read',False):
                with application.construction_read_guard():
                    for name in ('upstream','disaggregation','cells','aggregation','heterogeneity'):
                        with self.assertRaisesRegex(RuntimeError,'Forbidden prepared-data read'):
                            pd.read_parquet(root/'data'/name/'old.parquet')

    def test_reader_is_restored_when_construction_fails(self):
        before=pd.read_parquet
        with self.assertRaises(ValueError):
            with application.construction_read_guard():
                raise ValueError('test interrupted build')
        self.assertIs(pd.read_parquet,before)

    def test_only_fresh_run_paths_can_enter_queries(self):
        with tempfile.TemporaryDirectory(prefix='ns_clean_guard_') as folder:
            root=Path(folder);run=root/'runs/clean'
            connection=Mock()
            with patch.object(application,'ACTIVE_RUN',run):
                application.query(connection,'test',[run/'source.parquet'])
                connection.execute.assert_called_once()
                with self.assertRaisesRegex(RuntimeError,'outside this clean run'):
                    application.query(connection,'test',[root/'data/upstream/runs/old.parquet'])

    def test_query_glob_checks_its_directory_without_resolving_literal_wildcard(self):
        with tempfile.TemporaryDirectory(prefix='ns_glob_guard_') as folder:
            run=Path(folder)/'run';connection=Mock()
            with patch.object(application,'ACTIVE_RUN',run):
                application.query(connection,'test',[run/'fresh/y*.parquet'])
                connection.execute.assert_called_once()
                with self.assertRaisesRegex(RuntimeError,'outside this clean run'):
                    application.query(connection,'test',[run/'../outside/y*.parquet'])
                with self.assertRaisesRegex(RuntimeError,'Only a filename'):
                    application.query(connection,'test',[run/'*/y*.parquet'])

    def test_independent_comparison_rejects_changed_values_without_repair(self):
        with tempfile.TemporaryDirectory(prefix='ns_clean_guard_') as folder:
            root=Path(folder);path=root/'data/disaggregation/test.parquet';path.parent.mkdir(parents=True)
            old=pd.DataFrame({'ccg':[1,2],'year':[2000,2000],'value':[1.,2.]})
            old.to_parquet(path,index=False)
            bad=old.copy();bad.loc[1,'value']=3.
            anchors={'data/disaggregation/test.parquet':sha(path)}
            with patch.object(application,'ROOT',root),patch.object(application,'_anchors',anchors):
                with application.construction_read_guard():
                    with self.assertRaisesRegex(RuntimeError,'Independent input values differ'):
                        application.compare(bad,'disaggregation/test.parquet',['ccg','year'],root/'reports')
            self.assertEqual(bad.loc[1,'value'],3.)
            pd.testing.assert_frame_equal(pd.read_parquet(path),old)

    def test_unchanged_reconciliation_function_is_independently_anchored(self):
        function=application.original_function('provenance/original_4a/code/build/naics6_cells.py','split_parent')
        parent=pd.DataFrame({'ccg':[1,1],'parent':[1111,2222],'emp':[10.,20.]})
        six=pd.DataFrame({'ccg':[1,1,1],'parent':[1111,1111,2222],
                          'code':['111111','111112','222211'],'emp':[10.,10.,5.]})
        cells,accounting=function(parent,six)
        self.assertTrue(cells.emp.ge(0).all())
        np.testing.assert_allclose(cells.groupby('parent').emp.sum(),[10.,20.])
        self.assertEqual(accounting.factor.tolist(),[.5,1.])


if __name__=='__main__':
    unittest.main(testRunner=unittest.TextTestRunner(stream=sys.stdout,verbosity=2))
