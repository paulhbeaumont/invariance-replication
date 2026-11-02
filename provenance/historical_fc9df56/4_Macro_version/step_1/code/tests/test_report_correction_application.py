"""Source-backed application checks using an independently parsed FY1982 file."""
import importlib.util
import sys
from pathlib import Path
import unittest

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(CODE/'py'))
import dcads_common as dc
import report_corrections as rc


class Application(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.work = dc.MACRO_VERSION_TEMP_DIR/'contract_report_correction_application'
        cls.source = next(p for p in dc.data_files() if p.name=='RG330.DEFCON.FY82')
        cls.table = pq.read_table(cls.work/'parser_fixture_uncorrected_1982.parquet')
        cls.clean = rc.clean_table(cls.table,cls.source.name,cls.source)

    def test_approved_total_and_source_identity(self):
        self.assertEqual(sum(self.clean['report_correction_neutralized'].to_pylist()),3426)
        raw = self.table['dollars'].to_numpy()
        clean = self.clean['dollars'].to_numpy()
        self.assertEqual(int(np.maximum(raw,0).sum()-np.maximum(clean,0).sum())*1000,7469282000)
        self.assertTrue(rc.reported_table(self.clean).drop(rc.EXTRA).equals(self.table))

    def test_idempotent(self):
        self.assertTrue(rc.clean_table(self.clean,self.source.name,self.source).equals(self.clean))

    def test_st_james_replacement_survives(self):
        x=self.clean.to_pandas()
        x=x.loc[x.contract_no.eq('LA60082C5010') & x.report_no.eq('3082')]
        self.assertEqual(sorted(x.dollars_reported.tolist()),[-573747,57375,573747])
        self.assertEqual(sorted(x.dollars.tolist()),[0,0,57375])

    def test_wrong_reported_amount_rejected(self):
        index=int(rc.rules().loc[lambda x:x.source_file.eq(self.source.name)].record_number.iloc[0])-1
        dollars=self.table['dollars'].to_numpy().copy()
        dollars[index]+=1
        bad=self.table.set_column(self.table.schema.get_field_index('dollars'),'dollars',pa.array(dollars))
        with self.assertRaisesRegex(ValueError,'Parsed/source mismatch'):
            rc.clean_table(bad,self.source.name,self.source)

    def test_parser_hook(self):
        spec=importlib.util.spec_from_file_location('parse04',CODE/'py/04_parse_dcads.py')
        parser=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(parser)
        layouts,index=dc.load_layouts(include_delimited=False)
        output=self.work/'parser_fixture_corrected_1982.parquet'
        parser.parse_to_parquet(self.source,layouts[index[self.source.name]],output,'1982',self.source.name)
        self.assertTrue(pq.read_table(output).equals(self.clean))


if __name__=='__main__':
    unittest.main()
