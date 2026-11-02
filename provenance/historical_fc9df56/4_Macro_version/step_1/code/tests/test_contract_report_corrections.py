"""Conservative reversal matching: signs, ambiguity, chronology and chains."""

import importlib.util
from pathlib import Path
import unittest

import duckdb
import pandas as pd

PATH = Path(__file__).resolve().parents[1] / 'pipeline/103_audit_contract_report_corrections.py'
SPEC = importlib.util.spec_from_file_location('report_correction_audit', PATH)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def row(key, amount, code='', month='198001', **kwargs):
    result = {c: 'K' for c in AUDIT.IDENTITY + AUDIT.PAYLOAD}
    result.update(row_id=key, amount=amount, correction_code=code,
                  calendar_yearmonth=month, fiscal_file=month[:4])
    result.update(kwargs)
    return result


class CorrectionMatching(unittest.TestCase):
    def statuses(self, rows):
        with duckdb.connect() as con:
            frame = pd.DataFrame(rows)
            con.register('fixture', frame)
            con.execute('CREATE TABLE reports AS SELECT * FROM fixture')
            AUDIT.match(con, save=False)
            result = con.sql('SELECT row_id,match_status,original_id,corrected_id FROM status').df()
            return result.set_index('row_id').to_dict('index')

    def test_both_signs_and_cross_year(self):
        result = self.statuses([
            row('o',100,month='197912'),row('r',-100,'2'),row('c',10,'1'),
            row('on',-50,contract_no='N'),row('rn',50,'2',contract_no='N')])
        self.assertEqual(result['r']['match_status'],'unique_pair_and_replacement')
        self.assertEqual(result['r']['original_id'],'o')
        self.assertEqual(result['rn']['match_status'],'unique_pair_replacement_unresolved')

    def test_duplicate_original_and_reused_original(self):
        result = self.statuses([
            row('o',100),row('o2',100),row('r',-100,'2'),
            row('x',50,contract_no='N'),row('rx',-50,'2',contract_no='N'),
            row('rx2',-50,'2',contract_no='N')])
        self.assertTrue(all(x['match_status']=='ambiguous_original' for x in result.values()))

    def test_reject_future_missing_identity_payload_difference_and_zero(self):
        result = self.statuses([
            row('o',100,month='198002'),row('r',-100,'2'),
            row('b',30,contract_no=''),row('rb',-30,'2',contract_no=''),
            row('g',40,contract_no='G'),row('rg',-40,'2',contract_no='G',fsc='DIFFERENT'),
            row('z',0,'2')])
        for key in ['r','rb','rg']:
            self.assertEqual(result[key]['match_status'],'no_exact_original')
        self.assertEqual(result['z']['match_status'],'zero_reversal')

    def test_replacement_may_change_payload_but_must_be_unique(self):
        result = self.statuses([
            row('o',100),row('r',-100,'2',month='198002'),
            row('c',10,'1',month='198002',contractor_code='NEW',county_code='OTHER'),
            row('x',50,contract_no='N'),row('rx',-50,'2',contract_no='N'),
            row('cx',10,'1',contract_no='N'),row('cx2',20,'1',contract_no='N')])
        self.assertEqual(result['r']['corrected_id'],'c')
        self.assertEqual(result['rx']['match_status'],'unique_pair_replacement_unresolved')

    def test_correction_chain_and_ordinary_negative(self):
        result = self.statuses([
            row('o',100),row('r',-100,'2',month='198002'),row('c',20,'1',month='198002'),
            row('r2',-20,'2',month='198003'),row('c2',15,'1',month='198003'),
            row('ordinary_negative',-25,contract_no='N')])
        self.assertEqual(result['r']['corrected_id'],'c')
        self.assertEqual(result['r2']['original_id'],'c')
        self.assertEqual(result['r2']['corrected_id'],'c2')
        self.assertNotIn('ordinary_negative',result)


if __name__ == '__main__':
    unittest.main()
