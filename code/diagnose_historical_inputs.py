"""Compare completed historical outputs without bypassing the failed audit.

This diagnostic is deliberately separate from the production county-inputs
stage, which still requires a passing population audit. It never marks the
historical failure as passed or substitutes a frame into prepared replication.
"""
import sys
import argparse
import numpy as np
import pandas as pd

from package_io import ROOT, logs, read, require, sha, write
import historical_rebuild as historical


def compare_existing():
    prior=read(logs()/historical.RUNLOG/'source51_diagnostic_comparison.json')
    rows=[]
    for item in prior['files']:
        newpath=ROOT/prior['input_directory']/item['file']
        oldpath=ROOT/'data/upstream/baseline_roll5_gap5'/item['file']
        require(sha(oldpath)==item['reference_sha256'] and sha(newpath)==item['rebuilt_sha256'],
                'Diagnostic input or independent reference changed')
        old,new=pd.read_parquet(oldpath),pd.read_parquet(newpath)
        types=[]
        for column in old:
            if old[column].dtype!=new[column].dtype:
                require(np.issubdtype(old[column].dtype,np.integer) and np.issubdtype(new[column].dtype,np.integer),
                        'Non-integer storage difference requires separate investigation')
                require(np.array_equal(old[column].to_numpy(),new[column].to_numpy()),'Integer values differ')
                types.append(dict(column=str(column),original=str(old[column].dtype),rebuilt=str(new[column].dtype)))
        # Only integer storage width is set aside. Every float, key, missing
        # value, index, column order and other dtype must still match exactly.
        pd.testing.assert_frame_equal(old,new,check_exact=True,check_dtype=False)
        row=dict(item,status='EXACT_VALUES_INTEGER_WIDTH_ONLY' if types else 'EXACT_DATAFRAME_MATCH',
                 integer_storage_differences=types)
        rows.append(row)
        print(row['file'],row['status'],types,flush=True)
    write(logs()/historical.RUNLOG/'source51_exact_value_comparison.json',dict(
        input_directory=prior['input_directory'],files=rows,
        original_population_audit_status='FAILED',raw_to_results_certified=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compare-existing',action='store_true',help='Compare already constructed diagnostic inputs; preserve the original strict comparison')
    if parser.parse_args().compare_existing:
        compare_existing()
        return
    code, dc = historical.setup('source51_diagnostic')
    require(historical.passed_stage('employment') and historical.passed_stage('dcads-geography'),
            'Upstream source construction incomplete')
    audit = read(logs()/historical.RUNLOG/'population_attempt2_stage.json')
    require(audit['status']=='FAILED', 'This diagnostic records the unresolved original audit')
    require(all((dc.USASPENDING_OUT_DIR/('fy%d.parquet'%y)).exists() for y in range(2007,2019)),
            'Production USAspending block incomplete')
    destination = historical.WORKSPACE/'source51_diagnostic'
    require(not destination.exists(), 'Diagnostic inputs already exist; preserve prior comparison')
    destination.mkdir()
    producer = historical.load('diagnostic_source51',code/'py/51_county_multiplier_grid.py')
    panel = producer.base_panel()
    units = np.array(sorted(panel.ccg.unique()))
    require(len(units)==3058 and 15005 not in units, 'Historical retained sample differs')
    print('Diagnostic retained sample:',len(units),'counties; Kalawao absent',flush=True)
    matrices = producer.contract_matrices(units,1966,2018,drop_construction=True)
    frames = {'source51_county_panel.parquet':panel,
              'source51_contracts_net.parquet':matrices[('net',0.0)]}
    anchors = read(ROOT/'reference/upstream_source_manifest.json')
    comparisons = []
    for filename, frame in frames.items():
        frame.to_parquet(destination/filename,index=(filename=='source51_contracts_net.parquet'))
        original = ROOT/'data/upstream/baseline_roll5_gap5'/filename
        require(sha(original)==anchors['inputs'][filename], 'Independent source51 reference changed')
        expected = pd.read_parquet(original)
        try:
            pd.testing.assert_frame_equal(frame,expected,check_exact=True)
            row = dict(file=filename,status='EXACT_DATAFRAME_MATCH',rows=len(frame))
        except AssertionError as exc:
            row = dict(file=filename,status='DIFFERENT',rows=len(frame),reason=str(exc)[:6000])
        row.update(reference_sha256=anchors['inputs'][filename],rebuilt_sha256=sha(destination/filename))
        comparisons.append(row)
        print(row['file'],row['status'],flush=True)
    report = dict(status='DIAGNOSTIC_COMPARISON_COMPLETE_AUDIT_UNRESOLVED',
                  original_population_audit_status=audit['status'],excluded_county=15005,
                  excluded_county_in_retained_sample=False,retained_counties=len(units),
                  files=comparisons,raw_to_results_certified=False,
                  input_directory=destination.relative_to(ROOT).as_posix())
    write(logs()/historical.RUNLOG/'source51_diagnostic_comparison.json',report)
    compare_existing()
    print('Diagnostic input values match independent originals; the population audit remains FAILED.',flush=True)


if __name__=='__main__':
    main()
