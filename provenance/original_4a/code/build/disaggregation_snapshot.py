"""Snapshot and audit unchanged CBP cells for the approved h=2,3,4 comparison.

Run class: cache_rebuild, required_key_absent (industry keys and employment).
The source is the established annual naics2/naics4_alloc constructed panel.
No recleaning, negative-value correction or rowwise sample exclusion is allowed.
"""
from pathlib import Path
import sys
import shutil
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'lib'))
import numpy as np
import pandas as pd
from baseline_io import APP,DATA,SHARED,locked,require,sha,write_json


def main():
    _,sources,lock=locked()
    annual=DATA.parent/sources['snapshot_directory']/'source51_county_panel.parquet'
    require(sha(annual)==sources['inputs']['source51_county_panel.parquet'],'County source hash mismatch')
    B=pd.read_parquet(annual)
    units=np.sort(B.ccg.unique())
    root=DATA/'intermediate/disaggregation_h234'
    root.mkdir(parents=True,exist_ok=False)
    source=SHARED/'1_current_paper/_data/temp/employment'
    audits=[]
    inputs={}
    for rung in ('naics2','naics4_alloc'):
        (root/rung).mkdir()
        previous=None
        for year in range(1978,2017):
            src=source/rung/f'y{year}.parquet'
            dst=root/rung/src.name
            before=sha(src)
            shutil.copy2(src,dst)
            require(sha(dst)==before and sha(src)==before,f'Source changed during snapshot: {src}')
            inputs[f'{rung}/{src.name}']=before
            f=pd.read_parquet(dst,columns=['ccg','code','emp'])
            require(not f.duplicated(['ccg','code']).any(),f'Duplicate source keys {rung} {year}')
            f=f[f.ccg.isin(units)].sort_values(['ccg','code']).reset_index(drop=True)
            require(np.array_equal(np.sort(f.ccg.unique()),units),f'Absent county {rung} {year}')
            keys=f[['ccg','code']].to_numpy()
            support_same=previous is None or np.array_equal(keys,previous)
            previous=keys
            sums=f.groupby('ccg').emp.sum().reindex(units).to_numpy()
            reference=B[B.year.eq(year)].set_index('ccg').E.reindex(units).to_numpy()
            gaps=np.abs(sums-reference)/np.maximum(np.abs(reference),1)
            audits.append(dict(rung=rung,year=year,n_cells=len(f),n_counties=f.ccg.nunique(),
                               negative=int((f.emp<0).sum()),missing=int(f.emp.isna().sum()),
                               zero=int((f.emp==0).sum()),support_same=bool(support_same),
                               max_relative_county_gap=float(np.max(gaps))))
            print(audits[-1],flush=True)
    audit=pd.DataFrame(audits)
    audit.to_csv(APP/'results/manifests/disaggregation_cell_source_audit.csv',index=False)
    good=bool(audit.negative.eq(0).all() and audit.missing.eq(0).all()
              and audit.support_same.all() and audit.max_relative_county_gap.lt(1e-9).all())
    report=dict(status='PASS' if good else 'FAIL',inputs=inputs,baseline_lock=lock,
                source_root=str(source),snapshot_root=str(root),builder_sha256=sha(__file__),
                audit_sha256=sha(APP/'results/manifests/disaggregation_cell_source_audit.csv'),
                reason='No cell alteration is authorized; negative values, missingness, support changes or failed county reconstruction stop the build.')
    write_json(APP/'results/manifests/disaggregation_snapshot.json',report)
    require(good,'Cell-source audit failed; inspect disaggregation_cell_source_audit.csv. No cells were altered.')


if __name__=='__main__':
    main()
