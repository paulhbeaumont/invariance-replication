import numpy as np
import pandas as pd
from pathlib import Path
from baseline_math import lam_weights, wins
DATA=Path(__file__).resolve().parents[2]/"data"
from package_io import require, sha

def county_frame(spec,sources):
    source=DATA.parent/sources['snapshot_directory']
    for name,digest in sources['inputs'].items():
        require(sha(source/name)==digest,f'Snapshot hash mismatch: {name}')
    B=pd.read_parquet(source/'source51_county_panel.parquet')
    C=pd.read_parquet(source/'source51_contracts_net.parquet')
    require(not B.duplicated(['ccg','year']).any(),'Duplicate source county-years')
    units=np.array(sorted(B.ccg.unique()))
    require(len(units)==spec['n_counties'],'County support differs from baseline')
    require(np.array_equal(C.index.to_numpy(),units),'Procurement county order mismatch')
    years=list(range(spec['first_year'],spec['last_year']+1))
    h=spec['horizon']
    idx=pd.MultiIndex.from_product([units,years],names=['ccg','year'])
    f=pd.DataFrame(index=idx).reset_index()
    geography=B.drop_duplicates('ccg').set_index('ccg')
    f['state']=f.ccg.map(geography.state)
    f['division']=f.ccg.map(geography.division)
    piv={v:B.pivot(index='ccg',columns='year',values=v).reindex(units) for v in ('E','P')}
    for v in ('E','P'):
        f[v]=piv[v].stack().reindex(idx).to_numpy()
        f[v+'_b']=piv[v].reindex(columns=[t-h for t in years]).set_axis(years,axis=1).stack().reindex(idx).to_numpy()
    require(np.isfinite(f[['E','P','E_b','P_b']]).all().all(),'Missing baseline outcome or denominator; no row deletion permitted')
    require((f[['E','P','E_b','P_b']]>0).all().all(),'Nonpositive baseline outcome or denominator')
    f['y']=2.0*(f.E-f.E_b)/(f.E+f.E_b)
    lag={k:C.reindex(columns=[t-k for t in years]).set_axis(years,axis=1) for k in range(h+1)}
    wj=lam_weights(h,spec['lambda'])
    win=sum(wj[k]*lag[k] for k in range(h))
    f['G_win']=win.stack().reindex(idx).to_numpy()
    f['G_b']=lag[h].stack().reindex(idx).to_numpy()
    f['x_raw']=(f.G_win-f.G_b)/f.P_b
    f['national_shift']=(f.groupby('year').G_win.transform('sum')-f.groupby('year').G_b.transform('sum'))/f.groupby('year').P_b.transform('sum')
    exposures={}
    for t in years:
        hi=t-h-spec['exposure_gap']
        yy=list(range(hi-spec['exposure_length']+1,hi+1))
        require(C[yy].notna().all().all(),'Missing exposure spending')
        require(piv['P'][yy].notna().all().all(),'Missing exposure population')
        exposures[t]=C.loc[:,yy].sum(axis=1)/piv['P'].loc[:,yy].sum(axis=1,min_count=len(yy))
    f['exposure']=pd.DataFrame(exposures).reindex(units).stack().reindex(idx).to_numpy()
    f['z']=f.exposure*f.national_shift
    f['x']=wins(f.x_raw.to_numpy(),np.zeros(len(f),np.int64))
    f['weight']=1.0
    f['fe_county']=pd.factorize(f.ccg)[0].astype(np.int64)
    f['fe_year']=pd.factorize(f.year)[0].astype(np.int64)
    require(np.isfinite(f[['y','x_raw','x','z','exposure','national_shift']]).all().all(),'Nonfinite regression array')
    require(len(f)==spec['n_observations'],'Unexpected baseline sample size')
    require(f.groupby('ccg').size().eq(len(years)).all(),'Unbalanced county support')
    require((f.y.abs()<=2).all(),'MPGR bound failed')
    return f
