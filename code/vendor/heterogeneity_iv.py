"""Unchanged functions extracted from the original sandbox source."""
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from scipy.stats import t
from baseline_math import demean, ssc_k, fit2sls
def require(ok, message):
    if not ok: raise RuntimeError(message)

def iv(d,endog,excluded,exog,fe):
    w=d.weight.to_numpy(float);f1=pd.factorize(d[fe])[0];f2=pd.factorize(d.year)[0];cl,labels=pd.factorize(d.unit,sort=True)
    names=['std_y']+endog+excluded+exog
    raw=d[names].to_numpy(float);scale=np.maximum(abs(raw).max(axis=0),1e-30)
    m=demean(raw/scale,f1,f2,w)
    for ff in [f1,f2]:
        den=np.bincount(ff,weights=w)
        for v in m.T:require(abs(np.bincount(ff,weights=w*v)/den).max()<1e-9,'Incomplete FE absorption')
    nend=len(endog);nex=len(excluded);ne=len(exog)
    y=m[:,0];x0=m[:,1:1+nend];z0=m[:,1+nend:1+nend+nex];e=m[:,-ne:] if ne else np.empty((len(d),0))
    x=np.column_stack([x0,e]);z=np.column_stack([z0,e])
    zx=z.T@(w[:,None]*x);zz=z.T@(w[:,None]*z)
    require(np.linalg.matrix_rank(zx)==len(x[0]) and np.linalg.matrix_rank(zz)==len(z[0]),'Rank failure; stop rather than drop controls')
    b=np.linalg.solve(zx,z.T@(w*y));pred=z@np.linalg.solve(zz,zx)
    require(np.allclose(b,np.linalg.lstsq(pred*np.sqrt(w)[:,None],y*np.sqrt(w),rcond=None)[0],rtol=1e-7,atol=1e-9),'Projected-IV mismatch')
    G=len(labels);k=ssc_k(f1,f2,cl)+x.shape[1]-1;ssc=G/(G-1)*(len(d)-1)/(len(d)-k)
    residual=y-x@b;scores=np.zeros((G,x.shape[1]));np.add.at(scores,cl,z*(w*residual)[:,None])
    xs=np.r_[scale[1:1+nend],scale[-ne:] if ne else []];factor=scale[0]/xs
    psi=scores@np.linalg.inv(zx).T*np.sqrt(ssc)*factor
    result=[]
    for j,name in enumerate(endog+exog):
        beta=b[j]*factor[j];se=np.linalg.norm(psi[:,j]);F=np.nan
        if j<nend:
            pi=np.linalg.solve(zz,z.T@(w*x0[:,j]));v=x0[:,j]-z@pi
            ss=np.zeros((G,z.shape[1]));np.add.at(ss,cl,z*(w*v)[:,None]);bread=np.linalg.inv(zz)
            V=ssc*bread@(ss.T@ss)@bread.T
            F=float(pi[:nex]@np.linalg.solve(V[:nex,:nex],pi[:nex])/nex)
        result.append(dict(variable=name,beta=beta,se=se,p=2*t.sf(abs(beta/se),G-1),F=F))
    export=pd.DataFrame(m*scale,columns=['y']+endog+excluded+exog)
    export['weight']=w;export['unit']=cl.astype(np.int32)
    norm=zx/np.sqrt((w[:,None]*z**2).sum(axis=0))[:,None]/np.sqrt((w[:,None]*x**2).sum(axis=0))[None,:]
    return result,psi,export,dict(k=int(k),n=len(d),G=G,condition=float(np.linalg.cond(norm)))
