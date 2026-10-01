import numpy as np
from .evidence import JointEvidence
from .combination import combine_pair,extend_result,_finish

class MAKERModel:
    """Ordered prefix dependency model. Missing inputs select their available subset."""
    def __init__(self,tables,parameters,*,mode='dependent',order=None,prior=None):
        if mode not in ('dependent','independent','exclusive'): raise ValueError('Invalid mode.')
        self.tables=dict(tables);self.mode=mode;self.prior=prior
        self.order=tuple(order or [key[0] for key in tables if len(key)==1 and key!=prior])
        if len(set(self.order))!=len(self.order) or not self.order: raise ValueError('Invalid variable order.')
        if set(parameters)!=set(tables): raise ValueError('Parameters and acquisition tables must match.')
        self.parameters={}
        spaces=[table.space for table in tables.values()]
        if not spaces or any(space!=spaces[0] for space in spaces): raise ValueError('Inconsistent state spaces.')
        self.space=spaces[0]
        for key,table in tables.items():
            if not isinstance(key,tuple) or not key: raise ValueError('Table keys must be variable tuples.')
            p=parameters[key]
            w=np.asarray(p['weights'],float);r=np.asarray(p['reliabilities'],float)
            if w.shape!=(len(self.space),) or r.shape!=(len(table.values),): raise ValueError('Parameter shape mismatch.')
            if not np.isfinite(w).all() or not np.isfinite(r).all() or np.any((w<0)|(w>1)) or np.any((r<0)|(r>1)):
                raise ValueError('Weights/reliabilities must lie in [0,1].')
            self.parameters[key]={'weights':w.copy(),'reliabilities':r.copy()}
        if any((k,) not in tables for k in self.order) or (prior is not None and prior not in tables):
            raise ValueError('Missing individual/prior table.')

    def acquire(self,key,value):
        p=self.parameters[key]
        return self.tables[key].acquire(value,p['weights'],p['reliabilities'])

    def _joint(self,key,value):
        if key not in self.tables: raise ValueError(f'Missing prefix joint table {key}; cannot assume independence.')
        t=self.tables[key]
        if t.joint_support is None: raise ValueError(f'Missing prefix joint support for {key}.')
        p=self.parameters[key];i=t.index(value)
        return JointEvidence(t.probabilities[i],p['weights'],p['reliabilities'][i],t.joint_support[i])

    def infer(self,values,*,include_prior=True):
        supplied=values if isinstance(values,dict) else dict(zip(self.order,values))
        if not isinstance(values,dict) and len(values)!=len(self.order): raise ValueError('Input value count mismatch.')
        if set(supplied)-set(self.order): raise ValueError('Unknown input variable.')
        keys=[k for k in self.order if k in supplied and supplied[k] is not None]
        if not keys:
            if self.prior is None or not include_prior: raise ValueError('No observed input or prior.')
            a=self.acquire(self.prior,self.tables[self.prior].values[0])
            return _finish(a.mass,1-a.reliability,self.space,[])
        a=self.acquire((keys[0],),supplied[keys[0]])
        result=None
        for l in range(1,len(keys)):
            prefix=tuple(keys[:l]);jointkey=tuple(keys[:l+1])
            b=self.acquire((keys[l],),supplied[keys[l]])
            joint=self._joint(jointkey,tuple(supplied[k] for k in jointkey)) if self.mode=='dependent' else None
            if result is None: result=combine_pair(a,b,joint=joint,mode=self.mode)
            else:
                acquired=self.acquire(prefix,tuple(supplied[k] for k in prefix)) if self.mode=='dependent' else None
                result=extend_result(result,b,prefix=acquired,joint=joint,mode=self.mode)
        if self.prior is not None and include_prior:
            prior=self.acquire(self.prior,self.tables[self.prior].values[0])
            result=combine_pair(a,prior,mode='independent') if result is None else extend_result(result,prior)
        if result is None: result=_finish(a.mass,1-a.reliability,self.space,[])
        return result
