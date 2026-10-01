import numpy as np

def transform_attribute(values, attribute):
    """R06 numeric equivalence transformation; input NaN is zero commitment."""
    values=np.asarray(values,object).reshape(-1)
    out=np.zeros((len(values),len(attribute.references)))
    if attribute.kind=='categorical':
        for i,v in enumerate(values):
            if v is None or (isinstance(v,(float,np.floating)) and np.isnan(v)): continue
            if v not in attribute.references: raise ValueError(f'Unknown category {v} for {attribute.name}.')
            out[i,attribute.references.index(v)]=1
        return out
    order=np.argsort(np.asarray(attribute.references,float))
    refs=np.asarray(attribute.references,float)[order]
    for i,v in enumerate(values):
        if v is None or np.isnan(float(v)): continue
        x=float(v)
        if not np.isfinite(x): raise ValueError('Numeric inputs must be finite or NaN.')
        if x<=refs[0]: out[i,order[0]]=1
        elif x>=refs[-1]: out[i,order[-1]]=1
        else:
            hi=np.searchsorted(refs,x); lo=hi-1
            p=(x-refs[lo])/(refs[hi]-refs[lo])
            out[i,order[lo]]=1-p; out[i,order[hi]]=p
    return out
