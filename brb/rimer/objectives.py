from dataclasses import dataclass
import numpy as np

@dataclass
class RIMERSupervision:
    numeric: object = None
    beliefs: object = None
    observed_mask: object = None
    objective_weights: object = None
    scales: object = None
    ideal: object = None
    nadir: object = None
    decision_policy: object = None

def _mask(spec,kind,shape,objective):
    if isinstance(spec,dict): spec=spec.get(kind)
    if spec is None: return np.ones(shape,bool)
    a=np.asarray(spec,bool)
    # A 1-D mask always denotes samples, never consequent columns.
    if a.shape==(shape[0],):
        return a if len(shape)==1 else np.broadcast_to(a[:,None],shape)
    if kind=='numeric' and objective=='mixed' and a.ndim==2:
        # A full belief mask is specific to belief supervision.
        return np.ones(shape,bool)
    if a.shape!=shape: raise ValueError(f'{kind} observation mask shape mismatch.')
    return a

def objective_components(result, supervision, utilities, objective):
    """R07 (12), (15), (16), (19); masks are explicit, never imputed zeros."""
    s=supervision; terms=[]
    if objective in ('numeric','mixed'):
        if s.numeric is None or utilities is None: raise ValueError('Numeric supervision requires targets and utilities.')
        y=np.asarray(s.numeric,float)
        if y.shape!=(len(result.beliefs),): raise ValueError('Numeric target shape mismatch.')
        mask=np.isfinite(y)&_mask(s.observed_mask,'numeric',y.shape,objective)
        if not mask.any(): raise ValueError('No observed numeric targets.')
        if s.decision_policy is None:
            if np.any(result.ignorance[mask]>1e-9): raise ValueError('Incomplete observed output requires an explicit decision policy.')
            pred=result.beliefs@np.asarray(utilities)
        else: pred=result.expected_value(utilities,decision_policy=s.decision_policy)
        terms.append(np.mean((pred[mask]-y[mask])**2))
    if objective in ('belief','mixed'):
        if s.beliefs is None: raise ValueError('Belief targets required.')
        y=np.asarray(s.beliefs,float)
        if y.shape!=result.beliefs.shape: raise ValueError('Belief target shape mismatch.')
        mask=np.isfinite(y)&_mask(s.observed_mask,'beliefs',y.shape,objective)
        if np.any((y[mask]<0)|(y[mask]>1)): raise ValueError('Invalid observed target beliefs.')
        for j in range(y.shape[1]):
            valid=mask[:,j]
            if not valid.any(): raise ValueError('Every belief objective needs observed targets.')
            terms.append(np.mean((result.beliefs[valid,j]-y[valid,j])**2))
    a=np.asarray(terms)
    if not len(a): raise ValueError('Invalid supervision objective.')
    if objective!='numeric':
        ideal=np.zeros_like(a) if s.ideal is None else np.broadcast_to(s.ideal,a.shape)
        nadir=np.ones_like(a) if s.nadir is None else np.broadcast_to(s.nadir,a.shape)
        if np.any(nadir<=ideal): raise ValueError('Nadir must exceed ideal.')
        weights=np.ones_like(a) if s.objective_weights is None else np.broadcast_to(s.objective_weights,a.shape)
        if (weights<0).any() or not weights.any(): raise ValueError('Invalid objective weights.')
        scales=np.ones_like(a) if s.scales is None else np.broadcast_to(s.scales,a.shape)
        if np.any(scales<=0): raise ValueError('Scales must be positive.')
        a=weights*(a-ideal)/(nadir-ideal)/scales
    return a
