import numpy as np
from .results import MAKERResult
from .evidence import Evidence

def dependency_indices(marginal_a,marginal_b,joint_support):
    a=np.asarray(marginal_a,float);b=np.asarray(marginal_b,float);j=np.asarray(joint_support,float)
    den=np.outer(a,b)
    if j.shape!=den.shape or not np.isfinite(j).all() or (j<0).any(): raise ValueError('Invalid joint support.')
    if np.any((den==0)&(j>0)): raise ValueError('Positive joint support with zero marginal support.')
    return np.divide(j,den,out=np.zeros_like(j),where=den>0)

def _finish(mass,residual,space,trace,status='ok'):
    total=float(np.sum(mass));den=total+residual
    if den<=0 or total<=0:
        return MAKERResult(np.zeros(len(space)),np.zeros(len(space)),1.,space,'no_support',trace)
    return MAKERResult(mass/total,mass/den,float(residual/den),space,status,trace)

def _intersection_sum(products,space):
    return np.array([products[space.intersections==m].sum() for m in space.masks])

def _dependent_products(a,b,joint):
    if joint is None: raise ValueError('Dependent combination requires explicit joint evidence.')
    j=Evidence(joint.probabilities,joint.weights,joint.reliability,a.space)
    alpha=dependency_indices(a.probabilities,b.probabilities,joint.joint_support)
    weights=joint.weight_indices
    if weights is None:
        numerator=np.zeros((len(a.space),len(a.space)))
        for mask,w in zip(a.space.masks,j.weights): numerator[a.space.intersections==mask]=w
        den=np.outer(a.weights,b.weights)
        if np.any((den==0)&(numerator*alpha>0)): raise ValueError('Undefined weight index at zero marginal weight.')
        weights=np.divide(numerator,den,out=np.zeros_like(den),where=den>0)
    weights=np.asarray(weights,float)
    projection=joint.projection_index
    if projection is None: projection=j.projection_rate/(a.projection_rate*b.projection_rate)
    if weights.shape!=alpha.shape or not np.isfinite(weights).all() or (weights<0).any() or not np.isfinite(projection) or projection<0:
        raise ValueError('Invalid weight/projection index.')
    return projection*weights*alpha,alpha,weights,float(projection)

def combine_pair(a,b,*,joint=None,mode='dependent',derivative_rates=None,augment=False,tau=1e-4,perturbations=(1e-8,1e-8)):
    """M25 Theorem 1/Eqs 59–68; 2^Theta residual stays separate."""
    if a.space!=b.space: raise ValueError('State space mismatch.')
    space=a.space;ma=a.mass;mb=b.mass;ua=1-a.reliability;ub=1-b.reliability
    products=np.outer(ma,mb);trace=[];status='ok'
    if mode=='dependent':
        coeff,alpha,wi,pi=_dependent_products(a,b,joint);products*=coeff
        trace.append({'dependency_indices':alpha,'weight_indices':wi,'projection_index':pi})
    elif mode=='exclusive': products[:]=0
    elif mode!='independent': raise ValueError('Unknown combination mode.')
    support=_intersection_sum(products,space)
    if mode=='independent' and augment and ua==0 and ub==0:
        if not np.all(a.weights==a.weights[0]) or not np.all(b.weights==b.weights[0]):
            raise ValueError('Augmented Dempster requires state-constant weights.')
        sigma=float(_intersection_sum(np.outer(a.probabilities,b.probabilities),space).sum())
        d1,d2=perturbations
        if not 0<tau<1 or not (0<d1<tau and 0<d2<tau): raise ValueError('Invalid reliability perturbation.')
        if sigma==0: derivative_rates=(d1,d2);mode='exclusive'
        elif sigma<=tau:
            ua=4*d1*sigma/tau*(1-sigma/tau);ub=4*d2*sigma/tau*(1-sigma/tau)
            status='augmented_dempster'
    mass=ub*ma+ua*mb+support;residual=ua*ub
    if mass.sum()<=0 and residual==0:
        if derivative_rates is None:
            return MAKERResult(np.zeros(len(space)),np.zeros(len(space)),1.,space,'undefined_limit',trace)
        rates=np.asarray(derivative_rates,float)
        if rates.shape!=(2,) or not np.isfinite(rates).all() or (rates<=0).any(): raise ValueError('Positive limiting unreliability rates required.')
        mass=rates[1]*ma+rates[0]*mb;status='differential_limit'
    trace.append({'unnormalised_mass':mass.copy(),'residual':float(residual),'normalizer':float(mass.sum()+residual)})
    return _finish(mass,residual,space,trace,status)

def extend_result(current,b,*,prefix=None,joint=None,mode='independent'):
    """M25 Eqs 70–72. Prefix acquisition is required for dependent indices."""
    if current.space!=b.space or current.status not in ('ok','augmented_dempster','differential_limit'):
        raise ValueError('Cannot extend incompatible or undefined result.')
    products=np.outer(current.erps_mass,b.mass);trace=list(current.trace)
    if mode=='dependent':
        if prefix is None: raise ValueError('Dependent recursion requires the acquired prefix evidence.')
        coeff,alpha,wi,pi=_dependent_products(prefix,b,joint);products*=coeff
        trace.append({'dependency_indices':alpha,'weight_indices':wi,'projection_index':pi})
    elif mode=='exclusive': products[:]=0
    elif mode!='independent': raise ValueError('Unknown combination mode.')
    mass=(1-b.reliability)*current.erps_mass+current.untrustworthiness*b.mass+_intersection_sum(products,b.space)
    residual=current.untrustworthiness*(1-b.reliability)
    trace.append({'unnormalised_mass':mass.copy(),'residual':residual,'normalizer':float(mass.sum()+residual)})
    return _finish(mass,residual,b.space,trace)
