import numpy as np

def matching_and_activation(config, beliefs):
    """R06 (6a), (6b), (7), (8); delta normalisation uses rule participants."""
    l,t=config.rule_index.shape; b=len(beliefs[0])
    mask=config.participation_mask
    d=np.broadcast_to(config.attribute_weights,(l,t))*mask
    maximum=d.max(1,keepdims=True)
    exponents=np.divide(d,maximum,out=np.zeros_like(d),where=maximum>0)
    gathered=np.ones((b,l,t))
    for i in range(t):
        gathered[:,:,i]=beliefs[i][:,np.where(mask[:,i],config.rule_index[:,i],0)]
    if config.connective=='and':
        factors=np.ones_like(gathered)
        np.power(gathered,exponents[None,:,:],out=factors,where=exponents[None,:,:]>0)
        alpha=factors.prod(2)
    else:
        alpha=1-(1-gathered*exponents[None,:,:]).prod(2)
    weighted=alpha*config.rule_weights
    total=weighted.sum(1,keepdims=True)
    activation=np.divide(weighted,total,out=np.zeros_like(weighted),where=total>0)
    sums=np.stack([x.sum(1) for x in beliefs],axis=1)
    completeness=sums@mask.T/mask.sum(1)
    return activation, completeness, total[:,0]>0

def er_aggregate(activation, beliefs):
    """R06 ER recursion, retaining ignorance separately from weight discount.
    beliefs is [B,L,N] or [L,N]; return committed beliefs and ignorance.
    """
    w=np.asarray(activation,float)
    beta=np.asarray(beliefs,float)
    if beta.ndim==2: beta=np.broadcast_to(beta,(len(w),)+beta.shape)
    if w.ndim!=2 or beta.ndim!=3 or beta.shape[:2]!=w.shape: raise ValueError('ER dimension mismatch.')
    b,l,n=beta.shape
    mass=np.zeros((b,n)); residual=np.ones(b); discount=np.ones(b)
    for k in range(l):
        mk=w[:,k,None]*beta[:,k]
        hk=1-w[:,k]*beta[:,k].sum(1)
        bar=1-w[:,k]
        conflict=mass.sum(1)*mk.sum(1)-(mass*mk).sum(1)
        den=1-conflict
        if np.any(den<=0): raise ValueError('ER total conflict.')
        mass=(mass*mk+mass*hk[:,None]+residual[:,None]*mk)/den[:,None]
        residual=residual*hk/den
        discount=discount*bar/den
    den=1-discount
    valid=den>1e-14
    out=np.divide(mass,den[:,None],out=np.zeros_like(mass),where=valid[:,None])
    unknown=np.divide(residual-discount,den,out=np.ones(b),where=valid)
    # Roundoff only: do not renormalise committed belief.
    unknown=np.maximum(unknown,0)
    return out,unknown
