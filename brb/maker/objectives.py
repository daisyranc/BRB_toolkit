import numpy as np

def paper_objectives(model,data):
    """M25 Eqs 86–95; nested frequency weights cancel to count/total."""
    f1=[];f2=[]
    for group in data.groups:
        if group.key not in model.parameters or group.key not in data.reference_parameters:
            raise ValueError('Observed group requires parameters and reference parameters.')
        parameters=model.parameters[group.key];reference=data.reference_parameters[group.key]
        table=model.tables[group.key]
        if group.counts.shape[1]!=len(model.space): raise ValueError('Observation state dimension mismatch.')
        for kind,a in parameters.items():
            if kind not in reference or np.shape(reference[kind])!=a.shape:
                raise ValueError('Invalid reference parameter dimensions.')
            v=np.asarray(reference[kind])
            if not np.isfinite(v).all() or np.any((v<0)|(v>1)): raise ValueError('Invalid reference parameters.')
        for i,value in enumerate(group.values):
            inputs={group.key[0]:value} if len(group.key)==1 else dict(zip(group.key,value))
            result=model.infer(inputs)
            if result.status!='ok': raise ValueError('Undefined training prediction.')
            observed=group.counts[i];n=float(observed.sum());index=table.index(value)
            error=.5*np.sum((group.targets[i]-result.probabilities[None,:])**2,axis=1)
            f1.extend(observed/data.total*error)
            weight_error=float(np.sum(observed/n*(reference['weights']-parameters['weights'])**2))
            reliability_error=float((reference['reliabilities'][index]-parameters['reliabilities'][index])**2)
            f2.extend([n/data.total*weight_error,n/data.total*reliability_error])
    return np.asarray(f1),np.asarray(f2)
