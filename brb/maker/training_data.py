from dataclasses import dataclass
import numpy as np

@dataclass
class ObservationGroup:
    key: tuple
    values: object
    counts: object
    targets: object=None

    def __post_init__(self):
        self.key=tuple(self.key);self.values=tuple(self.values);self.counts=np.asarray(self.counts,float)
        if self.counts.ndim!=2 or self.counts.shape[0]!=len(self.values) or not np.isfinite(self.counts).all() or (self.counts<0).any() or (self.counts.sum(1)<=0).any():
            raise ValueError('Observation counts must have positive input totals.')
        k=self.counts.shape[1]
        if self.targets is None: self.targets=np.broadcast_to(np.eye(k),(len(self.values),k,k)).copy()
        else:
            from ..paper_common import validate_beliefs
            self.targets=validate_beliefs(self.targets,complete=True)
            if self.targets.shape!=(len(self.values),k,k): raise ValueError('Observed target shape mismatch.')

class MAKERTrainingData:
    def __init__(self,single_groups,prefix_groups,reference_parameters,counts=None,prior=None,*,bounds=None):
        self.single_groups=list(single_groups);self.prefix_groups=list(prefix_groups)
        self.groups=self.single_groups+self.prefix_groups;self.prior=prior
        if not self.groups: raise ValueError('Training observations required.')
        self.total=float(sum(g.counts.sum() for g in self.groups))
        if counts is not None and not np.isclose(counts,self.total): raise ValueError('Total frequency mismatch.')
        self.reference_parameters={k:{n:np.array(a,float,copy=True) for n,a in p.items()} for k,p in reference_parameters.items()}
        self.bounds=bounds or {}
