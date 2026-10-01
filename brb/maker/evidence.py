from dataclasses import dataclass
import numpy as np
from ..paper_common import validate_beliefs
from ..paper_common.validation import unit_interval

@dataclass
class Evidence:
    probabilities: object
    weights: object
    reliability: float
    space: object

    def __post_init__(self):
        self.probabilities=validate_beliefs(self.probabilities,complete=True)
        if self.probabilities.shape!=(len(self.space),): raise ValueError('Evidence shape mismatch.')
        self.weights=np.broadcast_to(unit_interval(self.weights,'weights'),self.probabilities.shape).copy()
        self.reliability=float(unit_interval(self.reliability,'reliability'))

    @property
    def mass(self): return self.weights*self.probabilities

    @property
    def projection_rate(self):
        denominator=self.mass.sum()+1-self.reliability
        if denominator<=0: raise ValueError('Zero projection denominator.')
        return 1/denominator

@dataclass
class JointEvidence:
    probabilities: object
    weights: object
    reliability: float
    joint_support: object
    weight_indices: object=None
    projection_index: object=None

class EvidenceTable:
    def __init__(self,space,values,probabilities,*,counts=None,source=None,joint_support=None):
        self.space=space;self.values=tuple(values);self.source=source
        if len(set(self.values))!=len(self.values): raise ValueError('Duplicate evidence values.')
        self.probabilities=validate_beliefs(probabilities,complete=True)
        if self.probabilities.shape!=(len(self.values),len(space)): raise ValueError('Table shape mismatch.')
        self.counts=None if counts is None else np.array(counts,float,copy=True)
        self.joint_support=None if joint_support is None else np.array(joint_support,float,copy=True)
        if self.joint_support is not None:
            if self.joint_support.shape!=(len(self.values),len(space),len(space)) or not np.isfinite(self.joint_support).all() or (self.joint_support<0).any():
                raise ValueError('Joint support shape/values invalid.')
            intersections=space.intersections
            acquired=np.stack([self.joint_support[:,intersections==mask].sum(1) for mask in space.masks],1)
            if not np.allclose(acquired,self.probabilities,atol=1e-10):
                raise ValueError('Joint supports must acquire the supplied intersection likelihood.')

    @classmethod
    def from_counts(cls,space,values,counts,*,source=None,joint_support=None):
        c=np.asarray(counts,float)
        if c.shape!=(len(values),len(space)) or not np.isfinite(c).all() or (c<0).any() or (c.sum(0)<=0).any():
            raise ValueError('Counts require positive state totals and finite nonnegative entries.')
        likelihood=c/c.sum(0)
        if (likelihood.sum(1)<=0).any(): raise ValueError('Every input value needs support.')
        p=likelihood/likelihood.sum(1,keepdims=True)
        return cls(space,values,p,counts=c,source=source,joint_support=joint_support)

    def index(self,value):
        try: return self.values.index(value)
        except ValueError as e: raise ValueError(f'Unknown evidence value: {value}') from e

    def acquire(self,value,weights,reliabilities):
        i=self.index(value)
        return Evidence(self.probabilities[i],weights,np.asarray(reliabilities)[i],self.space)
