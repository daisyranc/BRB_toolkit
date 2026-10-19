from dataclasses import dataclass
from itertools import product
from math import prod
import numpy as np
from ..paper_common.validation import validate_beliefs, unit_interval

@dataclass
class RIMERAttribute:
    name: str
    references: list
    kind: str = 'numeric'

    def __post_init__(self):
        self.references = list(self.references)
        if len(self.references) < 2 or len(set(self.references)) != len(self.references):
            raise ValueError('Reference values must be distinct; at least two are required.')
        if self.kind not in ('numeric','categorical'): raise ValueError('Unsupported attribute kind.')
        if self.kind == 'numeric' and not np.isfinite(np.asarray(self.references,float)).all():
            raise ValueError('Numeric references must be finite.')

@dataclass
class RIMERConfig:
    attributes: list
    consequents: list
    rule_index: object
    rule_beliefs: object
    rule_weights: object
    attribute_weights: object
    complete_rules: object = None
    participation_mask: object = None
    connective: str = 'and'
    utilities: object = None
    max_rules: int = 10000

    def __post_init__(self):
        t=len(self.attributes)
        if not t or len({a.name for a in self.attributes}) != t: raise ValueError('Invalid attribute names.')
        if len(self.consequents)<2 or len(set(self.consequents))!=len(self.consequents): raise ValueError('Invalid consequents.')
        if self.rule_index is None:
            n=prod(len(a.references) for a in self.attributes)
            if n>self.max_rules: raise ValueError('Rule resource limit exceeded.')
            self.rule_index=list(product(*[range(len(a.references)) for a in self.attributes]))
        raw=np.asarray(self.rule_index)
        if not np.equal(raw,raw.astype(int)).all(): raise ValueError('Rule indices must be integers.')
        self.rule_index=raw.astype(int)
        if self.rule_index.ndim!=2 or self.rule_index.shape[1]!=t: raise ValueError('Rule dimension mismatch.')
        l=len(self.rule_index)
        if not 0<l<=self.max_rules: raise ValueError('Rule resource limit exceeded.')
        self.participation_mask=(self.rule_index>=0) if self.participation_mask is None else np.asarray(self.participation_mask,bool)
        if self.participation_mask.shape!=(l,t) or not self.participation_mask.any(1).all():
            raise ValueError('Every rule must use at least one antecedent.')
        for i,a in enumerate(self.attributes):
            ix=self.rule_index[self.participation_mask[:,i],i]
            if ((ix<0)|(ix>=len(a.references))).any(): raise ValueError('Rule reference out of range.')
        self.rule_beliefs=validate_beliefs(self.rule_beliefs)
        if self.rule_beliefs.shape!=(l,len(self.consequents)): raise ValueError('Consequent dimension mismatch.')
        self.rule_weights=unit_interval(self.rule_weights,'Rule weights')
        if self.rule_weights.shape!=(l,) or not self.rule_weights.any(): raise ValueError('Invalid rule weights.')
        self.attribute_weights=unit_interval(self.attribute_weights,'Attribute weights')
        if self.attribute_weights.shape not in ((t,),(l,t)): raise ValueError('Attribute weight dimension mismatch.')
        delta=np.broadcast_to(self.attribute_weights,(l,t))
        if not (delta*self.participation_mask).max(1).all(): raise ValueError('Every rule requires a positive attribute weight.')
        self.complete_rules=np.isclose(self.rule_beliefs.sum(1),1) if self.complete_rules is None else np.broadcast_to(self.complete_rules,(l,)).astype(bool).copy()
        if not np.allclose(self.rule_beliefs[self.complete_rules].sum(1),1): raise ValueError('Complete rule beliefs must sum to one.')
        if self.connective not in ('and','or'): raise ValueError('Only and/or connectives are supported.')
        if self.utilities is not None:
            # Utilities are trainable model parameters, not a shared template.
            self.utilities=np.array(self.utilities,dtype=float,copy=True)
            if self.utilities.shape!=(len(self.consequents),) or not np.isfinite(self.utilities).all(): raise ValueError('Invalid utilities.')
