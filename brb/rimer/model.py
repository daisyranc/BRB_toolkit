import numpy as np
from ..paper_common import validate_beliefs
from .transform import transform_attribute
from .inference import matching_and_activation,er_aggregate
from .results import RIMERResult

class RIMERModel:
    def __init__(self, config): self.config=config

    def predict(self, raw_inputs):
        x=np.asarray(raw_inputs,object)
        if x.ndim!=2 or x.shape[1]!=len(self.config.attributes): raise ValueError('Input shape mismatch.')
        return self.infer([transform_attribute(x[:,i],a) for i,a in enumerate(self.config.attributes)])

    def infer(self, attribute_beliefs):
        return self._infer(attribute_beliefs)

    def _infer(self, attribute_beliefs, *, training_trial=False):
        c=self.config
        if len(attribute_beliefs)!=len(c.attributes): raise ValueError('Attribute count mismatch.')
        beliefs=[validate_beliefs(x) for x in attribute_beliefs]
        batch=beliefs[0].shape[0]
        for a,x in zip(c.attributes,beliefs):
            if x.shape!=(batch,len(a.references)): raise ValueError('Belief input dimensions mismatch.')
        w,completeness,active=matching_and_activation(c,beliefs)
        rule_beta=c.rule_beliefs.copy()
        if training_trial:
            # Extend the constrained objective to SLSQP's off-manifold probes.
            # The stored parameters and equality constraints remain untouched.
            sums=rule_beta.sum(1)
            selected=c.complete_rules | (sums>1)
            rule_beta[selected]/=sums[selected,None]
        beta=rule_beta[None,:,:]*completeness[:,:,None]
        out,unknown=er_aggregate(w,beta)
        interval=None
        if c.utilities is not None:
            v=out@c.utilities
            interval=np.stack([v+unknown*c.utilities.min(),v+unknown*c.utilities.max()],1)
        return RIMERResult(out,unknown,w,completeness,np.where(active,'ok','no_activation'),interval)
