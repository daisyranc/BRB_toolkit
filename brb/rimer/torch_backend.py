"""Device-resident RIMER matching and ER, with explicit unknown mass."""
from dataclasses import dataclass, field
import numpy as np
import torch
from .transform import transform_attribute

@dataclass
class TorchRIMERResult:
    beliefs: torch.Tensor
    ignorance: torch.Tensor
    activation: torch.Tensor
    completeness: torch.Tensor
    status: list
    utility_interval: object=None
    trace: dict=field(default_factory=dict)

    def expected_value(self, utilities, *, decision_policy=None):
        u=torch.as_tensor(utilities,device=self.beliefs.device,dtype=self.beliefs.dtype)
        value=self.beliefs@u
        if decision_policy in {'midpoint','lower','upper'}:
            replacement={'midpoint':(u.min()+u.max())/2,'lower':u.min(),'upper':u.max()}[decision_policy]
            return value+self.ignorance*replacement
        if torch.any(self.ignorance.detach()>1e-9): raise ValueError('Incomplete output requires an explicit decision policy.')
        return value

class TorchRIMERBackend:
    def __init__(self,model,*,device='cpu',dtype=torch.float64,batch_size=2048):
        self.model=model; self.device=torch.device(device); self.dtype=dtype; self.batch_size=batch_size
        if self.device.type not in {'cpu','cuda'} or batch_size<1: raise ValueError('Invalid device/batch size.')
        if self.device.type=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable; choose CPU explicitly.')
        self.models=model.models if hasattr(model,'models') else {'model':model}
        self.structures={}
        for name,m in self.models.items():
            c=m.config
            self.structures[name]=(torch.as_tensor(c.rule_index,device=self.device,dtype=torch.long),torch.as_tensor(c.participation_mask,device=self.device),torch.as_tensor(c.complete_rules,device=self.device))

    def tensor(self,value): return torch.as_tensor(value,device=self.device,dtype=self.dtype)

    def predict(self,raw_inputs,*,parameters=None):
        if hasattr(self.model,'nodes'): raise ValueError('Use infer with named belief inputs for a hierarchy.')
        x=np.asarray(raw_inputs,object); c=self.model.config
        if x.ndim!=2 or x.shape[1]!=len(c.attributes): raise ValueError('Input shape mismatch.')
        return self.infer([transform_attribute(x[:,i],a) for i,a in enumerate(c.attributes)],parameters=parameters)

    def infer(self,attribute_beliefs,*,parameters=None,training_trial=False):
        if hasattr(self.model,'nodes'):
            trace={}
            for node in self.model.nodes:
                inputs=[]
                for key in node.inputs:
                    if key in trace:
                        x=trace[key].beliefs; mapping=(node.mappings or {}).get(key)
                        if mapping is not None: x=x@self.tensor(mapping)
                    else: x=attribute_beliefs[key]
                    inputs.append(x)
                trace[node.name]=self._node(node.name,inputs,(parameters or {}).get(node.name,{}),training_trial)
            result=trace[self.model.output]; result.trace={k:v for k,v in trace.items() if k!=self.model.output}; return result
        return self._node('model',attribute_beliefs,(parameters or {}).get('model',parameters or {}),training_trial)

    def _node(self,name,inputs,parameters,trial):
        c=self.models[name].config; index,mask,complete=self.structures[name]
        if len(inputs)!=len(c.attributes): raise ValueError('Attribute count mismatch.')
        x=[self.tensor(a) for a in inputs]
        batch=len(x[0])
        for a,value in zip(c.attributes,x):
            if value.shape!=(batch,len(a.references)) or not torch.isfinite(value).all() or (value<0).any() or (value.sum(1)>1+1e-7).any(): raise ValueError('Invalid input beliefs.')
        beta=parameters['beta'] if 'beta' in parameters else self.tensor(c.rule_beliefs)
        theta=parameters['theta'] if 'theta' in parameters else self.tensor(c.rule_weights)
        delta=parameters['delta'] if 'delta' in parameters else self.tensor(c.attribute_weights)
        utilities=parameters['utilities'] if 'utilities' in parameters else None if c.utilities is None else self.tensor(c.utilities)
        if trial:
            sums=beta.sum(1,keepdim=True); normalise=complete[:,None]|(sums>1)
            beta=torch.where(normalise,beta/sums.clamp_min(torch.finfo(self.dtype).tiny),beta)
        outputs=[]
        for start in range(0,batch,self.batch_size):
            outputs.append(self._calculate(c,index,mask,beta,theta,delta,[v[start:start+self.batch_size] for v in x]))
        if not outputs:
            b=self.tensor(np.empty((0,len(c.consequents)))); r=self.tensor(np.empty((0,len(index)))); return TorchRIMERResult(b,self.tensor([]),r,r,[],None if utilities is None else self.tensor(np.empty((0,2))))
        belief,unknown,w,completeness,active=[torch.cat([v[i] for v in outputs]) for i in range(5)]
        interval=None if utilities is None else torch.stack((belief@utilities+unknown*utilities.min(),belief@utilities+unknown*utilities.max()),1)
        return TorchRIMERResult(belief,unknown,w,completeness,['ok' if v else 'no_activation' for v in active.detach().cpu().tolist()],interval)

    def _calculate(self,c,index,mask,beta,theta,delta,x):
        d=delta.expand(index.shape)*mask
        exponents=d/d.max(1,keepdim=True).values.clamp_min(1e-15)
        gathered=torch.stack([v[:,index[:,i].clamp_min(0)] for i,v in enumerate(x)],2)
        if c.connective=='and':
            factors=torch.where(exponents[None]>0,gathered.clamp_min(1e-15).pow(exponents[None]),torch.ones_like(gathered))
            alpha=factors.prod(2)
            alpha=torch.where(((gathered==0)&(exponents[None]>0)).any(2),torch.zeros_like(alpha),alpha)
        else: alpha=1-(1-gathered*exponents[None]).prod(2)
        weighted=alpha*theta; total=weighted.sum(1,keepdim=True); w=weighted/total.clamp_min(1e-15)
        completeness=torch.stack([v.sum(1) for v in x],1)@mask.to(self.dtype).T/mask.sum(1)
        rb=beta[None]*completeness[:,:,None]
        h=1-w*rb.sum(2); hp=h.prod(1); discount=(1-w).prod(1)
        committed=(w[:,:,None]*rb+h[:,:,None]).prod(1)
        denominator=committed.sum(1)-(beta.shape[1]-1)*hp-discount
        active=(total[:,0]>0)&(denominator>1e-14)
        safe=torch.where(active,denominator,torch.ones_like(denominator))
        # These products are mathematically ordered. Separate reductions can
        # reverse them by roundoff when a consequent has zero committed mass.
        # Preserve incomplete belief; do not redistribute unknown mass.
        beliefs=torch.where(active[:,None],(committed-hp[:,None]).clamp_min(0)/safe[:,None],torch.zeros_like(committed))
        unknown=torch.where(active,(hp-discount)/safe,torch.ones_like(safe)).clamp_min(0)
        return beliefs,unknown,w,completeness,total[:,0]>0
