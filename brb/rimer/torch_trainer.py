"""Constrained R07 objectives using the general Torch RIMER backend."""
import numpy as np
import torch
from ..optimization import TrainingConfig, solve
from ..optimization.constraints import TensorLayout
from ..paper_common import ParameterBlock,ParameterLayout,TrainingResult
from .torch_backend import TorchRIMERBackend
from .objectives import _mask

def components(result,s,utilities,objective):
    terms=[]; tensor=lambda a:torch.tensor(np.array(a,copy=True),device=result.beliefs.device,dtype=result.beliefs.dtype)
    if objective in {'numeric','mixed'}:
        if s.numeric is None or utilities is None: raise ValueError('Numeric targets and utilities required.')
        y=np.asarray(s.numeric,float)
        if y.shape!=(len(result.beliefs),): raise ValueError('Numeric target shape mismatch.')
        mask=np.isfinite(y)&_mask(s.observed_mask,'numeric',y.shape,objective)
        if not mask.any(): raise ValueError('No observed numeric targets.')
        if s.decision_policy is None:
            if torch.any(result.ignorance[mask].detach()>1e-9): raise ValueError('Incomplete observed output requires an explicit decision policy.')
            pred=result.beliefs@torch.as_tensor(utilities,device=result.beliefs.device,dtype=result.beliefs.dtype)
        else: pred=result.expected_value(utilities,decision_policy=s.decision_policy)
        terms.append((pred[mask]-tensor(y[mask])).square().mean())
    if objective in {'belief','mixed'}:
        y=np.asarray(s.beliefs,float)
        if y.shape!=tuple(result.beliefs.shape): raise ValueError('Belief target shape mismatch.')
        mask=np.isfinite(y)&_mask(s.observed_mask,'beliefs',y.shape,objective)
        if np.any((y[mask]<0)|(y[mask]>1)): raise ValueError('Invalid observed target beliefs.')
        for j in range(y.shape[1]):
            valid=mask[:,j]
            if not valid.any(): raise ValueError('Every belief objective needs observed targets.')
            terms.append((result.beliefs[valid,j]-tensor(y[valid,j])).square().mean())
    values=torch.stack(terms)
    if objective!='numeric':
        ideal=tensor(np.zeros(len(terms)) if s.ideal is None else np.broadcast_to(s.ideal,(len(terms),)))
        nadir=tensor(np.ones(len(terms)) if s.nadir is None else np.broadcast_to(s.nadir,(len(terms),)))
        weights=tensor(np.ones(len(terms)) if s.objective_weights is None else np.broadcast_to(s.objective_weights,(len(terms),)))
        scales=tensor(np.ones(len(terms)) if s.scales is None else np.broadcast_to(s.scales,(len(terms),)))
        if (nadir<=ideal).any() or (weights<0).any() or not weights.any() or (scales<=0).any(): raise ValueError('Invalid minimax normalization.')
        values=weights*(values-ideal)/(nadir-ideal)/scales
    return values

class TorchRIMERTrainer:
    def __init__(self,objective='numeric',*,config=None):
        if objective not in {'numeric','belief','mixed'}: raise ValueError('Unknown RIMER objective.')
        self.objective=objective; self.config=config or TrainingConfig()

    def fit(self,model,inputs,supervision,*,trainable=None,bounds=None,expert_constraints=(),validation=None,objective_fn=None):
        cfg=self.config; cfg.check_device()
        backend=TorchRIMERBackend(model,device=cfg.device,dtype=cfg.torch_dtype,batch_size=cfg.batch_size)
        models=backend.models; trainable={'beta':True,'theta':True,'delta':True} if trainable is None else trainable; bounds=bounds or {}; blocks=[]
        output=getattr(model,'output','model')
        signs={}; complete={}
        for name,m in models.items():
            c=m.config
            for kind,a in [('beta',c.rule_beliefs),('theta',c.rule_weights),('delta',c.attribute_weights),('utilities',c.utilities)]:
                if a is None or kind=='utilities' and name!=output: continue
                key=name+'.'+kind; flag=trainable.get(key,trainable.get(kind,False)); default=(float(np.min(a)),float(np.max(a))) if kind=='utilities' and not np.any(flag) else (0.,1.)
                lo,hi=bounds.get(key,bounds.get(kind,default)); blocks.append(ParameterBlock(key,a,lo,hi,flag))
                if kind=='beta': complete[key]=c.complete_rules
            if c.utilities is not None:
                diff=np.diff(c.utilities)
                if (diff==0).any() or not ((diff>0).all() or (diff<0).all()): raise ValueError('Utilities must be strictly ordered.')
                signs[name]=np.sign(diff)
        layout=ParameterLayout(blocks); tensors=TensorLayout(layout,cfg.device,cfg.torch_dtype)
        def params(z):
            values=tensors.unpack(z); return {name:{kind:values[name+'.'+kind] for kind in ['beta','theta','delta','utilities'] if name+'.'+kind in values} for name in models}
        # Fixed inputs are placed on the device once per fit.
        tx={k:backend.tensor(v) for k,v in inputs.items()} if isinstance(inputs,dict) else [backend.tensor(v) for v in inputs]
        def terms(z,inputs=tx,supervision=supervision):
            p=params(z); result=backend.infer(inputs,parameters=p,training_trial=True)
            if objective_fn is not None: return objective_fn(result,p).reshape(-1)
            u=p[output].get('utilities',None if models[output].config.utilities is None else backend.tensor(models[output].config.utilities))
            return components(result,supervision,u,self.objective)
        def constraints(z,kind):
            p=params(z); vals=[]
            for name,m in models.items():
                c=m.config; b=p[name]['beta']; theta=p[name]['theta']; delta=p[name]['delta']
                if kind=='eq':
                    block=next(a for a in blocks if a.name==name+'.beta'); selected=c.complete_rules & block.trainable.any(1)
                    vals.append(b[selected].sum(1)-1)
                else:
                    vals.extend([1-b[~c.complete_rules].sum(1),theta.sum().reshape(1)-1e-8,(delta.expand(c.rule_index.shape)*backend.tensor(c.participation_mask)).max(1).values-1e-8])
                    if name in signs:
                        u=p[name].get('utilities',backend.tensor(c.utilities)); vals.append(backend.tensor(signs[name])*torch.diff(u)-1e-8)
            for e in expert_constraints:
                if e['type']==kind:
                    callback=e.get('torch_fun')
                    if callback is None: raise ValueError('Torch training requires expert torch_fun callbacks; use the legacy trainer for NumPy callbacks.')
                    value=callback(z)
                    if not isinstance(value,torch.Tensor): raise ValueError('Expert torch_fun must return a tensor.')
                    vals.append(value.reshape(-1))
            return torch.cat(vals) if vals else z.new_empty(0)
        selection=None
        if validation is not None:
            if callable(validation): selection=lambda z:validation(backend,params(z))
            else:
                vx,vy=validation
                selection=lambda z:float(terms(z,vx,vy).max().detach().cpu())
        initial=backend.tensor(layout.pack())
        n=len(initial); epigraph=cfg.optimizer=='slsqp' and self.objective!='numeric' and objective_fn is None and n>0
        point=lambda z:z[:n]
        vector=torch.cat((initial,terms(initial).max().detach().reshape(1))) if epigraph else initial
        def inequalities(z):
            original=constraints(point(z),'ineq')
            return torch.cat((original,z[-1]-terms(point(z)))) if epigraph else original
        choose=None if selection is None else lambda z:selection(point(z))
        fitted=solve((lambda z:z[-1]) if epigraph else (lambda z:terms(z).max()),vector,layout.bounds()+([(-np.inf,np.inf)] if epigraph else []),equal=lambda z:constraints(point(z),'eq'),inequality=inequalities,project=lambda z:tensors.simplex(z,complete),selection=choose,config=cfg)
        selected=point(fitted.parameters)
        layout.unpack(selected.detach().cpu().numpy())
        info={'stage':1,'optimizer':cfg.optimizer,'device':cfg.device,'converged':fitted.converged,'feasible':fitted.feasible,'status':fitted.status,'selected_iteration':fitted.selected_iteration,'constraint_violation':fitted.constraint_violation,'history':fitted.history,'parameter_inventory':tensors.inventory()}
        terminal=next(h for h in reversed(fitted.history) if h['event']=='terminal')
        info.update(optimizer_converged=fitted.converged,terminal_objective=terminal['objective'],terminal_constraint_violation=terminal['constraint_violation'],selected_checkpoint_feasible=fitted.feasible)
        self.trace=info
        return TrainingResult(fitted.objective_before,float(terms(selected).max().detach().cpu()),fitted.converged and fitted.feasible,fitted.status,fitted.constraint_violation,fitted.iterations,layout.snapshot(),[info])
