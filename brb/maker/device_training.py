"""Prioritized MAKER optimization using the shared device solvers."""
import numpy as np
import torch
from ..optimization import solve
from ..optimization.constraints import TensorLayout
from ..paper_common import ParameterBlock,ParameterLayout,TrainingResult
from .torch_backend import TorchMAKERBackend

def fit_device(model,data,bounds,trainable,expert_constraints,validation,config,priority_tolerance):
    config.check_device(); blocks=[]; entries=[]
    for key,p in model.parameters.items():
        for kind,a in p.items():
            lo,hi=bounds[key][kind]; flag=True if trainable is None else trainable.get((key,kind),False)
            block=ParameterBlock('/'.join(key)+'.'+kind,a,lo,hi,flag); blocks.append(block); entries.append((key,kind,block.name))
    layout=ParameterLayout(blocks); tensors=TensorLayout(layout,config.device,config.torch_dtype)
    backend=TorchMAKERBackend(model,device=config.device,dtype=config.torch_dtype,batch_size=config.batch_size)
    def parameters(x):
        values=tensors.unpack(x); result={}
        for key,kind,name in entries: result.setdefault(key,{})[kind]=values[name]
        return result
    def terms(x): return backend.objectives(data,parameters=parameters(x))
    initial=backend.tensor(layout.pack()); before=float(terms(initial)[0].max().detach().cpu()); stages=[]
    def stage(start,index,limit=None):
        epigraph=config.optimizer=='slsqp' and len(start)>0
        vector=torch.cat((start,terms(start)[index].max().detach().reshape(1))) if epigraph else start
        box=layout.bounds()+[(0,np.inf)] if epigraph else layout.bounds()
        def point(z): return z[:-1] if epigraph else z
        def objective(z): return z[-1] if epigraph else terms(z)[index].max()
        def constraint(z,kind):
            x=point(z); a,b=terms(x); pieces=[]
            if kind=='ineq':
                if epigraph: pieces.append(z[-1]-(a,b)[index])
                if limit is not None: pieces.append(limit-a)
            for entry in expert_constraints:
                if entry['type']==kind:
                    if 'torch_fun' not in entry: raise ValueError('Device training requires expert torch_fun; use legacy trainer for NumPy callbacks.')
                    value=entry['torch_fun'](x)
                    if not isinstance(value,torch.Tensor): raise ValueError('Expert torch_fun must return a tensor.')
                    pieces.append(value.reshape(-1))
            return torch.cat(pieces) if pieces else z.new_empty(0)
        selection=None if validation is None else lambda z:validation(backend,parameters(point(z)))
        def diagnostics(z):
            a,b=terms(point(z)); return {'f1':float(a.max().detach().cpu()),'f2':float(b.max().detach().cpu()),'priority_limit':limit}
        result=solve(objective,vector,box,equal=lambda z:constraint(z,'eq'),inequality=lambda z:constraint(z,'ineq'),selection=selection,diagnostics=diagnostics,config=config)
        fitted=point(result.parameters); a,b=terms(fitted)
        info={'stage':index+1,'optimizer':config.optimizer,'device':config.device,'success':result.converged and result.feasible,'converged':result.converged,'feasible':result.feasible,'status':result.status,'iterations':result.iterations,'constraint_violation':result.constraint_violation,'priority_limit':limit,'f1':float(a.max().detach().cpu()),'f2':float(b.max().detach().cpu()),'history':result.history,'selected_iteration':result.selected_iteration,'parameter_inventory':tensors.inventory()}
        terminal=next(h for h in reversed(result.history) if h['event']=='terminal')
        info.update(optimizer_converged=result.converged,terminal_objective=terminal['objective'],terminal_constraint_violation=terminal['constraint_violation'],terminal_f1=terminal.get('f1'),terminal_f2=terminal.get('f2'),selected_checkpoint_feasible=result.feasible)
        return fitted,info
    first,info=stage(initial,0); stages.append(info); chosen=first; success=info['success']; status=info['status']
    if success:
        second,info2=stage(first,1,info['f1']+priority_tolerance); stages.append(info2); success=info2['success']; status=info2['status']
        if success: chosen=second
        else: status='Second stage failed; restored first-stage feasible parameters: '+status
    layout.unpack(chosen.detach().cpu().numpy()); after=float(terms(chosen)[0].max().detach().cpu())
    residual=stages[0]['constraint_violation'] if len(stages)==1 or not success else stages[-1]['constraint_violation']
    return TrainingResult(before,after,success,status,residual,sum(s['iterations'] for s in stages),layout.snapshot(),stages)
