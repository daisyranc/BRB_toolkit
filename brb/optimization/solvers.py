"""Box-projected augmented Lagrangian and exact-Jacobian SLSQP."""
from dataclasses import dataclass
import time
import numpy as np
import torch
from scipy.optimize import minimize
from .config import TrainingConfig

@dataclass
class SolverResult:
    parameters: torch.Tensor
    objective_before: float
    objective_after: float
    feasible: bool
    converged: bool
    status: str
    iterations: int
    constraint_violation: float
    history: list
    selected_iteration: int

def solve(objective, initial, bounds, *, equal=None, inequality=None, project=None, selection=None, diagnostics=None, config=None):
    """Inequality convention: g(x)>=0. Selection sees feasible candidates only."""
    cfg=config or TrainingConfig(); cfg.check_device()
    x=initial.detach().to(device=cfg.device,dtype=cfg.torch_dtype).clone()
    if x.ndim!=1 or len(bounds)!=len(x): raise ValueError('Parameter/bound dimensions differ.')
    limits=torch.as_tensor(bounds,device=x.device,dtype=x.dtype).reshape(-1,2)
    lower,upper=limits[:,0],limits[:,1]
    if not torch.isfinite(x).all() or torch.isnan(limits).any() or (lower>upper).any() or ((x<lower)|(x>upper)).any(): raise ValueError('Invalid initial parameters/bounds.')
    empty=x.new_empty(0)
    timing={'objective_forward_seconds':0.,'derivative_seconds':0.,'validation_seconds':0.,'optimizer_step_seconds':0.}
    def clock():
        if cfg.profile_compute and x.device.type=='cuda': torch.cuda.synchronize(x.device)
        return time.perf_counter()
    def residuals(z):
        eq=empty if equal is None else equal(z).reshape(-1)
        iq=empty if inequality is None else inequality(z).reshape(-1)
        if not torch.isfinite(eq).all() or not torch.isfinite(iq).all(): raise ValueError('Nonfinite constraint.')
        return eq,iq
    def violation(z):
        eq,iq=residuals(z)
        pieces=[eq.abs(),(-iq).clamp_min(0),(lower-z).clamp_min(0),(z-upper).clamp_min(0)]
        values=torch.cat(pieces)
        return float(values.max().detach().cpu()) if len(values) else 0.
    def scalar(z):
        started=clock()
        v=objective(z)
        timing['objective_forward_seconds']+=clock()-started
        if v.numel()!=1 or not torch.isfinite(v).all(): raise ValueError('Objective must be a finite scalar tensor.')
        return v.reshape(())
    def projection(z):
        z=z.clamp(lower,upper)
        return z if project is None else project(z).clamp(lower,upper)
    history=[]; start=time.perf_counter(); best=None; bestscore=np.inf; selected_iteration=0
    def record(z,iteration,event,gradient_norm=None,phase=0,projected_gradient_norm=None):
        nonlocal best,bestscore,selected_iteration
        value=float(scalar(z).detach().cpu()); residual=violation(z)
        started=clock()
        selected=selection(z) if selection is not None and residual<=cfg.feasibility_tol else value
        timing['validation_seconds']+=clock()-started
        metrics=selected if isinstance(selected,dict) else {}
        score=float(metrics['score'] if metrics else selected)
        if not np.isfinite(score): raise ValueError('Nonfinite validation score.')
        if residual<=cfg.feasibility_tol and (best is None or score<bestscore): best=z.detach().clone(); bestscore=score; selected_iteration=iteration
        if iteration%cfg.record_every==0 or event!='iteration':
            if cfg.optimizer=='slsqp' and event=='iteration':
                probe=z.detach().clone().requires_grad_(True); loss=scalar(probe); started=clock(); grad=torch.autograd.grad(loss,probe)[0];timing['derivative_seconds']+=clock()-started
                gradient_norm=float(grad.norm().detach().cpu())
            row={'iteration':iteration,'event':event,'phase':phase,'objective':value,'constraint_violation':residual,'selection_score':score if residual<=cfg.feasibility_tol else None,'gradient_norm':gradient_norm,'lr':cfg.lr if cfg.optimizer!='slsqp' else None,'elapsed_seconds':time.perf_counter()-start}
            row['projected_gradient_norm']=projected_gradient_norm
            row.update({'validation_'+k:float(v) for k,v in metrics.items() if k!='score'})
            if diagnostics is not None: row.update(diagnostics(z))
            row.update(timing)
            if iteration%cfg.checkpoint_every==0 or event!='iteration': row['parameters']=z.detach().cpu().tolist()
            history.append(row)
    before=float(scalar(x).detach().cpu()); record(x,0,'initial')
    converged=False; nit=0; status='Iteration limit reached'
    if not len(x) or cfg.maxiter==0:
        converged=not len(x) and violation(x)<=cfg.feasibility_tol
        status='No trainable parameters' if not len(x) else 'Iteration limit is zero'
    elif cfg.optimizer=='slsqp':
        if set(cfg.optimizer_options)-{'disp'}: raise ValueError('SLSQP only accepts disp in optimizer_options.')
        def val(z): return scalar(torch.as_tensor(z,device=x.device,dtype=x.dtype))
        def fun(z): return float(val(z).detach().cpu())
        def jac(z):
            v=torch.tensor(z,device=x.device,dtype=x.dtype,requires_grad=True); loss=scalar(v)
            started=clock(); grad=torch.autograd.grad(loss,v)[0];timing['derivative_seconds']+=clock()-started
            if not torch.isfinite(grad).all(): raise ValueError('Nonfinite objective gradient.')
            return grad.detach().cpu().numpy()
        constraints=[]
        for kind,callback in [('eq',equal),('ineq',inequality)]:
            if callback is not None and callback(x).numel():
                def f(z,callback=callback): return callback(torch.as_tensor(z,device=x.device,dtype=x.dtype)).detach().cpu().numpy().reshape(-1)
                def j(z,callback=callback):
                    v=torch.tensor(z,device=x.device,dtype=x.dtype,requires_grad=True)
                    started=clock(); grad=torch.autograd.functional.jacobian(lambda a:callback(a).reshape(-1),v,vectorize=True);timing['derivative_seconds']+=clock()-started
                    if not torch.isfinite(grad).all(): raise ValueError('Nonfinite constraint Jacobian.')
                    return grad.detach().cpu().numpy()
                constraints.append({'type':kind,'fun':f,'jac':j})
        def track(z):
            nonlocal nit
            nit+=1; record(torch.as_tensor(z,device=x.device,dtype=x.dtype),nit,'iteration')
        started=clock();sol=minimize(fun,x.cpu().numpy(),jac=jac,method='SLSQP',bounds=bounds,constraints=constraints,callback=track,options={'maxiter':cfg.maxiter,'ftol':cfg.ftol,**cfg.optimizer_options});timing['optimizer_step_seconds']+=clock()-started
        x=torch.as_tensor(sol.x,device=x.device,dtype=x.dtype); nit=int(sol.nit)
        converged=bool(sol.success and violation(x)<=cfg.feasibility_tol); status=str(sol.message)
    else:
        parameter=torch.nn.Parameter(x)
        classes={'adam':torch.optim.Adam,'adamw':torch.optim.AdamW,'sgd':torch.optim.SGD,'lbfgs':torch.optim.LBFGS}
        options=dict(cfg.optimizer_options)
        if cfg.optimizer=='adamw': options.setdefault('weight_decay',0.)
        if cfg.optimizer=='lbfgs':
            if set(options)&{'max_iter','max_eval'}: raise ValueError('L-BFGS max_iter/max_eval are controlled by the training budget.')
            options.update(max_iter=1,max_eval=10); options.setdefault('line_search_fn','strong_wolfe')
        optimizer=classes[cfg.optimizer]([parameter],lr=cfg.lr,**options)
        e,g=residuals(parameter); le=torch.zeros_like(e); li=torch.zeros_like(g)
        rho=float(cfg.constraint_options.get('penalty',10.)); outer=int(cfg.constraint_options.get('outer_steps',5)); interval=max(1,(cfg.maxiter+outer-1)//outer)
        for i in range(1,cfg.maxiter+1):
            def closure():
                optimizer.zero_grad(); trial=parameter.clamp(lower,upper); e,g=residuals(trial)
                penalty=(le*e).sum()+rho/2*e.square().sum()
                # Powell-Hestenes-Rockafellar inequality term, with g>=0.
                penalty=penalty+(((li-rho*g).clamp_min(0).square()-li.square())/(2*rho)).sum()
                box_penalty=rho/2*((lower-parameter).clamp_min(0).square()+(parameter-upper).clamp_min(0).square()).sum()
                loss=scalar(trial)+penalty+box_penalty;started=clock();loss.backward();timing['derivative_seconds']+=clock()-started
                if parameter.grad is None or not torch.isfinite(parameter.grad).all(): raise ValueError('Nonfinite gradient.')
                return loss
            started=clock()
            if cfg.optimizer=='lbfgs': optimizer.step(closure)
            else: closure(); optimizer.step()
            timing['optimizer_step_seconds']+=clock()-started
            with torch.no_grad(): parameter.copy_(projection(parameter))
            nit=i
            probe=parameter.detach().clone().requires_grad_(True); loss=scalar(probe);e,g=residuals(probe)
            gradient=torch.autograd.grad(loss,probe,retain_graph=True)[0]
            lagrangian=loss+(le*e).sum()-(li*g).sum()
            lag_gradient=torch.autograd.grad(lagrangian,probe)[0]
            with torch.no_grad():
                stationarity=float((probe-projection(probe-lag_gradient)).norm().cpu())
                complementarity=float((li*g).abs().max().cpu()) if len(g) else 0.
            record(parameter,i,'iteration',float(gradient.norm().detach().cpu()),(i-1)//interval,stationarity)
            if violation(parameter)<=cfg.feasibility_tol and stationarity<=np.sqrt(cfg.ftol) and complementarity<=np.sqrt(cfg.ftol): converged=True; status='Lagrangian stationarity, complementarity and feasibility tolerances satisfied'; break
            if i%interval==0:
                e,g=residuals(parameter); le=(le+rho*e).detach(); li=(li-rho*g).clamp_min(0).detach(); rho*=float(cfg.constraint_options.get('penalty_growth',10.))
                if cfg.optimizer=='lbfgs': optimizer.state.clear()
        x=parameter.detach()
    record(x,nit,'terminal')
    chosen=best if best is not None else initial.to(x).detach().clone()
    residual=violation(chosen); feasible=residual<=cfg.feasibility_tol
    if best is None: status+='; no feasible candidate found'
    record(chosen,selected_iteration,'selected')
    return SolverResult(chosen,before,float(scalar(chosen).detach().cpu()),feasible,bool(converged and feasible),status,nit,residual,history,selected_iteration)
