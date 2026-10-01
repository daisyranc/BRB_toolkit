import numpy as np
from scipy.optimize import minimize
from ..paper_common import ParameterBlock,ParameterLayout,TrainingResult
from .objectives import paper_objectives

class MAKERTrainer:
    """Local two-stage pre-emptive epigraph minimax, M25 Eqs 97–100."""
    def __init__(self,maxiter=1000,ftol=1e-10,priority_tolerance=1e-5):
        if maxiter<0 or ftol<=0 or priority_tolerance<0: raise ValueError('Invalid optimiser settings.')
        self.maxiter=maxiter;self.ftol=ftol;self.priority_tolerance=priority_tolerance

    def fit(self,model,data,*,bounds,trainable=None,expert_constraints=()):
        blocks=[]
        for key,p in model.parameters.items():
            for kind,a in p.items():
                lo,hi=bounds[key][kind]
                flag=True if trainable is None else trainable.get((key,kind),False)
                blocks.append(ParameterBlock('/'.join(key)+'.'+kind,a,lo,hi,flag))
        layout=ParameterLayout(blocks);initial=layout.pack();n=len(initial)
        before=float(paper_objectives(model,data)[0].max());stages=[];iterations=0
        def terms(x):
            layout.unpack(x[:n]);return paper_objectives(model,data)
        def extras(x):
            residual=0.
            for c in expert_constraints:
                a=np.asarray(c['fun'](x[:n]),float)
                residual=max(residual,float(np.max(np.abs(a) if c['type']=='eq' else -a,initial=0)))
            return residual
        def stage(start,index,primary_limit=None):
            v=terms(start);z0=np.r_[start,float(v[index].max())]
            def inequality(z):
                a,b=terms(z);selected=(a,b)[index]
                result=z[-1]-selected
                return result if primary_limit is None else np.r_[result,primary_limit-a]
            def violation(z):
                v=max(float(np.max(-inequality(z),initial=0)),extras(z),0.)
                for value,(lo,hi) in zip(z[:n],layout.bounds()): v=max(v,lo-value,value-hi)
                return v
            best=z0.copy() if violation(z0)<1e-8 else None
            score=z0[-1] if best is not None else np.inf
            def track(z):
                nonlocal best,score
                v=terms(z)[index].max()
                if violation(z)<1e-8 and (best is None or v<score): best=z.copy();score=float(v)
            constraints=[{'type':'ineq','fun':inequality}]
            constraints.extend({'type':c['type'],'fun':lambda z,c=c:c['fun'](z[:n])} for c in expert_constraints)
            if n and self.maxiter:
                sol=minimize(lambda z:z[-1],z0,method='SLSQP',bounds=layout.bounds()+[(0,None)],constraints=constraints,
                             callback=track,options={'maxiter':self.maxiter,'ftol':self.ftol})
                track(sol.x);success=bool(sol.success and violation(sol.x)<1e-7)
                status=str(sol.message);nit=int(sol.nit)
            else: success=not n and violation(z0)<1e-8;status='No trainable parameters' if not n else 'Iteration limit is zero';nit=0
            if best is None:
                best=z0.copy();success=False;status+='; no feasible candidate found'
            success=bool(success and violation(best)<1e-7)
            a,b=terms(best)
            info={'stage':index+1,'success':success,'status':status,'f1':float(a.max()),'f2':float(b.max()),
                  'iterations':nit,'constraint_violation':violation(best),'priority_limit':primary_limit}
            return best[:n].copy(),info
        first,info=stage(initial,0);stages.append(info);iterations+=info['iterations']
        chosen=first;success=info['success'];status=info['status']
        if success:
            second,info2=stage(first,1,info['f1']+self.priority_tolerance)
            stages.append(info2);iterations+=info2['iterations'];success=info2['success'];status=info2['status']
            if success: chosen=second
            else: status='Second stage failed; restored first-stage feasible parameters: '+status
        layout.unpack(chosen)
        after=float(paper_objectives(model,data)[0].max())
        residual=extras(chosen)
        if len(stages)==2: residual=max(residual,after-stages[0]['f1']-self.priority_tolerance)
        return TrainingResult(before,after,success,status,max(residual,0),iterations,layout.snapshot(),stages)
