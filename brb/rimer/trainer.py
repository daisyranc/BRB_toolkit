import numpy as np
from scipy.optimize import minimize
from ..paper_common import ParameterBlock,ParameterLayout,TrainingResult
from .objectives import objective_components

class RIMERTrainer:
    """R07 constrained local training with epigraph minimax objectives."""
    def __init__(self,objective='numeric',maxiter=500,ftol=1e-9):
        if objective not in ('numeric','belief','mixed'): raise ValueError('Unknown training objective.')
        self.objective=objective; self.maxiter=maxiter; self.ftol=ftol

    def fit(self,model,inputs,supervision,*,trainable=None,bounds=None,expert_constraints=(),validation=None):
        models=model.models if hasattr(model,'models') else {'model':model}
        trainable={'beta':True,'theta':True,'delta':True} if trainable is None else trainable
        bounds=bounds or {}
        blocks=[]; entries=[]
        for node,m in models.items():
            c=m.config
            for kind,a in [('beta',c.rule_beliefs),('theta',c.rule_weights),('delta',c.attribute_weights)]:
                name=node+'.'+kind
                enabled=trainable.get(name,trainable.get(kind,False))
                lo,hi=bounds.get(name,bounds.get(kind,(0,1)))
                block=ParameterBlock(name,a,lo,hi,enabled)
                blocks.append(block);entries.append((node,kind,block))
            if node==getattr(model,'output','model') and c.utilities is not None:
                name=node+'.utilities'
                enabled=trainable.get(name,trainable.get('utilities',False))
                default=(0,1) if np.any(enabled) else (c.utilities.min(),c.utilities.max())
                lo,hi=bounds.get(name,bounds.get('utilities',default))
                blocks.append(ParameterBlock(name,c.utilities,lo,hi,enabled))
        layout=ParameterLayout(blocks)
        x0=layout.pack(); n=len(x0); minimax=self.objective!='numeric'
        signs={}
        for name,m in models.items():
            if m.config.utilities is not None:
                diff=np.diff(m.config.utilities)
                if np.any(diff==0) or not (np.all(diff>0) or np.all(diff<0)):
                    raise ValueError('Utilities must be strictly ordered for training.')
                signs[name]=np.sign(diff)
        def predict():
            return model._infer(inputs,training_trial=True)
        final=models[getattr(model,'output','model')]
        def components(z):
            layout.unpack(z[:n])
            return objective_components(predict(),supervision,final.config.utilities,self.objective)
        initial=components(x0)
        before=float(initial.max())
        def equal(z):
            layout.unpack(z[:n])
            return np.concatenate([models[node].config.rule_beliefs[models[node].config.complete_rules & block.trainable.any(1)].sum(1)-1
                                   for node,kind,block in entries if kind=='beta'])
        def inequalities(z):
            layout.unpack(z[:n]); vals=[]
            for name,m in models.items():
                c=m.config
                vals.extend(1-c.rule_beliefs[~c.complete_rules].sum(1))
                vals.append(c.rule_weights.sum()-1e-8)
                vals.extend((np.broadcast_to(c.attribute_weights,c.rule_index.shape)*c.participation_mask).max(1)-1e-8)
                if name in signs: vals.extend(signs[name]*np.diff(c.utilities)-1e-8)
            if minimax: vals.extend(z[-1]-components(z))
            return np.asarray(vals)
        def violation(z):
            layout.unpack(z[:n])
            eq=np.concatenate([m.config.rule_beliefs[m.config.complete_rules].sum(1)-1 for m in models.values()])
            iq=inequalities(z)
            v=max(float(np.max(np.abs(eq),initial=0)),float(np.max(-iq,initial=0)),0)
            for constraint in expert_constraints:
                value=np.asarray(constraint['fun'](z[:n]),float)
                v=max(v,float(np.max(np.abs(value) if constraint['type']=='eq' else -value,initial=0)))
            for value,(lo,hi) in zip(z[:n],layout.bounds()): v=max(v,lo-value,value-hi)
            return v
        z0=np.r_[x0,before] if minimax else x0.copy()
        best=z0.copy() if violation(z0)<1e-7 else None
        best_score=before if best is not None else np.inf
        def track(z):
            nonlocal best,best_score
            score=float(components(z).max())
            if violation(z)<1e-7 and (best is None or score<best_score):
                best=z.copy();best_score=score
        constraints=[{'type':'ineq','fun':inequalities}]
        if len(equal(z0)): constraints.append({'type':'eq','fun':equal})
        for e in expert_constraints:
            constraints.append({'type':e['type'],'fun':lambda z,e=e:e['fun'](z[:n])})
        if n and self.maxiter>0:
            objective=(lambda z:z[-1]) if minimax else (lambda z:float(components(z)[0]))
            sol=minimize(objective,z0,method='SLSQP',bounds=layout.bounds()+([(None,None)] if minimax else []),
                         constraints=constraints,callback=track,options={'maxiter':self.maxiter,'ftol':self.ftol})
            track(sol.x)
            success=bool(sol.success and violation(sol.x)<1e-7)
            message=str(sol.message);iterations=int(sol.nit)
        else:
            success=not n and violation(z0)<1e-7;message='No trainable parameters' if not n else 'Iteration limit is zero';iterations=0
        if best is None:
            best=z0.copy();success=False;message+='; no feasible candidate found'
        success=bool(success and violation(best)<1e-7)
        layout.unpack(best[:n])
        after=float(components(best).max())
        residual=violation(best)
        layout.unpack(best[:n])
        result=TrainingResult(before,after,success,message,residual,iterations,layout.snapshot())
        if validation is not None:
            vx,vs=validation
            vr=model.infer(vx)
            result.stage_results.append({'validation_objective':float(objective_components(vr,vs,final.config.utilities,self.objective).max())})
        return result
