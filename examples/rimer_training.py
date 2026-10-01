"""R07 graphite training; run from toolkit root as python -m examples.rimer_training."""
import numpy as np
from brb.rimer import RIMERTrainer,RIMERSupervision
from examples.paper_cases.graphite import load_graphite_case,build_graphite_model,graphite_inputs

def run(maxiter=200):
    d=load_graphite_case();data=np.asarray(d['training'])
    m=build_graphite_model(training=True)
    x=graphite_inputs(data[:,:4])
    before=m.infer(x).expected_value([1,.5,0])
    trainer=RIMERTrainer(maxiter=maxiter,ftol=1e-9)
    fit=trainer.fit(m,x,RIMERSupervision(numeric=data[:,4]),trainable={'beta':True,'theta':True,'delta':True,'utilities':True})
    after=m.infer(x).expected_value(m.models['X9'].config.utilities)
    return m,fit,before,after
if __name__=='__main__':
    import json
    from pathlib import Path
    m,r,before,after=run()
    report={'training':r.to_dict(),'observed':np.asarray(load_graphite_case()['training'])[:,4].tolist(),
            'initial':before.tolist(),'trained':after.tolist()}
    destination=Path('outputs/rimer_run.json');destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print({k:getattr(r,k) for k in ['objective_before','objective_after','success','status','iterations','constraint_violation']})
    print('Observed / initial / trained:',np.column_stack([np.asarray(load_graphite_case()['training'])[:,4],before,after]))
