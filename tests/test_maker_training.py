import numpy as np
from brb.maker import MAKERTrainer,paper_objectives
from examples.paper_cases.diagnosis import build_diagnosis_case

def test_objective_frequency_and_reference_separation():
    model,data,case=build_diagnosis_case()
    a,b=paper_objectives(model,data)
    pred=model.infer({'saliva':'positive'}).probabilities
    independent=180/6300*.5*np.sum((np.eye(3)[0]-pred)**2)
    np.testing.assert_allclose(a[0],independent)
    np.testing.assert_allclose(b,0)
    model.parameters[('saliva',)]['weights'][0]-=.1
    _,b=paper_objectives(model,data)
    np.testing.assert_allclose(b[0],180/6300*.01)
    assert data.reference_parameters[('saliva',)]['weights'][0]==.9

def test_two_stage_minimax_and_bounds():
    model,data,case=build_diagnosis_case()
    result=MAKERTrainer(maxiter=300).fit(model,data,bounds=data.bounds)
    assert result.success,result.status
    assert len(result.stage_results)==2
    assert result.objective_after<result.objective_before
    assert result.constraint_violation<1e-7
    assert result.stage_results[1]['f1']<=result.stage_results[0]['f1']+1e-5+1e-8
    assert sum(p['weights'].size+p['reliabilities'].size for p in model.parameters.values())==21
    for key,p in model.parameters.items():
        for kind,a in p.items():
            lo,hi=data.bounds[key][kind]
            assert np.all((a>=lo-1e-8)&(a<=hi+1e-8))

def test_failed_first_stage_does_not_run_second():
    model,data,_=build_diagnosis_case()
    initial={k:{n:a.copy() for n,a in p.items()} for k,p in model.parameters.items()}
    result=MAKERTrainer(maxiter=0).fit(model,data,bounds=data.bounds)
    assert not result.success and len(result.stage_results)==1
    for key,p in model.parameters.items():
        for kind,a in p.items(): np.testing.assert_allclose(a,initial[key][kind])

def test_failed_second_stage_restores_first_feasible(monkeypatch):
    import brb.maker.trainer as module
    real=module.minimize;calls=[]
    def fail_second(*args,**kwargs):
        if not calls:
            sol=real(*args,**kwargs);calls.append(sol.x.copy());return sol
        from types import SimpleNamespace
        return SimpleNamespace(x=np.asarray(args[1]),success=False,message='Injected second-stage failure',nit=0)
    monkeypatch.setattr(module,'minimize',fail_second)
    model,data,_=build_diagnosis_case()
    result=MAKERTrainer(maxiter=300).fit(model,data,bounds=data.bounds)
    assert not result.success and len(result.stage_results)==2
    assert 'restored first-stage' in result.status
    assert result.constraint_violation<1e-7
    np.testing.assert_allclose(result.objective_after,result.stage_results[0]['f1'],atol=1e-10)
