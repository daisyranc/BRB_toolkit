import copy
import numpy as np
import pytest
from brb.maker.torch_trainer import TorchMAKERTrainer
from brb.optimization import TrainingConfig
from examples.paper_cases.diagnosis import build_diagnosis_case

@pytest.mark.parametrize('optimizer',['slsqp','adam','adamw','sgd','lbfgs'])
def test_mask_frozen_and_traces(optimizer):
    model,data,_=build_diagnosis_case(); initial=copy.deepcopy(model.parameters)
    bounds={k:{n:(0,1) for n in p} for k,p in model.parameters.items()}
    report=TorchMAKERTrainer(config=TrainingConfig(optimizer=optimizer,maxiter=3,lr=.01)).fit(model,data,bounds=bounds,trainable={})
    assert report.success and len(report.stage_results)==2
    for k,p in model.parameters.items():
        for n,a in p.items(): np.testing.assert_array_equal(a,initial[k][n])
    assert report.stage_results[1]['priority_limit']>=report.stage_results[0]['f1']
    assert report.stage_results[0]['history'][0]['parameters']==[]

def test_torch_optimizer_runs_and_returns_feasible():
    model,data,_=build_diagnosis_case()
    for p in model.parameters.values():
        for a in p.values(): a[:]=.8
    bounds={k:{n:(.05,.99) for n in p} for k,p in model.parameters.items()}
    report=TorchMAKERTrainer(config=TrainingConfig(optimizer='adam',maxiter=10,lr=.01)).fit(model,data,bounds=bounds)
    assert report.objective_after<=report.objective_before+1e-10
    assert report.stage_results[0]['feasible']
    assert report.stage_results[0]['history'][-1]['event']=='selected'
