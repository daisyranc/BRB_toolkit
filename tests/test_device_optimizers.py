import numpy as np
import pytest
import torch
from brb.optimization import TrainingConfig, solve

@pytest.mark.parametrize('optimizer',['slsqp','adam','adamw','sgd','lbfgs'])
def test_constrained_quadratic(optimizer):
    config=TrainingConfig(optimizer=optimizer,maxiter=120,lr=.1,constraint_options={'outer_steps':6},optimizer_options={'weight_decay':0.} if optimizer=='adamw' else {})
    x=torch.tensor([.2,.8],dtype=torch.float64)
    result=solve(lambda z:((z-torch.tensor([.7,.3],dtype=z.dtype,device=z.device))**2).sum(),x,[(0.,1.),(0.,1.)],equal=lambda z:z.sum().reshape(1)-1,project=lambda z:z/z.sum(),config=config)
    assert result.feasible
    assert result.objective_after < result.objective_before*.8
    assert len(result.history)>1
    assert all(np.isfinite(h['objective']) for h in result.history)

def test_invalid_and_zero_budget():
    with pytest.raises(ValueError): TrainingConfig(optimizer='invalid')
    result=solve(lambda x:x.square().sum(),torch.tensor([.5],dtype=torch.float64),[(0,1)],config=TrainingConfig(maxiter=0))
    assert not result.converged and result.feasible and result.iterations==0

def test_active_inequality_stationarity():
    result=solve(lambda x:(x[0]-1).square(),torch.tensor([.5],dtype=torch.float64),[(0,1)],inequality=lambda x:.5-x,config=TrainingConfig(optimizer='lbfgs',maxiter=120,lr=1.))
    assert result.feasible and result.converged
    assert abs(float(result.parameters[0])-.5)<1e-7

def test_lbfgs_line_search_keeps_objective_in_box():
    def objective(x):
        if ((x<0)|(x>1)).any():raise ValueError('Objective outside its domain')
        return (x[0]-1).square()
    result=solve(objective,torch.tensor([.1],dtype=torch.float64),[(0,1)],config=TrainingConfig(optimizer='lbfgs',maxiter=10,lr=10.))
    assert result.feasible and result.objective_after<result.objective_before
