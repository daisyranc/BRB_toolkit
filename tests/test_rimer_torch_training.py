import numpy as np
import pytest
import torch
from brb.rimer import RIMERAttribute,RIMERConfig,RIMERModel,RIMERSupervision
from brb.rimer.torch_trainer import TorchRIMERTrainer
from brb.optimization import TrainingConfig

def test_utility_training_does_not_mutate_shared_initialization():
    initial = np.array([0., 1.])
    def build():
        return RIMERModel(RIMERConfig([RIMERAttribute('x', [0., 1.])],
            ['a', 'b'], None, [[1., 0.], [0., 1.]], [1., 1.], [1.],
            utilities=initial))
    first, peer = build(), build()
    report = TorchRIMERTrainer(config=TrainingConfig(maxiter=20)).fit(
        first, [np.eye(2)], RIMERSupervision(numeric=[.2, .8]),
        trainable={'utilities': True}, bounds={'utilities': (0., 1.)})
    assert report.objective_after < report.objective_before
    np.testing.assert_array_equal(initial, [0., 1.])
    np.testing.assert_array_equal(peer.config.utilities, [0., 1.])
    np.testing.assert_array_equal(build().config.utilities, [0., 1.])

@pytest.mark.parametrize('optimizer',['slsqp','adam','adamw','sgd','lbfgs'])
def test_training_preserves_partial_freeze(optimizer):
    m=RIMERModel(RIMERConfig([RIMERAttribute('x',[0.,1.])],['a','b'],None,[[.6,.4],[.4,.6]],[1.,1.],[1.],utilities=[0.,1.]))
    x=[np.eye(2)]; y=RIMERSupervision(numeric=[.1,.9])
    report=TorchRIMERTrainer(config=TrainingConfig(optimizer=optimizer,maxiter=80,lr=.05)).fit(m,x,y,trainable={'beta':[[True,True],[True,True]],'theta':False,'delta':False})
    assert report.constraint_violation<1e-7
    assert report.objective_after<report.objective_before
    np.testing.assert_array_equal(m.config.rule_weights,[1.,1.])
    np.testing.assert_allclose(m.config.rule_beliefs.sum(1),1.,atol=1e-7)
    assert report.stage_results[0]['history'][-1]['event']=='selected'

def test_partly_frozen_row_and_zero_iterations():
    m=RIMERModel(RIMERConfig([RIMERAttribute('x',[0.,1.])],['a','b','c'],None,[[.2,.3,.5],[.4,.4,.2]],[1.,1.],[1.],utilities=[0.,.5,1.]))
    report=TorchRIMERTrainer(config=TrainingConfig(optimizer='adam',maxiter=10,lr=.01)).fit(m,[np.eye(2)],RIMERSupervision(numeric=[.3,.7]),trainable={'beta':[[False,True,True],[False,True,True]]})
    np.testing.assert_array_equal(m.config.rule_beliefs[:,0],[.2,.4])
    assert report.constraint_violation<1e-7
    report=TorchRIMERTrainer(config=TrainingConfig(maxiter=0)).fit(m,[np.eye(2)],RIMERSupervision(numeric=[.3,.7]))
    assert not report.success

def test_belief_minimax_uses_epigraph():
    m=RIMERModel(RIMERConfig([RIMERAttribute('x',[0.,1.])],['a','b'],None,[[.6,.4],[.4,.6]],[1.,1.],[1.]))
    report=TorchRIMERTrainer(objective='belief',config=TrainingConfig(maxiter=20)).fit(m,[np.eye(2)],RIMERSupervision(beliefs=[[.9,.1],[.1,.9]]),trainable={'beta':True})
    assert len(report.stage_results[0]['history'][0]['parameters'])==5
    assert report.objective_after<report.objective_before and report.success

def test_negative_normalized_objective_is_allowed():
    m=RIMERModel(RIMERConfig([RIMERAttribute('x',[0.,1.])],['a','b'],None,[[.6,.4],[.4,.6]],[1.,1.],[1.]))
    report=TorchRIMERTrainer(objective='belief',config=TrainingConfig(maxiter=20)).fit(m,[np.eye(2)],RIMERSupervision(beliefs=[[.9,.1],[.1,.9]],ideal=.2,nadir=.5),trainable={'beta':True})
    assert report.objective_before<0 and report.objective_after<report.objective_before

def test_unobserved_missing_numeric_row_needs_no_decision():
    m=RIMERModel(RIMERConfig([RIMERAttribute('x',[0.,1.])],['a','b'],None,[[.6,.4],[.4,.6]],[1.,1.],[1.],utilities=[0.,1.]))
    result=TorchRIMERTrainer(config=TrainingConfig(maxiter=0)).fit(m,[np.array([[1.,0.],[0.,0.]])],RIMERSupervision(numeric=[.4,np.nan]))
    assert result.objective_before==0

@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_cuda_training_matches_cpu():
    fitted=[]
    for device in ['cpu','cuda']:
        m=RIMERModel(RIMERConfig([RIMERAttribute('x',[0.,1.])],['a','b'],None,[[.6,.4],[.4,.6]],[1.,1.],[1.],utilities=[0.,1.]))
        report=TorchRIMERTrainer(config=TrainingConfig(optimizer='adam',device=device,maxiter=10,lr=.02)).fit(m,[np.eye(2)],RIMERSupervision(numeric=[.1,.9]),trainable={'beta':True})
        assert report.constraint_violation<1e-7
        fitted.append(m.config.rule_beliefs.copy())
    np.testing.assert_allclose(fitted[0],fitted[1],atol=1e-7)
