import numpy as np
import pytest
from brb.rimer import RIMERTrainer,RIMERSupervision,RIMERAttribute,RIMERConfig,RIMERModel
from brb.maker import MAKERTrainer,ObservationGroup,MAKERTrainingData,paper_objectives
from examples.paper_cases.diagnosis import build_diagnosis_case

def rimer():
    return RIMERModel(RIMERConfig([RIMERAttribute('x',[0,1])],['L','H'],None,[[.5,.5],[.5,.5]],[1,1],[1],utilities=[0,1]))

def test_frozen_infeasible_is_failure_in_both_trainers():
    impossible=({'type':'ineq','fun':lambda x:np.array([-1.])},)
    m=rimer()
    r=RIMERTrainer().fit(m,[np.eye(2)],RIMERSupervision(numeric=np.array([.5,.5])),trainable={'beta':False},expert_constraints=impossible)
    assert not r.success and r.constraint_violation>=1
    model,data,_=build_diagnosis_case()
    r=MAKERTrainer().fit(model,data,bounds=data.bounds,trainable={},expert_constraints=impossible)
    assert not r.success and len(r.stage_results)==1 and r.constraint_violation>=1

def test_initially_infeasible_but_zero_error_accepts_worse_feasible():
    m=rimer()
    r=RIMERTrainer().fit(m,[np.eye(2)],RIMERSupervision(numeric=np.array([.5,.5])),
        trainable={'beta':True,'theta':False,'delta':False},
        expert_constraints=({'type':'ineq','fun':lambda x:x[0]-.8},))
    assert r.success and r.constraint_violation<1e-7
    assert m.config.rule_beliefs[0,0]>=.8-1e-7
    assert r.objective_after>r.objective_before

def test_maker_rejects_observation_and_reference_broadcasting():
    model,data,_=build_diagnosis_case()
    bad=MAKERTrainingData([ObservationGroup(('saliva',),['positive'],[[5]])],[],data.reference_parameters)
    with pytest.raises(ValueError,match='state'): paper_objectives(model,bad)
    data.reference_parameters[('saliva',)]['weights']=np.ones(1)
    with pytest.raises(ValueError,match='reference'): paper_objectives(model,data)

def test_inactive_indices_are_never_gathered():
    c=RIMERConfig([RIMERAttribute('x',[0,1]),RIMERAttribute('y',[0,1])],['L','H'],[[0,999]],[[.5,.5]],[1],[1,1],participation_mask=[[True,False]])
    r=RIMERModel(c).infer([np.array([[1,0]]),np.array([[0,1]])])
    np.testing.assert_allclose(r.beliefs,[[.5,.5]])

def test_theta_only_training_does_not_add_frozen_beta_equalities():
    m=rimer();m.config.rule_beliefs[:]=np.eye(2)
    r=RIMERTrainer().fit(m,[np.array([[.5,.5]])],RIMERSupervision(numeric=np.array([.2])),trainable={'theta':True})
    assert r.success and r.objective_after<1e-6

def test_numeric_observed_mask_excludes_finite_label():
    m=rimer()
    r=RIMERTrainer().fit(m,[np.array([[1,0],[1,0]])],
        RIMERSupervision(numeric=np.array([.1,.9]),observed_mask=np.array([True,False])),
        trainable={'beta':True})
    assert r.success and r.objective_after<1e-7
    np.testing.assert_allclose(m.infer([np.array([[1,0]])]).expected_value([0,1]),[.1],atol=1e-4)

def test_mixed_masks_are_unambiguous_and_row_mask_is_rowwise():
    from brb.rimer.objectives import objective_components
    m=rimer();result=m.infer([np.eye(2)])
    s=RIMERSupervision(numeric=np.array([.5,9.]),beliefs=np.array([[.5,.5],[.9,.1]]),
        observed_mask={'numeric':[True,False],'beliefs':[True,False]})
    np.testing.assert_allclose(objective_components(result,s,[0,1],'mixed'),[0,0,0])

def test_frozen_utilities_allow_original_measurement_scale():
    m=rimer();m.config.utilities[:]=[0,100]
    r=RIMERTrainer().fit(m,[np.eye(2)],RIMERSupervision(numeric=np.array([10,90])),trainable={'beta':True})
    assert r.success and r.objective_after<1e-6
    np.testing.assert_array_equal(m.config.utilities,[0,100])

def test_rule_limit_rejects_huge_product_before_materialisation(monkeypatch):
    import brb.rimer.config as module
    def forbidden(*args,**kwargs): raise AssertionError('Oversized Cartesian product materialised')
    monkeypatch.setattr(module,'product',forbidden)
    with pytest.raises(ValueError,match='resource'):
        RIMERConfig([RIMERAttribute(f'x{i}',[0,1]) for i in range(64)],['L','H'],None,[[.5,.5]],[1],np.ones(64))
