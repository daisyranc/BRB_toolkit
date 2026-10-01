import numpy as np
from brb.rimer import RIMERAttribute,RIMERConfig,RIMERModel,RIMERTrainer,RIMERSupervision

def small():
    c=RIMERConfig([RIMERAttribute('x',[0,1])],['L','H'],None,[[.5,.5],[.5,.5]],[1,1],[1],utilities=[0,1])
    return RIMERModel(c)

def test_numeric_training_constraints_and_freeze():
    m=small(); x=[np.array([[1,0],[0,1]])]; y=RIMERSupervision(numeric=np.array([.1,.9]))
    r=RIMERTrainer().fit(m,x,y,trainable={'beta':True,'theta':False,'delta':False})
    assert r.success and r.objective_after < 1e-7 and r.constraint_violation<1e-7
    np.testing.assert_allclose(m.config.rule_weights,[1,1])
    np.testing.assert_allclose(m.config.rule_beliefs.sum(1),1,atol=1e-7)

def test_subjective_and_mixed_supervision():
    for objective in ['belief','mixed']:
        m=small(); x=[np.array([[1,0],[0,1]])]
        y=RIMERSupervision(numeric=np.array([.1,np.nan]),beliefs=np.array([[.9,.1],[.1,.9]]))
        r=RIMERTrainer(objective=objective).fit(m,x,y,trainable={'beta':True,'theta':False,'delta':False})
        assert r.success and r.objective_after<1e-7

def test_solver_failure_reports_feasible_result():
    m=small();before=m.config.rule_beliefs.copy()
    r=RIMERTrainer(maxiter=0).fit(m,[np.eye(2)],RIMERSupervision(numeric=np.array([0,1])))
    assert not r.success
    np.testing.assert_allclose(m.config.rule_beliefs,before)

def test_hierarchy_joint_training_and_utility_order():
    from brb.rimer import RIMERNode,HierarchicalRIMER
    a=small(); b=small(); b.config.attributes[0]=RIMERAttribute('mid',['L','H'],'categorical')
    a.config.rule_beliefs[:]=[[.8,.2],[.2,.8]]
    b.config.rule_beliefs[:]=[[.9,.1],[.1,.9]]
    h=HierarchicalRIMER([RIMERNode('mid',a,['x']),RIMERNode('out',b,['mid'])],'out')
    r=RIMERTrainer().fit(h,{'x':np.eye(2)},RIMERSupervision(numeric=np.array([.05,.95])),trainable={'beta':True,'utilities':True,'theta':False,'delta':False})
    assert r.success and r.objective_after<1e-7
    assert np.all(np.diff(b.config.utilities)>0)
