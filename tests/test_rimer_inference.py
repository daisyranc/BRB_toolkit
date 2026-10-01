import numpy as np
import pytest
from brb.rimer import RIMERAttribute, RIMERConfig, RIMERModel
from brb.rimer.inference import er_aggregate

def model(connective='and'):
    a = [RIMERAttribute('a',[0,1]), RIMERAttribute('b',[0,1])]
    c = RIMERConfig(a, ['L','H'], [[0,0],[1,1]], [[1,0],[0,1]], [1,1], [1,1], connective=connective, utilities=[0,1])
    return RIMERModel(c)

def test_single_incomplete_rule():
    b,u = er_aggregate([[1]], [[[.2,.3]]])
    np.testing.assert_allclose(b, [[.2,.3]])
    np.testing.assert_allclose(u, [.5])

def test_complete_er_hand_calculation():
    b,u = er_aggregate([[.5,.5]], [[[1,0],[0,1]]])
    np.testing.assert_allclose(b, [[.5,.5]])
    np.testing.assert_allclose(u, [0])

def test_zero_activation_unknown():
    r = model().infer([np.zeros((1,2)),np.zeros((1,2))])
    np.testing.assert_allclose(r.ignorance, [1])
    assert r.status[0] == 'no_activation'

def test_numeric_and_or_and_masks():
    m = model()
    r = m.predict([[.25,.75]])
    np.testing.assert_allclose(r.activation, [[.5,.5]])
    np.testing.assert_allclose(r.beliefs.sum(1)+r.ignorance, 1)
    m.config.attribute_weights[:] = [.5,1]
    r = m.infer([np.array([[.2,.8]]),np.array([[.6,.4]])])
    expected = np.array([.2**.5*.6,.8**.5*.4]); expected /= expected.sum()
    np.testing.assert_allclose(r.activation[0], expected)
    m = model('or')
    np.testing.assert_allclose(m.predict([[.25,.75]]).activation, [[.5,.5]])

def test_unknown_and_validation():
    r = model().predict([[float('nan'),.5]])
    assert r.ignorance[0] == 1
    with pytest.raises(ValueError): model().infer([np.array([[.6,.6]]), np.array([[.5,.5]])])
    with pytest.raises(ValueError):
        RIMERModel(RIMERConfig([RIMERAttribute('a',[0,1])],['L','H'],[[0]],[[1,0]],[1],[0]))
