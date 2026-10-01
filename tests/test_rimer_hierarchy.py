import numpy as np
import pytest
from brb.rimer import RIMERAttribute,RIMERConfig,RIMERModel,RIMERNode,HierarchicalRIMER

def identity(name):
    return RIMERModel(RIMERConfig([RIMERAttribute(name,['H','L'],'categorical')],['H','L'],[[0],[1]],[[1,0],[0,1]],[1,1],[1]))

def test_unknown_propagates():
    h = HierarchicalRIMER([RIMERNode('mid',identity('x'),['x']),RIMERNode('out',identity('mid'),['mid'])], 'out')
    r = h.infer({'x':np.array([[.5,0]])})
    np.testing.assert_allclose(r.ignorance, [.5])
    assert 'mid' in r.trace

def test_cycle_and_semantics():
    with pytest.raises(ValueError):
        HierarchicalRIMER([RIMERNode('a',identity('b'),['b']),RIMERNode('b',identity('a'),['a'])], 'b')
    a = identity('x'); b=identity('mid'); b.config.attributes[0].references=['L','H']
    with pytest.raises(ValueError): HierarchicalRIMER([RIMERNode('mid',a,['x']),RIMERNode('out',b,['mid'])], 'out')
