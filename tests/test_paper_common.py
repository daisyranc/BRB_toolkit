import numpy as np
import pytest
from brb.paper_common import validate_beliefs, ParameterBlock, ParameterLayout, TrainingResult

def test_belief_validation():
    np.testing.assert_array_equal(validate_beliefs([[.2,.3]]), [[.2,.3]])
    for b in [[[-.1,.2]], [[np.inf,0]], [[.6,.5]]]:
        with pytest.raises(ValueError): validate_beliefs(b)
    with pytest.raises(ValueError): validate_beliefs([[.2,.3]], complete=True)

def test_frozen_layout():
    b = ParameterBlock('beta', np.array([.2,.3]), 0, 1, np.array([True,False]))
    layout = ParameterLayout([b])
    assert len(layout.pack()) == 1
    layout.unpack([.8])
    np.testing.assert_allclose(b.values, [.8,.3])

def test_result_roundtrip():
    r = TrainingResult(.5,.2,True,'ok',0,3,{'beta': np.array([.2,.8])})
    restored = TrainingResult.from_dict(r.to_dict())
    np.testing.assert_allclose(restored.best_feasible['beta'], [.2,.8])
    assert restored.success
