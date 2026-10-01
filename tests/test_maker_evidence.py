import numpy as np
import pytest
from brb.maker import StateSpace, EvidenceTable

def test_likelihood_is_state_conditional_not_posterior():
    space=StateSpace(['H1','H2'])
    counts=np.array([[275,215,95],[20,2490,105]])
    table=EvidenceTable.from_counts(space,['positive','negative'],counts)
    np.testing.assert_allclose(table.probabilities[0],[.6270,.0535,.3195],atol=5e-5)
    np.testing.assert_allclose(table.probabilities[1],[.0448,.6083,.3469],atol=5e-5)
    assert space.masks==(1,2,3)
    assert space.intersections[0,2]==1
    assert not np.allclose(table.probabilities[0],counts[0]/counts[0].sum())

def test_invalid_counts_and_resource_limit():
    space=StateSpace(['A','B'])
    for c in ([[1,0,2],[1,0,2]], [[-1,2,2],[1,2,2]]):
        with pytest.raises(ValueError): EvidenceTable.from_counts(space,['a','b'],c)
    with pytest.raises(ValueError): StateSpace(list(range(20)))
    with pytest.raises(ValueError): EvidenceTable(space,['a'],[[.5,.5,.5]])
