import numpy as np
import pytest
from brb.maker import StateSpace,Evidence,JointEvidence,combine_pair,dependency_indices

def test_independent_intersections_and_bayes():
    s=StateSpace(['a','b'])
    a=Evidence([.7,.2,.1],1,1,s); b=Evidence([.3,.5,.2],1,1,s)
    r=combine_pair(a,b,mode='independent')
    # a gets a*a, a*Theta and Theta*a; similarly for b.
    np.testing.assert_allclose(r.probabilities,np.array([.38,.19,.02])/.59)
    assert r.untrustworthiness==0
    x=combine_pair(Evidence([.8,.2,0],1,1,s),Evidence([.1,.9,0],1,1,s),mode='independent')
    np.testing.assert_allclose(x.probabilities,[.08/.26,.18/.26,0])

def test_dependent_reduces_to_independent_with_explicit_indices():
    s=StateSpace(['a','b']); a=Evidence([.6,.3,.1],[.8,.7,.9],.85,s);b=Evidence([.2,.5,.3],[.7,.9,.8],.9,s)
    joint=JointEvidence([.4,.4,.2],1,.8,np.outer(a.probabilities,b.probabilities),
                        weight_indices=np.ones((3,3)),projection_index=1)
    x=combine_pair(a,b,joint=joint);y=combine_pair(a,b,mode='independent')
    np.testing.assert_allclose(x.erps_mass,y.erps_mass)
    np.testing.assert_allclose(dependency_indices(a.probabilities,b.probabilities,joint.joint_support),1)
    with pytest.raises(ValueError): combine_pair(a,b)

def test_exclusive_limits_no_support_and_ambiguity():
    s=StateSpace(['a','b']);a=Evidence([1,0,0],1,.8,s);b=Evidence([0,1,0],1,.6,s)
    r=combine_pair(a,b,mode='exclusive');np.testing.assert_allclose(r.probabilities,[2/3,1/3,0])
    assert r.untrustworthiness>0
    x=combine_pair(Evidence([1,0,0],1,1,s),Evidence([0,1,0],1,1,s),mode='exclusive')
    assert x.status=='undefined_limit'
    x=combine_pair(Evidence([1,0,0],1,1,s),Evidence([0,1,0],1,1,s),mode='exclusive',derivative_rates=(1,2))
    np.testing.assert_allclose(x.probabilities,[2/3,1/3,0])
    x=combine_pair(Evidence([0,0,1],1,.8,s),Evidence([0,0,1],1,.8,s),mode='independent')
    assert x.ambiguity==1 and x.untrustworthiness>0
    x=combine_pair(Evidence([1,0,0],0,0,s),Evidence([0,1,0],0,0,s),mode='independent')
    assert x.status=='no_support' and x.untrustworthiness==1

def test_augmented_dempster_total_conflict():
    s=StateSpace(['a','b'])
    x=combine_pair(Evidence([1,0,0],1,1,s),Evidence([0,1,0],1,1,s),mode='independent',augment=True)
    np.testing.assert_allclose(x.probabilities,[.5,.5,0])
    assert x.status=='differential_limit'
