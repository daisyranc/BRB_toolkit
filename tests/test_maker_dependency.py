import numpy as np
import pytest
from brb.maker import StateSpace,EvidenceTable,MAKERModel,combine_pair,extend_result

def test_prefix_metadata_required_and_recursive_trace():
    s=StateSpace(['a','b']);p=[[.6,.3,.1]]
    tables={(k,):EvidenceTable(s,['v'],p) for k in ('x','y','z')}
    parameters={k:{'weights':np.ones(3),'reliabilities':np.array([.8])} for k in tables}
    model=MAKERModel(tables,parameters,order=['x','y','z'])
    with pytest.raises(ValueError,match='prefix'): model.infer(('v','v','v'))
    for key in [('x','y'),('x','y','z')]:
        tables[key]=EvidenceTable(s,[tuple('v' for _ in key)],p,joint_support=[np.diag(p[0])])
        parameters[key]={'weights':np.ones(3),'reliabilities':np.array([.8])}
    r=MAKERModel(tables,parameters,order=['x','y','z']).infer(('v','v','v'))
    assert len(r.trace)==4 and r.status=='ok'
    np.testing.assert_allclose(r.erps_mass.sum()+r.untrustworthiness,1)

def test_independent_recursive_carries_erps_not_projected_probability():
    s=StateSpace(['a','b']);p=[[.6,.3,.1]]
    tables={(k,):EvidenceTable(s,['v'],p) for k in ('x','y','z')}
    parameters={k:{'weights':np.array([.8,.7,.6]),'reliabilities':np.array([.8])} for k in tables}
    model=MAKERModel(tables,parameters,mode='independent')
    a,b,c=[tables[(k,)].acquire('v',parameters[(k,)]['weights'],parameters[(k,)]['reliabilities']) for k in ('x','y','z')]
    direct=extend_result(combine_pair(a,b,mode='independent'),c)
    r=model.infer(('v','v','v'))
    np.testing.assert_allclose(r.erps_mass,direct.erps_mass)
