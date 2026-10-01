import numpy as np
from examples.paper_cases.graphite import load_graphite_case,build_graphite_model,graphite_inputs

def test_r06_complete_and_incomplete():
    d=load_graphite_case(); m=build_graphite_model()
    result=m.infer(graphite_inputs(d['training'][2][:4]))
    for node in ['X2','X4','X6','X8']:
        np.testing.assert_allclose(result.trace[node].beliefs[0],d['paper_reference'][node],atol=5e-5,rtol=0)
    x=graphite_inputs(d['training'][2][:4]); x['X5']=np.array([[0,0,.8]])
    r=m.infer(x)
    np.testing.assert_allclose(r.beliefs[0], d['paper_reference']['X9_incomplete'],atol=5e-5,rtol=0)
    np.testing.assert_allclose(r.ignorance[0], .0371,atol=5e-5,rtol=0)
