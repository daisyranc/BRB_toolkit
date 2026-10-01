import numpy as np
from examples.paper_cases.diagnosis import build_diagnosis_case,apply_paper_parameters
from brb.maker import dependency_indices

def test_acquisition_dependency_and_paper_predictions():
    model,data,case=build_diagnosis_case()
    a=model.tables[('saliva',)].probabilities[0];b=model.tables[('blood',)].probabilities[0]
    alpha=dependency_indices(a,b,model.tables[('saliva','blood')].joint_support[0])
    np.testing.assert_allclose(alpha[0,0],1.7018,atol=5e-5)
    apply_paper_parameters(model,case)
    results=[model.infer(dict(zip(model.order,v)),include_prior=False) for v in case['joint_order']]
    np.testing.assert_allclose([r.probabilities for r in results],case['paper_reference']['table18'],atol=1.5e-4)
    np.testing.assert_allclose([r.trustworthiness for r in results],case['paper_reference']['trustworthiness'],atol=1.5e-4)
    assert data.total==6300
