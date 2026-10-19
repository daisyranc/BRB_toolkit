import numpy as np
import pytest
import torch
from brb.rimer import RIMERAttribute, RIMERConfig, RIMERModel
from brb.rimer.torch_backend import TorchRIMERBackend
from examples.paper_cases.graphite import build_graphite_model, graphite_inputs

def test_sparse_er_output_is_valid_for_next_layer():
    from brb.paper_common.validation import validate_beliefs
    rng = np.random.default_rng(42)
    beta = rng.dirichlet([.1, .1, .1], 81)
    beta[:, 0] = 0.
    beta /= beta.sum(1, keepdims=True)
    model = RIMERModel(RIMERConfig(
        [RIMERAttribute(str(i), [0., 1., 2.]) for i in range(4)],
        ['a', 'b', 'c'], None, beta, rng.uniform(.01, 1., 81), [1.] * 4))
    inputs = [rng.dirichlet([1., 1., 1.], 1000) for _ in range(4)]
    actual = TorchRIMERBackend(model).infer(inputs)
    beliefs = actual.beliefs.detach().numpy()
    assert np.all(beliefs >= 0.)
    validate_beliefs(beliefs)
    np.testing.assert_allclose(beliefs[:, 0], 0., atol=2e-14)
    np.testing.assert_allclose(beliefs, model.infer(inputs).beliefs, atol=2e-13)

@pytest.mark.parametrize('connective', ['and', 'or'])
def test_partial_missing_and_gradients(connective):
    m = RIMERModel(RIMERConfig([RIMERAttribute('x',[0.,1.])], ['a','b'], None, [[.3,.7],[.2,.4]], [.6,.8],[.7], connective=connective, utilities=[0.,2.]))
    x = [np.array([[.2,.8],[0.,0.],[.1,.4]])]
    backend = TorchRIMERBackend(m)
    actual = backend.infer(x)
    expected = m.infer(x)
    np.testing.assert_allclose(actual.beliefs.detach(), expected.beliefs, atol=1e-10)
    np.testing.assert_allclose(actual.ignorance.detach(), expected.ignorance, atol=1e-10)
    assert actual.status == list(expected.status)
    beta = torch.tensor(m.config.rule_beliefs, dtype=torch.float64, requires_grad=True)
    result = backend.infer(x, parameters={'model':{'beta':beta}})
    result.beliefs.sum().backward()
    assert torch.isfinite(beta.grad).all()
    np.testing.assert_allclose(backend.predict([[.4],[None]]).beliefs.detach(), m.predict([[.4],[None]]).beliefs, atol=1e-10)

def test_hierarchy_parity_and_parameter_refresh():
    m=build_graphite_model(); x=graphite_inputs([1,2,3,4])
    backend=TorchRIMERBackend(m)
    np.testing.assert_allclose(backend.infer(x).beliefs.detach(),m.infer(x).beliefs,atol=1e-9)
    assert set(backend.infer(x).trace)==set(m.infer(x).trace)
    first=next(iter(m.models.values())); first.config.rule_weights[:]=.7
    np.testing.assert_allclose(backend.infer(x).beliefs.detach(),m.infer(x).beliefs,atol=1e-9)

@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_cuda_parity():
    m=build_graphite_model(); x=graphite_inputs([1,2,3,4])
    np.testing.assert_allclose(TorchRIMERBackend(m,device='cuda').infer(x).beliefs.detach().cpu(),m.infer(x).beliefs,atol=1e-8)
