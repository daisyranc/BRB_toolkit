import numpy as np
import pytest
import torch

from examples.paper_cases.diagnosis import build_diagnosis_case, apply_paper_parameters
from brb.maker.objectives import paper_objectives
from brb.maker.torch_backend import TorchMAKERBackend


@pytest.mark.parametrize('include_prior', [True, False])
def test_batch_matches_numpy(include_prior):
    model, data, case = build_diagnosis_case()
    apply_paper_parameters(model, case)
    inputs = [dict(zip(model.order, v)) for v in case['joint_order']]
    backend = TorchMAKERBackend(model)
    actual = backend.infer_batch(inputs, include_prior=include_prior)
    expected = [model.infer(v, include_prior=include_prior) for v in inputs]
    np.testing.assert_allclose(actual['probabilities'].detach(), [r.probabilities for r in expected], atol=1e-10, rtol=1e-8)
    np.testing.assert_allclose(actual['trustworthiness'].detach(), [r.trustworthiness for r in expected], atol=1e-10, rtol=1e-8)


def test_objective_vectors_match_numpy():
    model, data, _ = build_diagnosis_case()
    actual = TorchMAKERBackend(model).objectives(data)
    for a, e in zip(actual, paper_objectives(model, data)):
        np.testing.assert_allclose(a.detach(), e, atol=1e-10, rtol=1e-8)


def test_cuda_unavailable_raises():
    if torch.cuda.is_available():
        pytest.skip('Requires unavailable CUDA')
    model, _, _ = build_diagnosis_case()
    with pytest.raises(RuntimeError, match='CUDA'):
        TorchMAKERBackend(model, device='cuda')


def test_missing_subsets_preserve_order():
    model, _, case = build_diagnosis_case()
    inputs = [{}, {'blood': case['value_order'][0]}, dict(zip(model.order, case['joint_order'][0])), {'saliva': case['value_order'][1]}]
    actual = TorchMAKERBackend(model, batch_size=1).infer_batch(inputs)
    expected = [model.infer(v) for v in inputs]
    np.testing.assert_allclose(actual['probabilities'].detach(), [r.probabilities for r in expected], atol=1e-10)


@pytest.mark.parametrize('mode', ['dependent', 'independent'])
def test_composite_focal_intersections(mode):
    from brb.maker import StateSpace, EvidenceTable, MAKERModel
    s = StateSpace(['a', 'b']); p = [[.6, .3, .1]]
    tables = {(k,): EvidenceTable(s, ['v'], p) for k in ('x', 'y', 'z')}
    for key in [('x', 'y'), ('x', 'y', 'z')]:
        tables[key] = EvidenceTable(s, [tuple('v' for _ in key)], p, joint_support=[np.diag(p[0])])
    params = {key: {'weights': np.array([.8, .7, .6]), 'reliabilities': np.array([.8])} for key in tables}
    model = MAKERModel(tables, params, order=['x', 'y', 'z'], mode=mode)
    row = {'x': 'v', 'y': 'v', 'z': 'v'}
    result = TorchMAKERBackend(model).infer_batch([row])
    expected = model.infer(row)
    np.testing.assert_allclose(result['probabilities'][0], expected.probabilities, atol=1e-10)
    np.testing.assert_allclose(result['trustworthiness'][0], expected.trustworthiness, atol=1e-10)


def test_undefined_limit_and_unsupported_mode():
    from brb.maker import StateSpace, EvidenceTable, MAKERModel
    s = StateSpace(['a', 'b'])
    tables = {('x',): EvidenceTable(s, [0], [[1, 0, 0]]), ('y',): EvidenceTable(s, [0], [[0, 1, 0]])}
    params = {k: {'weights': np.ones(3), 'reliabilities': np.ones(1)} for k in tables}
    model = MAKERModel(tables, params, mode='independent')
    result = TorchMAKERBackend(model).infer_batch([{'x': 0, 'y': 0}])
    assert result['status'] == ['undefined_limit']
    model.mode = 'exclusive'
    with pytest.raises(ValueError, match='supports'):
        TorchMAKERBackend(model)


def test_single_evidence_no_support_status():
    model, _, case = build_diagnosis_case()
    model.parameters[('saliva',)]['weights'][:] = 0
    model.parameters[('saliva',)]['reliabilities'][:] = 1
    row = {'saliva': case['value_order'][0]}
    expected = model.infer(row, include_prior=False)
    result = TorchMAKERBackend(model).infer_batch([row], include_prior=False)
    assert expected.status == 'no_support'
    assert result['status'] == [expected.status]


def test_invalid_reference_parameters_raise():
    model, data, _ = build_diagnosis_case()
    key = data.groups[0].key
    data.reference_parameters[key]['weights'][:] = 2
    with pytest.raises(ValueError, match='reference'):
        TorchMAKERBackend(model).objectives(data)


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA hardware unavailable'))])
def test_reused_backend_reads_current_parameters(device):
    model, _, case = build_diagnosis_case()
    backend = TorchMAKERBackend(model, device=device)
    row = dict(zip(model.order, case['joint_order'][0]))
    before = backend.infer_batch([row])['probabilities'].cpu().numpy().copy()
    model.parameters[('saliva',)]['weights'] = np.array([.1, .9, .4])
    actual = backend.infer_batch([row])['probabilities'].cpu().numpy()
    np.testing.assert_allclose(actual[0], model.infer(row).probabilities, atol=1e-10)
    assert not np.allclose(actual, before)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA hardware unavailable')
def test_cuda_matches_cpu():
    model, data, case = build_diagnosis_case()
    cpu = TorchMAKERBackend(model)
    gpu = TorchMAKERBackend(model, device='cuda')
    rows = [dict(zip(model.order, v)) for v in case['joint_order']]
    for key in ['probabilities', 'trustworthiness']:
        np.testing.assert_allclose(gpu.infer_batch(rows)[key].cpu(), cpu.infer_batch(rows)[key], atol=1e-10, rtol=1e-8)
    for a, b in zip(gpu.objectives(data), cpu.objectives(data)):
        np.testing.assert_allclose(a.cpu(), b, atol=1e-10, rtol=1e-8)
