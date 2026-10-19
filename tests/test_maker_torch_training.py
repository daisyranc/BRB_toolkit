import numpy as np
from examples.paper_cases.diagnosis import build_diagnosis_case
from brb.maker.objectives import paper_objectives
from brb.maker.torch_trainer import TorchMAKERTrainer, TorchMAKERObjective
from brb.maker.torch_backend import TorchMAKERBackend
from brb.paper_common import ParameterBlock, ParameterLayout


def setup_objective():
    model, data, _ = build_diagnosis_case()
    for p in model.parameters.values():
        for a in p.values():
            a[:] = .8
    blocks = [ParameterBlock('/'.join(key)+'.'+kind, a, .05, .99) for key, p in model.parameters.items() for kind, a in p.items()]
    layout = ParameterLayout(blocks)
    return model, data, layout, TorchMAKERObjective(TorchMAKERBackend(model), data, layout)


def test_jacobian_matches_central_difference():
    model, data, layout, objective = setup_objective()
    x = layout.pack()
    a, b, ja, jb = objective.evaluate(x, jacobian=True)
    for column in range(len(x)):
        delta = np.zeros_like(x); delta[column] = 1e-6
        plus = objective.evaluate(x + delta)
        minus = objective.evaluate(x - delta)
        for j, hi, lo in zip([ja, jb], plus, minus):
            np.testing.assert_allclose(j[:, column], (hi-lo)/2e-6, atol=1e-6, rtol=1e-4)


def test_zero_iterations_and_frozen_layout():
    model, data, _ = build_diagnosis_case()
    bounds = {k: {kind: (0, 1) for kind in p} for k, p in model.parameters.items()}
    initial = {k: {n: v.copy() for n, v in p.items()} for k, p in model.parameters.items()}
    result = TorchMAKERTrainer(maxiter=0).fit(model, data, bounds=bounds)
    assert not result.success and len(result.stage_results) == 1
    for k, p in initial.items():
        for n, v in p.items():
            np.testing.assert_array_equal(model.parameters[k][n], v)
    fixed = TorchMAKERTrainer().fit(model, data, bounds=bounds, trainable={})
    assert fixed.success and len(fixed.stage_results) == 2
    assert abs(fixed.objective_after - paper_objectives(model, data)[0].max()) < 1e-10
    for stage in fixed.stage_results:
        assert stage['history'][0]['event'] == 'initial'
        assert stage['history'][-1]['event'] == 'final'
        assert np.isfinite(stage['history'][-1]['constraint_violation'])


def test_stage_one_failure_stops_and_stage_two_rolls_back(monkeypatch):
    from types import SimpleNamespace
    import brb.maker.torch_trainer as module
    model, data, _ = build_diagnosis_case()
    bounds = {k: {kind: (0, 1) for kind in p} for k, p in model.parameters.items()}
    calls = []
    def solve(fun, x, **kwargs):
        calls.append(x.copy())
        return SimpleNamespace(x=x.copy(), success=len(calls) == 1, message='test', nit=1)
    monkeypatch.setattr(module, 'minimize', solve)
    report = TorchMAKERTrainer().fit(model, data, bounds=bounds)
    assert len(report.stage_results) == 2 and not report.success
    assert 'restored' in report.status
    np.testing.assert_allclose(np.concatenate([a for p in model.parameters.values() for a in p.values()]), calls[0][:-1])
    calls.clear()
    monkeypatch.setattr(module, 'minimize', lambda fun, x, **kw: SimpleNamespace(x=x.copy(), success=False, message='failed', nit=1))
    report = TorchMAKERTrainer().fit(model, data, bounds=bounds)
    assert len(report.stage_results) == 1 and not report.success
