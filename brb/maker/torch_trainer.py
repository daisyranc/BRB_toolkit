"""SLSQP prioritized minimax with device-side automatic differentiation."""
import numpy as np
import torch
from scipy.optimize import minimize

from ..paper_common import ParameterBlock, ParameterLayout, TrainingResult
from .torch_backend import TorchMAKERBackend


class TorchMAKERObjective:
    def __init__(self, backend, data, layout):
        self.backend, self.data, self.layout = backend, data, layout
        self.cached = None
        self.blocks = []
        for (key, parameters) in backend.model.parameters.items():
            for kind in parameters:
                block = layout.blocks[len(self.blocks)]
                self.blocks.append((key, kind, block))

    def _terms(self, vector):
        parameters, offset = {}, 0
        for key, kind, block in self.blocks:
            base = self.backend.tensor(block.values).clone()
            mask = torch.as_tensor(block.trainable, device=self.backend.device)
            size = int(block.trainable.sum())
            parameters.setdefault(key, {})[kind] = base.masked_scatter(mask, vector[offset:offset + size])
            offset += size
        return self.backend.objectives(self.data, parameters=parameters)

    def evaluate(self, vector, *, jacobian=False):
        vector = np.asarray(vector, float)
        if self.cached is not None and np.array_equal(vector, self.cached[0]):
            if not jacobian or len(self.cached[1]) == 4:
                return self.cached[1] if jacobian else self.cached[1][:2]
        x = self.backend.tensor(vector).requires_grad_(True)
        a, b = self._terms(x)
        output = [a.detach().cpu().numpy(), b.detach().cpu().numpy()]
        if jacobian:
            # Chunked batched reverse-mode avoids one Python backward per objective item.
            combined = torch.cat((a, b))
            chunks = []
            for start in range(0, len(combined), 32):
                stop = min(start + 32, len(combined))
                seed = torch.zeros((stop-start, len(combined)), dtype=x.dtype, device=x.device)
                seed[:, start:stop] = torch.eye(stop-start, dtype=x.dtype, device=x.device)
                grad = torch.autograd.grad(combined, x, seed, is_grads_batched=True, retain_graph=True)[0]
                chunks.append(grad.detach().cpu().numpy())
            jac = np.concatenate(chunks, axis=0)
            output.extend([jac[:len(a)], jac[len(a):]])
        if not all(np.isfinite(v).all() for v in output):
            raise ValueError('Nonfinite MAKER objective or Jacobian.')
        self.cached = (vector.copy(), tuple(output))
        return tuple(output)


class TorchMAKERTrainer:
    def __init__(self, maxiter=1000, ftol=1e-10, priority_tolerance=1e-5, *, device='cpu', batch_size=256, config=None, optimizer='slsqp'):
        if maxiter < 0 or ftol <= 0 or priority_tolerance < 0:
            raise ValueError('Invalid optimiser settings.')
        self.maxiter, self.ftol, self.priority_tolerance = maxiter, ftol, priority_tolerance
        self.device, self.batch_size = device, batch_size
        self.config = config
        self.optimizer = optimizer

    def fit(self, model, data, *, bounds, trainable=None, expert_constraints=(), validation=None):
        if self.config is not None or self.optimizer != 'slsqp':
            from .device_training import fit_device
            from ..optimization import TrainingConfig
            config = self.config or TrainingConfig(optimizer=self.optimizer, device=self.device, maxiter=self.maxiter, ftol=self.ftol, batch_size=self.batch_size)
            return fit_device(model, data, bounds, trainable, expert_constraints, validation, config, self.priority_tolerance)
        blocks = []
        for key, p in model.parameters.items():
            for kind, a in p.items():
                lo, hi = bounds[key][kind]
                flag = True if trainable is None else trainable.get((key, kind), False)
                blocks.append(ParameterBlock('/'.join(key)+'.'+kind, a, lo, hi, flag))
        layout = ParameterLayout(blocks)
        initial = layout.pack(); n = len(initial)
        evaluator = TorchMAKERObjective(TorchMAKERBackend(model, device=self.device, batch_size=self.batch_size), data, layout)
        before = float(evaluator.evaluate(initial)[0].max())

        def extras(x):
            residual = 0.
            for c in expert_constraints:
                a = np.asarray(c['fun'](x[:n]), float)
                if not np.isfinite(a).all():
                    raise ValueError('Nonfinite expert constraint.')
                residual = max(residual, float(np.max(np.abs(a) if c['type'] == 'eq' else -a, initial=0)))
            return residual

        def stage(start, index, primary_limit=None):
            z0 = np.r_[start, float(evaluator.evaluate(start)[index].max())]

            def inequality(z):
                a, b = evaluator.evaluate(z[:n])
                result = z[-1] - (a, b)[index]
                return result if primary_limit is None else np.r_[result, primary_limit-a]

            def jacobian(z):
                a, b, ja, jb = evaluator.evaluate(z[:n], jacobian=True)
                result = np.column_stack((-(ja, jb)[index], np.ones(len((a, b)[index]))))
                return result if primary_limit is None else np.vstack((result, np.column_stack((-ja, np.zeros(len(a))))))

            def violation(z):
                value = max(float(np.max(-inequality(z), initial=0)), extras(z), 0.)
                for v, (lo, hi) in zip(z[:n], layout.bounds()):
                    value = max(value, lo-v, v-hi)
                return value

            best = z0.copy() if violation(z0) < 1e-8 else None
            score = z0[-1] if best is not None else np.inf
            history = []

            def record(z, event):
                a, b = evaluator.evaluate(z[:n])
                history.append({'event': event, 'iteration': sum(h['event'] == 'iteration' for h in history) + int(event == 'iteration'),
                                'f1': float(a.max()), 'f2': float(b.max()), 'epigraph': float(z[-1]),
                                'constraint_violation': violation(z), 'priority_limit': primary_limit})

            record(z0, 'initial')

            def track(z):
                nonlocal best, score
                value = evaluator.evaluate(z[:n])[index].max()
                if violation(z) < 1e-8 and (best is None or value < score):
                    best, score = z.copy(), float(value)

            def callback(z):
                track(z)
                record(z, 'iteration')

            constraints = [{'type': 'ineq', 'fun': inequality, 'jac': jacobian}]
            for c in expert_constraints:
                entry = {'type': c['type'], 'fun': lambda z, c=c: c['fun'](z[:n])}
                if 'jac' in c:
                    entry['jac'] = lambda z, c=c: np.column_stack((np.atleast_2d(c['jac'](z[:n])), np.zeros(np.atleast_2d(c['jac'](z[:n])).shape[0])))
                constraints.append(entry)
            if n and self.maxiter:
                gradient = np.r_[np.zeros(n), 1.]
                sol = minimize(lambda z: z[-1], z0, jac=lambda z: gradient, method='SLSQP', bounds=layout.bounds()+[(0, None)], constraints=constraints, callback=callback, options={'maxiter': self.maxiter, 'ftol': self.ftol})
                track(sol.x)
                success = bool(sol.success and violation(sol.x) < 1e-7)
                status, nit = str(sol.message), int(sol.nit)
            else:
                success = not n and violation(z0) < 1e-8
                status = 'No trainable parameters' if not n else 'Iteration limit is zero'
                nit = 0
            if best is None:
                best = z0.copy(); success = False; status += '; no feasible candidate found'
            success = bool(success and violation(best) < 1e-7)
            a, b = evaluator.evaluate(best[:n])
            record(best, 'final')
            info = {'stage': index+1, 'success': success, 'status': status, 'f1': float(a.max()), 'f2': float(b.max()), 'iterations': nit, 'constraint_violation': violation(best), 'priority_limit': primary_limit, 'history': history}
            return best[:n].copy(), info

        first, info = stage(initial, 0)
        stages = [info]; chosen = first; success = info['success']; status = info['status']
        if success:
            second, info2 = stage(first, 1, info['f1']+self.priority_tolerance)
            stages.append(info2); success, status = info2['success'], info2['status']
            if success:
                chosen = second
            else:
                status = 'Second stage failed; restored first-stage feasible parameters: '+status
        layout.unpack(chosen)
        after = float(evaluator.evaluate(chosen)[0].max())
        residual = extras(chosen)
        if len(stages) == 2:
            residual = max(residual, after-stages[0]['f1']-self.priority_tolerance)
        return TrainingResult(before, after, success, status, max(residual, 0), sum(s['iterations'] for s in stages), layout.snapshot(), stages)
