"""Batched float64 MAKER evidence calculations on CPU or CUDA."""
from collections import defaultdict

import numpy as np
import torch


class TorchMAKERBackend:
    def __init__(self, model, *, device='cpu', dtype=torch.float64, batch_size=256):
        self.device = torch.device(device)
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable; select device="cpu" explicitly.')
        if model.mode not in ('dependent', 'independent'):
            raise ValueError('Torch backend supports dependent and independent modes only.')
        if batch_size < 1:
            raise ValueError('batch_size must be positive.')
        self.model, self.dtype, self.batch_size = model, dtype, batch_size
        self.tables = {key: self.tensor(t.probabilities) for key, t in model.tables.items()}
        self.joints = {key: self.tensor(t.joint_support) for key, t in model.tables.items() if t.joint_support is not None}
        self.parameters = {k: {n: self.tensor(v).clone() for n, v in p.items()} for k, p in model.parameters.items()}
        self._parameter_snapshot = {k: {n: np.array(v, copy=True) for n, v in p.items()} for k, p in model.parameters.items()}
        self._indices = {}
        self._training = {}
        masks = model.space.masks
        index = np.full(model.space.intersections.shape, -1, dtype=np.int64)
        for j, mask in enumerate(masks):
            index[model.space.intersections == mask] = j
        self.valid = torch.as_tensor(index.ravel() >= 0, device=self.device)
        self.intersection = torch.as_tensor(np.maximum(index.ravel(), 0), device=self.device)

    def tensor(self, value):
        return torch.as_tensor(value, dtype=self.dtype, device=self.device)

    def _current_parameters(self):
        for key, p in self.model.parameters.items():
            for kind, value in p.items():
                old = self._parameter_snapshot[key][kind]
                if not np.array_equal(old, value):
                    value = np.asarray(value, float)
                    if value.shape != old.shape or not np.isfinite(value).all() or np.any((value < 0) | (value > 1)):
                        raise ValueError('Invalid updated MAKER parameters.')
                    self.parameters[key][kind] = self.tensor(value).clone()
                    self._parameter_snapshot[key][kind] = value.copy()
        return self.parameters

    def _acquire(self, key, values, parameters):
        cache_key = (key, tuple(values))
        if cache_key not in self._indices:
            self._indices[cache_key] = torch.as_tensor([self.model.tables[key].index(v) for v in values], device=self.device)
        index = self._indices[cache_key]
        p = self.tables[key][index]
        w = parameters[key]['weights'].expand_as(p)
        r = parameters[key]['reliabilities'][index]
        return p, w, r, index

    def _products(self, a, b, key, values, parameters):
        pa, wa, ra, _ = a
        pb, wb, rb, _ = b
        products = (wa * pa)[:, :, None] * (wb * pb)[:, None, :]
        if key is None:
            return products
        if key not in self.joints:
            raise ValueError(f'Missing prefix joint support for {key}.')
        pj, wj, rj, idx = self._acquire(key, values, parameters)
        denominator = pa[:, :, None] * pb[:, None, :]
        joint = self.joints[key][idx]
        if torch.any((denominator == 0) & (joint > 0)).item():
            raise ValueError('Positive joint support with zero marginal support.')
        alpha = joint / torch.where(denominator > 0, denominator, torch.ones_like(denominator))
        weight_num = wj[:, self.intersection].reshape(products.shape)
        weight_num = weight_num * self.valid.reshape(1, *products.shape[1:])
        weight_den = wa[:, :, None] * wb[:, None, :]
        if torch.any((weight_den == 0) & (weight_num * alpha > 0)).item():
            raise ValueError('Undefined weight index at zero marginal weight.')
        wi = weight_num / torch.where(weight_den > 0, weight_den, torch.ones_like(weight_den))
        wi = torch.where(weight_den > 0, wi, torch.zeros_like(wi))
        da = (wa * pa).sum(1) + 1 - ra
        db = (wb * pb).sum(1) + 1 - rb
        dj = (wj * pj).sum(1) + 1 - rj
        if torch.any((da <= 0) | (db <= 0) | (dj <= 0)).item():
            raise ValueError('Zero projection denominator.')
        return products * alpha * wi * (da * db / dj)[:, None, None]

    def _intersection_sum(self, products):
        flat = products.flatten(1)[:, self.valid]
        target = self.intersection[self.valid].expand(len(products), -1)
        return torch.zeros((len(products), len(self.model.space)), device=self.device, dtype=self.dtype).scatter_add(1, target, flat)

    @staticmethod
    def _finish(mass, residual):
        total = mass.sum(1)
        valid = (total > 0) & (total + residual > 0)
        den = torch.where(valid, total + residual, torch.ones_like(total))
        probabilities = mass / torch.where(valid, total, torch.ones_like(total))[:, None]
        probabilities = torch.where(valid[:, None], probabilities, torch.zeros_like(probabilities))
        erps = torch.where(valid[:, None], mass / den[:, None], torch.zeros_like(mass))
        unknown = torch.where(valid, residual / den, torch.ones_like(residual))
        undefined = (total <= 0) & (residual == 0)
        return probabilities, erps, unknown, valid, undefined

    def _infer_group(self, rows, keys, include_prior, parameters):
        if not keys:
            if self.model.prior is None or not include_prior:
                raise ValueError('No observed input or prior.')
            a = self._acquire(self.model.prior, [self.model.tables[self.model.prior].values[0]] * len(rows), parameters)
        else:
            a = self._acquire((keys[0],), [row[keys[0]] for row in rows], parameters)
        mass, residual = a[0] * a[1], 1 - a[2]
        result = None
        for step in range(1, len(keys)):
            b = self._acquire((keys[step],), [row[keys[step]] for row in rows], parameters)
            prefix = tuple(keys[:step])
            dependent = self.model.mode == 'dependent'
            acquired = a if step == 1 else (self._acquire(prefix, [tuple(row[k] for k in prefix) for row in rows], parameters) if dependent else None)
            jointkey = tuple(keys[:step + 1]) if dependent else None
            if dependent:
                coeff = self._dependent_coeff(acquired, b, jointkey, rows, parameters)
                products = mass[:, :, None] * (b[0] * b[1])[:, None, :] * coeff
            else:
                products = mass[:, :, None] * (b[0] * b[1])[:, None, :]
            mass = (1 - b[2])[:, None] * mass + residual[:, None] * (b[0] * b[1]) + self._intersection_sum(products)
            residual = residual * (1 - b[2])
            result = self._finish(mass, residual)
            if step > 1:
                result = (*result[:4], torch.zeros_like(result[4]))
            if not result[3].all().item() and step < len(keys) - 1:
                raise ValueError('Cannot extend incompatible or undefined result.')
            mass, residual = result[1], result[2]
        if keys and self.model.prior is not None and include_prior:
            extended = result is not None
            if result is not None and not result[3].all().item():
                raise ValueError('Cannot extend incompatible or undefined result.')
            b = self._acquire(self.model.prior, [self.model.tables[self.model.prior].values[0]] * len(rows), parameters)
            products = mass[:, :, None] * (b[0] * b[1])[:, None, :]
            mass = (1 - b[2])[:, None] * mass + residual[:, None] * (b[0] * b[1]) + self._intersection_sum(products)
            residual = residual * (1 - b[2])
            result = self._finish(mass, residual)
            if extended:
                result = (*result[:4], torch.zeros_like(result[4]))
        if result is None:
            result = self._finish(mass, residual)
            result = (*result[:4], torch.zeros_like(result[4]))
        return result

    def _dependent_coeff(self, a, b, key, rows, parameters):
        # Compute coefficients with unit pseudo-masses while retaining acquisition quantities.
        pa, wa, ra, _ = a
        pb, wb, rb, _ = b
        pj, wj, rj, idx = self._acquire(key, [tuple(row[k] for k in key) for row in rows], parameters)
        den = pa[:, :, None] * pb[:, None, :]
        support = self.joints[key][idx]
        if torch.any((den == 0) & (support > 0)).item():
            raise ValueError('Positive joint support with zero marginal support.')
        alpha = support / torch.where(den > 0, den, torch.ones_like(den))
        num = wj[:, self.intersection].reshape(len(rows), len(self.model.space), len(self.model.space)) * self.valid.reshape(1, len(self.model.space), len(self.model.space))
        wd = wa[:, :, None] * wb[:, None, :]
        if torch.any((wd == 0) & (num * alpha > 0)).item():
            raise ValueError('Undefined weight index at zero marginal weight.')
        wi = torch.where(wd > 0, num / torch.where(wd > 0, wd, torch.ones_like(wd)), torch.zeros_like(wd))
        da = (wa * pa).sum(1) + 1 - ra
        db = (wb * pb).sum(1) + 1 - rb
        dj = (wj * pj).sum(1) + 1 - rj
        if torch.any((da <= 0) | (db <= 0) | (dj <= 0)).item():
            raise ValueError('Zero projection denominator.')
        return alpha * wi * (da * db / dj)[:, None, None]

    def infer_batch(self, inputs, *, include_prior=True, parameters=None):
        params = self._current_parameters() if parameters is None else parameters
        groups = defaultdict(list)
        for i, row in enumerate(inputs):
            if set(row) - set(self.model.order):
                raise ValueError('Unknown input variable.')
            keys = tuple(k for k in self.model.order if row.get(k) is not None)
            groups[keys].append((i, row))
        values, positions, statuses = [], [], ['ok'] * len(inputs)
        for keys, entries in groups.items():
            for start in range(0, len(entries), self.batch_size):
                chunk = entries[start:start + self.batch_size]
                result = self._infer_group([r for _, r in chunk], keys, include_prior, params)
                values.append((result[0], 1 - result[2]))
                positions.extend(i for i, _ in chunk)
                for (i, _), valid, undefined in zip(chunk, result[3].detach().cpu().tolist(), result[4].detach().cpu().tolist()):
                    statuses[i] = 'ok' if valid else ('undefined_limit' if undefined else 'no_support')
        if not values:
            return {'probabilities': self.tensor(np.empty((0, len(self.model.space)))), 'trustworthiness': self.tensor([]), 'status': statuses}
        order = torch.as_tensor(np.argsort(positions), device=self.device)
        return {'probabilities': torch.cat([v[0] for v in values])[order], 'trustworthiness': torch.cat([v[1] for v in values])[order], 'status': statuses}

    def objectives(self, data, *, parameters=None):
        params = self._current_parameters() if parameters is None else parameters
        first, second = [], []
        for group in data.groups:
            cache_key = (id(data), id(group))
            if cache_key not in self._training:
                observed = self.tensor(group.counts)
                indices = torch.as_tensor([self.model.tables[group.key].index(v) for v in group.values], device=self.device)
                reference = data.reference_parameters[group.key]
                if group.counts.shape[1] != len(self.model.space):
                    raise ValueError('Observation state dimension mismatch.')
                for kind, value in self.parameters[group.key].items():
                    if kind not in reference or np.shape(reference[kind]) != tuple(value.shape):
                        raise ValueError('Invalid reference parameter dimensions.')
                    r = np.asarray(reference[kind])
                    if not np.isfinite(r).all() or np.any((r < 0) | (r > 1)):
                        raise ValueError('Invalid reference parameters.')
                self._training[cache_key] = {
                    'group': group, 'data': data,
                    'rows': [{group.key[0]: v} if len(group.key) == 1 else dict(zip(group.key, v)) for v in group.values],
                    'counts': observed, 'n': observed.sum(1), 'targets': self.tensor(group.targets),
                    'indices': indices, 'weights': self.tensor(reference['weights']),
                    'reliabilities': self.tensor(reference['reliabilities'])[indices]}
            cached = self._training[cache_key]
            rows = cached['rows']
            result = self.infer_batch(rows, parameters=params)
            if any(s != 'ok' for s in result['status']):
                raise ValueError('Undefined training prediction.')
            observed, n, targets = cached['counts'], cached['n'], cached['targets']
            error = .5 * ((targets - result['probabilities'][:, None, :]) ** 2).sum(-1)
            first.append((observed / data.total * error).flatten())
            p = params[group.key]
            we = (observed / n[:, None] * (cached['weights'] - p['weights']) ** 2).sum(1)
            re = (cached['reliabilities'] - p['reliabilities'][cached['indices']]) ** 2
            second.append(torch.stack((n / data.total * we, n / data.total * re), 1).flatten())
        return torch.cat(first), torch.cat(second)
