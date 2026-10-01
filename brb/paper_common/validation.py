import numpy as np

def validate_beliefs(values, *, complete=False, atol=1e-10):
    """Validate without normalising away unassigned belief. / 验证并保留未知质量。"""
    a = np.asarray(values, dtype=np.float64)
    if a.ndim < 1 or not a.size or not np.isfinite(a).all() or (a < 0).any():
        raise ValueError('Beliefs must be nonempty, finite and nonnegative.')
    sums = a.sum(axis=-1)
    if (sums > 1 + atol).any() or (complete and not np.allclose(sums, 1, atol=atol, rtol=0)):
        raise ValueError('Belief row sums violate the complete/incomplete contract.')
    return a.copy()

def unit_interval(values, name):
    a = np.asarray(values, dtype=float)
    if not np.isfinite(a).all() or ((a < 0) | (a > 1)).any():
        raise ValueError(f'{name} must be finite and in [0,1].')
    return a.copy()
