from dataclasses import dataclass
import numpy as np

@dataclass
class ParameterBlock:
    name: str
    values: np.ndarray
    lower: object = 0
    upper: object = 1
    trainable: object = True

    def __post_init__(self):
        self.values = np.asarray(self.values, float)
        self.lower = np.broadcast_to(self.lower, self.values.shape).copy()
        self.upper = np.broadcast_to(self.upper, self.values.shape).copy()
        self.trainable = np.broadcast_to(self.trainable, self.values.shape).astype(bool).copy()
        if not np.isfinite(self.values).all() or np.any(self.lower > self.upper):
            raise ValueError('Invalid parameter values/bounds.')
        if np.any(self.values < self.lower) or np.any(self.values > self.upper):
            raise ValueError('Initial parameters lie outside bounds.')

class ParameterLayout:
    def __init__(self, blocks):
        self.blocks = list(blocks)
        if len({b.name for b in blocks}) != len(blocks):
            raise ValueError('Parameter block names must be unique.')

    def pack(self):
        return np.concatenate([b.values[b.trainable] for b in self.blocks]) if self.blocks else np.empty(0)

    def unpack(self, vector):
        x = np.asarray(vector, float)
        if x.shape != self.pack().shape: raise ValueError('Parameter vector dimension mismatch.')
        offset = 0
        for b in self.blocks:
            n = int(b.trainable.sum())
            b.values[b.trainable] = x[offset:offset+n]
            offset += n

    def bounds(self):
        return [(lo, hi) for b in self.blocks for lo, hi in zip(b.lower[b.trainable], b.upper[b.trainable])]

    def snapshot(self):
        return {b.name: b.values.copy() for b in self.blocks}
