from dataclasses import dataclass, field
import numpy as np

@dataclass
class TrainingResult:
    objective_before: float
    objective_after: float
    success: bool
    status: str
    constraint_violation: float
    iterations: int
    best_feasible: dict
    stage_results: list = field(default_factory=list)

    def to_dict(self):
        def encode(x):
            if isinstance(x, np.ndarray): return {'array':x.tolist()}
            if isinstance(x, dict): return {k:encode(v) for k,v in x.items()}
            if isinstance(x, (list,tuple)): return [encode(v) for v in x]
            if isinstance(x, np.generic): return x.item()
            return x
        return encode(vars(self))

    @classmethod
    def from_dict(cls, value):
        def decode(x):
            if isinstance(x, dict):
                if set(x) == {'array'}: return np.array(x['array'], float)
                return {k:decode(v) for k,v in x.items()}
            if isinstance(x, list): return [decode(v) for v in x]
            return x
        return cls(**decode(value))
