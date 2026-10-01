from dataclasses import dataclass, field
import numpy as np

@dataclass
class RIMERResult:
    beliefs: np.ndarray
    ignorance: np.ndarray
    activation: np.ndarray
    completeness: np.ndarray
    status: np.ndarray
    utility_interval: object = None
    trace: dict = field(default_factory=dict)

    def expected_value(self, utilities, *, decision_policy=None):
        u=np.asarray(utilities,float)
        v=self.beliefs@u
        if np.any(self.ignorance>1e-9):
            if decision_policy=='midpoint': v+=self.ignorance*(u.min()+u.max())/2
            elif decision_policy=='lower': v+=self.ignorance*u.min()
            elif decision_policy=='upper': v+=self.ignorance*u.max()
            else: raise ValueError('Incomplete output requires an explicit decision policy.')
        return v
