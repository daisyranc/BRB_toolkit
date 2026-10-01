from dataclasses import dataclass,field
import numpy as np

@dataclass
class MAKERResult:
    probabilities: object
    erps_mass: object
    untrustworthiness: float
    space: object
    status: str='ok'
    trace: list=field(default_factory=list)

    @property
    def trustworthiness(self): return 1-self.untrustworthiness

    @property
    def ambiguity(self):
        return float(sum(p for m,p in zip(self.space.masks,self.probabilities) if m.bit_count()>1))

    def pignistic(self):
        """Explicit decision transform, not part of MAKER inference."""
        p=np.zeros(len(self.space.states))
        for m,value in zip(self.space.masks,self.probabilities):
            for i in range(len(p)):
                if m&(1<<i): p[i]+=value/m.bit_count()
        return p
