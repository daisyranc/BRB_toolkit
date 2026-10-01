import numpy as np

class StateSpace:
    """Nonempty focal sets, represented by bit masks; Theta is not 2^Theta."""
    def __init__(self,states,*,max_focal_sets=1023):
        self.states=tuple(states)
        if not self.states or len(set(self.states))!=len(self.states):
            raise ValueError('States must be nonempty and distinct.')
        if len(self.states)>max_focal_sets.bit_length() or 2**len(self.states)-1>max_focal_sets:
            raise ValueError('Full powerset exceeds resource limit.')
        self.masks=tuple(range(1,2**len(self.states)))
        self.theta=self.masks[-1]
        a=np.array(self.masks)
        self.intersections=a[:,None]&a[None,:]
        self.labels=tuple('{'+','.join(str(self.states[i]) for i in range(len(self.states)) if m&(1<<i))+'}' for m in self.masks)

    def __len__(self): return len(self.masks)

    def __eq__(self,other): return isinstance(other,StateSpace) and self.states==other.states
