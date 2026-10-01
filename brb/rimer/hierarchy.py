from dataclasses import dataclass
import numpy as np

@dataclass
class RIMERNode:
    name: str
    model: object
    inputs: list
    mappings: object = None

class HierarchicalRIMER:
    def __init__(self,nodes,output):
        self.nodes=list(nodes); self.output=output
        by_name={n.name:n for n in nodes}
        if len(by_name)!=len(nodes) or output not in by_name: raise ValueError('Invalid graph node names/output.')
        self.models={n.name:n.model for n in nodes}
        pending=list(nodes); ordered=[]
        while pending:
            ready=[n for n in pending if all(k not in by_name or k in [p.name for p in ordered] for k in n.inputs)]
            if not ready: raise ValueError('Hierarchy must be acyclic.')
            for n in ready:
                if len(n.inputs)!=len(n.model.config.attributes): raise ValueError('Node input count mismatch.')
                for i,k in enumerate(n.inputs):
                    if k in by_name:
                        parent=by_name[k].model.config.consequents
                        child=n.model.config.attributes[i].references
                        mapping=(n.mappings or {}).get(k)
                        if mapping is None and list(parent)!=list(child): raise ValueError('Hierarchy reference semantics mismatch; provide explicit mapping.')
                        if mapping is not None:
                            matrix=np.asarray(mapping,float)
                            if matrix.shape!=(len(parent),len(child)) or (matrix<0).any() or not np.allclose(matrix.sum(1),1):
                                raise ValueError('Mapping must preserve committed mass.')
                ordered.append(n);pending.remove(n)
        self.nodes=ordered

    def infer(self,inputs):
        return self._infer(inputs)

    def _infer(self,inputs, *, training_trial=False):
        trace={}
        for n in self.nodes:
            beliefs=[]
            for k in n.inputs:
                if k in trace:
                    x=trace[k].beliefs
                    mapping=(n.mappings or {}).get(k)
                    if mapping is not None: x=x@np.asarray(mapping,float)
                else:
                    if k not in inputs: raise ValueError(f'Missing external input {k}.')
                    x=inputs[k]
                beliefs.append(x)
            trace[n.name]=n.model._infer(beliefs,training_trial=training_trial)
        result=trace[self.output]
        result.trace={k:v for k,v in trace.items() if k!=self.output}
        return result
