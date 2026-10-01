"""Explicit table acquisition adapters; joint dependence is never guessed."""
import numpy as np
from .evidence import EvidenceTable

class EvidenceAcquisition:
    @staticmethod
    def from_counts(counts,*,values,space,smoothing=0,source=None):
        c=np.asarray(counts,float)
        if not np.isfinite(c).all() or (c<0).any(): raise ValueError('Invalid counts.')
        if not np.isfinite(smoothing) or smoothing<0: raise ValueError('Smoothing must be finite and nonnegative.')
        table=EvidenceTable.from_counts(space,values,c+smoothing,source=source)
        table.counts=c.copy()
        table.metadata={'smoothing':float(smoothing)}
        return table

    @staticmethod
    def from_frame(frame,*,inputs,state_column,space,discretizers=None):
        """Build singles and ordered prefix acquisitions from available records.

        This supplies acquired likelihoods, not A/B joint support allocations.
        General dependent models still require their explicit joint metadata.
        State entries may be singleton labels, nonempty subsets, integer masks,
        StateSpace.labels, or the conventional 'Theta' label.
        """
        import pandas as pd
        inputs=tuple(inputs);discretizers=discretizers or {}
        if not inputs or len(set(inputs))!=len(inputs) or state_column in inputs:
            raise ValueError('Distinct input and state columns required.')
        if any(k not in frame.columns for k in (*inputs,state_column)): raise ValueError('Missing frame column.')
        if set(discretizers)-set(inputs): raise ValueError('Discretizer names must be input columns.')
        work=frame.loc[:,list(inputs)+[state_column]].copy()
        available=work.loc[:,list(inputs)].notna()
        patterns={}
        for row in available.to_numpy():
            key=''.join('1' if v else '0' for v in row)
            patterns[key]=patterns.get(key,0)+1
        for key,transform in discretizers.items():
            selected=available[key]
            # Transform observed values only, preserving the original missing mask.
            values=transform(work.loc[selected,key].copy())
            if len(values)!=int(selected.sum()): raise ValueError('Discretizer changed record count.')
            work[key]=work[key].astype(object)
            work.loc[selected,key]=np.asarray(values,object)
            if work.loc[selected,key].isna().any(): raise ValueError('Discretizer removed observed values.')
        def mask(value):
            try:
                if value in space.states: return 1<<space.states.index(value)
                if value in space.labels: return space.masks[space.labels.index(value)]
            except (TypeError,ValueError): pass
            if isinstance(value,str) and value=='Theta': return space.theta
            if isinstance(value,(int,np.integer)) and 1<=int(value)<=space.theta: return int(value)
            if isinstance(value,(set,frozenset,tuple)) and value:
                if all(v in space.states for v in value):
                    return sum(1<<space.states.index(v) for v in set(value))
            raise ValueError(f'Unknown/missing observed state: {value!r}')
        states=np.array([mask(v) for v in work[state_column]],int)
        groups=[(k,) for k in inputs]+[inputs[:i] for i in range(2,len(inputs)+1)]
        result={}
        for key in groups:
            selected=available.loc[:,list(key)].all(axis=1).to_numpy()
            if not selected.any(): raise ValueError(f'No supported acquisition records for {key}.')
            rows=list(work.loc[selected,list(key)].itertuples(index=False,name=None))
            unique=list(dict.fromkeys(rows));indices={v:i for i,v in enumerate(unique)}
            counts=np.zeros((len(unique),len(space)))
            for row,state in zip(rows,states[selected]): counts[indices[row],space.masks.index(int(state))]+=1
            values=[v[0] for v in unique] if len(key)==1 else unique
            table=EvidenceAcquisition.from_counts(counts,values=values,space=space,source='explicit input dataframe')
            table.metadata.update({'inputs':list(key),'state_column':state_column,'missing_patterns':patterns.copy(),
                                   'available_records':int(selected.sum()),'requires_explicit_joint_support':len(key)>1})
            result[key]=table
        return result
