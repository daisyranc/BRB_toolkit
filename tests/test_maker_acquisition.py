import numpy as np
import pandas as pd
import pytest
from brb.maker import StateSpace,EvidenceAcquisition

def test_smoothing_requires_explicit_nonzero_setting():
    space=StateSpace(['H1','H2']);counts=np.array([[1,0,0],[0,1,0]])
    with pytest.raises(ValueError): EvidenceAcquisition.from_counts(counts,values=['a','b'],space=space)
    table=EvidenceAcquisition.from_counts(counts,values=['a','b'],space=space,smoothing=.1)
    np.testing.assert_array_equal(table.counts,counts)
    assert np.all(table.probabilities>0) and table.metadata['smoothing']==.1

def test_frame_preserves_available_rows_and_explicit_discretisation():
    space=StateSpace(['H1','H2'])
    frame=pd.DataFrame({'x':[.1,.2,.9,.1,.9,.9], 'y':['p',None,'n','p','n',None],
                        'state':['H1','Theta','H2','Theta','H1','H2']})
    tables=EvidenceAcquisition.from_frame(frame,inputs=['x','y'],state_column='state',space=space,
        discretizers={'x':lambda s:s.map(lambda v:'low' if v<.5 else 'high')})
    assert set(tables)=={('x',),('y',),('x','y')}
    assert tables[('x',)].counts.sum()==6 and tables[('y',)].counts.sum()==4
    assert tables[('x','y')].counts.sum()==4
    assert tables[('x','y')].joint_support is None
    assert tables[('x',)].metadata['missing_patterns']=={'11':4,'10':2}
    assert tables[('x',)].values==('low','high')
    with pytest.raises(ValueError): tables[('x',)].index('unseen')

def test_frame_invalid_states_not_silently_dropped():
    frame=pd.DataFrame({'x':['a'],'state':['not-a-state']})
    with pytest.raises(ValueError,match='state'):
        EvidenceAcquisition.from_frame(frame,inputs=['x'],state_column='state',space=StateSpace(['H1','H2']))
