import json
from pathlib import Path
import numpy as np
from brb.maker import StateSpace,EvidenceTable,MAKERModel,ObservationGroup,MAKERTrainingData

def build_diagnosis_case():
    case=json.loads(Path(__file__).with_suffix('.json').read_text(encoding='utf-8'))
    space=StateSpace(['H1','H2']);values=case['value_order'];joint_values=[tuple(v) for v in case['joint_order']]
    tables={}
    for key,counts in zip([('saliva',),('blood',)],case['single_counts']):
        tables[key]=EvidenceTable.from_counts(space,values,counts,source=case['source'])
    joint=EvidenceTable.from_counts(space,joint_values,case['joint_counts'],source=case['source'])
    # Same underlying observed state gives diagonal joint supports in this case.
    # This allocation is case-specific, never inferred automatically by MAKERModel.
    joint.joint_support=np.stack([np.diag(p) for p in joint.probabilities])
    tables[('saliva','blood')]=joint
    tables[('prior',)]=EvidenceTable(space,['all'],[np.array(case['prior'])/sum(case['prior'])],source=case['source'])
    upper={('saliva',):.9,('blood',):.95,('prior',):1.,('saliva','blood'):.85}
    params={k:{'weights':np.full(3,upper[k]),'reliabilities':np.full(len(t.values),upper[k])} for k,t in tables.items()}
    bounds={k:{'weights':(.5,upper[k]),'reliabilities':(.5,upper[k])} for k in tables}
    model=MAKERModel(tables,params,order=['saliva','blood'],prior=('prior',))
    singles=[ObservationGroup((k,),values,c) for k,c in zip(['saliva','blood'],case['training_single'])]
    prefixes=[ObservationGroup(('saliva','blood'),joint_values,case['joint_counts'])]
    data=MAKERTrainingData(singles,prefixes,params,6300,prior=('prior',),bounds=bounds)
    return model,data,case

def apply_paper_parameters(model,case):
    for i,key in enumerate([('saliva',),('blood',),('prior',),('saliva','blood')]):
        model.parameters[key]['weights'][:]=case['paper_reference']['weights'][i]
        model.parameters[key]['reliabilities'][:]=case['paper_reference']['reliabilities'][i]
