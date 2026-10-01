"""Run the 21-parameter M25 diagnosis case: python -m examples.maker_training."""
import json
from pathlib import Path
from brb.maker import MAKERTrainer,paper_objectives
from examples.paper_cases.diagnosis import build_diagnosis_case,apply_paper_parameters

def run(maxiter=500):
    model,data,case=build_diagnosis_case()
    result=MAKERTrainer(maxiter=maxiter).fit(model,data,bounds=data.bounds)
    report={'training':result.to_dict(),'paper_reference':case['paper_reference'],
            'trained_f1_f2':[float(x.max()) for x in paper_objectives(model,data)]}
    paper,_,_=build_diagnosis_case();apply_paper_parameters(paper,case)
    report['paper_parameter_f1_f2']=[float(x.max()) for x in paper_objectives(paper,data)]
    report['computed_table18']=[paper.infer(v,include_prior=False).probabilities.tolist() for v in case['joint_order']]
    return model,result,report

if __name__=='__main__':
    _,result,report=run()
    destination=Path('outputs/maker_run.json');destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('trained_f1_f2','paper_parameter_f1_f2','computed_table18')},indent=2))
    print('success:',result.success,'residual:',result.constraint_violation,'stages:',result.stage_results)
