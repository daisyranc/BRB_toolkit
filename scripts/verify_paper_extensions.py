"""Write a reproducible numerical validation report."""
from pathlib import Path
import json,sys,platform,importlib.metadata as metadata
from datetime import datetime, timezone
import numpy as np
import nbformat

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from examples.paper_cases.graphite import load_graphite_case,build_graphite_model,graphite_inputs
from examples.paper_cases.diagnosis import build_diagnosis_case,apply_paper_parameters
from brb.maker import paper_objectives

def main():
    case=load_graphite_case();m=build_graphite_model();x=graphite_inputs(case['training'][2][:4]);r=m.infer(x)
    middle=max(float(abs(r.trace[k].beliefs[0]-case['paper_reference'][k]).max()) for k in ['X2','X4','X6','X8'])
    complete=float(abs(r.beliefs[0]-[.2685,.6772,.0543]).max())
    x['X5']=np.array([[0,0,.8]]);inc=m.infer(x)
    incomplete=float(abs(np.r_[inc.beliefs[0],inc.ignorance[0]]-np.r_[case['paper_reference']['X9_incomplete'],.0371]).max())
    pm,data,dc=build_diagnosis_case();apply_paper_parameters(pm,dc)
    mr=[pm.infer(v,include_prior=False) for v in dc['joint_order']]
    maker=float(abs(np.array([v.probabilities for v in mr])-dc['paper_reference']['table18']).max())
    trust=float(abs(np.array([v.trustworthiness for v in mr])-dc['paper_reference']['trustworthiness']).max())
    notebooks=[]
    for path in sorted((ROOT/'examples/notebooks').glob('*.ipynb')):
        nb=nbformat.read(path,as_version=4);nbformat.validate(nb)
        code=[c for c in nb.cells if c.cell_type=='code']
        assert all(c.execution_count is not None for c in code)
        assert not any(o.output_type=='error' for c in code for o in c.outputs)
        notebooks.append({'name':path.name,'executed_code_cells':len(code)})
    result={'date':datetime.now(timezone.utc).date().isoformat(),'python':platform.python_version(),
            'versions':{p:metadata.version(p) for p in ['numpy','scipy','torch','pytest','nbformat','nbclient','ipykernel']},
            'max_absolute_errors':{'R06_intermediate':middle,'R06_complete_X9':complete,'R06_incomplete_X9':incomplete,'M25_table18':maker,'M25_trustworthiness':trust},
            'paper_parameters_recomputed_objectives':[float(v.max()) for v in paper_objectives(pm,data)],'notebooks':notebooks}
    limits=[5e-5,1e-4,5e-5,1.5e-4,1.5e-4]
    assert all(e <= t for e,t in zip(result['max_absolute_errors'].values(),limits)), result['max_absolute_errors']
    output=ROOT/'outputs/validation_results.json'
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
