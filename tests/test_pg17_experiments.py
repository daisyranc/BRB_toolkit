import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'project'))
import numpy as np
import pandas as pd
from brb.optimization import TrainingConfig
from pg17_experiments import run_device_comparison, experiment_environment

def test_pair_resets_seed_and_skips_unavailable(tmp_path):
    import torch
    seen=[]
    def task(config,output):
        seen.append(np.random.rand())
        pd.DataFrame([{'split':'test','n':2,'mse':.1}]).to_csv(output/'metrics.csv',index=False)
        return {'objective':1.}
    table=run_device_comparison(task,config=TrainingConfig(maxiter=0),repeats=2,devices=('cpu',),output_dir=tmp_path)
    assert len(table)==2 and table.status.eq('completed').all()
    assert seen[0]==seen[1]
    assert (tmp_path/'environment.json').exists()
    assert 'cuda_available' in experiment_environment()

def test_reports_accept_no_traces(tmp_path):
    from pg17_training_reports import build_reports
    pd.DataFrame([{'device':'cpu','repeat':0,'status':'completed','training_seconds':1.,'end_to_end_seconds':2.}]).to_csv(tmp_path/'timing.csv',index=False)
    build_reports(tmp_path)
    assert (tmp_path/'REPORT.md').exists() and (tmp_path/'REPORT.zh-CN.md').exists()

def test_shared_parent_snapshot_and_initialization_hash(tmp_path):
    from pg17_training import shared_parent_beliefs, parameter_fingerprint
    import torch
    first={'g':{'train':torch.tensor([[.2,.8]],dtype=torch.float64)}}
    second={'g':{'train':torch.tensor([[.3,.7]],dtype=torch.float64)}}
    shared_parent_beliefs(first,tmp_path,'cpu')
    matched=shared_parent_beliefs(second,tmp_path,'cpu')
    np.testing.assert_array_equal(matched['g']['train'],first['g']['train'])
    assert (tmp_path/'manifest.json').exists()

def test_unobserved_maker_reliabilities_are_frozen():
    from pg17_device_maker import maker_parameter_scope
    from pg17_maker import build_branch
    bundle=build_branch(pd.DataFrame({'risk_state_usage':[0,0],'ClaimCount':[0,1]}),['risk_state_usage'])
    bounds,mask=maker_parameter_scope(bundle)
    np.testing.assert_array_equal(mask[(('risk_state_usage',),'reliabilities')],[True,False,False])

def test_maker_prediction_preserves_requested_precision():
    import torch
    from pg17_maker import build_branch
    from pg17_maker_backend import predict_maker_rows
    frame=pd.DataFrame({'risk_state_usage':[0,0],'ClaimCount':[0,1]})
    bundle=build_branch(frame,['risk_state_usage'])
    p,trust=predict_maker_rows(bundle,frame,backend='torch',dtype=torch.float32)
    assert p.dtype==np.float32 and trust.dtype==np.float32
