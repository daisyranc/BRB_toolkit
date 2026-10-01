"""Execute in fresh kernels using the requested interpreter; save real outputs."""
from pathlib import Path
import json,sys,time
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager

ROOT=Path(__file__).resolve().parents[1]

def execute_notebooks(paths,python,timeout=1800):
    kernel_root=ROOT/'.cache/brb-toolkit/kernels';folder=kernel_root/'paper-demo'
    folder.mkdir(parents=True,exist_ok=True)
    (folder/'kernel.json').write_text(json.dumps({'argv':[str(Path(python).resolve()),'-m','ipykernel_launcher','-f','{connection_file}'],
        'display_name':'Paper demo isolated Python','language':'python',
        'env':{'IPYTHONDIR':str(ROOT/'.cache/brb-toolkit/ipython'),'MPLCONFIGDIR':str(ROOT/'.cache/brb-toolkit/matplotlib')}}),encoding='utf-8')
    reports={}
    for path in paths:
        path=Path(path);start=time.monotonic()
        nb=nbformat.read(path,as_version=4)
        km=KernelManager(kernel_name='paper-demo',kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernel_root)]))
        print('Executing:',path.name,flush=True)
        client=NotebookClient(nb,timeout=timeout,km=km,resources={'metadata':{'path':str(path.parent)}})
        try:
            client.execute(cleanup_kc=True)
        finally:
            # Preserve any failure output for diagnosis, never mark it successful.
            nbformat.write(nb,path)
        reports[path.name]={'success':True,'seconds':round(time.monotonic()-start,2),'python':str(python),
                            'code_cells':sum(c.cell_type=='code' for c in nb.cells)}
        print(reports[path.name],flush=True)
    (ROOT/'.cache/brb-toolkit/notebook_execution.json').write_text(json.dumps(reports,indent=2),encoding='utf-8')
    return reports

if __name__=='__main__':
    paths=[Path(p) for p in sys.argv[1:]] or sorted((ROOT/'examples/notebooks').glob('*.ipynb'))
    execute_notebooks(paths,Path(sys.executable))
