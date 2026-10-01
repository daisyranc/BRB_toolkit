from pathlib import Path
import nbformat
import pytest

@pytest.mark.parametrize('name',['01_brb_modelling_paper','02_rimer_inference_paper','03_maker_paper'])
def test_notebook_real_outputs_and_source(name):
    path=Path(__file__).resolve().parents[1]/'examples/notebooks'/f'{name}.ipynb'
    nb=nbformat.read(path,as_version=4);nbformat.validate(nb)
    source='\n'.join(c.source for c in nb.cells)
    assert 'https://' in source and 'paper_reference' in source and 'computed' in source
    code=[c for c in nb.cells if c.cell_type=='code']
    assert 8<=len(code)<=12
    assert all(c.execution_count is not None for c in code)
    assert not any(o.output_type=='error' for c in code for o in c.outputs)
    assert any(c.outputs for c in code)
    if name.startswith('01'): assert 'training' in source and '12' in source
    if name.startswith('02'): assert 'Unknown' in source
    if name.startswith('03'): assert 'stage_results' in source
