"""Check that the source distribution contains only public project assets."""
from pathlib import Path
import json
import nbformat

ROOT=Path(__file__).resolve().parents[1]

def test_no_local_environment_or_development_records():
    for name in ['.venv-paper','.paper-work','docs/superpowers','docs/paper_extensions']:
        assert not (ROOT/name).exists(),name
    assert not (ROOT/'scripts/build_paper_notebooks.py').exists()

def test_notebooks_have_no_machine_specific_paths():
    for path in (ROOT/'examples/notebooks').glob('*.ipynb'):
        nb=nbformat.read(path,as_version=4)
        public=json.dumps(nb,ensure_ascii=False).lower()
        for token in [r'c:\\users\\']:
            assert token not in public,(path.name,token)
        code=[c for c in nb.cells if c.cell_type=='code']
        assert all(c.execution_count is not None for c in code)

def test_documentation_and_runner_paths_are_current():
    for path in [ROOT/'README.md',ROOT/'README.zh-CN.md',ROOT/'examples/README.md',ROOT/'examples/README.zh-CN.md',
                 ROOT/'scripts/execute_paper_notebooks.py',ROOT/'examples/rimer_training.py',ROOT/'examples/maker_training.py']:
        text=path.read_text(encoding='utf-8')
        assert '.paper-work' not in text,path
