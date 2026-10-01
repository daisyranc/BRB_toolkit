import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def test_graphite_training_has_12_rows():
    d = json.loads((ROOT / 'examples/paper_cases/graphite.json').read_text(encoding='utf-8'))
    assert len(d['training']) == 12
    assert all(len(row) == 5 for row in d['training'])
    assert d['training'][2] == [0.98, 0.8, 0.2, 0.8, 0.6]
    assert len(d['rules']) == 36

def test_maker_counts():
    d = json.loads((ROOT / 'examples/paper_cases/diagnosis.json').read_text(encoding='utf-8'))
    assert d['prior'] == [565, 5435, 300]
    assert sum(map(sum, d['training_single'][0])) == 2100
    assert sum(map(sum, d['training_single'][1])) == 3100
    assert sum(map(sum, d['joint_counts'])) == 1100
