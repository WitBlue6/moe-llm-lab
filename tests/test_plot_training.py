import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('plot_training', Path(__file__).parents[1] / 'scripts/plot_training.py')
plot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plot)


def test_live_log_keeps_sparse_validation(tmp_path):
    path = tmp_path / 'metrics.jsonl'
    rows = [{'step': 1, 'train_loss': 4., 'aux_loss': 1.2},
            {'step': 2, 'train_loss': 3., 'val_loss': 3.5},
            {'step': 3, 'train_loss': 2.}]
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows) + '{"step": 4')
    with pytest.warns(UserWarning, match='final line'):
        data = plot.read_metrics(path)
    assert data['val_loss'] == ([2], [3.5])
    assert data['aux_loss'] == ([1], [1.2])
    assert plot.moving_average(data['train_loss'][1], 2) == [4., 3.5, 2.5]
    path.write_text('{bad json}\n')
    with pytest.raises(ValueError, match='invalid metrics'):
        plot.read_metrics(path)


def test_plot_missing_aux_and_no_overwrite(tmp_path):
    pytest.importorskip('matplotlib')
    (tmp_path / 'metrics.jsonl').write_text('{"step": 1, "train_loss": 2}\n')
    output = tmp_path / 'loss.png'
    counts = plot.plot_runs([tmp_path], output, labels=['fixture'])
    assert counts['fixture'] == {'train_loss': 1, 'aux_loss': 0, 'val_loss': 0}
    assert output.read_bytes().startswith(b'\x89PNG')
    with pytest.raises(FileExistsError):
        plot.plot_runs([tmp_path], output)
