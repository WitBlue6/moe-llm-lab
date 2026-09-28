import io
import json

import pytest

from moe_llm.progress import TrainingProgress, log_json


class Terminal(io.StringIO):
    def isatty(self):
        return True


def test_epoch_resume_boundary_and_last_validation(monkeypatch):
    monkeypatch.delenv('MOE_LAB_PROGRESS', raising=False)
    stream = Terminal()
    with TrainingProgress(rank=0, total=6, start=2, steps_per_epoch=3, epochs=2, stream=stream) as p:
        assert 'Epoch 1/2' in p.description() and '2/3' in p.description()
        log_json({'step': 3, 'train_loss': 2., 'val_loss': 3.})
        assert 'Epoch 1/2' in p.description() and '3/3' in p.description()
        log_json({'step': 4, 'train_loss': 1.})
        assert 'Epoch 2/2' in p.description() and '1/3' in p.description()
        assert 'val=3.0000@3' in p.description()
        log_json({'event': 'validation_start'})
        assert p.status == 'validate'
    assert p.status == 'stopped'
    assert '\033[2K' in stream.getvalue()


def test_redirected_output_is_json_and_worker_is_silent(monkeypatch):
    monkeypatch.delenv('MOE_LAB_PROGRESS', raising=False)
    stream = io.StringIO()
    record = {'step': 1, 'train_loss': 2.}
    with TrainingProgress(rank=0, total=1, start=0, steps_per_epoch=1, stream=stream) as p:
        log_json(record)
    assert json.loads(stream.getvalue()) == record
    assert '\r' not in stream.getvalue() and p.status == 'done'
    worker = Terminal()
    with TrainingProgress(rank=2, total=1, start=0, steps_per_epoch=1, stream=worker) as p:
        p.update(record)
        p.log(record)
    assert worker.getvalue() == ''


def test_failure_cleanup_and_disable(monkeypatch):
    monkeypatch.setenv('MOE_LAB_PROGRESS', '0')
    stream = Terminal()
    with pytest.raises(RuntimeError):
        with TrainingProgress(rank=0, total=2, start=0, steps_per_epoch=2, stream=stream) as p:
            raise RuntimeError('training failed')
    assert p.status == 'failed' and stream.getvalue() == ''
