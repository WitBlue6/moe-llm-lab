import json
from pathlib import Path

import pytest

from moe_llm.best_checkpoint import BestCheckpoint
from moe_llm.data import prepare
from moe_llm.training import TrainConfig, train, load_checkpoint


def test_finite_strict_minimum_and_scope(tmp_path):
    tracker = BestCheckpoint(None, {'limit': 2})
    assert not tracker.consider({'step': 1, 'val_loss': float('nan')}, tmp_path)
    assert tracker.consider({'step': 2, 'val_loss': 3.}, tmp_path)
    with pytest.raises(FileNotFoundError):
        tracker.publish(tmp_path)
    Path(tracker.best['checkpoint']).touch()
    tracker.publish(tmp_path)
    assert not tracker.consider({'step': 3, 'val_loss': 3.}, tmp_path)
    assert not tracker.consider({'step': 4}, tmp_path)
    with pytest.raises(ValueError, match='validation scope'):
        BestCheckpoint({'best': tracker.best}, {'limit': 3})
    assert BestCheckpoint({'best': tracker.best}, {'limit': 2}).best['step'] == 2


def test_best_off_schedule_resume_and_init_reset(tmp_path, tiny, monkeypatch):
    import moe_llm.training as training
    fixtures = Path(__file__).parents[1] / 'fixtures'
    data = tmp_path / 'data'
    prepare([fixtures / 'pretrain.jsonl'], data, 'byte', 'pretrain', 64, .25)
    c = TrainConfig(max_steps=4, warmup_steps=0, batch_size=2, grad_accum_steps=1,
                    eval_every=1, save_every=4, device='cpu', precision='fp32')
    losses = iter([2., 1., 3., 4., 5.])
    monkeypatch.setattr(training, 'evaluate', lambda *a, **kw: {'val_loss': next(losses)})
    partial, resumed = tmp_path / 'partial', tmp_path / 'resumed'
    train(tiny, c, data, 'byte', partial, stop_after=3)
    assert (partial / 'step-0000002.pt').is_file()  # best outside periodic save
    assert (partial / 'step-0000001.pt').is_file()
    result = train(tiny, c, data, 'byte', resumed, resume=partial / 'step-0000003.pt')
    best = json.loads((resumed / 'best.json').read_text())
    assert best['step'] == 2 and best['val_loss'] == 1.
    assert Path(best['checkpoint']) == partial / 'step-0000002.pt'
    assert load_checkpoint(resumed / 'step-0000004.pt')['best'] == best == result['best']
    fresh = tmp_path / 'fresh'
    train(tiny, c, data, 'byte', fresh, init_from=partial / 'step-0000003.pt', stop_after=1)
    assert json.loads((fresh / 'best.json').read_text())['val_loss'] == 5.
