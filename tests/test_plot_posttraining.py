import importlib.util
import json
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('plot_posttraining',Path(__file__).parents[1]/'scripts/plot_posttraining.py')
plot=importlib.util.module_from_spec(spec);spec.loader.exec_module(plot)


def test_sparse_validation_and_live_tail(tmp_path):
 p=tmp_path/'metrics.jsonl'
 p.write_text('{"step":1,"train_reward":0}\n{"step":2,"train_reward":0.5,"val_reward":0}\n{"step":3')
 with pytest.warns(UserWarning,match='final line'):d=plot.read_metrics(p)
 assert d['val_reward']==([2],[0])
 assert d['train_reward']==([1,2],[0,0.5])
 assert plot.moving_average([0,0.5],2)==[0,0.25]
 p.write_text('{"step":1,"val_reward":NaN}\n')
 with pytest.raises(ValueError,match='finite'):plot.read_metrics(p)


def test_auto_panels_preserve_zero_rewards_and_no_overwrite(tmp_path):
 pytest.importorskip('matplotlib')
 (tmp_path/'metrics.jsonl').write_text('{"step":1,"train_reward":0,"val_reward":0,"truncated_fraction":1}\n')
 out=tmp_path/'reward.png';counts=plot.plot_runs([tmp_path],out,['PPO'],2)
 assert counts=={'PPO':{'train_reward':1,'val_reward':1,'truncated_fraction':1}}
 assert out.read_bytes().startswith(b'\x89PNG')
 with pytest.raises(FileExistsError):plot.plot_runs([tmp_path],out)
 selected=plot.plot_runs([tmp_path],tmp_path/'explicit.svg',metrics=['val_reward','val_loss'])
 assert next(iter(selected.values()))=={'val_reward':1,'val_loss':0}
 with pytest.raises(ValueError,match='unique'):plot.plot_runs([tmp_path],tmp_path/'bad.png',metrics=['val_reward','val_reward'])
