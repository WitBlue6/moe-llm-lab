from dataclasses import replace
import json
import math
from pathlib import Path

import pytest
import torch

from moe_llm.model import LanguageModel
from moe_llm.tokenizer import TextTokenizer
from moe_llm.posttrain_data import prepare_posttrain, PosttrainDataset
from moe_llm.posttraining import (PosttrainConfig, ScalarHead, train_posttrain, sequence_output,
    sample_response, answer_reward, evaluate_posttrain)
from moe_llm.rl_objectives import (dpo_loss, gae, group_advantages, clipped_policy_loss,
    clipped_value_loss, reference_kl)


def setup_data(tmp_path, tiny):
    from dataclasses import asdict
    tok=TextTokenizer('byte')
    base=tmp_path/'sft.pt'
    model=LanguageModel(tiny)
    torch.save({'format':'moe-lab-checkpoint-v1','model':model.state_dict(),'model_config':asdict(tiny),
                'train_config':{'stage':'sft'},'provenance':{'tokenizer_sha256':tok.fingerprint}},base)
    raw=tmp_path/'raw.jsonl'
    raw.write_text(''.join(json.dumps({'prompt':f'q{i}','chosen':'1','rejected':'2','answer':'1'})+'\n' for i in range(60)))
    prefs,prompts=tmp_path/'prefs',tmp_path/'prompts'
    prepare_posttrain([raw],prefs,'byte','preferences',128,.2)
    prepare_posttrain([raw],prompts,'byte','prompts',128,.2)
    return base,prefs,prompts


def test_objectives_and_reward_math():
    a=torch.tensor([0.],requires_grad=True)
    b=torch.tensor([0.],requires_grad=True)
    loss,_=dpo_loss(a,b,torch.zeros(1),torch.zeros(1),.1)
    assert loss.item()==pytest.approx(math.log(2))
    loss.backward()
    assert a.grad.item()<0 and b.grad.item()>0
    adv,ret=gae(torch.tensor([0.,1.]),torch.tensor([.2,.4]),1.,1.)
    torch.testing.assert_close(ret,torch.ones(2))
    torch.testing.assert_close(adv,torch.tensor([.8,.6]))
    assert torch.equal(group_advantages(torch.ones(4)),torch.zeros(4))
    torch.testing.assert_close(group_advantages(torch.tensor([0.,1.])),torch.tensor([-1.,1.]),atol=3e-6,rtol=0)
    pg,ratio=clipped_policy_loss(torch.tensor([math.log(2.)]),torch.zeros(1),torch.ones(1),.2)
    assert pg.item()==pytest.approx(-1.2)
    assert clipped_value_loss(torch.tensor([2.]),torch.tensor([0.]),torch.tensor([2.]),.2).item()==pytest.approx(1.62)
    assert torch.equal(reference_kl(a.detach(),a.detach()),torch.zeros(1))
    assert answer_reward('work\n#### 3.0','3','numeric')==1
    assert answer_reward('maybe 3 or 4','3','numeric')==0
    assert answer_reward('<answer>3</answer>','3','numeric')==1


def test_prompt_split_completion_mask_and_sampler(tmp_path,tiny):
    base,prefs,prompts=setup_data(tmp_path,tiny)
    tok=TextTokenizer('byte')
    a=PosttrainDataset(prefs,'train'); b=PosttrainDataset(prefs,'val')
    assert {json.dumps(a[i]['messages']) for i in range(len(a))}.isdisjoint({json.dumps(b[i]['messages']) for i in range(len(b))})
    p=PosttrainDataset(prompts,'train')[0]
    assert p['prompt_ids']==tok.chat(p['messages'],True)[0]
    actor=LanguageModel(tiny).eval()
    prompt=p['prompt_ids']; response=tok.encode('1')+[tok.eos_id]
    out=sequence_output(actor,prompt,response,torch.device('cpu'))
    full=actor(torch.tensor([prompt+response[:-1]]))['logits'][0]
    expected=full[len(prompt)-1:].log_softmax(-1).gather(1,torch.tensor(response)[:,None]).squeeze(1)
    torch.testing.assert_close(out['response_logp'],expected)
    c=PosttrainConfig(max_steps=2,warmup_steps=0,max_new_tokens=4,device='cpu',precision='fp32')
    tokens,old=sample_response(actor,prompt,c,torch.device('cpu'))
    recompute=sequence_output(actor,prompt,tokens,torch.device('cpu'),temperature=c.temperature)['response_logp']
    torch.testing.assert_close(old,recompute,atol=2e-6,rtol=2e-6)
    assert all(t==2 or t>=6 for t in tokens)


@pytest.mark.parametrize('algorithm',['dpo','reward','ppo','grpo'])
def test_train_resume_and_generate_compatibility(tmp_path,tiny,algorithm):
    base,prefs,prompts=setup_data(tmp_path,tiny)
    data=prefs if algorithm in ('dpo','reward') else prompts
    c=PosttrainConfig(algorithm=algorithm,max_steps=2,warmup_steps=0,batch_size=2,grad_accum_steps=1,
        update_epochs=2,group_size=2,max_new_tokens=2,eval_every=1,save_every=2,eval_max_records=2,
        device='cpu',precision='fp32',reward='exact')
    full,part,resumed=[tmp_path/x for x in ('full','part','resumed')]
    train_posttrain(base,c,data,'byte',full)
    train_posttrain(base,c,data,'byte',part,stop_after=1)
    result=train_posttrain(base,c,data,'byte',resumed,resume=part/'step-0000001.pt')
    a=torch.load(full/'step-0000002.pt',weights_only=True)
    b=torch.load(resumed/'step-0000002.pt',weights_only=True)
    for key in a['model']:
        torch.testing.assert_close(a['model'][key],b['model'][key],rtol=0,atol=0)
    if a['scalar_head']:
        for key in a['scalar_head']:
            torch.testing.assert_close(a['scalar_head'][key],b['scalar_head'][key],rtol=0,atol=0)
    assert a['counts']==b['counts']
    assert result['step']==2
    if algorithm!='reward':
        from moe_llm.training import load_checkpoint
        assert load_checkpoint(full/'step-0000002.pt')['train_config']['stage']==algorithm
    if algorithm in ('dpo','reward'):
        best=json.loads((resumed/'best.json').read_text())
        assert Path(best['checkpoint']).is_file()
    with pytest.raises(ValueError,match='unchanged'):
        train_posttrain(base,replace(c,learning_rate=2e-6),data,'byte',tmp_path/'bad',resume=part/'step-0000001.pt')


def test_learned_reward_route(tmp_path,tiny):
    base,prefs,prompts=setup_data(tmp_path,tiny)
    common=dict(max_steps=2,warmup_steps=0,batch_size=1,grad_accum_steps=1,eval_every=1,
                save_every=1,eval_max_records=2,device='cpu',precision='fp32')
    train_posttrain(base,PosttrainConfig(algorithm='reward',**common),prefs,'byte',tmp_path/'rm',stop_after=1)
    result=train_posttrain(base,PosttrainConfig(algorithm='ppo',reward='model',max_new_tokens=2,**common),
        prompts,'byte',tmp_path/'ppo',stop_after=1,reward_checkpoint=tmp_path/'rm/step-0000001.pt')
    assert math.isfinite(result['val_reward'])


@pytest.mark.parametrize('algorithm', ['dpo', 'reward', 'ppo', 'grpo'])
def test_epoch_budget_and_reference_immutability(tmp_path,tiny,algorithm):
    base,prefs,prompts=setup_data(tmp_path,tiny)
    data = prompts if algorithm in ('ppo', 'grpo') else prefs
    digest=base.read_bytes()
    c=PosttrainConfig(algorithm=algorithm,epochs=1,max_steps=100,warmup_steps=0,
                     batch_size=7,grad_accum_steps=2,max_new_tokens=2,group_size=2,
                     update_epochs=2,eval_every=100,save_every=100,
                     eval_max_records=2,device='cpu',precision='fp32')
    n=len(PosttrainDataset(data,'train'))
    result=train_posttrain(base,c,data,'byte',tmp_path/'epoch')
    assert result['step']==math.ceil(math.ceil(n/7)/2)
    assert result['optimizer_updates']==result['step']*(2 if algorithm in ('ppo','grpo') else 1)
    if algorithm in ('ppo','grpo'):
        assert result['rollout_sequences']==n*(2 if algorithm=='grpo' else 1)
    assert result['epochs_completed']==1
    assert result['prompt_exposures']==n
    assert base.read_bytes()==digest
