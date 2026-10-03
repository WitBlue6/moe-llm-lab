import copy
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
import torch.distributed as dist

from moe_llm.model import LanguageModel, ModelConfig
from moe_llm.posttraining import sequence_output, synchronize_gradients
from moe_llm.rl_objectives import dpo_loss


def worker(rank,rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method=rendezvous,rank=rank,world_size=2)
    try:
        torch.manual_seed(77)
        c=ModelConfig(vocab_size=32,hidden_size=16,num_layers=1,num_attention_heads=2,num_key_value_heads=1,
                      expert_intermediate_size=24,num_experts=4,experts_per_token=2,max_seq_len=32)
        local=LanguageModel(c)
        expected=copy.deepcopy(local)
        prompts=[[1,4,8,2,5],[1,4,9,10,2,5]]
        def objective(model,i):
            a=sequence_output(model,prompts[i],[12,2],torch.device('cpu'))['response_logp'].sum()
            b=sequence_output(model,prompts[i],[13,14,2],torch.device('cpu'))['response_logp'].sum()
            return dpo_loss(a,b,a.detach()+.2,b.detach(),.1)[0]
        ((objective(expected,0)+objective(expected,1))/2).backward()
        objective(local,rank).backward()
        synchronize_gradients(list(local.parameters()),2)
        for (name,a),(_,b) in zip(local.named_parameters(),expected.named_parameters()):
            reference=torch.zeros_like(b) if b.grad is None else b.grad
            torch.testing.assert_close(a.grad,reference,atol=1e-7,rtol=1e-4,msg=name)
    finally:
        dist.destroy_process_group()


def test_pair_gradient_global_equivalence(tmp_path):
    torch.multiprocessing.spawn(worker,args=((tmp_path/'rendezvous').as_uri(),),nprocs=2,join=True)


@pytest.mark.parametrize('algorithm',['dpo','ppo','grpo'])
def test_two_rank_train_resume(tmp_path,tiny,algorithm):
    import json
    from dataclasses import asdict
    from moe_llm.posttrain_data import prepare_posttrain
    from moe_llm.posttraining import PosttrainConfig
    from moe_llm.tokenizer import TextTokenizer
    base=tmp_path/'base.pt'
    torch.save({'format':'moe-lab-checkpoint-v1','model':LanguageModel(tiny).state_dict(),
                'model_config':asdict(tiny),'train_config':{'stage':'sft'},
                'provenance':{'tokenizer_sha256':TextTokenizer('byte').fingerprint}},base)
    raw=tmp_path/'rows.jsonl'
    raw.write_text(''.join(json.dumps({'prompt':f'q{i}','chosen':'1','rejected':'2','answer':'1'})+'\n' for i in range(40)))
    data=tmp_path/'data'
    prepare_posttrain([raw],data,'byte','preferences' if algorithm=='dpo' else 'prompts',128,.2)
    c=PosttrainConfig(algorithm=algorithm,max_steps=2,warmup_steps=0,grad_accum_steps=1,
        update_epochs=2,group_size=2,max_new_tokens=2,device='cpu',precision='fp32',eval_every=1,save_every=2,eval_max_records=3)
    config=tmp_path/'config.json'; config.write_text(json.dumps(asdict(c)))
    env={**os.environ,'OMP_NUM_THREADS':'1'}
    if sys.platform=='darwin': env['GLOO_SOCKET_IFNAME']='lo0'
    command=[sys.executable,'-m','torch.distributed.run','--rdzv-backend=c10d','--rdzv-endpoint=127.0.0.1:0',
             '--local-addr=127.0.0.1','--rdzv-conf=is_host=true','--nnodes=1','--nproc-per-node=2',
             '-m','moe_llm.cli','post-train','--base-checkpoint',str(base),'--config',str(config),'--data',str(data)]
    for name,extra in [('full',[]),('part',['--stop-after','1']),('resumed',['--resume',str(tmp_path/'part/step-0000001.pt')])]:
        result=subprocess.run([*command,'--output',str(tmp_path/name),*extra],env=env,capture_output=True,text=True,timeout=60)
        assert result.returncode==0,result.stdout+result.stderr
    full=torch.load(tmp_path/'full/step-0000002.pt',weights_only=True)
    resumed=torch.load(tmp_path/'resumed/step-0000002.pt',weights_only=True)
    for key in full['model']:
        torch.testing.assert_close(full['model'][key],resumed['model'][key],atol=0,rtol=0)
    assert full['counts']==resumed['counts']
    assert len(full['rng_states'])==2
