"""Native DPO, reward ranking, shared-critic PPO and GRPO reference trainer.

Data parallelism uses explicit gradient averaging, with one full replica per
rank. Online sampling is local and sequential; no external rollout service.
"""
from dataclasses import asdict, dataclass, replace
import json
import math
import os
from pathlib import Path
import random
import re
import time
from decimal import Decimal, InvalidOperation

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F

from .model import LanguageModel, ModelConfig
from .tokenizer import TextTokenizer, sha256_file
from .training import (BatchStream, amp_context, code_fingerprint, git_revision,
                       learning_rate_at, load_checkpoint, resolve_training_budget,
                       restore_rng, rng_state, select_device)
from .posttrain_data import PosttrainDataset
from .progress import TrainingProgress, log_json
from .best_checkpoint import BestCheckpoint
from .rl_objectives import (dpo_loss, gae, group_advantages, clipped_policy_loss,
                            clipped_value_loss, reference_kl)


@dataclass(frozen=True)
class PosttrainConfig:
    algorithm: str = 'dpo'
    max_steps: int = 200
    epochs: int | None = None
    batch_size: int = 1
    grad_accum_steps: int = 4
    learning_rate: float = 1e-6
    warmup_steps: int = 10
    min_lr_ratio: float = .1
    weight_decay: float = 0.
    grad_clip: float = 1.
    aux_loss_coef: float = .01
    beta: float = .1
    kl_coef: float = .02
    clip_range: float = .2
    value_clip: float = .2
    value_coef: float = .5
    entropy_coef: float = 0.
    gamma: float = 1.
    gae_lambda: float = .95
    update_epochs: int = 2
    group_size: int = 4
    max_new_tokens: int = 64
    temperature: float = 1.
    reward: str = 'exact'
    reward_scale: float = 1.
    eval_every: int = 20
    save_every: int = 20
    eval_max_records: int = 32
    seed: int = 42
    precision: str = 'bf16'
    device: str = 'auto'
    cpu_threads: int = 1

    def __post_init__(self):
        if self.algorithm not in ('dpo', 'reward', 'ppo', 'grpo'):
            raise ValueError('unsupported post-training algorithm')
        for key in ('max_steps','batch_size','grad_accum_steps','update_epochs','group_size',
                    'max_new_tokens','eval_every','save_every','eval_max_records','cpu_threads'):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f'{key} must be a positive integer')
        if self.epochs is not None and (type(self.epochs) is not int or self.epochs < 1):
            raise ValueError('epochs must be a positive integer')
        if self.algorithm == 'grpo' and self.group_size < 2:
            raise ValueError('GRPO requires at least two responses per prompt')
        if self.reward not in ('exact', 'numeric', 'model'):
            raise ValueError('reward must be exact, numeric or model')
        if self.precision not in ('fp32','bf16') or self.device not in ('auto','cpu','cuda'):
            raise ValueError('post-training supports fp32/bf16 and cpu/cuda/auto')
        floats = ('learning_rate','grad_clip','aux_loss_coef','beta','kl_coef','clip_range',
                  'value_clip','value_coef','entropy_coef','gamma','gae_lambda','temperature',
                  'reward_scale','weight_decay','min_lr_ratio')
        if any(not math.isfinite(getattr(self,k)) for k in floats):
            raise ValueError('nonfinite hyperparameter')
        if min(self.learning_rate,self.grad_clip,self.beta,self.temperature,self.reward_scale) <= 0:
            raise ValueError('learning rate, clip norm, beta, temperature, reward scale must be positive')
        if min(self.aux_loss_coef,self.kl_coef,self.value_coef,self.entropy_coef,self.weight_decay) < 0:
            raise ValueError('loss coefficients and weight decay must be nonnegative')
        if not 0 < self.clip_range < 1 or self.value_clip <= 0 or not 0 <= self.gamma <= 1 or not 0 <= self.gae_lambda <= 1:
            raise ValueError('invalid PPO clipping/GAE')
        if not 0 <= self.min_lr_ratio <= 1 or self.warmup_steps < 0 or (self.epochs is None and self.warmup_steps >= self.max_steps):
            raise ValueError('invalid learning-rate schedule')

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text()))


class ScalarHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.projection = nn.Linear(hidden_size, 1)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, hidden):
        return self.projection(hidden).squeeze(-1).float()


def sequence_output(model, prompt, response, device, head=None, temperature=None):
    """Completion-only probabilities; no prompt, padding or invented EOS targets."""
    ids = torch.tensor([prompt + response], device=device)
    out = model(ids[:, :-1], return_hidden=head is not None)
    logits = out['logits'][0, len(prompt)-1:]
    if temperature is not None:
        logits = sampling_logits(logits, temperature)
    log_probs = logits.float().log_softmax(-1)
    target = torch.tensor(response, device=device)
    out['response_logp'] = log_probs.gather(-1, target[:, None]).squeeze(-1)
    out['entropy'] = -(log_probs.exp() * log_probs).sum(-1)
    if head is not None:
        out['values'] = head(out['hidden_states'][0, len(prompt)-1:])
    # Do not keep the full prompt vocabulary tensor alive during accumulation.
    del out['logits']
    out.pop('hidden_states', None)
    return out


def sampling_logits(logits, temperature):
    logits = logits.float() / temperature
    # Only EOS (2) and text IDs (>=6) are valid completion tokens. Use this
    # identical distribution in rollout, old-policy recording and updates.
    return logits.index_fill(-1, torch.tensor([0,1,3,4,5], device=logits.device), -1e9)


@torch.no_grad()
def sample_response(model, prompt, c, device, eos_id=2, greedy=False):
    model.eval()
    ids = torch.tensor([prompt], device=device)
    cache, response, old_logp = None, [], []
    for _ in range(c.max_new_tokens):
        with amp_context(device,c.precision):
            out = model(ids, past_key_values=cache, use_cache=True)
        logits = sampling_logits(out['logits'][0,-1], c.temperature)
        lp = logits.log_softmax(-1)
        token = lp.argmax() if greedy else torch.multinomial(lp.exp(),1)[0]
        response.append(int(token)); old_logp.append(lp[token])
        if int(token) == eos_id:
            break
        ids = token.reshape(1,1)
        cache = out['past_key_values']
    return response, torch.stack(old_logp)


def answer_reward(text, answer, kind):
    if kind == 'exact':
        return float(' '.join(text.split()).casefold() == ' '.join(answer.split()).casefold())
    # Accept only a complete numeric answer or an explicit final-answer field.
    match = re.search(r'<answer>\s*([^<>]+)\s*</answer>\s*$', text)
    candidate = match.group(1) if match else text.strip().split('####')[-1].strip()
    number = r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)'
    if not re.fullmatch(number,candidate) or not re.fullmatch(number,answer.strip()):
        return 0.
    try:
        return float(Decimal(candidate) == Decimal(answer.strip()))
    except InvalidOperation:
        return 0.


def reward_score(model, head, prompt, response, device, return_aux=False):
    ids = torch.tensor([prompt + response], device=device)
    out = model(ids, return_hidden=True)
    score = head(out['hidden_states'][0,-1])
    return (score, out['aux_loss']) if return_aux else score


def synchronize_gradients(parameters, world):
    if world > 1:
        for p in parameters:
            if p.grad is None:
                p.grad = torch.zeros_like(p)
            dist.all_reduce(p.grad)
            p.grad /= world


def reduce_stats(values, device, world):
    tensor = torch.tensor(values,dtype=torch.float64,device=device)
    if world > 1:
        dist.all_reduce(tensor)
    return tensor.tolist()


def normalize_advantages(rollouts, device, world):
    all_adv = torch.cat([r['advantages'] for r in rollouts])
    total, square, count = reduce_stats([all_adv.sum().item(),all_adv.square().sum().item(),all_adv.numel()],device,world)
    mean = total / count
    std = math.sqrt(max(0.,square/count-mean*mean))
    for r in rollouts:
        r['advantages'] = (r['advantages']-mean)/(std+1e-6)


def collect_rollouts(actor, reference, value_head, rows, c, tok, device, reward_model=None, reward_head=None):
    results, groups_zero = [], 0
    for row in rows:
        group = []
        for _ in range(c.group_size if c.algorithm == 'grpo' else 1):
            response, old_logp = sample_response(actor,row['prompt_ids'],c,device)
            text = tok.decode(response)
            with torch.no_grad(), amp_context(device,c.precision):
                ref = sequence_output(reference,row['prompt_ids'],response,device,temperature=c.temperature)['response_logp']
                values = sequence_output(actor,row['prompt_ids'],response,device,head=value_head,temperature=c.temperature).get('values')
                score = (c.reward_scale * torch.tanh(reward_score(reward_model,reward_head,row['prompt_ids'],response,device)).item()
                         if c.reward == 'model' else answer_reward(text,row['answer'],c.reward))
            item = {'prompt':row['prompt_ids'],'response':response,'text':text,'score':score,
                    'old_logp':old_logp.detach(),'ref_logp':ref.detach()}
            if c.algorithm == 'ppo':
                rewards = -c.kl_coef * (old_logp-ref)
                rewards[-1] += score
                item['old_values'] = values.detach()
                item['advantages'],item['returns'] = gae(rewards,values.detach(),c.gamma,c.gae_lambda)
            group.append(item)
        if c.algorithm == 'grpo':
            rewards = torch.tensor([r['score'] for r in group],device=device)
            adv = group_advantages(rewards)
            groups_zero += int(rewards.std(unbiased=False).item() < 1e-6)
            for item,a in zip(group,adv):
                item['advantages'] = a.expand(len(item['response'])).detach()
        results.extend(group)
    return results,groups_zero


@torch.no_grad()
def evaluate_posttrain(actor, reference, head, dataset, c, tok, device, reward_model=None, reward_head=None, rank=0, world=1):
    actor.eval()
    n = min(len(dataset),c.eval_max_records)
    sums = [0.,0.,0.,0.]
    for index in range(rank,n,world):
        row = dataset[index]
        with amp_context(device,c.precision):
            if c.algorithm == 'reward':
                a = reward_score(actor,head,row['prompt_ids'],row['chosen'],device)
                b = reward_score(actor,head,row['prompt_ids'],row['rejected'],device)
                sums[0] += F.softplus(b-a).item(); sums[1] += float(a>b)
            elif c.algorithm == 'dpo':
                a = sequence_output(actor,row['prompt_ids'],row['chosen'],device)['response_logp'].sum()
                b = sequence_output(actor,row['prompt_ids'],row['rejected'],device)['response_logp'].sum()
                ra = sequence_output(reference,row['prompt_ids'],row['chosen'],device)['response_logp'].sum()
                rb = sequence_output(reference,row['prompt_ids'],row['rejected'],device)['response_logp'].sum()
                loss, margin = dpo_loss(a,b,ra,rb,c.beta)
                sums[0] += loss.item(); sums[1] += float(margin>0)
            else:
                response,_ = sample_response(actor,row['prompt_ids'],c,device,greedy=True)
                text = tok.decode(response)
                score = (c.reward_scale*torch.tanh(reward_score(reward_model,reward_head,row['prompt_ids'],response,device)).item()
                         if c.reward == 'model' else answer_reward(text,row['answer'],c.reward))
                sums[1] += score
                out = sequence_output(actor,row['prompt_ids'],response,device,temperature=c.temperature)
                ref = sequence_output(reference,row['prompt_ids'],response,device,temperature=c.temperature)
                sums[2] += reference_kl(out['response_logp'],ref['response_logp']).mean().item()
            sums[3] += 1
    sums = reduce_stats(sums,device,world)
    result = {'val_records':int(sums[3]),'val_total_records':len(dataset),'val_full':n==len(dataset)}
    if c.algorithm in ('dpo','reward'):
        result.update(val_loss=sums[0]/sums[3],val_preference_accuracy=sums[1]/sums[3])
    else:
        result.update(val_reward=sums[1]/sums[3],val_kl=sums[2]/sums[3])
    actor.train()
    return result


def save_posttrain(root, actor, head, optimizer, step, stream, c, provenance, best, device, rank, world, counts):
    states = [None]*world
    local = rng_state(device)
    if world > 1:
        dist.all_gather_object(states,local)
    else:
        states[0] = local
    if rank == 0:
        path = root/f'step-{step:07d}.pt'
        if path.exists():
            raise FileExistsError(path)
        payload = {'format':'moe-lab-reward-v1' if c.algorithm=='reward' else 'moe-lab-checkpoint-v1',
            'model':actor.state_dict(),'model_config':asdict(actor.config),
            'train_config':{'stage':c.algorithm},'posttrain_config':asdict(c),'provenance':provenance,
            'optimizer':optimizer.state_dict(),'scalar_head':head.state_dict() if head else None,
            'step':step,'epoch':stream.epoch,'cursor':stream.cursor,'rng_states':states,
            'best':best,'counts':counts,'world_size':world}
        temp = path.with_suffix('.pt.tmp')
        with temp.open('xb') as f:
            torch.save(payload,f)
        temp.rename(path)
    if world > 1:
        dist.barrier()


def train_posttrain(base_checkpoint, c, data, tokenizer, output, resume=None, stop_after=None, reward_checkpoint=None):
    world,rank = int(os.environ.get('WORLD_SIZE','1')),int(os.environ.get('RANK','0'))
    device = select_device(c.device)
    torch.set_num_threads(c.cpu_threads)
    if world > 1:
        dist.init_process_group('nccl' if device.type=='cuda' else 'gloo')
    try:
        return _train_posttrain(base_checkpoint,c,data,tokenizer,output,resume,stop_after,reward_checkpoint,device,rank,world)
    finally:
        if world > 1:
            dist.destroy_process_group()


def _train_posttrain(base_checkpoint,c,data,tokenizer,output,resume,stop_after,reward_checkpoint,device,rank,world):
    with amp_context(device,c.precision):
        pass
    tok = TextTokenizer(tokenizer)
    train_data,val_data = PosttrainDataset(data,'train'),PosttrainDataset(data,'val')
    manifest = train_data.manifest
    expected = 'preferences' if c.algorithm in ('dpo','reward') else 'prompts'
    if manifest['kind'] != expected or manifest['tokenizer_sha256'] != tok.fingerprint:
        raise ValueError('post-training data kind/tokenizer mismatch')
    c,steps_per_epoch = resolve_training_budget(c,len(train_data),world)
    if c.warmup_steps >= c.max_steps:
        raise ValueError('warmup exceeds training budget')
    if stop_after is not None and not 1 <= stop_after <= c.max_steps:
        raise ValueError('stop-after must be within the full schedule')
    base = load_checkpoint(base_checkpoint)
    if base['train_config']['stage'] != 'sft' or base['provenance']['tokenizer_sha256'] != tok.fingerprint:
        raise ValueError('base-checkpoint must be a matching text SFT checkpoint')
    mc = ModelConfig(**base['model_config'])
    if manifest['max_seq_len'] > mc.max_seq_len:
        raise ValueError('prepared data exceeds model context')
    if c.algorithm in ('ppo','grpo'):
        for dataset in (train_data,val_data):
            for i in range(len(dataset)):
                row = dataset[i]
                if len(row['prompt_ids']) + c.max_new_tokens > mc.max_seq_len:
                    raise ValueError('prompt plus max-new-tokens exceeds context; reprepare/filter data')
                if c.reward != 'model' and 'answer' not in row:
                    raise ValueError('verifiable reward requires answer labels')
    if (c.reward=='model' and c.algorithm in ('ppo','grpo')) != bool(reward_checkpoint):
        raise ValueError('online reward=model requires reward-checkpoint; other modes must omit it')
    provenance = {'base_sha256':sha256_file(base_checkpoint),'base_checkpoint':str(Path(base_checkpoint).resolve()),
        'tokenizer_sha256':tok.fingerprint,'data_sha256':sha256_file(Path(data)/'manifest.json'),
        'code_sha256':code_fingerprint(),'git_revision':git_revision(),'torch':str(torch.__version__),
        'world_size':world,'device_type':device.type,
        'reward_sha256':sha256_file(reward_checkpoint) if reward_checkpoint else None,
        'external_model_calls':0}
    state = torch.load(resume,map_location='cpu',weights_only=True) if resume else None
    if state and (state.get('posttrain_config') != asdict(c) or state.get('provenance') != provenance):
        raise ValueError('exact post-training resume requires unchanged config/code/data/base/reward/world/device')
    root = Path(output)
    exists = torch.tensor(int(root.exists()),device=device)
    if world > 1:
        dist.all_reduce(exists)
    if exists.item():
        raise FileExistsError(f'use a new output directory: {root}')
    random.seed(c.seed); torch.manual_seed(c.seed)
    actor = LanguageModel(mc).to(device)
    actor.load_state_dict(state['model'] if state else base['model'])
    reference = None
    if c.algorithm != 'reward':
        reference = LanguageModel(mc).to(device).requires_grad_(False).eval()
        reference.load_state_dict(base['model'])
    head = ScalarHead(mc.hidden_size).to(device) if c.algorithm in ('ppo','reward') else None
    if state and head:
        head.load_state_dict(state['scalar_head'])
    reward_model = reward_head = None
    if reward_checkpoint:
        rm = torch.load(reward_checkpoint,map_location='cpu',weights_only=True)
        if rm['format']!='moe-lab-reward-v1' or rm['provenance']['tokenizer_sha256']!=tok.fingerprint or rm['model_config']!=asdict(mc):
            raise ValueError('reward-model architecture/tokenizer mismatch')
        reward_model = LanguageModel(mc).to(device).requires_grad_(False).eval()
        reward_model.load_state_dict(rm['model'])
        reward_head = ScalarHead(mc.hidden_size).to(device).requires_grad_(False).eval()
        reward_head.load_state_dict(rm['scalar_head'])
        del rm
    del base
    parameters = list(actor.parameters()) + (list(head.parameters()) if head else [])
    optimizer = torch.optim.AdamW(parameters,lr=c.learning_rate,weight_decay=c.weight_decay)
    start,epoch,cursor = (state['step'],state['epoch'],state['cursor']) if state else (0,0,0)
    counts = dict(state['counts']) if state else {'generated_tokens':0,'rollout_sequences':0,'optimizer_updates':0,'prompt_exposures':0}
    if state:
        optimizer.load_state_dict(state['optimizer'])
    end = stop_after or c.max_steps
    if end <= start:
        raise ValueError('checkpoint reached requested end')
    scope = {'eval_max_records':c.eval_max_records,'precision':c.precision,'world_size':world,
             'metric':'val_loss' if c.algorithm in ('dpo','reward') else 'val_reward','reward':c.reward}
    best = BestCheckpoint(state,scope) if c.algorithm in ('dpo','reward') else None
    if rank==0:
        root.mkdir(parents=True)
        (root/'run.json').write_text(json.dumps({'config':asdict(c),'model_config':asdict(mc),
            'provenance':provenance,'data':manifest,'reference':'frozen original SFT',
            'critic':'shared actor backbone + value head' if c.algorithm=='ppo' else None},indent=2))
        if tok.path != 'byte':
            (root/'tokenizer.json').write_bytes(Path(tok.path).read_bytes())
        if best:
            best.publish(root)
        log_json({'event':'training_budget','algorithm':c.algorithm,'max_steps':c.max_steps,
                  'epochs':c.epochs,'steps_per_epoch':steps_per_epoch,
                  'step_unit':'rollout batch' if c.algorithm in ('ppo','grpo') else 'optimizer update'})
    if world>1:
        dist.barrier()
    stream = BatchStream(train_data,c,rank,world,epoch,cursor,collate_fn=lambda rows:rows)
    if state:
        restore_rng(state['rng_states'][rank],device)
    else:
        torch.manual_seed(c.seed+rank); random.seed(c.seed+rank)
    del state
    started = time.perf_counter()
    if device.type=='cuda':
        torch.cuda.reset_peak_memory_stats(device)
    with TrainingProgress(rank=rank,total=c.max_steps,start=start,steps_per_epoch=steps_per_epoch,epochs=c.epochs) as progress:
        for step in range(start,end):
            progress.phase('rollout' if c.algorithm in ('ppo','grpo') else 'train')
            rows = sum(stream.next_group(c.grad_accum_steps,finish_epoch=c.epochs is not None),[])
            global_rows = reduce_stats([len(rows)],device,world)[0]
            counts['prompt_exposures'] += int(global_rows)
            lr = learning_rate_at(step,c)
            for group in optimizer.param_groups:
                group['lr'] = lr
            metrics = {'step':step+1,'epochs_completed':stream.epoch+stream.cursor/len(stream.loader),'learning_rate':lr}
            stats = [0.]*7
            rollouts = None
            if c.algorithm in ('ppo','grpo'):
                rollouts,zero = collect_rollouts(actor,reference,head,rows,c,tok,device,reward_model,reward_head)
                if c.algorithm=='ppo':
                    normalize_advantages(rollouts,device,world)
                tokens,sequences = reduce_stats([sum(len(r['response']) for r in rollouts),len(rollouts)],device,world)
                counts['generated_tokens'] += int(tokens); counts['rollout_sequences'] += int(sequences)
                scored,zero,totalgroups = reduce_stats([sum(r['score'] for r in rollouts),zero,len(rows)],device,world)
                metrics.update(train_reward=scored/sequences,zero_variance_group_fraction=zero/totalgroups,
                               truncated_fraction=reduce_stats([sum(r['response'][-1]!=tok.eos_id for r in rollouts)],device,world)[0]/sequences)
            progress.phase('train')
            repeats = c.update_epochs if rollouts is not None else 1
            for _ in range(repeats):
                actor.train(); optimizer.zero_grad(set_to_none=True)
                examples = rollouts if rollouts is not None else rows
                denominator = reduce_stats([len(examples)],device,world)[0]
                for row in examples:
                    with amp_context(device,c.precision):
                        if c.algorithm=='reward':
                            a, aa = reward_score(actor,head,row['prompt_ids'],row['chosen'],device,return_aux=True)
                            b, ba = reward_score(actor,head,row['prompt_ids'],row['rejected'],device,return_aux=True)
                            objective = F.softplus(b-a)
                            aux = (aa+ba)/2
                            stats[3] += float(a.detach()>b.detach())
                        elif c.algorithm=='dpo':
                            a = sequence_output(actor,row['prompt_ids'],row['chosen'],device)
                            b = sequence_output(actor,row['prompt_ids'],row['rejected'],device)
                            with torch.no_grad():
                                ra = sequence_output(reference,row['prompt_ids'],row['chosen'],device)['response_logp'].sum()
                                rb = sequence_output(reference,row['prompt_ids'],row['rejected'],device)['response_logp'].sum()
                            objective,margin = dpo_loss(a['response_logp'].sum(),b['response_logp'].sum(),ra,rb,c.beta)
                            aux = (a['aux_loss']+b['aux_loss'])/2
                            stats[3] += float(margin.detach()>0)
                        else:
                            out = sequence_output(actor,row['prompt'],row['response'],device,head=head,temperature=c.temperature)
                            policy,ratio = clipped_policy_loss(out['response_logp'],row['old_logp'],row['advantages'],c.clip_range)
                            kl = reference_kl(out['response_logp'],row['ref_logp']).mean()
                            value = clipped_value_loss(out['values'],row['old_values'],row['returns'],c.value_clip).mean() if head else kl.new_zeros(())
                            objective = policy.mean()+c.value_coef*value-c.entropy_coef*out['entropy'].mean()
                            if c.algorithm=='grpo':
                                objective = objective+c.kl_coef*kl
                            aux = out['aux_loss']
                            stats[2] += kl.detach().item()
                            stats[3] += float(((ratio.detach()-1).abs()>c.clip_range).float().mean())
                            stats[4] += value.detach().item()
                            stats[5] += ratio.detach().mean().item()
                        loss = objective + c.aux_loss_coef*aux
                    (loss * (world/denominator)).backward()
                    stats[0] += objective.detach().item(); stats[1] += aux.detach().item(); stats[6] += 1
                synchronize_gradients(parameters,world)
                grad = nn.utils.clip_grad_norm_(parameters,c.grad_clip,error_if_nonfinite=True)
                optimizer.step(); counts['optimizer_updates'] += 1
            sums = reduce_stats(stats,device,world)
            metrics.update(train_loss=sums[0]/sums[6],aux_loss=sums[1]/sums[6],grad_norm=float(grad),**counts)
            if rollouts is None:
                metrics['train_preference_accuracy'] = sums[3]/sums[6]
            else:
                metrics.update(reference_kl=sums[2]/sums[6],clip_fraction=sums[3]/sums[6],
                               value_loss=sums[4]/sums[6],mean_ratio=sums[5]/sums[6])
            progress.update(metrics)
            evaluated = (step+1)%c.eval_every==0 or step+1==end
            if evaluated:
                progress.phase('validate')
                # Evaluate greedily so checkpoint comparisons do not use a changing sampling seed.
                metrics.update(evaluate_posttrain(actor,reference,head,val_data,c,tok,device,reward_model,reward_head,rank,world))
            elapsed = time.perf_counter()-started
            metrics.update(elapsed_seconds_this_run=elapsed,allocated_gpu_seconds_this_run=elapsed*world if device.type=='cuda' else 0.)
            if device.type=='cuda':
                memory = torch.tensor(torch.cuda.max_memory_allocated(device),device=device)
                if world>1:
                    dist.all_reduce(memory,op=dist.ReduceOp.MAX)
                metrics['max_rank_peak_memory_bytes'] = memory.item()
            improved = best.consider(metrics,root,rank,world) if best and evaluated else False
            if rank==0:
                with (root/'metrics.jsonl').open('a') as f:
                    f.write(json.dumps(metrics)+'\n')
                log_json(metrics)
            if rollouts is not None:
                with (root/f'rollouts-rank{rank:03d}.jsonl').open('a') as f:
                    for r in rollouts:
                        f.write(json.dumps({'step':step+1,'rank':rank,'prompt_ids':r['prompt'],'response_ids':r['response'],
                                            'response':r['text'],'reward':r['score']})+'\n')
            if improved or (step+1)%c.save_every==0 or step+1==end:
                progress.phase('save')
                save_posttrain(root,actor,head,optimizer,step+1,stream,c,provenance,best.best if best else None,device,rank,world,counts)
                if rank==0 and best:
                    best.publish(root)
    metrics['best'] = best.best if best else None
    if rank==0:
        (root/'summary.json').write_text(json.dumps(metrics,indent=2))
    return metrics
