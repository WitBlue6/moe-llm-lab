"""Optional post-training commands using the project's native networks."""
from dataclasses import replace
import json
from pathlib import Path


def register(commands):
    p = commands.add_parser('post-prepare', help='prepare preference pairs or verifiable online prompts')
    p.add_argument('--input', nargs='+', required=True)
    p.add_argument('--kind', choices=('preferences','prompts'), required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--tokenizer', default='byte')
    p.add_argument('--max-seq-len', type=int, default=512)
    p.add_argument('--val-ratio', type=float, default=.05)
    p.add_argument('--seed', type=int, default=42)
    p = commands.add_parser('post-train', help='DPO, reward-model ranking, PPO or GRPO after text SFT')
    p.add_argument('--base-checkpoint', required=True)
    p.add_argument('--config', required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--tokenizer', default='byte')
    p.add_argument('--output', required=True)
    p.add_argument('--resume')
    p.add_argument('--stop-after', type=int)
    p.add_argument('--epochs', type=int)
    p.add_argument('--reward-checkpoint')
    p = commands.add_parser('post-evaluate', help='held-out preference loss/accuracy or generated-answer reward')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--base-checkpoint', required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--tokenizer', default='byte')
    p.add_argument('--reward-checkpoint')
    p.add_argument('--device', choices=('auto','cpu','cuda'), default='auto')
    p.add_argument('--max-records', type=int, default=128)
    p.add_argument('--output')


def dispatch(args):
    from .posttrain_data import prepare_posttrain, PosttrainDataset
    from .posttraining import (PosttrainConfig, train_posttrain, evaluate_posttrain,
                               ScalarHead, select_device, load_checkpoint)
    if args.command == 'post-prepare':
        result = prepare_posttrain(args.input,args.output,args.tokenizer,args.kind,args.max_seq_len,args.val_ratio,args.seed)
    elif args.command == 'post-train':
        c = PosttrainConfig.load(args.config)
        if args.epochs is not None:
            c = replace(c,epochs=args.epochs)
        return train_posttrain(args.base_checkpoint,c,args.data,args.tokenizer,args.output,args.resume,args.stop_after,args.reward_checkpoint)
    else:
        import torch
        from .model import LanguageModel, ModelConfig
        from .tokenizer import TextTokenizer, sha256_file
        if args.max_records < 1:
            raise ValueError('max-records must be positive')
        state = torch.load(args.checkpoint,map_location='cpu',weights_only=True)
        if 'posttrain_config' not in state:
            raise ValueError('post-evaluate requires a post-training checkpoint')
        c = replace(PosttrainConfig(**state['posttrain_config']),device=args.device,eval_max_records=args.max_records)
        torch.set_num_threads(c.cpu_threads)
        device, tok = select_device(c.device), TextTokenizer(args.tokenizer)
        if state['provenance']['tokenizer_sha256'] != tok.fingerprint or sha256_file(args.base_checkpoint) != state['provenance']['base_sha256']:
            raise ValueError('evaluation base/tokenizer mismatch')
        dataset = PosttrainDataset(args.data,'val')
        if dataset.manifest['tokenizer_sha256']!=tok.fingerprint or dataset.manifest['kind']!=('preferences' if c.algorithm in ('dpo','reward') else 'prompts'):
            raise ValueError('evaluation data kind/tokenizer mismatch')
        mc = ModelConfig(**state['model_config'])
        actor = LanguageModel(mc).to(device)
        actor.load_state_dict(state['model'])
        head = ScalarHead(mc.hidden_size).to(device) if state['scalar_head'] is not None else None
        if head:
            head.load_state_dict(state['scalar_head'])
        reference = reward_model = reward_head = None
        if c.algorithm != 'reward':
            reference = LanguageModel(mc).to(device).eval()
            reference.load_state_dict(load_checkpoint(args.base_checkpoint)['model'])
        if c.algorithm in ('ppo','grpo') and c.reward=='model':
            if not args.reward_checkpoint or sha256_file(args.reward_checkpoint)!=state['provenance']['reward_sha256']:
                raise ValueError('evaluation requires the same reward checkpoint')
            rm = torch.load(args.reward_checkpoint,map_location='cpu',weights_only=True)
            reward_model = LanguageModel(mc).to(device).eval()
            reward_model.load_state_dict(rm['model'])
            reward_head = ScalarHead(mc.hidden_size).to(device).eval()
            reward_head.load_state_dict(rm['scalar_head'])
        if c.algorithm in ('ppo','grpo'):
            for i in range(min(len(dataset),args.max_records)):
                row=dataset[i]
                if len(row['prompt_ids'])+c.max_new_tokens>mc.max_seq_len or (c.reward!='model' and 'answer' not in row):
                    raise ValueError('evaluation prompt context/answer invalid')
        result=evaluate_posttrain(actor,reference,head,dataset,c,tok,device,reward_model,reward_head)
        if args.output:
            with Path(args.output).open('x') as handle:
                json.dump(result,handle,indent=2)
    print(json.dumps(result,indent=2))
